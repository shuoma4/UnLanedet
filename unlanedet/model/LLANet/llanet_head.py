from calendar import c
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import random
import logging

from ..module.head.plaindecoder import PlainDecoder
from ..module.losses.focal_loss import FocalLoss
from .line_iou import line_iou, liou_loss
from .dynamic_assign import assign
from .roi_gather import ROIGather, LinearModule
from .prior import init_prior_embeddings_with_stats
from unlanedet.utils.detailed_loss_logger import DetailedLossLogger
from unlanedet.model.module.core.lane import Lane
from unlanedet.layers.ops import nms
from unlanedet.layers.ops.nms_demo import lane_nms
from unlanedet.model.LLANet.vis_monitor import visualize_training_progress


class LLANetHead(nn.Module):
    def __init__(
        self,
        num_points=72,
        prior_feat_channels=64,
        fc_hidden_dim=64,
        num_priors=192,
        num_fc=2,
        refine_layers=3,
        sample_points=36,
        cfg=None,
    ):
        super(LLANetHead, self).__init__()
        if cfg is None:
            raise ValueError("cfg must be provided")
        self.cfg = cfg
        self.enable_category = cfg.enable_category
        self.enable_attribute = cfg.enable_attribute
        self.num_lane_categories = cfg.num_lane_categories
        self.num_lr_attributes = cfg.num_lr_attributes
        self.scale_factor = cfg.scale_factor
        self.detailed_loss_logger = cfg.get("detailed_loss_logger", None)
        if (
            self.detailed_loss_logger is None
            and cfg.get("detailed_loss_logger_config") is not None
        ):
            conf = cfg.get("detailed_loss_logger_config")
            self.detailed_loss_logger = DetailedLossLogger(
                output_dir=conf.output_dir, filename=conf.filename
            )
        self.img_w = self.cfg.img_w
        self.img_h = self.cfg.img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim

        # Dynamic weight scheduling parameters
        self.epoch_per_iter = cfg.epoch_per_iter
        self.warmup_epochs = cfg.warmup_epochs
        self.start_cls_loss_weight = cfg.start_cls_loss_weight
        self.cls_loss_weight = cfg.cls_loss_weight
        self.start_category_loss_weight = cfg.start_category_loss_weight
        self.category_loss_weight = cfg.category_loss_weight
        self.start_attribute_loss_weight = cfg.start_attribute_loss_weight
        self.attribute_loss_weight = cfg.attribute_loss_weight
        self.dataset_statistics = cfg.dataset_statistics
        self.logger = logging.getLogger("unlanedet")  # 获取 logger 实例
        self.stats_data = None
        if self.dataset_statistics and os.path.exists(self.dataset_statistics):
            try:
                self.stats_data = np.load(self.dataset_statistics, allow_pickle=True)
                self.logger.info(
                    f"Loaded dataset statistics: {self.dataset_statistics}"
                )
            except Exception as e:
                self.logger.error(f"Failed to load dataset statistics: {e}")

        self.prior_feat_channels = prior_feat_channels
        # Buffers
        self.__init_buffers()
        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings()
        self.register_buffer("priors", init_priors)
        self.register_buffer("priors_on_featmap", priors_on_featmap)

        self.seg_conv = nn.Conv2d(
            self.prior_feat_channels * self.refine_layers, self.prior_feat_channels, 1
        )
        self.seg_decoder = PlainDecoder(cfg)

        reg_modules, cls_modules = [], []
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.ModuleList(reg_modules)
        self.cls_modules = nn.ModuleList(cls_modules)

        self.roi_gather = ROIGather(
            in_channels=self.prior_feat_channels,
            num_priors=self.num_priors,
            sample_points=self.sample_points,
            fc_hidden_dim=self.fc_hidden_dim,
            refine_layers=self.refine_layers,
            mid_channels=self.fc_hidden_dim,
        )
        self.reg_layers = nn.Linear(self.fc_hidden_dim, self.n_offsets + 1 + 2 + 1)
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)
        if self.enable_category:
            self.category_modules = nn.ModuleList(
                [
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            self.prototypes = nn.Parameter(
                torch.randn(self.num_lane_categories, self.fc_hidden_dim)
            )
            nn.init.normal_(self.prototypes, mean=0.0, std=0.01)

        if self.enable_attribute:
            self.attribute_modules = nn.ModuleList(
                [
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            self.attribute_layers = nn.Linear(
                self.fc_hidden_dim, self.num_lr_attributes
            )

        weights = torch.ones(self.cfg.num_classes)
        weights[0] = self.cfg.bg_weight
        self.criterion = torch.nn.NLLLoss(
            ignore_index=self.cfg.ignore_label, weight=weights
        )

        if self.enable_category:
            cat_weights = None
            if (
                self.stats_data is not None
                and "cls_category_weights" in self.stats_data
            ):
                # The loaded weights are actually frequencies. We need to invert them to get proper loss weights.
                freqs = torch.tensor(
                    self.stats_data["cls_category_weights"], dtype=torch.float32
                )
                # Invert: 1 / (freq + epsilon)
                cat_weights = 1.0 / (freqs + 1e-6)
                # Normalize so mean is 1
                cat_weights = cat_weights / cat_weights.mean()

                self.logger.info(
                    f"Using statistics for category loss weights. Weights: {cat_weights}"
                )
            self.category_criterion = torch.nn.NLLLoss(
                weight=cat_weights, reduction="sum"
            )

        if self.enable_attribute:
            self.attribute_criterion = torch.nn.NLLLoss(reduction="sum")

        self.init_weights()

    def __init_buffers(self):
        # sample_x_indexs: sample points for ROI Gather (Top-to-Bottom: 0 -> n_strips)
        self.register_buffer(
            name="sample_x_indexs",
            tensor=(
                torch.linspace(0, 1, steps=self.sample_points, dtype=torch.float32)
                * self.n_strips
            ).long(),
        )
        use_pdf = False
        if self.stats_data is not None:
            try:
                has_y = (
                    "start_y_grid" in self.stats_data
                    and "start_y_pdf" in self.stats_data
                )
                has_x = (
                    "start_x_grid" in self.stats_data
                    and "start_x_pdf" in self.stats_data
                )
                if has_y:
                    y_grid = np.array(self.stats_data["start_y_grid"]).astype(
                        np.float32
                    )
                    y_pdf = np.array(self.stats_data["start_y_pdf"]).astype(np.float32)
                    y_grid = np.clip(y_grid, 0.0, 1.0)
                    y_pdf = np.maximum(y_pdf, 0.0)
                    y_pdf_sum = (
                        float(np.sum(y_pdf)) if float(np.sum(y_pdf)) > 0 else 1.0
                    )
                    y_cdf = np.cumsum(y_pdf) / y_pdf_sum
                    q_feat = np.linspace(0.0, 1.0, self.sample_points, dtype=np.float32)
                    q_off = np.linspace(0.0, 1.0, self.n_offsets, dtype=np.float32)
                    y_feat = np.interp(q_feat, y_cdf, y_grid)
                    y_off = np.interp(q_off, y_cdf, y_grid)
                    # prior_feat_ys: sample points for ROI Gather (Top-to-Bottom: 0 -> 1)
                    # prior_ys: physical y-coordinates for regression/IoU (Bottom-to-Top: 1 -> 0)
                    # This matches self.offsets_ys = np.linspace(self.img_h, 0, self.num_points) in generate_lane_line_openlane.py
                    # which maps index 0 to img_h (Bottom) and index n_offsets-1 to 0 (Top).
                    # Normalized: index 0 -> 1.0 (Bottom), index n_offsets-1 -> 0.0 (Top).
                    self.register_buffer(
                        name="prior_feat_ys",
                        tensor=torch.from_numpy(y_feat).float(),
                    )
                    self.register_buffer(
                        name="prior_ys", tensor=torch.from_numpy(1.0 - y_off).float()
                    )
                    self.register_buffer(
                        name="start_y_pdf_grid", tensor=torch.from_numpy(y_grid).float()
                    )
                    self.register_buffer(
                        name="start_y_pdf", tensor=torch.from_numpy(y_pdf).float()
                    )
                    use_pdf = True
                    self.logger.info(
                        f"Init prior Y buffers from PDF: feat_points={self.sample_points}, offsets={self.n_offsets}"
                    )
                if has_x:
                    x_grid = np.array(self.stats_data["start_x_grid"]).astype(
                        np.float32
                    )
                    x_pdf = np.array(self.stats_data["start_x_pdf"]).astype(np.float32)
                    x_grid = np.clip(x_grid, 0.0, 1.0)
                    x_pdf = np.maximum(x_pdf, 0.0)
                    self.register_buffer(
                        name="start_x_pdf_grid", tensor=torch.from_numpy(x_grid).float()
                    )
                    self.register_buffer(
                        name="start_x_pdf", tensor=torch.from_numpy(x_pdf).float()
                    )
                    self.logger.info("Registered start X PDF buffers.")
            except Exception as e:
                self.logger.warning(f"Init buffers from PDF failed: {e}")
        if not use_pdf:
            self.register_buffer(
                name="prior_feat_ys",
                tensor=torch.flip(
                    (1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]
                ),
            )
            self.register_buffer(
                name="prior_ys",
                tensor=torch.linspace(
                    1.0, 0.0, steps=self.n_offsets, dtype=torch.float32
                ),
            )
            self.logger.info(
                "Init prior Y buffers using uniform linspace (Top-to-Bottom)."
            )
        self.logger.info(
            f"Init buffers: sample_points={self.sample_points}, n_offsets={self.n_offsets}"
        )
        self.logger.info(
            f"Buffer shapes: sample_x_indexs={self.sample_x_indexs.shape}, prior_feat_ys={self.prior_feat_ys.shape}, prior_ys={self.prior_ys.shape}"
        )

    def init_weights(self):
        # Initialize cls_layers (Single Linear layer)
        if isinstance(self.cls_layers, nn.Linear):
            nn.init.normal_(self.cls_layers.weight, mean=0.0, std=1e-3)
            # Use statistics for bias initialization
            prior_prob = 0.01
            if self.stats_data is not None and "seg_positive_ratios" in self.stats_data:
                try:
                    pos_ratios = self.stats_data["seg_positive_ratios"]
                    if len(pos_ratios) > 0:
                        prior_prob = float(np.mean(pos_ratios))
                        prior_prob = max(1e-4, min(1.0 - 1e-4, prior_prob))
                        self.logger.info(
                            f"Init cls bias using prior seg_positive_ratios, prior_prob={prior_prob:.5f}"
                        )
                except Exception as e:
                    self.logger.error(f"Error reading seg_positive_ratios: {e}")
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            if self.cls_layers.bias is not None:
                nn.init.constant_(self.cls_layers.bias, bias_value)
                self.logger.info(f"Init cls bias to {bias_value:.5f}")
        # Initialize reg_layers (Single Linear layer)
        if isinstance(self.reg_layers, nn.Linear):
            nn.init.normal_(self.reg_layers.weight, mean=0.0, std=1e-3)
            if self.reg_layers.bias is not None:
                nn.init.constant_(self.reg_layers.bias, 0)
        if hasattr(self, "priors"):
            try:
                p = self.priors
                start_y_mean = float(p[:, 2].mean().item())
                start_x_mean = float(p[:, 3].mean().item())
                theta_mean = float(p[:, 4].mean().item())
                length_mean = float(p[:, 5].mean().item())
                self.logger.info(
                    f"Priors summary: y_mean={start_y_mean:.4f}, x_mean={start_x_mean:.4f}, theta_mean={theta_mean:.4f}, length_mean={length_mean:.4f}"
                )
            except Exception as e:
                self.logger.warning(f"Log priors summary failed: {e}")
        # Initialize category_modules (ModuleList)
        if self.enable_category:
            for m in self.category_modules:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        # Initialize attribute_layers (Linear)
        if self.enable_attribute:
            if isinstance(self.attribute_layers, nn.Linear):
                nn.init.normal_(self.attribute_layers.weight, mean=0.0, std=1e-3)
                if self.attribute_layers.bias is not None:
                    nn.init.constant_(self.attribute_layers.bias, 0)
            # Also init attribute_modules if they exist (ModuleList)
            if hasattr(self, "attribute_modules"):
                for m in self.attribute_modules:
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def _init_prior_embeddings_default(self):
        # [start_y, start_x, theta] -> all normalize
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8
        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], (i // 2) * strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.0)
            nn.init.constant_(
                self.prior_embeddings.weight[i, 2], 0.16 if i % 2 == 0 else 0.32
            )
        for i in range(left_priors_nums, left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.0)
            nn.init.constant_(
                self.prior_embeddings.weight[i, 1],
                ((i - left_priors_nums) // 4 + 1) * bottom_strip_size,
            )
            nn.init.constant_(self.prior_embeddings.weight[i, 2], 0.2 * (i % 4 + 1))
        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) * strip_size,
            )
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.0)
            nn.init.constant_(
                self.prior_embeddings.weight[i, 2], 0.68 if i % 2 == 0 else 0.84
            )

    def _init_prior_embeddings(self):
        # Force default initialization to match CLRNet
        # self.stats_data check removed to ensure parity with CLRNet's heuristic initialization
        self.logger.info(
            "Using default prior embeddings initialization (CLRNet style)."
        )
        self._init_prior_embeddings_default()

    def generate_priors_from_embeddings(self):
        device = self.prior_embeddings.weight.device
        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates, score[0] = negative prob, score[1] = positive prob
        priors = torch.zeros((self.num_priors, 6 + self.n_offsets), device=device)
        priors[:, 2:5] = self.prior_embeddings.weight.clone()  # y, x, theta
        # DEBUG: Check for NaNs in priors
        if torch.isnan(priors).any():
            self.logger.error("NaN detected in priors after embedding clone!")
        if torch.isinf(priors).any():
            self.logger.error("Inf detected in priors after embedding clone!")
        priors[:, 5] = 0.0
        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets)
            * (self.img_w - 1)
            + (
                (
                    self.prior_ys.repeat(self.num_priors, 1).to(device)
                    - priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)
                )
                * self.img_h
                / torch.tan(
                    priors[:, 4].unsqueeze(1).clone().repeat(1, self.n_offsets)
                    * math.pi
                    + 1e-5
                )
            )
        ) / (self.img_w - 1)
        priors_on_featmap = priors[..., 6 + self.sample_x_indexs]
        return priors, priors_on_featmap

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        batch_size = batch_features.shape[0]
        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        prior_ys = self.prior_feat_ys.view(1, 1, -1, 1).expand(
            batch_size, num_priors, -1, 1
        )
        prior_xs = prior_xs * 2.0 - 1.0
        grid_y = (prior_ys * 2.0) - 1.0
        grid = torch.cat((prior_xs, grid_y), dim=-1)
        feature = F.grid_sample(batch_features, grid, align_corners=True)
        feature = feature.permute(0, 2, 1, 3).reshape(
            batch_size * num_priors, self.prior_feat_channels, self.sample_points, 1
        )
        return feature

    def forward(self, features, img_metas=None, **kwargs):
        batch_features = list(features[-self.refine_layers :])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]
        device = batch_features[-1].device

        if self.training:
            if self.prior_embeddings.weight.device != device:
                self.prior_embeddings = self.prior_embeddings.to(device)
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()
        if self.priors.device != device:
            self.priors = self.priors.to(device)
            self.priors_on_featmap = self.priors_on_featmap.to(device)

        priors = self.priors.expand(batch_size, -1, -1)
        priors_on_featmap = self.priors_on_featmap.expand(batch_size, -1, -1)

        predictions_lists = []
        final_fc_features = None
        prior_features_stages = []

        prior_ys_expanded = (
            self.prior_ys.to(device)
            .view(1, 1, -1)
            .expand(batch_size, self.num_priors, -1)
        )

        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])
            if torch.isnan(prior_xs).any():
                self.logger.error(f"NaN detected in prior_xs at stage {stage}!")
            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs
            )
            # DEBUG: Check batch_prior_features
            if torch.isnan(batch_prior_features).any():
                self.logger.error(
                    f"NaN detected in batch_prior_features at stage {stage}!"
                )
            prior_features_stages.append(batch_prior_features)
            fc_features = self.roi_gather(
                prior_features_stages, batch_features[stage], stage
            )
            # DEBUG: Check fc_features
            if torch.isnan(fc_features).any():
                self.logger.error(f"NaN detected in fc_features at stage {stage}!")
            if torch.isinf(fc_features).any():
                self.logger.error(f"Inf detected in fc_features at stage {stage}!")
            if stage == self.refine_layers - 1:
                final_fc_features = fc_features
            # Flatten (B, N, C) -> (B*N, C) without scrambling
            fc_features_flat = fc_features.view(
                batch_size * num_priors, self.fc_hidden_dim
            )

            cls_features = fc_features_flat.clone()
            reg_features = fc_features_flat.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)

            cls_logits = self.cls_layers(cls_features).reshape(batch_size, -1, 2)
            reg = self.reg_layers(reg_features).reshape(
                batch_size, -1, self.n_offsets + 1 + 2 + 1
            )

            # DEBUG: Check logits and reg
            if torch.isnan(cls_logits).any():
                self.logger.error(f"NaN detected in cls_logits at stage {stage}!")
            if torch.isnan(reg).any():
                self.logger.error(f"NaN detected in reg at stage {stage}!")

            # 1. Gradient Flow: Calculate Raw Params
            pred_start_y = priors[:, :, 2] + reg[:, :, 0]
            pred_start_x = priors[:, :, 3] + reg[:, :, 1]
            pred_theta = priors[:, :, 4] + reg[:, :, 2]
            pred_len = reg[:, :, 3]  # Direct prediction like CLRNet
            pred_len = torch.clamp(pred_len, min=1e-3)

            # 2. Stop Gradient: Clamp for Projection Safety
            clamped_start_y = torch.clamp(pred_start_y, 0.0, 1.0)
            clamped_start_x = torch.clamp(pred_start_x, 0.0, 1.0)
            clamped_theta = pred_theta  # Theta can be outside 0-1 slightly
            clamped_len = (
                pred_len  # Length is unnormalized (points), do not clamp to 1.0
            )

            # 3. Geometric Projection
            def tran_tensor(t):
                return t.unsqueeze(2).expand(-1, -1, self.n_offsets)

            pred_start_y_exp = tran_tensor(clamped_start_y)
            pred_start_x_exp = tran_tensor(clamped_start_x)
            pred_theta_exp = tran_tensor(clamped_theta)

            # prior_ys is [1.0, ..., 0.0] (Bottom to Top)
            # (1 - prior_ys - start_y) * H: distance from start_y (relative to Top-Down) to current row
            # If start_y=0 (Bottom start), at Bottom (prior_ys=1), dy = 1-1-0 = 0.
            # At Top (prior_ys=0), dy = 1-0-0 = 1.
            coords = (
                pred_start_x_exp * (self.img_w - 1)
                + (
                    (1.0 - prior_ys_expanded - pred_start_y_exp)
                    * self.img_h
                    / torch.tan(pred_theta_exp * math.pi + 1e-5)
                )
            ) / (self.img_w - 1)

            # 4. Add Offsets
            pred_points = coords + reg[:, :, 4:]

            # 5. Concat
            predictions = torch.cat(
                [
                    cls_logits,
                    pred_start_y.unsqueeze(-1),
                    pred_start_x.unsqueeze(-1),
                    pred_theta.unsqueeze(-1),
                    pred_len.unsqueeze(-1),
                    pred_points,
                ],
                dim=-1,
            )

            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = predictions.detach().clone()
                priors[:, :, 2] = clamped_start_y.detach()
                priors[:, :, 3] = clamped_start_x.detach()
                priors[:, :, 5] = clamped_len.detach()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        if self.training:
            seg_features = torch.cat(
                [
                    F.interpolate(
                        feature,
                        size=[batch_features[-1].shape[2], batch_features[-1].shape[3]],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for feature in batch_features
                ],
                dim=1,
            )
            seg_out = self.seg_decoder(self.seg_conv(seg_features))
        else:
            seg_out = None

        category_out = None
        attribute_out = None
        if self.enable_category and final_fc_features is not None:
            cat_feat = final_fc_features.clone()
            for m in self.category_modules:
                cat_feat = m(cat_feat)
            category_out = self.scale_factor * torch.matmul(
                F.normalize(cat_feat, p=2, dim=-1),
                F.normalize(self.prototypes, p=2, dim=-1).transpose(0, 1),
            )

        if self.enable_attribute and final_fc_features is not None:
            attr_feat = final_fc_features.view(batch_size, self.num_priors, -1)
            for m in self.attribute_modules:
                attr_feat = m(attr_feat)
            attribute_out = self.attribute_layers(attr_feat)

        final_preds = predictions_lists[-1]
        if self.training:
            outputs = {"predictions_lists": predictions_lists}
            if seg_out is not None:
                outputs.update(**seg_out)
            if category_out is not None:
                outputs["category"] = category_out.view(batch_size, self.num_priors, -1)
            if attribute_out is not None:
                outputs["attribute"] = attribute_out
            return outputs
        else:
            ret = {"lane_lines": final_preds}
            if category_out is not None:
                ret["category"] = category_out.view(batch_size, self.num_priors, -1)
            if attribute_out is not None:
                ret["attribute"] = attribute_out
            return ret

    def get_lanes(self, output, as_lanes=True):
        """
        Convert model output to lanes.
        """
        if isinstance(output, dict):
            output = output["lane_lines"]

        softmax = nn.Softmax(dim=1)

        decoded = []
        for predictions in output:
            # filter out the conf lower than conf threshold
            if self.cfg.test_parameters.conf_threshold is not None:
                threshold = self.cfg.test_parameters.conf_threshold
            else:
                threshold = 0.4
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= threshold
            # Fallback: if no lanes found, take top K or lower threshold to ensure evaluation works
            if keep_inds.sum() == 0 and len(scores) > 0:
                # Use a very low threshold to ensure we output something for evaluation
                # This is critical for early training stages or debugging
                self.logger.debug(f"No lanes >= {threshold}, falling back to 0.01")
                threshold = 0.01
                keep_inds = scores >= threshold

            self.logger.debug(
                f"DEBUG: Keeping {keep_inds.sum()} lanes after threshold {threshold}"
            )

            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                self.logger.debug("DEBUG: No predictions left after filtering")
                decoded.append([])
                continue
            nms_predictions = predictions.detach().clone()
            nms_predictions = torch.cat(
                [nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1
            )
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[..., 5:] = nms_predictions[..., 5:] * (self.img_w - 1)

            # Use lane_nms (Python implementation) instead of generic box NMS
            # Generic NMS expects boxes (x1, y1, x2, y2), but we have (score, score, y, x...)
            # lane_nms uses line_iou which is correct for lanes
            keep = lane_nms(
                nms_predictions[..., 5:],  # Pass points only
                scores,
                nms_overlap_thresh=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.max_lanes,
                img_w=self.cfg.img_w,
            )

            num_to_keep = len(keep)
            self.logger.debug(
                f"Keeping {num_to_keep} lanes after Lane NMS (thres={self.cfg.test_parameters.nms_thres})"
            )
            predictions = predictions[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions)
            else:
                pred = predictions
            decoded.append(pred)

        return decoded

    def predictions_to_pred(self, predictions):
        """
        Convert predictions to internal Lane structure for evaluation.
        """
        self.prior_ys = self.prior_ys.to(predictions.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(
                max(0, int(round(lane[2].item() * self.n_strips))), self.n_strips
            )
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~(
                (
                    ((lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0))
                    .cpu()
                    .numpy()[::-1]
                    .cumprod()[::-1]
                ).astype(bool)
            )
            lane_xs[end + 1 :] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            lane_ys = (
                lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height)
                + self.cfg.cut_height
            ) / self.cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1
            ).squeeze(2)
            lane = Lane(
                points=points.cpu().numpy(),
                metadata={"start_x": lane[3], "start_y": lane[2], "conf": lane[1]},
            )
            lanes.append(lane)
        return lanes

    def _normalize_targets(self, batch_targets):
        """
        Normalize Ground Truth (Pixels -> 0~1) to match Model Output
        Structure: [flag, flag, y, x, theta, len, points...]
        """
        if batch_targets.numel() > 0:
            # Helper to normalize only unnormalized values (> 1.0) and fix invalid ones
            def normalize_part(tensor, index):
                mask_unnorm = tensor[..., index] > 1.0
                if mask_unnorm.any():
                    tensor[..., index][mask_unnorm] /= self.img_w

            # Index 3: Start X
            normalize_part(batch_targets, 3)

            # Index 4: Theta (Degrees -> Normalized)
            # Theta is usually small, but in degrees it's > 1.
            mask_theta = (
                batch_targets[..., 4] > 2.0
            )  # Assume > 2 degrees is unnormalized
            if mask_theta.any():
                batch_targets[..., 4][mask_theta] /= 180.0

            # Index 5: Length
            normalize_part(batch_targets, 5)

            # Index 6+: Points
            if batch_targets.shape[-1] > 6:
                # 1. Normalize unnormalized valid points
                mask_unnorm_pts = batch_targets[..., 6:] > 1.0
                if mask_unnorm_pts.any():
                    batch_targets[..., 6:][mask_unnorm_pts] /= self.img_w

                # 2. Fix invalid points (negative or garbage) to -1.0
                # This prevents huge negative values (-8e7) from causing numerical issues
                mask_invalid = batch_targets[..., 6:] <= 0.0
                if mask_invalid.any():
                    batch_targets[..., 6:][mask_invalid] = -1.0

    def loss(self, outputs, batch, current_iter=0):
        predictions_lists = outputs["predictions_lists"]
        targets_list = batch["lane_line"]
        batch_lane_categories = batch.get("lane_categories", None)
        batch_lane_attributes = batch.get("lane_attributes", None)
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        device = self.priors.device
        batch_size = len(targets_list)
        max_lanes = 0
        target_dim = 0
        if len(targets_list) > 0:
            lengths = [len(t) for t in targets_list]
            max_lanes = max(lengths) if lengths else 0
            if max_lanes > 0:
                target_dim = targets_list[0].shape[1]
        max_lanes = max(max_lanes, 1)
        if target_dim == 0:
            target_dim = 6 + 72
        batch_targets = torch.zeros((batch_size, max_lanes, target_dim), device=device)
        batch_masks = torch.zeros((batch_size, max_lanes), device=device)
        for i, t in enumerate(targets_list):
            num_t = t.shape[0]
            if num_t > 0:
                valid_mask = t[:, 1] == 1
                valid_t = t[valid_mask]
                num_valid = valid_t.shape[0]
                if num_valid > 0:
                    num_valid = min(num_valid, max_lanes)
                    batch_targets[i, :num_valid] = valid_t[:num_valid]
                    batch_masks[i, :num_valid] = 1
        # self._normalize_targets(batch_targets)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_xytl_loss = torch.tensor(0.0, device=device)
        total_iou_loss = torch.tensor(0.0, device=device)
        total_category_loss = torch.tensor(0.0, device=device)
        total_attribute_loss = torch.tensor(0.0, device=device)
        log_ly = torch.tensor(0.0, device=device)
        log_lx = torch.tensor(0.0, device=device)
        log_lt = torch.tensor(0.0, device=device)
        log_ll = torch.tensor(0.0, device=device)
        total_stage_count = 0
        for stage in range(self.refine_layers):
            predictions = predictions_lists[stage]
            with torch.no_grad():
                assigned_mask, assigned_ids = assign(
                    predictions,
                    batch_targets,
                    batch_masks,
                    self.img_w,
                    self.img_h,
                    self.cfg,
                    current_iter=current_iter,
                    prior_ys=self.prior_ys,
                )

            cls_targets = assigned_mask.long().view(-1)
            pred_cls_flat = predictions[..., :2].view(-1, 2)
            num_positives = assigned_mask.sum()
            cls_norm = max(num_positives.item(), 1.0)

            stage_cls_loss = cls_criterion(pred_cls_flat, cls_targets).sum()
            total_cls_loss += stage_cls_loss / cls_norm

            if num_positives > 0:
                batch_idx, prior_idx = torch.where(assigned_mask)
                target_idx = assigned_ids[batch_idx, prior_idx]
                pos_preds = predictions[batch_idx, prior_idx]
                pos_targets = batch_targets[batch_idx, target_idx]
                reg_yxtl = pos_preds[:, 2:6].clone()
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= self.img_w - 1
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips
                target_yxtl = pos_targets[:, 2:6].clone()
                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 1] *= self.img_w - 1
                target_yxtl[:, 2] *= 180
                target_yxtl[:, 3] *= self.n_strips
                # with torch.no_grad():
                #     target_yxtl[:, 3] -= reg_yxtl[:, 0] - target_yxtl[:, 0]
                #     target_yxtl[:, 3] = torch.clamp(target_yxtl[:, 3], min=0)
                loss_y = F.smooth_l1_loss(
                    reg_yxtl[:, 0], target_yxtl[:, 0], reduction="none"
                )
                loss_x = F.smooth_l1_loss(
                    reg_yxtl[:, 1], target_yxtl[:, 1], reduction="none"
                )
                loss_theta = F.smooth_l1_loss(
                    reg_yxtl[:, 2], target_yxtl[:, 2], reduction="none"
                )
                loss_len = F.smooth_l1_loss(
                    reg_yxtl[:, 3], target_yxtl[:, 3], reduction="none"
                )
                reg_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
                loss_components = torch.stack(
                    [loss_y, loss_x, loss_theta, loss_len], dim=1
                )
                total_reg_xytl_loss += (loss_components * reg_weights).mean()
                if self.detailed_loss_logger is not None:
                    # Log the normalized losses for consistency with optimization
                    log_ly += loss_y.mean()
                    log_lx += loss_x.mean()
                    log_lt += loss_theta.mean()
                    log_ll += loss_len.mean()
                    total_stage_count += 1
                # IoU Loss
                # Line IoU Loss
                line_pred = pos_preds[:, 6:] * (self.img_w - 1)
                line_target = pos_targets[:, 6:] * (self.img_w - 1)

                # DEBUG: Check for NaNs in line_pred
                if torch.isnan(line_pred).any() or torch.isinf(line_pred).any():
                    print(f"[LLANetHead] NaN/Inf in line_pred! Iter={current_iter}")
                    # Clamp to avoid crash
                    line_pred = torch.clamp(line_pred, min=-1000, max=5000)

                ious = line_iou(
                    line_pred,
                    line_target,
                    self.img_w,
                    length=30,
                    aligned=True,
                )

                # DEBUG: Check for NaNs in ious
                if torch.isnan(ious).any():
                    self.logger.debug(f"[LLANetHead] NaN in IoUs! Iter={current_iter}")
                    ious = torch.nan_to_num(ious, nan=0.0)

                total_iou_loss += (1 - ious).mean()
                ious_raw = line_iou(
                    line_pred,
                    line_target,
                    self.img_w,
                    length=15,
                    aligned=True,
                )

                # ================== 【增强版可视化】 ==================
                if (
                    current_iter % 50 == 0
                    and torch.distributed.get_rank() == 0
                    and stage == self.refine_layers - 1
                ):
                    visualize_training_progress(
                        current_iter,
                        batch,
                        targets_list,
                        batch_lane_categories,
                        batch_lane_attributes,
                        pos_preds,
                        batch_idx,
                        prior_idx,
                        outputs,
                        ious,
                        loss_y,
                        loss_x,
                        loss_theta,
                        loss_len,
                        target_idx,
                        self.img_h,
                        self.img_w,
                        self.prior_ys,
                        self.enable_category,
                        self.enable_attribute,
                    )

                if stage == self.refine_layers - 1:
                    if (
                        self.enable_category
                        and "category" in outputs
                        and isinstance(batch_lane_categories, torch.Tensor)
                    ):
                        cat_preds = outputs["category"][assigned_mask]
                        cat_targets = batch_lane_categories[batch_idx, target_idx]

                        # Safety check for category targets
                        max_cat = self.cfg.num_lane_categories
                        if (cat_targets >= max_cat).any():
                            # print(f"[LLANetHead] Warning: Invalid category targets detected! Max allowed: {max_cat-1}, Found: {cat_targets.max().item()}")
                            cat_targets = torch.clamp(cat_targets, max=max_cat - 1)
                        if (cat_targets < 0).any():
                            # print(f"[LLANetHead] Warning: Negative category targets detected!")
                            cat_targets = torch.clamp(cat_targets, min=0)

                        # Use log_softmax for NLLLoss if not already applied
                        total_category_loss += (
                            self.category_criterion(
                                F.log_softmax(cat_preds, dim=-1), cat_targets
                            )
                            / cls_norm
                        )

                    if (
                        self.enable_attribute
                        and "attribute" in outputs
                        and isinstance(batch_lane_attributes, torch.Tensor)
                    ):
                        attr_preds = outputs["attribute"][assigned_mask]
                        attr_targets = batch_lane_attributes[batch_idx, target_idx]

                        # Safety check for attribute targets
                        max_attr = self.cfg.num_lr_attributes
                        if (attr_targets >= max_attr).any():
                            # print(f"[LLANetHead] Warning: Invalid attribute targets detected! Max allowed: {max_attr-1}, Found: {attr_targets.max().item()}")
                            attr_targets = torch.clamp(attr_targets, max=max_attr - 1)
                        if (attr_targets < 0).any():
                            # print(f"[LLANetHead] Warning: Negative attribute targets detected!")
                            attr_targets = torch.clamp(attr_targets, min=0)

                        total_attribute_loss += (
                            self.attribute_criterion(
                                F.log_softmax(attr_preds, dim=-1), attr_targets
                            )
                            / cls_norm
                        )

        seg_loss = torch.tensor(0.0, device=device)
        if outputs.get("seg", None) is not None and batch.get("seg", None) is not None:
            seg_pred = outputs["seg"]
            if seg_pred.shape[-2:] != batch["seg"].shape[-2:]:
                seg_pred = F.interpolate(
                    seg_pred,
                    size=batch["seg"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            seg_target = batch["seg"].long()
            # Safety check for seg target
            if seg_pred.shape[1] > 0:
                max_seg = seg_pred.shape[1]
                # invalid if >= max_seg AND != 255
                mask = (seg_target >= max_seg) & (seg_target != 255)
                if mask.any():
                    # print(f"[LLANetHead] Warning: Invalid seg targets detected! Found values >= {max_seg} (excluding 255)")
                    seg_target[mask] = 255  # Ignore invalid labels

            seg_loss = self.criterion(F.log_softmax(seg_pred, dim=1), seg_target)

        if self.detailed_loss_logger is not None and total_stage_count > 0:
            detailed_loss_dict = {
                "loss_start_y": log_ly.item() / total_stage_count,
                "loss_start_x": log_lx.item() / total_stage_count,
                "loss_theta": log_lt.item() / total_stage_count,
                "loss_length": log_ll.item() / total_stage_count,
                "reg_xytl_loss": total_reg_xytl_loss.item() / self.refine_layers,
                "cls_loss": total_cls_loss.item() / self.refine_layers,
                "iou_loss": total_iou_loss.item() / self.refine_layers,
                "loss_category": total_category_loss.item(),
                "loss_attribute": total_attribute_loss.item(),
                "seg_loss": seg_loss.item(),
            }
            self.detailed_loss_logger.log(None, detailed_loss_dict)

        # Warmup weights
        alpha = current_iter / (self.warmup_epochs * self.epoch_per_iter)
        alpha = max(0.0, min(1.0, alpha))

        current_category_loss_weight = self.start_category_loss_weight + alpha * (
            self.category_loss_weight - self.start_category_loss_weight
        )
        current_attribute_loss_weight = self.start_attribute_loss_weight + alpha * (
            self.attribute_loss_weight - self.start_attribute_loss_weight
        )

        losses = {}
        losses["cls_loss"] = total_cls_loss / self.refine_layers * self.cls_loss_weight
        losses["reg_xytl_loss"] = (
            total_reg_xytl_loss / self.refine_layers * self.cfg.xyt_loss_weight
        )
        losses["iou_loss"] = (
            total_iou_loss / self.refine_layers * self.cfg.iou_loss_weight
        )
        losses["loss_category"] = total_category_loss * current_category_loss_weight
        losses["loss_attribute"] = total_attribute_loss * current_attribute_loss_weight
        losses["seg_loss"] = seg_loss * self.cfg.seg_loss_weight

        return losses
