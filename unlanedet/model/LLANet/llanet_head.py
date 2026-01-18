import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..module.head.plaindecoder import PlainDecoder
from ..module.losses.focal_loss import FocalLoss
from ..CLRNet.line_iou import line_iou, liou_loss
from ..CLRNet.dynamic_assign import assign
from ..CLRNet.roi_gather import ROIGather, LinearModule


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
        enable_category=True,
        enable_attribute=True,
        num_lane_categories=15,
        scale_factor=20.0,
    ):
        super(LLANetHead, self).__init__()
        self.cfg = cfg
        self.enable_category = enable_category
        self.enable_attribute = enable_attribute
        self.num_lane_categories = num_lane_categories
        self.scale_factor = scale_factor

        self.img_w = 800 if cfg is None else self.cfg.img_w
        self.img_h = 320 if cfg is None else self.cfg.img_h
        self.num_lr_attributes = 4 if cfg is None else self.cfg.num_lr_attributes

        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim

        # Buffers
        self.register_buffer(
            "sample_x_indexs",
            (
                torch.linspace(0, 1, steps=self.sample_points, dtype=torch.float32)
                * self.n_strips
            ).long(),
        )
        self.register_buffer(
            "prior_feat_ys",
            torch.flip((1 - self.sample_x_indexs.float() / self.n_strips), dims=[-1]),
        )
        self.register_buffer(
            "prior_ys", torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
        )

        self.prior_feat_channels = prior_feat_channels
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
            self.prior_feat_channels,
            self.num_priors,
            self.sample_points,
            self.fc_hidden_dim,
            self.refine_layers,
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

        # Use SUM reduction for vectorized calculation, normalize manually later
        if self.enable_category:
            self.category_criterion = torch.nn.NLLLoss(reduction="sum")
        if self.enable_attribute:
            self.attribute_criterion = torch.nn.NLLLoss(reduction="sum")

        self.init_weights()

    def init_weights(self):
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)
        if self.enable_category:
            for m in self.category_modules.parameters():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m, mean=0.0, std=1e-3)
        if self.enable_attribute:
            for m in self.attribute_layers.parameters():
                nn.init.normal_(m, mean=0.0, std=1e-3)

    def _init_prior_embeddings(self):
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        nn.init.normal_(self.prior_embeddings.weight, mean=0.0, std=0.01)

    def generate_priors_from_embeddings(self):
        device = self.prior_embeddings.weight.device
        priors = torch.zeros((self.num_priors, 6 + self.n_offsets), device=device)
        priors[:, 0] = 0.1
        priors[:, 1] = 0.9
        priors[:, 2] = 0.5
        priors[:, 3] = 0.1
        priors[:, 4] = 0.0
        priors[:, 5] = 50.0
        for i in range(self.num_priors):
            x_pos = (i + 0.5) / self.num_priors * (self.img_w - 1)
            for j in range(self.n_offsets):
                y_pos = (1 - j / (self.n_offsets - 1)) * (self.img_h - 1)
                priors[i, 6 + j] = x_pos / (self.img_w - 1)
        priors_on_featmap = priors[..., 6 + self.sample_x_indexs]
        return priors, priors_on_featmap

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        batch_size = batch_features.shape[0]
        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        prior_ys = self.prior_feat_ys.view(1, 1, -1, 1).expand(
            batch_size, num_priors, -1, 1
        )
        prior_xs = prior_xs * 2.0 - 1.0
        prior_ys = prior_ys * 2.0 - 1.0
        grid = torch.cat((prior_xs, prior_ys), dim=-1)
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
            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs
            )
            prior_features_stages.append(batch_prior_features)
            fc_features = self.roi_gather(
                prior_features_stages, batch_features[stage], stage
            )
            fc_features = (
                fc_features.view(num_priors, batch_size, -1)
                .permute(1, 0, 2)
                .contiguous()
            )
            if stage == self.refine_layers - 1:
                final_fc_features = fc_features
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

            predictions = priors.clone()
            predictions[..., :2] = cls_logits
            predictions[..., 2:5] += reg[..., :3]
            predictions[..., 5] = reg[..., 3]

            def unsqueeze_repeat(t):
                return t.unsqueeze(2).expand(-1, -1, self.n_offsets)

            pred_start_x = unsqueeze_repeat(predictions[..., 3])
            pred_theta = unsqueeze_repeat(predictions[..., 4])
            delta_y = 1.0 - prior_ys_expanded - unsqueeze_repeat(predictions[..., 2])
            cot_theta = 1.0 / torch.tan(pred_theta * math.pi + 1e-5)
            predictions[..., 6:] = (
                (pred_start_x * (self.img_w - 1)) + (delta_y * self.img_h * cot_theta)
            ) / (self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]
            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
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
                outputs["category"] = category_out
            if attribute_out is not None:
                outputs["attribute"] = attribute_out
            return outputs
        else:
            ret_list = []
            batch_size = final_preds.shape[0]
            for i in range(batch_size):
                sample_ret = {"lane_lines": final_preds[i]}
                if category_out is not None:
                    sample_ret["category"] = category_out[i]
                if attribute_out is not None:
                    sample_ret["attribute"] = attribute_out[i]
                ret_list.append(sample_ret)
            return ret_list

    def loss(self, outputs, batch):
        """Vectorized Loss Calculation"""
        predictions_lists = outputs["predictions_lists"]
        targets = batch["lane_line"]
        batch_lane_categories = batch.get("lane_categories", None)
        batch_lane_attributes = batch.get("lane_attributes", None)

        cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        device = self.priors.device

        total_cls_loss = 0.0
        total_reg_xytl_loss = 0.0
        total_iou_loss = 0.0
        total_category_loss = 0.0
        total_attribute_loss = 0.0

        batch_size = len(targets)

        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]

            # --- Vectorized Collection Buffers ---
            stage_reg_preds = []
            stage_reg_targets = []
            stage_pred_lines = []
            stage_target_lines = []

            stage_cls_targets = torch.zeros(
                (batch_size, self.num_priors), dtype=torch.long, device=device
            )
            stage_cat_preds, stage_cat_targets = [], []
            stage_attr_preds, stage_attr_targets = [], []

            num_positives = 0

            # 1. Dynamic Matching (Optimized Loop)
            for b in range(batch_size):
                prediction = predictions_list[b]
                target = targets[b][targets[b][:, 1] == 1]  # Valid GT

                if len(target) > 0:
                    with torch.no_grad():
                        rows, cols = assign(prediction, target, self.img_w, self.img_h)

                    if len(rows) > 0:
                        stage_cls_targets[b, rows] = 1
                        num_positives += len(rows)

                        reg_pred = prediction[rows, 2:6]
                        reg_target = target[cols, 2:6].clone()

                        with torch.no_grad():
                            pred_starts = torch.clamp(
                                (reg_pred[:, 0] * self.n_strips).round().long(),
                                0,
                                self.n_strips,
                            )
                            target_starts = (
                                (reg_target[:, 0] * self.n_strips).round().long()
                            )
                            reg_target[:, 3] -= (
                                pred_starts - target_starts
                            ) / self.n_strips

                        stage_reg_preds.append(reg_pred)
                        stage_reg_targets.append(reg_target)
                        stage_pred_lines.append(prediction[rows, 6:])
                        stage_target_lines.append(target[cols, 6:])

                        if stage == self.refine_layers - 1:
                            if self.enable_category and "category" in outputs:
                                stage_cat_preds.append(outputs["category"][b, rows])
                                stage_cat_targets.append(batch_lane_categories[b, cols])
                            if self.enable_attribute and "attribute" in outputs:
                                stage_attr_preds.append(outputs["attribute"][b, rows])
                                stage_attr_targets.append(
                                    batch_lane_attributes[b, cols]
                                )

            # 2. Vectorized Loss Computation
            # Normalize by total positives to match "Mean" logic (batch-wise average)
            cls_norm = max(num_positives, 1.0)
            stage_cls_loss = cls_criterion(
                predictions_list[..., :2].view(-1, 2), stage_cls_targets.view(-1)
            ).sum()
            total_cls_loss += stage_cls_loss / cls_norm

            if num_positives > 0:
                cat_reg_preds = torch.cat(stage_reg_preds, dim=0)
                cat_reg_targets = torch.cat(stage_reg_targets, dim=0)

                # Scale to Absolute
                cat_reg_preds_abs = cat_reg_preds.clone()
                cat_reg_preds_abs[:, 0] *= self.n_strips
                cat_reg_preds_abs[:, 1] *= self.img_w - 1
                cat_reg_preds_abs[:, 2] *= 180
                cat_reg_preds_abs[:, 3] *= self.n_strips

                cat_reg_targets_abs = cat_reg_targets.clone()
                cat_reg_targets_abs[:, 0] *= self.n_strips
                cat_reg_targets_abs[:, 1] *= self.img_w - 1
                cat_reg_targets_abs[:, 2] *= 180
                cat_reg_targets_abs[:, 3] *= self.n_strips

                # Mean Reduction
                total_reg_xytl_loss += F.smooth_l1_loss(
                    cat_reg_preds_abs, cat_reg_targets_abs, reduction="mean"
                )

                cat_pred_lines = torch.cat(stage_pred_lines, dim=0) * (self.img_w - 1)
                cat_target_lines = torch.cat(stage_target_lines, dim=0) * (
                    self.img_w - 1
                )
                total_iou_loss += liou_loss(
                    cat_pred_lines, cat_target_lines, self.img_w, length=15
                )

                if stage == self.refine_layers - 1:
                    if len(stage_cat_preds) > 0:
                        total_category_loss += self.category_criterion(
                            torch.cat(stage_cat_preds).log_softmax(dim=-1),
                            torch.cat(stage_cat_targets),
                        ) / max(
                            len(stage_cat_preds), 1
                        )  # Manual mean if needed, or if criterion is sum

                    if len(stage_attr_preds) > 0:
                        total_attribute_loss += self.attribute_criterion(
                            torch.cat(stage_attr_preds).log_softmax(dim=-1),
                            torch.cat(stage_attr_targets),
                        ) / max(len(stage_attr_preds), 1)

        # 3. Final Weighted Sum
        # 归一化说明：
        # - cls_loss: 已经除以num_positives（匹配到的车道线数），再除以refine_layers得到平均每层的每根车道线损失
        # - reg_xytl_loss: 已经用mean reduction（除以num_positives），再除以refine_layers得到平均每层的每根车道线损失
        # - iou_loss: 已经除以num_positives，再除以refine_layers得到平均每层的每根车道线损失
        # - loss_category: 只在最后一层计算，需要除以batch_size得到每张图的平均损失
        # - loss_attribute: 只在最后一层计算，需要除以batch_size得到每张图的平均损失
        losses = {}
        losses["cls_loss"] = (
            total_cls_loss
            / self.refine_layers
            * self.cfg.cls_loss_weight
        )
        losses["reg_xytl_loss"] = (
            total_reg_xytl_loss
            / self.refine_layers
            * self.cfg.xyt_loss_weight
        )
        losses["iou_loss"] = (
            total_iou_loss
            / self.refine_layers
            * self.cfg.iou_loss_weight
        )
        # category和attribute只在最后一层计算，且已经除以了各自匹配到的数量
        # 这里再除以batch_size得到每张图的平均损失
        losses["loss_category"] = (
            total_category_loss
            / batch_size
            * self.cfg.category_loss_weight
        )
        losses["loss_attribute"] = (
            total_attribute_loss
            / batch_size
            * self.cfg.attribute_loss_weight
        )

        seg_loss = torch.tensor(0.0, device=device)
        if outputs.get("seg", None) is not None and batch.get("seg", None) is not None:
            # PlainDecoder已经将seg上采样到[img_h, img_w]尺寸
            # 如果batch["seg"]尺寸不同才需要上采样
            seg_pred = outputs["seg"]
            if seg_pred.shape[-2:] != batch["seg"].shape[-2:]:
                seg_pred = F.interpolate(
                    seg_pred,
                    size=batch["seg"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            seg_loss = self.criterion(
                F.log_softmax(seg_pred, dim=1), batch["seg"].long()
            )
        losses["seg_loss"] = seg_loss * self.cfg.seg_loss_weight

        return losses
