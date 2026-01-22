import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import random

from ..module.head.plaindecoder import PlainDecoder
from ..module.losses.focal_loss import FocalLoss
from .line_iou import line_iou, liou_loss
from .dynamic_assign import assign
from .roi_gather import ROIGather, LinearModule


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
        detailed_loss_logger=None,
    ):
        super(LLANetHead, self).__init__()
        self.cfg = cfg
        self.enable_category = enable_category
        self.enable_attribute = enable_attribute
        self.num_lane_categories = num_lane_categories
        self.scale_factor = scale_factor
        self.detailed_loss_logger = detailed_loss_logger

        self.img_w = 800 if cfg is None else self.cfg.img_w
        self.img_h = 320 if cfg is None else self.cfg.img_h
        self.num_lr_attributes = 4 if cfg is None else self.cfg.num_lr_attributes

        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_dim = fc_hidden_dim

        # Dynamic weight scheduling parameters
        self.epoch_per_iter = getattr(cfg, "epoch_per_iter", 1)
        self.warmup_epochs = getattr(cfg, "warmup_epochs", 5)

        self.start_cls_loss_weight = getattr(cfg, "start_cls_loss_weight", 0.0)
        self.cls_loss_weight = getattr(cfg, "cls_loss_weight", 2.0)

        self.start_category_loss_weight = getattr(
            cfg, "start_category_loss_weight", 0.0
        )
        self.category_loss_weight = getattr(cfg, "category_loss_weight", 1.0)

        self.start_attribute_loss_weight = getattr(
            cfg, "start_attribute_loss_weight", 0.0
        )
        self.attribute_loss_weight = getattr(cfg, "attribute_loss_weight", 0.5)

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
            torch.linspace(1.0, 0.0, steps=self.sample_points).float(),
        )
        self.register_buffer(
            "prior_ys",
            torch.linspace(1.0, 0.0, steps=self.n_offsets, dtype=torch.float32),
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
            in_channels=self.prior_feat_channels,
            num_priors=self.num_priors,
            sample_points=self.sample_points,
            fc_hidden_dim=self.fc_hidden_dim,
            refine_layers=self.refine_layers,
            mid_channels=self.fc_hidden_dim,
        )

        self.reg_layers = nn.Linear(self.fc_hidden_dim, self.n_offsets + 1 + 2 + 1)
        self.cls_layers = nn.Linear(self.fc_hidden_dim, 2)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_layers.bias, bias_value)

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
            self.category_criterion = torch.nn.NLLLoss(reduction="sum")
        if self.enable_attribute:
            self.attribute_criterion = torch.nn.NLLLoss(reduction="sum")

        self.init_weights()

    def init_weights(self):
        for m in self.cls_layers.parameters():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                if m.bias is not None:
                    nn.init.constant_(m.bias, bias_value)
            else:
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
        # [start_y, start_x, theta]
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)

        # 【关键修复】Prior 初始化策略 (仿照 CLRNet)
        # 不再全部初始化在底部，而是均匀分布在 Y 轴和 X 轴
        # 0 ~ N-1
        with torch.no_grad():
            # Y: 均匀分布 [0.0, 1.0]
            # X: 均匀分布 [0.0, 1.0]
            # Theta: 0.5 (垂直)

            # 分层初始化：底部 Prior 和 左/右 Prior
            # 这里简化为全图均匀分布，这对于 OpenLane 这种多样化场景更稳健

            # Start Y: 0.0 (Top) -> 1.0 (Bottom)
            # 我们让大部分 Prior 集中在底部 (0.5 - 1.0)，少部分在中间
            bottom_priors = int(self.num_priors * 0.7)
            mid_priors = self.num_priors - bottom_priors

            # Bottom Priors: Y=1.0, X=0~1
            self.prior_embeddings.weight[:bottom_priors, 0] = 1.0
            self.prior_embeddings.weight[:bottom_priors, 1] = torch.linspace(
                0, 1, bottom_priors
            )
            self.prior_embeddings.weight[:bottom_priors, 2] = 0.5

            # Mid Priors: Y=0.4~0.9, X=0~1
            self.prior_embeddings.weight[bottom_priors:, 0] = torch.linspace(
                0.4, 0.9, mid_priors
            )
            self.prior_embeddings.weight[bottom_priors:, 1] = torch.linspace(
                0, 1, mid_priors
            )
            self.prior_embeddings.weight[bottom_priors:, 2] = 0.5

    def generate_priors_from_embeddings(self):
        device = self.prior_embeddings.weight.device
        priors = torch.zeros((self.num_priors, 6 + self.n_offsets), device=device)

        # 从 embedding 加载初始参数
        # prediction = embedding.weight
        priors[:, 2:5] = self.prior_embeddings.weight.clone()  # y, x, theta
        priors[:, 5] = 0.3  # Initial length

        # 初始点坐标 (垂直投影)
        # x = start_x (因为 theta=0.5, tan=inf, dx=0)
        for i in range(self.num_priors):
            priors[i, 6:] = priors[i, 3]  # start_x

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

            # 1. Gradient Flow: Calculate Raw Params
            pred_start_y = priors[:, :, 0] + reg[:, :, 0]
            pred_start_x = priors[:, :, 1] + reg[:, :, 1]
            pred_theta = priors[:, :, 2] + reg[:, :, 2]
            pred_len = reg[:, :, 3]

            # 2. Stop Gradient: Clamp for Projection Safety
            clamped_start_y = torch.clamp(pred_start_y, 0.0, 1.0)
            clamped_start_x = torch.clamp(pred_start_x, 0.0, 1.0)
            clamped_theta = pred_theta  # Theta can be outside 0-1 slightly
            clamped_len = torch.clamp(pred_len, 0.0, 1.0)

            # 3. Geometric Projection
            def tran_tensor(t):
                return t.unsqueeze(2).expand(-1, -1, self.n_offsets)

            pred_start_y_exp = tran_tensor(clamped_start_y)
            pred_start_x_exp = tran_tensor(clamped_start_x)
            pred_theta_exp = tran_tensor(clamped_theta)

            coords = (
                pred_start_x_exp * (self.img_w - 1)
                + (
                    (self.prior_ys.repeat(batch_size, num_priors, 1) - pred_start_y_exp)
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
            ret_list = []
            batch_size = final_preds.shape[0]
            if category_out is not None:
                category_out = category_out.view(batch_size, self.num_priors, -1)

            for i in range(batch_size):
                sample_ret = {
                    "lane_lines": self.predictions_to_pred(final_preds[i].unsqueeze(0))
                }
                if category_out is not None:
                    sample_ret["category"] = category_out[i]
                if attribute_out is not None:
                    sample_ret["attribute"] = attribute_out[i]
                ret_list.append(sample_ret)
            return ret_list

    def predictions_to_pred(self, predictions):
        lanes = []
        for lane in predictions:
            lane = lane.detach().cpu().numpy()
            lanes.append(lane)
        return lanes

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

        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_xytl_loss = torch.tensor(0.0, device=device)
        total_iou_loss = torch.tensor(0.0, device=device)
        total_category_loss = torch.tensor(0.0, device=device)
        total_attribute_loss = torch.tensor(0.0, device=device)

        # 统计变量初始化 (Fix NameError)
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

                # ============ 量纲对齐 ============
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

                # 【Length】Direct Regression (No Residual)
                # target_yxtl[:, 3] = target_yxtl[:, 3]

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
                total_reg_xytl_loss += (loss_components * reg_weights).sum() / max(
                    num_positives, 1
                )

                if self.detailed_loss_logger is not None:
                    log_ly += loss_y.mean()
                    log_lx += loss_x.mean()
                    log_lt += loss_theta.mean()
                    log_ll += loss_len.mean()
                    total_stage_count += 1

                # IoU Loss (Target Masking)
                line_pred = pos_preds[:, 6:] * (self.img_w - 1)
                line_target = pos_targets[:, 6:] * (self.img_w - 1)

                # 【Fix】处理 Target 中的无效点 (-1e5)
                # liou_loss 内部通常有 valid_mask 处理，但最好确保传入的有效值
                # 这里假设 liou_loss 处理 -1e5 为 invalid
                ious = line_iou(
                    line_pred, line_target, self.img_w, length=15, aligned=True
                )
                total_iou_loss += (1 - ious).mean()

                # ================== 【增强版可视化】 ==================
                if (
                    current_iter % 50 == 0
                    and torch.distributed.get_rank() == 0
                    and stage == self.refine_layers - 1
                ):
                    import os
                    import cv2
                    import numpy as np

                    save_dir = "debug_vis"
                    os.makedirs(save_dir, exist_ok=True)
                    vis_txt_path = os.path.join(save_dir, f"iter_{current_iter}.txt")

                    # 1. 图像
                    pad = 100
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(
                        1, 3, 1, 1
                    )
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(
                        1, 3, 1, 1
                    )
                    img_tensor = batch["img"][0]
                    img_tensor = img_tensor * std[0] + mean[0]
                    img_np = (
                        (img_tensor.permute(1, 2, 0).cpu().numpy() * 255)
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    img_vis = cv2.copyMakeBorder(
                        img_vis,
                        pad,
                        pad,
                        pad,
                        pad,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )

                    # 2. 绘制 GT Prioirs (蓝色虚线) - 可选
                    # for k in range(min(10, self.num_priors)):
                    #    py = int(priors[0, k, 0].item() * self.img_h) + pad
                    #    px = int(priors[0, k, 1].item() * self.img_w) + pad
                    #    cv2.circle(img_vis, (px, py), 2, (255, 0, 0), -1)

                    # 3. 绘制 GT (绿色)
                    if len(targets_list) > 0:
                        gt_lanes = targets_list[0]
                        gt_cats = (
                            batch_lane_categories[0]
                            if batch_lane_categories is not None
                            else None
                        )
                        gt_attrs = (
                            batch_lane_attributes[0]
                            if batch_lane_attributes is not None
                            else None
                        )

                        for i_gt, lane in enumerate(gt_lanes):
                            if lane[1] == 1:
                                pts_x = lane[6:]
                                points = []
                                for idx_pt, x in enumerate(pts_x):
                                    if x > 0 and x < 1:
                                        y = (
                                            int(
                                                self.img_h
                                                * (1.0 - idx_pt / (self.n_offsets - 1))
                                            )
                                            + pad
                                        )
                                        x_pixel = int(x * self.img_w) + pad
                                        points.append((x_pixel, y))
                                if len(points) > 1:
                                    cv2.polylines(
                                        img_vis,
                                        [np.array(points)],
                                        False,
                                        (0, 255, 0),
                                        2,
                                    )
                                    start_pt = points[0]
                                    gt_c = (
                                        gt_cats[i_gt].item()
                                        if gt_cats is not None
                                        else -1
                                    )
                                    gt_a = (
                                        gt_attrs[i_gt].item()
                                        if gt_attrs is not None
                                        else -1
                                    )
                                    cv2.putText(
                                        img_vis,
                                        f"GT_{i_gt}|C{gt_c}|A{gt_a}",
                                        (start_pt[0], start_pt[1] + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 0),
                                        1,
                                    )

                    # 4. 绘制 Matched Preds & 写 Log
                    b0_mask = batch_idx == 0
                    with open(vis_txt_path, "w") as f:
                        f.write(f"Iteration {current_iter} - Batch 0 Analysis\n")
                        f.write("=" * 60 + "\n")

                        if b0_mask.sum() > 0:
                            b0_pos_preds = pos_preds[b0_mask]
                            b0_priors_idx = prior_idx[b0_mask]

                            b0_cat_logits = (
                                outputs["category"][0][b0_priors_idx]
                                if (self.enable_category and "category" in outputs)
                                else None
                            )
                            b0_attr_logits = (
                                outputs["attribute"][0][b0_priors_idx]
                                if (self.enable_attribute and "attribute" in outputs)
                                else None
                            )

                            b0_iou = ious[b0_mask]
                            b0_ly = loss_y[b0_mask]
                            b0_lx = loss_x[b0_mask]
                            b0_lt = loss_theta[b0_mask]
                            b0_ll = loss_len[b0_mask]

                            b0_tgt_idx = target_idx[b0_mask]

                            for k in range(len(b0_pos_preds)):
                                lane = b0_pos_preds[k].detach().cpu().numpy()
                                score = lane[1]
                                prob = 1 / (1 + np.exp(-score))

                                cat_id = (
                                    torch.argmax(b0_cat_logits[k]).item()
                                    if b0_cat_logits is not None
                                    else -1
                                )
                                attr_id = (
                                    torch.argmax(b0_attr_logits[k]).item()
                                    if b0_attr_logits is not None
                                    else -1
                                )

                                color = (
                                    np.random.randint(50, 255),
                                    np.random.randint(50, 100),
                                    np.random.randint(100, 255),
                                )

                                points_x = lane[6:]
                                points = []
                                for idx_pt, x in enumerate(points_x):
                                    if x > 0 and x < 1:
                                        y = (
                                            int(
                                                self.img_h
                                                * (1.0 - idx_pt / (self.n_offsets - 1))
                                            )
                                            + pad
                                        )
                                        x_pixel = int(x * self.img_w) + pad
                                        points.append((x_pixel, y))

                                if len(points) > 1:
                                    cv2.polylines(
                                        img_vis, [np.array(points)], False, color, 2
                                    )
                                    start_pt = points[0]
                                    cv2.circle(img_vis, start_pt, 4, color, -1)
                                    info = f"P{k}|{prob:.2f}|C{cat_id}"
                                    cv2.putText(
                                        img_vis,
                                        info,
                                        (start_pt[0] + 10, start_pt[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        (0, 0, 0),
                                        2,
                                    )
                                    cv2.putText(
                                        img_vis,
                                        info,
                                        (start_pt[0] + 10, start_pt[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4,
                                        (255, 255, 255),
                                        1,
                                    )

                                gt_id_val = b0_tgt_idx[k].item()
                                f.write(
                                    f"Pred #{k} (Matched GT_{gt_id_val}) | Conf: {prob:.4f} | Cat: {cat_id} | Attr: {attr_id}\n"
                                )
                                f.write(f"  IoU: {b0_iou[k]:.4f}\n")
                                f.write(
                                    f"  Reg Loss -> Y: {b0_ly[k]:.2f}, X: {b0_lx[k]:.2f}, Theta: {b0_lt[k]:.2f}, Len: {b0_ll[k]:.2f}\n"
                                )
                                f.write("-" * 40 + "\n")
                        else:
                            f.write("No positive matches in Batch 0.\n")

                    cv2.imwrite(
                        os.path.join(save_dir, f"vis_iter_{current_iter}.jpg"), img_vis
                    )

                if stage == self.refine_layers - 1:
                    if (
                        self.enable_category
                        and "category" in outputs
                        and isinstance(batch_lane_categories, torch.Tensor)
                    ):
                        cat_preds = outputs["category"][assigned_mask]
                        cat_targets = batch_lane_categories[batch_idx, target_idx]
                        total_category_loss += self.category_criterion(
                            cat_preds.log_softmax(dim=-1), cat_targets
                        ) / max(batch_size, 1)

                    if (
                        self.enable_attribute
                        and "attribute" in outputs
                        and isinstance(batch_lane_attributes, torch.Tensor)
                    ):
                        attr_preds = outputs["attribute"][assigned_mask]
                        attr_targets = batch_lane_attributes[batch_idx, target_idx]
                        total_attribute_loss += self.attribute_criterion(
                            attr_preds.log_softmax(dim=-1), attr_targets
                        ) / max(batch_size, 1)

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
            seg_loss = self.criterion(
                F.log_softmax(seg_pred, dim=1), batch["seg"].long()
            )

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

        current_cls_loss_weight = self.start_cls_loss_weight + alpha * (
            self.cls_loss_weight - self.start_cls_loss_weight
        )
        current_category_loss_weight = self.start_category_loss_weight + alpha * (
            self.category_loss_weight - self.start_category_loss_weight
        )
        current_attribute_loss_weight = self.start_attribute_loss_weight + alpha * (
            self.attribute_loss_weight - self.start_attribute_loss_weight
        )

        losses = {}
        losses["cls_loss"] = (
            total_cls_loss / self.refine_layers * current_cls_loss_weight
        )
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
