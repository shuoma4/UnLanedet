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

        # 设置分类层 bias，使得初始正样本概率极低（0.01），避免 Mode Collapse
        # bias = -log((1-p)/p), p=0.01 => bias ≈ -4.59
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

        # Use SUM reduction for vectorized calculation, normalize manually later
        if self.enable_category:
            self.category_criterion = torch.nn.NLLLoss(reduction="sum")
        if self.enable_attribute:
            self.attribute_criterion = torch.nn.NLLLoss(reduction="sum")

        self.init_weights()

    def init_weights(self):
        import math

        # 分类层初始化：设置 bias 使得初始正样本概率极低（0.01），避免 Mode Collapse
        for m in self.cls_layers.parameters():
            if isinstance(m, nn.Linear):
                # Weight 使用小的正态分布
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                # Bias 设置使得初始预测概率约为 0.01
                # bias = -log((1-p)/p), p=0.01 => bias ≈ -4.59
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(m.bias, bias_value)
            else:
                nn.init.normal_(m, mean=0.0, std=1e-3)

        # 回归层初始化
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0.0, std=1e-3)

        # Category 和 Attribute 层初始化
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
        """Vectorized Loss Calculation Optimized for GPU"""
        predictions_lists = outputs["predictions_lists"]
        targets_list = batch["lane_line"]  # List of [N, Dim] tensors
        batch_lane_categories = batch.get("lane_categories", None)
        batch_lane_attributes = batch.get("lane_attributes", None)

        cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        device = self.priors.device
        batch_size = len(targets_list)

        # ============================================================
        # 1. 预处理 Targets: Pad to Tensor (B, Max_Lanes, Dim)
        # ============================================================
        # 找到当前 Batch 中最大的车道线数量
        max_lanes = 0
        target_dim = 0
        if len(targets_list) > 0:
            lengths = [len(t) for t in targets_list]
            max_lanes = max(lengths) if lengths else 0
            if max_lanes > 0:
                target_dim = targets_list[0].shape[1]

        # 至少保证有维度，避免报错
        max_lanes = max(max_lanes, 1)
        if target_dim == 0:
            target_dim = 6 + 72  # 默认维度防止空数据报错

        # 初始化 Batch Tensor
        batch_targets = torch.zeros((batch_size, max_lanes, target_dim), device=device)
        batch_masks = torch.zeros(
            (batch_size, max_lanes), device=device
        )  # 1 for Valid, 0 for Padding

        # 填充数据
        for i, t in enumerate(targets_list):
            num_t = t.shape[0]
            if num_t > 0:
                # 过滤有效线 (通常 index 1 是 valid flag)
                # 假设 t 的格式与原始代码一致，其中 t[:, 1] == 1 表示有效
                valid_mask = t[:, 1] == 1
                valid_t = t[valid_mask]

                num_valid = valid_t.shape[0]
                if num_valid > 0:
                    # 如果超过 max_lanes (理论上不会，但为了安全)
                    num_valid = min(num_valid, max_lanes)
                    batch_targets[i, :num_valid] = valid_t[:num_valid]
                    batch_masks[i, :num_valid] = 1

        # ============================================================
        # 2. 向量化训练循环
        # ============================================================
        total_cls_loss = 0.0
        total_reg_xytl_loss = 0.0
        total_iou_loss = 0.0
        total_category_loss = 0.0
        total_attribute_loss = 0.0

        # 导入新的向量化 assign
        from ..CLRNet.dynamic_assign import assign

        for stage in range(self.refine_layers):
            predictions = predictions_lists[stage]  # (B, Num_Priors, Dim)

            # --- ONE-SHOT ASSIGNMENT FOR THE WHOLE BATCH ---
            # assigned_mask: (B, Num_Priors) Bool - True if prior matched
            # assigned_ids:  (B, Num_Priors) Long - Index of matched GT (0 to Max_Lanes-1)
            with torch.no_grad():
                assigned_mask, assigned_ids = assign(
                    predictions, batch_targets, batch_masks, self.img_w, self.img_h
                )

            # --- Classification Loss ---
            # Focal Loss 输入: (N, C) logits, (N) targets
            cls_targets = assigned_mask.long().view(-1)  # Flatten (B*N)
            pred_cls_flat = predictions[..., :2].view(-1, 2)

            # 计算正样本总数
            num_positives = assigned_mask.sum()
            cls_norm = max(num_positives.item(), 1.0)

            stage_cls_loss = cls_criterion(pred_cls_flat, cls_targets).sum()
            total_cls_loss += stage_cls_loss / cls_norm

            if num_positives > 0:
                # --- 选取正样本进行回归 ---
                # predictions: (Num_Positives, Dim)
                pos_preds = predictions[assigned_mask]

                # 获取对应的 GT
                # batch_idx, prior_idx 用于定位
                batch_idx, prior_idx = torch.where(assigned_mask)
                target_idx = assigned_ids[batch_idx, prior_idx]  # (Num_Positives,)

                pos_targets = batch_targets[
                    batch_idx, target_idx
                ]  # (Num_Positives, Dim)

                # --- Regression Loss (XYTL) ---
                reg_pred = pos_preds[:, 2:6]
                reg_target = pos_targets[:, 2:6].clone()

                # 调整 Target Start X (保持原逻辑)
                with torch.no_grad():
                    pred_starts = torch.clamp(
                        (reg_pred[:, 0] * self.n_strips).round().long(),
                        0,
                        self.n_strips,
                    )
                    target_starts = (reg_target[:, 0] * self.n_strips).round().long()
                    reg_target[:, 3] -= (pred_starts - target_starts) / self.n_strips

                # 将 Theta 从 [0, 1] 转换为角度，再转换为 sin/cos 编码以解决周期性
                # Theta: [0, 1] -> [0, 180] -> [0, pi] -> sin/cos
                theta_pred_rad = reg_pred[:, 2] * math.pi  # 转换为弧度 [0, pi]
                theta_target_rad = reg_target[:, 2] * math.pi

                # 计算 sin/cos
                theta_pred_sin = torch.sin(theta_pred_rad)
                theta_pred_cos = torch.cos(theta_pred_rad)
                theta_target_sin = torch.sin(theta_target_rad)
                theta_target_cos = torch.cos(theta_target_rad)

                # 还原到绝对坐标，但 Theta 使用 sin/cos
                reg_pred_abs = torch.stack(
                    [
                        reg_pred[:, 0] * self.n_strips,  # X
                        reg_pred[:, 1] * (self.img_w - 1),  # Y
                        theta_pred_sin,  # Theta sin
                        theta_pred_cos,  # Theta cos
                        reg_pred[:, 3] * self.n_strips,  # Length
                    ],
                    dim=1,
                )

                reg_target_abs = torch.stack(
                    [
                        reg_target[:, 0] * self.n_strips,
                        reg_target[:, 1] * (self.img_w - 1),
                        theta_target_sin,
                        theta_target_cos,
                        reg_target[:, 3] * self.n_strips,
                    ],
                    dim=1,
                )

                total_reg_xytl_loss += F.smooth_l1_loss(
                    reg_pred_abs, reg_target_abs, reduction="mean"
                )

                # --- IoU Loss ---
                # pos_preds index 6 开始是点集
                line_pred = pos_preds[:, 6:] * (self.img_w - 1)
                line_target = pos_targets[:, 6:] * (self.img_w - 1)

                total_iou_loss += liou_loss(
                    line_pred, line_target, self.img_w, length=15
                )

                # --- Category & Attribute Loss (仅在最后一层) ---
                if stage == self.refine_layers - 1:
                    # Category
                    if self.enable_category and "category" in outputs:
                        cat_preds = outputs["category"][assigned_mask]

                        # 处理 Category Targets (Padding check)
                        if isinstance(batch_lane_categories, torch.Tensor):
                            # 如果已经是 Tensor (B, Max_Lanes)，直接 gather
                            cat_targets = batch_lane_categories[batch_idx, target_idx]

                            # 注意：这里除以 batch_size，符合你之前代码的意图
                            total_category_loss += self.category_criterion(
                                cat_preds.log_softmax(dim=-1), cat_targets
                            ) / max(batch_size, 1)

                    # Attribute
                    if self.enable_attribute and "attribute" in outputs:
                        attr_preds = outputs["attribute"][assigned_mask]
                        if isinstance(batch_lane_attributes, torch.Tensor):
                            attr_targets = batch_lane_attributes[batch_idx, target_idx]
                            total_attribute_loss += self.attribute_criterion(
                                attr_preds.log_softmax(dim=-1), attr_targets
                            ) / max(batch_size, 1)

        # 3. 汇总 Loss
        losses = {}
        losses["cls_loss"] = (
            total_cls_loss / self.refine_layers * self.cfg.cls_loss_weight
        )
        losses["reg_xytl_loss"] = (
            total_reg_xytl_loss / self.refine_layers * self.cfg.xyt_loss_weight
        )
        losses["iou_loss"] = (
            total_iou_loss / self.refine_layers * self.cfg.iou_loss_weight
        )
        losses["loss_category"] = total_category_loss * self.cfg.category_loss_weight
        losses["loss_attribute"] = total_attribute_loss * self.cfg.attribute_loss_weight

        # Seg Loss (保持原样)
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
        losses["seg_loss"] = seg_loss * self.cfg.seg_loss_weight

        return losses
