import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.llanet.priors.category_weight import CATEGORY_WEIGHT
from unlanedet.layers.ops import nms
from unlanedet.model.fCLRNet.f_line_iou import line_iou
from unlanedet.model.fCLRNet.fclr_head import FCLRHead
from unlanedet.model.module.core.lane import Lane
from unlanedet.model.module.losses import FocalLoss

from .assigner import assign
from .prior import init_priors


class LLANetV1Head(FCLRHead):
    """fCLRNet-based lane head with optional category branch and data-driven priors."""

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
        if cfg is None:
            raise ValueError('cfg must be provided')

        self.cfg = cfg
        self.enable_lane_category = bool(
            getattr(cfg, 'enable_lane_category', getattr(cfg, 'enable_category_head', False))
        )
        self.category_head_type = getattr(cfg, 'category_head_type', 'prototype')
        self.num_lane_categories = int(getattr(cfg, 'num_lane_categories', len(CATEGORY_WEIGHT)))
        self.category_loss_weight = float(getattr(cfg, 'category_loss_weight', 1.0))
        self.category_scale_factor = float(getattr(cfg, 'category_scale_factor', 20.0))
        self.enable_supcon = bool(getattr(cfg, 'enable_supcon', False))
        self.lambda_con = float(getattr(cfg, 'lambda_con', 0.5)) if self.enable_supcon else 0.0
        self.tau_con = float(getattr(cfg, 'tau_con', 0.07))
        self.combined_alpha = float(getattr(cfg, 'combined_alpha', 0.5))
        self.con_loss_weight = float(getattr(cfg, 'con_loss_weight', 0.1))
        self.prior_statistics = None

        super().__init__(
            num_points=num_points,
            prior_feat_channels=prior_feat_channels,
            fc_hidden_dim=fc_hidden_dim,
            num_priors=num_priors,
            num_fc=num_fc,
            refine_layers=refine_layers,
            sample_points=sample_points,
            cfg=cfg,
        )

        if self.enable_lane_category:
            self.category_modules = nn.Sequential(
                nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
            )
            if self.category_head_type in ['prototype', 'combined']:
                self.category_prototypes = nn.Parameter(torch.randn(self.num_lane_categories, self.fc_hidden_dim))
                nn.init.orthogonal_(self.category_prototypes)
                if self.category_head_type == 'combined':
                    self.category_tau = nn.Parameter(torch.tensor(self.category_scale_factor, dtype=torch.float32))
            if self.category_head_type in ['linear', 'combined']:
                self.category_layers = nn.Linear(self.fc_hidden_dim, self.num_lane_categories)
                nn.init.normal_(self.category_layers.weight, mean=0.0, std=1e-3)
                if self.category_layers.bias is not None:
                    nn.init.constant_(self.category_layers.bias, 0.0)

            category_weights = torch.tensor(CATEGORY_WEIGHT, dtype=torch.float32)
            if len(category_weights) < self.num_lane_categories:
                category_weights = F.pad(
                    category_weights,
                    (0, self.num_lane_categories - len(category_weights)),
                    value=1.0,
                )
            elif len(category_weights) > self.num_lane_categories:
                category_weights = category_weights[: self.num_lane_categories]
            self.register_buffer('category_weights_buf', category_weights)
            self.category_criterion = nn.CrossEntropyLoss(weight=self.category_weights_buf, reduction='sum')

    def _init_prior_embeddings(self):
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        self.prior_statistics = init_priors(self.prior_embeddings, self.cfg, self.img_w, self.img_h, self.num_priors)

    def generate_priors_from_embeddings(self):
        predictions = self.prior_embeddings.weight
        priors = predictions.new_zeros((self.num_priors, 2 + 2 + 2 + self.n_offsets), device=predictions.device)
        priors[:, 2:5] = predictions.clone()

        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().repeat(1, self.n_offsets) * (self.img_w - 1)
            + (
                (
                    1
                    - self.prior_ys.repeat(self.num_priors, 1)
                    - priors[:, 2].unsqueeze(1).clone().repeat(1, self.n_offsets)
                )
                * self.img_h
                / torch.tan(priors[:, 4].unsqueeze(1).clone().repeat(1, self.n_offsets) * math.pi + 1e-5)
            )
        ) / (self.img_w - 1)

        priors_on_featmap = priors.clone()[..., 6 + self.sample_x_indexs]
        return priors, priors_on_featmap

    def _category_forward(self, flat_features, batch_size, num_priors):
        # 采用勾子缩放反传给 Backbone 的梯度 (x0.1)：
        # 既允许 Backbone 学习部分语义特征(恢复分类F1)，又防止 loss_weight(5.0) 导致梯度过大破坏定位(保护检测F1)
        category_features = flat_features.clone()
        if self.training and category_features.requires_grad:
            category_features.register_hook(lambda grad: grad * 0.1)

        for layer in self.category_modules:
            category_features = layer(category_features)

        z = F.normalize(category_features, p=2, dim=-1)

        if self.category_head_type == 'combined':
            # 直接使用归一化前的特征喂给 linear，保证 logits 尺度能与 prototype 平起平坐
            logits_linear = self.category_layers(category_features)
            logits_proto = self.category_tau * torch.matmul(
                z,
                F.normalize(self.category_prototypes, p=2, dim=-1).transpose(0, 1)
            )
            logits = self.combined_alpha * logits_linear + (1 - self.combined_alpha) * logits_proto
        elif self.category_head_type == 'prototype':
            logits = self.category_scale_factor * torch.matmul(
                z,
                F.normalize(self.category_prototypes, p=2, dim=-1).transpose(0, 1),
            )
        else:
            logits = self.category_layers(category_features)
        return logits.reshape(batch_size, num_priors, -1), z.reshape(batch_size, num_priors, -1)

    def forward(self, x, **kwargs):
        batch_features = list(x[len(x) - self.refine_layers :])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]

        if self.training:
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()

        priors = self.priors.repeat(batch_size, 1, 1)
        priors_on_featmap = self.priors_on_featmap.repeat(batch_size, 1, 1)

        predictions_lists = []
        prior_features_stages = []
        final_flat_features = None

        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])
            batch_prior_features = self.pool_prior_features(batch_features[stage], num_priors, prior_xs)
            prior_features_stages.append(batch_prior_features)

            fc_features = self.roi_gather(prior_features_stages, batch_features[stage], stage)
            flat_features = fc_features.view(num_priors, batch_size, -1).reshape(batch_size * num_priors, self.fc_hidden_dim)
            if stage == self.refine_layers - 1:
                final_flat_features = flat_features

            cls_features = flat_features.clone()
            reg_features = flat_features.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)

            cls_logits = self.cls_layers(cls_features).reshape(batch_size, -1, 2)
            reg = self.reg_layers(reg_features).reshape(batch_size, -1, self.reg_layers.out_features)

            predictions = priors.clone()
            predictions[:, :, :2] = cls_logits
            predictions[:, :, 2:5] += reg[:, :, :3]
            predictions[:, :, 5] = reg[:, :, 3]

            def tran_tensor(t):
                return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)

            predictions[..., 6:] = (
                tran_tensor(predictions[..., 3]) * (self.img_w - 1)
                + (
                    (1 - self.prior_ys.repeat(batch_size, num_priors, 1) - tran_tensor(predictions[..., 2]))
                    * self.img_h
                    / torch.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5)
                )
            ) / (self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]
            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        output = {
            'predictions_lists': predictions_lists,
            'distill_cls_logits': predictions_lists[-1][..., :2],
        }

        if self.training:
            seg_features = torch.cat(
                [
                    F.interpolate(
                        feature,
                        size=[batch_features[-1].shape[2], batch_features[-1].shape[3]],
                        mode='bilinear',
                        align_corners=False,
                    )
                    for feature in batch_features
                ],
                dim=1,
            )
            seg = self.seg_decoder(seg_features)
            output.update(**seg)

        if self.enable_lane_category and final_flat_features is not None:
            output['category'], output['category_z'] = self._category_forward(final_flat_features, batch_size, num_priors)

        output['distill_features'] = (
            final_flat_features.reshape(batch_size, num_priors, -1) if final_flat_features is not None else None
        )
        if not self.training:
            output['last_pred_lanes'] = predictions_lists[-1]
        return output

    def _lane_category_targets(self, batch, device):
        lane_categories = batch.get('lane_categories', None)
        if lane_categories is None:
            return None
        if torch.is_tensor(lane_categories):
            return lane_categories.to(device)
        return torch.as_tensor(lane_categories, dtype=torch.long, device=device)

    def loss(
        self,
        output,
        batch,
        cls_loss_weight: float = 2.0,
        xyt_loss_weight: float = 0.5,
        iou_loss_weight: float = 2.0,
        seg_loss_weight: float = 1.0,
    ):
        cfg = self.cfg
        if 'cls_loss_weight' in cfg:
            cls_loss_weight = cfg.cls_loss_weight
        if 'xyt_loss_weight' in cfg:
            xyt_loss_weight = cfg.xyt_loss_weight
        if 'iou_loss_weight' in cfg:
            iou_loss_weight = cfg.iou_loss_weight
        if 'seg_loss_weight' in cfg:
            seg_loss_weight = cfg.seg_loss_weight

        predictions_lists = output['predictions_lists']
        targets = batch['lane_line'].to(self.priors.device).clone()
        valid_mask = targets[..., 1] == 1
        num_gt_per_img = valid_mask.sum(dim=1).float()
        num_gt_per_img[num_gt_per_img == 0] = 1.0
        batch_size = targets.shape[0]
        device = self.priors.device
        lane_categories = self._lane_category_targets(batch, device)

        cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        cls_loss = torch.tensor(0.0, device=device)
        reg_xytl_loss = torch.tensor(0.0, device=device)
        iou_loss = torch.tensor(0.0, device=device)
        category_loss = torch.tensor(0.0, device=device)

        for stage in range(self.refine_layers):
            predictions = predictions_lists[stage]
            matching_matrix = assign(
                predictions,
                targets,
                valid_mask,
                self.img_w,
                self.img_h,
                cfg=self.cfg,
                sample_ys=self.prior_ys,
            )
            if stage == self.refine_layers - 1:
                output['final_matching_matrix'] = matching_matrix
            matched_indices = matching_matrix.nonzero(as_tuple=False)
            if matched_indices.numel() == 0:
                batch_idx = predictions.new_zeros((0,), dtype=torch.long)
                prior_idx = predictions.new_zeros((0,), dtype=torch.long)
                gt_idx = predictions.new_zeros((0,), dtype=torch.long)
            else:
                batch_idx = matched_indices[:, 0]
                prior_idx = matched_indices[:, 1]
                gt_idx = matched_indices[:, 2]

            num_priors = predictions.shape[1]
            cls_target = torch.zeros((batch_size, num_priors), dtype=torch.long, device=device)
            if len(batch_idx) > 0:
                cls_target[batch_idx, prior_idx] = 1
            cls_pred = predictions[..., :2].permute(0, 2, 1)
            focal_loss = cls_criterion(cls_pred, cls_target)
            cls_loss += (focal_loss.sum(dim=1) / num_gt_per_img).sum()

            if len(batch_idx) == 0:
                continue

            matched_preds = predictions[batch_idx, prior_idx]
            matched_targets = targets[batch_idx, gt_idx]

            reg_yxtl = matched_preds[:, 2:6].clone().float()
            reg_yxtl[:, 0] *= self.n_strips
            reg_yxtl[:, 1] *= self.img_w - 1
            reg_yxtl[:, 2] *= 180
            reg_yxtl[:, 3] *= self.n_strips

            target_yxtl = matched_targets[:, 2:6].clone().float()
            with torch.no_grad():
                pred_starts = (matched_preds[:, 2] * self.n_strips).round().long().clamp(0, self.n_strips)
                target_starts = (matched_targets[:, 2] * self.n_strips).round().long()
                target_yxtl[:, -1] -= pred_starts - target_starts

            target_yxtl[:, 0] *= self.n_strips
            target_yxtl[:, 2] *= 180

            reg_loss_each = F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean(dim=1)
            reg_loss_sum = torch.zeros(batch_size, device=device, dtype=torch.float32)
            reg_loss_sum.index_add_(0, batch_idx, reg_loss_each)

            pred_xs = matched_preds[:, 6:].clone().float() * (self.img_w - 1)
            target_xs = matched_targets[:, 6:].clone().float()
            iou_each = 1.0 - line_iou(pred_xs, target_xs, self.img_w, length=15, aligned=True)
            iou_loss_sum = torch.zeros(batch_size, device=device)
            iou_loss_sum.index_add_(0, batch_idx, iou_each)

            count_per_img = torch.bincount(batch_idx, minlength=batch_size).float()
            valid_img_mask = count_per_img > 0
            reg_xytl_loss += (reg_loss_sum[valid_img_mask] / count_per_img[valid_img_mask]).sum()
            iou_loss += (iou_loss_sum[valid_img_mask] / count_per_img[valid_img_mask]).sum()

            if self.enable_lane_category and stage == self.refine_layers - 1 and lane_categories is not None and 'category' in output:
                category_logits = output['category'][batch_idx, prior_idx]
                category_targets = lane_categories[batch_idx, gt_idx].long()
                
                L_type = self.category_criterion(
                    category_logits.float(),
                    category_targets,
                ) / max(int(len(batch_idx)), 1)
                
                L_con = torch.tensor(0.0, device=device)
                if (self.category_head_type == 'combined' or (self.category_head_type == 'prototype' and self.enable_supcon)) and len(batch_idx) > 1:
                    z_norm = output['category_z'][batch_idx, prior_idx] # (M, D)
                    M = z_norm.size(0)
                    sim_matrix = torch.matmul(z_norm, z_norm.T) / self.tau_con
                    
                    mask_self = torch.eye(M, device=device).bool()
                    labels = category_targets
                    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_self
                    
                    valid = mask_pos.sum(dim=1) > 0
                    
                    sim_matrix.masked_fill_(mask_self, float('-inf'))
                    log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
                    
                    log_prob_pos = torch.where(mask_pos, log_prob, torch.zeros_like(log_prob))
                    loss_con = -log_prob_pos.sum(dim=1)
                    
                    loss_con = loss_con[valid] / mask_pos[valid].sum(dim=1).float()
                    if valid.sum() > 0:
                        L_con = loss_con.mean()

                    if self.category_head_type == 'combined':
                        category_loss += L_type + self.con_loss_weight * L_con
                    else:
                        category_loss += L_type + self.lambda_con * L_con
                else:
                    category_loss += L_type

        seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1), batch['seg'].long().to(device))
        norm = float(batch_size * self.refine_layers)
        cls_loss /= norm
        reg_xytl_loss /= norm
        iou_loss /= norm

        losses = {
            'cls_loss': cls_loss * cls_loss_weight,
            'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
            'seg_loss': seg_loss * seg_loss_weight,
            'iou_loss': iou_loss * iou_loss_weight,
        }
        if self.enable_lane_category:
            losses['category_loss'] = category_loss * self.category_loss_weight
        return losses

    def predictions_to_pred(self, predictions, category_logits=None):
        self.prior_ys = self.prior_ys.to(predictions.device)
        self.prior_ys = self.prior_ys.double()
        lanes = []
        category_probs = F.softmax(category_logits, dim=-1) if category_logits is not None else None

        for lane_idx, lane in enumerate(predictions):
            lane_xs = lane[6:]
            start = min(max(0, int(round(lane[2].item() * self.n_strips))), self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            mask = ~(
                (((lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0)).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool)
            )
            lane_xs[end + 1 :] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0].flip(0).double()
            lane_ys = lane_ys.flip(0)
            lane_ys = (lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height) + self.cfg.cut_height) / self.cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            metadata = {'start_x': lane[3], 'start_y': lane[2], 'conf': lane[1]}
            if category_probs is not None:
                category_score, category_id = category_probs[lane_idx].max(dim=-1)
                metadata['category_id'] = int(category_id.item())
                metadata['category_score'] = float(category_score.item())
            lanes.append(Lane(points=points.cpu().numpy(), metadata=metadata))
        return lanes

    def get_lanes(self, output, as_lanes=True):
        softmax = nn.Softmax(dim=1)
        if isinstance(output, dict):
            output_predictions = output['last_pred_lanes']
            category_predictions = output.get('category')
        else:
            output_predictions = output
            category_predictions = None

        decoded = []
        for batch_idx, predictions in enumerate(output_predictions):
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= self.cfg.test_parameters.conf_threshold
            predictions = predictions[keep_inds]
            category_logits = category_predictions[batch_idx][keep_inds] if category_predictions is not None else None

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            nms_predictions = predictions.detach().clone()
            nms_predictions = torch.cat([nms_predictions[..., :4], nms_predictions[..., 5:]], dim=-1)
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[..., 5:] = nms_predictions[..., 5:] * (self.img_w - 1)

            keep, num_to_keep, _ = nms(
                nms_predictions,
                scores[keep_inds],
                overlap=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.max_lanes,
            )
            keep = keep[:num_to_keep]
            predictions = predictions[keep]
            category_logits = category_logits[keep] if category_logits is not None else None

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = torch.round(predictions[:, 5] * self.n_strips)
            decoded.append(self.predictions_to_pred(predictions, category_logits) if as_lanes else predictions)
        return decoded
