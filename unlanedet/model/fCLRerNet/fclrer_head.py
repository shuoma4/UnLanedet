"""
FCLRerHead — drop-in replacement for CLRerHead with a vectorised loss().

Model architecture, forward(), get_lanes(), and predictions_to_pred() are
**unchanged** (fully inherited from CLRerHead).  Only loss() is overridden.

Optimisations vs CLRerHead.loss()
────────────────────────────────
1. ``dynamic_k_assign`` (f_dynamic_assign.py) — Python for-loop over GT
   indices replaced by argsort + scatter_.  Zero Python round-trips per GT.

2. ``distance_cost`` (f_dynamic_assign.py) — repeat_interleave/cat replaced
   by direct broadcasting; avoids two large intermediate copies.

3. Loss batching — Assignments and losses are computed for the entire batch
   in parallel using Batched Assignment and index-based aggregation.
   Kernel launches are reduced by factor of B (Batch Size).

4. FocalLoss — computed on batched tensors.

The public interface (loss signature, return dict keys) is identical to
CLRerHead so trainers require no changes.
"""

import torch
import torch.nn.functional as F

from ..CLRerNet.head import CLRerHead
from ..module.losses import FocalLoss
from .f_dynamic_assign import assign
from .f_lane_iou import LaneIoUCost, LaneIoULoss, CLRNetIoUCost


class FCLRerHead(CLRerHead):
    """
    Architecturally identical to CLRerHead.
    Only ``loss()`` is overridden with a vectorised implementation.
    """

    def loss(
        self,
        output,
        batch,
        cls_loss_weight: float = 2.0,
        xyt_loss_weight: float = 0.5,
        iou_loss_weight: float = 2.0,
        seg_loss_weight: float = 1.0,
    ):
        # ── Weight overrides from config ─────────────────────────────────────
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
        targets = batch['lane_line'].clone()
        batch_size = targets.shape[0]
        device = self.priors.device

        cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        cls_loss = torch.tensor(0.0, device=device)
        reg_xytl_loss = torch.tensor(0.0, device=device)
        iou_loss = torch.tensor(0.0, device=device)

        valid_mask = targets[..., 1] == 1
        num_gt_per_img = valid_mask.sum(dim=1).float()
        num_gt_per_img[num_gt_per_img == 0] = 1.0

        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]  # (B, num_priors, D)
            num_priors = predictions_list.shape[1]

            # ── Batched Assignment ───────────────────────────────────────────
            with torch.no_grad():
                # matching_matrix: (B, Np, Nt)
                matching_matrix = assign(predictions_list, targets, self.img_w, self.img_h, valid_mask=valid_mask)

            # Extract matched indices: (N_match, 3) -> (b, p, t)
            matched_indices = matching_matrix.nonzero(as_tuple=False)
            batch_idx = matched_indices[:, 0]
            prior_idx = matched_indices[:, 1]
            gt_idx = matched_indices[:, 2]

            # ── Classification Loss ──────────────────────────────────────────
            # Construct target tensor (B, Np)
            cls_target = torch.zeros((batch_size, num_priors), dtype=torch.long, device=device)
            cls_target[batch_idx, prior_idx] = 1

            # Predictions: (B, Np, 2) -> (B, 2, Np) for FocalLoss
            cls_pred = predictions_list[..., :2].permute(0, 2, 1)

            # FocalLoss returns (B, Np) with reduction='none'
            focal_loss = cls_criterion(cls_pred, cls_target)

            # Sum over priors per image, then normalize by num_gt_per_img, then sum over batch
            # This matches CLRNet's per-image normalization logic
            cls_loss_stage = (focal_loss.sum(dim=1) / num_gt_per_img).sum()
            cls_loss += cls_loss_stage

            # ── Regression & IoU Loss ────────────────────────────────────────
            if len(batch_idx) > 0:
                # Extract matched predictions and targets
                matched_preds = predictions_list[batch_idx, prior_idx]  # (N_match, D)
                matched_targets = targets[batch_idx, gt_idx]  # (N_match, D)

                # 1. Reg XYTL
                reg_yxtl = matched_preds[:, 2:6].clone().float()
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= self.img_w - 1
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                tgt_yxtl = matched_targets[:, 2:6].clone().float()

                # Length correction
                with torch.no_grad():
                    pred_starts = (matched_preds[:, 2] * self.n_strips).round().long().clamp(0, self.n_strips)
                    tgt_starts = (matched_targets[:, 2] * self.n_strips).round().long()
                    tgt_yxtl[:, -1] -= pred_starts - tgt_starts

                tgt_yxtl[:, 0] *= self.n_strips
                tgt_yxtl[:, 2] *= 180

                # Smooth L1 Loss (N_match, 4) -> Mean over 4 params -> (N_match,)
                loss_reg = F.smooth_l1_loss(reg_yxtl, tgt_yxtl, reduction='none').mean(dim=1)

                # 2. IoU Loss
                reg_pred_xs = matched_preds[:, 6:].clone().float() * (self.img_w - 1)
                reg_target_xs = matched_targets[:, 6:].clone().float()

                # lane_iou expects relative coordinates (N_match, Nr)
                pred_rel = reg_pred_xs / self.img_w
                target_rel = reg_target_xs / self.img_w
                
                iou_loss_func = LaneIoULoss()
                pred_width, target_width = iou_loss_func._calc_lane_width(pred_rel, target_rel)

                # calc_iou computes IoU (N_match,)
                iou_score = iou_loss_func.calc_iou(pred_rel, target_rel, pred_width, target_width)
                loss_iou = 1.0 - iou_score

                # 3. Aggregate per image (Sum of Means)
                # We calculate mean loss for each image separately to match CLRNet behavior
                # count_per_img[b] = Count(matches in image b)
                count_per_img = torch.bincount(batch_idx, minlength=batch_size).float()

                sum_reg_loss = torch.zeros(batch_size, device=device)
                sum_reg_loss.index_add_(0, batch_idx, loss_reg)

                sum_iou_loss = torch.zeros(batch_size, device=device)
                sum_iou_loss.index_add_(0, batch_idx, loss_iou)

                # Handle images with 0 matches (avoid division by zero)
                valid_img_mask = count_per_img > 0

                if valid_img_mask.any():
                    reg_loss_stage = (sum_reg_loss[valid_img_mask] / count_per_img[valid_img_mask]).sum()
                    iou_loss_stage = (sum_iou_loss[valid_img_mask] / count_per_img[valid_img_mask]).sum()

                    reg_xytl_loss += reg_loss_stage
                    iou_loss += iou_loss_stage

        # ── Segmentation loss (unchanged) ─────────────────────────────────────
        seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1), batch['seg'].long())

        # ── Normalise ─────────────────────────────────────────────────────────
        # Normalize by batch_size * refine_layers to get average loss per image per stage
        norm = float(batch_size * self.refine_layers)
        cls_loss /= norm
        reg_xytl_loss /= norm
        iou_loss /= norm

        return {
            'cls_loss': cls_loss * cls_loss_weight,
            'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
            'seg_loss': seg_loss * seg_loss_weight,
            'iou_loss': iou_loss * iou_loss_weight,
        }
