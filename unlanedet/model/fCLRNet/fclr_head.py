"""
FCLRHead — drop-in replacement for CLRHead with a vectorised loss().

Model architecture, forward(), get_lanes(), and predictions_to_pred() are
**unchanged** (fully inherited from CLRHead).  Only loss() is overridden.

Optimisations vs CLRHead.loss()
────────────────────────────────
1. ``dynamic_k_assign`` (f_dynamic_assign.py) — Python for-loop over GT
   indices replaced by argsort + scatter_.  Zero Python round-trips per GT.

2. ``distance_cost`` (f_dynamic_assign.py) — repeat_interleave/cat replaced
   by direct broadcasting; avoids two large intermediate copies.

3. Loss batching — matched (pred, target) pairs from ALL samples within a
   stage are cat-ed and their regression / IoU losses computed in a **single**
   kernel call instead of once per sample.

4. FocalLoss — still computed per sample (needed for correct normalisation)
   but uses the vectorised assign result.

The public interface (loss signature, return dict keys) is identical to
CLRHead so trainers require no changes.
"""
import torch
import torch.nn.functional as F

from ..CLRNet.clr_head import CLRHead
from ..module.losses import FocalLoss
from .f_dynamic_assign import assign
from .f_line_iou import liou_loss


class FCLRHead(CLRHead):
    """
    Architecturally identical to CLRHead.
    Only ``loss()`` is overridden with a vectorised implementation.
    """

    def loss(
        self,
        output,
        batch,
        cls_loss_weight:  float = 2.0,
        xyt_loss_weight:  float = 0.5,
        iou_loss_weight:  float = 2.0,
        seg_loss_weight:  float = 1.0,
    ):
        # ── Weight overrides from config ─────────────────────────────────────
        cfg = self.cfg
        if "cls_loss_weight"  in cfg: cls_loss_weight  = cfg.cls_loss_weight
        if "xyt_loss_weight"  in cfg: xyt_loss_weight  = cfg.xyt_loss_weight
        if "iou_loss_weight"  in cfg: iou_loss_weight  = cfg.iou_loss_weight
        if "seg_loss_weight"  in cfg: seg_loss_weight  = cfg.seg_loss_weight

        predictions_lists = output["predictions_lists"]
        targets           = batch["lane_line"].clone()
        batch_size        = targets.shape[0]
        device            = self.priors.device

        cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        cls_loss      = torch.tensor(0.0, device=device)
        reg_xytl_loss = torch.tensor(0.0, device=device)
        iou_loss      = torch.tensor(0.0, device=device)

        # Pre-filter valid GT lanes for every sample  (list of tensors, may be empty)
        targets_valid = [t[t[:, 1] == 1] for t in targets]

        # ─────────────────────────────────────────────────────────────────────
        # Stage loop  (3 stages — fusing across stages is not possible because
        # each stage has its own set of refined priors)
        # ─────────────────────────────────────────────────────────────────────
        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]  # (B, num_priors, D)

            # Accumulators for batched loss computation
            cls_preds_list   = []   # (num_priors, 2) per sample
            cls_targets_list = []   # (num_priors,)  long  per sample
            cls_norm_list    = []   # normalisation denominator per sample

            reg_yxtl_list = []      # matched regression predictions
            tgt_yxtl_list = []      # matched regression targets
            reg_pred_list = []      # matched x-coord predictions
            reg_tgt_list  = []      # matched x-coord targets

            # ── Per-sample assignment ─────────────────────────────────────────
            # Cannot be fully fused: each sample has a variable number of GTs.
            for i in range(batch_size):
                predictions = predictions_list[i]   # (num_priors, D)
                target      = targets_valid[i]       # (num_valid_i, D)
                num_priors  = predictions.shape[0]
                cls_pred    = predictions[:, :2]     # (num_priors, 2)

                if len(target) == 0:
                    # All-negative sample — only cls loss, no reg/iou
                    cls_preds_list.append(cls_pred)
                    cls_targets_list.append(predictions.new_zeros(num_priors).long())
                    cls_norm_list.append(1.0)
                    continue

                # ── Vectorised assignment ─────────────────────────────────
                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h
                    )

                # Classification
                cls_target = predictions.new_zeros(num_priors).long()
                cls_target[matched_row_inds] = 1
                cls_preds_list.append(cls_pred)
                cls_targets_list.append(cls_target)
                cls_norm_list.append(float(target.shape[0]))

                # Regression — collect matched pairs for batched loss later
                reg_yxtl = predictions[matched_row_inds, 2:6].clone()
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= self.img_w - 1
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                tgt_yxtl = target[matched_col_inds, 2:6].clone()

                # Length correction (no-grad)
                with torch.no_grad():
                    pred_starts = (
                        (predictions[matched_row_inds, 2] * self.n_strips)
                        .round().long().clamp(0, self.n_strips)
                    )
                    tgt_starts = (
                        (target[matched_col_inds, 2] * self.n_strips).round().long()
                    )
                    tgt_yxtl[:, -1] -= (pred_starts - tgt_starts)

                tgt_yxtl[:, 0] *= self.n_strips
                tgt_yxtl[:, 2] *= 180

                reg_pred    = predictions[matched_row_inds, 6:].clone() * (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()

                reg_yxtl_list.append(reg_yxtl)
                tgt_yxtl_list.append(tgt_yxtl)
                reg_pred_list.append(reg_pred)
                reg_tgt_list.append(reg_targets)

            # ── Focal loss for this stage (per-sample, correct normalisation) ─
            for cls_pred, cls_target, norm in zip(
                cls_preds_list, cls_targets_list, cls_norm_list
            ):
                cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum() / norm

            # ── Batched regression + IoU loss for this stage ──────────────────
            # Cat all matched pairs from every sample → single GPU kernel
            if reg_yxtl_list:
                batch_reg_yxtl = torch.cat(reg_yxtl_list, dim=0)  # (M, 4)
                batch_tgt_yxtl = torch.cat(tgt_yxtl_list, dim=0)
                batch_reg_pred = torch.cat(reg_pred_list, dim=0)   # (M, Nr)
                batch_reg_tgt  = torch.cat(reg_tgt_list,  dim=0)

                reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(
                    batch_reg_yxtl, batch_tgt_yxtl, reduction="none"
                ).mean()

                iou_loss = iou_loss + liou_loss(
                    batch_reg_pred, batch_reg_tgt, self.img_w, length=15
                )

        # ── Segmentation loss (unchanged) ─────────────────────────────────────
        seg_loss = self.criterion(
            F.log_softmax(output["seg"], dim=1), batch["seg"].long()
        )

        # ── Normalise ─────────────────────────────────────────────────────────
        norm = float(batch_size * self.refine_layers)
        cls_loss      /= norm
        reg_xytl_loss /= norm
        iou_loss      /= norm

        return {
            "cls_loss":      cls_loss      * cls_loss_weight,
            "reg_xytl_loss": reg_xytl_loss * xyt_loss_weight,
            "seg_loss":      seg_loss      * seg_loss_weight,
            "iou_loss":      iou_loss      * iou_loss_weight,
        }
