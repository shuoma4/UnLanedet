"""
Vectorized Line IoU — functionally identical to CLRNet/line_iou.py.
All ops were already tensor-wise; this module re-exports them under the
fCLRNet namespace and adds minor clarity improvements (pairwise uses
cleaner broadcasting).
"""
import torch


def line_iou(pred: torch.Tensor,
             target: torch.Tensor,
             img_w: float,
             length: float = 15,
             aligned: bool = True) -> torch.Tensor:
    """
    Compute Line-IoU between predictions and targets.

    Args:
        pred   : (N, Nr) or (Np, Nr)  – x-coordinates of predictions
        target : (N, Nr) or (Nt, Nr)  – x-coordinates of ground-truth
        img_w  : image width (used as validity range)
        length : half-width of the virtual lane band
        aligned: True  → element-wise IoU, shape (N,)
                 False → pairwise  IoU, shape (Np, Nt)
    """
    px1 = pred   - length
    px2 = pred   + length
    tx1 = target - length
    tx2 = target + length

    if aligned:
        # ── aligned: shapes match exactly ────────────────────────────────
        ovr   = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
        invalid_masks = (target < 0) | (target >= img_w)
    else:
        # ── pairwise: broadcast (Np,1,Nr) vs (1,Nt,Nr) ───────────────────
        ovr   = torch.min(px2[:, None, :], tx2[None, :, :]) \
              - torch.max(px1[:, None, :], tx1[None, :, :])
        union = torch.max(px2[:, None, :], tx2[None, :, :]) \
              - torch.min(px1[:, None, :], tx1[None, :, :])
        invalid_masks = (target[None, :, :] < 0) | (target[None, :, :] >= img_w)

    ovr  [invalid_masks] = 0.0
    union[invalid_masks] = 0.0
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def liou_loss(pred: torch.Tensor,
              target: torch.Tensor,
              img_w: float,
              length: float = 15) -> torch.Tensor:
    """Mean (1 - LineIoU) loss over aligned pairs."""
    return (1.0 - line_iou(pred, target, img_w, length, aligned=True)).mean()


class CLRNetIoULoss(torch.nn.Module):
    """nn.Module wrapper around Line-IoU loss (CLRNet style)."""

    def __init__(self, loss_weight: float = 1.0, lane_width: float = 15 / 800):
        super().__init__()
        self.loss_weight = loss_weight
        self.lane_width  = lane_width

    def _calc_iou(self,
                  pred:         torch.Tensor,
                  target:       torch.Tensor,
                  pred_width:   torch.Tensor,
                  target_width: torch.Tensor) -> torch.Tensor:
        px1 = pred   - pred_width
        px2 = pred   + pred_width
        tx1 = target - target_width
        tx2 = target + target_width

        ovr   = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)

        invalid_masks = (target < 0) | (target >= 1.0)
        ovr  [invalid_masks] = 0.0
        union[invalid_masks] = 0.0
        return ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.shape == target.shape, \
            "prediction and target must have the same shape!"
        width = torch.ones_like(target) * self.lane_width
        iou   = self._calc_iou(pred, target, width, width)
        return (1.0 - iou).mean() * self.loss_weight
