"""
Vectorized Line IoU — functionally identical to CLRNet/line_iou.py.

Bug fix (pairwise mode):
    The original draft used boolean-index assignment
        ovr[invalid_masks] = 0.0
    where ``invalid_masks`` has shape (1, Nt, Nr) but ``ovr`` has shape
    (Np, Nt, Nr).  PyTorch boolean indexing does NOT auto-broadcast at index 0,
    which would raise an IndexError for Np > 1.  We now use element-wise
    multiplication with a float validity mask instead.

Bug fix (clamp):
    ``ovr`` and ``union`` can be negative when lanes do not overlap at all.
    Summing negative ovr values reduces the IoU numerator below zero, which
    produces IoU > 1 after dividing by a small union and then causes the
    dynamic-k topk to select wrong priors.  Clamping at 0 is physically
    correct: negative overlap means no overlap.
"""

import torch


def line_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    img_w: float,
    length: float = 15,
    aligned: bool = True,
) -> torch.Tensor:
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
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length

    if aligned:
        # ── aligned: shapes match exactly ──────────────────────────────────────
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
        # valid_mask shape: same as target (..., Nr)
        valid_mask = ((target >= 0) & (target < img_w)).float()
        ovr = torch.clamp(ovr, min=0.0) * valid_mask
        union = torch.clamp(union, min=0.0) * valid_mask
    else:
        # ── pairwise: broadcast (..., Np, 1, Nr) vs (..., 1, Nt, Nr) ────────
        # pred: (..., Np, Nr) -> (..., Np, 1, Nr)
        # target: (..., Nt, Nr) -> (..., 1, Nt, Nr)
        px1 = px1.unsqueeze(-2)
        px2 = px2.unsqueeze(-2)
        tx1 = tx1.unsqueeze(-3)
        tx2 = tx2.unsqueeze(-3)

        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)

        # valid_mask shape: (..., 1, Nt, Nr)
        target_broad = target.unsqueeze(-3)
        valid_mask = ((target_broad >= 0) & (target_broad < img_w)).float()

        ovr = torch.clamp(ovr, min=0.0) * valid_mask
        union = torch.clamp(union, min=0.0) * valid_mask

    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def liou_loss(
    pred: torch.Tensor, target: torch.Tensor, img_w: float, length: float = 15
) -> torch.Tensor:
    """Mean (1 - LineIoU) loss over aligned pairs."""
    return (1.0 - line_iou(pred, target, img_w, length, aligned=True)).mean()


class CLRNetIoULoss(torch.nn.Module):
    """nn.Module wrapper around Line-IoU loss (CLRNet style)."""

    def __init__(self, loss_weight: float = 1.0, lane_width: float = 15 / 800):
        super().__init__()
        self.loss_weight = loss_weight
        self.lane_width = lane_width

    def _calc_iou(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_width: torch.Tensor,
        target_width: torch.Tensor,
    ) -> torch.Tensor:
        px1 = pred - pred_width
        px2 = pred + pred_width
        tx1 = target - target_width
        tx2 = target + target_width

        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)

        valid_mask = ((target >= 0) & (target < 1.0)).float()
        ovr = torch.clamp(ovr, min=0.0) * valid_mask
        union = torch.clamp(union, min=0.0) * valid_mask
        return ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape == target.shape
        ), "prediction and target must have the same shape!"
        width = torch.ones_like(target) * self.lane_width
        iou = self._calc_iou(pred, target, width, width)
        return (1.0 - iou).mean() * self.loss_weight
