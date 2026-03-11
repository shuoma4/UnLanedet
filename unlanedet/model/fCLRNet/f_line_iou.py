"""
Vectorized Line IoU — functionally identical to CLRNet/line_iou.py.

Bug fix (pairwise mode):
    The original draft used boolean-index assignment
        ovr[invalid_masks] = 0.0
    where ``invalid_masks`` has shape (1, Nt, Nr) but ``ovr`` has shape
    (Np, Nt, Nr).  PyTorch boolean indexing does NOT auto-broadcast at index 0,
    which would raise an IndexError for Np > 1.  We now use element-wise
    multiplication with a float validity mask instead.

Bug fix (clamp ovr/union at 0):
    ``ovr`` and ``union`` can be negative when lanes do not overlap at all.
    Summing negative ovr values reduces the IoU numerator below zero.
    Clamping at 0 is physically correct: negative overlap means no overlap.

Bug fix (AMP / FP16 NaN in DivBackward0):
    With AMP enabled (FP16), the epsilon ``1e-9`` underflows to exactly 0.0
    in half-precision arithmetic.  When union_sum is also near-zero in FP16,
    the backward gradient  d(a/b)/db = -a/b^2  becomes NaN (DivBackward0
    crash observed at ~8000 iters).

    Fix: clamp the denominator to a minimum of 1.0 (pixel units) before
    dividing, then use torch.where to zero-out IoU for the all-invalid case.
    This keeps the full autograd graph intact — unlike in-place mask
    assignment on a zeros_like() tensor, which silently detaches from the
    graph and produces zero IoU gradients everywhere.
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
        # ── aligned: shapes match exactly ─────────────────────────────────────
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

    ovr_sum = ovr.sum(dim=-1)
    union_sum = union.sum(dim=-1)

    # ── NaN-safe division (AMP/FP16 compatible) ─────────────────────────────
    # 1e-9 underflows to 0 in FP16, causing DivBackward0 NaN when union_sum
    # is also near-zero.  clamp(min=1.0) is safe because a single valid
    # point contributes union >= 2*length = 30 pixels; it only activates for
    # the degenerate all-invalid case.
    # torch.where keeps both branches in the computation graph so gradients
    # flow correctly through the valid entries.
    safe_union = union_sum.clamp(min=1.0)
    iou_raw = ovr_sum / safe_union
    # Zero out entries where the lane is entirely outside the image.
    iou = torch.where(union_sum > 0, iou_raw, torch.zeros_like(iou_raw))
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

        ovr_sum = ovr.sum(dim=-1)
        union_sum = union.sum(dim=-1)
        safe_union = union_sum.clamp(min=1.0)
        iou_raw = ovr_sum / safe_union
        return torch.where(union_sum > 0, iou_raw, torch.zeros_like(iou_raw))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape == target.shape
        ), "prediction and target must have the same shape!"
        width = torch.ones_like(target) * self.lane_width
        iou = self._calc_iou(pred, target, width, width)
        return (1.0 - iou).mean() * self.loss_weight
