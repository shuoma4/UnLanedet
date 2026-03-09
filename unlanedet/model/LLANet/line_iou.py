"""
line_iou.py — Line IoU utilities for structured lane detection.

Key design decisions
--------------------
* A point is treated as **invalid** when its coordinate is close to ``invalid_value``
  (default -1e5).  Concretely, any value ≤ ``invalid_value / 2`` is masked out.
  This lets slightly off-screen but geometrically meaningful points (e.g. x = -5)
  participate in the IoU computation, while true sentinel values (-1e5) are excluded.
* **Both** pred and target are sanitised and masked.  The original code only masked
  target invalids, causing silent IoU collapse whenever pred contained sentinels.
* NaN is converted to ``invalid_value`` for both tensors before any arithmetic.
* The element-wise ``clamp(union, min=1e-9)`` that appeared before the sum was
  removed — it was redundant and obscured the intent.  Instead the *sum* is clamped
  to ≥ 1.0 to keep gradients finite even in FP16.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _invalid_threshold(invalid_value: float) -> float:
    """Return the scalar threshold below which a coordinate is considered invalid.

    For the canonical sentinel -1e5 this returns -5e4, safely below any real
    pixel coordinate while staying far above -1e5 itself.
    """
    if invalid_value < 0:
        return invalid_value * 0.5  # e.g. -1e5 → -5e4
    # Positive sentinel is unusual but handled symmetrically
    return invalid_value * 0.5


def _make_invalid_mask(x: torch.Tensor, invalid_value: float) -> torch.Tensor:
    """Boolean mask: True where *x* carries a sentinel / out-of-range value.

    Args:
        x: arbitrary-shape float tensor of lane x-coordinates.
        invalid_value: sentinel used to mark invisible points (e.g. -1e5).

    Returns:
        bool tensor with the same shape as *x*.
    """
    threshold = _invalid_threshold(invalid_value)
    if invalid_value < 0:
        return x <= threshold
    return x >= threshold


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def pairwise_line_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    img_w: int,
    length: float = 15,
    invalid_value: float = -1e5,
) -> torch.Tensor:
    """Pairwise Line IoU — all predicted lanes vs. all GT lanes.

    Args:
        pred:          ``[N_pred, L]`` or ``[B, N_pred, L]``
        target:        ``[N_gt,   L]`` or ``[B, N_gt,   L]``
        img_w:         image width in pixels (unused in masking, kept for API
                       consistency and future range checks).
        length:        virtual half-lane width in pixels.
        invalid_value: sentinel for invisible / missing x-coordinates.

    Returns:
        iou: ``[N_pred, N_gt]`` or ``[B, N_pred, N_gt]``

    Raises:
        ValueError: on unsupported tensor dimensions or mismatched batch sizes.
    """
    if pred.dim() != target.dim():
        raise ValueError(
            f'pred and target must have the same number of dimensions, '
            f'got pred.dim()={pred.dim()}, target.dim()={target.dim()}'
        )
    if pred.dim() not in (2, 3):
        raise ValueError(f'Expected 2-D or 3-D tensors, got pred.dim()={pred.dim()}')
    if pred.dim() == 3 and pred.shape[0] != target.shape[0]:
        raise ValueError(f'Batch size mismatch: pred.shape[0]={pred.shape[0]}, target.shape[0]={target.shape[0]}')

    # --- Sanitise: replace NaN with sentinel so arithmetic stays finite -------
    pred = torch.nan_to_num(pred, nan=invalid_value)
    target = torch.nan_to_num(target, nan=invalid_value)

    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length

    if pred.dim() == 2:
        # shapes: [N_pred, L], [N_gt, L]  →  [N_pred, N_gt, L]
        ovr = torch.min(px2[:, None, :], tx2[None, :, :]) - torch.max(px1[:, None, :], tx1[None, :, :])
        union = torch.max(px2[:, None, :], tx2[None, :, :]) - torch.min(px1[:, None, :], tx1[None, :, :])

        pred_inv = _make_invalid_mask(pred, invalid_value)  # [N_pred, L]
        target_inv = _make_invalid_mask(target, invalid_value)  # [N_gt,   L]
        # A lane-point slot is invalid when *either* side carries a sentinel
        combined_inv = pred_inv[:, None, :] | target_inv[None, :, :]  # [N_pred, N_gt, L]

    else:  # dim == 3
        # shapes: [B, N_pred, L], [B, N_gt, L]  →  [B, N_pred, N_gt, L]
        ovr = torch.min(px2[:, :, None, :], tx2[:, None, :, :]) - torch.max(px1[:, :, None, :], tx1[:, None, :, :])
        union = torch.max(px2[:, :, None, :], tx2[:, None, :, :]) - torch.min(px1[:, :, None, :], tx1[:, None, :, :])

        pred_inv = _make_invalid_mask(pred, invalid_value)  # [B, N_pred, L]
        target_inv = _make_invalid_mask(target, invalid_value)  # [B, N_gt,   L]
        combined_inv = pred_inv[:, :, None, :] | target_inv[:, None, :, :]  # [B, N_pred, N_gt, L]

    ovr = torch.clamp(ovr, min=0.0)
    union = torch.clamp(union, min=0.0)

    ovr = ovr.masked_fill(combined_inv, 0.0)
    union = union.masked_fill(combined_inv, 0.0)

    ovr_sum = ovr.sum(dim=-1)
    union_sum = union.sum(dim=-1)
    # Clamp to ≥ 1.0: avoids division-by-zero and FP16 overflow (1/1e-9 ≈ 1e9 >> 65504)
    iou = ovr_sum / torch.clamp(union_sum, min=1.0)
    return iou


def aligned_line_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    img_w: int,
    length: float = 15,
    invalid_value: float = -1e5,
) -> torch.Tensor:
    """Aligned Line IoU — element-wise (one-to-one) matching.

    Args:
        pred:          ``[N, L]`` or ``[B, N, L]``
        target:        ``[N, L]`` or ``[B, N, L]``
        img_w:         image width in pixels (kept for API consistency).
        length:        virtual half-lane width in pixels.
        invalid_value: sentinel for invisible / missing x-coordinates.

    Returns:
        iou: ``[N]`` or ``[B, N]``

    Raises:
        ValueError: if pred and target shapes differ.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f'pred and target must have identical shapes, got pred.shape={pred.shape}, target.shape={target.shape}'
        )
    if pred.dim() not in (2, 3):
        raise ValueError(f'Expected 2-D or 3-D tensors, got pred.dim()={pred.dim()}')

    # --- Sanitise both tensors -----------------------------------------------
    pred = torch.nan_to_num(pred, nan=invalid_value)
    target = torch.nan_to_num(target, nan=invalid_value)

    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length

    ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
    union = torch.max(px2, tx2) - torch.min(px1, tx1)

    ovr = torch.clamp(ovr, min=0.0)
    union = torch.clamp(union, min=0.0)

    # Mask positions where *either* coordinate carries a sentinel
    invalid_mask = _make_invalid_mask(pred, invalid_value) | _make_invalid_mask(target, invalid_value)
    ovr = ovr.masked_fill(invalid_mask, 0.0)
    union = union.masked_fill(invalid_mask, 0.0)

    ovr_sum = ovr.sum(dim=-1)
    union_sum = union.sum(dim=-1)
    iou = ovr_sum / torch.clamp(union_sum, min=1.0)
    return iou


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------


class LaneIouLoss(nn.Module):
    """Line IoU Loss for structured lane representations.

    Expects lanes encoded with a sentinel value (default -1e5) for invisible
    or out-of-range sample points, as produced by a typical LaneEncoder.

    Args:
        loss_weight:   Global scalar weight applied to the returned loss.
        lane_width:    Virtual lane half-width in pixels (e.g. 15 → 30 px wide lane).
        invalid_value: Sentinel marking invisible GT points.
        reduction:     ``'mean'`` | ``'sum'`` | ``'none'``.
                       Use ``'none'`` to obtain per-lane losses for custom weighting.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        lane_width: float = 15.0,
        invalid_value: float = -1e5,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'")
        self.loss_weight = loss_weight
        self.lane_width = lane_width
        self.invalid_value = invalid_value
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        img_w: int,
    ) -> torch.Tensor:
        """Compute the IoU loss.

        Args:
            pred   (Tensor): ``(N, L)`` predicted x-coords at sampled y rows.
            target (Tensor): ``(N, L)`` GT x-coords at sampled y rows.
            img_w  (int):    image width in pixels.

        Returns:
            Scalar tensor (or ``[N]`` when ``reduction='none'``).
        """
        iou = aligned_line_iou(
            pred=pred,
            target=target,
            img_w=img_w,
            length=self.lane_width,
            invalid_value=self.invalid_value,
        )

        loss = 1.0 - iou  # per-lane loss ∈ [0, 1]

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # "none" → return the [N] tensor unchanged

        return loss * self.loss_weight

    def extra_repr(self) -> str:
        return (
            f'loss_weight={self.loss_weight}, lane_width={self.lane_width}, '
            f"invalid_value={self.invalid_value}, reduction='{self.reduction}'"
        )
