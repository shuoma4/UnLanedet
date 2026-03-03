import torch


def pairwise_line_iou(pred, target, img_w, length=15, invalid_value=-1e5):
    """
    Pairwise Line IoU (all-to-all)
    Args:
        pred:   [N_pred, L] or [B, N_pred, L]
        target: [N_gt, L]   or [B, N_gt, L]
        img_w: image width
        length: extended radius
        invalid_value: sentinel value for invalid points
    Returns:
        iou: [N_pred, N_gt] or [B, N_pred, N_gt]
    """
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if pred.dim() == 2:
        # [N_pred, L] x [N_gt, L] -> [N_pred, N_gt, L]
        ovr = torch.min(px2[:, None, :], tx2[None, :, :]) - torch.max(px1[:, None, :], tx1[None, :, :])
        union = torch.max(px2[:, None, :], tx2[None, :, :]) - torch.min(px1[:, None, :], tx1[None, :, :])
    elif pred.dim() == 3:
        # [B, N_pred, L] x [B, N_gt, L] -> [B, N_pred, N_gt, L]
        ovr = torch.min(px2[:, :, None, :], tx2[:, None, :, :]) - torch.max(px1[:, :, None, :], tx1[:, None, :, :])
        union = torch.max(px2[:, :, None, :], tx2[:, None, :, :]) - torch.min(px1[:, :, None, :], tx1[:, None, :, :])
    else:
        raise ValueError(f'Unsupported pred dim: {pred.dim()}')

    if pred.dim() == 2:
        invalid_mask = (target < 0) | (target >= img_w)
        invalid_mask = invalid_mask[None, :, :]  # [1, N_gt, L]
    else:
        invalid_mask = (target < 0) | (target >= img_w)
        invalid_mask = invalid_mask[:, None, :, :]  # [B, 1, N_gt, L]

    ovr = torch.clamp(ovr, min=0.0)
    union = torch.clamp(union, min=1e-9)

    ovr = ovr.masked_fill(invalid_mask, 0.0)
    union = union.masked_fill(invalid_mask, 0.0)

    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def aligned_line_iou(pred, target, img_w, length=15, invalid_value=-1e5):
    """
    Aligned Line IoU (one-to-one)
    Args:
        pred:   [N, L] or [B, N, L]
        target: [N, L] or [B, N, L]
        img_w: image width
        length: extended radius
        invalid_value: sentinel value for invalid points
    Returns:
        iou: [N] or [B, N]
    """
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
    union = torch.max(px2, tx2) - torch.min(px1, tx1)

    ovr = torch.clamp(ovr, min=0.0)
    union = torch.clamp(union, min=1e-9)

    invalid_mask = (target < -100) | (target >= img_w + 100)
    ovr = ovr.masked_fill(invalid_mask, 0.0)
    union = union.masked_fill(invalid_mask, 0.0)
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


class LaneIouLoss(torch.nn.Module):
    """
    Line IoU Loss for structured lane representation.

    This loss is designed to work with your LaneEncoder, where:
        - Lane points outside the visible segment are encoded as delta_x = -1e5
        - IoU is computed by integrating virtual lane overlaps along sampled rows
    """

    def __init__(self, loss_weight=1.0, lane_width=15.0, invalid_value=-1e5):
        """
        Args:
            loss_weight (float):
                Global weight for LineIoU loss term.
            lane_width (float):
                Virtual lane half-width in pixel space.
                Example: 15 px means a 30 px wide virtual lane.
            invalid_value (float):
                Sentinel value used to mark invisible points in GT encoding.
        """
        super(LaneIouLoss, self).__init__()
        self.loss_weight = loss_weight
        self.lane_width = lane_width
        self.invalid_value = invalid_value

    def forward(self, pred, target, img_w):
        """
        Args:
            pred (Tensor): (N, L) predicted x-coordinates at sampled y positions.
            target (Tensor): (N, L) GT x-coordinates at sampled y positions.
            img_w (int): image width in pixels.

        Returns:
            Tensor: scalar LineIoU loss.
        """
        assert pred.shape == target.shape, 'Prediction and target must have the same shape.'

        iou = aligned_line_iou(
            pred=pred,
            target=target,
            img_w=img_w,
            length=self.lane_width,
        )

        loss = (1.0 - iou).mean() * self.loss_weight
        return loss
