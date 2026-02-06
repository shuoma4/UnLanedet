import torch


def line_iou(pred, target, img_w, length=15.0, aligned=True, invalid_value=-1e5):
    """
    Calculate Line IoU between predicted lanes and GT lanes.

    This version is adapted for structured lane encoding, where:
        - Invalid / invisible sample points are marked by delta_x = invalid_value (e.g. -1e5)
        - Coordinates are in pixel space (not normalized)

    Args:
        pred (Tensor):
            Predicted lane x-coordinates at sampled y positions.
            Shape:
                aligned=True:   (N, L)
                aligned=False:  (N_pred, L)
                batch mode:     (B, N_pred, L)
        target (Tensor):
            Ground-truth lane x-coordinates at sampled y positions.
            Shape:
                aligned=True:   (N, L)
                aligned=False:  (N_gt, L)
                batch mode:     (B, N_gt, L)
        img_w (int):
            Image width (pixel space).
        length (float):
            Half width of the virtual lane (pixel space).
        aligned (bool):
            - True:  compute IoU for aligned pairs (used in loss)
            - False: compute pair-wise IoU for matching / assign
        invalid_value (float):
            Sentinel value used in GT encoding for invisible points (default: -1e5)

    Returns:
        Tensor:
            IoU scores.
            Shape:
                aligned=True:   (N,)
                aligned=False:  (N_pred, N_gt)
                batch mode:     (B, N_pred, N_gt)
    """

    # -------------------------------------------------
    # 1. 构造每个采样点处的“虚拟车道区间”
    # -------------------------------------------------
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length

    # -------------------------------------------------
    # 2. 计算重叠区间与并集区间
    # -------------------------------------------------
    if aligned:
        # pred:   (N, L)
        # target: (N, L)
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)

    else:
        if pred.dim() == 2:
            # pred:   (N_pred, L)
            # target: (N_gt,   L)
            num_pred = pred.shape[0]
            invalid_mask = target.repeat(num_pred, 1, 1)

            ovr = torch.min(px2[:, None, :], tx2[None, ...]) - torch.max(px1[:, None, :], tx1[None, ...])
            union = torch.max(px2[:, None, :], tx2[None, ...]) - torch.min(px1[:, None, :], tx1[None, ...])
        else:
            # Batch mode:
            # pred:   (B, N_pred, L)
            # target: (B, N_gt,   L)
            num_pred = pred.shape[1]
            invalid_mask = target.unsqueeze(1).expand(-1, num_pred, -1, -1)

            px1 = px1.unsqueeze(2)  # (B, N_pred, 1, L)
            px2 = px2.unsqueeze(2)
            tx1 = tx1.unsqueeze(1)  # (B, 1, N_gt, L)
            tx2 = tx2.unsqueeze(1)

            ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
            union = torch.max(px2, tx2) - torch.min(px1, tx1)

    # -------------------------------------------------
    # 3. 交集非负截断（几何上 IoU 不能为负）
    # -------------------------------------------------
    ovr = torch.clamp(ovr, min=0.0)

    # -------------------------------------------------
    # 4. 构造无效点掩码（基于 GT 编码语义）
    #    - delta_x == invalid_value  → 不可见点
    # -------------------------------------------------
    invalid_masks = invalid_mask <= invalid_value + 1.0

    ovr[invalid_masks] = 0.0
    union[invalid_masks] = 0.0

    # -------------------------------------------------
    # 5. 沿采样点维度求 IoU
    # -------------------------------------------------
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

        iou = line_iou(
            pred=pred,
            target=target,
            img_w=img_w,
            length=self.lane_width,
            aligned=True,
            invalid_value=self.invalid_value,
        )

        loss = (1.0 - iou).mean() * self.loss_weight
        return loss
