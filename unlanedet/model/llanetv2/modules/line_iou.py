import torch


def line_iou(pred, target, img_w, length=15, aligned=True):
    """
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72) or (B, num_pred, 72)
        target: ground truth, shape: (num_target, 72) or (B, num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    """
    # Vectorized implementation of line_iou
    # If aligned=True, assumes pred and target match (e.g. 1-to-1 or same shape)
    # If aligned=False, computes pairwise IoU between all preds and all targets

    # Expand length for broadcasting
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length

    if aligned:
        # 1-to-1 matching (e.g. for Loss calculation)
        # pred: [N, 72], target: [N, 72]
        # or pred: [B, N, 72], target: [B, N, 72]
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        # Pairwise matching (e.g. for Assignment)
        # pred: [N, 72], target: [M, 72]
        # output: [N, M]
        # Need to broadcast: [N, 1, 72] vs [1, M, 72]

        # Ensure input dimensions are correct for broadcasting
        # If input is 2D [N, L] and [M, L]
        if pred.dim() == 2 and target.dim() == 2:
            num_pred = pred.shape[0]
            num_target = target.shape[0]

            # [N, 1, 72]
            px1 = px1.unsqueeze(1)
            px2 = px2.unsqueeze(1)
            # [1, M, 72]
            tx1 = tx1.unsqueeze(0)
            tx2 = tx2.unsqueeze(0)

            # Broadcasting for overlap and union
            # [N, M, 72]
            ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
            union = torch.max(px2, tx2) - torch.min(px1, tx1)

            # Invalid mask based on target (if target point is invalid)
            # target invalid points are usually -1e5 or outside [0, img_w]
            # target is [M, 72] -> [1, M, 72] -> broadcast to [N, M, 72]
            invalid_mask = target.unsqueeze(0).expand(num_pred, num_target, -1)

        else:
            raise NotImplementedError('Pairwise line_iou only supports 2D inputs [N, L] and [M, L]')

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)

    # Zero out invalid points
    # ovr = torch.clamp(ovr, min=0.0) # Ensure overlap is non-negative
    ovr[invalid_masks] = 0.0
    # union = torch.clamp(union, min=0.0)
    union[invalid_masks] = 0.0

    # Sum over points (dim -1)
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def liou_loss(pred, target, img_w, length=15):
    return (1 - line_iou(pred, target, img_w, length, aligned=True)).mean()
