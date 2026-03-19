import torch
from ..CLRNet import CLRNetIoULoss

class LaneIoULoss(CLRNetIoULoss):
    def __init__(self, loss_weight=1.0, lane_width=7.5 / 800, img_h=320, img_w=1640):
        """
        LaneIoU loss employed in CLRerNet.
        Args:
            weight (float): loss weight.
            lane_width (float): half virtual lane width.
        """
        super(LaneIoULoss, self).__init__(loss_weight, lane_width)
        self.max_dx = 1e4
        self.img_h = img_h
        self.img_w = img_w

    def _calc_lane_width(self, pred, target):
        n_strips = pred.shape[-1] - 1
        dy = self.img_h / n_strips * 2  # two horizontal grids
        _pred = pred.clone().detach()
        
        pred_dx = (
            _pred[..., 2:] - _pred[..., :-2]
        ) * self.img_w  
        pred_width = self.lane_width * torch.sqrt(pred_dx.pow(2) + dy**2) / dy
        pred_width = torch.cat(
            [pred_width[..., 0:1], pred_width, pred_width[..., -1:]], dim=-1
        )
        
        target_dx = (target[..., 2:] - target[..., :-2]) * self.img_w
        target_dx[torch.abs(target_dx) > self.max_dx] = 0
        target_width = self.lane_width * torch.sqrt(target_dx.pow(2) + dy**2) / dy
        target_width = torch.cat(
            [target_width[..., 0:1], target_width, target_width[..., -1:]], dim=-1
        )

        return pred_width, target_width

    def forward(self, pred, target):
        assert (
            pred.shape == target.shape
        ), "prediction and target must have the same shape!"
        pred_width, target_width = self._calc_lane_width(pred, target)
        iou = self.calc_iou(pred, target, pred_width, target_width)
        return (1 - iou).mean() * self.loss_weight

# Class implementation of LineIoU cost
class CLRNetIoUCost:
    def __init__(self, weight=1.0, lane_width=15 / 800):
        """
        LineIoU cost employed in CLRNet.
        Adapted from:
        https://github.com/Turoad/CLRNet/blob/main/clrnet/models/losses/lineiou_loss.py
        Args:
            weight (float): cost weight.
            lane_width (float): half virtual lane width.
        """
        self.weight = weight
        self.lane_width = lane_width

    def _calc_over_union(self, pred, target, pred_width, target_width):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
            pred_width (torch.Tensor): virtual lane half-widths for prediction at pre-defined rows, shape (Nl, Nr).
            target_width (torch.Tensor): virtual lane half-widths for GT at pre-defined rows, shape (Nl, Nr).
        Returns:
            torch.Tensor: calculated overlap, shape (Nlp, Nlt, Nr).
            torch.Tensor: calculated union, shape (Nlp, Nlt, Nr).
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        

        px1 = pred - pred_width
        px2 = pred + pred_width
        tx1 = target - target_width
        tx2 = target + target_width

        # Handle batched dimensions dynamically
        if pred.dim() == 3:  # (B, Nlp, Nr)
            px1 = px1.unsqueeze(2)  # (B, Nlp, 1, Nr)
            px2 = px2.unsqueeze(2)
            tx1 = tx1.unsqueeze(1)  # (B, 1, Nlt, Nr)
            tx2 = tx2.unsqueeze(1)
        else: # (Nlp, Nr)
            px1 = px1.unsqueeze(1)
            px2 = px2.unsqueeze(1)
            tx1 = tx1.unsqueeze(0)
            tx2 = tx2.unsqueeze(0)
            
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)

        return ovr, union

    def __call__(self, pred, target):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate
            target: ground truth, shape: (Nlt, Nr), relative coordinate
        Returns:
            torch.Tensor: calculated IoU matrix, shape (Nlp, Nlt)
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        ovr, union = self._calc_over_union(
            pred, target, self.lane_width, self.lane_width
        )
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        invalid_masks = (invalid_mask < 0) | (invalid_mask >= 1.0)
        ovr[invalid_masks] = 0.0
        union[invalid_masks] = 0.0
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou * self.weight

class LaneIoUCost(CLRNetIoUCost, LaneIoULoss):
    def __init__(
        self,
        weight=1.0,
        lane_width=7.5 / 800,
        img_h=320,
        img_w=1640,
        use_pred_start_end=False,
        use_giou=True,
    ):
        """
        Angle- and length-aware LaneIoU employed in CLRerNet.
        Args:
            weight (float): cost weight.
            lane_width (float): half virtual lane width.
            use_pred_start_end (bool): apply the lane range (in horizon indices) for pred lanes
            use_giou (bool): GIoU-style calculation that allow negative overlap
               when the lanes are separated
        """
        super(LaneIoUCost, self).__init__(weight, lane_width)
        self.use_pred_start_end = use_pred_start_end
        self.use_giou = use_giou
        self.max_dx = 1e4
        self.img_h = img_h
        self.img_w = img_w

    @staticmethod
    def _set_invalid_with_start_end(
        pred, target, ovr, union, start, end, pred_width, target_width
    ):
        if pred.dim() == 3:  # (B, Np, Nr)
            B, Np, Nr = pred.shape
            Nt = target.shape[1]
            pred_mask = pred.unsqueeze(2).expand(B, Np, Nt, Nr)
            target_mask = target.unsqueeze(1).expand(B, Np, Nt, Nr)
            
            invalid_mask_pred = (pred_mask < 0) | (pred_mask >= 1.0)
            invalid_mask_gt = (target_mask < 0) | (target_mask >= 1.0)
            
            assert start is not None and end is not None
            yind = torch.ones_like(invalid_mask_pred) * torch.arange(0, Nr).float().to(pred.device)
            h = Nr - 1
            start_idx = (start * h).long().view(B, Np, 1, 1)
            end_idx = (end * h).long().view(B, Np, 1, 1)
            invalid_mask_pred = invalid_mask_pred | (yind < start_idx) | (yind >= end_idx)
            
            invalid_mask_pred_gt = invalid_mask_pred | invalid_mask_gt
            ovr[invalid_mask_pred_gt] = 0
            union[invalid_mask_pred_gt] = 0
            
            union_sep_target = target_width.unsqueeze(1).expand(B, Np, Nt, Nr) * 2
            union_sep_pred = pred_width.unsqueeze(2).expand(B, Np, Nt, Nr) * 2
            
            union[invalid_mask_pred_gt & ~invalid_mask_pred] += union_sep_pred[invalid_mask_pred_gt & ~invalid_mask_pred]
            union[invalid_mask_pred_gt & ~invalid_mask_gt] += union_sep_target[invalid_mask_pred_gt & ~invalid_mask_gt]
            return ovr, union

        else:
            num_gt = target.shape[0]
            pred_mask = pred.repeat(num_gt, 1, 1).permute(1, 0, 2)
            invalid_mask_pred = (pred_mask < 0) | (pred_mask >= 1.0)
            target_mask = target.repeat(pred.shape[0], 1, 1)
            invalid_mask_gt = (target_mask < 0) | (target_mask >= 1.0)

            assert start is not None and end is not None
            yind = torch.ones_like(invalid_mask_pred) * torch.arange(
                0, pred.shape[-1]
            ).float().to(pred.device)
            h = pred.shape[-1] - 1
            start_idx = (start * h).long().view(-1, 1, 1)
            end_idx = (end * h).long().view(-1, 1, 1)
            invalid_mask_pred = invalid_mask_pred | (yind < start_idx) | (yind >= end_idx)

            invalid_mask_pred_gt = invalid_mask_pred | invalid_mask_gt
            ovr[invalid_mask_pred_gt] = 0
            union[invalid_mask_pred_gt] = 0

            union_sep_target = target_width.repeat(pred.shape[0], 1, 1) * 2
            union_sep_pred = pred_width.repeat(num_gt, 1, 1).permute(1, 0, 2) * 2
            union[invalid_mask_pred_gt & ~invalid_mask_pred] += union_sep_pred[
                invalid_mask_pred_gt & ~invalid_mask_pred
            ]
            union[invalid_mask_pred_gt & ~invalid_mask_gt] += union_sep_target[
                invalid_mask_pred_gt & ~invalid_mask_gt
            ]
            return ovr, union
    @staticmethod
    def _set_invalid_without_start_end(pred, target, ovr, union):
        if pred.dim() == 3:
            B, Nlp, Nr = pred.shape
            Nlt = target.shape[1]
            invalid_mask_gt = (target < 0) | (target >= 1.0)
            invalid_mask_gt = invalid_mask_gt.unsqueeze(1).repeat(1, Nlp, 1, 1)
        else:
            Nlp = pred.shape[0]
            target_mask = target.repeat(Nlp, 1, 1)
            invalid_mask_gt = (target_mask < 0) | (target_mask >= 1.0)
            
        ovr[invalid_mask_gt] = 0.0
        union[invalid_mask_gt] = 0.0
        return ovr, union

    def __call__(self, pred, target, start=None, end=None):
        """
        Calculate the line iou value between predictions and targets
        Args:
            pred: lane predictions, shape: (Nlp, Nr), relative coordinate.
            target: ground truth, shape: (Nlt, Nr), relative coordinate.
        Returns:
            torch.Tensor: calculated IoU matrix, shape (Nlp, Nlt)
        Nlp, Nlt: number of prediction and target lanes, Nr: number of rows.
        """
        pred_width, target_width = self._calc_lane_width(pred, target)
        ovr, union = self._calc_over_union(pred, target, pred_width, target_width)
        if self.use_pred_start_end is True:
            ovr, union = self._set_invalid_with_start_end(
                pred, target, ovr, union, start, end, pred_width, target_width
            )
        else:
            ovr, union = self._set_invalid_without_start_end(pred, target, ovr, union)
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
        return iou * self.weight