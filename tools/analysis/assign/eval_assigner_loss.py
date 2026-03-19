import torch
import torch.nn.functional as F
import numpy as np

# Directly import the instantiated config objects
import config.llanetv1.culane.resnet34_fpn as clr_cfg
from unlanedet.config import instantiate
from unlanedet.model.fCLRNet.f_dynamic_assign import assign as clrnet_assign
from tools.analysis.assign.assigner import geometry_aware_assign
from unlanedet.model.module.losses import FocalLoss
from unlanedet.model.fCLRNet.f_line_iou import line_iou

def compute_loss(predictions, targets, matching_matrix, n_strips, img_w, prior_ys):
    device = predictions.device
    batch_size = targets.shape[0]
    
    valid_mask = targets[..., 1] == 1
    num_gt_per_img = valid_mask.sum(dim=1).float()
    num_gt_per_img[num_gt_per_img == 0] = 1.0

    cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    matched_indices = matching_matrix.nonzero(as_tuple=False)
    if matched_indices.numel() == 0:
        batch_idx = predictions.new_zeros((0,), dtype=torch.long)
        prior_idx = predictions.new_zeros((0,), dtype=torch.long)
        gt_idx = predictions.new_zeros((0,), dtype=torch.long)
    else:
        batch_idx = matched_indices[:, 0]
        prior_idx = matched_indices[:, 1]
        gt_idx = matched_indices[:, 2]

    num_priors = predictions.shape[1]
    cls_target = torch.zeros((batch_size, num_priors), dtype=torch.long, device=device)
    if len(batch_idx) > 0:
        cls_target[batch_idx, prior_idx] = 1
        
    cls_pred = predictions[..., :2].permute(0, 2, 1)
    focal_loss = cls_criterion(cls_pred, cls_target)
    cls_loss = (focal_loss.sum(dim=1) / num_gt_per_img).sum()

    reg_xytl_loss = torch.tensor(0.0, device=device)
    iou_loss = torch.tensor(0.0, device=device)

    if len(batch_idx) > 0:
        matched_preds = predictions[batch_idx, prior_idx]
        matched_targets = targets[batch_idx, gt_idx]

        reg_yxtl = matched_preds[:, 2:6].clone().float()
        reg_yxtl[:, 0] *= n_strips
        reg_yxtl[:, 1] *= img_w - 1
        reg_yxtl[:, 2] *= 180
        reg_yxtl[:, 3] *= n_strips

        target_yxtl = matched_targets[:, 2:6].clone().float()
        with torch.no_grad():
            pred_starts = (matched_preds[:, 2] * n_strips).round().long().clamp(0, n_strips)
            target_starts = (matched_targets[:, 2] * n_strips).round().long()
            target_yxtl[:, -1] -= pred_starts - target_starts

        target_yxtl[:, 0] *= n_strips
        target_yxtl[:, 2] *= 180

        reg_loss_each = F.smooth_l1_loss(reg_yxtl, target_yxtl, reduction='none').mean(dim=1)
        reg_loss_sum = torch.zeros(batch_size, device=device, dtype=torch.float32)
        reg_loss_sum.index_add_(0, batch_idx, reg_loss_each)

        pred_xs = matched_preds[:, 6:].clone().float() * (img_w - 1)
        target_xs = matched_targets[:, 6:].clone().float()
        iou_each = 1.0 - line_iou(pred_xs, target_xs, img_w, length=15, aligned=True)
        iou_loss_sum = torch.zeros(batch_size, device=device)
        iou_loss_sum.index_add_(0, batch_idx, iou_each)

        count_per_img = torch.bincount(batch_idx, minlength=batch_size).float()
        valid_img_mask = count_per_img > 0
        reg_xytl_loss += (reg_loss_sum[valid_img_mask] / count_per_img[valid_img_mask]).sum()
        iou_loss += (iou_loss_sum[valid_img_mask] / count_per_img[valid_img_mask]).sum()

    return cls_loss, reg_xytl_loss, iou_loss

def debug_assigners():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Building model with dataloader...")
    clr_model = instantiate(clr_cfg.model)
    dataloader_cfg = clr_cfg.dataloader.train
    # dataloader_cfg.dataset.split = "val"
    dataloader = instantiate(dataloader_cfg)
    clr_param_config = clr_cfg.param_config

    weights = torch.load("assets/clrnet_model_best_culane.pth", map_location="cpu")
    if "model" in weights:
        clr_model.load_state_dict(weights["model"], strict=False)
    elif "net" in weights:
        clr_model.load_state_dict(weights["net"], strict=False)
    else:
        clr_model.load_state_dict(weights, strict=False)
    clr_model.eval()
    clr_model.to(device)
    
    print("Fetching a batch of data...")
    iterator = iter(dataloader)
    batch = next(iterator)
    
    def to_device(batch_data):
        if isinstance(batch_data, dict):
            return {k: to_device(v) for k, v in batch_data.items()}
        elif isinstance(batch_data, torch.Tensor):
            return batch_data.to(device)
        elif isinstance(batch_data, list):
            return [to_device(v) for v in batch_data]
        return batch_data
        
    batch = to_device(batch)

    with torch.no_grad():
        if hasattr(clr_model, 'module'):
            model_core = clr_model.module
        else:
            model_core = clr_model
            
        if hasattr(model_core, 'forward_features_and_aux'):
            features, _ = model_core.forward_features_and_aux(batch)
        else:
            features = model_core.neck(model_core.backbone(batch['img']))
        
        model_core.head.train()
        output = model_core.head(features)
        predictions_lists = output['predictions_lists']
        model_core.head.eval()

    head = model_core.head
    img_w, img_h = head.img_w, head.img_h
    n_strips = head.n_strips
    sample_ys = head.prior_ys

    targets = batch['lane_line'].clone()
    valid_mask = targets[..., 1] == 1
    
    print(f"\n=====================================")
    print(f"Loss Comparison Output:")
    print(f"=====================================")
    
    for stage, predictions in enumerate(predictions_lists):
        print(f"\n[Stage {stage}] Refining Layer:")
        clr_match = clrnet_assign(predictions, targets, img_w, img_h, valid_mask=valid_mask)
        ga_match = geometry_aware_assign(predictions, targets, img_w, img_h, cfg=clr_param_config, sample_ys=sample_ys)
        
        c_cls, c_reg, c_iou = compute_loss(predictions, targets, clr_match, n_strips, img_w, sample_ys)
        g_cls, g_reg, g_iou = compute_loss(predictions, targets, ga_match, n_strips, img_w, sample_ys)
        
        total_clr = c_cls + c_reg + c_iou
        total_ga = g_cls + g_reg + g_iou
        
        print(f"  [CLRNet] Cls Loss: {c_cls:.4f} | Reg Loss: {c_reg:.4f} | IoU Loss: {c_iou:.4f} | Total: {total_clr:.4f}")
        print(f"  [GA Ass] Cls Loss: {g_cls:.4f} | Reg Loss: {g_reg:.4f} | IoU Loss: {g_iou:.4f} | Total: {total_ga:.4f}")
        
        # Also print priors matching stats
        for b in range(1): # check first batch item
            vt = valid_mask[b].sum().item()
            print(f"      Image 0 (GT={int(vt)}): CLR Priors={clr_match[b].sum(dim=0)[:int(vt)].tolist()} | GA Priors={ga_match[b].sum(dim=0)[:int(vt)].tolist()}")

if __name__ == '__main__':
    debug_assigners()
