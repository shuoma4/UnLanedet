import torch
import sys
import os

sys.path.append('.')
from unlanedet.model.fCLRNet.f_dynamic_assign import assign as clrnet_assign
from tools.analysis.assign.assigner import geometry_aware_assign
from config.llanetv1.culane.resnet34_fpn import dataloader
from unlanedet.data.build import build_dataloader

dataloader_cfg = dataloader.val
dataloader_cfg['batch_size'] = 8
dataloader_cfg['workers'] = 0
dataloader_cfg['shuffle'] = False

dl = build_dataloader(dataloader_cfg)

class DummyCfg:
    w_cls = 1.0
    w_geom = 3.0

cfg = DummyCfg()

clrnet_assigned_total = 0
ga_assigned_total = 0
total_gt_lanes = 0

print("Testing Recall...")
for i, batch in enumerate(dl):
    if i > 5: break
    
    preds = torch.rand(batch['lane_line'].shape[0], 192, 78).cuda()
    targets = batch['lane_line'].cuda()
    masks = (targets[..., 1] == 1)
    
    img_w, img_h = 1640.0, 590.0
    
    clr_match = clrnet_assign(preds, targets, img_w, img_h, valid_mask=masks)
    ga_match = geometry_aware_assign(preds, targets, img_w, img_h, valid_mask=masks, cfg=cfg)
    
    for b in range(preds.shape[0]):
        vt = masks[b].sum()
        total_gt_lanes += int(vt)
        clr_assigned = clr_match[b, :, :vt].sum(dim=0) > 0
        ga_assigned = ga_match[b, :, :vt].sum(dim=0) > 0
        
        clrnet_assigned_total += clr_assigned.sum().item()
        ga_assigned_total += ga_assigned.sum().item()

print(f"Total GT: {total_gt_lanes}")
print(f"CLRNet Recall: {clrnet_assigned_total / total_gt_lanes:.4f}")
print(f"GA Recall: {ga_assigned_total / total_gt_lanes:.4f}")
