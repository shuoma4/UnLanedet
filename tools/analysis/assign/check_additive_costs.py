import torch
import sys
import os

sys.path.append('.')
from unlanedet.model.fCLRNet.f_dynamic_assign import distance_cost, focal_cost
from unlanedet.model.llanetv1.assigner import _to_absolute, _target_to_absolute

img_w = 1640.0
img_h = 590.0

preds = torch.zeros((1, 5, 78))
preds[..., 0] = -1.0
preds[..., 1] = 1.0 # score = 1
# relative coordinates
preds[..., 2] = 0.5 # y
preds[..., 3] = 0.5 # x
preds[..., 4] = 0.5 # theta
preds[..., 5] = 0.5 # len
preds[..., 6:] = 0.5 # xs

targets = torch.zeros((1, 1, 78))
targets[..., 1] = 1
targets[..., 2] = 71 * 0.5 # absolute y
targets[..., 3] = 1640 * 0.5 # absolute x
targets[..., 4] = 180 * 0.5
targets[..., 5] = 71 * 0.5
targets[..., 6:] = 1640 * 0.5

preds_abs = _to_absolute(preds, img_w, img_h)
targets_abs = _target_to_absolute(targets, img_w, img_h)

n_strips = 71.0

# 1. Start point absolute distances
delta_x = torch.abs(preds_abs[..., 3].unsqueeze(2) - targets_abs[..., 3].unsqueeze(1)) / img_w
delta_y = torch.abs(preds_abs[..., 2].unsqueeze(2) - targets_abs[..., 2].unsqueeze(1)) / n_strips
theta_c = torch.abs(preds_abs[..., 4].unsqueeze(2) - targets_abs[..., 4].unsqueeze(1)) / 180.0
length_c = torch.abs(preds_abs[..., 5].unsqueeze(2) - targets_abs[..., 5].unsqueeze(1)) / n_strips
line_dist_c = distance_cost(preds_abs, targets_abs, img_w) / img_w

print(f"delta_x: {delta_x[0,0,0]}")
print(f"delta_y: {delta_y[0,0,0]}")
print(f"theta_c: {theta_c[0,0,0]}")
print(f"length_c: {length_c[0,0,0]}")
print(f"line_dist_c: {line_dist_c[0,0,0]}")

cls_cost = focal_cost(preds[..., :2], targets[..., 1].long())
print(f"cls_cost: {cls_cost[0,0,0]}")

