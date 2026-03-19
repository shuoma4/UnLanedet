import torch
import sys
import os

sys.path.append('.')
from unlanedet.model.fCLRNet.f_dynamic_assign import distance_cost
from unlanedet.model.llanetv1.assigner import _to_absolute, _target_to_absolute

img_w = 1640
img_h = 590

# create dummy predictions (relative)
preds = torch.zeros((1, 10, 78))
preds[..., 6:] = 0.5 # relative x

# dummy targets (already absolute)
targets = torch.zeros((1, 1, 78))
targets[..., 1] = 1 # valid class
targets[..., 6:] = 820.0 # absolute x

# test current code distance (bugged)
bad_dist = distance_cost(preds, targets, img_w)
print(f"Bad distance: {bad_dist[0, 0, 0].item()}")

# test fixed code distance (using absolute)
preds_abs = _to_absolute(preds, img_w, img_h)
targets_abs = _target_to_absolute(targets, img_w, img_h)

good_dist = distance_cost(preds_abs, targets_abs, img_w)
print(f"Good distance: {good_dist[0, 0, 0].item()}")

