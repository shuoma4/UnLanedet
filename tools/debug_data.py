
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import cv2
from unlanedet.config import LazyConfig, instantiate
from unlanedet.utils import comm

def print_stats(name, batch):
    print(f"--- {name} Statistics ---")
    if 'img' in batch:
        img = batch['img']
        print(f"Image Shape: {img.shape}")
        print(f"Image Type: {img.dtype}")
        print(f"Image Mean: {img.mean().item():.4f}")
        print(f"Image Std: {img.std().item():.4f}")
        print(f"Image Min: {img.min().item():.4f}")
        print(f"Image Max: {img.max().item():.4f}")
    
    if 'lane_line' in batch:
        lanes = batch['lane_line']
        if isinstance(lanes, torch.Tensor):
            print(f"Lane Lines Shape: {lanes.shape}")
            print(f"Lane Lines Mean: {lanes.float().mean().item():.4f}")
            print(f"Lane Lines Max: {lanes.float().max().item():.4f}")
        elif isinstance(lanes, list):
             print(f"Lane Lines List Length: {len(lanes)}")
             if len(lanes) > 0:
                 print(f"First Sample Lane Count: {len(lanes[0])}")

    if 'seg' in batch:
        seg = batch['seg']
        print(f"Seg Shape: {seg.shape}")
        print(f"Seg Unique Values: {torch.unique(seg)}")

def main():
    # Load configs
    cfg_openlane = LazyConfig.load("config/llanet/llanetv2_openlane.py")
    cfg_culane = LazyConfig.load("config/clrnet/resnet34_culane.py")

    # Instantiate dataloaders (using test loader for simplicity, or train loader with batch_size 1)
    # Using train loader to see training augmentations
    
    # Adjust batch size for debugging
    cfg_openlane.dataloader.train.total_batch_size = 4
    cfg_openlane.dataloader.train.num_workers = 0 # Main process
    cfg_culane.dataloader.train.total_batch_size = 4
    cfg_culane.dataloader.train.num_workers = 0

    print("Building OpenLane Loader...")
    loader_openlane = instantiate(cfg_openlane.dataloader.train)
    
    print("Building CULane Loader...")
    # CULane loader might need data_root adjustment if not absolute
    # cfg_culane.dataloader.train.dataset.data_root = "/data0/lxy_data/mslanedet/CULane/" # Assuming this path from config file
    loader_culane = instantiate(cfg_culane.dataloader.train)

    print("\nFetching OpenLane Batch...")
    for batch_openlane in loader_openlane:
        print_stats("OpenLane (LLANetV2)", batch_openlane)
        break

    print("\nFetching CULane Batch...")
    try:
        for batch_culane in loader_culane:
            print_stats("CULane (CLRNet)", batch_culane)
            break
    except Exception as e:
        print(f"Failed to fetch CULane batch (likely due to missing path): {e}")

if __name__ == "__main__":
    main()
