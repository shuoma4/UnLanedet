import argparse
import pickle
import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

# Add project root to path to allow imports if needed, though we try to be standalone
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

# OpenLane Category Mapping to Train ID
# 0 (Unknown) -> 1
# 1-12 -> 2-13
# 20 (Left Curbside) -> 14
# 21 (Right Curbside) -> 15
# Background -> 0
OPENLANE_CATEGORY_MAP = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    20: 14,
    21: 15,
}

NUM_CLASSES = 16
IMG_W = 800
IMG_H = 320
LINE_WIDTH = 5  # Pixel width for segmentation labels


def generate_lane_mask_ids(image_shape, lane_points_list, category_list, line_width=5):
    """
    Generate a lane line segmentation mask with Train IDs.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Sort lanes to draw? Or just draw. Order might matter for overlap.
    # Usually doesn't matter much for weights.

    for points, category in zip(lane_points_list, category_list):
        if points is None or len(points) < 2:
            continue

        # Map category to Train ID
        train_id = OPENLANE_CATEGORY_MAP.get(category, 0)
        # If category is not in map (e.g. unexpected), treat as 0 or skip?
        # If 0 is background, we shouldn't draw with 0.
        # OpenLane categories should be in the map.
        if train_id == 0:
            continue

        pts = np.array(points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Draw the polyline with Train ID value
        cv2.polylines(
            mask, [pts], isClosed=False, color=int(train_id), thickness=line_width
        )

    return mask


def calculate_weights(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"Error: pkl file not found at {pkl_path}")
        return

    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data_infos = pickle.load(f)

    print(f"Loaded {len(data_infos)} samples.")
    print("Calculating segmentation weights...")

    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for sample in tqdm(data_infos):
        lanes = sample.get("lanes", [])
        categories = sample.get("lane_categories", [])

        # We need to rescale points if they are in original resolution (1920x1280)
        # to the target resolution (800x320).
        # Assuming data_infos contains points in original resolution.
        # Let's check a sample usually.
        # But wait, standard OpenLane processing often keeps points in original.
        # We need to scale them.

        # Default OpenLane shape
        ori_h, ori_w = 1280, 1920

        scaled_lanes = []
        for lane in lanes:
            lane = np.array(lane)
            # Scale x and y
            lane[:, 0] *= IMG_W / ori_w
            lane[:, 1] *= IMG_H / ori_h  # This is simple scaling.
            # Note: Real augmentation might do crop/resize.
            # For weight calculation, simple resize is a good approximation.
            scaled_lanes.append(lane)

        mask = generate_lane_mask_ids(
            (IMG_H, IMG_W), scaled_lanes, categories, line_width=LINE_WIDTH
        )

        counts = np.bincount(mask.flatten(), minlength=NUM_CLASSES)
        pixel_counts += counts

    # Calculate frequencies
    total_pixels = np.sum(pixel_counts)
    frequency = pixel_counts / total_pixels

    print("\nClass Frequencies:")
    for i, f in enumerate(frequency):
        print(f"Class {i}: {f:.6f}")

    # Median Frequency Balancing
    # We only consider classes that are present
    present_classes = frequency[frequency > 0]
    median_freq = np.median(present_classes)

    weights = np.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        if frequency[i] > 0:
            weights[i] = median_freq / frequency[i]
        else:
            weights[i] = 0.0  # Or 1.0? Usually 0 or high penalty?
            # If class never appears, weight doesn't matter much.

    print("\nCalculated Weights (Median Frequency Balancing):")
    print("-" * 30)
    print("seg_weights = [")
    for i, w in enumerate(weights):
        print(f"    {w:.4f},  # Class {i}")
    print("]")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate segmentation weights for OpenLane"
    )
    parser.add_argument("pkl_path", help="Path to the dataset cache pkl file")
    args = parser.parse_args()

    calculate_weights(args.pkl_path)
