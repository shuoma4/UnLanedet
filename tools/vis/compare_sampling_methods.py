import os
import os.path as osp
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

# Add project root to path
import sys

sys.path.append(os.getcwd())

# Import sampling classes
from unlanedet.data.transform.generate_lane_line import (
    GenerateLaneLine as GenerateLaneLinev1,
)
from unlanedet.data.transform.res_lane_encoder import (
    ResLaneEncoder,
)
from config.llanet.priors import SAMPLE_YS_IOSDENSITY

plt.rcParams["font.family"] = "AR PL UMing CN"

# ============================ Configuration ============================

CACHE_PATH = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/openlane_lane3d_1000_train_cuth-270_800x320_cache_v1.pkl"
OUTPUT_DIR = "./vis/compare_sampling/"
IMG_W = 800
IMG_H = 320
NUM_POINTS = 192  # Equivalent to n_strips + 1 roughly
SAMPLE_IDS = [
    47642,
    34490,
    99180,
    83308,
    91254,
    85227,
    7456,
    43216,
    105047,
    118108,
    94301,
    76735,
    127923,
    111017,
    72330,
    115971,
    88157,
    76146,
    111823,
    142119,
]  # Includes the problematic one mentioned by user

if not osp.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class MockConfig:
    def __init__(self):
        self.img_w = IMG_W
        self.img_h = IMG_H
        self.num_points = 72
        self.max_lanes = 20
        self.cut_height = 0
        self.img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# Initialize Transformers
cfg = MockConfig()
cfg.sample_y = SAMPLE_YS_IOSDENSITY
cfg.num_points = len(SAMPLE_YS_IOSDENSITY)
gen_v1 = GenerateLaneLinev1(transforms=[], cfg=cfg, training=False)
gen_openlane = ResLaneEncoder(transforms=[], cfg=cfg, training=False)

# Define Y sample points (same as used in the classes)
# Note: GenerateLaneLine uses np.arange(self.img_h, -1, -self.strip_size)
sample_ys_equidistant = np.linspace(IMG_H, 0, cfg.num_points)
sample_ys_isodistribution = SAMPLE_YS_IOSDENSITY


def method1_polyfit_sample(points, num_points):
    """
    Method 1: Fit a continuous curve (Polyfit) and sample at fixed Y intervals.
    """
    points = np.array(points)
    if len(points) < 3:
        return points[:, 0], points[:, 1]

    x = points[:, 0]
    y = points[:, 1]

    # Fit x = f(y)
    # Degree 2 or 3 usually works best for lanes
    try:
        # Check if we have enough variance in Y
        if np.std(y) < 1e-3:
            return x, y

        z = np.polyfit(y, x, 4)
        p = np.poly1d(z)

        # Determine range of Y for sampling
        min_y, max_y = np.min(y), np.max(y)

        # Generate fixed Ys within the lane's range
        # Let's use global sample_ys_equidistant intersected with lane range to be consistent with others
        valid_sample_ys = sample_ys_equidistant[
            (sample_ys_equidistant >= min_y) & (sample_ys_equidistant <= max_y)
        ]

        if len(valid_sample_ys) < 2:
            # Fallback to linear linspace between min/max
            valid_sample_ys = np.linspace(max_y, min_y, num_points)  # Descending

        sampled_x = p(valid_sample_ys)
        return sampled_x, valid_sample_ys

    except Exception as e:
        print(f"Polyfit failed: {e}")
        return x, y


def load_dataset_cache(cache_path):
    with open(cache_path, "rb") as f:
        data_infos = pickle.load(f)
    return data_infos


def draw_method1(img, lane, cfg):
    """Method 1: Polyfit + Fixed Y Sampling"""
    try:
        m1_x, m1_y = method1_polyfit_sample(lane, cfg.num_points)
        for px, py in zip(m1_x, m1_y):
            if 0 <= px < IMG_W and 0 <= py < IMG_H:
                cv2.circle(img, (int(px), int(py)), 3, (0, 255, 255), -1)
    except Exception as e:
        print(f"Method 1 failed: {e}")


def draw_method2(img, lane, gen_v1):
    """Method 2: GenerateLaneLine (v1) - Spline Interpolation (Uses Fixed Code)"""
    try:
        # GenerateLaneLine.sample_lane expects points sorted by y descending and strictly decreasing
        # Filter duplicate Ys and sort
        unique_y_points = []
        seen_y = set()
        # Sort by Y descending first
        lane = sorted(lane, key=lambda p: -p[1])

        for p in lane:
            if p[1] not in seen_y:
                unique_y_points.append(p)
                seen_y.add(p[1])

        if len(unique_y_points) < 2:
            return

        lane_filtered = np.array(unique_y_points)

        # Call the ACTUAL modified sample_lane function
        # Note: We recently changed sample_lane to return all_xs directly
        all_xs = gen_v1.sample_lane(lane_filtered, sample_ys_equidistant)

        # all_xs corresponds to sample_ys_equidistant
        valid_mask = (all_xs >= 0) & (all_xs < IMG_W) & (all_xs > -1e4)

        final_xs = all_xs[valid_mask]
        final_ys = sample_ys_equidistant[valid_mask]

        # Calculate GT Y range to filter extrapolation for visualization
        # lane is sorted by Y descending
        gt_max_y = lane[0][1]
        gt_min_y = lane[-1][1]

        # Filter points outside GT Y range (allow small tolerance)
        tolerance = 5
        range_mask = (final_ys <= gt_max_y + tolerance) & (
            final_ys >= gt_min_y - tolerance
        )

        final_xs = final_xs[range_mask]
        final_ys = final_ys[range_mask]

        for px, py in zip(final_xs, final_ys):
            cv2.circle(img, (int(px), int(py)), 3, (255, 0, 255), -1)

    except Exception as e:
        print(f"Method 2 failed: {e}")


def draw_method3(img, lane, gen_openlane):
    """Method 3: GenerateLaneLineOpenLane - ArcLength Sample -> Linear Interp"""
    try:
        interp_xs, _ = gen_openlane.sample_lane(
            lane, gen_openlane.offsets_ys, visibility=[]
        )

        valid_mask = (interp_xs > -1e4) & (interp_xs >= 0) & (interp_xs < IMG_W)
        final_xs = interp_xs[valid_mask]
        final_ys = gen_openlane.offsets_ys[valid_mask]

        for px, py in zip(final_xs, final_ys):
            cv2.circle(img, (int(px), int(py)), 3, (255, 255, 0), -1)

    except Exception as e:
        print(f"Method 3 failed: {e}")


def process_and_visualize():
    data_infos = load_dataset_cache(CACHE_PATH)
    print(f"Loaded {len(data_infos)} samples.")

    # Filter IDs
    selected_samples = []
    if SAMPLE_IDS:
        for idx in SAMPLE_IDS:
            if idx < len(data_infos):
                selected_samples.append((idx, data_infos[idx]))
    else:
        # Random 5
        import random

        idxs = random.sample(range(len(data_infos)), 5)
        for idx in idxs:
            selected_samples.append((idx, data_infos[idx]))

    for idx, sample in tqdm(selected_samples):
        # Handle different cache formats
        if "img_info" in sample:
            rel_path = sample["img_info"]["filename"]
        elif "img_path" in sample:
            rel_path = sample["img_path"]
        else:
            print(f"Sample {idx} missing image path info. Keys: {sample.keys()}")
            continue

        img_path = osp.join("/data1/lxy_log/workspace/ms/OpenLane/images", rel_path)
        if not osp.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_W, IMG_H))

        # Get lanes
        lanes = sample["lanes"]

        # Prepare 4 images for the grid
        img_gt = img.copy()
        img_m1 = img.copy()
        img_m2 = img.copy()
        img_m3 = img.copy()

        # 1. Original GT
        for lane in lanes:
            lane = np.array(lane)
            # for i in range(len(lane) - 1):
            #     cv2.line(
            #         img_gt,
            #         (int(lane[i][0]), int(lane[i][1])),
            #         (int(lane[i + 1][0]), int(lane[i + 1][1])),
            #         (0, 255, 0),
            #         2,
            #     )
            for p in lane:
                cv2.circle(img_gt, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        # Process each lane for methods
        for lane in lanes:
            lane = np.array(lane)
            if len(lane) < 2:
                continue

            # Ensure sorted by Y descending
            if lane[0, 1] < lane[-1, 1]:
                lane = lane[::-1]

            draw_method1(img_m1, lane, cfg)
            draw_method2(img_m2, lane, gen_v1)
            draw_method3(img_m3, lane, gen_openlane)

        # Add Titles
        font_scale = 0.8
        thickness = 2

        def add_title(image, text):
            # Add white background for text
            cv2.rectangle(image, (0, 0), (IMG_W, 40), (0, 0, 0), -1)
            cv2.putText(
                image,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )
            # Add border
            cv2.rectangle(image, (0, 0), (IMG_W - 1, IMG_H - 1), (255, 255, 255), 2)

        add_title(img_gt, "Original GT (Red)")
        add_title(img_m1, "Method 1: Polyfit + Fixed Y (Yellow)")
        add_title(img_m2, "Method 2: GenerateLaneLine v1 (Magenta)")
        add_title(img_m3, "Method 3: GenerateLaneLineOpenLane (Cyan)")

        # Combine into (2,2) grid
        top = np.hstack([img_gt, img_m1])
        bottom = np.hstack([img_m2, img_m3])
        grid = np.vstack([top, bottom])

        # Save
        out_path = osp.join(OUTPUT_DIR, f"compare_{idx}.png")
        cv2.imwrite(out_path, grid)


if __name__ == "__main__":
    process_and_visualize()
    print(f"Visualization complete. Saved to {OUTPUT_DIR}")
