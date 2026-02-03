import pickle
import numpy as np

# import matplotlib.pyplot as plt
import os

CACHE_PATH = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/openlane_lane3d_1000_train_cuth-270_800x320_cache_v1.pkl"
NUM_POINTS = 72
OUTPUT_DIR = "tools/analysis/openlane"


from tqdm import tqdm


def sample_lane_fixed(points, num_samples):
    """固定点弧长均匀采样"""
    points = np.array(points, dtype=np.float32)
    if len(points) < 2:
        return np.array([]), np.array([])
    sort_idx = np.argsort(points[:, 1])[::-1]
    points = points[sort_idx]
    diffs = points[1:] - points[:-1]
    seg_lens = np.sqrt((diffs**2).sum(axis=1))
    arc_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = arc_len[-1]
    if total_len < 1e-6:
        return np.array([]), np.array([])
    target_lens = np.linspace(0, total_len, num_samples)
    sampled = []
    for t in target_lens:
        idx = np.searchsorted(arc_len, t) - 1
        idx = np.clip(idx, 0, len(points) - 2)
        l0, l1 = arc_len[idx], arc_len[idx + 1]
        ratio = (t - l0) / (l1 - l0 + 1e-6)
        p = points[idx] + ratio * (points[idx + 1] - points[idx])
        sampled.append(p)
    sampled = np.array(sampled)
    return sampled[:, 0], sampled[:, 1]


def get_y_distribution():
    print(f"Loading cache from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        data_infos = pickle.load(f)

    all_ys = []
    print("Processing lanes with arc-length sampling...")
    # Process a subset to speed up (every 50th sample)
    for info in tqdm(data_infos[::50]):
        lanes = info.get("lanes", [])
        for lane in lanes:
            # lane is a list of [x, y]
            lane = np.array(lane)
            if len(lane) > 1:
                # Use sample_lane_fixed to get uniform points along the lane
                _, ys = sample_lane_fixed(lane, num_samples=320)
                if len(ys) > 0:
                    all_ys.extend(ys)

    all_ys = np.array(all_ys)
    print(f"Total points collected: {len(all_ys)}")
    print(f"Y range: {all_ys.min()} - {all_ys.max()}")

    # Calculate quantiles
    # We want sample_ys to go from Bottom (High Y) to Top (Low Y) or vice versa.
    # Usually algorithms expect sorted Ys.
    # If we want equal number of points between samples, we use percentiles.

    # Check distribution
    # plt.figure()
    # plt.hist(all_ys, bins=100)
    # plt.title("Y Coordinate Distribution")
    # plt.xlabel("Y (pixel)")
    # plt.ylabel("Count")
    # if not os.path.exists(OUTPUT_DIR):
    #     os.makedirs(OUTPUT_DIR)
    # plt.savefig(os.path.join(OUTPUT_DIR, "y_dist.png"))
    # print(f"Saved distribution plot to {os.path.join(OUTPUT_DIR, 'y_dist.png')}")

    # Generate sample_ys
    # We want NUM_POINTS samples.
    # The first sample should probably be at max Y (bottom), last at min Y (top)?
    # Or just spread them according to density.

    # np.percentile(all_ys, q) where q is 0..100
    # Let's generate linearly spaced quantiles
    q_points = np.linspace(0, 100, NUM_POINTS)
    sample_ys = np.percentile(all_ys, q_points)

    # Sort descending (Bottom to Top) as is common in lane detection (near to far)
    sample_ys = np.sort(sample_ys)[::-1]

    print("\nGenerated sample_ys (Equal Frequency):")
    # Format for easy copy-pasting
    print("sample_ys = np.array([")
    print(", ".join([f"{y:.4f}" for y in sample_ys]))
    print("], dtype=np.float32)")

    return sample_ys


if __name__ == "__main__":
    get_y_distribution()
