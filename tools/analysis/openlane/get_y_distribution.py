import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ============================
# Config
# ============================
CACHE_PATH = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/openlane_lane3d_1000_train_cuth-270_800x320_cache_v1.pkl"
NUM_POINTS = 72
OUTPUT_DIR = "tools/analysis/openlane"
IMG_H = 320


# ============================
# 曲率估计
# ============================
def compute_curvature(points, eps=1e-6):
    """
    基于离散角度变化的曲率近似
    κ_i = arccos( (v_i · v_{i+1}) / (||v_i|| ||v_{i+1}||) )
    """
    curv = np.zeros(len(points), dtype=np.float32)
    if len(points) < 3:
        return curv

    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < eps or n2 < eps:
            curv[i] = 0.0
        else:
            cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            curv[i] = np.arccos(cos_theta)
    return curv


# ============================
# 高度 + 曲率联合加权弧长重采样
# ============================
def sample_lane_weighted(points, num_samples, alpha=1.0, beta=1.0, img_h=320):
    """
    联合加权弧长重采样
    w_i = 1 + α (1 - y_i/H) + β κ_i
    Δs*_i = w_i Δs_i
    """
    points = np.array(points, dtype=np.float32)
    if len(points) < 2:
        return np.array([]), np.array([])

    # 按 y 从下到上排序（图像坐标系）
    sort_idx = np.argsort(points[:, 1])[::-1]
    points = points[sort_idx]

    diffs = points[1:] - points[:-1]
    seg_lens = np.sqrt((diffs**2).sum(axis=1))

    # 曲率
    curv = compute_curvature(points)
    curv = (curv - curv.min()) / (curv.max() - curv.min() + 1e-6)

    # 高度权重
    y_norm = points[:, 1] / img_h
    w_y = 1.0 + alpha * (1.0 - y_norm)

    # 联合权重
    w = 1.0 + (w_y - 1.0) + beta * curv
    w_mid = (w[:-1] + w[1:]) * 0.5

    weighted_arc = seg_lens * w_mid
    weighted_cum = np.concatenate([[0.0], np.cumsum(weighted_arc)])

    total_len = weighted_cum[-1]
    if total_len < 1e-6:
        return np.array([]), np.array([])

    target_lens = np.linspace(0, total_len, num_samples)

    sampled = []
    for t in target_lens:
        idx = np.searchsorted(weighted_cum, t) - 1
        idx = np.clip(idx, 0, len(points) - 2)
        l0, l1 = weighted_cum[idx], weighted_cum[idx + 1]
        ratio = (t - l0) / (l1 - l0 + 1e-6)
        p = points[idx] + ratio * (points[idx + 1] - points[idx])
        sampled.append(p)

    sampled = np.array(sampled)
    return sampled[:, 0], sampled[:, 1]


# ============================
# 主流程：统计 y 分布
# ============================
def get_y_distribution():
    print(f"Loading cache from {CACHE_PATH}...")
    with open(CACHE_PATH, "rb") as f:
        data_infos = pickle.load(f)

    all_ys = []
    print("Processing lanes with weighted arc-length sampling...")

    for info in tqdm(data_infos[::5]):
        lanes = info.get("lanes", [])
        for lane in lanes:
            lane = np.array(lane, dtype=np.float32)
            if len(lane) < 2:
                continue

            y_min, y_max = lane[:, 1].min(), lane[:, 1].max()
            length_y = abs(y_max - y_min)
            num_sample = int(np.clip(length_y, 30, 180))

            xs, ys = sample_lane_weighted(
                lane,
                num_samples=num_sample,
                alpha=1.2,  # 上部区域先验
                beta=1.5,  # 曲率增强 
                img_h=IMG_H,
            )
            if len(ys) > 0:
                all_ys.extend(ys)

    all_ys = np.array(all_ys)
    print(f"Total points collected: {len(all_ys)}")
    print(f"Y range: {all_ys.min()} - {all_ys.max()}")

    plt.figure()
    plt.hist(all_ys, bins=100)
    plt.title("Y Coordinate Distribution (Height + Curvature Weighted)")
    plt.xlabel("Y (pixel)")
    plt.ylabel("Count")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "y_dist_weighted.png"))

    q_points = np.linspace(0, 100, NUM_POINTS)
    sample_ys = np.percentile(all_ys, q_points)
    sample_ys = np.sort(sample_ys)[::-1]

    print("\nGenerated sample_ys:")
    print("sample_ys = np.array([")
    print(", ".join([f"{y:.4f}" for y in sample_ys]))
    print("], dtype=np.float32)")

    return sample_ys


if __name__ == "__main__":
    get_y_distribution()
