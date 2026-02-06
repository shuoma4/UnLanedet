import os
import os.path as osp
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

plt.rcParams["font.family"] = "AR PL UMing CN"

# ============================ 配置参数 ============================

CACHE_PATH = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/openlane_lane3d_1000_train_cuth-270_800x320_cache_v1.pkl"
OUTPUT_DIR = "./vis/sample_lane/spline"
NUM_SAMPLE_POINTS = 192
IMG_W = 800
IMG_H = 320
SHOW_IMAGES = False

# ======== ID 选择模式 ========
ID_MODE = "hook"  # ["hook", "range", "list"]

# ============================ 工具函数 ============================


def compute_lane_curvature_circle(points):
    """
    基于圆拟合的车道线曲率估计
    points: (N,2)
    return: scalar curvature (1 / R)
    """
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 3:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]

    # 构造线性最小二乘：x^2 + y^2 + ax + by + c = 0
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    B = -(x**2 + y**2)

    try:
        a, b, c = np.linalg.lstsq(A, B, rcond=None)[0]
        R_sq = (a * a + b * b) / 4.0 - c
        if R_sq <= 1e-6:
            return 0.0
        R = np.sqrt(R_sq)
        return float(1.0 / R)
    except np.linalg.LinAlgError:
        return 0.0


def compute_lane_curvature_spline(points, num_samples=50):
    """
    基于样条插值 + 二阶导的车道线曲率估计（鲁棒版本）
    points: (N,2)
    return: scalar curvature (mean curvature)
    """
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 4:
        return 0.0

    # 按 y 排序
    order = np.argsort(pts[:, 1])
    pts = pts[order]

    # 去除重复点 / 极近点
    filtered = [pts[0]]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - filtered[-1]) > 1e-3:
            filtered.append(pts[i])
    pts = np.array(filtered, dtype=np.float32)

    if len(pts) < 4:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]

    # 弧长参数化
    diffs = pts[1:] - pts[:-1]
    seg_lens = np.sqrt((diffs**2).sum(axis=1))

    # 过滤掉 seg_lens = 0 的情况
    valid = seg_lens > 1e-4
    if not np.any(valid):
        return 0.0

    t = np.concatenate([[0.0], np.cumsum(seg_lens)])

    # 强制严格递增（数值兜底）
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-4

    if t[-1] < 1e-3:
        return 0.0

    try:
        k = min(3, len(t) - 1)
        fx = InterpolatedUnivariateSpline(t, x, k=k)
        fy = InterpolatedUnivariateSpline(t, y, k=k)

        ts = np.linspace(0, t[-1], num_samples)

        dx = fx.derivative(1)(ts)
        dy = fy.derivative(1)(ts)
        ddx = fx.derivative(2)(ts)
        ddy = fy.derivative(2)(ts)

        denom = (dx * dx + dy * dy) ** 1.5 + 1e-6
        curvature = np.abs(dx * ddy - dy * ddx) / denom

        return float(np.mean(curvature))

    except Exception as e:
        # 兜底：样条失败直接返回 0
        return 0.0
 

def load_dataset_cache(cache_path):
    if not osp.exists(cache_path):
        raise FileNotFoundError(f"缓存文件不存在: {cache_path}")
    with open(cache_path, "rb") as f:
        data_infos = pickle.load(f)
    print(f"成功加载缓存: {cache_path}")
    print(f"数据集大小: {len(data_infos)} 个样本")
    return data_infos


def sample_lane_fixed(points, num_samples):
    """固定点弧长均匀采样 (Improved with Spline Interpolation)"""
    points = np.array(points, dtype=np.float32)
    if len(points) < 2:
        return np.array([]), np.array([])

    # 1. Sort by Y (descending)
    sort_idx = np.argsort(points[:, 1])[::-1]
    points = points[sort_idx]

    # 2. Remove duplicates
    valid_points = [points[0]]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[i - 1]) > 1e-4:
            valid_points.append(points[i])
    points = np.array(valid_points)

    if len(points) < 2:
        return np.array([]), np.array([])

    # 3. Calculate cumulative distance (approx arc length)
    diffs = points[1:] - points[:-1]
    seg_lens = np.sqrt((diffs**2).sum(axis=1))
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_dist[-1]

    if total_len < 1e-6:
        return np.array([]), np.array([])

    # 4. Spline Interpolation
    try:
        # Use cubic spline if enough points, else lower order
        k = min(3, len(points) - 1)
        if k < 1:
            k = 1

        # Parameterize by cumulative distance t
        # x = f(t), y = g(t)
        spline_x = InterpolatedUnivariateSpline(cum_dist, points[:, 0], k=k)
        spline_y = InterpolatedUnivariateSpline(cum_dist, points[:, 1], k=k)

        # Sample uniformly along t
        t_samples = np.linspace(0, total_len, num_samples)
        xs = spline_x(t_samples)
        ys = spline_y(t_samples)

        return xs, ys

    except Exception as e:
        print(f"Spline interpolation failed: {e}. Fallback to linear.")
        # Fallback to linear interpolation
        target_lens = np.linspace(0, total_len, num_samples)
        sampled = []
        for t in target_lens:
            idx = np.searchsorted(cum_dist, t) - 1
            idx = np.clip(idx, 0, len(points) - 2)
            l0, l1 = cum_dist[idx], cum_dist[idx + 1]
            ratio = (t - l0) / (l1 - l0 + 1e-6)
            p = points[idx] + ratio * (points[idx + 1] - points[idx])
            sampled.append(p)
        sampled = np.array(sampled)
        return sampled[:, 0], sampled[:, 1]


def visualize_sample_fixed(data_info, idx, num_sample_points, output_dir, img_w, img_h):
    """可视化固定点采样算法"""
    os.makedirs(output_dir, exist_ok=True)

    img_path = data_info["img_path"]
    if not osp.exists(img_path):
        print(f"警告: 图像文件不存在: {img_path}")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"警告: 无法加载图像: {img_path}")
        return

    if img.shape[1] != img_w or img.shape[0] != img_h:
        img = cv2.resize(img, (img_w, img_h))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.imshow(img_rgb)
    ax1.set_title(f"Original Points", fontsize=12)
    ax2.imshow(img_rgb)
    ax2.set_title(f"Fixed Sample Points (n={num_sample_points})", fontsize=12)

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(data_info["lanes"]))))

    for lane_idx, lane_points in enumerate(data_info["lanes"]):
        if len(lane_points) < 2:
            continue

        color = colors[lane_idx % len(colors)]
        points = np.array(lane_points)
        x_orig, y_orig = points[:, 0], points[:, 1]

        # 原始点
        ax1.scatter(x_orig, y_orig, c=[color], s=5, alpha=0.7)
        ax1.plot(x_orig, y_orig, color=color, linewidth=0.5, alpha=0.7)

        # 固定点采样
        xs, ys = sample_lane_fixed(lane_points, num_sample_points)
        if len(xs) == 0:
            continue

        valid_mask = (xs >= 0) & (xs < img_w) & (ys >= 0) & (ys < img_h)
        ax2.scatter(xs[valid_mask], ys[valid_mask], c=[color], s=5, alpha=0.7)
        ax2.plot(xs[valid_mask], ys[valid_mask], color=color, linewidth=0.5, alpha=0.7)

    for ax in [ax1, ax2]:
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = osp.join(output_dir, f"sample_{idx:06d}_slope_std_topk.png")
    plt.savefig(output_path, dpi=500, bbox_inches="tight")

    if SHOW_IMAGES:
        plt.show()
    else:
        plt.close()

    print(f"Saved: {output_path}")


# ============================ Hook 选择弯曲车道（曲率） ============================


def analyze_openlane_slope_std_topk(pkl_path, top_k=50):
    """Hook 模式：选取斜率 std 最大的 top-k 弯曲车道"""
    with open(pkl_path, "rb") as f:
        data_infos = pickle.load(f)

    lane_records = []  # (slope_std, sample_id, lane_id)

    for sample_id, sample in tqdm(
        enumerate(data_infos), total=len(data_infos), desc="计算车道曲率"
    ):
        for lane_id, lane in enumerate(sample["lanes"]):
            curv_val = compute_lane_curvature_spline(lane)
            lane_records.append((curv_val, sample_id, lane_id))

    lane_records.sort(key=lambda x: x[0], reverse=True)  # 弯曲程度高的在前
    topk_lanes = lane_records[:top_k]
    topk_sample_ids = np.array([x[1] for x in topk_lanes], dtype=np.int64)

    stats = {
        "lane_mean_curvature": float(np.mean([x[0] for x in lane_records])),
        "lane_max_curvature": float(lane_records[0][0]),
    }

    return topk_sample_ids, stats


ID_HOOK = analyze_openlane_slope_std_topk
ID_RANGE = (1000, 5)
ID_LIST = [
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
]


def get_sample_ids(pkl_path, data_infos):
    if ID_MODE == "hook":
        ids, stats = ID_HOOK(pkl_path)
        print("使用 Hook 选择样本（斜率最大 top-k 弯曲车道）")
        print("Hook Stats:", stats)
        return ids
    elif ID_MODE == "range":
        start_idx, num_samples = ID_RANGE
        end_idx = min(start_idx + num_samples, len(data_infos))
        ids = np.arange(start_idx, end_idx)
        print(f"使用 Range 选择样本: {start_idx} ~ {end_idx-1}")
        return ids
    elif ID_MODE == "list":
        ids = np.array([i for i in ID_LIST if i < len(data_infos)], dtype=np.int64)
        print(f"使用 List 选择样本: {ids.tolist()}")
        return ids
    else:
        raise ValueError(f"未知 ID_MODE: {ID_MODE}")


# ============================ 主函数 ============================


def main():
    print("=" * 60)
    print("OpenLane 可视化（固定点采样 + 弯曲车道 top-k based on curvature）")
    print("=" * 60)

    data_infos = load_dataset_cache(CACHE_PATH)
    sample_ids = get_sample_ids(CACHE_PATH, data_infos)

    for idx in sample_ids:
        print(f"\n处理样本 {idx}/{len(data_infos)-1}")
        print(f"样本信息: {data_infos[idx]['img_path']}")
        print(f"车道线数量: {len(data_infos[idx]['lanes'])}")

        visualize_sample_fixed(
            data_infos[idx], idx, NUM_SAMPLE_POINTS, OUTPUT_DIR, IMG_W, IMG_H
        )

    print(f"\n可视化完成! 结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
