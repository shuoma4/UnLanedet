import os
import os.path as osp
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

plt.rcParams["font.family"] = "AR PL UMing CN"

# ============================ 配置参数 ============================
CACHE_PATH = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/openlane_lane3d_1000_train_cuth-270_800x320_cache_v1.pkl"
OUTPUT_DIR = "./vis/sample_openlane/weight"
NUM_SAMPLE_POINTS = 72
IMG_W = 800
IMG_H = 320
SHOW_IMAGES = False

# 选择模式
ID_MODE = "curvature"  # ["curvature", "hook", "range", "list"]
TOP_K = 20  # 选择前k个弯曲车道线

# ============================ 改进的工具函数 ============================


def sample_lane_fixed(points, num_samples):
    """
    稳健的固定点弧长均匀采样（论文级实现）
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return np.array([]), np.array([])

    # ========= 1. 去重 =========
    _, unique_idx = np.unique(points, axis=0, return_index=True)
    points = points[np.sort(unique_idx)]
    if len(points) < 2:
        return np.array([]), np.array([])

    # ========= 2. 按 y 从下到上排序（图像坐标系） =========
    order = np.argsort(points[:, 1])[::-1]
    points = points[order]

    # ========= 3. 计算弧长参数 =========
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    seg_len = np.sqrt(dx * dx + dy * dy)

    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = s[-1]
    if total_len < 1e-6:
        return points[:, 0], points[:, 1]

    s_norm = s / total_len

    # ========= 4. 样条重参数化（保证平滑） =========
    if len(points) >= 3:
        try:
            cs_x = CubicSpline(s_norm, points[:, 0])
            cs_y = CubicSpline(s_norm, points[:, 1])

            t_samples = np.linspace(0, 1, num_samples)
            xs = cs_x(t_samples)
            ys = cs_y(t_samples)

            return xs, ys
        except Exception as e:
            pass  # 回退到线性插值

    # ========= 5. 线性插值回退 =========
    t_samples = np.linspace(0, total_len, num_samples)
    sampled = []

    for t in t_samples:
        idx = np.searchsorted(s, t) - 1
        idx = np.clip(idx, 0, len(points) - 2)

        l0, l1 = s[idx], s[idx + 1]
        if l1 - l0 > 1e-6:
            ratio = (t - l0) / (l1 - l0)
        else:
            ratio = 0.0

        p = points[idx] + ratio * (points[idx + 1] - points[idx])
        sampled.append(p)

    sampled = np.asarray(sampled)
    return sampled[:, 0], sampled[:, 1]


def compute_curvature(points, smoothing_sigma=1.5):
    """
    计算车道线曲率
    使用参数化曲线和曲率公式: k = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    """
    if len(points) < 4:  # 至少需要4个点才能计算曲率
        return 0.0, 0.0

    points = np.array(points, dtype=np.float32)

    # 按y坐标从下到上排序
    if len(points) > 1:
        sort_idx = np.argsort(points[:, 1])[::-1]
        points = points[sort_idx]

    # 提取x, y坐标
    x = points[:, 0]
    y = points[:, 1]

    # 如果点太少，返回0
    if len(x) < 4:
        return 0.0, 0.0

    # 高斯平滑去除噪声
    x_smooth = gaussian_filter1d(x, sigma=smoothing_sigma)
    y_smooth = gaussian_filter1d(y, sigma=smoothing_sigma)

    # 计算一阶和二阶导数
    t = np.linspace(0, 1, len(x_smooth))

    try:
        # 对x和y分别计算导数
        dx = np.gradient(x_smooth, t)
        dy = np.gradient(y_smooth, t)
        ddx = np.gradient(dx, t)
        ddy = np.gradient(dy, t)

        # 计算曲率
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5

        # 移除无穷大和NaN值
        valid_mask = np.isfinite(curvature)
        if not np.any(valid_mask):
            return 0.0, 0.0

        curvature = curvature[valid_mask]

        # 返回平均曲率和最大曲率
        return float(np.mean(curvature)), float(np.max(curvature))

    except Exception as e:
        return 0.0, 0.0


def compute_bending_score(points):
    """
    计算车道线弯曲程度的综合评分
    结合了：曲率、方向变化、长度
    """
    if len(points) < 4:
        return 0.0

    points = np.array(points, dtype=np.float32)

    # 1. 计算平均曲率
    mean_curvature, max_curvature = compute_curvature(points)

    # 2. 计算方向变化（角度变化）
    vectors = points[1:] - points[:-1]

    # 计算每段的角度（弧度）
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # 计算相邻段的角度变化
    if len(angles) > 1:
        angle_changes = np.abs(np.diff(angles))
        # 处理角度环绕
        angle_changes = np.minimum(angle_changes, 2 * np.pi - angle_changes)
        mean_angle_change = np.mean(angle_changes)
    else:
        mean_angle_change = 0.0

    # 3. 计算总长度
    lengths = np.sqrt(np.sum(vectors**2, axis=1))
    total_length = np.sum(lengths)

    # 4. 计算曲线与弦长的比值（弯曲程度）
    chord_length = np.linalg.norm(points[-1] - points[0])
    if chord_length > 1e-6:
        bend_ratio = total_length / chord_length
    else:
        bend_ratio = 1.0

    # 5. 综合评分（可调整权重）
    # 曲率贡献 + 角度变化贡献 + 弯曲比贡献
    score = (
        mean_curvature * 1000  # 曲率（乘以1000放大）
        + mean_angle_change * 50  # 角度变化
        + (bend_ratio - 1) * 20
    )  # 弯曲比（直线为1，越大越弯曲）

    return score


def is_high_quality_lane(points, min_points=8, min_length=100, max_y_variation=300):
    """
    判断是否为高质量的车道线标注
    """
    if len(points) < min_points:
        return False

    points = np.array(points, dtype=np.float32)

    # 计算长度
    vectors = points[1:] - points[:-1]
    total_length = np.sum(np.sqrt(np.sum(vectors**2, axis=1)))

    if total_length < min_length:
        return False

    # 检查y坐标变化范围
    y_coords = points[:, 1]
    y_range = y_coords.max() - y_coords.min()
    if y_range < 50:  # y方向变化太小，可能是短车道线
        return False

    # 检查点是否单调变化（避免锯齿）
    # 允许轻微的非单调，但不能太多
    y_diff = np.diff(y_coords)
    non_monotonic_count = np.sum(y_diff > 0)  # y增加的数量

    # 如果超过25%的点是y增加的，认为是锯齿
    if non_monotonic_count > len(y_diff) * 0.25:
        return False

    # 检查x坐标变化范围
    x_coords = points[:, 0]
    x_range = x_coords.max() - x_coords.min()
    if x_range < 20:  # x方向变化太小，可能是垂直线
        return False

    return True


def extract_scene_id(img_name):
    """
    从img_name中提取场景ID，用于去重
    假设img_name格式: segment-xxxx/video_xxxx/.../timestamp.jpg
    提取: segment-xxxx/video_xxxx
    """
    parts = img_name.split("/")
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"
    elif len(parts) == 2:
        return parts[0]
    else:
        return img_name


def analyze_openlane_by_curvature(
    pkl_path, top_k=20, min_quality=True, scene_deduplicate=True
):
    """
    使用曲率分析挑选弯曲车道线，支持场景去重
    """
    with open(pkl_path, "rb") as f:
        data_infos = pickle.load(f)

    print(f"数据集大小: {len(data_infos)} 个样本")

    # 存储所有高质量车道线的记录
    lane_records = []
    scene_records = {}  # 按场景分组，用于去重

    for sample_id, sample in tqdm(
        enumerate(data_infos), total=len(data_infos), desc="分析车道线弯曲度"
    ):
        # 提取场景ID
        scene_id = extract_scene_id(sample["img_path"])

        for lane_id, lane in enumerate(sample["lanes"]):
            # 质量过滤
            if min_quality and not is_high_quality_lane(
                lane, min_points=8, min_length=120
            ):
                continue

            # 计算弯曲度
            score = compute_bending_score(lane)

            # 计算曲率详情
            mean_curv, max_curv = compute_curvature(lane)

            # 计算其他统计信息
            points = np.array(lane)
            y_range = points[:, 1].max() - points[:, 1].min()
            x_range = points[:, 0].max() - points[:, 0].min()

            record = {
                "sample_id": sample_id,
                "lane_id": lane_id,
                "scene_id": scene_id,
                "img_path": sample["img_path"],
                "bending_score": score,
                "mean_curvature": mean_curv,
                "max_curvature": max_curv,
                "num_points": len(lane),
                "y_range": y_range,
                "x_range": x_range,
                "total_length": np.sum(
                    np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
                ),
            }

            lane_records.append(record)

    print(f"找到 {len(lane_records)} 条高质量车道线")

    if len(lane_records) == 0:
        print("警告: 没有找到高质量车道线!")
        return [], {}

    # 按弯曲度排序
    lane_records.sort(key=lambda x: x["bending_score"], reverse=True)

    # 场景去重
    selected_records = []
    if scene_deduplicate:
        seen_scenes = set()
        for record in lane_records:
            if record["scene_id"] not in seen_scenes:
                selected_records.append(record)
                seen_scenes.add(record["scene_id"])
                if len(selected_records) >= top_k:
                    break
    else:
        selected_records = lane_records[:top_k]

    # 统计信息
    stats = {
        "total_lanes_analyzed": len(lane_records),
        "selected_lanes": len(selected_records),
        "unique_scenes": len(set([r["scene_id"] for r in selected_records])),
        "avg_bending_score": np.mean([r["bending_score"] for r in selected_records]),
        "max_bending_score": (
            selected_records[0]["bending_score"] if selected_records else 0
        ),
        "avg_curvature": np.mean([r["mean_curvature"] for r in selected_records]),
        "avg_points": np.mean([r["num_points"] for r in selected_records]),
        "avg_y_range": np.mean([r["y_range"] for r in selected_records]),
        "avg_total_length": np.mean([r["total_length"] for r in selected_records]),
    }

    # 打印统计信息
    print("\n" + "=" * 60)
    print("弯曲车道线选择统计信息:")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\nTop 5 最弯曲的车道线 (已去重):")
    for i, record in enumerate(selected_records[:5]):
        print(
            f"  {i+1}. 场景: {record['scene_id']}, 样本: {record['sample_id']:04d}, "
            f"车道: {record['lane_id']}, 弯曲度: {record['bending_score']:.2f}"
        )
        print(
            f"     平均曲率: {record['mean_curvature']:.5f}, 点数: {record['num_points']}, "
            f"y范围: {record['y_range']:.1f}, 总长: {record['total_length']:.1f}"
        )

    return selected_records, stats


def visualize_sample_with_curvature(
    data_info, idx, num_sample_points, output_dir, img_w, img_h, selected_lane_ids=None
):
    """
    可视化样本，特别标注弯曲车道线
    """
    os.makedirs(output_dir, exist_ok=True)

    img_path = data_info["img_path"]
    if not osp.exists(img_path):
        print(f"警告: 图像文件不存在: {img_path}")
        return None

    img = cv2.imread(img_path)
    if img is None:
        print(f"警告: 无法加载图像: {img_path}")
        return None

    if img.shape[1] != img_w or img.shape[0] != img_h:
        img = cv2.resize(img, (img_w, img_h))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    # 子图1: 原始标注点
    ax1.imshow(img_rgb)
    ax1.set_title(f"原始标注点 (样本 {idx})", fontsize=14, fontweight="bold")

    # 子图2: 固定点采样
    ax2.imshow(img_rgb)
    ax2.set_title(f"固定点采样 (n={num_sample_points})", fontsize=14, fontweight="bold")

    # 子图3: 弯曲度分析
    ax3.imshow(img_rgb)
    ax3.set_title(f"弯曲度分析", fontsize=14, fontweight="bold")

    # 为不同车道线分配颜色
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(data_info["lanes"]))))

    for lane_idx, lane_points in enumerate(data_info["lanes"]):
        if len(lane_points) < 2:
            continue

        color = colors[lane_idx % len(colors)]
        points = np.array(lane_points)

        # 确保点按y坐标从上到下排序
        if len(points) > 1:
            sort_idx = np.argsort(points[:, 1])[::-1]
            points = points[sort_idx]

        x_orig, y_orig = points[:, 0], points[:, 1]

        # 计算车道线信息
        is_selected = selected_lane_ids is not None and lane_idx in selected_lane_ids
        bending_score = compute_bending_score(lane_points)
        mean_curv, max_curv = compute_curvature(lane_points)
        is_quality = is_high_quality_lane(lane_points)

        # 子图1: 原始点
        ax1.scatter(
            x_orig,
            y_orig,
            c=[color],
            s=15,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )
        ax1.plot(x_orig, y_orig, color=color, linewidth=1.5, alpha=0.7)

        # 子图2: 固定点采样
        xs, ys = sample_lane_fixed(lane_points, num_sample_points)
        if len(xs) > 0:
            valid_mask = (xs >= 0) & (xs < img_w) & (ys >= 0) & (ys < img_h)
            if np.any(valid_mask):
                ax2.scatter(
                    xs[valid_mask],
                    ys[valid_mask],
                    c=[color],
                    s=10,
                    alpha=0.8,
                    edgecolors="white",
                    linewidth=0.5,
                )
                ax2.plot(
                    xs[valid_mask],
                    ys[valid_mask],
                    color=color,
                    linewidth=1.0,
                    alpha=0.7,
                )

        # 子图3: 弯曲度分析
        linewidth = 3.0 if is_selected else 1.5
        alpha = 1.0 if is_selected else 0.6
        linestyle = "-" if is_quality else "--"

        ax3.plot(
            x_orig,
            y_orig,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
        )

        # 如果是选中的弯曲车道线，特别标注
        if is_selected:
            # 在起点和终点标记
            ax3.scatter(
                [x_orig[0], x_orig[-1]],
                [y_orig[0], y_orig[-1]],
                c="yellow",
                s=100,
                marker="*",
                zorder=5,
                edgecolors="red",
                linewidth=1.5,
            )

            # 显示弯曲度信息
            info_text = f"弯曲度: {bending_score:.1f}\n"
            info_text += f"平均曲率: {mean_curv:.4f}"
            if not is_quality:
                info_text += "\n(低质量)"

            # 在车道线中点显示信息
            mid_idx = len(x_orig) // 2
            ax3.text(
                x_orig[mid_idx],
                y_orig[mid_idx] - 20,
                info_text,
                fontsize=9,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8),
            )

    # 设置子图属性
    for ax, title in zip([ax1, ax2, ax3], ["原始标注点", "固定点采样", "弯曲度分析"]):
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_xlabel("X (pixels)", fontsize=10)
        ax.set_ylabel("Y (pixels)", fontsize=10)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_aspect("equal")

    # 添加统计信息
    total_lanes = len(data_info["lanes"])
    selected_lanes = len(selected_lane_ids) if selected_lane_ids is not None else 0
    fig.suptitle(
        f"样本 {idx} | 总车道线: {total_lanes} | 选中弯曲车道: {selected_lanes} | {data_info['img_path']}",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()
    output_path = osp.join(output_dir, f"sample_{idx:06d}_curvature_analysis.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")

    if SHOW_IMAGES:
        plt.show()
    else:
        plt.close()

    print(f"已保存: {output_path}")
    return output_path


# ============================ 主函数 ============================
def main():
    print("=" * 60)
    print("OpenLane 可视化 - 基于曲率的弯曲车道线选择 (改进版)")
    print("=" * 60)
    with open(CACHE_PATH, "rb") as f:
        data_infos = pickle.load(f)
    print(f"数据集大小: {len(data_infos)} 个样本")
    if ID_MODE == "curvature":
        print("\n使用曲率分析方法选择最弯曲的车道线...")
        selected_records, stats = analyze_openlane_by_curvature(
            CACHE_PATH, top_k=TOP_K, min_quality=True, scene_deduplicate=True
        )
    else:
        print("警告: 使用旧的选择模式，可能无法选择到真正的弯曲车道线")
        return

    print(f"\n选择了 {len(selected_records)} 个包含弯曲车道线的样本 (已去重)")

    # 可视化选中的样本
    for i, record in enumerate(selected_records):
        idx = record["sample_id"]
        lane_id = record["lane_id"]

        if idx >= len(data_infos):
            continue

        print(
            f"\n[{i+1}/{len(selected_records)}] 处理样本 {idx}: {data_infos[idx]['img_path']}"
        )
        print(f"  场景ID: {record['scene_id']}")
        print(
            f"  车道线 {lane_id}: 弯曲度={record['bending_score']:.2f}, "
            f"平均曲率={record['mean_curvature']:.5f}"
        )
        print(
            f"  统计: 点数={record['num_points']}, y范围={record['y_range']:.1f}, "
            f"总长度={record['total_length']:.1f}"
        )

        # 可视化
        output_path = visualize_sample_with_curvature(
            data_infos[idx],
            idx,
            NUM_SAMPLE_POINTS,
            OUTPUT_DIR,
            IMG_W,
            IMG_H,
            [lane_id],  # 只可视化选中的车道线
        )

        if output_path:
            print(f"  已保存: {osp.basename(output_path)}")

    print(f"\n{'='*60}")
    print(f"可视化完成! 结果保存在: {OUTPUT_DIR}")
    print(f"共处理了 {len(selected_records)} 个样本")
    print(f"平均弯曲度: {stats['avg_bending_score']:.2f}")
    print(f"平均曲率: {stats['avg_curvature']:.5f}")
    print(f"平均点数: {stats['avg_points']:.1f}")
    print(f"平均y范围: {stats['avg_y_range']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
