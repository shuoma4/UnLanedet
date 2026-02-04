import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.getcwd())

from unlanedet.data.transform.lane_encoder import LaneEncoder
from config.llanet.priors.sample_ys import (
    SAMPLE_YS_IOSDENSITY,
    SAMPLE_YS_EQUIDISTANT,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

CACHE_PATH = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/openlane_lane3d_1000_train_cuth-270_800x320_cache_v1.pkl"
OUTPUT_DIR = "./vis/encoder_precision"
IMG_W = 800
IMG_H = 320

CURVE_IDS = [111017, 91254]

SAMPLE_IDS = [
    47642,
    34490,
    99180,
    83308,
    91254,
    85227,
    7456,
    43216,
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
]


class MockConfig:
    def __init__(self):
        self.img_w = IMG_W
        self.img_h = IMG_H
        self.num_points = 72
        self.max_lanes = 20
        self.cut_height = 0
        self.sample_ys_mode = "equal_interval"
        self.sample_lane_mode = "linear_interp"
        self.sample_ys = SAMPLE_YS_IOSDENSITY


def load_cache(cache_path):
    with open(cache_path, "rb") as f:
        data_infos = pickle.load(f)
    return data_infos


def lane_to_arrays(lane_pts):
    lane = np.array(lane_pts, dtype=np.float32)
    if lane.shape[0] < 2:
        return None, None
    if lane[0, 1] < lane[-1, 1]:
        lane = lane[::-1]
    return lane[:, 0], lane[:, 1]


def gt_interp(xs, ys, sample_ys):
    return np.interp(sample_ys, ys[::-1], xs[::-1])


def reconstruct_xs_from_reg(reg, sample_ys):
    start_x, start_y, theta = reg[0], reg[1], reg[2]
    theta_rad = np.deg2rad(theta)
    x_prior = start_x + (start_y - sample_ys) / (np.tan(theta_rad) + 1e-6)
    delta_x = reg[4:]
    return x_prior + delta_x


def eval_group(sample_lane_mode, sample_ys_mode, data_infos, vis_data_collector=None):
    cfg = MockConfig()
    cfg.sample_lane_mode = sample_lane_mode
    cfg.sample_ys_mode = sample_ys_mode

    if sample_ys_mode == "equal_interval":
        cfg.sample_ys = SAMPLE_YS_EQUIDISTANT
        cfg.num_points = len(SAMPLE_YS_EQUIDISTANT)
    else:
        cfg.sample_ys = SAMPLE_YS_IOSDENSITY
        cfg.num_points = len(SAMPLE_YS_IOSDENSITY)

    encoder = LaneEncoder(cfg)
    method_key = f"{sample_lane_mode}|{sample_ys_mode}"

    mae_list, rmse_list, cover_list = [], [], []

    for idx in SAMPLE_IDS:
        if idx >= len(data_infos):
            continue
        sample = data_infos[idx]
        lanes = sample.get("lanes", [])

        for lane in lanes:
            xs, ys = lane_to_arrays(lane)
            if xs is None:
                continue

            if (
                vis_data_collector is not None
                and method_key == "linear_interp|equal_interval"
            ):
                vis_data_collector.setdefault(idx, {}).setdefault("GT", []).append(
                    (xs, ys)
                )

            reg, sample_ys = encoder.encode_lane(lane)
            xs_rec = reconstruct_xs_from_reg(reg, sample_ys)
            xs_gt = gt_interp(xs, ys, sample_ys)

            y_min, y_max = ys.min(), ys.max()
            mask = (
                (sample_ys >= y_min)
                & (sample_ys <= y_max)
                & (xs_rec >= 0)
                & (xs_rec < IMG_W)
                & (xs_gt >= 0)
                & (xs_gt < IMG_W)
            )

            if mask.sum() == 0:
                continue

            if vis_data_collector is not None:
                vis_data_collector.setdefault(idx, {}).setdefault(
                    method_key, []
                ).append((xs_rec[mask], sample_ys[mask]))

            err = xs_rec[mask] - xs_gt[mask]
            mae_list.append(np.mean(np.abs(err)))
            rmse_list.append(np.sqrt(np.mean(err**2)))
            cover_list.append(mask.sum() / len(sample_ys))

    return dict(
        mae=float(np.mean(mae_list)),
        rmse=float(np.mean(rmse_list)),
        cover=float(np.mean(cover_list)),
        count=len(mae_list),
    )


# ======================= 高质量绘图 =======================


def create_high_quality_figure(
    image_paths, lanes_data_list, titles, output_path, dpi=300
):
    """
    创建高质量4行2列可视化图，适合论文使用

    重新排列顺序：
    第一行: (0,0) GT | (0,1) 原图
    第二行: (1,0) Linear + Eq-Int | (1,1) ArcLen + Eq-Int
    第三行: (2,0) Linear + Eq-Den | (2,1) ArcLen + Eq-Den
    第四行: (3,0) Linear + Adapt | (3,1) ArcLen + Adapt

    修改为绘制点而非线条
    """
    # 设置matplotlib样式 - 论文质量
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.5,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.format": "pdf",  # 输出为矢量图格式
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,  # 减小边距使布局更紧凑
        }
    )

    # 创建4行2列的子图，更紧凑的布局
    fig, axes = plt.subplots(4, 2, figsize=(7.5, 10))  # 更紧凑的宽高比

    # 定义布局映射: 新位置 -> (方法键, 标题)
    layout_mapping = [
        (0, 0, "GT", "GT"),
        (0, 1, None, "Original"),  # 原图，没有车道线数据
        (1, 0, "linear_interp|equal_interval", "Linear + Eq-Int"),
        (1, 1, "arc_length|equal_interval", "ArcLen + Eq-Int"),
        (2, 0, "linear_interp|equal_density", "Linear + Eq-Den"),
        (2, 1, "arc_length|equal_density", "ArcLen + Eq-Den"),
        (3, 0, "linear_interp|lane_adaptive", "Linear + Adapt"),
        (3, 1, "arc_length|lane_adaptive", "ArcLen + Adapt"),
    ]

    # 定义颜色方案，确保不同方法有足够的对比度
    colors = {
        "GT": (1.0, 0.0, 0.0),  # 红色，真实车道线
        "linear_interp|equal_interval": (0.0, 0.4, 1.0),  # 蓝色
        "linear_interp|equal_density": (0.0, 0.7, 0.0),  # 绿色
        "linear_interp|lane_adaptive": (0.7, 0.0, 0.7),  # 紫色
        "arc_length|equal_interval": (1.0, 0.5, 0.0),  # 橙色
        "arc_length|equal_density": (0.6, 0.3, 0.1),  # 棕色
        "arc_length|lane_adaptive": (1.0, 0.0, 0.5),  # 洋红色
    }

    # 点的大小和透明度
    point_size = 1
    point_alpha = 0.8

    # 为每个布局位置绘制图像
    for row, col, method_key, title in layout_mapping:
        ax = axes[row, col]

        # 获取对应的样本索引
        # 我们假设传入的image_paths和lanes_data_list已经按照正确的顺序排列
        # 但现在需要根据layout_mapping重新映射
        idx = layout_mapping.index((row, col, method_key, title))

        # 确保索引在范围内
        if idx >= len(image_paths):
            continue

        img_path = image_paths[idx]

        # 读取并显示图像
        img = cv2.imread(img_path)
        if img is None:
            ax.text(
                0.5,
                0.5,
                "Image not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            continue

        # 转换为RGB并调整大小到800x320
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 320))

        # 显示图像
        ax.imshow(img, aspect="auto")

        # 如果不是原图（原图没有方法键），则绘制车道线点
        if method_key is not None and idx < len(lanes_data_list):
            lanes_data = lanes_data_list[idx]

            if method_key in lanes_data:
                lanes = lanes_data[method_key]

                # 为每条车道线绘制点
                for xs, ys in lanes:
                    if len(xs) > 0 and len(ys) > 0:
                        # 使用散点图绘制点，而非折线图
                        ax.scatter(
                            xs,
                            ys,
                            color=colors.get(method_key, (0, 0, 0)),
                            s=point_size,
                            alpha=point_alpha,
                            # edgecolors="white",  # 白色边缘增强可见性
                            # linewidths=0.3,
                            marker="o",
                        )

        # 设置子图标题
        ax.set_title(title, fontsize=10, fontweight="bold", pad=5)

        # 隐藏坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 800)
        ax.set_ylim(320, 0)  # 图像坐标系，y轴向下

        # 移除边框
        for spine in ax.spines.values():
            spine.set_visible(False)

    # 调整子图间距，使布局更紧凑
    plt.subplots_adjust(
        left=0.05,  # 减小左边距
        right=0.95,  # 减小右边距
        bottom=0.05,  # 减小底边距
        top=0.95,  # 减小顶边距
        wspace=0.05,  # 减小水平间距（使两列更紧凑）
        hspace=0.12,  # 垂直间距
    )

    # 添加图例说明（放置在图形底部，不占用太多空间）
    from matplotlib.patches import Patch

    legend_elements = []
    for method_key, color in colors.items():
        # 简化图例标签
        if method_key == "GT":
            label = "Ground Truth"
        else:
            # 解析方法名称
            parts = method_key.split("|")
            if len(parts) == 2:
                label = f"{parts[0].replace('_', ' ')} + {parts[1].replace('_', '-')}"
            else:
                label = method_key
        legend_elements.append(
            Patch(
                facecolor=color,
                edgecolor="black",
                linewidth=0.3,
                label=label,
                alpha=0.8,
            )
        )

    # 将图例放在图形底部中央，使用紧凑布局
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=7,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),  # 更靠近底部
        handletextpad=0.3,
        columnspacing=0.5,
    )

    # 保存为矢量图
    plt.savefig(output_path, format="pdf", dpi=dpi, bbox_inches="tight")
    print(f"保存高质量矢量图: {output_path}")

    # 同时保存为高分辨率PNG
    png_path = output_path.replace(".pdf", ".png")
    plt.savefig(
        png_path, format="png", dpi=600, bbox_inches="tight"
    )  # 600 DPI 超高分辨率
    print(f"保存高分辨率PNG: {png_path}")

    plt.close(fig)


def prepare_data_for_figure(sample_ids, data_infos, vis_data_collector):
    """
    准备数据用于创建图形
    新的布局需要重新排列数据顺序
    """
    # 只使用第一个样本（假设所有方法都在同一个样本上评估）
    sample_id = sample_ids[0]
    if sample_id >= len(data_infos):
        return [], [], []

    sample = data_infos[sample_id]

    # 获取图像路径（重复8次用于8个子图）
    img_path = osp.join(
        "/data1/lxy_log/workspace/ms/OpenLane/images", sample.get("img_path", "")
    )
    image_paths = [img_path] * 8

    # 定义新的布局顺序
    layout_methods = [
        "GT",
        None,  # Original
        "linear_interp|equal_interval",
        "arc_length|equal_interval",
        "linear_interp|equal_density",
        "arc_length|equal_density",
        "linear_interp|lane_adaptive",
        "arc_length|lane_adaptive",
    ]

    layout_titles = [
        "GT",
        "Original",
        "Linear + Eq-Int",
        "ArcLen + Eq-Int",
        "Linear + Eq-Den",
        "ArcLen + Eq-Den",
        "Linear + Adapt",
        "ArcLen + Adapt",
    ]

    # 准备车道线数据
    lanes_data_list = []
    sample_data = vis_data_collector.get(sample_id, {})

    for method_key in layout_methods:
        if method_key is None:  # Original，没有车道线数据
            lanes_data_list.append({})
        else:
            # 获取该方法下的车道线数据
            if method_key in sample_data:
                lanes_data_list.append({method_key: sample_data[method_key]})
            else:
                lanes_data_list.append({method_key: []})

    return image_paths, lanes_data_list, layout_titles


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_infos = load_cache(CACHE_PATH)

    vis_data_collector = {}
    results = {}

    # 1️⃣ 先跑评估 + 收集可视化数据（所有 sample 共用一次）
    for slm in ["linear_interp", "arc_length"]:
        for sym in ["equal_interval", "equal_density", "lane_adaptive"]:
            key = f"{slm}|{sym}"
            results[key] = eval_group(slm, sym, data_infos, vis_data_collector)

    # 2️⃣ 对每个 sample_id 单独绘制一张对比图
    for sample_id in SAMPLE_IDS:
        if sample_id >= len(data_infos):
            continue

        image_paths, lanes_data_list, titles = prepare_data_for_figure(
            [sample_id], data_infos, vis_data_collector
        )

        out_pdf = osp.join(OUTPUT_DIR, f"lane_sampling_precision_{sample_id}.pdf")
        create_high_quality_figure(image_paths, lanes_data_list, titles, out_pdf)

    print("Done. All samples visualization finished.")


if __name__ == "__main__":
    main()
