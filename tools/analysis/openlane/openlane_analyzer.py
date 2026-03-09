import json
import os
import os.path as osp
import pickle
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'AR PL UMing CN'


def save_detailed_lane_info(all_lane_params, output_dir):
    """保存每条车道线的详细信息到NPZ文件"""
    # 提取结构化数据
    sample_indices = np.array([info['sample_idx'] for info in all_lane_params])
    lane_indices = np.array([info['lane_idx'] for info in all_lane_params])
    start_x = np.array([info['start_x'] for info in all_lane_params])
    start_y = np.array([info['start_y'] for info in all_lane_params])
    thetas = np.array([info['theta'] for info in all_lane_params])
    lengths = np.array([info['length'] for info in all_lane_params])
    num_points = np.array([info['num_points'] for info in all_lane_params])

    # 保存到NPZ
    output_path = osp.join(output_dir, 'detailed_lane_parameters.npz')
    np.savez(
        output_path,
        sample_indices=sample_indices,
        lane_indices=lane_indices,
        start_x=start_x,
        start_y=start_y,
        thetas=thetas,
        lengths=lengths,
        num_points=num_points,
        total_lanes=len(all_lane_params),
    )
    print(f'详细车道线参数已保存到: {output_path}')
    print(f'包含 {len(all_lane_params)} 条车道线的详细信息')

    # 同时保存文本摘要
    summary_path = osp.join(output_dir, 'lane_parameters_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('OpenLane车道线详细参数摘要\n')
        f.write('=' * 50 + '\n')
        f.write(f'总车道线数量: {len(all_lane_params)}\n')
        f.write(f'起点X范围: [{start_x.min():.4f}, {start_x.max():.4f}]\n')
        f.write(f'起点Y范围: [{start_y.min():.4f}, {start_y.max():.4f}]\n')
        f.write(f'角度θ范围: [{thetas.min():.4f}, {thetas.max():.4f}]\n')
        f.write(f'长度范围: [{lengths.min():.2f}, {lengths.max():.2f}] 像素\n')
        f.write(f'平均点数: {num_points.mean():.1f}\n')

        # 角度分布
        left_ratio = np.mean(thetas < 0.5)
        right_ratio = np.mean(thetas > 0.5)
        vertical_ratio = np.mean((thetas >= 0.45) & (thetas <= 0.55))
        f.write(f'左倾比例: {left_ratio:.2%}\n')
        f.write(f'右倾比例: {right_ratio:.2%}\n')
        f.write(f'垂直比例: {vertical_ratio:.2%}\n')

    return output_path


def analyze_openlane_distribution_vector_method(pkl_path, output_dir='./statistics'):
    """
    使用向量累加法分析OpenLane数据集分布，并保存每条车道线的详细信息
    只进行基础的数据分析和绘图，不进行聚类
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f'加载pkl文件: {pkl_path}')
    with open(pkl_path, 'rb') as f:
        data_infos = pickle.load(f)

    print(f'成功加载 {len(data_infos)} 个样本')

    all_lane_params = []  # 存储每条车道线的详细参数
    all_params = []  # 用于统计的参数
    positive_ratios = []
    all_categories = []

    print('处理样本并提取车道线参数...')
    for sample_idx, sample in enumerate(tqdm(data_infos, desc='处理样本')):
        lanes = sample.get('lanes', [])
        lane_categories_sample = sample.get('lane_categories', [])
        if not lanes:
            continue
        all_categories.extend(lane_categories_sample)
        # 构建分割掩码并统计正样本比例
        try:
            img_w_default, img_h_default = 800, 320
            mask_scale = 8
            mask_h, mask_w = int(img_h_default // mask_scale), int(img_w_default // mask_scale)
            seg_map = np.zeros((mask_h, mask_w), dtype=np.uint8)
            for lane_points in lanes:
                pts = np.array(lane_points, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
                    continue
                if np.max(pts[:, 0]) <= 1.0 and np.max(pts[:, 1]) <= 1.0:
                    pts_px = pts.copy()
                    pts_px[:, 0] = pts[:, 0] * img_w_default
                    pts_px[:, 1] = pts[:, 1] * img_h_default
                else:
                    pts_px = pts
                pts_scaled = (pts_px / mask_scale).astype(np.int32)
                cv2.polylines(seg_map, [pts_scaled], isClosed=False, color=1, thickness=1)
            total_pixels = seg_map.size
            positive_pixels = int(np.sum(seg_map == 1))
            positive_ratio = positive_pixels / float(total_pixels)
            positive_ratios.append(positive_ratio)
        except Exception:
            pass

        for lane_idx, lane_points in enumerate(lanes):
            if len(lane_points) < 3:  # 至少需要3个点才能使用向量法
                continue

            # 1. 找到y坐标最大的点（底部）
            y_coords = lane_points[:, 1]
            start_idx = np.argmax(y_coords)
            start_x, start_y = lane_points[start_idx, 0], lane_points[start_idx, 1]

            # 2. 使用向量累加法计算角度
            theta = calculate_angle_by_vector_method(lane_points)

            # 3. 计算长度
            end_idx = np.argmin(y_coords)
            end_x, end_y = lane_points[end_idx, 0], lane_points[end_idx, 1]
            length = abs(end_y - start_y)

            # 过滤异常值
            if length < 0.1 or not (0.1 <= theta <= 0.9):
                continue

            # 存储详细的车道线信息
            lane_info = {
                'sample_idx': sample_idx,
                'lane_idx': lane_idx,
                'start_x': start_x,
                'start_y': start_y,
                'theta': theta,
                'length': length,
                'num_points': len(lane_points),
            }
            all_lane_params.append(lane_info)

            # 用于统计的参数
            all_params.append([start_x, start_y, theta, length])

    if not all_lane_params:
        print('错误：没有提取到有效的车道线参数！')
        return None, None

    all_params = np.array(all_params)
    print(f'提取了 {len(all_lane_params)} 条有效车道线参数')

    # 保存每条车道线的详细信息到NPZ
    detailed_params_path = save_detailed_lane_info(all_lane_params, output_dir)

    # 分离各个参数用于统计
    start_x = all_params[:, 0]
    start_y = all_params[:, 1]
    thetas = all_params[:, 2]
    lengths = all_params[:, 3]

    # 基本统计
    stats = calculate_statistics_vector(start_x, start_y, thetas, lengths)
    print_statistics_vector(stats)

    # 可视化
    visualize_distributions_vector_method(start_x, start_y, thetas, lengths, output_dir)

    # 保存统计结果（不包含聚类中心）
    save_statistics_without_clustering(stats, output_dir)

    try:
        regression_parameters = {
            'start_x': start_x,
            'start_y': start_y,
            'thetas': thetas,
            'lengths': lengths,
        }
        reg_stats, x_grid_500, y_grid_500, x_pdf_500, y_pdf_500 = analyze_regression_prior(
            regression_parameters, output_dir
        )
    except Exception as _e:
        print(f'回归先验统计失败: {_e}')

    # 分类与分割统计与可视化
    try:
        cls_stats = analyze_classification_prior(
            {'lane_categories': [np.array(all_categories, dtype=np.int64)]},
            num_categories=14,
            output_dir=output_dir,
        )
    except Exception as _e:
        print(f'分类先验统计失败: {_e}')
        cls_stats = None

    try:
        if len(positive_ratios) > 0:
            seg_stats = {
                'positive_ratio_mean': float(np.mean(positive_ratios)),
                'positive_ratio_std': float(np.std(positive_ratios)),
                'positive_ratio_min': float(np.min(positive_ratios)),
                'positive_ratio_max': float(np.max(positive_ratios)),
                'positive_ratio_median': float(np.median(positive_ratios)),
                'total_images': int(len(positive_ratios)),
            }
            plt.figure(figsize=(10, 6))
            plt.hist(positive_ratios, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(
                seg_stats['positive_ratio_mean'],
                color='red',
                linestyle='--',
                label=f'均值: {seg_stats["positive_ratio_mean"]:.4f}',
            )
            plt.xlabel('正样本比例')
            plt.ylabel('图片数量')
            plt.title('分割任务正样本比例分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                osp.join(output_dir, 'segmentation_positive_ratio_distribution.png'),
                dpi=300,
                bbox_inches='tight',
            )
            plt.close()
        else:
            seg_stats = None
    except Exception as _e:
        print(f'分割先验统计失败: {_e}')
        seg_stats = None

    try:
        save_all_priors_npz(
            output_dir=output_dir,
            start_x=start_x,
            start_y=start_y,
            thetas=thetas,
            lengths=lengths,
            vector_stats=stats,
            start_x_grid=x_grid_500 if 'x_grid_500' in locals() else None,
            start_x_pdf=x_pdf_500 if 'x_pdf_500' in locals() else None,
            start_y_grid=y_grid_500 if 'y_grid_500' in locals() else None,
            start_y_pdf=y_pdf_500 if 'y_pdf_500' in locals() else None,
            seg_stats=seg_stats,
            seg_positive_ratios=np.array(positive_ratios, dtype=np.float32),
            cls_stats=cls_stats,
        )
    except Exception as _e:
        print(f'保存综合NPZ失败: {_e}')

    return stats, detailed_params_path


def calculate_angle_by_vector_method(lane_points):
    """
    使用向量累加法计算车道线角度
    对连续两点之间的向量计算sin,cos并累加，最后用atan2求平均角度
    """
    # 确保车道线点按y坐标排序（从下到上）
    sorted_indices = np.argsort(lane_points[:, 1])[::-1]  # 从大到小排序
    sorted_points = lane_points[sorted_indices]

    if len(sorted_points) < 2:
        return 0.5  # 默认垂直

    # 初始化sin和cos的累加值
    total_sin = 0.0
    total_cos = 0.0
    valid_vectors = 0

    # 计算每个相邻点对之间的向量
    for i in range(len(sorted_points) - 1):
        # 当前点对
        p1 = sorted_points[i]  # 下面的点（y值较大）
        p2 = sorted_points[i + 1]  # 上面的点（y值较小）

        # 计算向量 (从p1指向p2)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]  # 注意：dy为负值（向上）

        # 计算向量长度
        vector_length = np.sqrt(dx**2 + dy**2)

        # 过滤掉太短的向量（可能是噪声）
        if vector_length < 1e-5:
            continue

        # 计算单位向量的sin和cos
        # 注意：我们需要计算与y轴负方向的夹角（因为y轴向下为正）
        # 所以实际的角度是atan2(dx, -dy)
        sin_val = dx / vector_length
        cos_val = -dy / vector_length  # 使用-dy因为y轴方向

        total_sin += sin_val
        total_cos += cos_val
        valid_vectors += 1

    if valid_vectors == 0:
        return 0.5  # 默认垂直

    # 计算平均sin和cos
    avg_sin = total_sin / valid_vectors
    avg_cos = total_cos / valid_vectors

    # 使用atan2计算平均角度
    # atan2(avg_sin, avg_cos) 给出与x轴的夹角，但我们需要与y轴的夹角
    angle_rad = np.arctan2(avg_sin, avg_cos)

    # 将角度归一化到[0, 1]范围
    # 0.5表示垂直（与y轴平行）
    # <0.5表示左倾，>0.5表示右倾
    theta = 0.5 + angle_rad / (2 * np.pi)  # 缩放因子使角度变化更平缓
    theta = np.clip(theta, 0.1, 0.9)  # 限制在合理范围

    return theta


def calculate_statistics_vector(start_x, start_y, thetas, lengths):
    """计算统计信息（向量法）"""
    stats = {
        'start_x': {
            'mean': np.mean(start_x),
            'std': np.std(start_x),
            'min': np.min(start_x),
            'max': np.max(start_x),
            'median': np.median(start_x),
        },
        'start_y': {
            'mean': np.mean(start_y),
            'std': np.std(start_y),
            'min': np.min(start_y),
            'max': np.max(start_y),
            'median': np.median(start_y),
        },
        'theta': {
            'mean': np.mean(thetas),
            'std': np.std(thetas),
            'min': np.min(thetas),
            'max': np.max(thetas),
            'median': np.median(thetas),
            'left_leaning_ratio': np.mean(thetas < 0.5),
            'right_leaning_ratio': np.mean(thetas > 0.5),
            'vertical_ratio': np.mean((thetas >= 0.49) & (thetas <= 0.51)),
        },
        'length': {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'median': np.median(lengths),
        },
    }

    return stats


def print_statistics_vector(stats):
    """打印统计结果（向量法）"""
    print('\n' + '=' * 70)
    print('车道线分布统计结果（向量累加法）')
    print('角度定义: 0.5=垂直, <0.5=左倾(\\), >0.5=右倾(/)')
    print('=' * 70)

    for param_name, param_stats in stats.items():
        print(f'\n{param_name.upper()} 统计:')
        for stat_name, value in param_stats.items():
            if 'ratio' in stat_name:
                print(f'  {stat_name}: {value:.2%}')
            else:
                print(f'  {stat_name}: {value:.4f}')


def visualize_distributions_vector_method(start_x, start_y, thetas, lengths, output_dir):
    """可视化分布（向量法）"""
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 起点X分布
    axes[0, 0].hist(start_x, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('起点X坐标 (0=左, 1=右)')
    axes[0, 0].set_ylabel('频率')
    axes[0, 0].set_title('起点X分布')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 起点Y分布
    axes[0, 1].hist(start_y, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('起点Y坐标 (0=上, 1=下)')
    axes[0, 1].set_ylabel('频率')
    axes[0, 1].set_title('起点Y分布')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 角度分布（重点）
    n, bins, patches = axes[0, 2].hist(thetas, bins=50, alpha=0.7, color='coral', edgecolor='black')

    # 标记不同区域
    axes[0, 2].axvline(0.5, color='red', linestyle='--', linewidth=2, label='垂直 (0.5)')
    axes[0, 2].axvspan(0.1, 0.5, alpha=0.2, color='blue', label='左倾 (<0.5)')
    axes[0, 2].axvspan(0.5, 0.9, alpha=0.2, color='green', label='右倾 (>0.5)')

    axes[0, 2].set_xlabel('角度θ (0.5=垂直, <0.5=左倾, >0.5=右倾)')
    axes[0, 2].set_ylabel('频率')
    axes[0, 2].set_title('车道线角度分布（向量累加法）')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 长度分布
    axes[1, 0].hist(lengths, bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 0].set_xlabel('车道线长度')
    axes[1, 0].set_ylabel('频率')
    axes[1, 0].set_title('长度分布')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. X-Y分布与角度关系
    scatter = axes[1, 1].scatter(start_x, start_y, c=thetas, cmap='coolwarm', alpha=0.3, s=1, vmin=0.3, vmax=0.7)
    axes[1, 1].set_xlabel('起点X坐标')
    axes[1, 1].set_ylabel('起点Y坐标')
    axes[1, 1].set_title('起点分布（颜色表示角度）')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='角度θ')

    # 6. 角度详细分布
    axes[1, 2].hist(thetas, bins=100, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 2].axvline(0.5, color='red', linestyle='--', linewidth=2, label='垂直')

    # 标记典型角度值
    for angle, label in [
        (0.3, '陡左倾'),
        (0.4, '缓左倾'),
        (0.5, '垂直'),
        (0.6, '缓右倾'),
        (0.7, '陡右倾'),
    ]:
        axes[1, 2].axvline(angle, color='gray', linestyle=':', alpha=0.7)
        axes[1, 2].text(
            angle,
            axes[1, 2].get_ylim()[1] * 0.9,
            label,
            rotation=90,
            verticalalignment='top',
            horizontalalignment='center',
        )

    axes[1, 2].set_xlabel('角度θ')
    axes[1, 2].set_ylabel('频率')
    axes[1, 2].set_title('角度详细分布')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        osp.join(output_dir, 'vector_method_distributions.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # 创建角度分布专题图
    plt.figure(figsize=(12, 8))

    # 计算角度分布
    n, bins = np.histogram(thetas, bins=100, range=(0.1, 0.9))
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # 创建颜色映射：左倾蓝色，垂直红色，右倾绿色
    colors = []
    for center in bin_centers:
        if center < 0.5:
            colors.append('blue')
        elif center > 0.5:
            colors.append('green')
        else:
            colors.append('red')

    plt.bar(bin_centers, n, width=bins[1] - bins[0], alpha=0.7, color=colors)

    plt.axvline(0.5, color='red', linestyle='--', linewidth=3, label='垂直车道线 (|)')

    # 添加区域标注
    plt.axvspan(0.1, 0.5, alpha=0.1, color='blue', label='左倾车道线 (\\)')
    plt.axvspan(0.5, 0.9, alpha=0.1, color='green', label='右倾车道线 (/)')

    plt.xlabel('角度θ (0.5=垂直, <0.5=左倾, >0.5=右倾)')
    plt.ylabel('车道线数量')
    plt.title('OpenLane车道线角度分布（向量累加法）')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    left_ratio = np.mean(thetas < 0.5)
    right_ratio = np.mean(thetas > 0.5)
    vertical_ratio = np.mean((thetas >= 0.49) & (thetas <= 0.51))

    stats_text = f'左倾: {left_ratio:.1%}\n右倾: {right_ratio:.1%}\n垂直: {vertical_ratio:.1%}'
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        osp.join(output_dir, 'vector_method_angle_distribution.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # 创建起点分布热力图
    plt.figure(figsize=(10, 8))

    # 计算起点分布的2D密度
    if len(start_x) > 0 and len(start_y) > 0:
        # 创建2D直方图
        h, xedges, yedges = np.histogram2d(start_x, start_y, bins=50)

        # 显示热力图
        plt.imshow(
            h.T,
            origin='lower',
            aspect='auto',
            extent=[0, 1, 0, 1],
            cmap='hot',
            interpolation='nearest',
        )
        plt.colorbar(label='密度')
        plt.xlabel('起点X坐标')
        plt.ylabel('起点Y坐标')
        plt.title('起点分布热力图')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            osp.join(output_dir, 'start_point_heatmap.png'),
            dpi=300,
            bbox_inches='tight',
        )
        plt.close()

    print(f'可视化图表已保存到: {output_dir}')


def save_statistics_without_clustering(stats, output_dir):
    """保存统计结果（不包含聚类）"""
    # 保存为文本文件
    with open(osp.join(output_dir, 'statistics_summary.txt'), 'w') as f:
        f.write('OpenLane车道线分布统计结果（基础分析）\n')
        f.write('角度定义: 0.5=垂直, <0.5=左倾(\\), >0.5=右倾(/)\n')
        f.write('=' * 70 + '\n\n')

        for param_name, param_stats in stats.items():
            f.write(f'{param_name.upper()} 统计:\n')
            for stat_name, value in param_stats.items():
                if 'ratio' in stat_name:
                    f.write(f'  {stat_name}: {value:.2%}\n')
                else:
                    f.write(f'  {stat_name}: {value:.6f}\n')
            f.write('\n')

    # 保存为numpy文件（只包含统计信息，不包含聚类中心）
    np.savez(osp.join(output_dir, 'statistics_data.npz'), stats=stats)

    print(f'统计结果已保存到: {output_dir}')


def analyze_segmentation_prior(batch_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seg_masks = batch_data.get('seg', [])
    if not seg_masks:
        print('警告: 未找到分割标签数据')
        return None
    import torch

    positive_ratios = []
    total_pixels_list = []
    positive_pixels_list = []
    for seg_mask in seg_masks:
        if isinstance(seg_mask, torch.Tensor):
            seg_mask = seg_mask.cpu().numpy()
        total_pixels = seg_mask.size
        positive_pixels = np.sum(seg_mask == 1)
        positive_ratio = positive_pixels / total_pixels
        positive_ratios.append(positive_ratio)
        total_pixels_list.append(total_pixels)
        positive_pixels_list.append(positive_pixels)
    seg_stats = {
        'positive_ratio_mean': float(np.mean(positive_ratios)),
        'positive_ratio_std': float(np.std(positive_ratios)),
        'positive_ratio_min': float(np.min(positive_ratios)),
        'positive_ratio_max': float(np.max(positive_ratios)),
        'positive_ratio_median': float(np.median(positive_ratios)),
        'total_images': len(seg_masks),
        'avg_positive_pixels_per_image': float(np.mean(positive_pixels_list)),
        'avg_total_pixels_per_image': float(np.mean(total_pixels_list)),
    }
    plt.figure(figsize=(10, 6))
    plt.hist(positive_ratios, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(
        seg_stats['positive_ratio_mean'],
        color='red',
        linestyle='--',
        label=f'均值: {seg_stats["positive_ratio_mean"]:.4f}',
    )
    plt.xlabel('正样本比例')
    plt.ylabel('图片数量')
    plt.title('分割任务正样本比例分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(
        osp.join(output_dir, 'segmentation_positive_ratio_distribution.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()
    print('分割正样本比例统计完成')
    print(f'  平均正样本比例: {seg_stats["positive_ratio_mean"]:.4f}')
    print(f'  标准差: {seg_stats["positive_ratio_std"]:.6f}')
    print(f'  范围: [{seg_stats["positive_ratio_min"]:.4f}, {seg_stats["positive_ratio_max"]:.4f}]')
    return seg_stats


def analyze_classification_prior(batch_data, num_categories=14, output_dir='./statistics'):
    os.makedirs(output_dir, exist_ok=True)
    lane_categories = batch_data.get('lane_categories', [])
    if not lane_categories:
        print('警告: 未找到车道线类别数据')
        return None
    import torch

    all_categories = []
    for categories in lane_categories:
        if isinstance(categories, torch.Tensor):
            categories = categories.cpu().numpy()
        valid_categories = categories[categories >= 0]
        all_categories.extend(valid_categories.tolist())
    if not all_categories:
        print('警告: 未找到有效的类别标签')
        return None
    all_categories = np.array(all_categories)
    num_categories_eff = int(np.max(all_categories)) + 1
    if num_categories_eff < num_categories:
        num_categories_eff = num_categories
    category_counts = np.bincount(all_categories.astype(int), minlength=num_categories_eff)
    total_samples = len(all_categories)
    category_ratios = category_counts / total_samples
    category_weights = np.where(category_ratios > 0, 1.0 / (category_ratios + 1e-8), 0.0)
    sum_w = np.sum(category_weights)
    if sum_w > 0:
        category_weights = category_weights / sum_w
    cls_stats = {
        'category_counts': category_counts.astype(int).tolist(),
        'category_ratios': category_ratios.astype(float).tolist(),
        'category_weights': category_weights.astype(float).tolist(),
        'total_samples': int(total_samples),
        'num_categories': int(num_categories_eff),
        'most_frequent_category': int(np.argmax(category_counts)),
        'least_frequent_category': int(
            (np.where(category_counts > 0)[0][np.argmin(category_counts[category_counts > 0])])
            if np.any(category_counts > 0)
            else -1
        ),
        'max_category_ratio': float(np.max(category_ratios)),
        'min_category_ratio': float(
            np.min(category_ratios[category_ratios > 0]) if np.any(category_ratios > 0) else 0.0
        ),
        'class_imbalance_ratio': float(
            (np.max(category_ratios) / np.min(category_ratios[category_ratios > 0]))
            if np.any(category_ratios > 0)
            else 0.0
        ),
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    categories = np.arange(num_categories_eff)
    bars = ax1.bar(categories, category_counts, alpha=0.7, color='lightcoral', edgecolor='black')
    ax1.set_xlabel('类别ID')
    ax1.set_ylabel('样本数量')
    ax1.set_title('车道线类别分布')
    ax1.grid(True, alpha=0.3)
    for bar, count in zip(bars, category_counts):
        if count > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{count}',
                ha='center',
                va='bottom',
                fontsize=8,
            )
    ax2.bar(categories, category_weights, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('类别ID')
    ax2.set_ylabel('类别权重')
    ax2.set_title('分类任务类别权重（逆频率加权）')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        osp.join(output_dir, 'classification_category_distribution.png'),
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()
    print('分类任务统计完成')
    print(f'  总样本数: {cls_stats["total_samples"]}')
    print(f'  类别数量: {num_categories_eff}')
    print(f'  最频繁类别: {cls_stats["most_frequent_category"]} (比例: {cls_stats["max_category_ratio"]:.2%})')
    print(f'  最稀有类别: {cls_stats["least_frequent_category"]} (比例: {cls_stats["min_category_ratio"]:.2%})')
    print(f'  类别不平衡比例: {cls_stats["class_imbalance_ratio"]:.2f}')
    return cls_stats


def analyze_regression_prior(parameters, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print('分析回归任务起始坐标分布...')
    start_x = parameters['start_x']
    start_y = parameters['start_y']
    thetas = parameters['thetas']
    lengths = parameters['lengths']
    max_points_env = os.environ.get('OPENLANE_MAX_POINTS_REGRESSION')
    hist_bins_1d_env = os.environ.get('OPENLANE_HIST_BINS_1D')
    hist_bins_2d_env = os.environ.get('OPENLANE_HIST_BINS_2D')
    kde_thr_1d_env = os.environ.get('OPENLANE_KDE_THRESHOLD_1D')
    hist2d_thr_env = os.environ.get('OPENLANE_HIST2D_THRESHOLD')
    skip_vis_env = os.environ.get('OPENLANE_SKIP_VIS')
    smooth_env = os.environ.get('OPENLANE_HIST2D_SMOOTH')
    max_points = int(max_points_env) if max_points_env and max_points_env.isdigit() else None
    bins_1d = int(hist_bins_1d_env) if hist_bins_1d_env and hist_bins_1d_env.isdigit() else 500
    bins_2d = int(hist_bins_2d_env) if hist_bins_2d_env and hist_bins_2d_env.isdigit() else 300
    thr1d = int(kde_thr_1d_env) if kde_thr_1d_env and kde_thr_1d_env.isdigit() else 200000
    thr2d = int(hist2d_thr_env) if hist2d_thr_env and hist2d_thr_env.isdigit() else 100000
    skip_vis = True if skip_vis_env == '1' else False
    smooth_sigma = float(smooth_env) if smooth_env else 1.0
    n_total = len(start_x)
    if max_points and n_total > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, size=max_points, replace=False)
        start_x = start_x[idx]
        start_y = start_y[idx]
    stats = {
        'start_x': {
            'mean': float(np.mean(start_x)),
            'std': float(np.std(start_x)),
            'min': float(np.min(start_x)),
            'max': float(np.max(start_x)),
            'median': float(np.median(start_x)),
        },
        'start_y': {
            'mean': float(np.mean(start_y)),
            'std': float(np.std(start_y)),
            'min': float(np.min(start_y)),
            'max': float(np.max(start_y)),
            'median': float(np.median(start_y)),
        },
        'theta': {
            'mean': float(np.mean(thetas)),
            'std': float(np.std(thetas)),
            'min': float(np.min(thetas)),
            'max': float(np.max(thetas)),
            'median': float(np.median(thetas)),
            'left_leaning_ratio': float(np.mean(thetas < 0.5)),
            'right_leaning_ratio': float(np.mean(thetas > 0.5)),
            'vertical_ratio': float(np.mean((thetas >= 0.49) & (thetas <= 0.51))),
        },
        'length': {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'min': float(np.min(lengths)),
            'max': float(np.max(lengths)),
            'median': float(np.median(lengths)),
        },
    }
    n_points = len(start_x)
    use_hist_1d = n_points > thr1d
    if use_hist_1d:
        print('使用直方图近似计算一维PDF')
    kde_x = gaussian_kde(start_x) if (len(start_x) > 1 and not use_hist_1d) else None
    kde_y = gaussian_kde(start_y) if (len(start_y) > 1 and not use_hist_1d) else None
    xs = np.linspace(0.0, 1.0, 500)
    ys = np.linspace(0.0, 1.0, 500)
    if kde_x is not None:
        px = kde_x(xs)
    else:
        hist_x, _ = np.histogram(start_x, bins=bins_1d, range=(0.0, 1.0))
        px = hist_x.astype(np.float64)
        px = px + 1e-12
        px = px / px.sum()
    if kde_y is not None:
        py = kde_y(ys)
    else:
        hist_y, _ = np.histogram(start_y, bins=bins_1d, range=(0.0, 1.0))
        py = hist_y.astype(np.float64)
        py = py + 1e-12
        py = py / py.sum()
    if not skip_vis:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(xs, px, color='blue')
        axes[0].set_xlabel('start_x')
        axes[0].set_ylabel('PDF')
        axes[0].set_title('起始X坐标PDF')
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(ys, py, color='green')
        axes[1].set_xlabel('start_y')
        axes[1].set_ylabel('PDF')
        axes[1].set_title('起始Y坐标PDF')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            osp.join(output_dir, 'regression_start_coordinate_pdf.png'),
            dpi=300,
            bbox_inches='tight',
        )
        plt.close()
    if len(start_x) > 1 and len(start_y) > 1:
        use_hist_2d = n_points > thr2d
        if use_hist_2d:
            print('使用直方图近似计算二维密度')
            H, xedges, yedges = np.histogram2d(start_x, start_y, bins=bins_2d, range=[[0.0, 1.0], [0.0, 1.0]])
            H = H.astype(np.float64)
            H = gaussian_filter(H, sigma=smooth_sigma)
            H = H / (H.sum() + 1e-12)
            pxy = H
            if not skip_vis:
                plt.figure(figsize=(7, 6))
                plt.imshow(
                    pxy.T,
                    origin='lower',
                    extent=[0, 1, 0, 1],
                    aspect='auto',
                    cmap='hot',
                )
                plt.colorbar(label='密度')
                plt.xlabel('start_x')
                plt.ylabel('start_y')
                plt.title('起始坐标二维密度')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    osp.join(output_dir, 'regression_start_xy_density.png'),
                    dpi=300,
                    bbox_inches='tight',
                )
                plt.close()
        else:
            xi, yi = np.meshgrid(xs, ys)
            xy = np.vstack([xi.ravel(), yi.ravel()])
            print('使用KDE计算二维密度')
            kde_xy = gaussian_kde(np.vstack([start_x, start_y]))
            pxy = kde_xy(xy).reshape(xi.shape)
            if not skip_vis:
                plt.figure(figsize=(7, 6))
                plt.imshow(
                    pxy.T,
                    origin='lower',
                    extent=[0, 1, 0, 1],
                    aspect='auto',
                    cmap='hot',
                )
                plt.colorbar(label='密度')
                plt.xlabel('start_x')
                plt.ylabel('start_y')
                plt.title('起始坐标二维密度')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    osp.join(output_dir, 'regression_start_xy_density.png'),
                    dpi=300,
                    bbox_inches='tight',
                )
                plt.close()
    np.savez(
        osp.join(output_dir, 'regression_prior_distributions.npz'),
        start_x_grid=xs,
        start_x_pdf=px,
        start_y_grid=ys,
        start_y_pdf=py,
        stats=stats,
    )
    print('回归任务起始坐标分布统计完成')
    return stats, xs, ys, px, py


def save_all_priors_npz(
    output_dir,
    start_x,
    start_y,
    thetas,
    lengths,
    vector_stats=None,
    start_x_grid=None,
    start_x_pdf=None,
    start_y_grid=None,
    start_y_pdf=None,
    seg_stats=None,
    seg_positive_ratios=None,
    cls_stats=None,
):
    os.makedirs(output_dir, exist_ok=True)
    npz_path = osp.join(output_dir, 'openlane_priors.npz')
    data_to_save = {
        'start_x': np.array(start_x, dtype=np.float32),
        'start_y': np.array(start_y, dtype=np.float32),
        'thetas': np.array(thetas, dtype=np.float32),
        'lengths': np.array(lengths, dtype=np.float32),
    }
    if vector_stats is not None:
        data_to_save['vector_stats_json'] = np.array([json.dumps(vector_stats)], dtype=object)
    if start_x_grid is not None and start_x_pdf is not None:
        data_to_save['start_x_grid'] = np.array(start_x_grid, dtype=np.float32)
        data_to_save['start_x_pdf'] = np.array(start_x_pdf, dtype=np.float32)
    if start_y_grid is not None and start_y_pdf is not None:
        data_to_save['start_y_grid'] = np.array(start_y_grid, dtype=np.float32)
        data_to_save['start_y_pdf'] = np.array(start_y_pdf, dtype=np.float32)
    if seg_positive_ratios is not None:
        data_to_save['seg_positive_ratios'] = np.array(seg_positive_ratios, dtype=np.float32)
    if seg_stats is not None:
        data_to_save['seg_stats_json'] = np.array([json.dumps(seg_stats)], dtype=object)
    if cls_stats is not None:
        data_to_save['cls_category_counts'] = np.array(cls_stats['category_counts'], dtype=np.int64)
        data_to_save['cls_category_ratios'] = np.array(cls_stats['category_ratios'], dtype=np.float32)
        data_to_save['cls_category_weights'] = np.array(cls_stats['category_weights'], dtype=np.float32)
        data_to_save['cls_total_samples'] = np.array([cls_stats['total_samples']], dtype=np.int64)
        data_to_save['cls_num_categories'] = np.array([cls_stats['num_categories']], dtype=np.int64)
        data_to_save['cls_most_frequent_category'] = np.array([cls_stats['most_frequent_category']], dtype=np.int64)
        data_to_save['cls_least_frequent_category'] = np.array([cls_stats['least_frequent_category']], dtype=np.int64)
        data_to_save['cls_max_category_ratio'] = np.array([cls_stats['max_category_ratio']], dtype=np.float32)
        data_to_save['cls_min_category_ratio'] = np.array([cls_stats['min_category_ratio']], dtype=np.float32)
        data_to_save['cls_class_imbalance_ratio'] = np.array([cls_stats['class_imbalance_ratio']], dtype=np.float32)
    np.savez(npz_path, **data_to_save)
    print(f'综合先验NPZ已保存到: {npz_path}')


# 主执行代码
if __name__ == '__main__':
    import glob
    import sys
    from concurrent.futures import ProcessPoolExecutor, as_completed

    def _run_worker(pkl_path, output_dir, gpu_id=None):
        try:
            if gpu_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            stats, detailed_params_path = analyze_openlane_distribution_vector_method(pkl_path, output_dir=output_dir)
            return (pkl_path, True, stats, detailed_params_path, None)
        except Exception as e:
            return (pkl_path, False, None, None, str(e))

    # 处理命令行参数或环境变量
    if len(sys.argv) > 1:
        pkl_path = sys.argv[1]
    else:
        # 默认路径
        pkl_path = '/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/openlane_lane3d_1000_train_cache_v5.pkl'

    # 如果文件不存在，尝试自动查找
    if not osp.exists(pkl_path):
        candidates = glob.glob('/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/*openlane*train*.pkl')
        if candidates:
            pkl_path = candidates[0]
            print(f'使用找到的文件: {pkl_path}')
        else:
            # 在当前目录查找
            candidates = glob.glob('./*openlane*train*.pkl')
            if candidates:
                pkl_path = candidates[0]
                print(f'使用当前目录的文件: {pkl_path}')
            else:
                print('请手动指定pkl文件路径')
                print('用法: python script.py /path/to/your/file.pkl')
                sys.exit(1)

    # 输出目录
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = './source/openlane_statistics'

    print('=' * 60)
    print('OpenLane车道线基础分析（向量累加法）')
    print('角度定义: 0.5=垂直, <0.5=左倾(\\), >0.5=右倾(/)')
    print('=' * 60)
    print(f'输入文件: {pkl_path}')
    print(f'输出目录: {output_dir}')
    print('=' * 60)

    try:
        import time

        gpu_ids_env = os.environ.get('OPENLANE_GPU_IDS')
        pkl_paths_env = os.environ.get('OPENLANE_PKL_PATHS')

        if gpu_ids_env and pkl_paths_env:
            GPU_IDS = [int(x) for x in gpu_ids_env.split(',') if x.strip() != '']
            PKL_PATHS = [x for x in pkl_paths_env.split(',') if x.strip() != '']
            print(f'检测到多GPU配置: {GPU_IDS}，启动并行分析')
            tasks = []
            for i, path in enumerate(PKL_PATHS):
                gid = GPU_IDS[i % len(GPU_IDS)]
                out_dir_i = os.path.join(output_dir, f'analysis_gpu{gid}')
                os.makedirs(out_dir_i, exist_ok=True)
                tasks.append((path, out_dir_i, gid))
            start_time = time.time()
            with ProcessPoolExecutor(max_workers=len(GPU_IDS)) as ex:
                futures = [ex.submit(_run_worker, p, od, gid) for p, od, gid in tasks]
                for fut in as_completed(futures):
                    path, ok, stats, detailed_params_path, err = fut.result()
                    if ok:
                        print(f'✓ {path} 分析完成，输出目录: {os.path.dirname(detailed_params_path)}')
                    else:
                        print(f'✗ {path} 分析失败: {err}')
            elapsed_time = time.time() - start_time
            print(f'\n并行分析完成！总耗时: {elapsed_time:.2f}秒 ({elapsed_time / 60:.2f}分钟)')
        else:
            gpu_env = os.environ.get('OPENLANE_GPU_ID')
            if gpu_env is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_env)

            start_time = time.time()
            stats, detailed_params_path = analyze_openlane_distribution_vector_method(pkl_path, output_dir=output_dir)
            elapsed_time = time.time() - start_time
            print(f'\n分析完成！总耗时: {elapsed_time:.2f}秒 ({elapsed_time / 60:.2f}分钟)')
            if stats is not None:
                print(f'统计结果已保存到: {output_dir}')
                print(f'详细车道线参数已保存到: {detailed_params_path}')
                print('\n关键发现:')
                print(f'- 左倾车道线比例: {stats["theta"]["left_leaning_ratio"]:.2%}')
                print(f'- 右倾车道线比例: {stats["theta"]["right_leaning_ratio"]:.2%}')
                print(f'- 垂直车道线比例: {stats["theta"]["vertical_ratio"]:.2%}')
                print(f'- 平均角度: {stats["theta"]["mean"]:.3f} (0.5=垂直)')
                print(f'- 角度范围: [{stats["theta"]["min"]:.3f}, {stats["theta"]["max"]:.3f}]')
                print(f'- 起点X范围: [{stats["start_x"]["min"]:.3f}, {stats["start_x"]["max"]:.3f}]')
                print(f'- 起点Y范围: [{stats["start_y"]["min"]:.3f}, {stats["start_y"]["max"]:.3f}]')
                print(f'- 平均长度: {stats["length"]["mean"]:.3f}')
                print('\n下一步: 可以使用 clustering_analyzer.py 对详细参数进行聚类分析')
            else:
                print('分析失败，请检查输入数据')
    except Exception as e:
        print(f'分析过程中出错: {str(e)}')
        import traceback

        traceback.print_exc()
