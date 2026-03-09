import logging

import numpy as np


class LaneEncoder(object):
    """
    结构化车道线编码器（归一化版本）

    编码参数含义：
        reg =[start_y_norm, start_x_norm, theta_far, length_norm, delta_x_0, ..., delta_x_{N-1}]

    - start_y_norm, start_x_norm:
        结构化车道线起点归一化坐标 (y_norm, x_norm)，取值范围[0,1]
        y_norm = start_y / (img_h - 1)
        x_norm = start_x / (img_w - 1)
    - theta_far:
        归一化后的整体倾斜角，取值范围[-1,1]。
        计算逻辑：计算所有有效采样点相对于起点的角度并取平均值。
        theta_i = arctan(-dx_i/dy_i) / (π/2)
        theta_far = mean(theta_i)
    - length_norm:
        结构化车道线在图像垂直方向上的归一化长度
        length_norm = (start_y - end_y) / (img_h - 1)
    - delta_x:
        相对于先验直线的归一化横向残差。
        当采样点超出结构化起终点范围时，delta_x 为默认值 -1e5。
        有效采样点: delta_x = (xs_sampled - prior_xs) / (img_w - 1)
        注：构建先验直线 prior_xs 时，theta_far 对应的物理角度会被强制截断限制在[-85°, 85°] 之间。
    """

    def __init__(self, cfg):
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.img_w = cfg.img_w
        self.img_h = cfg.img_h
        self.n_offsets = cfg.num_points

        self.sample_ys_mode = cfg.sample_ys_mode
        self.sample_lane_mode = cfg.sample_lane_mode

        # -------------------------
        # 预定义采样高度模式
        # -------------------------
        if self.sample_ys_mode == 'equal_interval':
            self.base_sample_ys = np.linspace(self.img_h - 1, 0, self.n_offsets, dtype=np.float32)
        elif self.sample_ys_mode == 'equal_density':
            self.base_sample_ys = np.array(cfg.sample_ys, dtype=np.float32)
        elif self.sample_ys_mode == 'lane_adaptive':
            self.base_sample_ys = None
        else:
            raise ValueError(f'Unknown sample_ys_mode: {self.sample_ys_mode}')

    # ============================================================
    # 自适应弧长重采样
    # ============================================================
    def resample_by_arclength_adaptive(self, xs, ys):
        pts = np.stack([xs, ys], axis=1)
        order = np.argsort(pts[:, 1])[::-1]
        pts = pts[order]

        if len(pts) < 2:
            return pts[:, 0], pts[:, 1]

        diffs = pts[1:] - pts[:-1]
        seg_lens = np.sqrt((diffs**2).sum(axis=1))
        arc_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = arc_len[-1]

        if total_len < 1e-6:
            return pts[:, 0], pts[:, 1]

        # 注释说明: 此处的 270 是硬编码的内部高密度插值点数，
        # 用于在弧长积分前生成平滑的密集点集，与外部的 cfg.num_points (n_offsets) 无关。
        num_sample = 270
        target_lens = np.linspace(0, total_len, num_sample)

        sampled = []
        for t in target_lens:
            idx = np.searchsorted(arc_len, t) - 1
            idx = np.clip(idx, 0, len(pts) - 2)
            l0, l1 = arc_len[idx], arc_len[idx + 1]
            ratio = (t - l0) / (l1 - l0 + 1e-6)
            p = pts[idx] + ratio * (pts[idx + 1] - pts[idx])
            sampled.append(p)

        sampled = np.array(sampled)
        return sampled[:, 0], sampled[:, 1]

    # ============================================================
    # 车道线采样函数
    # ============================================================
    def sample_lane(self, xs, ys, sample_ys):
        if self.sample_lane_mode == 'linear_interp':
            xs_out = np.interp(sample_ys, ys[::-1], xs[::-1])
            return xs_out

        elif self.sample_lane_mode == 'arc_length':
            xs_dense, ys_dense = self.resample_by_arclength_adaptive(xs, ys)
            xs_out = np.interp(sample_ys, ys_dense[::-1], xs_dense[::-1])
            return xs_out

        else:
            raise ValueError(f'Unknown sample_lane_mode: {self.sample_lane_mode}')

    # ============================================================
    # 主入口：编码单条车道线
    # ============================================================
    def encode(self, lane_pts):
        """
        编码单条车道线

        Args:
            lane_pts (list of tuple): 车道线点集，每个点为 (x, y) 像素坐标

        Returns:
            reg (np.ndarray): 结构化车道线归一化编码参数，形状为 (n_offsets + 4,)
                前4个元素:[start_y_norm, start_x_norm, theta_far, length_norm]
                后续n_offsets个元素: delta_x 归一化残差
            end_pts (list of float): 结构化车道线终点像素坐标 (y, x)
            xs_sampled (np.ndarray): 结构化车道线在采样高度上的x像素坐标 (n_offsets,)
            sample_ys (np.ndarray): y轴采样点像素坐标序列 (n_offsets,)
        """
        xs = np.asarray([p[0] for p in lane_pts], dtype=np.float32)
        ys = np.asarray([p[1] for p in lane_pts], dtype=np.float32)
        order = np.argsort(ys)[::-1]
        xs = xs[order]
        ys = ys[order]
        if len(ys) > 1:
            uniq_mask = np.concatenate(([True], ys[1:] != ys[:-1]))
            xs = xs[uniq_mask]
            ys = ys[uniq_mask]

        # -------------------------
        # 1️⃣ 采样高度
        # -------------------------
        if self.sample_ys_mode == 'lane_adaptive':
            sample_ys = np.linspace(ys[0], ys[-1], self.n_offsets, dtype=np.float32)
        else:
            sample_ys = self.base_sample_ys

        # -------------------------
        # 2️⃣ 采样车道线
        # -------------------------
        xs_sampled = self.sample_lane(xs, ys, sample_ys)

        # 过滤掉超出真实 y 范围的采样点 (防止 np.interp 的常量外推导致竖直线)
        # 允许一定的容差 (例如 1 像素)
        y_min, y_max = ys.min(), ys.max()
        valid_y_mask = (sample_ys >= y_min - 1e-3) & (sample_ys <= y_max + 1e-3)
        xs_sampled[~valid_y_mask] = -1e5

        inside_mask = (xs_sampled >= 0) & (xs_sampled < self.img_w)
        if inside_mask.sum() < 2:
            raise ValueError('Not enough valid points for encoding.')

        valid_indices = np.where(inside_mask)[0]
        start_index = int(valid_indices[0])
        end_index = int(valid_indices[-1])

        start_x = xs_sampled[start_index]
        start_y = sample_ys[start_index]
        end_x = xs_sampled[end_index]
        end_y = sample_ys[end_index]

        xs_inside = xs_sampled[inside_mask]
        ys_inside = sample_ys[inside_mask]

        # 计算 theta_far：所有点相对于起点的角度并取均值
        thetas = []
        for i in range(1, len(xs_inside)):
            dy = ys_inside[i] - ys_inside[0]
            dx = xs_inside[i] - xs_inside[0]
            tan_theta = -dx / (dy + 1e-5)
            theta = np.arctan(tan_theta) / (np.pi / 2.0)
            thetas.append(theta)
        theta_far = float(np.mean(thetas)) if len(thetas) > 0 else 0.0

        # 归一化处理
        start_y_norm = start_y / float(self.img_h - 1)
        start_x_norm = start_x / float(self.img_w - 1)
        length_norm = (start_y - end_y) / float(self.img_h - 1)

        delta_x = np.ones_like(xs_sampled, dtype=np.float32) * -1e5
        theta_deg = theta_far * 90.0

        theta_deg = np.clip(theta_deg, -85.0, 85.0)
        tan_theta = np.tan(np.deg2rad(theta_deg))
        prior_xs = start_x + (start_y - sample_ys) * tan_theta

        # 计算相对残差
        delta_x[inside_mask] = (xs_sampled[inside_mask] - prior_xs[inside_mask]) / float(self.img_w - 1)

        reg = np.concatenate(
            [
                np.array([start_y_norm, start_x_norm, theta_far, length_norm], dtype=np.float32),
                delta_x.astype(np.float32),
            ]
        )

        return reg, [end_y, end_x], xs_sampled, sample_ys
