import logging

import numpy as np


class LaneEncoder(object):
    """
    结构化车道线编码器（非归一化版本）

    编码参数含义：
        reg = [start_x, start_y, theta, length, delta_x_0, ..., delta_x_{N-1}]

    - start_x, start_y:
        结构化车道线起点像素坐标（对齐到最近采样高度）
    - theta:
        车道线整体倾斜角(-90~90°), <0为左倾, >0为右倾, 0为垂直
    - length:
        结构化车道线在图像垂直方向上的长度（对齐到最近采样高度）
    - delta_x:
        相对于先验直线的横向残差（像素单位）
        当采样点超出结构化起终点范围时, delta_x 为默认值 -1e5
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
            lane_pts (list of tuple): 车道线点集，每个点为 (x, y) 格式

        Returns:
            reg (np.ndarray): 结构化车道线编码参数，形状为 (n_offsets, 5),
                start_y : 结构化车道线起点y像素坐标（对齐到最近采样高度）
                start_x : 结构化车道线起点x像素坐标（对齐到最近采样高度）
                theta : 车道线整体倾斜角(-90~90°), <0为左倾, >0为右倾, 0为垂直
                length : 结构化车道线在图像垂直方向上的长度（对齐到最近采样高度）
                delta_x : 采样范围内，车道线x坐标相对于先验直线的横向残差（像素单位）
            end_pts (list of float): 结构化车道线终点像素坐标 (y, x)
            xs_sampled (np.ndarray): 结构化车道线在采样高度上的x坐标 (n_offsets,)
            sample_ys (np.ndarray): y轴采样点序列 (n_offsets,)
        """
        xs = np.asarray([p[0] for p in lane_pts], dtype=np.float32)
        ys = np.asarray([p[1] for p in lane_pts], dtype=np.float32)

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

        # -------------------------
        # 3️⃣ 结构化起点 & 终点（对齐到最近采样高度）
        # -------------------------
        idx_start = int(np.argmin(np.abs(sample_ys - ys[0])))
        idx_end = int(np.argmin(np.abs(sample_ys - ys[-1])))

        start_y = sample_ys[idx_start]
        start_x = xs_sampled[idx_start]
        end_x = xs_sampled[idx_end]
        end_y = sample_ys[idx_end]

        # -------------------------
        # 4️⃣ 可见性掩码（严格以结构化起终点为边界）
        # -------------------------
        y_high = start_y
        y_low = end_y

        vis = (sample_ys <= y_high + 1e-3) & (sample_ys >= y_low - 1e-3)
        # -------------------------
        # 5️⃣ 拟合整体方向角 theta
        # -------------------------
        k, b = np.polyfit(ys, xs, 1)
        theta = -np.degrees(np.arctan(k))  # 左负右正（你也可以去掉负号换定义）
        theta = np.clip(theta, -90.0, 90.0)

        # -------------------------
        # 6️⃣ 结构化长度（对齐到采样坐标）
        # -------------------------
        length = float(start_y - end_y)

        # -------------------------
        # 7️⃣ 构造先验直线并计算残差
        # -------------------------
        tan_theta = np.tan(np.deg2rad(theta))
        tan_theta = np.clip(tan_theta, -1e3, 1e3)  # 防止接近水平的车道线产生数值爆炸
        x_prior = start_x + (start_y - sample_ys) * tan_theta

        delta_x = xs_sampled - x_prior  # 固定Y采样点下，X坐标的偏移
        xs_sampled[~vis] = -1e5
        delta_x[~vis] = -1e5
        # -------------------------
        # 8️⃣ 拼接回归向量
        # -------------------------
        reg = np.concatenate(
            [
                np.array([start_y, start_x, theta, length], dtype=np.float32),
                delta_x.astype(np.float32),
            ]
        )

        return reg, [end_y, end_x], xs_sampled, sample_ys
