import numpy as np
import logging


class LaneEncoder(object):
    """
    结构化车道线编码器（非归一化版本）

    编码参数含义：
        reg = [start_x, start_y, theta, length, delta_x_0, ..., delta_x_{N-1}]

    - start_x, start_y:
        车道线起点像素坐标（图像坐标系）
    - theta:
        车道线整体倾斜角（0~180°）
        90° 表示竖直向上
        <90° 表示向左倾斜
        >90° 表示向右倾斜
    - length:
        车道线在图像垂直方向上的长度（像素尺度）
    - delta_x:
        相对于先验直线的横向残差（像素单位）
        当采样点超出图像或者原始真值gt标注的范围时，delta_x为默认值-1e5
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
        if self.sample_ys_mode == "equal_interval":
            # 在整张图像高度范围内均匀采样
            self.base_sample_ys = np.linspace(
                self.img_h - 1, 0, self.n_offsets, dtype=np.float32
            )
        elif self.sample_ys_mode == "equal_density":
            # 使用数据统计得到的采样分布
            self.base_sample_ys = np.array(cfg.sample_ys, dtype=np.float32)
        elif self.sample_ys_mode == "lane_adaptive":
            # 运行时根据车道线自身范围生成，均匀生成
            self.base_sample_ys = None
        else:
            raise ValueError(f"Unknown sample_ys_mode: {self.sample_ys_mode}")

    # ============================================================
    # 自适应弧长重采样（按垂直跨度确定采样密度）
    # ============================================================
    def resample_by_arclength_adaptive(self, xs, ys):
        """
        对原始车道线进行等弧长重采样，采样点数根据车道线垂直跨度自适应设置

        目的：
        - 短车道线：避免无意义过密采样
        - 长车道线：避免欠采样导致几何精度不足
        """

        pts = np.stack([xs, ys], axis=1)

        # 按 y 从大到小排序（从车道起点到终点）
        order = np.argsort(pts[:, 1])[::-1]
        pts = pts[order]
        xs = pts[:, 0]
        ys = pts[:, 1]

        # 1️⃣ 根据垂直跨度自适应设置采样点数
        y_min = ys.min()
        y_max = ys.max()
        # num_sample = int(abs(y_max - y_min))
        # num_sample = np.clip(num_sample, 10, 270)  # 下限防止过稀，上限防止过慢
        num_sample = 270

        if len(xs) < 2 or num_sample < 2:
            return xs, ys

        # 2️⃣ 弧长参数化
        diffs = pts[1:] - pts[:-1]
        seg_lens = np.sqrt((diffs**2).sum(axis=1))
        arc_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = arc_len[-1]

        if total_len < 1e-6:
            return xs, ys

        # 3️⃣ 等弧长重采样（几何一致逐段插值）
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
    # 车道线采样函数（映射到 sample_ys）
    # ============================================================
    def sample_lane(self, xs, ys, sample_ys):
        """
        将车道线映射到给定的 sample_ys 上，得到对应的 x 坐标
        """

        if self.sample_lane_mode == "linear_interp":
            # 直接基于原始标注线性插值
            xs_out = np.interp(sample_ys, ys[::-1], xs[::-1])
            vis = (sample_ys <= ys[0]) & (sample_ys >= ys[-1])
            return xs_out, vis

        elif self.sample_lane_mode == "arc_length":
            # 先等弧长重采样，再插值
            xs_dense, ys_dense = self.resample_by_arclength_adaptive(xs, ys)
            xs_out = np.interp(sample_ys, ys_dense[::-1], xs_dense[::-1])
            vis = (sample_ys <= ys_dense[0]) & (sample_ys >= ys_dense[-1])
            return xs_out, vis

        else:
            raise ValueError(f"Unknown sample_lane_mode: {self.sample_lane_mode}")

    # ============================================================
    # 主入口：编码单条车道线
    # ============================================================
    def encode_lane(self, lane_pts):
        """
        输入：
            lane_pts: [(x0, y0), (x1, y1), ...]，已按从下到上排序

        输出：
            reg: [start_x, start_y, theta, length, delta_x...]
            sample_ys: 采样高度
        """

        xs = np.asarray([p[0] for p in lane_pts], dtype=np.float32)
        ys = np.asarray([p[1] for p in lane_pts], dtype=np.float32)

        # -------------------------
        # 1️⃣ 生成采样高度
        # -------------------------
        if self.sample_ys_mode == "lane_adaptive":
            sample_ys = np.linspace(ys[0], ys[-1], self.n_offsets, dtype=np.float32)
        else:
            sample_ys = self.base_sample_ys

        # -------------------------
        # 2️⃣ 采样得到真实车道线 x
        # -------------------------
        xs_sampled, vis = self.sample_lane(xs, ys, sample_ys)

        # -------------------------
        # 3️⃣ 起点参数
        # -------------------------
        start_x = xs[0]
        start_y = ys[0]

        # -------------------------
        # 4️⃣ 拟合整体方向角 theta（0~180°）
        # -------------------------
        k, b = np.polyfit(ys, xs, 1)  # x = k*y + b
        theta = 90.0 - np.degrees(np.arctan(k))
        theta = np.clip(theta, 0.0, 180.0)

        # -------------------------
        # 5️⃣ 车道线垂直长度
        # -------------------------
        length = float(ys[0] - ys[-1])

        # -------------------------
        # 6️⃣ 构造先验直线并计算残差
        # -------------------------
        theta_rad = np.deg2rad(theta)
        x_prior = start_x + (start_y - sample_ys) / (np.tan(theta_rad) + 1e-6)
        delta_x = xs_sampled - x_prior
        delta_x[~vis] = -1e5

        # -------------------------
        # 7️⃣ 拼接回归向量
        # -------------------------
        reg = np.concatenate(
            [
                np.array([start_x, start_y, theta, length], dtype=np.float32),
                delta_x.astype(np.float32),
            ]
        )

        return reg, sample_ys
