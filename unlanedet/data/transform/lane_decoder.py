import torch
import numpy as np


class LaneDecoder(object):
    """
    将网络输出的结构化车道线参数解码为像素坐标车道线点集
    支持归一化 / 非归一化模式，可配置
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.img_w = cfg.img_w
        self.img_h = cfg.img_h

        # 是否归一化（由外部配置控制）
        self.normalize_start_xy = cfg.normalize_start_xy
        self.normalize_theta = cfg.normalize_theta
        self.normalize_length = cfg.normalize_length
        self.normalize_offset = cfg.normalize_offset

        # 角度定义方式
        # 0~180°，90°表示竖直向上，<90左倾，>90右倾
        self.theta_mode = getattr(cfg, "theta_mode", "deg_0_180")

        # 是否开启置信度过滤
        self.conf_threshold = getattr(cfg, "conf_threshold", 0.0)

    # ------------------------------------------------
    # 单步解码（核心几何解码）
    # ------------------------------------------------
    def decode_reg(self, reg, sample_ys):
        """
        将回归参数解码为 (xs, ys)

        reg: Tensor (B, M, 4 + N)
            [start_x, start_y, theta, length, delta_x...]
        sample_ys: Tensor (N,)  像素坐标（非归一化）

        return:
            xs: (B, M, N)
            ys: (N,)
        """
        device = reg.device
        B, M, D = reg.shape
        N = sample_ys.shape[0]

        start_x = reg[..., 0]
        start_y = reg[..., 1]
        theta = reg[..., 2]
        length = reg[..., 3]
        delta_x = reg[..., 4:]  # (B, M, N)

        # ----------------------------
        # 反归一化
        # ----------------------------
        if self.normalize_start_xy:
            start_x = start_x * self.img_w
            start_y = start_y * self.img_h

        if self.normalize_theta:
            theta = theta * 180.0

        if self.normalize_length:
            length = length * self.img_h

        if self.normalize_offset:
            delta_x = delta_x * self.img_w

        # ----------------------------
        # 几何先验：直线外推
        # ----------------------------
        # theta: 0~180°，90°竖直
        theta_rad = torch.deg2rad(theta)
        tan_theta = torch.tan(theta_rad).clamp(min=1e-6)

        # sample_ys 扩展为 (1, 1, N)
        sample_ys = sample_ys.to(device).view(1, 1, -1)

        # 直线先验
        x_prior = start_x.unsqueeze(-1) + (
            start_y.unsqueeze(-1) - sample_ys
        ) / tan_theta.unsqueeze(-1)

        # 最终预测 x
        xs = x_prior + delta_x
        ys = sample_ys

        return xs, ys

    # ------------------------------------------------
    # 掩码过滤 + 裁剪
    # ------------------------------------------------
    def filter_points(self, xs, ys):
        """
        过滤越界点
        xs: (B, M, N)
        ys: (1, 1, N)
        """
        valid = (xs >= 0) & (xs < self.img_w) & (ys >= 0) & (ys < self.img_h)
        return valid

    # ------------------------------------------------
    # 主入口：批量解码
    # ------------------------------------------------
    def decode(self, reg, sample_ys):
        """
        reg: Tensor (B, MAX_LINES, 4 + N)
        sample_ys: Tensor (N,)   像素坐标（非归一化）

        return:
            lanes_all: List[List[List[(x, y)]]]
        """
        with torch.no_grad():
            xs, ys = self.decode_reg(reg, sample_ys)
            valid_mask = self.filter_points(xs, ys)

            B, M, N = xs.shape
            lanes_all = []

            xs_np = xs.cpu().numpy()
            ys_np = ys.cpu().numpy()[0, 0]
            valid_np = valid_mask.cpu().numpy()

            for b in range(B):
                lanes_b = []
                for m in range(M):
                    mask = valid_np[b, m]
                    if mask.sum() < 2:
                        lanes_b.append([])
                        continue
                    pts = [
                        (float(x), float(y))
                        for x, y in zip(xs_np[b, m][mask], ys_np[mask])
                    ]
                    lanes_b.append(pts)
                lanes_all.append(lanes_b)

        return lanes_all

    # ------------------------------------------------
    # 带置信度过滤的接口（可扩展NMS）
    # ------------------------------------------------
    def get_lanes(self, reg, sample_ys, scores=None):
        """
        reg: (B, MAX_LINES, 4 + N)
        scores: (B, MAX_LINES) 车道线置信度（可选）
        """
        if scores is not None and self.conf_threshold > 0:
            keep = scores >= self.conf_threshold
            reg = reg * keep.unsqueeze(-1)

        return self.decode(reg, sample_ys)
