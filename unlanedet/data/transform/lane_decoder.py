import torch
import torch.nn as nn


class LaneDecoder(nn.Module):
    """
    reg (归一化参数) -> 像素坐标车道线
    只负责几何解码，不包含置信度过滤等任务逻辑
    """

    def __init__(self, cfg):
        super().__init__()
        self.img_w = cfg.img_w
        self.img_h = cfg.img_h
        self.num_points = cfg.num_points

    # ------------------------------------------------
    # 纯几何解码（GPU / CPU 通用）
    # ------------------------------------------------
    def forward(self, reg, sample_ys=None):
        """
        reg: (B, ... , 6 + N)
            0 - background logits
            1 - lane logits
            2 - start_y normalized
            3 - start_x normalized
            4 - theta normalized
            5 - length normalized
            6 - delta_x normalized
        sample_ys:   (N,)
            None        -> lane_adaptive
            Tensor (N,) -> 固定采样高度（像素坐标）

        return:
            xs: (B, M, N)
            ys: (B, M, N)
            valid_mask: (B, M, N)
        """
        device = reg.device

        reg[..., 2] *= self.img_h - 1
        reg[..., 3] *= self.img_w - 1
        reg[..., 4] *= 90
        reg[..., 5] *= self.img_h
        reg[..., 6:] *= self.img_w - 1

        start_y = reg[..., 2]
        start_x = reg[..., 3]
        theta = reg[..., 4]
        length = reg[..., 5]
        delta_x = reg[..., 6:]

        if sample_ys is None:
            t = torch.linspace(0, 1, self.num_points, device=device)
            sample_ys = start_y.unsqueeze(-1) - t.view(1, 1, -1) * length.unsqueeze(-1)
        else:
            sample_ys = sample_ys.to(device).view(1, 1, -1)

        tan_theta = torch.tan(torch.deg2rad(theta))
        tan_theta = torch.clamp(tan_theta, -1e3, 1e3)

        xs = start_x.unsqueeze(-1) + (start_y.unsqueeze(-1) - sample_ys) * tan_theta.unsqueeze(-1) + delta_x
        ys = sample_ys

        valid_mask = (xs >= 0) & (xs < self.img_w) & (ys >= 0) & (ys < self.img_h)

        return xs, ys, valid_mask
