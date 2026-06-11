import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG = logging.getLogger(__name__)


def _unwrap(x):
    """Unwrap DataLoader-wrapped list-of-lists."""
    if hasattr(x, 'data'):
        x = x.data
    if isinstance(x, list) and len(x) == 1 and isinstance(x[0], list):
        x = x[0]
    return x


class TemporalConsistencyLoss(nn.Module):
    """时序一致性损失 (论文 Eq 4-7 ~ 4-12)

    几何分支：matching_matrix.nonzero 一次取出全 batch 匹配，track 对齐与
    T_rel/K 变换、投影、pred 插值均为张量批量运算；仅构建变长 xyz 填充时有
    对匹配条数 M（≲ B×max_lanes）的短循环。
    """

    def __init__(self, loss_weight=1.0, cfg=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.cfg = cfg
        # Internal step counter for temporal warmup/ramp.
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self, current_preds, previous_preds, batch=None, outputs=None, assigner=None):
        if current_preds is None:
            return torch.tensor(0.0, device='cpu')

        device = current_preds.device
        # AMP(fp16) 下对整段纵向坐标做 smooth_l1 易在反传中炸 Linear(Addmm) 梯度；全程用 fp32 计算
        min_p = min(current_preds.shape[1], previous_preds.shape[1])
        cur = current_preds[:, :min_p].float()
        prv = previous_preds[:, :min_p].detach().float()
        
        # Apply a conservative clamp before anything to prevent infinite/nan propagation
        cur = torch.clamp(cur, min=-10.0, max=10.0)
        prv = torch.clamp(prv, min=-10.0, max=10.0)
        
        cls_loss = cur.new_tensor(0.0)
        reg_loss = cur.new_tensor(0.0)
        used_matched = False
        self._step = self._step + 1

        # --- 1. 匹配上的 positive anchor：保留分类特征的一致性约束，移除静态 2D 坐标约束 ---
        if outputs is not None and 'final_matching_matrix' in outputs:
            mm = outputs['final_matching_matrix']
            if mm.device != cur.device:
                mm = mm.to(cur.device)
            idx = mm.nonzero(as_tuple=False)
            if idx.numel() > 0:
                b_all, p_all = idx[:, 0], idx[:, 1]
                bp = torch.stack([b_all, p_all], dim=1)
                uniq = torch.unique(bp, dim=0)
                bu, pu = uniq[:, 0], uniq[:, 1]
                # Keep a light classification consistency term only on strict matches.
                cls_loss = F.smooth_l1_loss(cur[bu, pu, :2], prv[bu, pu, :2])
                cls_loss = torch.nan_to_num(cls_loss, nan=0.0, posinf=0.0, neginf=0.0)
                used_matched = True

        # --- 2. 无匹配信息时的 fallback：在「当前或上一帧任一视为前景」的 prior 上约束（仅分类） ---
        # REMOVED: 之前这里的 fallback 逻辑会错误地将同一 prior 在帧间的正常车道线出现/消失作为损失进行惩罚，
        # 以及未考虑车辆运动导致车道线在 prior 间转移的情况。现已删除该不合理约束。
        
        # Keep cls branch always light; geometric branch carries the major constraint.
        cls_loss = torch.clamp(cls_loss, max=2.0)

        # --- 3. 几何分支：与 anchor 级 reg 相加，而不是替换（避免 geo==None 时只剩过小的 cls） ---
        if (batch is not None and outputs is not None
                and 'final_matching_matrix' in outputs
                and 'extrinsic' in batch):
            try:
                # 传入归一化前的坐标给 geo_loss，因为 geo_loss 内部有自己的反归一化或像素尺度计算
                # 同时也传入未反归一化的 cur 供其使用（内部实现如果期望归一化的输入，这里需要一致）
                geo = self._geo_loss(current_preds, batch, outputs, device)
                if geo is not None:
                    geo = torch.nan_to_num(geo, nan=0.0, posinf=0.0, neginf=0.0)
                    if torch.isfinite(geo).all() and float(geo.item()) >= 0.0:
                        # Robust add with mild cap; avoid hard saturation to fixed constants.
                        geo_cap = float(getattr(self.cfg, "temporal_geo_cap", 8.0))
                        reg_loss = reg_loss + geo.clamp(min=0.0, max=geo_cap)
                    else:
                        _LOG.warning("TemporalConsistencyLoss: geo loss is non-finite.")
            except Exception as exc:
                _LOG.warning("TemporalConsistencyLoss geo error: %s", exc)

        # Weight schedule: warmup -> linear ramp -> full weight.
        warmup_iters = int(getattr(self.cfg, "temporal_warmup_iters", 2000))
        ramp_iters = int(getattr(self.cfg, "temporal_ramp_iters", 4000))
        step = int(self._step.item())
        if step <= warmup_iters:
            sched = 0.0
        elif step <= warmup_iters + max(ramp_iters, 1):
            sched = float(step - warmup_iters) / float(max(ramp_iters, 1))
        else:
            sched = 1.0

        total = (0.2 * cls_loss + reg_loss) * (self.loss_weight * sched)
        total_cap = float(getattr(self.cfg, "temporal_total_cap", 20.0))
        total = torch.clamp(total, min=0.0, max=total_cap)
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(total).all():
            _LOG.warning("TemporalConsistencyLoss: non-finite total, using 0 for this step")
            # Detached scalar: must not retain graph links to preds (avoids poisoned backward).
            return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=False)
        return total

    # ------------------------------------------------------------------
    # 几何一致性损失（向量化核心）
    # ------------------------------------------------------------------
    def _geo_loss(self, current_preds, batch, outputs, device):
        cfg = self.cfg
        cut_h   = float(getattr(cfg, 'cut_height', 600))
        img_w   = float(getattr(cfg, 'img_w',     800))
        img_h   = float(getattr(cfg, 'img_h',     320))
        ori_w   = float(getattr(cfg, 'ori_img_w', 1920))
        ori_h   = float(getattr(cfg, 'ori_img_h', 1280))
        n_off   = int(getattr(cfg,   'num_points', 72))
        strip   = img_h / (n_off - 1)
        su      = img_w / ori_w
        sv      = img_h / (ori_h - cut_h)

        seq_ext = batch['extrinsic']    # [B, T, 4, 4]
        seq_int = batch['intrinsic']    # [B, T, 3, 3] or [B, 3, 3]
        seq_pose = batch.get('pose')    # [B, T, 4, 4]
        if seq_ext.dim() != 4 or seq_pose is None or seq_pose.dim() != 4:
            return None

        # 全 batch 一次性计算 T_rel = E_t^{-1} @ pose_t^{-1} @ pose_{t-1} @ E_{t-1}
        E_t1  = seq_ext[:, -2].double()          # [B, 4, 4]
        E_t   = seq_ext[:, -1].double()          # [B, 4, 4]
        P_t1  = seq_pose[:, -2].double()         # [B, 4, 4]
        P_t   = seq_pose[:, -1].double()         # [B, 4, 4]
        
        # ego_t1 to ego_t: inv(P_t) @ P_t1
        # cam_t1 to cam_t: inv(E_t) @ inv(P_t) @ P_t1 @ E_t1
        T_rel = torch.linalg.inv(E_t) @ torch.linalg.inv(P_t) @ P_t1 @ E_t1    # [B, 4, 4]
        K_all = (seq_int[:, -1] if seq_int.dim() == 4 else seq_int).double()  # [B, 3, 3]

        matching_matrix = outputs['final_matching_matrix']  # [B, num_priors, max_lanes]
        seq_xyz   = _unwrap(batch.get('seq_xyz'))
        seq_track = _unwrap(batch.get('seq_track_id'))
        seq_vis   = _unwrap(batch.get('seq_visibility'))

        B = current_preds.shape[0]
        max_lanes = int(getattr(cfg, 'max_lanes', 12))

        track_t_all = torch.full((B, max_lanes), -1, dtype=torch.long, device=device)
        track_t1_all = torch.full((B, max_lanes), -1, dtype=torch.long, device=device)
        for bb in range(B):
            try:
                tt = seq_track[bb][-1]
                t1 = seq_track[bb][-2]
            except (IndexError, TypeError, KeyError):
                continue
            if not tt or not t1:
                continue
            n = min(len(tt), max_lanes)
            if n:
                track_t_all[bb, :n] = torch.as_tensor(tt[:n], device=device, dtype=torch.long)
            n = min(len(t1), max_lanes)
            if n:
                track_t1_all[bb, :n] = torch.as_tensor(t1[:n], device=device, dtype=torch.long)

        idx = matching_matrix.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return None
        b, prior_idx, gt_idx = idx[:, 0], idx[:, 1], idx[:, 2]

        ok = gt_idx < max_lanes
        b, prior_idx, gt_idx = b[ok], prior_idx[ok], gt_idx[ok]
        if b.numel() == 0:
            return None

        cur_tid = track_t_all[b, gt_idx]
        ok = cur_tid >= 0
        b, prior_idx, gt_idx, cur_tid = b[ok], prior_idx[ok], gt_idx[ok], cur_tid[ok]
        if b.numel() == 0:
            return None

        rows = track_t1_all[b]
        eq = rows == cur_tid.unsqueeze(1)
        has = eq.any(dim=1)
        if not has.any():
            return None
        b = b[has]
        prior_idx = prior_idx[has]
        prev_lane_idx = eq[has].long().argmax(dim=1)
        M = prior_idx.shape[0]
        if M == 0:
            return None

        n_pts_list = []
        for i in range(M):
            bi = int(b[i].item())
            pi = int(prev_lane_idx[i].item())
            try:
                xyz_row = seq_xyz[bi][-2][pi]
                n_pts_list.append(len(xyz_row))
            except (IndexError, TypeError, KeyError):
                n_pts_list.append(0)
        max_N = max(n_pts_list) if n_pts_list else 0
        if max_N == 0:
            return None

        pts_pad = torch.zeros(M, 4, max_N, dtype=torch.float64, device=device)
        vis_pad = torch.zeros(M, max_N, dtype=torch.float32, device=device)
        pt_mask = torch.zeros(M, max_N, dtype=torch.bool, device=device)

        for i in range(M):
            n = n_pts_list[i]
            if n == 0:
                continue
            bi = int(b[i].item())
            pi = int(prev_lane_idx[i].item())
            try:
                xyz_t1 = seq_xyz[bi][-2][pi]
                vis_t1 = seq_vis[bi][-2][pi]
            except (IndexError, TypeError, KeyError):
                continue
            pts_pad[i, :3, :n] = torch.as_tensor(xyz_t1, dtype=torch.float64, device=device).t()
            pts_pad[i, 3, :n] = 1.0
            vis_pad[i, :n] = torch.as_tensor(vis_t1, dtype=torch.float32, device=device)
            pt_mask[i, :n] = True

        T_bn = T_rel[b].double()
        K_bn = K_all[b].double()

        pts_t = torch.einsum('nij,njk->nik', T_bn, pts_pad)
        X, Y, Z = pts_t[:, 0], pts_t[:, 1], pts_t[:, 2]

        # OpenLane/Waymo camera coords: X=forward, Y=left, Z=up.
        # Depth is X. So points must be in front of the camera (X > 0)
        X_safe = X.clamp(min=1e-6)
        fx, fy = K_bn[:, 0, 0], K_bn[:, 1, 1]
        cx, cy = K_bn[:, 0, 2], K_bn[:, 1, 2]
        # projection mapping to image plane: u = -Y/X*fx + cx, v = -Z/X*fy + cy
        u = ((-Y / X_safe) * fx.unsqueeze(1) + cx.unsqueeze(1)).float() * su
        v = (((-Z / X_safe) * fy.unsqueeze(1) + cy.unsqueeze(1)) - cut_h).float() * sv

        valid = (
            pt_mask
            & (X > 0.1)  # Filter points behind camera
            & (u >= 0) & (u < img_w)
            & (v >= 0) & (v < img_h)
            & (vis_pad > 0.5)
        )
        if not valid.any():
            return None

        y_idx = ((img_h - v) / strip).clamp(0, n_off - 1)
        fi = y_idx.long().clamp(0, n_off - 1)
        ci = (y_idx + 1).long().clamp(0, n_off - 1)
        frac = (y_idx - fi.float()).clamp(0, 1)

        pred_lanes = current_preds[b, prior_idx].float()
        # clamp is necessary here to prevent exploding gradients when predictions are wild
        lane_xs = torch.clamp(pred_lanes[:, 6 : 6 + n_off] * img_w, -100.0, img_w + 100.0)

        pred_u_floor = lane_xs.gather(1, fi)
        pred_u_ceil = lane_xs.gather(1, ci)
        pred_u = pred_u_floor * (1 - frac) + pred_u_ceil * frac

        # Reliability gate: require enough overlap points per lane.
        min_pts = int(getattr(cfg, "temporal_min_valid_points", 8))
        valid_cnt = valid.sum(dim=1)
        lane_ok = valid_cnt >= min_pts
        if not lane_ok.any():
            return None

        # Robust point residual: pseudo-Huber + per-point confidence gate.
        delta = float(getattr(cfg, "temporal_huber_delta", 3.0))
        tau = float(getattr(cfg, "temporal_reproj_tau", 8.0))
        max_point_err = float(getattr(cfg, "temporal_max_point_error", 150.0))

        err = (pred_u - u.detach()).abs()
        huber = (delta * delta) * (torch.sqrt(1.0 + (err / delta) ** 2) - 1.0)
        w_reproj = torch.exp(-(err.detach() / max(tau, 1e-6)))

        point_mask = valid & lane_ok.unsqueeze(1) & (err.detach() <= max_point_err)
        if not point_mask.any():
            return None

        weighted = huber * w_reproj
        num = weighted[point_mask].sum()
        den = w_reproj[point_mask].sum().clamp_min(1e-6)
        geo = num / den
        geo_scale = float(getattr(cfg, "temporal_geo_scale", 1.0))
        out = geo * geo_scale
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class ContMixTemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Overview-Net：DilatedRepConv × 2 → GAP → Conv1×1 → G
        self.overview_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
        )
        
        # Focus-Net：动态卷积核生成
        self.fc1 = nn.Linear(channels * 2, 512)
        self.fc2 = nn.Linear(512, channels * kernel_size * kernel_size)
        
        # 自适应融合权重
        self.alpha_proj = nn.Sequential(
            nn.Conv2d(channels * 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, current_feat, prev_feat=None, guidance=None):
        if prev_feat is None:
            return current_feat
        if prev_feat.shape[-2:] != current_feat.shape[-2:]:
            prev_feat = F.interpolate(prev_feat, size=current_feat.shape[-2:], mode='bilinear', align_corners=False)

        B, C, H, W = current_feat.shape
        mix_input = torch.cat([current_feat, prev_feat], dim=1)
        
        # 1. Overview-Net
        G = self.overview_net(mix_input)  # B, C, 1, 1
        
        # 2. Focus-Net
        gap_Ft = F.adaptive_avg_pool2d(current_feat, 1)  # B, C, 1, 1
        concat_feats = torch.cat([G.view(B, C), gap_Ft.view(B, C)], dim=1)  # B, 2C
        
        x_fc1 = F.silu(self.fc1(concat_feats))
        W_dyn = self.fc2(x_fc1)  # B, C*K*K
        
        W_dyn = W_dyn.view(B * C, 1, self.kernel_size, self.kernel_size)
        x = current_feat.view(1, B * C, H, W)
        pad = self.kernel_size // 2
        out = F.conv2d(x, W_dyn, padding=pad, groups=B * C)
        out = out.view(B, C, H, W)
        
        # 3. 自适应融合
        alpha = self.alpha_proj(mix_input)
        alpha = torch.clamp(alpha, min=0.3, max=0.7)
        
        fused = alpha * out + (1.0 - alpha) * prev_feat
        
        return current_feat + fused


class ContMixTemporalAggregator(nn.Module):
    def __init__(self, in_channels, temporal_weight=1.0):
        super().__init__()
        self.blocks = nn.ModuleList([ContMixTemporalBlock(c) for c in in_channels])
        self.temporal_loss = TemporalConsistencyLoss(loss_weight=temporal_weight)

    def forward(self, sequence_features, stage_predictions=None):
        if not sequence_features:
            return None, {}
        if len(sequence_features) == 1:
            aux = {'temporal_consistency_loss': None}
            return sequence_features[0], aux

        aggregated = sequence_features[0]
        for t in range(1, len(sequence_features)):
            current = sequence_features[t]
            guidance = aggregated[-1]
            aggregated = [
                block(curr_feat, prev_feat, guidance if idx == len(current) - 1 else None)
                for idx, (block, curr_feat, prev_feat) in enumerate(zip(self.blocks, current, aggregated))
            ]

        temporal_loss = None
        return aggregated, {'temporal_consistency_loss': temporal_loss}
