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

        cls_loss = cur.new_tensor(0.0)
        reg_loss = cur.new_tensor(0.0)
        used_matched = False

        # --- 1. 匹配上的 positive anchor：避免对全体 prior 做 softmax-MSE（背景占满巨量 anchor，差分被平均到≈0） ---
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
                # logit 级 smooth_l1：对未饱和的正负样本都比 softmax-MSE 敏感
                cls_loss = F.smooth_l1_loss(cur[bu, pu, :2], prv[bu, pu, :2])
                reg_loss = F.smooth_l1_loss(cur[bu, pu, 2:6], prv[bu, pu, 2:6])
                n_off = int(getattr(self.cfg, 'num_points', 72) if self.cfg else 72)
                end = 6 + n_off
                if cur.shape[-1] >= end:
                    # 72 维采样横坐标对总梯度的贡献远大于 2:6；单独降权抑制偶发梯度爆炸
                    reg_loss = reg_loss + 0.2 * F.smooth_l1_loss(
                        cur[bu, pu, 6:end], prv[bu, pu, 6:end]
                    )
                used_matched = True

        # --- 2. 无匹配信息时的 fallback：在「当前或上一帧任一视为前景」的 prior 上约束 ---
        if not used_matched:
            p_cur = F.softmax(cur[..., :2], dim=-1)[..., 1]
            p_prv = F.softmax(prv[..., :2], dim=-1)[..., 1]
            fg = ((p_cur > 0.05) | (p_prv > 0.05)).float()
            if fg.sum() > 0:
                cls_l1 = F.smooth_l1_loss(cur[..., :2], prv[..., :2], reduction='none').mean(2)
                cls_loss = (cls_l1 * fg).sum() / (fg.sum() + 1e-5)
                rd = F.smooth_l1_loss(cur[..., 2:6], prv[..., 2:6], reduction='none').mean(2)
                reg_loss = (rd * fg).sum() / (fg.sum() + 1e-5)

        # --- 3. 几何分支：与 anchor 级 reg 相加，而不是替换（避免 geo==None 时只剩过小的 cls） ---
        if (batch is not None and outputs is not None
                and 'final_matching_matrix' in outputs
                and 'extrinsic' in batch):
            try:
                geo = self._geo_loss(current_preds, batch, outputs, device)
                if geo is not None:
                    reg_loss = reg_loss + geo
            except Exception as exc:
                _LOG.warning("TemporalConsistencyLoss geo error: %s", exc)

        total = (cls_loss + reg_loss) * self.loss_weight
        if not torch.isfinite(total):
            _LOG.warning("TemporalConsistencyLoss: non-finite total, using 0 for this step")
            return current_preds.new_zeros((), device=device, dtype=torch.float32)
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
        if seq_ext.dim() != 4:
            return None

        # 全 batch 一次性计算 T_rel = E_t^{-1} @ E_{t-1}  (Eq 4-7)
        E_t1  = seq_ext[:, -2].double()          # [B, 4, 4]
        E_t   = seq_ext[:, -1].double()          # [B, 4, 4]
        T_rel = torch.linalg.inv(E_t) @ E_t1    # [B, 4, 4]
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

        Z_safe = Z.clamp(min=1e-6)
        fx, fy = K_bn[:, 0, 0], K_bn[:, 1, 1]
        cx, cy = K_bn[:, 0, 2], K_bn[:, 1, 2]
        u = ((X / Z_safe) * fx.unsqueeze(1) + cx.unsqueeze(1)).float() * su
        v = (((Y / Z_safe) * fy.unsqueeze(1) + cy.unsqueeze(1)) - cut_h).float() * sv

        valid = (
            pt_mask
            & (Z > 0)
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
        lane_xs = pred_lanes[:, 6 : 6 + n_off] * img_w

        pred_u_floor = lane_xs.gather(1, fi)
        pred_u_ceil = lane_xs.gather(1, ci)
        pred_u = pred_u_floor * (1 - frac) + pred_u_ceil * frac

        return F.l1_loss(pred_u[valid], u[valid].detach()) * 5.0


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
