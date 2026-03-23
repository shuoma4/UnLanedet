import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_homography_from_poses(extrinsic_k, extrinsic_t, intrinsic, z=0):
    # 简化：使用内参和外参计算单应矩阵，假设路面 z=0
    # 由于实现暂不包含完整的3D反投影逻辑，我们这里可以用一个基础版：
    H = torch.eye(3, device=extrinsic_k.device)
    return H


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, cfg=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.cfg = cfg

    def forward(self, current_preds, previous_preds, batch=None, outputs=None, assigner=None):
        if current_preds is None:
            return torch.tensor(0.0)

        # Baseline constraint (if no batch info available)
        min_priors = min(current_preds.shape[1], previous_preds.shape[1])
        curr_p = current_preds[:, :min_priors]
        prev_p = previous_preds[:, :min_priors].detach()
        cls_loss = F.mse_loss(F.softmax(curr_p[..., :2], dim=-1), F.softmax(prev_p[..., :2], dim=-1)) * 5.0
        
        geo_loss = curr_p.new_tensor(0.0)
        reg_loss = curr_p.new_tensor(0.0)
        
        # pure anchor smooth loss as fallback
        fg_mask = (F.softmax(prev_p[..., :2], dim=-1)[..., 1] > 0.1).float()
        if fg_mask.sum() > 0:
            reg_diff = F.smooth_l1_loss(curr_p[..., 2:6], prev_p[..., 2:6], reduction='none')
            reg_loss = (reg_diff.mean(dim=-1) * fg_mask).sum() / (fg_mask.sum() + 1e-5) * 5.0

        if batch is not None and outputs is not None and 'final_matching_matrix' in outputs and 'extrinsic' in batch:
            B = current_preds.shape[0]
            matching_matrix = outputs['final_matching_matrix'] # [B, num_priors, max_lanes]
            matched_indices = matching_matrix.nonzero(as_tuple=False)
            
            seq_xyz = batch.get('seq_xyz')
            if hasattr(seq_xyz, 'data'):
                seq_xyz = seq_xyz.data
                if isinstance(seq_xyz, list) and len(seq_xyz) == 1 and isinstance(seq_xyz[0], list): seq_xyz = seq_xyz[0]
            seq_track = batch.get('seq_track_id')
            if hasattr(seq_track, 'data'):
                seq_track = seq_track.data
                if isinstance(seq_track, list) and len(seq_track) == 1 and isinstance(seq_track[0], list): seq_track = seq_track[0]
            seq_vis = batch.get('seq_visibility')
            if hasattr(seq_vis, 'data'):
                seq_vis = seq_vis.data
                if isinstance(seq_vis, list) and len(seq_vis) == 1 and isinstance(seq_vis[0], list): seq_vis = seq_vis[0]
            seq_ext = batch.get('extrinsic')
            seq_int = batch.get('intrinsic', batch.get('intrinsic')) # in TemporalToTensor, it's just 'intrinsic'
            
            cut_height = float(getattr(self.cfg, 'cut_height', 600))
            img_w = float(getattr(self.cfg, 'img_w', 800))
            img_h = float(getattr(self.cfg, 'img_h', 320))
            ori_img_w = float(getattr(self.cfg, 'ori_img_w', 1920))
            ori_img_h = float(getattr(self.cfg, 'ori_img_h', 1280))
            
            n_offsets = int(getattr(self.cfg, 'num_points', 72))
            n_strips = n_offsets - 1
            strip_size = img_h / n_strips
            
            total_l1 = 0.0
            num_valid = 0
            
            for b in range(B):
                try:
                    if seq_ext.dim() == 4:
                        E_t_1 = seq_ext[b, -2].double()
                        E_t = seq_ext[b, -1].double()
                        K_t = seq_int[b, -1].double() if seq_int.dim() == 4 else seq_int[b].double()
                    else:
                        continue
                        
                    T_rel = torch.inverse(E_t) @ E_t_1
                    
                    track_ids_t_1 = seq_track[b][-2]
                    track_ids_t = seq_track[b][-1]
                    xyz_t_1 = seq_xyz[b][-2]
                    vis_t_1 = seq_vis[b][-2]
                    
                    b_mask = (matched_indices[:, 0] == b)
                    if not b_mask.any(): continue
                    
                    for prior_idx, gt_idx in zip(matched_indices[b_mask, 1], matched_indices[b_mask, 2]):
                        g_idx = gt_idx.item()
                        if g_idx >= len(track_ids_t): continue
                        tid = track_ids_t[g_idx]
                        if tid == -1: continue
                        
                        try:
                            prev_idx = track_ids_t_1.index(tid)
                        except ValueError:
                            continue
                            
                        xyz = xyz_t_1[prev_idx]
                        if len(xyz) == 0: continue
                        
                        pts = current_preds.new_tensor(xyz, dtype=torch.float64).transpose(0, 1) # 3, N
                        ones = torch.ones((1, pts.shape[1]), device=pts.device, dtype=torch.float64)
                        pts_homo = torch.cat([pts, ones], dim=0)
                        
                        pts_t_c = T_rel @ pts_homo
                        X = pts_t_c[0, :]
                        Y = pts_t_c[1, :]
                        Z = pts_t_c[2, :]
                        
                        z_mask = Z > 0
                        if not z_mask.any(): continue
                        
                        uv_proj_org = K_t @ torch.stack([X/Z, Y/Z, torch.ones_like(Z)], dim=0)
                        u_org = uv_proj_org[0, :]
                        v_org = uv_proj_org[1, :]
                        
                        u = u_org * (img_w / ori_img_w)
                        v = (v_org - cut_height) * (img_h / (ori_img_h - cut_height))
                        
                        valid = z_mask & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
                        if not valid.any(): continue
                        
                        u_valid = u[valid].float()
                        v_valid = v[valid].float()
                        
                        y_idx = (img_h - v_valid) / strip_size
                        floor_idx = torch.floor(y_idx).long()
                        ceil_idx = torch.ceil(y_idx).long()
                        
                        floor_idx = torch.clamp(floor_idx, 0, n_offsets - 1)
                        ceil_idx = torch.clamp(ceil_idx, 0, n_offsets - 1)
                        
                        alpha = y_idx - floor_idx
                        
                        pred = current_preds[b, prior_idx]
                        lane_xs = pred[6:6+n_offsets] * img_w
                        
                        floor_x = lane_xs[floor_idx]
                        ceil_x = lane_xs[ceil_idx]
                        pred_u = floor_x * (1 - alpha) + ceil_x * alpha
                        
                        vis = current_preds.new_tensor(vis_t_1[prev_idx])
                        vis_sub = vis[valid]
                        
                        l1 = F.l1_loss(pred_u, u_valid, reduction='none')
                        vis_mask = vis_sub > 0.5
                        if vis_mask.any():
                            total_l1 += l1[vis_mask].mean()
                            num_valid += 1
                            
                except Exception as e:
                    # failsafe silent fallback for this batch 
                    import logging
                    logging.getLogger(__name__).warning("Temporal projection error: " + str(e))
                    pass

            if num_valid > 0:
                geo_loss = total_l1 / num_valid * 5.0 # Give it a reasonable scale
                # Reduce weight of anchor fallback if we have real geometry loss
                reg_loss = geo_loss
            
        return (cls_loss + reg_loss) * self.loss_weight


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
