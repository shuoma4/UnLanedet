import importlib
import logging
from types import SimpleNamespace

import torch
import torch.nn as nn

from unlanedet.config import instantiate

from .deploy import convert_deploy, prepare_qat
from .distill import LaneDistillationLoss


LOGGER = logging.getLogger(__name__)

# 将多帧拼成一次 backbone 前向可加速，但峰值显存 ∝ B*T；超过阈值则逐帧前向以防 OOM（B=6,T=3→18）
_MAX_STACKED_SEQUENCE_BT = 18


def _maybe_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class LLANetV1(nn.Module):
    def __init__(self, backbone, neck, head, temporal_model=None, cfg=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.temporal_model = temporal_model
        self.cfg = cfg if cfg is not None else SimpleNamespace()
        self.teacher = None
        self.distiller = None
        self._init_teacher_if_needed()

    def _init_teacher_if_needed(self):
        distill_cfg = _maybe_get(self.cfg, 'distill_cfg', None)
        if not distill_cfg or not _maybe_get(distill_cfg, 'enable', False):
            return
        teacher_cfg_module = _maybe_get(distill_cfg, 'teacher_cfg_module', None)
        if not teacher_cfg_module:
            LOGGER.warning('Distillation is enabled but `teacher_cfg_module` is empty.')
            return
        try:
            module = importlib.import_module(teacher_cfg_module)
            teacher_model_cfg = getattr(module, 'model')
            self.teacher = instantiate(teacher_model_cfg)
            teacher_checkpoint = _maybe_get(distill_cfg, 'teacher_checkpoint', None)
            if teacher_checkpoint:
                state = torch.load(teacher_checkpoint, map_location='cpu')
                state_dict = state.get('model', state)
                missing, unexpected = self.teacher.load_state_dict(state_dict, strict=False)
                LOGGER.info('Teacher checkpoint loaded. Missing=%s Unexpected=%s', len(missing), len(unexpected))
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()
            feature_pairs = _maybe_get(distill_cfg, 'feature_pairs', [])
            self.distiller = LaneDistillationLoss(
                feature_pairs=feature_pairs,
                feature_weight=float(_maybe_get(distill_cfg, 'feature_weight', 1.0)),
                logits_weight=float(_maybe_get(distill_cfg, 'logits_weight', 1.0)),
                temperature=float(_maybe_get(distill_cfg, 'temperature', 4.0)),
            )
        except Exception as exc:
            LOGGER.warning('Failed to initialize teacher model: %s', exc)
            self.teacher = None
            self.distiller = None

    def _move_to_device(self, batch):
        device = next(self.parameters()).device
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _forward_single_image(self, img):
        features = self.backbone(img)
        if self.neck is not None:
            features = self.neck(features)
        return features

    def _forward_sequence_stacked(self, img):
        """将 T 帧拼成 B*T 一次过 backbone+neck，再拆回各帧（避免 T 次重复卷积）。"""
        if img.dim() != 5:
            raise ValueError("expected 5D img (B,T,C,H,W)")
        B, T, C, H, W = img.shape
        x = img.reshape(B * T, C, H, W)
        feat_bt = self.backbone(x)
        if self.neck is not None:
            feat_bt = self.neck(feat_bt)
        if not isinstance(feat_bt, (list, tuple)):
            feat_bt = [feat_bt]
        sequence_features = []
        for t in range(T):
            frame_feats = []
            for f in feat_bt:
                _, Cc, fh, fw = f.shape
                f4 = f.view(B, T, Cc, fh, fw)
                frame_feats.append(f4[:, t].contiguous())
            sequence_features.append(frame_feats)
        return sequence_features

    def extract_features(self, batch):
        if isinstance(batch, dict):
            img = batch['img']
        else:
            img = batch
        if img.dim() == 5:
            B, T = img.shape[0], img.shape[1]
            if B * T <= _MAX_STACKED_SEQUENCE_BT:
                sequence_features = self._forward_sequence_stacked(img)
            else:
                sequence_features = [self._forward_single_image(img[:, t]) for t in range(T)]
            if self.temporal_model is not None:
                aggregated_features, _ = self.temporal_model(sequence_features)
            else:
                aggregated_features = sequence_features[-1]
            return aggregated_features
        return self._forward_single_image(img)

    def forward_features_and_aux(self, batch):
        img = batch['img']
        if img.dim() == 5:
            B, T = img.shape[0], img.shape[1]
            if B * T <= _MAX_STACKED_SEQUENCE_BT:
                sequence_features = self._forward_sequence_stacked(img)
            else:
                sequence_features = [self._forward_single_image(img[:, t]) for t in range(T)]
            if self.temporal_model is not None:
                features, temporal_aux = self.temporal_model(sequence_features)
            else:
                features, temporal_aux = sequence_features[-1], {'temporal_consistency_loss': None}
            temporal_aux['sequence_features'] = sequence_features
            return features, temporal_aux
        return self._forward_single_image(img), {'temporal_consistency_loss': None}

    def get_lanes(self, output):
        return self.head.get_lanes(output)

    def forward(self, batch):
        batch = self._move_to_device(batch)
        features, temporal_aux = self.forward_features_and_aux(batch)

        if self.training:
            outputs = self.head(features, batch=batch)
            losses = self.head.loss(outputs, batch)
            temporal_loss = temporal_aux.get('temporal_consistency_loss')
            if temporal_loss is not None:
                losses['temporal_consistency_loss'] = temporal_loss * float(_maybe_get(self.cfg, 'temporal_loss_weight', 0.5))
            
            # 时序模式A：当 dataloader 提供 5D video 时，计算预测层可见区域的几何稳定性约束
            sequence_features = temporal_aux.get('sequence_features')
            if batch['img'].dim() == 5 and sequence_features is not None and len(sequence_features) >= 2:
                if self.temporal_model is not None and getattr(self.temporal_model, 'temporal_loss', None) is not None:
                    # 仅此时序分支：head 只产出 predictions_lists 供 L_t，不传 predictions_only 则与其它 LLANetV1 训练完全一致
                    with torch.no_grad():
                        prev_outputs = self.head(
                            sequence_features[-2],
                            batch=batch,
                            predictions_only=True,
                        )
                    preds = outputs.get('predictions_lists', [])
                    prev_preds = prev_outputs.get('predictions_lists', [])
                    if len(preds) > 0 and len(prev_preds) > 0:
                        current_preds = preds[-1]
                        previous_preds = prev_preds[-1]
                        real_temporal_loss = self.temporal_model.temporal_loss(
                            current_preds, 
                            previous_preds, 
                            batch, 
                            outputs=outputs,
                            assigner=getattr(self.head, 'assigner', None)
                        )
                        if real_temporal_loss is not None:
                            losses['temporal_consistency_loss'] = real_temporal_loss * float(_maybe_get(self.cfg, 'temporal_loss_weight', 0.5))

            # 兼容模式B：当 dataloader 仅提供 4D batch 时（非 5D video），使用批次内平移前一个样本假装作为"上一帧"对齐。
            if batch['img'].dim() == 4 and self.temporal_model is not None and getattr(self.temporal_model, 'temporal_loss', None) is not None:
                preds = outputs.get('predictions_lists', [])
                if len(preds) > 0:
                    current_preds = preds[-1]
                    # We cannot just roll the tensor over the batch and call small difference
                    # because the coordinates would be slightly different
                    previous_preds = current_preds.roll(shifts=1, dims=0)
                    pseudo_temporal_loss = self.temporal_model.temporal_loss(current_preds, previous_preds, batch)
                    if pseudo_temporal_loss is not None:
                        # use a proper scale since we are comparing totally different frames in the batch in Mode B
                        losses['temporal_consistency_loss'] = pseudo_temporal_loss * float(_maybe_get(self.cfg, 'temporal_loss_weight', 0.5))

            if self.teacher is not None and self.distiller is not None:
                with torch.no_grad():
                    teacher_features = self.teacher.extract_features(batch)
                    teacher_outputs = self.teacher.head(teacher_features)
                distill_losses = self.distiller(features, teacher_features, outputs, teacher_outputs)
                losses.update(distill_losses)
            return losses

        return self.head(features)

    def prepare_for_qat(self, backend='fbgemm'):
        return prepare_qat(self, backend=backend)

    def convert_for_deploy(self):
        return convert_deploy(self)
