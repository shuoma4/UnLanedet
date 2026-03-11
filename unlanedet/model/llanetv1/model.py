import importlib
import logging
from types import SimpleNamespace

import torch
import torch.nn as nn

from unlanedet.config import instantiate

from .deploy import convert_deploy, prepare_qat
from .distill import LaneDistillationLoss


LOGGER = logging.getLogger(__name__)


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

    def extract_features(self, batch):
        if isinstance(batch, dict):
            img = batch['img']
        else:
            img = batch
        if img.dim() == 5:
            sequence_features = [self._forward_single_image(img[:, t]) for t in range(img.shape[1])]
            if self.temporal_model is not None:
                aggregated_features, _ = self.temporal_model(sequence_features)
            else:
                aggregated_features = sequence_features[-1]
            return aggregated_features
        return self._forward_single_image(img)

    def forward_features_and_aux(self, batch):
        img = batch['img']
        if img.dim() == 5:
            sequence_features = [self._forward_single_image(img[:, t]) for t in range(img.shape[1])]
            if self.temporal_model is not None:
                features, temporal_aux = self.temporal_model(sequence_features)
            else:
                features, temporal_aux = sequence_features[-1], {'temporal_consistency_loss': None}
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
                losses['temporal_consistency_loss'] = temporal_loss * float(_maybe_get(self.cfg, 'temporal_loss_weight', 0.0))

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
