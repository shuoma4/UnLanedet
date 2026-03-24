from unlanedet.config import LazyCall as L
from unlanedet.model.LLANet.gsa_fpn import GSAFPN
from unlanedet.model.backbone.timm_mobilenetv4 import TimmMobileNetV4Wrapper
from unlanedet.model.llanetv1 import ContMixTemporalAggregator, LLANetV1, LLANetV1Head
from unlanedet.model.module.backbone.resnet import ResNetWrapper
from unlanedet.model.module.neck.fpn import FPN


def _get_resnet_channels(backbone_name):
    if backbone_name in ('resnet18', 'resnet34'):
        return [128, 256, 512]
    return [512, 1024, 2048]


def _get_mobilenet_channels(backbone_name):
    channel_map = {
        'mobilenetv4_conv_small': [64, 96, 960],
        'mobilenetv4_conv_medium': [80, 160, 960],
    }
    return channel_map.get(backbone_name, [80, 160, 960])


def _get_efficientnet_channels(backbone_name):
    # timm features_only=True, out_indices=[2, 3, 4] 对应 stride 8/16/32 的输出通道
    channel_map = {
        'efficientnet_b0': [40, 112, 320],
        'efficientnet_b1': [40, 112, 320],
        'efficientnet_b2': [48, 120, 352],
        'efficientnet_b3': [48, 136, 384],
        'tf_efficientnet_lite0': [32, 96, 320],
        'tf_efficientnet_lite1': [32, 96, 320],
    }
    return channel_map.get(backbone_name, [40, 112, 320])


def create_llanetv1_model(cfg):
    feature_dim = getattr(cfg, 'neck_out_channels', cfg.featuremap_out_channel)
    hidden_dim = getattr(cfg, 'fc_hidden_dim', feature_dim)
    backbone_type = getattr(cfg, 'backbone_type', 'mobilenetv4')
    backbone_name = getattr(cfg, 'backbone_name', 'mobilenetv4_conv_small')

    if backbone_type == 'mobilenetv4':
        backbone = L(TimmMobileNetV4Wrapper)(
            model_name=backbone_name,
            pretrained=getattr(cfg, 'use_pretrained_backbone', True),
            features_only=True,
            out_indices=[2, 3, 4],
        )
        in_channels = _get_mobilenet_channels(backbone_name)
    elif backbone_type == 'efficientnet':
        backbone = L(TimmMobileNetV4Wrapper)(
            model_name=backbone_name,
            pretrained=getattr(cfg, 'use_pretrained_backbone', True),
            features_only=True,
            out_indices=[2, 3, 4],
        )
        in_channels = _get_efficientnet_channels(backbone_name)
    else:
        backbone = L(ResNetWrapper)(
            resnet=backbone_name,
            pretrained=getattr(cfg, 'use_pretrained_backbone', True),
            replace_stride_with_dilation=[False, False, False],
            out_conv=False,
        )
        in_channels = _get_resnet_channels(backbone_name)

    neck_type = getattr(cfg, 'neck_type', 'GSAFPN')
    if neck_type == 'GSAFPN':
        neck = L(GSAFPN)(
            in_channels=in_channels,
            out_channels=feature_dim,
            num_outs=3,
            scm_kernel_size=getattr(cfg, 'scm_kernel_size', 3),
            enable_global_semantic=getattr(cfg, 'enable_global_semantic', True),
        )
    else:
        neck = L(FPN)(in_channels=in_channels, out_channels=feature_dim, num_outs=3, attention=False)

    temporal_model = None
    if getattr(cfg, 'enable_temporal_model', False):
        temporal_model = L(ContMixTemporalAggregator)(
            in_channels=[feature_dim, feature_dim, feature_dim],
            temporal_weight=getattr(cfg, 'temporal_loss_weight', 1.0),
        )

    return L(LLANetV1)(
        backbone=backbone,
        neck=neck,
        head=L(LLANetV1Head)(
            num_priors=cfg.num_priors,
            refine_layers=getattr(cfg, 'refine_layers', 3),
            prior_feat_channels=feature_dim,
            fc_hidden_dim=hidden_dim,
            sample_points=getattr(cfg, 'sample_points', 36),
            cfg=cfg,
        ),
        temporal_model=temporal_model,
        cfg=cfg,
    )
