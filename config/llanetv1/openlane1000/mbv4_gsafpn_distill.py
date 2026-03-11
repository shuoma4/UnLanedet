from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name='mbv4_gsafpn_distill',
    backbone_type='mobilenetv4',
    backbone_name='mobilenetv4_conv_small',
    neck_type='GSAFPN',
    enable_category_head=True,
    use_data_driven_priors=True,
    assign_method='GeometryAware',
    enable_temporal_model=False,
    enable_global_semantic=True,
    distill_cfg=dict(
        enable=True,
        teacher_cfg_module='config.llanetv1.openlane1000.resnet50_fpn_teacher',
        teacher_checkpoint='',
        feature_weight=1.0,
        logits_weight=0.5,
        temperature=4.0,
        feature_pairs=[
            dict(student_idx=0, teacher_idx=0, student_channels=64, teacher_channels=64),
            dict(student_idx=1, teacher_idx=1, student_channels=64, teacher_channels=64),
            dict(student_idx=2, teacher_idx=2, student_channels=64, teacher_channels=64),
        ],
    ),
)
