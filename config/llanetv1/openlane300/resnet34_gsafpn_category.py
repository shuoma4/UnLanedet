from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name='resnet34_gsafpn_category',
    backbone_type='resnet',
    backbone_name='resnet34',
    neck_type='GSAFPN',
    enable_category_head=True,
    use_data_driven_priors=True,
    assign_method='GeometryAware',
    enable_temporal_model=False,
    enable_global_semantic=True,
)
