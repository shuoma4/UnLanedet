from ..common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name='category/clrnet_prototype_resnet34',
    backbone_type='resnet',
    backbone_name='resnet34',
    neck_type='FPN',
    enable_category_head=True,
    category_head_type='prototype',
    use_data_driven_priors=False,
    assign_method='CLRNet',
    enable_temporal_model=False,
    enable_global_semantic=False,
)
param_config.category_loss_weight = 5.0
