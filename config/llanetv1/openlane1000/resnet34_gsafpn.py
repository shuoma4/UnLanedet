from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="resnet34_gsafpn",
    backbone_type="resnet",
    backbone_name="resnet34",
    neck_type="GSAFPN",
    enable_category_head=False,
    use_data_driven_priors=False,
    assign_method="CLRNet",
    enable_temporal_model=False,
    enable_global_semantic=False,
)
