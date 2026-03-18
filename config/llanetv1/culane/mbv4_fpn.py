from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
        run_name="mbv4_fpn",
        backbone_type="mobilenetv4",
        backbone_name="mobilenetv4_conv_small",
        neck_type="FPN",
        enable_category_head=False,
        use_data_driven_priors=False,
        assign_method="CLRNet",
        enable_global_semantic=False,
    )
