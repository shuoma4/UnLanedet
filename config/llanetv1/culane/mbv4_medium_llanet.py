from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="mbv4_medium_llanet",
    backbone_type="mobilenetv4",
    backbone_name="mobilenetv4_conv_medium",
    neck_type="GSAFPN",
    enable_category_head=False,
    use_data_driven_priors=False,
    assign_method="CLRerAssign",
    enable_global_semantic=True,
    batch_size=24,
    epochs=15,
)

param_config.scm_kernel_size = 9
