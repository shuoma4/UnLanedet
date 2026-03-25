from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="resnet34_llanet",
    backbone_type="resnet34",
    backbone_name="resnet34",
    neck_type="GSAFPN",
    enable_category_head=False,
    use_data_driven_priors=False,
    sample_points=72,
    assign_method="CLRerNet",
    enable_global_semantic=True,
    batch_size=24,
    epochs=15,
)

param_config.scm_kernel_size = 9
