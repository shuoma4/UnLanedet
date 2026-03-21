from ..common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="scm/resnet34_scm_13",
    backbone_type="resnet",
    backbone_name="resnet34",
    neck_type="GSAFPN",
    enable_category_head=False,
    use_data_driven_priors=False,
    assign_method="CLRNet",
    enable_global_semantic=False,
    epochs=5,
    batch_size=12,
)

param_config.scm_kernel_size = 13
