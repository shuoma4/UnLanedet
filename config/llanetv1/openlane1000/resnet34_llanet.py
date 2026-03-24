from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="resnet34_llanet",
    backbone_type="resnet34",
    backbone_name="resnet34",
    neck_type="GSAFPN",
    enable_category_head=True,
    category_head_type="combined",
    use_data_driven_priors=False,
    assign_method="GeometryAware",
    enable_global_semantic=True,
    batch_size=48,
    epochs=20,
    use_category_weights=True,
    enable_supcon=True,
)

param_config.scm_kernel_size = 9
