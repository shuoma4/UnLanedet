from ..common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="scm/resnet34_scm_9",
    backbone_type="resnet",
    backbone_name="resnet34",
    neck_type="GSAFPN",
    enable_category_head=False,
    use_data_driven_priors=False,
    assign_method="CLRNet",
    enable_global_semantic=False,
    epochs=15,
    batch_size=48,
)

optimizer.lr = 0.6e-3 * (48 / 24)
dataloader.train.num_workers = 8
dataloader.test.total_batch_size = 8

param_config.scm_kernel_size = 9
