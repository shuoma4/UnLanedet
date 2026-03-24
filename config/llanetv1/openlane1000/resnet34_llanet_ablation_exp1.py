# Exp1: 逆频率权重×, 原型头×, SupCon×
# 基线：线性分类头 + 均匀权重 CE Loss，无 SupCon
from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="resnet34_llanet_ablation_exp1",
    backbone_type="resnet34",
    backbone_name="resnet34",
    neck_type="GSAFPN",
    enable_category_head=True,
    category_head_type="linear",
    use_data_driven_priors=False,
    assign_method="CLRNet",
    enable_global_semantic=True,
    batch_size=24,
    epochs=20,
    use_category_weights=False,
    enable_supcon=False,
)

param_config.scm_kernel_size = 9
