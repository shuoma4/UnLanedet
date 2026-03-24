# Exp4: 逆频率权重✓, 原型头✓ (combined: prototype + linear), SupCon×
# 在 Exp3 基础上改为 combined 头（原型+线性联合分类），但不加 SupCon
from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="resnet34_llanet_ablation_exp4",
    backbone_type="resnet34",
    backbone_name="resnet34",
    neck_type="GSAFPN",
    enable_category_head=True,
    category_head_type="combined",
    use_data_driven_priors=False,
    assign_method="CLRNet",
    enable_global_semantic=True,
    batch_size=24,
    epochs=20,
    use_category_weights=True,
    enable_supcon=False,
)

param_config.scm_kernel_size = 9
