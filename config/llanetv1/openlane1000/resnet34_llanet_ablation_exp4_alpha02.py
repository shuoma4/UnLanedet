#
# Exp4' : combined 头，但降低线性分支权重 (combined_alpha=0.2)
# 目的：让 combined 更接近 prototype-only（因为 Exp3 的 Cat_F1_Macro 明显更好）
#
from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="resnet34_llanet_ablation_exp4_alpha02",
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
param_config.combined_alpha = 0.2  # linear 权重；越小越接近 prototype-only

