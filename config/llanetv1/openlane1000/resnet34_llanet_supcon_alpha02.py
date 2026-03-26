#
# Exp5' : full + SupCon，但降低 combined 的线性分支权重 (combined_alpha=0.2)
# 目的：减少线性分支对 prototype 的干扰，同时检验 SupCon 导致 rare 类塌陷的问题是否缓解。
#
from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="resnet34_llanet_supcon_alpha02_bs48",
    backbone_type="resnet34",
    backbone_name="resnet34",
    neck_type="GSAFPN",
    enable_category_head=True,
    category_head_type="combined",
    use_data_driven_priors=False,
    assign_method="CLRNet",
    enable_global_semantic=True,
    batch_size=48,
    epochs=20,
    use_category_weights=True,
    enable_supcon=True,
)

param_config.scm_kernel_size = 9
param_config.combined_alpha = 0.2  # linear 权重；越小越接近 prototype-only

