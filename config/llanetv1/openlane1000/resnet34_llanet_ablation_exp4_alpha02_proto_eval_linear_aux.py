# Exp4'' : combined 头，但分类输出使用 prototype；linear 仅作为训练辅助 CE
#
# 目标：
# 1) 评估/推理阶段：避免 linear 决策边界拖累 Macro 类别 F1
# 2) 训练阶段：保留 linear 分支对共享特征的额外梯度信号，从而不完全丢掉“linear 对回归反馈更好”的潜在收益

from .common import build_config

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="resnet34_llanet_ablation_exp4_alpha02_proto_eval_linear_aux",
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

# combined 内部 logits 的解耦策略：
# - 输出给 eval 的 category logits：只使用 prototype 分支
param_config.combined_proto_as_output = True

# - 训练时额外使用 linear 分支做辅助 CE（以免 linear 分类“只在推理侧被屏蔽”而完全没有梯度）
param_config.combined_enable_linear_aux_loss = True

# linear 辅助 CE 的权重（越小越接近“prototype 为主”的训练目标）
param_config.combined_linear_aux_loss_weight = 0.2

# 仍保留 combined_alpha，便于与现有实验对齐/复现
param_config.combined_alpha = 0.2

