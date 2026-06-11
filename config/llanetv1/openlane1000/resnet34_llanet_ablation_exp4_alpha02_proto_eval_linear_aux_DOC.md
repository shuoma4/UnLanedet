# `resnet34_llanet_ablation_exp4_alpha02_proto_eval_linear_aux`

## 你要求的“linear 不如 prototype，但 linear 对回归梯度可能更有用”的解决思路
现有 `category_head_type="combined"` 会在 logits 级别做线性融合：`alpha * linear + (1-alpha) * prototype`。
这会导致一个问题：**训练/评估时 linear 分支也参与最终决策边界**，但你观察到 linear 的类别能力整体不如 prototype（尤其 Macro F1 更敏感）。

因此本次改进做了“优点结合”的分离：
1. **评估/推理只用 prototype logits**：让类别决策边界尽量贴近 prototype-only 的效果。
2. **训练仍保留 linear 分支作为辅助 CE**：让 linear 分支产生的梯度仍能给共享特征提供额外学习信号，从而尽量不丢掉你认为的“linear 对回归反馈更好”的潜在收益。

## 我对代码做了什么改进
核心改动在：
- `unlanedet/model/llanetv1/head.py`
- `unlanedet/model/llanetv1/temporal_head.py`（保持一致性；即使当前 openlane1000 配置默认不启用 temporal 模式）

### 1) combined 头新增 3 个可配置开关
在 `LLANetV1Head.__init__` 和 `LLANetV1TemporalHead.__init__` 中加入：
- `combined_proto_as_output`  
  - `True`：输出给下游（eval/推理）的 `output['category']` 只使用 prototype 分支 logits  
  - `False`：保持原逻辑（output 使用 `alpha*linear + (1-alpha)*prototype`）
- `combined_enable_linear_aux_loss`  
  - `True`：forward 额外返回 `category_linear_logits` 与 `category_proto_logits`，以便 loss 里分别计算
- `combined_linear_aux_loss_weight`  
  - 用于控制训练时 linear 辅助 CE 的权重（`L_type = L_proto + w * L_linear`）

### 2) forward：允许对 combined logits 进行“输出/训练”解耦
当 `category_head_type == "combined"` 且 `combined_enable_linear_aux_loss == True` 时：
- 仍会计算：
  - `logits_proto`（prototype 分支）
  - `logits_linear`（linear 分支）
  - `logits_mixed`（按原来的 combined_alpha 混合）
- 但：
  - `output['category']` 将由 `combined_proto_as_output` 决定选择 prototype-only 或 mixed
  - 同时把 `category_linear_logits` / `category_proto_logits` 放到 output 中供 loss 使用

### 3) loss：训练时按“prototype 为主 + linear 辅助”的方式计算类别损失
在 `LLANetV1Head.loss()`（以及 temporal_head 的对应位置）里：
- 当 head 为 `combined` 且启用了 `combined_enable_linear_aux_loss` 时：
  - `L_type = CE(prototype_logits, targets) + w * CE(linear_logits, targets)`
- 若未启用线性辅助损失，则完全保持旧逻辑：用 `output['category']` 直接算 CE

SupCon 部分不做额外改动（本次配置默认 `enable_supcon=False`），仍可按你后续实验需求叠加。

## 我新增的训练配置文件做了什么
新增文件：
- `config/llanetv1/openlane1000/resnet34_llanet_ablation_exp4_alpha02_proto_eval_linear_aux.py`

该配置在基础 Exp4'(alpha02) 上设置：
- `category_head_type="combined"`
- `param_config.combined_proto_as_output = True`  
  - 评估/推理使用 prototype logits
- `param_config.combined_enable_linear_aux_loss = True`  
  - 训练时额外用 linear logits 算辅助 CE
- `param_config.combined_linear_aux_loss_weight = 0.2`  
  - linear 辅助梯度强度较小（偏向 prototype 主导）
- `param_config.combined_alpha = 0.2`  
  - 保持与既有实验对齐，便于对比/复现

## 建议你如何比较结果
你可以把新 run 与以下两类实验对照：
1. prototype-only 对照：看 Macro 类别 F1 是否更接近 prototype 的优势
2. combined(old) 对照：看“推理决策边界”是否改善（尤其 Macro）

日志里重点关注：
- `Cat_F1_Macro`
- 稀有类（per_class 中 f1 很低的类）是否出现缓解
- `F1` 总检测指标是否仍保持在之前 combined(alpha02) 的水平附近

