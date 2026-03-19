# 样本分配策略验证工具 (Assigner Analysis Tool)

本文档说明了如何使用本目录下的工具对比、分析、和验证不同车道线样本分配(Assign)策略（如默认 `CLRNet` 与 自研 `GeometryAware`）的代码表现和分配差异。

## 1. 核心工具介绍

**工具脚本:** `compare_assigners.py`
**其目的是:** **实现免训练的秒级正负样本分配评估**。
由于直接启动训练动辄数小时才能看到分配策略改变导致的 F1 指标变化，该脚本通过手动截取一个训练 Batch 的数据，走通一次前向传播（Forward），并将同一个特征层的网络预测（Predictions）**同时送入 `CLRNetAssginer` 和 `GeometryAware Assigner` 中进行样本匹配。**

它将分别打印出两种分配策略为每张图片中的每一条真实车道线（GT）究竟分配了多少个先验框（Priors/Positive Samples）。

## 2. 工具使用方法

当您对 `unlanedet/model/llanetv1/assigner.py` 内部的 `geometry_aware_assign` 或是成本矩阵（Cost Matrix）函数做出了任何修改并希望验证时，请执行以下命令：

```bash
# 1. 激活虚拟环境
source /home/lixiyang/anaconda3/bin/activate unlanedet

# 2. 指定 PYTHONPATH 和任意闲置的单卡 GPU 执行分析脚本
PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 python tools/analysis/assign/compare_assigners.py
```

*(注意: 首次运行需等待数秒左右加载验证集/训练集标注。)*

## 3. 输出看图指南与诊断

成功运行后会得到阶段性的详细匹配数据打印，例如：

```text
=====================================
Assigner Comparison Output:
=====================================

[Stage 0] Refining Layer:
  --> Image 0: (4 Ground Truth Lanes)
      [CLRNet] Positive Priors per GT:  [1.0, 1.0, 1.0, 1.0]  (Total: 4)
      [GA Ass] Positive Priors per GT:  [1.0, 4.0, 3.0, 2.0]  (Total: 10)
```

### 关键输出含义:
* **Image X: (N Ground Truth Lanes)**: 代表第 X 张抽样图片本身有 N 条真值（GT）车道线标签。
* **Positive Priors per GT**: 列表长度对应上述的 GT 数量。例如 `[1.0, 4.0, ...]` 意味着：对应该图片的第一条 GT，分配器给了 1 个正样本 Prior；对于第二条 GT，分配器给了 4 个正样本 Priors。

### 本地掉点原因分析 (诊断与改进逻辑):
* **基线状态 (CLRNet):** 默认展现了非常严格的二分图 1对1 匹配（列表始终为 `[1.0, 1.0...]`）。
* **自研状态 (GeometryAware 初始状态):** 呈现出 `1对多 (One-to-many)` 的匹配模式。这正是 GSA-FPN 在不改网络架构前提下相比基线掉点近 1.9% F1 分数的原因。多对一的匹配在缺乏特定 Loss 正则或 NMS (非极大值抑制) 的情况下，会导致多个具备不同特征与角度的 Priors 去向同一条 GT 回归，引起梯度的“互相拉扯”，破坏了收敛稳定性。
* **下一步修改代码及优化目标:**  
  请修改 `assigner.py` 中的距离/角度惩罚阈值、Cost 结构，或者是最后计算 `matching_matrix` 时增加类似于 Top-K/二分图硬性过滤的逻辑。
  **最终目标**: 让此脚本打印出的 `[GA Ass]` 列表中各 GT 的正样本数收敛至类似 `[1.0, 1.0, 1.0]` 的形式，并保证这一由几何约束选出的“那个 $1$”优于 CLRNet 原本基于分类选出的那个最佳参考点。

---
*注：如果在后续实验中 `compare_assigners.py` 需要支持新的自定义模型或 Dataloader 配置路径，请进入该脚本直接更新文件头部的 config 依赖导入即可。*
