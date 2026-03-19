# CLRNet vs CLRerNet 统一分配器深度对比分析 (Offline Assignment Analysis)

## 核心差异剖析

通过对 `unlanedet/model/CLRNet/dynamic_assign.py` 和 `unlanedet/model/CLRerNet/assign.py` 的架构及离线验证基准的严格对比，我们发现了 CLRerNet 相比于原版 CLRNet （或 fCLRNet）在 F1 性能上高出约 1.0 个百分点的三大本源原因：

### 1. 放弃 1D Ad-hoc 组合惩罚，回归 2D 拓扑结构 (GIoU)
* **CLRNet 的几何惩罚（存在缺陷）**：
  CLRNet 使用了多个基于坐标点的启发式度量相乘来代表几何相似度：
  `cost = - ((distances_score * start_xys_score * theta_score) ** 2) * w`
  **致命缺陷**：这种设计对局部特征极度敏感。如果一个 Prior 的形状完美贴合 GT（重合度99%），但仅仅由于其起始点（Y轴）比 GT 延伸得短了/宽了2个像素，导致 `start_xys_score` 变得极差。由于连乘机制（AND Gate），整个几何得分会被瞬间清零拉低，直接导致该高质量 Prior 被抛除。这导致大量的优质先验框被误杀。
* **CLRerNet 的几何惩罚（极其优越）**：
  CLRerNet 彻底抛弃了 `start_x`、`theta` 和绝对 `distance` 三个代理指标，直接将其全部替换为 **LaneIoU / GIoU** 作为唯一且统一的结构代价：
  `cost_iou = 1 - (1 - iou_cost) / max_iou ... `
  `cost = -iou_score * reg_weight + cls_score`
  **解决之道**：IoU/GIoU 是从二维拓扑域真正反映两根车道线重合程度的核心指标。一根长线稍微短了一点，其 IoU 仍然极高。这种宽容度完美切合了 CULane 等数据集的真实评估要求。

### 2. 通过 GIoU (Generalized IoU) 解决梯度与区分度消失
* 原版 CLRNet 在计算 `Dynamic-K` 获取 topk 门槛时，使用的是标准 `line_iou`。当线与真实线在空间中完全无交叉时，标准 IoU=0。这意味着所有“偏离 20像素的预测”与“偏离 500像素的预测” 在分配上的度量得分都是一样的 (0)。
* **CLRerNet 使用了 LaneIoUCost(use_giou=True)**。GIoU 引入了最小边界闭包（Union Envelope）的惩罚。这意味着，虽然都不重叠，但他能明显分配更低的 cost 给那个**虽然不相交但离得更近**的先验线。这在线条初始化早期的训练中极大地巩固了优质边界框的富集。

### 3. 先验样本数量 (Dynamic K) 更加克制
我们的离线分析显示（由 `experiments.py` 计算输出）：
* **CLRNet** 倾向于频繁把动态 K 打满，单根 GT 车道往往分配满 **3.5 ~ 4.0** 根 先验线。
* **CLRerNet** 则因为 GIoU 惩罚分明，它的分配数量均指落在了 **2.5 ~ 2.8** 左右。
* **效果**：更克制的正样本分配减少了冲突（Conflict），大幅削减了最后推理阶段去重时残留的 False Positives(多检数)，换来了 Precision 的显著飞跃（这一项往往占了 F1 那 1.0 的绝大头）。

## 改造 Geometry_Aware Assigner (GA) 的行动指南

根据这些底层逻辑，我们需要重构我们现有的 `geometry_aware_assign`：
1. **导入 CLRerNet 的 LaneIoUCost** 作为我们 Dynamic-K 和 Total Cost 的基底判断。
2. 彻底移除现有的加法或乘法型几何代理（`delta_x`, `theta_c`, `start_xy`）。这些属于“强迫症”状的代理指标并不比闭包 IoU 有效。
3. 把动态的 G-IoU 生成的 Cost 进行与 Batch 相关的标准化，然后加上 `focal_cost` 并直接作为匹配指标，从而真正利用上 CLRerNet 中能够提高 F1 的优秀特性。
