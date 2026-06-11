# 时序模型（TSM）损失定义修复与重构报告

本文档用于持续记录时序损失相关的 bug 修复、统计结论与后续改造方案。当前版本包含两部分：  
1) 已完成修复（代码层面）  
2) 面向当前训练症状的新版损失重构方案（待落地）

---

## 一、已完成修复（代码已合入）

### 1. 相机坐标系定义与投影公式修复

#### 问题
OpenLane / Waymo 使用相机坐标系：`X=前方(深度)`、`Y=左`、`Z=上`。  
旧版 `_geo_loss` 沿用了不匹配的投影定义，导致 `u/v` 异常放大并引入不稳定梯度。

#### 修复
在 `unlanedet/model/llanetv1/temporal.py` 中改为与 OpenLane 坐标一致的投影：

```python
X_safe = X.clamp(min=1e-6)
u = ((-Y / X_safe) * fx.unsqueeze(1) + cx.unsqueeze(1)).float() * su
v = (((-Z / X_safe) * fy.unsqueeze(1) + cy.unsqueeze(1)) - cut_h).float() * sv
valid = (pt_mask & (X > 0.1) & ...)
```

---

### 2. Ego-Motion 补偿修复

#### 问题
旧实现仅使用 `extrinsic` 计算 `T_rel`，在同段视频内近似恒等，等价于默认车辆静止。

#### 修复
引入并传递 `pose`，采用真实车体运动补偿：

```python
T_rel = torch.linalg.inv(E_t) @ torch.linalg.inv(P_t) @ P_t1 @ E_t1
```

并已在数据流中补齐 `pose` 透传与缓存更新。

---

### 3. 删除错误的 fallback 分类对齐惩罚

#### 问题
旧 fallback 在 prior 级强制跨帧分类一致，会误罚“正常出现/消失”和“prior 漂移”。

#### 修复
移除该 fallback；仅在存在有效匹配时施加一致性监督。

---

### 4. 训练资源稳定性修复（配置侧）

在 `config/llanetv1/openlane1000/tsm/llanet_tsm_v3.py` 中已完成：
- 降低 `batch_size`
- 收缩 `num_workers`
- 降低 `prefetch_factor`

用于避免训练阶段 CPU/内存资源过载导致卡死。

---

## 二、当前症状与诊断（截至本轮分析）

### 1. `temporal_consistency_loss` 长时间固定在 `100`

当前 `TemporalConsistencyLoss` 内存在如下饱和路径：
- `geo` 分支被 `geo.clamp(-20, 20)` 截断
- 再乘以 `10.0`
- 总损失再乘 `loss_weight=0.5`

当几何项持续打满上限时：

```text
20 * 10 * 0.5 = 100
```

因此日志出现“时序损失恒为 100”并非无效，而是“长期饱和”。

### 2. 重投影误差统计的现实含义

基于已生成统计，帧间误差存在明显长尾，且非小量噪声。  
直接逐点 L1 约束 `uv_pred` 对齐 `uv_proj`，会把几何误差与匹配噪声直接回灌给主分支，容易拖累 `xytl_loss / iou_loss` 收敛。

---

## 三、新版时序损失重构方案（建议实施）

目标：让时序损失“只在可信样本上监督”，并且“即使样本含噪也不产生过强梯度”。

### 总体形式

\[
L_{temp} = \lambda_{geo} L_{geo} + \lambda_{cont} L_{cont} + \lambda_{cls} L_{cls}
\]

---

### 1) 几何一致性项：鲁棒回归 + 严格门控

\[
L_{geo}
=\frac{1}{|M|}\sum_{j\in M}\frac{1}{\sum_k w_{jk}+\epsilon}\sum_k w_{jk}\,\rho(r_{jk}),
\quad
r_{jk}=u^{pred,j}_t(k)-u^{proj,j}_{t\leftarrow t-1}(k)
\]

其中 \(\rho\) 采用 pseudo-Huber / Charbonnier，而不是 L1：

\[
\rho(r)=\delta^2\left(\sqrt{1+(r/\delta)^2}-1\right)
\]

建议 `delta=2~3 px(@800w)`。

点权重：

\[
w_{jk}=w^{occ}_{jk}\cdot w^{curve}_{jk}\cdot w^{reproj}_{jk}\cdot w^{depth}_{jk}
\]

- `w_occ`：遮挡/不可见门控，仅高可信点参与  
- `w_curve`：曲率越大，权重越低（弯道误差放大区域降权）  
- `w_reproj`：历史重投影误差过大（如 >3~4px@800w）时降权或剔除  
- `w_depth`：远距离点降权，抑制深度敏感区的像素噪声放大

---

### 2) 连续性检验项（漏检惩罚）

仅对“理论上应连续存在”的轨迹惩罚当前漏检：

\[
L_{cont}=\frac{1}{|S|}\sum_{j\in S}\mathrm{BCE}(p^{exist}_{t,j},1)
\]

集合 \(S\) 的样本条件：
- \(t-1\) 高置信可检测
- 经姿态补偿后在当前帧仍处于有效视野
- 非重遮挡状态

该项只惩罚“该检未检”，不惩罚“合理消失”。

---

### 3) 类别连续性项

在稳定轨迹上做分类一致性：

\[
L_{cls}=\frac{1}{|S|}\sum_{j\in S}
\mathrm{KL}(p^{cls}_{t-1,j}\parallel p^{cls}_{t,j})
\]

仅对稳定匹配样本启用，避免噪声类别监督。

---

## 四、训练期防拖累策略（强建议）

为避免时序项压制主任务，建议加入调度：

1. **Warmup**：前 `2k~5k` iter，`lambda_geo=0` 或极小  
2. **Ramp-up**：`5k~15k` iter 将 `lambda_geo` 线性升到目标值  
3. **占比上限**：限制 `L_temp <= alpha * L_main`（如 `alpha=0.3`）  
4. **梯度裁剪**：对时序分支或整体做 `clip_grad_norm`

---

## 五、可视化与统计改进（已完成）

原 `lane_reprojection_error_curves_800_robust.png` 高频尖峰过密，趋势不可读。  
已在同目录新增更可读图：

- `lane_reprojection_error_trend_quantile_800.png`  
  分箱分位趋势（`p50/p75/p90`）
- `lane_reprojection_error_trend_ewma_800.png`  
  `winsorize(99%) + EWMA` 长趋势图

Notebook 已更新并可复现：  
`output/analysis/time_series/reproj_stats_train_20260327_111007/plot_reprojection_error_curves.ipynb`

---

## 六、下一步落地清单（待执行）

1. 在 `TemporalConsistencyLoss` 中实现 pseudo-Huber 版本 `L_geo`  
2. 加入曲率门控与重投影误差门控  
3. 实装 `L_cont`（连续性漏检惩罚）  
4. 实装 `L_cls`（稳定轨迹类别一致性）  
5. 配置新增 warmup / ramp-up 超参数并启用  
6. 进行短程对照实验（建议 `1k~2k` iter）对比：
   - `xytl_loss`
   - `iou_loss`
   - `temporal_consistency_loss` 是否脱离饱和
   - 验证集关键指标趋势

---

## 七、文档维护约定

后续每次迭代按以下格式追加：
- `修改日期`
- `代码改动摘要`
- `超参数改动`
- `关键曲线变化`
- `结论与下一步`

该文档作为时序损失唯一持续更新记录。

---

## 2026-03-27 审计补充：重投影统计与可视化有效性复核

### A. 当前重投影流程是否符合定义

当前脚本 `tools/analysis/temporal/temporal_reprojection_stats.py` 的核心流程为：
1. 用 `pose + extrinsic` 计算两帧相对变换 `T_rel`
2. 将上一帧 `xyz_{t-1}` 变换到当前相机系并投影到图像平面，得到 `uv_t'`
3. 将 `uv_t'` 与当前帧 `uv_t` 都按固定 `Y` 采样做插值并对齐
4. 统计 `|u_t' - u_t|` 误差

这与目标定义在几何上是一致的（等价于“先按姿态变化重投影，再固定 Y 比较”）。

### B. 为什么会出现非常大的误差长尾

复核发现当前统计口径会放大长尾：
- 存在大量低重叠样本（`num_samples <= 3` 的 lane 约 3.94 万条）
- 当可比较点数极少时，单点误差会直接主导 lane 级 mean/median/p90/max
- lane 指标间高度相关（相关系数约 0.97~0.999），平滑后外观会非常接近

因此：
- “EWMA 四图看起来很像”并不一定是代码错误，而是**指标高度共线 + 强平滑**导致
- “几千像素误差”主要来自少量极端样本，不应直接用于时序监督强约束

### C. 统计口径修正建议（后续统一采用）

为避免把异常样本当成主结论，建议在统计与训练门控中统一采用：
1. 最小重叠采样点阈值：`num_samples >= 10`
2. 最大误差硬裁剪用于可视化：例如 `max_abs_error_px@800 <= 300`
3. 报告同时输出：
   - 全量统计（保留）
   - 可信子集统计（作为主结论）
4. 趋势图优先使用分箱分位图（p50/p75/p90），弱化逐样本索引曲线

### D. 对时序损失设计的直接影响

该审计结果进一步确认：
- **不应**对全部匹配点做硬 L1 强约束
- 应先做可靠性筛选，再施加鲁棒损失（pseudo-Huber / Charbonnier）
- 低质量不可见性标注不适合作为强监督来源，只应作弱门控或屏蔽

结论：时序损失应从“全量刚性回归”切换为“可信样本上的软约束”。
