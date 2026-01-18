# LLANet OpenLane数据集训练指南

## 准备工作

1. 导入opencv动态库路径（自行选择），lane2d评估，调用的程序需要加载opencv动态库，需要设置环境变量
···bash
 export LD_LIBRARY_PATH=/home/lixiyang/anaconda3/envs/dataset-manger/lib:$LD_LIBRARY_PATH
···

## 杀死中断的训练进程

```bash
ps -ef | grep train  # 查看训练进程
kill -9 <PID>        # 强制终止训练进程
ps -ef | grep train | grep -v grep | awk '{print $2}' | xargs kill -9 # 直接杀死训练进程
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9  # 杀死所有python进程
```

## 监控程序用法

在tmux中运行守护监控程序
```bash
python tools/safety_daemon.py
# 或者使用nohup
nohup python tools/safety_daemon.py &
```

停止守护监控程序
```bash
kill $(pgrep -f safety_daemon.py)
```

你可以通过命令行参数调整阈值：

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--pid` | None | 手动指定要监控的进程 ID，指定后忽略 keyword。 |
| `--keyword` | `tools/train_net.py` | 自动查找时匹配命令行的关键字。 |
| `--my-mem-limit` | `55.0` | **你的内存配额 (GB)**。只有当你超过此值且系统内存不足时才杀。 |
| `--my-cpu-limit` | `2500.0` | **你的 CPU 配额 (%)**。`2500` 代表占用 25 个物理核心。 |
| `--sys-min-free-mem`| `20.0` | **系统危险线**：系统可用内存 (Free/Available) 低于此值 (GB) 时视为危险。 |
| `--sys-max-cpu` | `96.0` | **系统危险线**：系统总 CPU 占用率超过此值 (%) 时视为危险。 |
| `--patience` | `5` | **耐心值**。连续多少次检测到危险才执行查杀。 |
| `--interval` | `3` | **检测间隔 (秒)**。 |
| `--log-dir` | `./montor_logs/` | **日志目录**。 |
| `--log-interval-steps` | `1` | **日志记录间隔 (步)**。 |

## ⚠️ 重要：多GPU训练环境变量设置

**必须在使用多GPU训练前设置CUDA_VISIBLE_DEVICES环境变量！**

```bash
# 使用GPU 1和2进行训练（必须是2的倍数个GPU）
export CUDA_VISIBLE_DEVICES=1,2

# 然后使用训练命令
python tools/train_net.py \
    --config-file configs/llanet/mobilenetv4_openlane.py
```

**GPU数量要求：**
- 一般设备数目要是**2的倍数**（如：1, 2, 4, 8张GPU）
- 单GPU训练：不需要设置或设为 `export CUDA_VISIBLE_DEVICES=0`
- 多GPU训练：必须明确指定，且数量为2的倍数

## 概述
本文档提供了在OpenLane数据集上训练、评估和管理LLANet模型的完整说明，支持车道属性预测（车道类别和左右位置属性）。

LLANet采用基于原型的类别预测方法，使用余弦相似度进行车道类别分类，相比传统的线性分类器具有更好的泛化能力。

## 主要功能
- **车道检测**：预测车道几何形状和存在性
- **车道类别预测**：15个类别（13种车道类型 + 2种路缘类型）- **基于原型**
- **左右位置属性预测**：4个位置属性（最左、左、右、最右）
- **边界线支持**：正确处理路缘类别（原始类别20、21）
- **GSA-FPN**：使用非对称卷积和全局语义注入的特征金字塔网络
- **MobileNetV4-Small**：轻量级骨干网络，适合移动端和边缘设备

## 文件结构
```
config/llanet/
├── mobilenetv4_openlane.py      # 主训练配置文件（MobileNetV4-Small骨干）
├── README_CN.md                  # 本文件（中文版）
└── README.md                     # 英文版文档
```

## 快速开始

### 1. 环境准备
确保已安装所需依赖：
```bash
conda activate unlanedet
cd /data1/lxy_log/workspace/ms/UnLanedet
```

### 2. 单GPU训练
```bash
# 使用第0号GPU
export CUDA_VISIBLE_DEVICES=0

python tools/train_net.py \
    --config-file configs/llanet/mobilenetv4_openlane.py
```

### 3. 多GPU训练（推荐）
```bash
# 使用第1、2号GPU（必须是2的倍数）
export CUDA_VISIBLE_DEVICES=1,2

python tools/train_net.py \
    --config-file configs/llanet/mobilenetv4_openlane.py
```

**训练配置：**
- 训练轮数：15 epochs
- 批次大小：120（4张GPU × 每张30个样本）
- 学习率：0.0006
- 图像尺寸：800×320
- 裁剪高度：270像素
- 最大车道数：24
- 每车道点数：72
- 骨干网络：MobileNetV4-Small
- 特征金字塔：GSA-FPN (3层，64通道)

### 4. 从检查点恢复训练
如果训练中断，可以从上次的检查点继续：

```bash
# 设置GPU（重要！）
export CUDA_VISIBLE_DEVICES=1,2

# 自动从最佳检查点恢复
python tools/train_net.py \
    --config-file configs/llanet/mobilenetv4_openlane.py \
    --resume
```

或指定特定检查点：
```bash
export CUDA_VISIBLE_DEVICES=1,2

python tools/train_net.py \
    --config-file configs/llanet/mobilenetv4_openlane.py \
    --opts train.init_checkpoint=/path/to/checkpoint.pth
```

### 5. 评估训练好的模型
在验证集上评估模型性能：
```bash
export CUDA_VISIBLE_DEVICES=0  # 评估可以用单GPU

python tools/train_net.py \
    --config-file configs/llanet/mobilenetv4_openlane.py \
    --eval-only \
    --opts train.init_checkpoint=/path/to/model_best.pth
```

### 6. 单张图片推理
对单张图片进行检测：
```bash
export CUDA_VISIBLE_DEVICES=0

python tools/detect.py \
    --config-file configs/llanet/mobilenetv4_openlane.py \
    --input /path/to/image.jpg \
    --output /path/to/output/directory \
    --ckpt /path/to/model_best.pth
```

### 7. 批量推理
处理多张图片：
```bash
export CUDA_VISIBLE_DEVICES=0

python tools/detect.py \
    --config-file configs/llanet/mobilenetv4_openlane.py \
    --input /path/to/image/folder/*.jpg \
    --output /path/to/output/directory \
    --ckpt /path/to/model_best.pth
```

### 8. 速度测试
测试模型推理速度：
```bash
export CUDA_VISIBLE_DEVICES=0

python tools/test_speed.py \
    configs/llanet/mobilenetv4_openlane.py \
    --ckpt /path/to/model_best.pth
```

## 📁 修改输出目录（重要！）

### 问题：如何改变训练相关文件的保存地址？
**当前位置**：所有文件保存在 `output/` 目录下
**目标位置**：希望保存到 `output/llanet/mobilenetv4_small_gsafpn_openlane/` 目录

### 解决方案：修改配置中的 `train.output_dir`

编辑 `config/llanet/mobilenetv4_openlane.py`，在训练配置部分修改 `output_dir` 设置：

```python
# Training configuration
train = get_config("config/common/train.py").train

# === 修改输出目录 ===
train.output_dir = "./output/llanet/mobilenetv4_custom"  # 自定义输出目录
# ====================

epochs = 15
batch_size = 4 * 30
epoch_per_iter = 200000 // batch_size + 1
total_iter = epoch_per_iter * epochs
train.max_iter = total_iter
train.checkpointer.period = 100
train.eval_period = 500
```

### 动态设置输出目录（推荐）
为了更好地组织不同骨干网络的模型，建议使用动态命名：

```python
# Training configuration
train = get_config("config/common/train.py").train

# === 动态设置输出目录（根据骨干网络自动命名）===
backbone_name = "mobilenetv4_small"  # 可以根据实际配置修改
train.output_dir = f"./output/llanet/{backbone_name}_gsafpn_openlane"
print(f"训练输出目录: {train.output_dir}")  # 可选：打印确认
# ===============================================

epochs = 15
batch_size = 4 * 30
etc...
```

## 训练详情

### 模型架构
- **主干网络**：MobileNetV4-Small（轻量级，适合移动端）
- **特征金字塔**：GSA-FPN（Geometry-Semantic Alignment FPN），3层，每层64通道
  - 非对称卷积（SCM模块）
  - 全局语义注入
  - 支持跨层特征融合
- **检测头**：LLANetHead，3个精炼层
- **属性预测**：
  - 类别（15类）：基于原型的余弦相似度分类
  - 左右位置（4类）

### 基于原型的类别预测
LLANet使用一种新颖的类别预测方法：
- **可学习原型**：每个车道类别有一个可学习的原型向量（形状：15×64）
- **特征变换**：通过MLP将ROI特征映射到特征空间
- **余弦相似度**：计算特征向量与每个原型的余弦相似度
- **温度缩放**：使用可学习的温度系数缩放相似度
- **优势**：相比线性分类器，原型方法具有更好的泛化能力和可解释性

### 损失函数
- **分类损失**：Focal Loss（权重：2.0）
- **回归损失**：Smooth L1 + LIoU Loss（权重：0.2 + 2.0）
- **分割损失**：交叉熵（权重：1.0）
- **类别损失**：NLL Loss（权重：1.0）- **基于原型**
- **属性损失**：NLL Loss（权重：0.5）

### 数据增强
训练包含以下增强：
- 随机水平翻转（100%概率）
- 通道重排（10%概率）
- 亮度调整（60%概率）
- 色调/饱和度调整（70%概率）
- 运动/中值模糊（20%概率）
- 仿射变换（70%概率）

### 输出目录结构（默认）
```
output/
└── llanet/mobilenetv4_small_gsafpn_openlane/
    ├── config.yaml              # 训练配置
    ├── metrics.json             # 训练指标
    ├── model_best.pth           # 最佳模型（基于验证集）
    ├── model_final.pth          # 最终模型
    └── events.out.tfevents.*    # TensorBoard日志
```

## 配置参数

### mobilenetv4_openlane.py中的主要设置
- `epochs`：训练轮数（默认：15）
- `batch_size`：总批次大小（默认：120）
- `train.checkpointer.period`：检查点保存频率（默认：100次迭代）
- `train.eval_period`：评估频率（默认：500次迭代）
- `train.output_dir`：**输出目录**（可自定义，见上文说明）
- `optimizer.lr`：学习率（默认：0.0006）
- `img_w/img_h`：输入图像尺寸（默认：800×320）
- `cut_height`：图像裁剪高度（默认：270）
- `num_lane_categories`：车道类别数量（默认：15）
- `num_lr_attributes`：左右属性数量（默认：4）
- `scale_factor`：原型相似度的温度缩放系数（默认：20.0）
- `width_mult`：MobileNetV4宽度倍数（默认：1.0）

### 修改保存频率
要更改模型保存频率，编辑 `mobilenetv4_openlane.py`：
```python
train.checkpointer.period = 100  # 每100次迭代保存一次
train.eval_period = 500         # 每500次迭代评估一次
```

## 车道类别说明

### 常规车道类型（13类）
- 0: 未知
- 1: 白色虚线
- 2: 白色实线
- 3: 双白虚线
- 4: 双白实线
- 5: 白虚左实右
- 6: 白实左虚右
- 7: 黄色虚线
- 8: 黄色实线
- 9: 双黄虚线
- 10: 双黄实线
- 11: 黄虚左实右
- 12: 黄实左虚右

### 路缘类型（2类）
- 13: 左侧路缘（原始类别20）
- 14: 右侧路缘（原始类别21）

## 左右位置属性
- 1: 最左（left-left）- 比普通左侧车道更靠左
- 2: 左侧（left）- 车辆左侧的正常车道线
- 3: 右侧（right）- 车辆右侧的正常车道线
- 4: 最右（right-right）- 比普通右侧车道更靠右

## 常见问题解决

### 问题："部分模型参数未参与损失计算"
**解决方案**：已在配置中修复。`find_unused_parameters=True` 设置允许DDP处理属性分支中的未使用参数。

### 问题：CUDA显存不足
**解决方案**：
- 减小批次大小：修改配置中的 `batch_size`
- 减小宽度倍数：修改 `width_mult`（从1.0减到0.75）
- 关闭其他GPU密集型应用
- 确保正确设置 `CUDA_VISIBLE_DEVICES`

### 问题：数据加载慢
**解决方案**：
- 增加数据加载器的 `num_workers`（当前：2）
- 使用SSD存储数据集
- 启用 `pin_memory=True`（已启用）

### 问题：收敛效果差
**解决方案**：
- 检查学习率（当前：0.0006）
- 调整温度缩放系数 `scale_factor`（默认：20.0）
- 验证数据预处理
- 确保边界线的类别映射正确

## 训练监控

### 查看日志
```bash
# 查看默认位置的日志
tail -f output/llanet/mobilenetv4_small_gsafpn_openlane/metrics.json
```

### TensorBoard可视化
```bash
tensorboard --logdir=output/llanet/mobilenetv4_small_gsafpn_openlane
```

### 关键指标监控
- `total_loss`：总体训练损失（应该下降）
- `cls_loss`：分类损失
- `reg_xytl_loss`：回归损失
- `seg_loss`：分割损失
- `iou_loss`：IoU损失
- `loss_category`：车道类别预测损失（基于原型）
- `loss_attribute`：左右属性预测损失
- `lr`：学习率（余弦衰减）

## 高级用法

### 自定义数据集路径
修改 `mobilenetv4_openlane.py` 中的 `data_root`：
```python
data_root = "/your/custom/path/to/OpenLane/dataset/raw"
```

### 修改骨干网络宽度
MobileNetV4支持通过 `width_mult` 调整网络宽度：
```python
backbone=L(MobileNetV4Small)(
    width_mult=0.75,  # 0.75x width (更轻量)
    # width_mult=1.0,  # 1.0x width (标准)
    # width_mult=1.25,  # 1.25x width (更大容量)
),
```

### 多GPU训练（带环境变量）
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 4张GPU
python tools/train_net.py \
    --config-file configs/llanet/mobilenetv4_openlane.py
```

### FP16混合精度训练
通过修改 `train.amp` 启用混合精度训练：
```python
train.amp.enabled = True
```

### 调整原型温度缩放
原型相似度的温度系数影响分类锐度：
```python
head=L(LLANetHead)(
    # ... 其他参数
    scale_factor=10.0,  # 较小值 -> 更锐利的分类
    # scale_factor=20.0,  # 默认值
    # scale_factor=30.0,  # 较大值 -> 更平滑的分类
),
```

## LLANet vs FLanet 对比

| 特性 | LLANet | FLanet |
|------|---------|---------|
| 骨干网络 | MobileNetV4-Small | ResNet34 |
| FPN类型 | GSA-FPN (非对称卷积 + 全局语义) | 标准FPN |
| 类别预测方法 | 基于原型的余弦相似度 | 线性分类器 |
| 模型大小 | 轻量级 | 中等 |
| 推理速度 | 更快 | 中等 |
| 适用场景 | 移动端/边缘设备 | 服务器端 |

## 引用
如果使用此代码，请引用LLANet论文和OpenLane数据集论文。

## 技术支持
如有问题和疑问，请参考项目文档或在代码仓库中创建issue。
