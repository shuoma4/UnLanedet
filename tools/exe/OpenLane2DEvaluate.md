# 2D车道线评估工具使用指南

## 目录
1. [工具概述](#工具概述)
2. [评估程序使用](#评估程序使用)
3. [CSV输出文件格式](#csv输出文件格式)
4. [Python读取工具使用](#python读取工具使用)
5. [完整使用流程](#完整使用流程)

---

## 工具概述

本工具用于评估2D车道线检测模型的性能，主要功能包括：

- 计算TP、FP、FN等评估指标
- 支持按类别和属性进行详细统计
- 生成Excel兼容的CSV格式输出
- 提供Python脚本进行结果分析和可视化

---

## 评估程序使用

### 1. 编译程序

```bash
cd /path/to/OpenLane/OpenLane/eval/LANE_evaluation/lane2d
make clean
make
```

编译成功后会生成 `evaluate` 可执行文件。

### 2. 准备数据文件

评估程序需要以下三类文件：

#### 2.1 标注文件 (annotations/)
JSON格式的车道线标注，每个图片对应一个同名JSON文件：

```json
{
  "lane_lines": [
    {
      "uv": [[x1, x2, ...], [y1, y2, ...]],
      "category": 1,
      "attribute": 2
    }
  ]
}
```

**字段说明：**
- `uv`: 车道线点的坐标，`uv[0]`为x坐标数组，`uv[1]`为y坐标数组
- `category`: 车道线类别ID（参考[类别ID表](#类别id表)）
- `attribute`: 车道线属性ID（参考[属性ID表](#属性id表)）

#### 2.2 检测结果文件 (results/)
模型输出的检测结果，格式与标注文件相同。

#### 2.3 图片文件 (images/)
对应的原始图片文件。

#### 2.4 图片列表文件 (test_list.txt)
包含所有需要评估的图片名称，每行一个：
```
segment-xxx_xxx.jpg
segment-yyy_yyy.jpg
```

### 3. 运行评估程序

#### 3.1 基本用法

```bash
./evaluate \
  -a ./annotations/ \
  -d ./results/ \
  -i ./images/ \
  -l ./test_list.txt \
  -o ./output/
```

#### 3.2 命令行参数详解

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-h` | 显示帮助信息 | - |
| `-a <path>` | 标注文件目录 | /data/driving/eval_data/anno_label/ |
| `-d <path>` | 检测结果目录 | /data/driving/eval_data/predict_label/ |
| `-i <path>` | 图片文件目录 | /data/driving/eval_data/img/ |
| `-l <path>` | 图片列表文件 | /data/driving/eval_data/img/all.txt |
| `-o <path>` | 输出目录 | ./output |
| `-w <int>` | 车道线宽度 | 30 |
| `-t <float>` | IoU阈值 | 0.3 |
| `-c <int>` | 最大图片宽度 | 1920 |
| `-r <int>` | 最大图片高度 | 1280 |
| `-s` | 显示可视化 | false |
| `-e <name>` | 实验名称 | exp |
| `-j <int>` | 线程数（0为自动） | 0 |
| `-v` | 生成可视化图片 | false |

#### 3.3 示例：使用example文件夹测试

```bash
cd example
export LD_LIBRARY_PATH=/home/lixiyang/anaconda3/envs/dataset-manger/lib:$LD_LIBRARY_PATH
bash eval_demo.sh
```

或手动运行：
```bash
../evaluate \
  -a ./annotations/ \
  -d ./results/ \
  -i ./images/ \
  -l ./test_list.txt \
  -o ./
```

#### 3.4 输出说明

程序运行后会输出：

```
------------Configuration---------
anno_dir: ./annotations/
detect_dir: ./results/
...
visualize: false
-----------------------------------
Auto-detected 72 CPU cores, will use all available threads
Starting evaluation with category/attribute statistics...
Processed 0/2 images
Evaluation complete!
========================================
Overall Statistics:
tp: 8 fp: 4 fn: 2
precision: 0.666667
recall: 0.8
Fmeasure: 0.727273
========================================
CSV results saved to: ./csv_results/
  - iou_list.csv
  - category_stats.csv
  - attribute_stats.csv
========================================
```

输出文件位置：
- `{output_folder}csv_results/iou_list.csv` - IoU列表
- `{output_folder}csv_results/category_stats.csv` - 类别统计
- `{output_folder}csv_results/attribute_stats.csv` - 属性统计

---

## CSV输出文件格式

评估程序会生成三个CSV文件，保存在 `{output_folder}csv_results/` 目录下。

### 1. iou_list.csv - IoU列表

记录每个成功匹配的车道线对的详细信息。

#### 文件格式
```csv
ImageName,AnnoIdx,DetectIdx,IoU,AnnoCategory,DetectCategory,AnnoAttribute,DetectAttribute
segment-xxx.jpg,0,0,0.786725,20,20,0,0
segment-xxx.jpg,2,2,0.775137,1,1,4,0
```

#### 字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| ImageName | string | 图片文件名（包含相对路径） |
| AnnoIdx | int | 标注车道线在JSON中的索引 |
| DetectIdx | int | 检测车道线在JSON中的索引 |
| IoU | float | 交并比（0-1之间，越接近1越好） |
| AnnoCategory | int | 标注车道线的类别ID |
| DetectCategory | int | 检测车道线的类别ID |
| AnnoAttribute | int | 标注车道线的属性ID |
| DetectAttribute | int | 检测车道线的属性ID |

### 2. category_stats.csv - 类别统计

按14种车道线类型分类统计检测结果。

#### 文件格式
```csv
CategoryID,CategoryName,TP,FP,FN,Precision,Recall,F1-Score
1,white-dash,4,0,0,1,1,1
2,white-solid,2,2,0,0.5,1,0.666667
...
OVERALL,Overall,8,4,2,0.666667,0.8,0.727273
```

#### 字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| CategoryID | int/string | 车道线类别ID或"OVERALL" |
| CategoryName | string | 车道线类别名称 |
| TP | int | True Positive（正确检测数） |
| FP | int | False Positive（误检数） |
| FN | int | False Negative（漏检数） |
| Precision | float/N/A | 精确率 = TP / (TP + FP) |
| Recall | float/N/A | 召回率 = TP / (TP + FN) |
| F1-Score | float/N/A | F1分数 = 2 * P * R / (P + R) |

### 3. attribute_stats.csv - 属性统计

按4种位置属性分类统计检测结果。

#### 文件格式
```csv
AttributeID,AttributeName,TP,FP,FN,Precision,Recall,F1-Score
0,unknown,4,4,2,0.5,0.666667,0.571429
3,right,2,0,0,1,1,1
4,right-right,2,0,0,1,1,1
```

#### 字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| AttributeID | int | 车道线属性ID |
| AttributeName | string | 车道线属性名称 |
| TP | int | True Positive（正确检测数） |
| FP | int | False Positive（误检数） |
| FN | int | False Negative（漏检数） |
| Precision | float/N/A | 精确率 = TP / (TP + FP) |
| Recall | float/N/A | 召回率 = TP / (TP + FN) |
| F1-Score | float/N/A | F1分数 = 2 * P * R / (P + R) |

### 类别ID表

| ID | 类别名称 | 说明 |
|----|----------|------|
| 0 | unknown | 未知类型 |
| 1 | white-dash | 白色虚线 |
| 2 | white-solid | 白色实线 |
| 3 | double-white-dash | 双白色虚线 |
| 4 | double-white-solid | 双白色实线 |
| 5 | white-ldash-rsolid | 左白虚右白实 |
| 6 | white-lsolid-rdash | 左白实右白虚 |
| 7 | yellow-dash | 黄色虚线 |
| 8 | yellow-solid | 黄色实线 |
| 9 | double-yellow-dash | 双黄色虚线 |
| 10 | double-yellow-solid | 双黄色实线 |
| 11 | yellow-ldash-rsolid | 左黄虚右黄实 |
| 12 | yellow-lsolid-rdash | 左黄实右黄虚 |
| 20 | left-curbside | 左路沿 |
| 21 | right-curbside | 右路沿 |

### 属性ID表

| ID | 属性名称 | 说明 |
|----|----------|------|
| 0 | unknown | 未知属性（默认值） |
| 1 | left-left | 最左侧车道线 |
| 2 | left | 左侧车道线 |
| 3 | right | 右侧车道线 |
| 4 | right-right | 最右侧车道线 |

### 评估指标说明

- **TP (True Positive)**: 正确检测到的车道线数量
- **FP (False Positive)**: 误检的车道线数量
- **FN (False Negative)**: 漏检的车道线数量
- **Precision (精确率)**: TP / (TP + FP)，表示检测到的车道线中正确的比例
- **Recall (召回率)**: TP / (TP + FN)，表示真实车道线中被检测到的比例
- **F1-Score**: 2 * P * R / (P + R)，精确率和召回率的调和平均数

---

## Python读取工具使用

### 1. 安装依赖

```bash
pip install pandas numpy matplotlib openpyxl
```

### 2. 基本用法

#### 2.1 最简单用法

```bash
python read_csv_results.py --csv-folder ./csv_results/
```

输出：
```
✓ 加载IoU列表: 8 条记录
✓ 加载类别统计: 4 个类别
✓ 加载属性统计: 3 个属性

============================================================
评估结果摘要
============================================================
TP: 8, FP: 4, FN: 2
Precision: 0.6667
Recall: 0.8000
F1-Score: 0.7273
============================================================
```

#### 2.2 显示最佳类别

```bash
python read_csv_results.py --csv-folder ./csv_results/ --top-n 5
```

仅显示F1分数最高的前5个类别。

#### 2.3 查看最差的匹配

```bash
python read_csv_results.py --csv-folder ./csv_results/ --worst-n 10
```

显示IoU最低的10个车道线匹配。

#### 2.4 导出为Excel

```bash
python read_csv_results.py --csv-folder ./csv_results/ --excel results.xlsx
```

将三个CSV文件合并为一个Excel文件，包含三个Sheet：
- Sheet 1: IoU_List
- Sheet 2: Category_Stats
- Sheet 3: Attribute_Stats

#### 2.5 生成性能对比图

```bash
python read_csv_results.py --csv-folder ./csv_results/ --plot performance.png
```

生成各类别F1-Score对比的柱状图。

#### 2.6 综合使用

```bash
python read_csv_results.py \
  --csv-folder ./csv_results/ \
  --top-n 10 \
  --worst-n 5 \
  --excel results.xlsx \
  --plot performance.png
```

### 3. 在Python脚本中使用

```python
from read_csv_results import CSVResultReader

# 创建读取器
reader = CSVResultReader('./csv_results/')

# 加载所有CSV文件
reader.load_all()

# 打印摘要
reader.print_summary()

# 打印类别统计（前5名）
reader.print_category_stats(top_n=5)

# 打印属性统计
reader.print_attribute_stats()

# 打印IoU分布
reader.print_iou_summary()

# 获取最差的10个匹配
worst_matches = reader.get_worst_matches(n=10)

# 导出为Excel
reader.export_to_excel('results.xlsx')

# 绘制性能对比图
reader.plot_category_performance(save_path='performance.png')

# 获取类别统计DataFrame
category_df = reader.get_category_wise_stats()
print(category_df)

# 获取属性统计DataFrame
attribute_df = reader.get_attribute_wise_stats()
print(attribute_df)
```

### 4. 自定义分析

```python
import pandas as pd
from read_csv_results import CSVResultReader

reader = CSVResultReader('./csv_results/')
reader.load_all()

# 分析特定类别的性能
if reader.category_df is not None:
    # 找出F1分数低于0.7的类别
    poor_performance = reader.category_df[
        (reader.category_df['CategoryID'] != 'OVERALL') &
        (reader.category_df['F1-Score'] < 0.7)
    ]
    print("性能较差的类别:")
    print(poor_performance[['CategoryName', 'F1-Score']])

# 分析IoU分布
if reader.iou_df is not None:
    high_iou = reader.iou_df[reader.iou_df['IoU'] > 0.8]
    low_iou = reader.iou_df[reader.iou_df['IoU'] < 0.6]

    print(f"\n高IoU匹配 (>0.8): {len(high_iou)}")
    print(f"低IoU匹配 (<0.6): {len(low_iou)}")

    # 计算各类别的平均IoU
    avg_iou_by_category = reader.iou_df.groupby('AnnoCategory')['IoU'].mean()
    print("\n各类别平均IoU:")
    print(avg_iou_by_category)
```

### 5. 命令行参数完整说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--csv-folder` | CSV文件所在文件夹路径 | ./csv_results/ |
| `--top-n` | 仅显示前N个类别/属性（按F1分数降序） | None（全部显示） |
| `--worst-n` | 显示最差的N个匹配 | 10 |
| `--excel` | 导出为Excel文件的路径 | None（不导出） |
| `--plot` | 保存类别性能对比图的路径 | None（不保存） |

---

## 完整使用流程

### 场景1：评估新模型

```bash
# 1. 准备数据
# - 标注文件放在 annotations/
# - 模型输出放在 results/
# - 图片放在 images/
# - 创建 test_list.txt 列出所有图片

# 2. 编译评估程序
cd /path/to/lane2d
make clean && make

# 3. 运行评估
./evaluate \
  -a ./annotations/ \
  -d ./results/ \
  -i ./images/ \
  -l ./test_list.txt \
  -o ./output/ \
  -t 0.3 \
  -j 0

# 4. 查看评估结果
cat output/output.txt

# 5. 使用Python工具分析
python read_csv_results.py \
  --csv-folder output/csv_results/ \
  --excel analysis.xlsx \
  --plot performance.png
```

### 场景2：对比不同模型

```bash
# 模型A评估
./evaluate -a anno/ -d modelA/ -i img/ -l list.txt -o resultsA/

# 模型B评估
./evaluate -a anno/ -d modelB/ -i img/ -l list.txt -o resultsB/

# 对比分析
python compare_models.py \
  --folder1 resultsA/csv_results/ \
  --folder2 resultsB/csv_results/ \
  --name1 ModelA \
  --name2 ModelB
```

### 场景3：调试特定问题

```bash
# 1. 运行评估并生成可视化
./evaluate -a anno/ -d model/ -i img/ -l list.txt -o debug/ -v

# 2. 查找性能较差的类别
python read_csv_results.py \
  --csv-folder debug/csv_results/ \
  --top-n 10

# 3. 查看IoU较低的匹配
python read_csv_results.py \
  --csv-folder debug/csv_results/ \
  --worst-n 20

# 4. 查看可视化图片，分析问题原因
ls debug/visual/
```

### 场景4：生成实验报告

```python
# generate_report.py
from read_csv_results import CSVResultReader
import matplotlib.pyplot as plt

reader = CSVResultReader('./csv_results/')
reader.load_all()

# 1. 总体性能
reader.print_summary()

# 2. 各类别性能
reader.print_category_stats()

# 3. 各属性性能
reader.print_attribute_stats()

# 4. IoU分布
reader.print_iou_summary()

# 5. 生成图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 类别F1对比
category_df = reader.get_category_wise_stats()
axes[0, 0].barh(category_df['CategoryName'], category_df['F1-Score'])
axes[0, 0].set_title('Category F1-Score')

# 属性F1对比
attribute_df = reader.get_attribute_wise_stats()
axes[0, 1].barh(attribute_df['AttributeName'], attribute_df['F1-Score'])
axes[0, 1].set_title('Attribute F1-Score')

# IoU分布
reader.iou_df['IoU'].hist(bins=20, ax=axes[1, 0])
axes[1, 0].set_title('IoU Distribution')

# TP/FP/FN对比
overall = reader.category_df[reader.category_df['CategoryID'] == 'OVERALL'].iloc[0]
axes[1, 1].bar(['TP', 'FP', 'FN'], [overall['TP'], overall['FP'], overall['FN']])
axes[1, 1].set_title('TP/FP/FN')

plt.tight_layout()
plt.savefig('experiment_report.png', dpi=150)
plt.show()
```

---

## 常见问题

### Q1: 编译时提示找不到OpenCV库

```bash
# 设置库路径
export LD_LIBRARY_PATH=/path/to/opencv/lib:$LD_LIBRARY_PATH
```

### Q2: 评估速度很慢

- 检查是否使用了 `-v` 参数（生成可视化图片）
- 增加线程数：`-j 16` 或 `-j 32`
- 减少IoU阈值：`-t 0.3` 会比 `-t 0.5` 更快

### Q3: CSV文件无法用Excel正常显示

- 确保使用UTF-8编码
- 用Excel打开时选择"导入"而不是直接打开
- 或者使用Python工具导出为xlsx格式

### Q4: 类别或属性统计显示为N/A

- 当TP+FP=0时，Precision为N/A（没有检测到任何车道线）
- 当TP+FN=0时，Recall为N/A（标注中没有该类别）
- 这是正常现象，表示该类别在数据中不存在或未检测到

---

## 更新日志

- v1.0: 初始版本
  - 基础评估功能
  - 类别和属性统计
  - CSV输出
  - Python读取工具
