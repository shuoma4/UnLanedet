# OpenLane 车道线检测评估套件

这是 OpenLane 车道检测的官方评估套件。

## 概述
- #环境要求与安装
- #2d-车道线评估
- #致谢

## <a name="requirement"></a> 环境要求与安装
参见 ../README.md

## <a name="2d_lane"></a> 2D 车道线评估

### 数据格式要求
- 按照以下结构准备结果 JSON 文件目录：
```
├── 结果目录
|   ├── validation（验证集）
|   |   ├── segment-xxx（路段ID）
|   |   |   ├── xxx.json（结果文件）
|   |   |   └── ...
|   |   ├── segment-xxx
|   |   |   ├── xxx.json
|   |   |   └── ...
|   |   └── ...
```

- 准备测试列表文件(.txt)，包含数据集中所有图像的相对路径，路径结构需与上述一致：
```
validation/segment-xxx/xxx.jpg
validation/segment-xxx/xxx.jpg
validation/segment-xxx/xxx.jpg
...
```

- 每个结果 JSON 文件应遵循以下结构：
```json
{
    "file_path":                            "<字符串> -- 图像路径",
    "lane_lines": [
        {
            "category":                     "<整数> -- 车道线类别",
            "uv":                           "<浮点数数组> [2, n] -- 像素坐标系下的2D车道点"
        },
        ...                                 （`lane_lines` 字典中包含 k 条车道线）
    ]
}
```

### 评估执行
运行以下命令来评估您的方法：
```
cd lane2d
./evaluate -a $dataset_dir -d $result_dir -i $image_dir -l $test_list -w $w_lane -t $iou -o $output_file
```

参数说明如下：

`dataset_dir`：OpenLane 数据集的数据（标注）路径

`result_dir`：您的模型检测结果路径。参见上面的"数据格式要求"

`image_dir`：OpenLane 数据集的图像路径

`test_list`：测试列表文件(.txt)，包含每张图像的相对路径。参见上面的"数据格式要求"

`w_lane`：车道线宽度，原始 https://github.com/XingangPan/SCNN 论文中设为 30

`iou`：评估使用的 IOU 阈值，原始 https://github.com/XingangPan/SCNN 论文中使用 0.3/0.5

`output_file`：评估输出文件路径

示例命令：
```
./evaluate \
-a ./Dataset/OpenLane/lane3d_v2.0/ \
-d ./Evaluation/PersFormer/result_dir/ \
-i ./Dataset/OpenLane/images/ \
-l ./Evaluation/PersFormer/test_list.txt \
-w 30 \
-t 0.3 \
-o ./Evaluation/PersFormer/ \
```

我们在 `example` 文件夹中提供了一些 JSON 文件，您可以运行演示评估：
```
cd example
bash eval_demo.sh
```

### 已知问题
- 找不到 libopencv 库 `error while loading shared libraries: libopencv_core.so.3.4: cannot open shared object file: No such file or directory`  
请尝试 `export LD_LIBRARY_PATH=/path/to/opencv/lib64/:$LD_LIBRARY_PATH`（例如：`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib`），然后运行 `bash eval_demo.sh`

### 评估指标公式
我们采用了 CULane 数据集在 https://github.com/XingangPan/SCNN 中使用的评估指标

## <a name="3d_lane"></a> 3D 车道线评估

### 数据格式要求
- 按照以下结构准备结果 JSON 文件目录：
```
├── 结果目录
|   ├── validation（验证集）
|   |   ├── segment-xxx（路段ID）
|   |   |   ├── xxx.json（结果文件）
|   |   |   └── ...
|   |   ├── segment-xxx
|   |   |   ├── xxx.json
|   |   |   └── ...
|   |   └── ...
```

- 准备测试列表文件(.txt)，包含数据集中所有图像的相对路径，路径结构需与上述一致：
```
validation/segment-xxx/xxx.jpg
validation/segment-xxx/xxx.jpg
validation/segment-xxx/xxx.jpg
...
```

- 每个结果 JSON 文件应包含以下结构的结果：
```json
{
    "intrinsic":                            "<浮点数数组> [3, 3] -- 相机内参矩阵",
    "extrinsic":                            "<浮点数数组> [4, 4] -- 相机外参矩阵",
    "file_path":                            "<字符串> -- 图像路径",
    "lane_lines": [
        {
            "xyz":                          "<浮点数数组> [3, n] -- 车辆坐标系下的采样点x,y,z坐标",
            "category":                     "<整数> -- 车道线类别"
        },
        ...                                 （`lane_lines` 字典中包含 k 条车道线）
    ]
}
```

### 评估执行
运行以下命令来评估您的方法：
```
cd lane3d
python eval_3D_lane.py --dataset_dir $dataset_dir --pred_dir $pred_dir --test_list $test_list
```

基本参数说明如下。更多参数请参见脚本 `utils.py`

`dataset_dir`：OpenLane 数据集的数据（标注）路径

`pred_dir`：您的模型预测结果路径。参见上面的"数据格式要求"

`test_list`：测试列表文件(.txt)，包含每张图像的相对路径。参见上面的"数据格式要求"

示例命令：
```
python eval_3D_lane.py \
--dataset_dir=./Dataset/OpenLane/lane3d_v2.0/ \
--pred_dir=./Evaluation/PersFormer/result_dir/ \
--test_list=./Evaluation/PersFormer/test_list.txt \
```

我们在 `example` 文件夹中提供了一些 JSON 文件，您可以运行演示评估：
```
cd example
bash eval_demo.sh
```

### 评估指标公式
为了评估 3D 车道预测结果，我们首先根据可见性剪裁真实车道线（ground truth），只考虑那些在采样范围内有重叠的车道线（包括真实值和预测值）

在每个 y 步长处重新采样车道线后，我们为每个点定义一个新的可见性值：只有在 x 和 y 范围内的点才被设置为可见点

匹配成本定义为每条真实车道线和预测车道线之间的欧几里得距离，公式如下：

$$
d_{i}^{j,k}=
\begin{cases}
(x_{i}^{j}-x_{i}^{k})^2+(z_{i}^{j}-z_{i}^{k})^2, \quad 如果两者都可见\\
0, {\kern 100pt} 如果两者都不可见\\
1.5, {\kern 100pt} 其他情况\\
\end{cases}
$$

然后使用最小成本流算法（minimum-cost flow）获取全局最佳匹配结果

对于每个真实-预测匹配对，我们还统计了欧几里得距离小于阈值（此处我们设为 1.5）的匹配点数量。根据上述定义，当满足以下条件时，预测车道线可以被计为真正例（true positive）：

$$
\begin{cases}
\frac {匹配点数量}{真实点数量}\geq0.75,\\
\frac {匹配点数量}{预测点数量}\geq0.75,\\
\end{cases}
$$

此外，我们将误差度量分为两部分：近距离误差（前 40 个点）和远距离误差（剩余的 60 个点）

## <a name="ack"></a> 致谢
我们的 2D 评估代码基于 https://github.com/XingangPan/SCNN，3D 评估代码基于 https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection。