# LLANetV1 配置整理说明

## 1. 目录结构

- openlane300/
  - `resnet34_fpn_baseline.py`：当前基线，整体写法对齐 `config/fclrnet/resnet34_openlane_300d.py`
  - `resnet34_gsafpn_category.py`：ResNet34 + GSA-FPN + 分类头 + 统计先验
  - `mbv4_fpn_category.py`：MobileNetV4 + FPN 消融
  - `mbv4_gsafpn_category.py`：MobileNetV4 + GSA-FPN
  - `mbv4_gsafpn_temporal.py`：时序版本
  - `resnet50_fpn_teacher.py`：教师模型
  - `mbv4_gsafpn_distill.py`：蒸馏训练
  - `mbv4_gsafpn_qat.py`：QAT / 部署实验

- openlane1000/
  - `resnet34_fpn_baseline.py`：当前基线，整体写法对齐 `config/fclrnet/resnet34_openlane_1000d.py`
  - `resnet34_gsafpn_category.py`
  - `mbv4_fpn_category.py`
  - `mbv4_gsafpn_category.py`
  - `mbv4_gsafpn_temporal.py`
  - `resnet50_fpn_teacher.py`
  - `mbv4_gsafpn_distill.py`
  - `mbv4_gsafpn_qat.py`

- culane/
  - 仅保留目录占位，后续单独补配置

## 2. 这次整理的原则

1. 删除根目录下无意义的平铺配置与错误别名配置。
2. 基线配置回到 `fclrnet` 风格：数据集、训练轮数、batch size、eval period、optimizer 等写法保持一致。
3. 只保留真正有用途的训练配置：基线、neck/backbone 消融、时序、蒸馏、QAT。
4. 按数据集拆分文件夹，避免 300d / 1000d / CULane 混放。

## 3. 推荐训练入口

### OpenLane 300d 基线

python tools/train_net.py --config-file config/llanetv1/openlane300/resnet34_fpn_baseline.py

### OpenLane 300d 第三章模型

python tools/train_net.py --config-file config/llanetv1/openlane300/mbv4_gsafpn_category.py

### OpenLane 300d 时序模型

python tools/train_net.py --config-file config/llanetv1/openlane300/mbv4_gsafpn_temporal.py

### OpenLane 1000d 基线

python tools/train_net.py --config-file config/llanetv1/openlane1000/resnet34_fpn_baseline.py

### OpenLane 1000d 第三章模型

python tools/train_net.py --config-file config/llanetv1/openlane1000/mbv4_gsafpn_category.py

## 4. 说明

- 当前 OpenLane 配置使用的是基于 `GenerateLaneLine` 改写的 `OpenLaneGenerator`，既保持 fCLRNet 兼容的编码方式，也会同步保留 `lane_categories`、`lane_attributes`、`lane_track_ids`。
- 但配置整体组织方式、训练超参和 evaluator 接法，已经按 `config/fclrnet/resnet34_openlane_300d.py` / `1000d.py` 的风格重写。
- 如果后续要补 CULane，请直接在 `config/llanetv1/culane/` 下单独新增，不要再放回根目录。