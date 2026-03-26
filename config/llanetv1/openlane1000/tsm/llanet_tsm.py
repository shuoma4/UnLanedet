from ..common import build_config
from unlanedet.config import LazyCall as L

# Import new modules
from unlanedet.data.openlane_temporal import OpenLaneTemporal
from unlanedet.data.transform.openlane_generator_temporal import (
    OpenLaneTemporalGenerator,
    TemporalToTensor,
    TemporalNormalize,
)
from unlanedet.data.transform.custom_transforms import BGR2RGB
from unlanedet.model.llanetv1.temporal_modules import TemporalFusionWrapper
from ..common import TRAIN_TRANSFORMS, VAL_TRANSFORMS

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="tsm/baseline",
    backbone_type="resnet",
    backbone_name="resnet34",
    neck_type="GSAFPN",
    enable_category_head=True,
    category_head_type="combined",
    use_data_driven_priors=False,
    assign_method="CLRNet",
    enable_temporal_model=False,  # We override it manually below
    temporal_loss_weight=0.5,
    enable_global_semantic=True,
    # 单卡：total_batch=20（与原双卡时每卡 batch 相同），AMP 下约 8–9 GB
    batch_size=20,
)

param_config.scm_kernel_size = 9
param_config.category_loss_weight = 5.0
train.amp.enabled = True           # fp16 混合精度，约减 30–40% per-iter 时间
train.float32_precision = "high"   # TF32（Ada Lovelace 上有额外加速）
train.cudnn_benchmark = True

dataloader.train.num_workers = 4
dataloader.train.persistent_workers = True

# 1. Override the Datasets
dataloader.train.dataset = L(OpenLaneTemporal)(
    data_root=dataloader.train.dataset.data_root,
    split="train",
    cut_height=dataloader.train.dataset.cut_height,
    seq_len=3,
    cfg=param_config,
)
dataloader.test.dataset = L(OpenLaneTemporal)(
    data_root=dataloader.test.dataset.data_root,
    split="val",
    cut_height=dataloader.test.dataset.cut_height,
    seq_len=3,
    cfg=param_config,
)

# 2. Override the Data Process pipelines
dataloader.train.dataset.processes = [
    L(BGR2RGB)(),
    L(OpenLaneTemporalGenerator)(
        transforms=TRAIN_TRANSFORMS, cfg=param_config, training=True
    ),
    L(TemporalToTensor)(
        keys=["img", "lane_line", "seg", "lane_vis", "intrinsic", "extrinsic"],
        collect_keys=[
            "lane_categories",
            "lane_attributes",
            "lane_track_ids",
            "track_id",
            "seq_visibility",
            "seq_xyz",
            "seq_track_id",
        ],
    ),
]

dataloader.test.dataset.processes = [
    L(BGR2RGB)(),
    L(OpenLaneTemporalGenerator)(
        transforms=VAL_TRANSFORMS, cfg=param_config, training=False
    ),
    L(TemporalToTensor)(
        keys=["img", "lane_vis", "intrinsic", "extrinsic"],
        collect_keys=[
            "img_path",
            "lane_categories",
            "lane_attributes",
            "lane_track_ids",
            "track_id",
            "seq_visibility",
            "seq_xyz",
            "seq_track_id",
        ],
    ),
]


# 3. Override the Temporal Model，传入 cfg 使 TemporalConsistencyLoss 能读取图像尺寸等参数
model.temporal_model = L(TemporalFusionWrapper)(
    in_channels=64,
    num_levels=getattr(param_config, "refine_layers", 3),
    cfg=param_config,
)
dataloader.train.total_batch_size = 20
# 评估时 batch 小；单卡 seq_len=3 不宜过大
dataloader.test.total_batch_size = 4

# 4. 训练 I/O
dataloader.train.pin_memory = True
dataloader.train.prefetch_factor = 2
dataloader.test.num_workers = 2
dataloader.test.persistent_workers = False  # eval 结束后立即释放 worker 内存
dataloader.test.pin_memory = True
dataloader.test.prefetch_factor = 2

# 权重调回 5.0（因为我们内部重新加回了 0.05 的安全缩放）
param_config.temporal_loss_weight = 5.0
