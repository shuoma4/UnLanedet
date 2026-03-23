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
    batch_size=2,
)

param_config.scm_kernel_size = 9
param_config.category_loss_weight = 5.0
train.amp.enabled = False

# 提升数据读取并行度，解决T=3时序数据加载瓶颈导致的速度过慢问题
dataloader.train.num_workers = 8
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


# 3. Override the Temporal Model
model.temporal_model = L(TemporalFusionWrapper)(
    in_channels=64,  # Default FPN out_channels
    num_levels=getattr(param_config, "refine_layers", 3),
)
dataloader.train.total_batch_size = 2
dataloader.test.total_batch_size = 2
