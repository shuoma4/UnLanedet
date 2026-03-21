from fvcore.common.param_scheduler import CosineParamScheduler
from omegaconf import OmegaConf

from unlanedet.config import LazyCall as L
from unlanedet.data.transform.custom_transforms import BGR2RGB
from unlanedet.data.transform.openlane_generator import OpenLaneGenerator
from unlanedet.data.transform.transforms import ToTensor

from ...modelzoo import get_config
from ..model_factory import create_llanetv1_model

IOU_LOSS_WEIGHT = 2.0
CLS_LOSS_WEIGHT = 2.0
XYT_LOSS_WEIGHT = 0.2
SEG_LOSS_WEIGHT = 1.0
CATEGORY_LOSS_WEIGHT = 1.0
NUM_POINTS = 72
MAX_LANES = 12
NUM_PRIORS = 192
NUM_LANE_CATEGORIES = 15
ORI_IMG_W = 1920
ORI_IMG_H = 1280
IMG_W = 800
IMG_H = 320
CUT_HEIGHT = 270
BATCH_SIZE = 24
EPOCHS = 15
EPOCH_PER_ITER = 142226 // BATCH_SIZE + 1
DATA_ROOT = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/lane3d_1000"
DATASET_STATISTICS = "/data1/lxy_log/workspace/ms/UnLanedet/source/openlane_statistics/openlane_priors_with_clusters.npz"

TEST_PARAMETERS = dict(conf_threshold=0.4, nms_thres=50, nms_topk=MAX_LANES)
IMG_NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TRAIN_TRANSFORMS = [
    dict(name="Resize", parameters=dict(size=dict(height=IMG_H, width=IMG_W)), p=1.0),
    dict(name="HorizontalFlip", parameters=dict(p=1.0), p=0.5),
    dict(name="ChannelShuffle", parameters=dict(p=1.0), p=0.1),
    dict(
        name="MultiplyAndAddToBrightness",
        parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
        p=0.6,
    ),
    dict(name="AddToHueAndSaturation", parameters=dict(value=(-10, 10)), p=0.7),
    dict(
        name="OneOf",
        transforms=[
            dict(name="MotionBlur", parameters=dict(k=(3, 5))),
            dict(name="MedianBlur", parameters=dict(k=(3, 5))),
        ],
        p=0.2,
    ),
    dict(
        name="Affine",
        parameters=dict(
            translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)),
            rotate=(-10, 10),
            scale=(0.8, 1.2),
        ),
        p=0.7,
    ),
    dict(name="Resize", parameters=dict(size=dict(height=IMG_H, width=IMG_W)), p=1.0),
]

VAL_TRANSFORMS = [
    dict(name="Resize", parameters=dict(size=dict(height=IMG_H, width=IMG_W)), p=1.0),
]


def build_config(
    run_name,
    backbone_type="resnet",
    backbone_name="resnet34",
    neck_type="FPN",
    enable_category_head=False,
    category_head_type="linear",
    use_data_driven_priors=False,
    assign_method="CLRNet",
    enable_temporal_model=False,
    temporal_loss_weight=0.0,
    enable_global_semantic=False,
    featuremap_out_channel=64,
    fc_hidden_dim=64,
    distill_cfg=None,
    deploy_cfg=None,
    epochs=15,
):
    param_config = OmegaConf.create()
    param_config.backbone_type = backbone_type
    param_config.backbone_name = backbone_name
    param_config.use_pretrained_backbone = True
    param_config.neck_type = neck_type
    param_config.enable_global_semantic = enable_global_semantic
    param_config.enable_category_head = enable_category_head
    param_config.enable_lane_category = enable_category_head
    param_config.category_head_type = category_head_type
    param_config.category_scale_factor = 20.0
    param_config.use_data_driven_priors = use_data_driven_priors
    param_config.assign_method = assign_method
    param_config.enable_temporal_model = enable_temporal_model
    param_config.temporal_loss_weight = temporal_loss_weight
    param_config.distill_cfg = distill_cfg or dict(enable=False)
    param_config.deploy_cfg = deploy_cfg or dict(
        enable_qat=False, enable_onnx_export=False
    )
    param_config.dataset_statistics = DATASET_STATISTICS
    param_config.iou_loss_weight = IOU_LOSS_WEIGHT
    param_config.cls_loss_weight = CLS_LOSS_WEIGHT
    param_config.xyt_loss_weight = XYT_LOSS_WEIGHT
    param_config.seg_loss_weight = SEG_LOSS_WEIGHT
    param_config.category_loss_weight = CATEGORY_LOSS_WEIGHT
    param_config.num_points = NUM_POINTS
    param_config.max_lanes = MAX_LANES
    param_config.num_priors = NUM_PRIORS
    param_config.num_lane_categories = NUM_LANE_CATEGORIES
    param_config.refine_layers = 3
    param_config.sample_points = 36
    param_config.sample_y = [i for i in range(1279, 270, -20)]
    param_config.test_parameters = TEST_PARAMETERS
    param_config.ori_img_w = ORI_IMG_W
    param_config.ori_img_h = ORI_IMG_H
    param_config.img_w = IMG_W
    param_config.img_h = IMG_H
    param_config.cut_height = CUT_HEIGHT
    param_config.img_norm = IMG_NORM
    param_config.ignore_label = 255
    param_config.bg_weight = 0.4
    param_config.neck_out_channels = featuremap_out_channel
    param_config.featuremap_out_channel = featuremap_out_channel * 3
    param_config.fc_hidden_dim = fc_hidden_dim
    param_config.num_classes = 2
    param_config.data_root = DATA_ROOT
    param_config.use_offline_resized = True
    param_config.w_cls = 2.0
    param_config.w_geom = 4.0
    param_config.w_iou = 2.0
    param_config.simota_q = 10

    model = create_llanetv1_model(param_config)

    train = get_config("config/common/train.py").train
    total_iter = EPOCH_PER_ITER * epochs
    train.max_iter = total_iter
    train.checkpointer.period = EPOCH_PER_ITER
    train.eval_period = EPOCH_PER_ITER
    train.output_dir = f"output/llanetv1/openlane1000/{run_name}"
    train.amp.enabled = True

    param_config.output_dir = train.output_dir
    param_config.epoch_per_iter = EPOCH_PER_ITER

    optimizer = get_config("config/common/optim.py").AdamW
    optimizer.lr = 2.6e-4
    optimizer.weight_decay = 0.01

    lr_multiplier = L(CosineParamScheduler)(start_value=1.0, end_value=0.001)

    train_process = [
        L(BGR2RGB)(),
        L(OpenLaneGenerator)(transforms=TRAIN_TRANSFORMS, cfg=param_config),
        L(ToTensor)(
            keys=["img", "lane_line", "seg"],
            collect_keys=["lane_categories", "lane_attributes", "lane_track_ids"],
        ),
    ]
    val_process = [
        L(BGR2RGB)(),
        L(OpenLaneGenerator)(
            transforms=VAL_TRANSFORMS, training=False, cfg=param_config
        ),
        L(ToTensor)(
            keys=["img"],
            collect_keys=[
                "img_path",
                "lane_categories",
                "lane_attributes",
                "lane_track_ids",
            ],
        ),
    ]

    dataloader = get_config("config/common/openlane.py").dataloader
    dataloader.train.dataset.data_root = DATA_ROOT
    dataloader.train.dataset.processes = train_process
    dataloader.train.dataset.cut_height = CUT_HEIGHT
    dataloader.train.dataset.cfg = param_config
    dataloader.train.total_batch_size = BATCH_SIZE
    dataloader.train.num_workers = 4
    dataloader.train.persistent_workers = True
    dataloader.train.pin_memory = True
    dataloader.train.prefetch_factor = 4

    dataloader.test.dataset.data_root = DATA_ROOT
    dataloader.test.dataset.processes = val_process
    dataloader.test.dataset.cut_height = CUT_HEIGHT
    dataloader.test.dataset.cfg = param_config
    dataloader.test.total_batch_size = BATCH_SIZE
    dataloader.test.num_workers = 4

    dataloader.evaluator.cfg = param_config
    dataloader.evaluator.output_dir = f"{train.output_dir}/eval_results"

    return model, dataloader, train, optimizer, lr_multiplier, param_config
