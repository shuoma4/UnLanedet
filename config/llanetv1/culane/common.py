from fvcore.common.param_scheduler import CosineParamScheduler
from omegaconf import OmegaConf

from unlanedet.config import LazyCall as L
from unlanedet.data.transform.transforms import ToTensor
from unlanedet.data.transform import GenerateLaneLine

from ...modelzoo import get_config
from ..model_factory import create_llanetv1_model


def build_config(
    run_name,
    backbone_type="resnet",
    backbone_name="resnet34",
    neck_type="FPN",
    enable_category_head=False,
    use_data_driven_priors=False,
    assign_method="CLRNet",
    enable_global_semantic=False,
    featuremap_out_channel=64,
    sample_points=36,
    batch_size=24,
    fc_hidden_dim=64,
    epochs=15,
):
    iou_loss_weight = 2.0
    cls_loss_weight = 2.0
    xyt_loss_weight = 0.2
    seg_loss_weight = 1.0
    num_points = 72
    max_lanes = 4
    sample_y = [i for i in range(589, 230, -20)]
    test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)
    ori_img_w = 1640
    ori_img_h = 590
    img_w = 800
    img_h = 320
    cut_height = 270
    img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1.0, 1.0, 1.0])
    ignore_label = 255
    bg_weight = 0.4
    num_classes = 4 + 1
    data_root = "/data0/lxy_data/mslanedet/CULane/"

    param_config = OmegaConf.create()

    # Standard LLANetV1 settings
    param_config.backbone_type = backbone_type
    param_config.backbone_name = backbone_name
    param_config.use_pretrained_backbone = True
    param_config.neck_type = neck_type
    param_config.enable_global_semantic = enable_global_semantic
    param_config.enable_category_head = enable_category_head
    param_config.use_data_driven_priors = use_data_driven_priors
    param_config.assign_method = assign_method
    param_config.enable_temporal_model = False
    param_config.num_priors = 192
    param_config.refine_layers = 3
    param_config.sample_points = sample_points
    param_config.neck_out_channels = featuremap_out_channel
    param_config.featuremap_out_channel = featuremap_out_channel * 3
    param_config.fc_hidden_dim = fc_hidden_dim
    param_config.w_cls = 2.0
    param_config.w_geom = 4.0
    param_config.w_iou = 2.0
    param_config.simota_q = 10

    # fCLRNet-like CULane settings
    param_config.iou_loss_weight = iou_loss_weight
    param_config.cls_loss_weight = cls_loss_weight
    param_config.xyt_loss_weight = xyt_loss_weight
    param_config.seg_loss_weight = seg_loss_weight
    param_config.num_points = num_points
    param_config.max_lanes = max_lanes
    param_config.sample_y = sample_y
    param_config.test_parameters = test_parameters
    param_config.ori_img_w = ori_img_w
    param_config.ori_img_h = ori_img_h
    param_config.img_w = img_w
    param_config.img_h = img_h
    param_config.cut_height = cut_height
    param_config.img_norm = img_norm
    param_config.data_root = data_root
    param_config.ignore_label = ignore_label
    param_config.bg_weight = bg_weight
    param_config.num_classes = num_classes

    model = create_llanetv1_model(param_config)

    train = get_config("config/common/train.py").train

    epoch_per_iter = 88880 // batch_size + 1
    total_iter = epoch_per_iter * epochs
    train.max_iter = total_iter
    train.checkpointer.period = epoch_per_iter
    train.eval_period = epoch_per_iter
    train.output_dir = f"output/llanetv1/culane/{run_name}"
    train.amp.enabled = False

    param_config.output_dir = train.output_dir
    param_config.epoch_per_iter = epoch_per_iter

    optimizer = get_config("config/common/optim.py").AdamW
    optimizer.lr = 0.6e-3
    optimizer.weight_decay = 0.01

    lr_multiplier = L(CosineParamScheduler)(start_value=1.0, end_value=0.001)

    train_process = [
        L(GenerateLaneLine)(
            transforms=[
                dict(
                    name="Resize",
                    parameters=dict(size=dict(height=img_h, width=img_w)),
                    p=1.0,
                ),
                dict(name="HorizontalFlip", parameters=dict(p=1.0), p=0.5),
                dict(name="ChannelShuffle", parameters=dict(p=1.0), p=0.1),
                dict(
                    name="MultiplyAndAddToBrightness",
                    parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                    p=0.6,
                ),
                dict(
                    name="AddToHueAndSaturation",
                    parameters=dict(value=(-10, 10)),
                    p=0.7,
                ),
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
            ],
            cfg=param_config,
        ),
        L(ToTensor)(keys=["img", "lane_line", "seg"]),
    ]

    val_process = [
        L(GenerateLaneLine)(
            transforms=[
                dict(
                    name="Resize",
                    parameters=dict(size=dict(height=img_h, width=img_w)),
                    p=1.0,
                ),
            ],
            training=False,
            cfg=param_config,
        ),
        L(ToTensor)(keys=["img"]),
    ]

    dataloader = get_config("config/common/culane.py").dataloader
    dataloader.train.dataset.processes = train_process
    dataloader.train.dataset.data_root = data_root
    dataloader.train.dataset.cut_height = cut_height
    dataloader.train.dataset.cfg = param_config
    dataloader.train.total_batch_size = batch_size
    dataloader.train.num_workers = 4
    dataloader.train.persistent_workers = True
    dataloader.train.pin_memory = True
    dataloader.train.prefetch_factor = 4

    dataloader.test.dataset.processes = val_process
    dataloader.test.dataset.data_root = data_root
    dataloader.test.dataset.cut_height = cut_height
    dataloader.test.dataset.cfg = param_config
    dataloader.test.total_batch_size = batch_size
    dataloader.test.num_workers = 4

    # Evaluation config
    dataloader.evaluator.data_root = data_root
    dataloader.evaluator.output_basedir = f"{train.output_dir}/eval_results"
    dataloader.evaluator.cfg = param_config

    return model, dataloader, train, optimizer, lr_multiplier, param_config
