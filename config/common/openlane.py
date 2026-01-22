from omegaconf import OmegaConf
from unlanedet.data.openlane import OpenLane
from unlanedet.config import LazyCall as L
from unlanedet.data.build import build_batch_data_loader
from unlanedet.evaluation import OpenLaneEvaluator

ori_img_h = 1280
ori_img_w = 1920
img_h = 320
img_w = 800
cut_height = 270

dataloader = OmegaConf.create()

dataloader.train = L(build_batch_data_loader)(
    dataset=L(OpenLane)(
        data_root="/data1/lxy_log/workspace/ms/OpenLane/dataset/raw",
        split="train",
        cut_height=cut_height,
        processes=None,
    ),
    total_batch_size=256,
    num_workers=8,
    shuffle=True,
)

dataloader.test = L(build_batch_data_loader)(
    dataset=L(OpenLane)(
        data_root="/data1/lxy_log/workspace/ms/OpenLane/dataset/raw",
        split="val",
        cut_height=cut_height,
        processes=None,
    ),
    total_batch_size=256,
    num_workers=4,
    drop_last=False,
    shuffle=False,
)

dataloader.evaluator = L(OpenLaneEvaluator)(
    iou_threshold=0.5, width=30, metric="detection/f1"
)
