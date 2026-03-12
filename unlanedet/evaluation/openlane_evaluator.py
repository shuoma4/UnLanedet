import json
import logging
import os
import os.path as osp
from functools import partial

import numpy as np
from p_tqdm import p_map
from tqdm import tqdm

from . import culane_metric
from .evaluator import DatasetEvaluator


def _linear_interp(points, n=5):
    """用线性插值替代 spline，避免 OpenLane GT 点分布不均时 splprep 崩溃。"""
    if len(points) < 2:
        return np.array(points)
    pts = np.array(points)
    dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum_d = np.concatenate(([0], np.cumsum(dists)))
    total = cum_d[-1]
    if total == 0:
        return np.tile(pts[0], (50, 1))
    target_n = max(50, len(pts) * n)
    td = np.linspace(0, total, int(target_n))
    xi = np.interp(td, cum_d, pts[:, 0])
    yi = np.interp(td, cum_d, pts[:, 1])
    return np.stack([xi, yi], axis=1)


# 替换 culane_metric 中的 interp，避免 OpenLane GT 不规则点导致的 spline 失败
culane_metric.interp = _linear_interp


def _openlane_metric_per_sample(
    pred_pts, gt_pts, width=30, iou_threshold=0.5, img_shape=(1280, 1920)
):
    tp, fp, fn, _, _ = culane_metric.culane_metric(
        pred_pts,
        gt_pts,
        width=width,
        iou_threshold=iou_threshold,
        official=True,  # Back to discrete math but optimized with BBox Crops
        img_shape=(img_shape[0], img_shape[1]), 
    )
    return tp, fp, fn


class OpenLaneEvaluator(DatasetEvaluator):
    """
    OpenLane 数据集的 IoU-based 评估器（与 CULane 同一套 IoU 逻辑）。

    流程：
      1. 预测: Lane.to_array(cfg) -> (N, 2) 原始像素坐标 [x, y]
         要求 cfg.sample_y 为原始图像空间的 y 采样序列（如 range(1279, 270, -8)）
      2. 真值: data_infos[i]['lanes'] 中的 numpy 数组，已在原始图像坐标系 (u, v)
      3. 在 (ori_img_h × ori_img_w) 画布上用宽度 width 绘制车道线，计算 IoU

    self.data_infos 由 inference_on_dataset 从 dataset.data_infos 注入，无需手动传入。
    """

    def __init__(
        self,
        cfg=None,
        output_dir=None,
        iou_threshold=0.5,
        width=30,
        metric="F1",
        **kwargs,  # 兼容旧 config 中的多余参数（如 evaluate_bin_path）
    ):
        self.cfg = cfg
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        self.width = width
        self.metric = metric
        self.logger = logging.getLogger(__name__)
        self.data_infos = None  # 由 inference_on_dataset 注入

        self.ori_img_w = getattr(cfg, "ori_img_w", 1920) if cfg else 1920
        self.ori_img_h = getattr(cfg, "ori_img_h", 1280) if cfg else 1280

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _pred_lane_to_pts(self, lane):
        """Lane 对象 -> list of [x, y]（原始像素空间）"""
        arr = lane.to_array(
            self.cfg
        )  # (N, 2), x in [0, ori_img_w), y in [cut_h, ori_img_h)
        return arr.tolist() if len(arr) >= 2 else []

    def _gt_lane_to_pts(self, lane):
        """GT numpy 数组 (N,2) [u,v] -> list of [x, y]"""
        arr = np.array(lane) if not isinstance(lane, np.ndarray) else lane
        return arr.tolist() if len(arr) >= 2 else []

    def evaluate(self, predictions):
        self.logger.info("OpenLane evaluation start...")

        if not self.data_infos:
            self.logger.warning("data_infos 为空，无法评估。")
            return {}

        img_shape = (self.ori_img_h, self.ori_img_w, 3)
        pred_collection = []
        gt_collection = []

        for i, pred_lanes in enumerate(tqdm(predictions, desc="Computing IoU")):
            if i >= len(self.data_infos):
                break

            pred_pts = [self._pred_lane_to_pts(l) for l in pred_lanes]
            pred_pts = [p for p in pred_pts if len(p) >= 2]

            gt_pts = [self._gt_lane_to_pts(l) for l in self.data_infos[i]["lanes"]]
            gt_pts = [g for g in gt_pts if len(g) >= 2]

            pred_collection.append(pred_pts)
            gt_collection.append(gt_pts)

        import multiprocessing
        num_workers = min(32, multiprocessing.cpu_count() or 8)
        self.logger.info(f"Computing OpenLane IoU with {num_workers} workers...")
        
        # Prevent Thread Contentions in OpenCV and Numpy
        import cv2
        import multiprocessing
        import os
        cv2.setNumThreads(0)
        os.environ["OMP_NUM_THREADS"] = "1"

        partial_func = partial(
            _openlane_metric_per_sample,
            width=self.width,
            iou_threshold=self.iou_threshold,
            img_shape=img_shape,
        )

        with multiprocessing.Pool(num_workers) as p:
            # starmap will automatically unpack the zipped tuple into the two arguments
            results = p.starmap(
                partial_func, 
                zip(pred_collection, gt_collection), 
                chunksize=100
            )

        total_tp = sum(tp for tp, _, _ in results)
        total_fp = sum(fp for _, fp, _ in results)
        total_fn = sum(fn for _, _, fn in results)

        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        result = {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }

        self.logger.info("=== OpenLane Evaluation Results ===")
        for k, v in result.items():
            self.logger.info(
                f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
            )

        if self.output_dir:
            save_path = osp.join(self.output_dir, "openlane_eval_results.json")
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2)
            self.logger.info(f"结果已保存至 {save_path}")

        return result
