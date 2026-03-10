import json
import logging
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from . import culane_metric
from .evaluator import DatasetEvaluator


def _linear_interp(points, n=5):
    """
    线性插值替代样条插值，避免GT点稀疏/不规则时的振荡问题。
    n: 插值密度倍数（与原 culane_metric.interp 接口一致）
    """
    points = np.array(points, dtype=np.float64)
    if len(points) < 2:
        return points

    diffs = np.diff(points, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_lens = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_len = cum_lens[-1]

    if total_len < 1e-6:
        return np.tile(points[0], (50, 1))

    target_num = max(50, len(points) * n)
    target_lens = np.linspace(0.0, total_len, int(target_num))

    xs = np.interp(target_lens, cum_lens, points[:, 0])
    ys = np.interp(target_lens, cum_lens, points[:, 1])
    return np.stack([xs, ys], axis=1)


# 用线性插值替换原 culane_metric 中的三次样条，避免稀疏GT点导致的插值失败
culane_metric.interp = _linear_interp


class OpenLaneEvaluator(DatasetEvaluator):
    """
    OpenLane 2D 车道线评估器。

    评估流程：
        1. 将预测 Lane 对象通过 lane.to_array(cfg) 转换为原始图像坐标系下的点集。
        2. GT 车道线直接使用 data_infos[i]['lanes']（原始1920×1280坐标系）。
        3. 对每张图像用 culane_metric（IoU + 匈牙利匹配）计算 TP/FP/FN。
        4. 汇总全局 Precision / Recall / F1。

    参数：
        cfg          : param_config，需含 img_w, img_h, cut_height, ori_img_w, ori_img_h, sample_y
        output_dir   : 评估结果保存路径
        iou_threshold: IoU 阈值，默认 0.5
        width        : 绘制车道线宽度（像素，原始图像空间），默认 30
        metric       : 'F1'（目前只支持F1）
    """

    def __init__(
        self,
        cfg=None,
        output_dir=None,
        iou_threshold: float = 0.5,
        width: int = 30,
        metric: str = 'F1',
        **kwargs,
    ):
        self.cfg = cfg
        self.output_dir = output_dir
        self.iou_threshold = iou_threshold
        self.width = width
        self.metric = metric
        self.logger = logging.getLogger(__name__)

        # 这两个属性由 inference_on_dataset 在评估前注入
        self.data_infos = None   # list of dict，含 'lanes' 字段（原始坐标系）

    # ------------------------------------------------------------------
    # 预测车道线转换
    # ------------------------------------------------------------------
    def _pred_to_points(self, lane):
        """
        将单条预测 Lane 对象转为原始图像坐标系下的点列表 [(x, y), ...]。
        lane.to_array(cfg) 已经在 ori_img_h×ori_img_w 空间输出坐标。
        """
        arr = lane.to_array(self.cfg)   # shape (N, 2)，坐标在原始图像空间
        if arr is None or len(arr) == 0:
            return []
        # 过滤掉坐标超出图像范围的点（to_array 可能输出边界外的点）
        ori_h = self.cfg.ori_img_h
        ori_w = self.cfg.ori_img_w
        mask = (arr[:, 0] >= 0) & (arr[:, 0] < ori_w) & \
               (arr[:, 1] >= 0) & (arr[:, 1] < ori_h)
        arr = arr[mask]
        if len(arr) < 2:
            return []
        return arr.tolist()

    # ------------------------------------------------------------------
    # 单图 IoU 评估（复用 culane_metric）
    # ------------------------------------------------------------------
    def _eval_one_image(self, pred_lanes, gt_lanes):
        """
        对单张图像计算 TP/FP/FN。

        Args:
            pred_lanes : list of list of (x, y)，原始坐标系
            gt_lanes   : list of list of (x, y)，原始坐标系
        Returns:
            tp, fp, fn (int)
        """
        img_shape = (self.cfg.ori_img_h, self.cfg.ori_img_w, 3)

        # 过滤点数不足的车道线
        pred_lanes = [l for l in pred_lanes if len(l) >= 2]
        gt_lanes   = [l for l in gt_lanes   if len(l) >= 2]

        if len(pred_lanes) == 0 and len(gt_lanes) == 0:
            return 0, 0, 0
        if len(pred_lanes) == 0:
            return 0, 0, len(gt_lanes)
        if len(gt_lanes) == 0:
            return 0, len(pred_lanes), 0

        tp, fp, fn, _, _ = culane_metric.culane_metric(
            pred=pred_lanes,
            anno=gt_lanes,
            width=self.width,
            iou_threshold=self.iou_threshold,
            official=True,
            img_shape=img_shape,
        )
        return tp, fp, fn

    # ------------------------------------------------------------------
    # 主评估接口
    # ------------------------------------------------------------------
    def evaluate(self, predictions):
        """
        Args:
            predictions: list，每个元素是一张图对应的预测车道线列表
                         (list of Lane objects，由模型 get_lanes() 输出)
        Returns:
            dict: {'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'}
        """
        self.logger.info('OpenLaneEvaluator: 开始评估...')

        if self.data_infos is None or len(self.data_infos) == 0:
            self.logger.warning('data_infos 为空，跳过评估')
            return {}

        total_tp = 0
        total_fp = 0
        total_fn = 0

        n_imgs = min(len(predictions), len(self.data_infos))

        for i in tqdm(range(n_imgs), desc='OpenLane 评估'):
            # ---- 预测车道线 ----
            pred_raw = predictions[i]   # list of Lane objects
            pred_lanes = []
            for lane in pred_raw:
                pts = self._pred_to_points(lane)
                if len(pts) >= 2:
                    pred_lanes.append(pts)

            # ---- GT 车道线（原始坐标系，v >= cut_height）----
            raw_gt = self.data_infos[i].get('lanes', [])
            gt_lanes = []
            for lane_arr in raw_gt:
                arr = np.array(lane_arr) if not isinstance(lane_arr, np.ndarray) else lane_arr
                if len(arr) < 2:
                    continue
                # lanes 中存储的是 (u, v) 形式，已过滤 v < cut_height
                pts = arr[:, :2].tolist()   # [(x, y), ...]
                pts.sort(key=lambda p: p[1])
                gt_lanes.append(pts)

            tp, fp, fn = self._eval_one_image(pred_lanes, gt_lanes)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # ---- 全局指标 ----
        if total_tp == 0:
            precision = recall = f1 = 0.0
        else:
            precision = total_tp / (total_tp + total_fp)
            recall    = total_tp / (total_tp + total_fn)
            f1        = 2 * precision * recall / (precision + recall)

        results = {
            'TP':        total_tp,
            'FP':        total_fp,
            'FN':        total_fn,
            'Precision': precision,
            'Recall':    recall,
            'F1':        f1,
        }

        self.logger.info(
            f'评估完成: TP={total_tp}, FP={total_fp}, FN={total_fn} | '
            f'P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}'
        )

        # ---- 保存结果 ----
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            save_path = os.path.join(self.output_dir, 'openlane_eval_results.json')
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f'评估结果已保存至: {save_path}')

        return results
