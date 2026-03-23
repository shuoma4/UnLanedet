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
    pred_data, gt_data, width=30, iou_threshold=0.5, img_shape=(1280, 1920)
):
    pred_pts, pred_cats = pred_data
    gt_pts, gt_cats = gt_data

    if len(pred_pts) == 0:
        return 0, 0, len(gt_pts), [], []
    if len(gt_pts) == 0:
        return 0, len(pred_pts), 0, [], []

    interp_pred = np.array([culane_metric.interp(pred_lane, n=5) for pred_lane in pred_pts], dtype=object)
    interp_anno = np.array([culane_metric.interp(anno_lane, n=5) for anno_lane in gt_pts], dtype=object)

    ious = culane_metric.discrete_cross_iou(interp_pred, interp_anno, width=width, img_shape=(img_shape[0], img_shape[1]))
    
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(1 - ious)

    tp_mask = ious[row_ind, col_ind] > iou_threshold
    tp = int(tp_mask.sum())
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp

    matched_pred_cats = []
    matched_gt_cats = []
    if pred_cats is not None and gt_cats is not None:
        for i, is_tp in enumerate(tp_mask):
            if is_tp:
                r = row_ind[i]
                c = col_ind[i]
                matched_pred_cats.append(pred_cats[r])
                matched_gt_cats.append(gt_cats[c])

    return tp, fp, fn, matched_pred_cats, matched_gt_cats


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

            pred_pts = []
            pred_cats = []
            for l in pred_lanes:
                p = self._pred_lane_to_pts(l)
                if len(p) >= 2:
                    pred_pts.append(p)
                    pred_cats.append(l.metadata.get('category_id', -1))

            gt_pts = []
            gt_cats = []
            raw_gts = self.data_infos[i]["lanes"]
            raw_gt_cats = self.data_infos[i].get("lane_categories", [-1] * len(raw_gts))

            for g, c in zip(raw_gts, raw_gt_cats):
                p = self._gt_lane_to_pts(g)
                if len(p) >= 2:
                    gt_pts.append(p)
                    gt_cats.append(c)

            pred_collection.append((pred_pts, pred_cats))
            gt_collection.append((gt_pts, gt_cats))

        import multiprocessing
        num_workers = min(8, multiprocessing.cpu_count() or 4)
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

        # Use spawn context to prevent PyTorch deadlocks during fork when CUDA is initialized
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(num_workers) as p:
            # starmap will automatically unpack the zipped tuple into the two arguments
            results = p.starmap(
                partial_func, 
                zip(pred_collection, gt_collection), 
                chunksize=100
            )

        total_tp = sum(res[0] for res in results)
        total_fp = sum(res[1] for res in results)
        total_fn = sum(res[2] for res in results)

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

        all_pred_cats = []
        all_gt_cats = []
        for res in results:
            if len(res) == 5:
                all_pred_cats.extend(res[3])
                all_gt_cats.extend(res[4])

        if len(all_pred_cats) > 0 and len(all_gt_cats) > 0 and any(c != -1 for c in all_pred_cats):
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            def get_cls_metrics(gt_cats, pred_cats):
                if len(gt_cats) > 0 and len(pred_cats) > 0 and any(c != -1 for c in pred_cats):
                    p_mac = float(precision_score(gt_cats, pred_cats, average="macro", zero_division=0))
                    r_mac = float(recall_score(gt_cats, pred_cats, average="macro", zero_division=0))
                    f1_mac = float(f1_score(gt_cats, pred_cats, average="macro", zero_division=0))
                    p_wei = float(precision_score(gt_cats, pred_cats, average="weighted", zero_division=0))
                    r_wei = float(recall_score(gt_cats, pred_cats, average="weighted", zero_division=0))
                    f1_wei = float(f1_score(gt_cats, pred_cats, average="weighted", zero_division=0))
                    return p_mac, r_mac, f1_mac, p_wei, r_wei, f1_wei
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            cat_p, cat_r, cat_f1, p_wei, r_wei, f1_wei = get_cls_metrics(all_gt_cats, all_pred_cats)
            cm = confusion_matrix(all_gt_cats, all_pred_cats)
            
            result["Cat_Precision_Macro"] = cat_p
            result["Cat_Recall_Macro"] = cat_r
            result["Cat_F1_Macro"] = cat_f1
            result["Cat_Precision_Weighted"] = p_wei
            result["Cat_Recall_Weighted"] = r_wei
            result["Cat_F1_Weighted"] = f1_wei
            result["Confusion_Matrix"] = cm.tolist()
            
        import glob
        # Note: here we assume the current cfg has data_root
        data_root = getattr(self.cfg, "data_root", "")
        if data_root and "lane3d" in data_root:
            test_dir = osp.join(data_root, "test")
            scenario_files = glob.glob(osp.join(test_dir, "*.txt"))
            self.logger.info(f"Computing Sub-scenarios... Found {len(scenario_files)} lists")
            for s_file in scenario_files:
                scenario_name = osp.basename(s_file).replace('.txt', '').replace('1000_', '')
                with open(s_file, 'r') as f:
                    lines = set([ln.strip() for ln in f.readlines() if ln.strip()])
                
                s_tp = 0
                s_fp = 0
                s_fn = 0
                s_pred_cats = []
                s_gt_cats = []
                
                for i, res_item in enumerate(results):
                    img_path = self.data_infos[i].get("img_path", "")
                    parts = img_path.split('/')
                    if len(parts) >= 3:
                        rel_path = f"{parts[-3]}/{parts[-2]}/{parts[-1]}"
                    else:
                        rel_path = img_path
                        
                    matched = False
                    if rel_path in lines:
                        matched = True
                    else:
                        for idx_line in lines:
                            if idx_line in img_path:
                                matched = True
                                break
                    if matched:
                        s_tp += res_item[0]
                        s_fp += res_item[1]
                        s_fn += res_item[2]
                        if len(res_item) == 5:
                            s_pred_cats.extend(res_item[3])
                            s_gt_cats.extend(res_item[4])
                
                if s_tp + s_fp + s_fn > 0:
                    s_precision = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
                    s_recall = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
                    s_f1 = 2 * s_precision * s_recall / (s_precision + s_recall) if (s_precision + s_recall) > 0 else 0.0
                    result[f"{scenario_name}_F1"] = s_f1
                    if len(s_gt_cats) > 0 and 'get_cls_metrics' in locals():
                        _, _, f1_mac, _, _, f1_wei = get_cls_metrics(s_gt_cats, s_pred_cats)
                        result[f"{scenario_name}_Cat_F1_Macro"] = f1_mac
                        result[f"{scenario_name}_Cat_F1_Weighted"] = f1_wei

        self.logger.info("=== OpenLane Evaluation Results ===")
        for k, v in result.items():
            if k == "Confusion_Matrix":
                self.logger.info("  Confusion Matrix:")
                for row in v:
                    self.logger.info(f"    {row}")
            else:
                self.logger.info(
                    f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
                )

        if "Confusion_Matrix" in result:
            del result["Confusion_Matrix"]

        if self.output_dir:
            save_path = osp.join(self.output_dir, "openlane_eval_results.json")
            with open(save_path, "w") as f:
                json.dump(result, f, indent=2)
            self.logger.info(f"结果已保存至 {save_path}")

        return result
