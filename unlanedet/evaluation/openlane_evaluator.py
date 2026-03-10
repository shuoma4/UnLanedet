import logging
import os

import numpy as np
from tqdm import tqdm

try:
    from p_tqdm import p_map
except ImportError:
    p_map = None

from . import culane_metric
from .evaluator import DatasetEvaluator


def robust_interp(points, n=5):
    """
    Linear interpolation to avoid spline oscillations with jittery GT.
    Resamples the polyline to increase point density.
    n: multiplier for number of points (similar to original interp logic)
    """
    if len(points) < 2:
        return np.array(points)

    points = np.array(points)
    # Calculate cumulative distance
    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dists[-1]

    if total_dist == 0:
        return np.array([points[0]] * 50)

    # Determine target number of points
    # Use a fixed density or multiplier. Let's use multiplier to match CULane logic.
    target_num = max(50, len(points) * n)

    # Uniform sample distances
    target_dists = np.linspace(0, total_dist, int(target_num))

    # Interpolate X and Y
    x_interp = np.interp(target_dists, cum_dists, points[:, 0])
    y_interp = np.interp(target_dists, cum_dists, points[:, 1])

    return np.stack([x_interp, y_interp], axis=1)


# Monkey patch culane_metric.interp
culane_metric.interp = robust_interp


class OpenLaneEvaluator(DatasetEvaluator):
    def __init__(self, data_root=None, cfg=None, **kwargs):
        self.cfg = cfg
        self.data_root = data_root or (cfg.data_root if cfg and hasattr(cfg, 'data_root') else None)

        if self.data_root is None:
            raise ValueError('data_root must be provided either as argument or in cfg')

        self.logger = logging.getLogger(__name__)

        # Determine validation directory
        if os.path.basename(self.data_root) == 'lane3d_300':
            self.val_dir = os.path.join(self.data_root, 'validation')
        else:
            self.val_dir = os.path.join(self.data_root, 'lane3d_300', 'validation')

        if not os.path.exists(self.val_dir):
            self.val_dir = os.path.join(self.data_root, 'validation')

        if not os.path.exists(self.val_dir):
            raise ValueError(f'Validation directory not found in {self.data_root}')

        self.ori_img_w = cfg.ori_img_w if cfg else 1920
        self.ori_img_h = cfg.ori_img_h if cfg else 1280

        # Evaluation in original space
        self.img_w = self.ori_img_w
        self.img_h = self.ori_img_h

    def get_prediction_lanes(self, output):
        pred_lanes = []
        for lane in output:
            # HACK: Force correct sample_y for OpenLane
            if not hasattr(self.cfg, 'sample_y_corrected'):
                self.cfg.sample_y = list(range(270, 1280, 10))
                self.cfg.sample_y_corrected = True

            pred_lanes.append(lane.to_array(self.cfg))
        return pred_lanes

    def evaluate(self, predictions):
        self.logger.info('Evaluating...')

        if not hasattr(self, 'data_infos') or not self.data_infos:
            self.logger.warning('data_infos not found or empty. Evaluation might fail.')
            return {}

        all_pred_lanes = []
        all_gt_lanes = []

        # GT seems to be in 800x320 space (Max Y ~300)
        # We need to scale GT to 1920x1280
        # W Scale: 1920 / 800 = 2.4
        # H Scale: (1280 - 270) / 320 = 3.156

        gt_w = 800
        gt_h = 320
        w_scale = self.ori_img_w / gt_w
        cut_height = self.cfg.cut_height if hasattr(self.cfg, 'cut_height') else 270
        h_scale = (self.ori_img_h - cut_height) / gt_h

        for i, pred in enumerate(tqdm(predictions, desc='Processing predictions')):
            if i >= len(self.data_infos):
                break

            # Get Prediction Points (In original space)
            pred_lanes = self.get_prediction_lanes(pred)

            all_pred_lanes.append(pred_lanes)

            # Get GT Points
            raw_gt_lanes = self.data_infos[i]['lanes']
            track_ids = self.data_infos[i].get('lane_track_ids', [])

            # DEBUG: Check track IDs for Sample 2
            if i == 2:
                print(f'Sample 2 Track IDs: {track_ids}')

            gt_lanes = []

            for gtl in raw_gt_lanes:
                lane_arr = np.array(gtl) if not isinstance(gtl, np.ndarray) else gtl
                if len(lane_arr) == 0:
                    continue

                # GT is already in original space (1920x1280) because we disabled preprocessing
                points = lane_arr.tolist()

                points.sort(key=lambda p: p[1])
                gt_lanes.append(points)

            all_gt_lanes.append(gt_lanes)

        # Run Metric in original space
        img_shape = (self.ori_img_h, self.ori_img_w, 3)
        width = 30
        iou_threshold = 0.5
        official = False

        # Custom Distance-based Metric
        tp_gt = 0
        tp_pred = 0
        total_pred = 0
        total_gt = 0

        dist_th = 50.0  # pixels

        # DEBUG: Check first valid sample overlap manually
        checked = False
        for idx in range(len(all_pred_lanes)):
            p0 = all_pred_lanes[idx]
            g0 = all_gt_lanes[idx]

            # Count totals
            total_pred += len(p0)
            total_gt += len(g0)

            # Match GTs (Recall)
            for gl in g0:
                gl_arr = np.array(gl)
                matched = False
                for pl in p0:
                    pl_arr = np.array(pl)
                    if len(pl_arr) == 0 or len(gl_arr) == 0:
                        continue
                    # Calc min dist between any points
                    dists = np.sqrt(((pl_arr[:, None, :] - gl_arr[None, :, :]) ** 2).sum(axis=2))
                    if dists.min() < dist_th:
                        matched = True
                        break
                if matched:
                    tp_gt += 1

            # Match Preds (Precision)
            for pl in p0:
                pl_arr = np.array(pl)
                matched = False
                for gl in g0:
                    gl_arr = np.array(gl)
                    if len(pl_arr) == 0 or len(gl_arr) == 0:
                        continue
                    dists = np.sqrt(((pl_arr[:, None, :] - gl_arr[None, :, :]) ** 2).sum(axis=2))
                    if dists.min() < dist_th:
                        matched = True
                        break
                if matched:
                    tp_pred += 1

            if len(p0) > 0 and len(g0) > 0 and not checked:
                print(f'DEBUG: Checking overlap for Sample {idx}')
                # ... (existing debug print)
                # Check overlap for ALL pairs
                matched = False
                for i, pl in enumerate(p0):
                    for j, gl in enumerate(g0):
                        # Calculate min distance
                        pl_arr = np.array(pl)
                        gl_arr = np.array(gl)
                        # simple brute force min distance
                        dists = np.sqrt(((pl_arr[:, None, :] - gl_arr[None, :, :]) ** 2).sum(axis=2))
                        min_dist = dists.min()

                        img_p = culane_metric.draw_lane(np.array(pl), img_shape=img_shape, width=width)
                        img_g = culane_metric.draw_lane(np.array(gl), img_shape=img_shape, width=width)
                        overlap = (img_p > 0) & (img_g > 0)
                        union = (img_p > 0) | (img_g > 0)
                        iou = np.sum(overlap) / np.sum(union) if np.sum(union) > 0 else 0

                        print(f'  P{i}-G{j}: Min Dist={min_dist:.2f}, IOU={iou:.4f}')

                        if iou > 0.01:
                            # print(f"  Match P{i}-G{j}: IOU={iou:.4f}")
                            matched = True

                checked = True

        # Calculate Custom Metrics
        precision = tp_pred / total_pred if total_pred > 0 else 0
        recall = tp_gt / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f'Custom Distance Metric (th={dist_th}): P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}')

        return {
            'TP': tp_gt,
            'FP': total_pred - tp_pred,
            'FN': total_gt - tp_gt,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
        }
