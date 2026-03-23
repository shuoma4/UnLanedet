import argparse
import os
import glob
import time
import json
import logging
from functools import partial
import pandas as pd
import numpy as np
from tqdm import tqdm
from p_tqdm import p_map, t_map

import torch
import torch.nn as nn
from scipy.interpolate import splprep, splev

from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import default_setup
from unlanedet.engine.defaults import create_ddp_model
from unlanedet.checkpoint import Checkpointer
from unlanedet.evaluation import inference_on_dataset
from unlanedet.evaluation.evaluator import DatasetEvaluator

logger = logging.getLogger("unlanedet_pr")

class InterceptEvaluator(DatasetEvaluator):
    def __init__(self, target_evaluator):
        self.target = target_evaluator
        self.predictions = None
        self.data_infos = None

    def evaluate(self, predictions):
        self.predictions = predictions
        return {}

# ----------------- Patched CULane Metric -----------------
def culane_metric_patched(pred, anno, width=30, iou_threshold=0.5, official=True, img_shape=(590, 1640)):
    from unlanedet.evaluation.culane_metric import interp, discrete_cross_iou, continuous_cross_iou
    from scipy.optimize import linear_sum_assignment
    if len(pred) == 0:
        return 0, 0, len(anno), [], []
    if len(anno) == 0:
        return 0, len(pred), 0, [], []
    interp_pred = np.array([interp(pred_lane, n=5) for pred_lane in pred], dtype=object)
    interp_anno = np.array([interp(anno_lane, n=5) for anno_lane in anno], dtype=object)

    if official:
        ious = discrete_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)
    else:
        ious = continuous_cross_iou(interp_pred, interp_anno, width=width, img_shape=img_shape)

    row_ind, col_ind = linear_sum_assignment(1 - ious)
    tp = int((ious[row_ind, col_ind] > iou_threshold).sum())
    fp = len(pred) - tp
    fn = len(anno) - tp
    return tp, fp, fn, [], []

def eval_predictions_patched(pred_dir, anno_dir, list_path, iou_threshold=0.5, width=30, official=True, sequential=False):
    from unlanedet.evaluation.culane_metric import load_culane_data
    predictions = load_culane_data(pred_dir, list_path)
    annotations = load_culane_data(anno_dir, list_path)
    img_shape = (590, 1640, 3)
    
    if sequential:
        results = t_map(partial(culane_metric_patched, width=width, iou_threshold=iou_threshold, official=official, img_shape=img_shape), predictions, annotations)
    else:
        results = p_map(partial(culane_metric_patched, width=width, iou_threshold=iou_threshold, official=official, img_shape=img_shape), predictions, annotations)
    
    total_tp = sum(res[0] for res in results)
    total_fp = sum(res[1] for res in results)
    total_fn = sum(res[2] for res in results)
    
    if total_tp == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = float(total_tp) / (total_tp + total_fp)
        recall = float(total_tp) / (total_tp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)
        
    return {'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'Precision': precision, 'Recall': recall, 'F1': f1}

# ----------------- Main Loop -----------------

def main():
    parser = argparse.ArgumentParser(description="Generate PR Curve over IoU and Width")
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('ckpt', help='The path of checkpoint')
    parser.add_argument('--mode', choices=['iou', 'width', 'both'], default='both', help='Variable to iterate')
    args = parser.parse_args()

    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, [])
    default_setup(cfg, args)

    device = cfg.train.device
    model = instantiate(cfg.model)
    model.to(device)
    model = create_ddp_model(model)
    model.eval()
    Checkpointer(model).load(args.ckpt)

    print("=" * 40)
    print(f"Loading Evaluator for inference...")
    
    if "evaluator" not in cfg.dataloader:
        print("No evaluator in cfg.dataloader! Exiting.")
        return
        
    orig_evaluator = instantiate(cfg.dataloader.evaluator)
    evaluator_target = cfg.dataloader.evaluator.get('_target_', None)
    eval_name = getattr(orig_evaluator, '__class__', None)
    if eval_name:
        eval_name = eval_name.__name__
    elif evaluator_target:
        eval_name = getattr(evaluator_target, '__name__', '')
        
    is_culane = 'CULane' in eval_name
    is_openlane = 'OpenLane' in eval_name
    
    print(f"Detected Evaluator: {eval_name}")
    
    # Give CULane output dir manually if it is CULane, because CULane evaluator MUST write predictions to disk!
    interceptor = InterceptEvaluator(orig_evaluator)
    if is_culane:
        # CULane evaluator needs to dump. So we run the orig evaluator first
        print("Running inference and CULane line dumping...")
        inference_on_dataset(model, instantiate(cfg.dataloader.test), orig_evaluator)
    else:
        # OpenLane evaluator does not need to write to disk. We use interceptor
        test_dataloader = instantiate(cfg.dataloader.test)
        interceptor.data_infos = test_dataloader.dataset.data_infos if hasattr(test_dataloader.dataset, 'data_infos') else None
        
        # Override evaluate to just return the results dict and not throw away
        original_evaluate = orig_evaluator.evaluate
        def mock_evaluate(preds):
            interceptor.predictions = preds
            return {}
        orig_evaluator.evaluate = mock_evaluate
        print("Running OpenLane inference...")
        inference_on_dataset(model, test_dataloader, orig_evaluator)
        # Restore
        orig_evaluator.evaluate = original_evaluate

    settings = []
    if args.mode in ['iou', 'both']:
         for iou in np.arange(0.3, 0.82, 0.02):
             settings.append({"iou_threshold": round(iou, 2), "width": 30})
    if args.mode in ['width', 'both']:
         for w in range(10, 31, 1):
             settings.append({"iou_threshold": 0.5, "width": w})
             
    # Deduplicate if (0.5, 30) is repeated
    seen = set()
    unique_settings = []
    for s in settings:
         tup = (s["iou_threshold"], s["width"])
         if tup not in seen:
             seen.add(tup)
             unique_settings.append(s)
    settings = unique_settings

    train_out_dir = cfg.train.output_dir
    test_res_dir = os.path.join(train_out_dir, "pr_curves")
    os.makedirs(test_res_dir, exist_ok=True)
    json_path = os.path.join(test_res_dir, "pr_metrics.json")
    excel_path = os.path.join(test_res_dir, "pr_metrics.xlsx")
    
    all_results = []
    completed_settings = set()
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                all_results = json.load(f)
            for res in all_results:
                # Be careful with floating point matching if types differ slightly
                completed_settings.add((round(float(res.get("IoU_Threshold", 0)), 2), int(res.get("Width", 0))))
            print(f"Loaded {len(all_results)} existing results. Resuming...")
        except Exception as e:
            print(f"Could not load existing {json_path}: {e}")
    
    for s in tqdm(settings, "Processing configurations"):
        iou = round(float(s["iou_threshold"]), 2)
        w = int(s["width"])
        if (iou, w) in completed_settings:
            print(f"Skipping already processed IoU={iou}, Width={w}")
            continue
            
        res_row = {"IoU_Threshold": iou, "Width": w}
        
        if is_culane:
            data_root = cfg.dataloader.evaluator.data_root
            output_basedir = cfg.dataloader.evaluator.output_basedir
            # Overall
            overall_list = os.path.join(data_root, 'list', 'test.txt')
            overall_res = eval_predictions_patched(
                output_basedir, data_root, overall_list, 
                iou_threshold=iou, width=w, official=True
            )
            res_row["Total_F1"] = overall_res["F1"]
            res_row["Total_Precision"] = overall_res["Precision"]
            res_row["Total_Recall"] = overall_res["Recall"]
            
            # Sub-scenarios
            split_dir = os.path.join(data_root, 'list', 'test_split')
            splits = glob.glob(os.path.join(split_dir, '*.txt'))
            for split_file in splits:
                split_name = os.path.basename(split_file).replace('.txt', '')
                res = eval_predictions_patched(
                    output_basedir, data_root, split_file, 
                    iou_threshold=iou, width=w, official=True
                )
                if "cross" in split_name.lower():
                    res_row[split_name + '_FP'] = res.get('FP', 0)
                else:
                    res_row[split_name + '_F1'] = res.get('F1', 0)
                    
        elif is_openlane:
            # We already have interceptor.predictions
            # Or use orig_evaluator if it was properly stored
            cur_preds = interceptor.predictions
            orig_evaluator.iou_threshold = iou
            orig_evaluator.width = w
            
            # Avoid the logger spam in openlane_evaluator during loop calculation
            orig_logger_level = orig_evaluator.logger.level
            orig_evaluator.logger.setLevel(logging.WARNING)
            res = orig_evaluator.evaluate(cur_preds)
            orig_evaluator.logger.setLevel(orig_logger_level)
            
            res_row["Total_F1"] = res.get("F1", 0)
            res_row["Total_Precision"] = res.get("Precision", 0)
            res_row["Total_Recall"] = res.get("Recall", 0)
            res_row["Cat_F1_Macro"] = res.get("Cat_F1_Macro", 0)
            res_row["Cat_F1_Weighted"] = res.get("Cat_F1_Weighted", 0)
            
            # Sub scenarios. Ensure OpenLaneEvaluator has evaluated sub-scenarios!
            for scenario in ["updown", "curve", "extreme_weather", "night", "intersection", "merge_split"]:
                if f"{scenario}_F1" in res:
                    res_row[f"{scenario}_F1"] = res.get(f"{scenario}_F1", 0)
                    res_row[f"{scenario}_Cat_F1_Macro"] = res.get(f"{scenario}_Cat_F1_Macro", 0)
                    res_row[f"{scenario}_Cat_F1_Weighted"] = res.get(f"{scenario}_Cat_F1_Weighted", 0)
                
        all_results.append(res_row)

        df = pd.DataFrame(all_results)
        excel_path = os.path.join(test_res_dir, "pr_metrics.xlsx")
        json_path = os.path.join(test_res_dir, "pr_metrics.json")
        
        df.to_excel(excel_path, index=False)
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Incremental results saved to {json_path}")

    print("=" * 40)
    print(f"Evaluation complete! Final results saved to:\n  {excel_path}\n  {json_path}")


if __name__ == '__main__':
    main()
