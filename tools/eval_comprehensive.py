import os
import glob
import argparse
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import torch
import logging
import pandas as pd
import json
from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import default_setup
from unlanedet.engine.defaults import create_ddp_model
from unlanedet.evaluation import inference_on_dataset

logger = logging.getLogger("unlanedet")

def get_model_size_and_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    buffer_num = sum(b.numel() for b in model.buffers())
    total_params = param_num + buffer_num

    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return total_params, total_size_mb

def test_speed(cfg, model, device):
    print("=" * 40)
    print("Evaluating Inference Speed on RTX 4090 (or current GPU)...")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    repeat_time = 2000

    input_tensor = torch.zeros((1, 3, cfg.param_config.img_h, cfg.param_config.img_w), device=device)
    data = {'img': input_tensor}

    for _ in range(200):
        with torch.no_grad():
            _ = model(data)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in tqdm(range(repeat_time)):
        with torch.no_grad():
            _ = model(data)

    torch.cuda.synchronize()
    end = time.perf_counter()
    fps = 1 / ((end - start) / repeat_time)
    print(f"Inference Speed: {fps:.2f} FPS")
    print("=" * 40)
    return fps

def evaluate_metrics(cfg, model):
    print("=" * 40)
    print("Evaluating Metrics...")
    if "evaluator" in cfg.dataloader:
        evaluator = instantiate(cfg.dataloader.evaluator)
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            evaluator,
        )
        
        evaluator_target = cfg.dataloader.evaluator.get('_target_', None)
        is_culane = False
        if evaluator_target is not None and getattr(evaluator_target, '__name__', '') == 'CULaneEvaluator':
            is_culane = True

        if is_culane:
            from unlanedet.evaluation import culane_metric
            data_root = cfg.dataloader.evaluator.data_root
            output_basedir = cfg.dataloader.evaluator.output_basedir
            split_dir = os.path.join(data_root, 'list', 'test_split')
            splits = glob.glob(os.path.join(split_dir, '*.txt'))
            print("Computing CULane 9 sub-scenarios...")
            for split_file in splits:
                split_name = os.path.basename(split_file).replace('.txt', '')
                try:
                    res = culane_metric.eval_predictions(
                        output_basedir, data_root, split_file, official=True
                    )
                    # For cross scenario (often split number 7 or named 'cross'), we report FP
                    if "cross" in split_name.lower():
                        ret[split_name + '_FP'] = res.get('FP', 0)
                    else:
                        ret[split_name + '_F1'] = res.get('F1', 0)
                except Exception as e:
                    print(f"Error computing metric for {split_name}: {e}")
        return ret
    else:
        print("No evaluator found in config.")
        return {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Script")
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('ckpt', help='The path of checkpoint')
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

    train_out_dir = cfg.train.output_dir
    test_res_dir = os.path.join(train_out_dir, "test_res")
    os.makedirs(test_res_dir, exist_ok=True)

    b_name = "Unknown"
    try:
        if hasattr(model, 'module'):
            b_name = type(model.module.backbone).__name__
        else:
            b_name = type(model.backbone).__name__
            
        if b_name == "ResNetWrapper" or "ResNet" in b_name:
            if hasattr(cfg.model.backbone, "resnet"):
                b_name = cfg.model.backbone.resnet
    except Exception as e:
        logger.warning(f"Failed to infer backbone: {e}")
        b_name = "Unknown"
    
    total_params, total_size_mb = get_model_size_and_params(model)
    fps = test_speed(cfg, model, device)
    metrics = evaluate_metrics(cfg, model)

    res_dict = {
        "Backbone": b_name,
        "Params (M)": round(total_params / 1e6, 2),
        "Size (MB)": round(total_size_mb, 2),
        "FPS": round(fps, 2),
    }

    if "Fmeasure" in metrics:
        res_dict["Total F1"] = metrics["Fmeasure"]
    if "F1" in metrics:
        res_dict["Total F1"] = metrics["F1"]

    for k, v in metrics.items():
        if isinstance(v, float) or isinstance(v, int):
            res_dict[k] = v
        elif k == "Confusion_Matrix":
            res_dict["Confusion_Matrix_JSON"] = json.dumps(v)

    df = pd.DataFrame([res_dict])
    excel_path = os.path.join(test_res_dir, "evaluation_results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Metrics saved to {excel_path}")
