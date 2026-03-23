import os
import glob
from unlanedet.config import LazyConfig, instantiate
from unlanedet.engine import default_setup
from tools.eval_pr_curve import eval_predictions_patched

args = type('Args', (), {'config': 'config/clrnet/resnet34_culane.py'})()
cfg = LazyConfig.load(args.config)

data_root = cfg.dataloader.evaluator.data_root
output_basedir = cfg.dataloader.evaluator.output_basedir

overall_list = os.path.join(data_root, 'list', 'test.txt')
print(f"overall_list: {overall_list}")
overall_res = eval_predictions_patched(
    output_basedir, data_root, overall_list, 
    iou_threshold=0.5, width=30, official=True
)
print(overall_res)
