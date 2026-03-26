"""
在 curve + 指定 segment 上生成论文式网格图（原图 | CLRNet-linear | Exp2 | GT），
输出与 run_curve_clrnet_linear_vs_ablation_exp2 同目录，文件名 qualitative_grid_compare.png。
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_PR_DIR = os.path.dirname(os.path.abspath(__file__))
if _PR_DIR not in sys.path:
    sys.path.insert(0, _PR_DIR)

import qualitative_failure_analysis_compare_models as M
import qualitative_comparison_grid as G

M.CFG_PATH = "config/llanetv1/openlane1000/category/clrnet_linear_resnet34.py"
M.MODEL2_CFG_PATH = "config/llanetv1/openlane1000/resnet34_llanet_ablation_exp2.py"
M.MODEL1_CKPT_PATH = (
    "output/llanetv1/openlane1000/category/clrnet_linear_resnet34/model_best.pth"
)
M.MODEL2_CKPT_PATH = (
    "output/llanetv1/openlane1000/resnet34_llanet_ablation_exp2/model_best.pth"
)

_SEGMENT = (
    "segment-10203656353524179475_7625_000_7645_000_with_camera_labels"
)
M.SEQUENCE_NAMES = ["curve"]
M.SEGMENT_PATH_CONTAINS = _SEGMENT
M.OUT_DIR = os.path.join("output", "analysis", "curve", _SEGMENT)
M.APPEND_SCENARIO_SUBDIR = False

G.NUM_GRID_COLS = 8
G.CELL_W = 480
G.CELL_H = 270
G.MODEL1_DISPLAY_ZH = "CLRNet-linear"
G.MODEL2_DISPLAY_ZH = "LLANet Exp2"

if __name__ == "__main__":
    G.main()
