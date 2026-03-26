"""
一次性运行：curve 场景下某条序列上，对比
  - clrnet_linear_resnet34 (model_best)
  - resnet34_llanet_ablation_exp2 (model_best)

输出：output/analysis/curve/<segment 目录名>/
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

# OpenLane 场景列表名（会自动尝试 lane3d_1000/test/1000_curve.txt）
M.CFG_PATH = "config/llanetv1/openlane1000/category/clrnet_linear_resnet34.py"
M.MODEL2_CFG_PATH = "config/llanetv1/openlane1000/resnet34_llanet_ablation_exp2.py"
M.MODEL1_CKPT_PATH = (
    "output/llanetv1/openlane1000/category/clrnet_linear_resnet34/model_best.pth"
)
M.MODEL2_CKPT_PATH = (
    "output/llanetv1/openlane1000/resnet34_llanet_ablation_exp2/model_best.pth"
)

# 1000_curve.txt 中第一条序列（segment）
_SEGMENT = (
    "segment-10203656353524179475_7625_000_7645_000_with_camera_labels"
)
M.SEQUENCE_NAMES = ["curve"]
M.SEGMENT_PATH_CONTAINS = _SEGMENT
M.OUT_DIR = os.path.join("output", "analysis", "curve", _SEGMENT)
M.APPEND_SCENARIO_SUBDIR = False  # 结果直接落在 curve/<序列>/ 下

M.MAX_FRAMES_PER_SEQUENCE = 120
M.TOPK_FAILURES_TO_DRAW_PER_SEQUENCE = 16
M.ALSO_DRAW_RANDOM = 8

if __name__ == "__main__":
    M.main()
