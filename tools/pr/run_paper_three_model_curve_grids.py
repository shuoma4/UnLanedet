"""
OpenLane1000 子场景：按 segment 导出论文风格拼图，输出目录
  output/analysis/vis_model/<场景名>/<segment>/paper_grid.png

行1：RGB + 绿色真值；行2–4：黑底掩码（白=预测，红=类别错配）。
左侧：较大字号的行名（img / clrnet-l / clrnet-p / llanet）。

默认一次跑 6 个子场景（见 VIS_SCENARIOS）。仅调试某场景：环境变量 PAPER_SCENARIO=curve
仅调试前 N 个 segment：PAPER_MAX_SEGMENTS=3
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_pr = os.path.join(_REPO, "tools", "pr")
if _pr not in sys.path:
    sys.path.insert(0, _pr)

import cv2
import torch

import qualitative_failure_analysis_compare_models as M
from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.data.build import build_batch_data_loader


def in_repo(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(_REPO, rel)


# 与 lane3d_1000/test/1000_<name>.txt 对应（_match_indices_for_sequence 会自动试 1000_ 前缀）
VIS_SCENARIOS = [
    "curve",
    "night",
    "intersection",
    "extreme_weather",
    "merge_split_case",
    "updown",
]

NUM_COLUMNS = 8
POOL_MAX = 600
CROP_TOP_PX = 90
OUT_VIS_ROOT = "output/analysis/vis_model"
CELL_W = 420
CELL_H = 238
LABEL_STRIP_W = 68
LABEL_FONT_SIZE = 28
BATCH = 4
NUM_WORKERS = 0

_ms = os.environ.get("PAPER_MAX_SEGMENTS", "").strip()
MAX_SEGMENTS = int(_ms) if _ms else None
_only = os.environ.get("PAPER_SCENARIO", "").strip()  # 只跑单场景，如 curve

MODELS = [
    (
        "config/llanetv1/openlane1000/category/clrnet_linear_resnet34.py",
        "output/llanetv1/openlane1000/category/clrnet_linear_resnet34_base/model_best.pth",
        "clrnet-l",
    ),
    (
        "config/llanetv1/openlane1000/category/clrnet_prototype_resnet34.py",
        "output/llanetv1/openlane1000/category/clrnet_prototype_resnet34_base/model_best.pth",
        "clrnet-p",
    ),
    (
        "config/llanetv1/openlane1000/category/clrnet_combined_resnet34.py",
        "output/llanetv1/openlane1000/category/clrnet_combined_resnet34/model_best.pth",
        "llanet",
    ),
]


def run_one_scenario(
    scenario_name: str,
    test_dataset,
    cfgs,
    param_ref,
    data_root: str,
) -> None:
    sel = M._match_indices_for_sequence(test_dataset, scenario_name, data_root=data_root)
    sel = sorted(set(sel))
    if not sel:
        skip_msg = f"[SKIP] {scenario_name}: 无匹配帧（检查 data_root/test/1000_{scenario_name}.txt）"
        print(skip_msg)
        return
    sel = M.sort_indices_temporally(test_dataset, sel)
    by_seg = M.group_indices_by_segment(test_dataset, sel)
    if not by_seg:
        print(f"[SKIP] {scenario_name}: 路径中无 segment-* 子串")
        return

    seg_names = sorted(by_seg.keys())
    if MAX_SEGMENTS is not None:
        seg_names = seg_names[: int(MAX_SEGMENTS)]

    seg_to_cols: dict = {}
    for seg_name in seg_names:
        pool = by_seg[seg_name]
        if len(pool) > POOL_MAX:
            pool = M._subsample_evenly(pool, POOL_MAX)
        pool = M.sort_indices_temporally(test_dataset, pool)
        seg_to_cols[seg_name] = M.pick_column_indices_even(pool, NUM_COLUMNS)

    all_needed = sorted({i for cols in seg_to_cols.values() for i in cols})
    g2p = {g: p for p, g in enumerate(all_needed)}
    print(
        f"[RUN] scenario={scenario_name!r} segments={len(seg_names)} "
        f"union_frames={len(all_needed)} DEVICE={M.DEVICE}"
    )

    preds_triple: list = []
    for mi, (_, ckpt_p, _) in enumerate(MODELS):
        model = instantiate(cfgs[mi].model).to(M.DEVICE)
        model.eval()
        Checkpointer(model).load(in_repo(ckpt_p))
        subset = torch.utils.data.Subset(test_dataset, all_needed)
        loader = build_batch_data_loader(
            dataset=subset,
            total_batch_size=BATCH,
            num_workers=NUM_WORKERS,
            drop_last=False,
            shuffle=False,
        )
        preds_triple.append(M._run_inference_on_subset(model, loader, device=M.DEVICE))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all(len(preds_triple[i]) == len(all_needed) for i in range(3)):
        raise RuntimeError("pred length mismatch")

    out_base = in_repo(os.path.join(OUT_VIS_ROOT, scenario_name))

    for seg_name in seg_names:
        cols = seg_to_cols[seg_name]
        if not cols:
            continue
        stats_triple: list = [[], [], []]
        for li, orig_idx in enumerate(cols):
            data_info = test_dataset.data_infos[orig_idx]
            gt_lanes = data_info.get("lanes", [])
            gt_cats = data_info.get("lane_categories", [-1] * len(gt_lanes))
            for mi in range(3):
                pc = getattr(cfgs[mi], "param_config", None) or param_ref
                pr = preds_triple[mi][g2p[orig_idx]]
                stats_triple[mi].append(
                    M._compute_frame_match_stats(
                        pred_lanes=pr,
                        gt_lanes=gt_lanes,
                        param_cfg=pc,
                        gt_cats=gt_cats,
                        iou_threshold=M.IOU_THRESHOLD,
                        iou_width=M.IOU_WIDTH,
                    )
                )

        row_cells: list = [[] for _ in range(4)]
        gt_ref_stats = stats_triple[0]
        for li in range(len(cols)):
            orig_idx = cols[li]
            data_info = test_dataset.data_infos[orig_idx]
            ip = M._img_path_from_data_info(data_info)
            im = cv2.imread(ip)
            if im is None:
                raise FileNotFoundError(ip)
            rgb_gt = M.render_paper_cell_rgb_with_gt(im, gt_ref_stats[li], CROP_TOP_PX)
            row_cells[0].append(rgb_gt)
            hw = rgb_gt.shape[:2]
            for ri, mi in enumerate([0, 1, 2], start=1):
                mask = M.render_paper_cell_pred_mask(hw, stats_triple[mi][li], CROP_TOP_PX)
                row_cells[ri].append(mask)

        row_specs = [
            ("img", row_cells[0]),
            (MODELS[0][2], row_cells[1]),
            (MODELS[1][2], row_cells[2]),
            (MODELS[2][2], row_cells[3]),
        ]
        grid = M.compose_paper_grid_tight(
            row_specs,
            cell_w=CELL_W,
            cell_h=CELL_H,
            label_strip_w=LABEL_STRIP_W,
            label_font_size=LABEL_FONT_SIZE,
        )
        out_dir = os.path.join(out_base, seg_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "paper_grid.png")
        cv2.imwrite(out_path, grid)
        print(f"[OK] {scenario_name}/{seg_name} -> {out_path}")


def main() -> None:
    os.chdir(_REPO)
    M.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    M.IOU_THRESHOLD = 0.5
    M.IOU_WIDTH = 30

    for cfg_p, ckpt_p, _ in MODELS:
        for path, tag in ((in_repo(cfg_p), "cfg"), (in_repo(ckpt_p), "ckpt")):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"missing {tag}: {path}")

    cfgs = [LazyConfig.load(in_repo(m[0])) for m in MODELS]
    param_ref = getattr(cfgs[0], "param_config", None)
    assert param_ref is not None
    data_root = getattr(param_ref, "data_root", None)
    if not data_root:
        ds0 = instantiate(cfgs[0].dataloader.test).dataset
        data_root = ds0.data_root

    test_dataset = instantiate(cfgs[0].dataloader.test.dataset)

    scenarios = [_only] if _only else list(VIS_SCENARIOS)
    for scenario_name in scenarios:
        run_one_scenario(scenario_name, test_dataset, cfgs, param_ref, data_root)


if __name__ == "__main__":
    main()
