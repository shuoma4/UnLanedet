"""一次跑通 qualitative_lane_compare.ipynb 中的推理与存图逻辑（无 IPython）。"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_pr = os.path.join(REPO_ROOT, "tools", "pr")
if _pr not in sys.path:
    sys.path.insert(0, _pr)

import cv2
import torch

import qualitative_failure_analysis_compare_models as M
from unlanedet.checkpoint import Checkpointer
from unlanedet.config import LazyConfig, instantiate
from unlanedet.data.build import build_batch_data_loader


def in_repo(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


def pick_column_indices(sorted_sel, n):
    u = len(sorted_sel)
    if n <= 0 or u == 0:
        return []
    if u <= n:
        return list(sorted_sel)
    if n == 1:
        return [sorted_sel[u // 2]]
    return [sorted_sel[int(round(i * (u - 1) / (n - 1)))] for i in range(n)]


def main() -> None:
    os.chdir(REPO_ROOT)

    M.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    M.CFG_PATH = "config/llanetv1/openlane1000/category/clrnet_linear_resnet34.py"
    M.MODEL2_CFG_PATH = "config/llanetv1/openlane1000/resnet34_llanet_ablation_exp2.py"
    M.MODEL1_CKPT_PATH = "output/llanetv1/openlane1000/category/clrnet_linear_resnet34/model_best.pth"
    M.MODEL2_CKPT_PATH = "output/llanetv1/openlane1000/resnet34_llanet_ablation_exp2/model_best.pth"
    M.SEQUENCE_NAMES = ["curve"]
    M.SEGMENT_PATH_CONTAINS = (
        "segment-10203656353524179475_7625_000_7645_000_with_camera_labels"
    )
    M.OUT_DIR = os.path.join("output", "analysis", "curve", M.SEGMENT_PATH_CONTAINS)
    M.APPEND_SCENARIO_SUBDIR = False
    M.CROP_TOP_PX = 90
    M.SUBSET_TOTAL_BATCH_SIZE = 2
    M.SUBSET_NUM_WORKERS = 0

    NUM_GRID_COLS = 8
    CELL_W, CELL_H = 480, 270
    GRID_POOL_MAX_FRAMES = 500
    GRID_OUT_NAME = "qualitative_grid_compare.png"
    MODEL1_DISPLAY_ZH = "CLRNet-linear"
    MODEL2_DISPLAY_ZH = "LLANet Exp2"
    PREVIEW_LOCAL_IDX = 0

    cfg1_abs = in_repo(M.CFG_PATH)
    for path, label in (
        (cfg1_abs, "cfg1"),
        (in_repo(M.MODEL2_CFG_PATH), "cfg2"),
        (in_repo(M.MODEL1_CKPT_PATH), "ckpt1"),
        (in_repo(M.MODEL2_CKPT_PATH), "ckpt2"),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing {label}: {path}")

    cfg = LazyConfig.load(cfg1_abs)
    param_cfg = getattr(cfg, "param_config", None)
    assert param_cfg is not None

    data_root = getattr(param_cfg, "data_root", None)
    if not data_root:
        test_d0 = instantiate(cfg.dataloader.test).dataset
        data_root = test_d0.data_root

    test_dataset = instantiate(cfg.dataloader.test.dataset)
    seq_name = M.SEQUENCE_NAMES[0]

    sel = M._match_indices_for_sequence(test_dataset, seq_name, data_root=data_root)
    sel = sorted(set(sel))
    sel = M._filter_indices_by_path_substr(test_dataset, sel, M.SEGMENT_PATH_CONTAINS)
    pool = sel
    if len(pool) > GRID_POOL_MAX_FRAMES:
        pool = M._subsample_evenly(pool, GRID_POOL_MAX_FRAMES)

    union_indices = pick_column_indices(pool, NUM_GRID_COLS)
    print(f"columns={len(union_indices)} idx={union_indices}")
    if not union_indices:
        raise RuntimeError("no frames")

    subset_dataset = torch.utils.data.Subset(test_dataset, union_indices)
    subset_dataloader = build_batch_data_loader(
        dataset=subset_dataset,
        total_batch_size=M.SUBSET_TOTAL_BATCH_SIZE,
        num_workers=M.SUBSET_NUM_WORKERS,
        drop_last=False,
        shuffle=False,
    )

    m1_name = os.path.splitext(os.path.basename(M.MODEL1_CKPT_PATH))[0]
    m2_name = os.path.splitext(os.path.basename(M.MODEL2_CKPT_PATH))[0]

    model1 = instantiate(cfg.model).to(M.DEVICE)
    model1.eval()
    Checkpointer(model1).load(in_repo(M.MODEL1_CKPT_PATH))
    print("infer model1 …")
    pred1 = M._run_inference_on_subset(model1, subset_dataloader, device=M.DEVICE)
    del model1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cfg2_abs = in_repo(M.MODEL2_CFG_PATH or M.CFG_PATH)
    cfg2 = LazyConfig.load(cfg2_abs) if cfg2_abs != cfg1_abs else cfg
    model2 = instantiate(cfg2.model).to(M.DEVICE)
    model2.eval()
    Checkpointer(model2).load(in_repo(M.MODEL2_CKPT_PATH))
    print("infer model2 …")
    pred2 = M._run_inference_on_subset(model2, subset_dataloader, device=M.DEVICE)
    del model2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pred_stats_m1, pred_stats_m2 = [], []
    for local_idx, orig_idx in enumerate(union_indices):
        data_info = test_dataset.data_infos[orig_idx]
        gt_lanes = data_info.get("lanes", [])
        gt_cats = data_info.get("lane_categories", [-1] * len(gt_lanes))
        pred_stats_m1.append(
            M._compute_frame_match_stats(
                pred_lanes=pred1[local_idx],
                gt_lanes=gt_lanes,
                param_cfg=param_cfg,
                gt_cats=gt_cats,
                iou_threshold=M.IOU_THRESHOLD,
                iou_width=M.IOU_WIDTH,
            )
        )
        param_for_m2 = getattr(cfg2, "param_config", None) or param_cfg
        pred_stats_m2.append(
            M._compute_frame_match_stats(
                pred_lanes=pred2[local_idx],
                gt_lanes=gt_lanes,
                param_cfg=param_for_m2,
                gt_cats=gt_cats,
                iou_threshold=M.IOU_THRESHOLD,
                iou_width=M.IOU_WIDTH,
            )
        )

    out_dir = in_repo(
        os.path.join(M.OUT_DIR, seq_name) if M.APPEND_SCENARIO_SUBDIR else M.OUT_DIR
    )
    os.makedirs(out_dir, exist_ok=True)

    li = int(min(max(PREVIEW_LOCAL_IDX, 0), len(union_indices) - 1))
    orig_idx = union_indices[li]
    data_info = test_dataset.data_infos[orig_idx]
    img_path = M._img_path_from_data_info(data_info)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"cannot read image: {img_path}")

    gt_s, s1, s2 = pred_stats_m1[li], pred_stats_m1[li], pred_stats_m2[li]
    safe = (M._img_name_from_data_info(data_info) or "frame").replace("/", "_")
    single_path = os.path.join(out_dir, f"{safe}__compare_notebook.png")
    M._draw_failure_comparison(
        out_path=single_path,
        img_bgr=img_bgr,
        gt_stats=gt_s,
        m1_stats=s1,
        m2_stats=s2,
        m1_name=m1_name,
        m2_name=m2_name,
    )
    print("saved", single_path)

    rows = [
        ("raw", "输入图像"),
        ("m1", f"{MODEL1_DISPLAY_ZH} 预测（红·含类别）"),
        ("m2", f"{MODEL2_DISPLAY_ZH} 预测（蓝·含类别）"),
        ("gt", "真值 GT（绿·车道类别）"),
    ]
    row_blocks = []
    for mode_key, row_title in rows:
        cells = []
        for i, oi in enumerate(union_indices):
            data_info_i = test_dataset.data_infos[oi]
            ip = M._img_path_from_data_info(data_info_i)
            im = cv2.imread(ip)
            if im is None:
                raise FileNotFoundError(ip)
            cells.append(
                M.render_grid_cell_panel(
                    im,
                    mode_key,
                    pred_stats_m1[i],
                    pred_stats_m1[i],
                    pred_stats_m2[i],
                    m1_display=MODEL1_DISPLAY_ZH,
                    m2_display=MODEL2_DISPLAY_ZH,
                )
            )
        row_blocks.append(M.compose_grid_row(row_title, cells, CELL_W, CELL_H))

    grid_bgr = cv2.vconcat(row_blocks)
    grid_path = os.path.join(out_dir, GRID_OUT_NAME)
    cv2.imwrite(grid_path, grid_bgr)
    print("saved", grid_path)


if __name__ == "__main__":
    main()
