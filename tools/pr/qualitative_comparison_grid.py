"""
论文式定性拼图：多列（每列一帧）× 多行（原图 | 模型1 | 模型2 | GT）。
类别与置信度标注复用 qualitative_failure_analysis_compare_models 中的车道旁标签逻辑。

使用方式：与 qualitative_failure_analysis_compare_models.py 相同，先在本文件或外层入口里改
`qualitative_failure_analysis_compare_models` 模块的各项 M.* 配置，再执行：

  python tools/pr/qualitative_comparison_grid.py

或：

  python tools/pr/run_curve_qualitative_grid.py
"""
from __future__ import annotations

import os
import sys
from typing import List

import cv2

import qualitative_failure_analysis_compare_models as M
from unlanedet.config import LazyConfig, instantiate
from unlanedet.checkpoint import Checkpointer
from unlanedet.data.build import build_batch_data_loader
import torch

# ======================= 仅网格脚本参数 =======================
NUM_GRID_COLS = 8
CELL_W = 480
CELL_H = 270
GRID_OUT_FILENAME = "qualitative_grid_compare.png"
# 叠加图上行标题（中文，建议短一些）
MODEL1_DISPLAY_ZH = "模型1"
MODEL2_DISPLAY_ZH = "模型2"
# 从匹配到的序列帧中先均匀池化到至多这么多帧，再从中均匀取 NUM_GRID_COLS 列
GRID_POOL_MAX_FRAMES = 500


def _pick_column_indices(sorted_sel: List[int], n: int) -> List[int]:
    """在有序 dataset 下标列表上均匀取 n 个索引（尽量覆盖时间跨度）。"""
    if n <= 0:
        return []
    u = len(sorted_sel)
    if u == 0:
        return []
    if u <= n:
        return list(sorted_sel)
    if n == 1:
        return [sorted_sel[u // 2]]
    return [sorted_sel[int(round(i * (u - 1) / (n - 1)))] for i in range(n)]


def main() -> None:
    repo_root = M._get_repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    cfg = LazyConfig.load(M.CFG_PATH)
    param_cfg = getattr(cfg, "param_config", None)
    if param_cfg is None:
        raise ValueError(f"cfg 中找不到 param_config，请检查 {M.CFG_PATH}")

    data_root = getattr(param_cfg, "data_root", None)
    if not data_root:
        try:
            test_dataset0 = instantiate(cfg.dataloader.test).dataset
            data_root = test_dataset0.data_root
        except Exception:
            data_root = None
    if not data_root:
        raise ValueError("无法推断 data_root，请确保 config 正确配置。")

    test_dataset = instantiate(cfg.dataloader.test.dataset)

    if not M.SEQUENCE_NAMES:
        raise ValueError("SEQUENCE_NAMES 不能为空。")
    if len(M.SEQUENCE_NAMES) > 1:
        print(
            f"[WARN] GRID 仅使用 SEQUENCE_NAMES[0]={M.SEQUENCE_NAMES[0]!r} 采样各列；"
            f"忽略其余 {M.SEQUENCE_NAMES[1:]!r}。"
        )
    seq_name = M.SEQUENCE_NAMES[0]

    sel = M._match_indices_for_sequence(test_dataset, seq_name, data_root=data_root)
    sel = sorted(set(sel))
    sel = M._filter_indices_by_path_substr(test_dataset, sel, M.SEGMENT_PATH_CONTAINS)
    pool = sel
    if len(pool) > GRID_POOL_MAX_FRAMES:
        pool = M._subsample_evenly(pool, GRID_POOL_MAX_FRAMES)

    union_indices = _pick_column_indices(pool, NUM_GRID_COLS)
    if len(union_indices) < NUM_GRID_COLS:
        print(
            f"[WARN] 匹配帧仅 {len(union_indices)} 个，少于 NUM_GRID_COLS={NUM_GRID_COLS}，"
            "将用现有帧填满各列。"
        )

    if not union_indices:
        raise RuntimeError(
            "没有可用于网格的帧：请检查 SEQUENCE_NAMES / SEGMENT_PATH_CONTAINS / data_root。"
        )

    print(f"[GRID] columns={len(union_indices)} dataset_idx={union_indices}")

    subset_dataset = torch.utils.data.Subset(test_dataset, union_indices)
    subset_dataloader = build_batch_data_loader(
        dataset=subset_dataset,
        total_batch_size=M.SUBSET_TOTAL_BATCH_SIZE,
        num_workers=M.SUBSET_NUM_WORKERS,
        drop_last=False,
        shuffle=False,
    )

    m1_ckpt_stem = os.path.splitext(os.path.basename(M.MODEL1_CKPT_PATH))[0] or "model1"
    m2_ckpt_stem = os.path.splitext(os.path.basename(M.MODEL2_CKPT_PATH))[0] or "model2"

    print("[MODEL] grid: loading model1 ...")
    model1 = instantiate(cfg.model)
    model1.to(M.DEVICE)
    model1.eval()
    Checkpointer(model1).load(M.MODEL1_CKPT_PATH)
    print("[MODEL] grid: inference model1 ...")
    pred1 = M._run_inference_on_subset(model1, subset_dataloader, device=M.DEVICE)

    cfg2_path = M.MODEL2_CFG_PATH or M.CFG_PATH
    cfg2 = LazyConfig.load(cfg2_path) if cfg2_path != M.CFG_PATH else cfg
    pc2 = getattr(cfg2, "param_config", None)
    if pc2 is not None and getattr(pc2, "data_root", None) != data_root:
        print(
            f"[WARN] 模型2 data_root 与模型1 不一致: {getattr(pc2, 'data_root', None)} vs {data_root}"
        )

    print("[MODEL] grid: loading model2 ...")
    model2 = instantiate(cfg2.model)
    model2.to(M.DEVICE)
    model2.eval()
    Checkpointer(model2).load(M.MODEL2_CKPT_PATH)
    print("[MODEL] grid: inference model2 ...")
    pred2 = M._run_inference_on_subset(model2, subset_dataloader, device=M.DEVICE)

    if len(pred1) != len(pred2) or len(pred1) != len(union_indices):
        raise RuntimeError(
            f"pred 长度异常: len(pred1)={len(pred1)}, len(pred2)={len(pred2)}, union={len(union_indices)}"
        )

    pred_stats_m1: List[M.FrameMatchStats] = []
    pred_stats_m2: List[M.FrameMatchStats] = []

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

    disp1 = MODEL1_DISPLAY_ZH
    disp2 = MODEL2_DISPLAY_ZH

    rows = [
        ("raw", "输入图像"),
        ("m1", f"{disp1} 预测（红·含类别）"),
        ("m2", f"{disp2} 预测（蓝·含类别）"),
        ("gt", "真值 GT（绿·车道类别）"),
    ]

    row_blocks: List = []
    for mode_key, row_title in rows:
        cells = []
        for li, orig_idx in enumerate(union_indices):
            data_info = test_dataset.data_infos[orig_idx]
            img_path = M._img_path_from_data_info(data_info)
            if not img_path or not os.path.exists(img_path):
                raise FileNotFoundError(f"图像不存在: {img_path}")
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise RuntimeError(f"cv2.imread 失败: {img_path}")

            gt_stats = pred_stats_m1[li]
            m1s = pred_stats_m1[li]
            m2s = pred_stats_m2[li]
            panel = M.render_grid_cell_panel(
                img_bgr,
                mode_key,
                gt_stats,
                m1s,
                m2s,
                m1_display=disp1,
                m2_display=disp2,
            )
            cells.append(panel)
        row_blocks.append(M.compose_grid_row(row_title, cells, CELL_W, CELL_H))

    grid = cv2.vconcat(row_blocks)

    out_dir = (
        os.path.join(M.OUT_DIR, seq_name) if M.APPEND_SCENARIO_SUBDIR else M.OUT_DIR
    )
    M._safe_mkdir(out_dir)
    out_path = os.path.join(out_dir, GRID_OUT_FILENAME)
    cv2.imwrite(out_path, grid)
    print(f"[GRID] 已保存: {out_path} （ckpt: {m1_ckpt_stem} vs {m2_ckpt_stem}）")


if __name__ == "__main__":
    main()
