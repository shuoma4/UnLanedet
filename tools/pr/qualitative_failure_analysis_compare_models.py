import os
import re
import sys
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Set

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# 使用 matplotlib_font 里的字体安装逻辑（同时复用其字体候选）
from config.matplotlib_font import *  # noqa: F401,F403

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

# 让 `python tools/pr/xxx.py` 也能正确导入顶层 `unlanedet/` 包
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from unlanedet.config import LazyConfig, instantiate
from unlanedet.checkpoint import Checkpointer
from unlanedet.data.build import build_batch_data_loader

# 让 culane_metric 的插值逻辑适配 OpenLane
from unlanedet.evaluation import openlane_evaluator as _openlane_evaluator  # noqa: F401
from unlanedet.evaluation import culane_metric


# ======================= 可配置区（只改这里） =======================

# 1) 使用哪份 config 来实例化模型1 与 dataloader（数据管线以第一份为准）
CFG_PATH = "config/llanetv1/openlane1000/tsm/llanet_tsm.py"

# 1b) 模型2 的 config：与模型1 架构不同时必填（例如 FPN vs GSAFPN）；相同时可保持 None
MODEL2_CFG_PATH = None  # 例如: "config/llanetv1/openlane1000/resnet34_llanet_ablation_exp2.py"

# 2) 两个模型权重路径（直接填 .pth/.pt 等）
MODEL1_CKPT_PATH = "/path/to/model1.pth"
MODEL2_CKPT_PATH = "/path/to/model2.pth"

# 3) 你要分析的“序列名称/场景名称”列表。
#    - 支持两种方式：
#      A. 如果在 data_root 下存在 `test/{name}.txt` 或 `list/test_split/{name}.txt`，则按这些文件筛选帧（推荐：night/intersection/occlusion 等）。
#      B. 否则当作字符串子集匹配：`img_path` / `img_name` 中包含该名字即可。
SEQUENCE_NAMES = [
    "night",
    "intersection",
    "occlusion",
]

# 3b) 可选：在场景列表匹配的帧中，仅保留图像路径包含该子串的序列（如某个 segment-xxx）
SEGMENT_PATH_CONTAINS = None  # 例如: "segment-10203656353524179475_7625_000_7645_000_with_camera_labels"

# 4) 输出目录（脚本会在下面创建子文件夹）
OUT_DIR = "output/pr/qualitative_failure_analysis_compare"

# 4b) 是否在 OUT_DIR 下再建一层 SEQUENCE_NAMES 子目录；单场景且你希望直接落在 OUT_DIR 时可设为 False
APPEND_SCENARIO_SUBDIR = True

# 5) 推理与绘图参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBSET_TOTAL_BATCH_SIZE = 4  # 只用于子集推理
SUBSET_NUM_WORKERS = 2

GT_LINE_WIDTH = 3
MODEL1_LINE_WIDTH = 3
MODEL2_LINE_WIDTH = 3
LABEL_FONT_SCALE = 0.45
LABEL_THICKNESS = 1

# 5b) 中文字体/排版（PIL 渲染，避免 OpenCV 中文乱码）
FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
HEADER_FONT_SIZE = 22
LEGEND_FONT_SIZE = 18
LEGEND_MAX_ITEMS = 18  # 图例最多显示多少条（太多会遮挡）
LEGEND_BOX_ALPHA = 0.75  # 半透明底框

# 5c) 画面裁剪：裁掉顶部多余天空（注意会同步平移所有车道点坐标）
CROP_TOP_PX = 90  # 0 表示不裁剪

# 5d) 逐车道文字标注（贴在车道线旁边，避免重叠）
LANE_LABEL_FONT_SIZE = 22
LANE_LABEL_PAD_PX = 4
LANE_LABEL_BOX_ALPHA = 0.90
LANE_LABEL_MAX_PER_GROUP = 12  # 每组(GT/红/蓝)最多标多少条，避免刷屏

# 6) 失败案例筛选：先把每个序列最多抽取 MAX_FRAMES_PER_SEQUENCE 帧用于“计算失败原因”
MAX_FRAMES_PER_SEQUENCE = 80
TOPK_FAILURES_TO_DRAW_PER_SEQUENCE = 12
ALSO_DRAW_RANDOM = 6  # 每个序列额外随机抽一些帧，便于对照定性效果

# 7) 用于“客观失败原因”的匹配参数（OpenLane/CULane IoU 逻辑）
IOU_THRESHOLD = 0.5
IOU_WIDTH = 30

# 8) 可选：绘制时是否对 FP/FN 做加粗提示（不改变颜色，仍满足“真值绿色”）
HIGHLIGHT_FP_FN = True

# 9) 论文风格：掩码行线宽（黑底白线，类别错误为红线）
PAPER_MASK_LINE_WIDTH = 5


# ======================= 类别映射（用于显示标签） =======================

LANE_CATEGORIES_ZH = {
    0: "未知",
    1: "白色虚线",
    2: "白色实线",
    3: "双白虚线",
    4: "双白实线",
    5: "左白虚右白实",
    6: "左白实右白虚",
    7: "黄色虚线",
    8: "黄色实线",
    9: "双黄虚线",
    10: "双黄实线",
    11: "左黄虚右黄实",
    12: "左黄实右黄虚",
    13: "左侧路缘",
    14: "右侧路缘",
}

MODEL1_COLOR_BGR = (0, 0, 255)  # 红
MODEL2_COLOR_BGR = (255, 0, 0)  # 蓝
GT_COLOR_BGR = (0, 255, 0)  # 绿


# ======================= 工具函数 =======================


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _get_repo_root() -> str:
    # 兼容：直接 `python tools/pr/xxx.py` 运行
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _img_path_from_data_info(data_info: Dict[str, Any]) -> Optional[str]:
    # OpenLane(OpenLaneTemporal) 中常见字段：img_path / origin_img_path
    for k in ["img_path", "origin_img_path", "full_img_path"]:
        v = data_info.get(k)
        if v:
            return str(v)
    return None


def _img_name_from_data_info(data_info: Dict[str, Any]) -> str:
    # 用于输出文件名的稳定标识
    for k in ["img_name", "img_path", "origin_img_path"]:
        v = data_info.get(k)
        if v:
            return str(v)
    return "unknown_img"


def _img_path_to_rel_key(img_path: str) -> str:
    # 尽量复刻 openlane_evaluator 的子场景匹配逻辑
    img_path = img_path.replace("\\", "/")
    parts = img_path.split("/")
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return img_path


def _try_load_scenario_lines(data_root: str, name: str) -> Optional[Set[str]]:
    # OpenLane1000：test 下常见 1000_curve.txt；也支持 test/curve.txt
    n = name.replace(".txt", "").strip()
    cand = [
        os.path.join(data_root, "test", f"{n}.txt"),
    ]
    if not n.startswith("1000_"):
        cand.append(os.path.join(data_root, "test", f"1000_{n}.txt"))
    cand.extend(
        [
            os.path.join(data_root, "list", "test_split", f"{n}.txt"),
            os.path.join(data_root, "list", f"{n}.txt"),
        ]
    )
    for p in cand:
        if os.path.exists(p):
            with open(p, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            return set(lines)
    return None


def _match_indices_for_sequence(
    dataset,
    seq_name: str,
    data_root: str,
) -> List[int]:
    """
    返回 dataset 的“索引”（对 dataset.data_infos 的索引），用于抽帧/绘制。
    """
    lines = _try_load_scenario_lines(data_root, seq_name)
    selected = []
    for idx, data_info in enumerate(dataset.data_infos):
        img_path = _img_path_from_data_info(data_info) or ""
        img_name = str(data_info.get("img_name", "")) or ""
        if lines is not None:
            rel_key = _img_path_to_rel_key(img_path)
            matched = rel_key in lines
            if not matched:
                # 兼容：lines 可能是子串（而不是严格 rel_path）
                for ln in lines:
                    if ln and ln in img_path:
                        matched = True
                        break
            if matched:
                selected.append(idx)
        else:
            # 退化：子串匹配
            if seq_name in img_path or seq_name in img_name:
                selected.append(idx)
    return selected


def _filter_indices_by_path_substr(
    dataset, indices: List[int], substr: Optional[str]
) -> List[int]:
    if not substr:
        return indices
    out: List[int] = []
    for idx in indices:
        p = _img_path_from_data_info(dataset.data_infos[idx]) or ""
        if substr in p:
            out.append(idx)
    return out


def _subsample_evenly(sorted_indices: List[int], max_count: int) -> List[int]:
    if max_count <= 0 or len(sorted_indices) <= max_count:
        return list(sorted_indices)
    step = max(1, len(sorted_indices) // max_count)
    out = sorted_indices[::step]
    return out[:max_count]


def _build_index_mapping(orig_indices: List[int]) -> Dict[int, int]:
    return {orig_idx: i for i, orig_idx in enumerate(orig_indices)}


def _lane_label_for_cat(cat_id: int) -> str:
    name = LANE_CATEGORIES_ZH.get(int(cat_id), str(cat_id))
    return name


@dataclass
class FrameMatchStats:
    tp: int
    fp: int
    fn: int
    cat_acc: float
    cat_wrong_tp: int
    pred_is_tp: List[bool]
    gt_is_tp: List[bool]
    pred_cat_mismatch_indices: List[int]
    # 用于绘制
    pred_lane_points: List[np.ndarray]  # list of (N,2)
    pred_cats: List[int]
    pred_confs: List[float]
    gt_lane_points: List[Any]  # keep original structure (list of points)
    gt_cats: List[int]


def _compute_frame_match_stats(
    pred_lanes,
    gt_lanes,
    param_cfg,
    gt_cats: List[int],
    iou_threshold: float,
    iou_width: int,
) -> FrameMatchStats:
    """
    pred_lanes: List[Lane] (带 metadata['category_id'])
    gt_lanes: List[(u,v)]，坐标系与 param_cfg.ori_img_* 一致（OpenLane 的原图像素坐标）
    """
    ori_h = int(param_cfg.ori_img_h)
    ori_w = int(param_cfg.ori_img_w)

    # 预测 lane -> 原图坐标点
    pred_lane_points: List[np.ndarray] = []
    pred_cats: List[int] = []
    pred_confs: List[float] = []
    for lane in pred_lanes:
        if hasattr(lane, "to_array"):
            pts = lane.to_array(param_cfg)  # (N,2) in original pixel coords
            if isinstance(pts, np.ndarray) and pts.shape[0] >= 2:
                pred_lane_points.append(pts)
            else:
                pred_lane_points.append(np.zeros((0, 2), dtype=np.float32))
        else:
            pred_lane_points.append(np.zeros((0, 2), dtype=np.float32))
        md = getattr(lane, "metadata", {}) or {}
        pred_cats.append(int(md.get("category_id", -1)))
        try:
            pred_confs.append(float(md.get("conf", 0.0)))
        except Exception:
            pred_confs.append(0.0)

    pred_valid = [len(p) >= 2 for p in pred_lane_points]
    # 过滤掉空 lane，避免离散IoU里出现奇怪的 mask
    pred_lane_points = [p for p, ok in zip(pred_lane_points, pred_valid) if ok]
    pred_cats = [c for c, ok in zip(pred_cats, pred_valid) if ok]
    pred_confs = [s for s, ok in zip(pred_confs, pred_valid) if ok]

    gt_lane_points = gt_lanes
    gt_valid_mask = [len(l) >= 2 for l in gt_lane_points]
    gt_lane_points = [l for l, ok in zip(gt_lane_points, gt_valid_mask) if ok]
    gt_cats = [c for c, ok in zip(gt_cats, gt_valid_mask) if ok]

    if len(pred_lane_points) == 0:
        return FrameMatchStats(
            tp=0,
            fp=0,
            fn=len(gt_lane_points),
            cat_acc=0.0,
            cat_wrong_tp=0,
            pred_is_tp=[False] * 0,
            gt_is_tp=[False] * len(gt_lane_points),
            pred_cat_mismatch_indices=[],
            pred_lane_points=[],
            pred_cats=[],
            pred_confs=[],
            gt_lane_points=gt_lane_points,
            gt_cats=gt_cats,
        )
    if len(gt_lane_points) == 0:
        return FrameMatchStats(
            tp=0,
            fp=len(pred_lane_points),
            fn=0,
            cat_acc=0.0,
            cat_wrong_tp=0,
            pred_is_tp=[False] * len(pred_lane_points),
            gt_is_tp=[],
            pred_cat_mismatch_indices=[],
            pred_lane_points=pred_lane_points,
            pred_cats=pred_cats,
            pred_confs=pred_confs,
            gt_lane_points=[],
            gt_cats=[],
        )

    # 构造 discrete IoU 的输入：需要“可迭代点列表”
    pred_pts_list = [p.tolist() for p in pred_lane_points]
    gt_pts_list = [np.asarray(l).tolist() for l in gt_lane_points]

    # OpenLaneEvaluator 在导入时已把 culane_metric.interp 替换成线性插值
    interp_pred = np.array(
        [culane_metric.interp(pred_lane, n=5) for pred_lane in pred_pts_list],
        dtype=object,
    )
    interp_gt = np.array(
        [culane_metric.interp(anno_lane, n=5) for anno_lane in gt_pts_list],
        dtype=object,
    )

    ious = culane_metric.discrete_cross_iou(
        interp_pred,
        interp_gt,
        width=iou_width,
        img_shape=(ori_h, ori_w),
    )

    row_ind, col_ind = linear_sum_assignment(1 - ious)
    tp_mask = ious[row_ind, col_ind] > float(iou_threshold)

    pred_is_tp = [False] * len(pred_lane_points)
    gt_is_tp = [False] * len(gt_lane_points)
    pred_cat_mismatch_indices: List[int] = []

    cat_correct = 0
    cat_wrong_tp = 0

    for k, is_tp in enumerate(tp_mask):
        if not is_tp:
            continue
        pr_idx = int(row_ind[k])
        gt_idx = int(col_ind[k])
        pred_is_tp[pr_idx] = True
        gt_is_tp[gt_idx] = True

        pr_cat = int(pred_cats[pr_idx])
        gt_cat = int(gt_cats[gt_idx])
        if pr_cat == gt_cat:
            cat_correct += 1
        else:
            cat_wrong_tp += 1
            pred_cat_mismatch_indices.append(pr_idx)

    tp = int(tp_mask.sum())
    fp = int(len(pred_lane_points) - tp)
    fn = int(len(gt_lane_points) - tp)
    cat_acc = float(cat_correct / tp) if tp > 0 else 0.0

    return FrameMatchStats(
        tp=tp,
        fp=fp,
        fn=fn,
        cat_acc=cat_acc,
        cat_wrong_tp=cat_wrong_tp,
        pred_is_tp=pred_is_tp,
        gt_is_tp=gt_is_tp,
        pred_cat_mismatch_indices=pred_cat_mismatch_indices,
        pred_lane_points=pred_lane_points,
        pred_cats=pred_cats,
        pred_confs=pred_confs,
        gt_lane_points=gt_lane_points,
        gt_cats=gt_cats,
    )


def _draw_lane_polyline(
    img_bgr: np.ndarray,
    pts_xy: np.ndarray,
    color_bgr: Tuple[int, int, int],
    thickness: int,
) -> None:
    pts = np.asarray(pts_xy, dtype=np.int32)
    if pts.shape[0] < 2:
        return
    for i in range(len(pts) - 1):
        p1 = tuple(pts[i])
        p2 = tuple(pts[i + 1])
        cv2.line(img_bgr, p1, p2, color_bgr, thickness, lineType=cv2.LINE_AA)


def _pick_font_path() -> Optional[str]:
    for p in FONT_PATH_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


def _pil_draw_text(
    img_bgr: np.ndarray,
    xy: Tuple[int, int],
    text: str,
    font_size: int,
    fill_rgb: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    使用 PIL 在图上写中文，避免 OpenCV putText 乱码。
    """
    if Image is None or ImageDraw is None or ImageFont is None:
        # fallback：退化为英文/数字能用的 OpenCV（中文仍可能乱码）
        cv2.putText(
            img_bgr,
            text,
            xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (fill_rgb[2], fill_rgb[1], fill_rgb[0]),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font_path = _pick_font_path()
    try:
        font = (
            ImageFont.truetype(font_path, font_size)
            if font_path
            else ImageFont.load_default()
        )
    except Exception:
        font = ImageFont.load_default()
    draw.text(xy, text, font=font, fill=fill_rgb)
    out_rgb = np.asarray(pil_img)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def _pil_text_size(text: str, font_size: int) -> Tuple[int, int]:
    if Image is None or ImageDraw is None or ImageFont is None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        return int(tw), int(th)
    font_path = _pick_font_path()
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    # PIL 需要一个临时 draw 才能测量
    tmp = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(tmp)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except Exception:
        w, h = draw.textsize(text, font=font)
        return int(w), int(h)


def _rects_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def _place_label_box(
    anchor_xy: Tuple[int, int],
    text: str,
    font_size: int,
    img_w: int,
    img_h: int,
    occupied: List[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    为标签选择一个不重叠的落点。
    返回 (x0,y0,x1,y1,tx,ty)：背景框与文字起点。
    """
    tw, th = _pil_text_size(text, font_size)
    pad = int(LANE_LABEL_PAD_PX)
    bw, bh = tw + 2 * pad, th + 2 * pad
    ax, ay = anchor_xy

    # 候选偏移：优先放在线的外侧（左右上方/左右下方/正上/正下）
    candidates = [
        (16, -bh - 12),
        (-bw - 16, -bh - 12),
        (16, 12),
        (-bw - 16, 12),
        (0, -bh - 12),
        (0, 12),
        (32, -bh // 2),
        (-bw - 32, -bh // 2),
    ]

    for dx, dy in candidates:
        x0 = int(ax + dx)
        y0 = int(ay + dy)
        x0 = max(2, min(img_w - bw - 2, x0))
        y0 = max(2, min(img_h - bh - 2, y0))
        x1, y1 = x0 + bw, y0 + bh
        rect = (x0, y0, x1, y1)
        if any(_rects_intersect(rect, r) for r in occupied):
            continue
        occupied.append(rect)
        tx, ty = x0 + pad, y0 + pad
        return x0, y0, x1, y1, tx, ty

    # 实在放不下：允许与已有轻微相交（取第一个）
    dx, dy = candidates[0]
    x0 = int(ax + dx)
    y0 = int(ay + dy)
    x0 = max(2, min(img_w - bw - 2, x0))
    y0 = max(2, min(img_h - bh - 2, y0))
    x1, y1 = x0 + bw, y0 + bh
    rect = (x0, y0, x1, y1)
    occupied.append(rect)
    tx, ty = x0 + pad, y0 + pad
    return x0, y0, x1, y1, tx, ty


def _lane_anchor_point(pts_xy: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    选一个靠近车道线“上半段”的锚点，便于贴标签但不遮挡路面关键区域。
    """
    pts = np.asarray(pts_xy, dtype=np.float32)
    if pts.shape[0] < 2:
        return None
    # y 从小到大排序，取 25% 分位处（偏上但不在最顶端）
    order = np.argsort(pts[:, 1])
    k = int(max(0, min(len(order) - 1, round(0.25 * (len(order) - 1)))))
    p = pts[order[k]]
    return int(p[0]), int(p[1])

def _alpha_box(
    img_bgr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color_bgr: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.7,
) -> None:
    x0 = max(0, min(img_bgr.shape[1] - 1, x0))
    x1 = max(0, min(img_bgr.shape[1], x1))
    y0 = max(0, min(img_bgr.shape[0] - 1, y0))
    y1 = max(0, min(img_bgr.shape[0], y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img_bgr[y0:y1, x0:x1].astype(np.float32)
    overlay = np.zeros_like(roi)
    overlay[:] = np.array(color_bgr, dtype=np.float32)
    blended = roi * (1 - alpha) + overlay * alpha
    img_bgr[y0:y1, x0:x1] = blended.astype(np.uint8)


def _shift_points_after_crop(pts_xy: np.ndarray, crop_top_px: int) -> np.ndarray:
    p = np.asarray(pts_xy, dtype=np.float32).copy()
    if p.size == 0:
        return p
    p[:, 1] -= float(crop_top_px)
    keep = p[:, 1] >= 0
    return p[keep]


def _crop_and_shift_lanes(
    img_bgr: np.ndarray,
    crop_top_px: int,
    gt_stats: FrameMatchStats,
    m1_stats: FrameMatchStats,
    m2_stats: FrameMatchStats,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """裁掉图像顶部 crop_top_px，并把 GT/两模型预测点同步平移到裁剪后坐标系。"""
    canvas = img_bgr.copy()
    crop = min(int(crop_top_px), canvas.shape[0] - 2) if crop_top_px > 0 else 0
    if crop > 0:
        canvas = canvas[crop:, :, :].copy()
    gt_pts = [
        _shift_points_after_crop(np.asarray(p, dtype=np.float32), crop)
        for p in gt_stats.gt_lane_points
    ]
    m1_pts = [
        _shift_points_after_crop(np.asarray(p, dtype=np.float32), crop)
        for p in m1_stats.pred_lane_points
    ]
    m2_pts = [
        _shift_points_after_crop(np.asarray(p, dtype=np.float32), crop)
        for p in m2_stats.pred_lane_points
    ]
    return canvas, gt_pts, m1_pts, m2_pts


def _apply_lane_label(
    canvas: np.ndarray,
    occupied: List[Tuple[int, int, int, int]],
    anchor: Tuple[int, int],
    text: str,
    color_bgr: Tuple[int, int, int],
    font_size: int,
) -> np.ndarray:
    placed = _place_label_box(
        anchor_xy=anchor,
        text=text,
        font_size=font_size,
        img_w=canvas.shape[1],
        img_h=canvas.shape[0],
        occupied=occupied,
    )
    if placed is None:
        return canvas
    x0, y0, x1, y1, tx, ty = placed
    _alpha_box(canvas, x0, y0, x1, y1, color_bgr, alpha=float(LANE_LABEL_BOX_ALPHA))
    return _pil_draw_text(canvas, (tx, ty), text, font_size, (255, 255, 255))


def render_grid_cell_panel(
    img_bgr: np.ndarray,
    mode: str,
    gt_stats: FrameMatchStats,
    m1_stats: FrameMatchStats,
    m2_stats: FrameMatchStats,
    m1_display: str,
    m2_display: str,
    crop_top_px: Optional[int] = None,
) -> np.ndarray:
    """
    论文式网格中的单格渲染。
    mode: "raw" | "gt" | "m1" | "m2"
    """
    crop = int(CROP_TOP_PX if crop_top_px is None else crop_top_px)
    canvas, gt_pts, m1_pts, m2_pts = _crop_and_shift_lanes(
        img_bgr, crop, gt_stats, m1_stats, m2_stats
    )
    if mode == "raw":
        return canvas

    occupied: List[Tuple[int, int, int, int]] = []

    if mode == "gt":
        for i, lane_pts_np in enumerate(gt_pts):
            if lane_pts_np.shape[0] < 2:
                continue
            is_tp = gt_stats.gt_is_tp[i] if gt_stats.gt_is_tp else False
            thickness = GT_LINE_WIDTH if (is_tp or not HIGHLIGHT_FP_FN) else GT_LINE_WIDTH + 2
            _draw_lane_polyline(canvas, lane_pts_np, GT_COLOR_BGR, thickness)
        gt_order = sorted(
            range(len(gt_pts)),
            key=lambda i: float(np.min(gt_pts[i][:, 1])) if gt_pts[i].size else 1e9,
        )
        for i in gt_order[: int(LANE_LABEL_MAX_PER_GROUP)]:
            pts = gt_pts[i]
            if pts.shape[0] < 2:
                continue
            anchor = _lane_anchor_point(pts)
            if anchor is None:
                continue
            cat = int(gt_stats.gt_cats[i]) if gt_stats.gt_cats else 0
            canvas = _apply_lane_label(
                canvas,
                occupied,
                anchor,
                f"GT 真值 类别{cat}:{_lane_label_for_cat(cat)}",
                GT_COLOR_BGR,
                LANE_LABEL_FONT_SIZE,
            )
        return canvas

    if mode == "m1":
        stats, pts_list, color = m1_stats, m1_pts, MODEL1_COLOR_BGR
        name = m1_display
    elif mode == "m2":
        stats, pts_list, color = m2_stats, m2_pts, MODEL2_COLOR_BGR
        name = m2_display
    else:
        raise ValueError(f"unknown mode: {mode}")

    for i, lane_pts_np in enumerate(pts_list):
        if lane_pts_np.shape[0] < 2:
            continue
        is_tp = bool(stats.pred_is_tp[i]) if i < len(stats.pred_is_tp) else False
        thickness = (
            MODEL1_LINE_WIDTH if mode == "m1" else MODEL2_LINE_WIDTH
        ) + (1 if (HIGHLIGHT_FP_FN and (not is_tp)) else 0)
        _draw_lane_polyline(canvas, lane_pts_np, color, thickness)

    n = len(pts_list)
    order = list(range(n))
    if len(stats.pred_confs) == n:
        order.sort(key=lambda k: float(stats.pred_confs[k]), reverse=True)
    for i in order[: int(LANE_LABEL_MAX_PER_GROUP)]:
        pts = pts_list[i]
        if pts.shape[0] < 2:
            continue
        anchor = _lane_anchor_point(pts)
        if anchor is None:
            continue
        cat = int(stats.pred_cats[i]) if i < len(stats.pred_cats) else -1
        is_tp = bool(stats.pred_is_tp[i]) if i < len(stats.pred_is_tp) else False
        tag = (
            "FP"
            if not is_tp
            else ("MIS" if i in (stats.pred_cat_mismatch_indices or []) else "TP")
        )
        conf_s = (
            f"  conf={float(stats.pred_confs[i]):.2f}"
            if stats.pred_confs and i < len(stats.pred_confs)
            else ""
        )
        canvas = _apply_lane_label(
            canvas,
            occupied,
            anchor,
            f"{name} {tag} 预测{cat}:{_lane_label_for_cat(cat)}{conf_s}",
            color,
            LANE_LABEL_FONT_SIZE,
        )
    return canvas


def compose_grid_row(
    row_title: str,
    cells: List[np.ndarray],
    cell_w: int,
    cell_h: int,
    bar_h: int = 44,
) -> np.ndarray:
    """一行：上方中文行标题 + 横向拼接的等尺寸 cell。"""
    resized = []
    for c in cells:
        resized.append(cv2.resize(c, (cell_w, cell_h)))
    body = cv2.hconcat(resized)
    bar = np.zeros((bar_h, body.shape[1], 3), dtype=np.uint8)
    bar[:] = (32, 32, 32)
    fs = max(16, HEADER_FONT_SIZE - 2)
    tw, th = _pil_text_size(row_title, fs)
    tx = max(8, (body.shape[1] - tw) // 2)
    ty = max(6, (bar_h - th) // 2)
    bar = _pil_draw_text(bar, (tx, ty), row_title, fs, (255, 255, 255))
    return cv2.vconcat([bar, body])


# ---------- 论文风格：RGB+GT / 二值掩码（错类红线）+ 左侧竖排小字 ----------


def extract_segment_id_from_img_path(img_path: str) -> Optional[str]:
    """从 OpenLane 路径中抽出 segment-xxx 子场景名。"""
    if not img_path:
        return None
    m = re.search(r"(segment-[0-9A-Za-z_\-]+)", img_path.replace("\\", "/"))
    return m.group(1) if m else None


def frame_time_sort_key_for_data_info(data_info: Dict[str, Any]) -> Tuple[int, str]:
    """按时间序排序帧：优先取文件名中最长连续数字为时间戳键。"""
    path = _img_path_from_data_info(data_info) or ""
    base = os.path.basename(path)
    nums = re.findall(r"\d+", base)
    if nums:
        k = max(nums, key=len)
        try:
            return (int(k), base)
        except ValueError:
            pass
    return (0, base)


def sort_indices_temporally(dataset, indices: List[int]) -> List[int]:
    return sorted(indices, key=lambda i: frame_time_sort_key_for_data_info(dataset.data_infos[i]))


def group_indices_by_segment(dataset, indices: List[int]) -> Dict[str, List[int]]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i in indices:
        p = _img_path_from_data_info(dataset.data_infos[i]) or ""
        seg = extract_segment_id_from_img_path(p)
        if seg:
            buckets[seg].append(i)
    return dict(buckets)


def render_paper_cell_rgb_with_gt(
    img_bgr: np.ndarray,
    gt_stats: FrameMatchStats,
    crop_top_px: int,
    line_width: int = 3,
) -> np.ndarray:
    """裁剪后原图，仅画真值车道线（绿色），无任何文字。"""
    crop = min(int(crop_top_px), img_bgr.shape[0] - 2) if crop_top_px > 0 else 0
    canvas = img_bgr[crop:, :, :].copy() if crop > 0 else img_bgr.copy()
    for lane in gt_stats.gt_lane_points:
        pts = _shift_points_after_crop(np.asarray(lane, dtype=np.float32), crop)
        if pts.shape[0] >= 2:
            _draw_lane_polyline(canvas, pts, GT_COLOR_BGR, line_width)
    return canvas


def render_paper_cell_pred_mask(
    out_hw: Tuple[int, int],
    pred_stats: FrameMatchStats,
    crop_top_px: int,
) -> np.ndarray:
    """
    黑底；预测车道为白线；与真值 IoU 匹配但类别错误为红线（pred_cat_mismatch_indices）。
    """
    h, w = int(out_hw[0]), int(out_hw[1])
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mismatch = set(pred_stats.pred_cat_mismatch_indices or [])
    white = (255, 255, 255)
    wrong = (0, 0, 255)  # BGR 红
    crop = int(crop_top_px)
    for i, lane in enumerate(pred_stats.pred_lane_points):
        pts = _shift_points_after_crop(np.asarray(lane, dtype=np.float32), crop)
        if pts.shape[0] < 2:
            continue
        col = wrong if i in mismatch else white
        _draw_lane_polyline(mask, pts, col, PAPER_MASK_LINE_WIDTH)
    return mask


def _paper_left_label_strip_bgr(height: int, width: int, text: str, font_size: int = 26) -> np.ndarray:
    """白底左侧行名：整段文字横向绘制后旋转 90°，再缩放居中，便于阅读且字号足够大。"""
    strip = np.full((height, width, 3), 255, dtype=np.uint8)
    if not text or Image is None or ImageDraw is None or ImageFont is None:
        return strip
    fp = _pick_font_path()
    try:
        font = ImageFont.truetype(fp, font_size) if fp else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    probe = Image.new("RGB", (8, 32))
    dr0 = ImageDraw.Draw(probe)
    bbox = dr0.textbbox((0, 0), text, font=font)
    tw = max(1, bbox[2] - bbox[0] + 12)
    th = max(1, bbox[3] - bbox[1] + 12)
    horiz = Image.new("RGB", (tw, th), (255, 255, 255))
    dr = ImageDraw.Draw(horiz)
    dr.text((6, 6), text, font=font, fill=(0, 0, 0))
    vert = horiz.rotate(90, expand=True, fillcolor=(255, 255, 255))
    vw, vh = vert.size
    if vw <= 0 or vh <= 0:
        return strip
    inner_w = max(1, width - 4)
    inner_h = max(1, height - 4)
    scale = min(inner_w / float(vw), inner_h / float(vh)) * 0.95
    nw = max(1, int(round(vw * scale)))
    nh = max(1, int(round(vh * scale)))
    vert = vert.resize((nw, nh), Image.Resampling.LANCZOS)
    strip_rgb = Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
    x = (width - nw) // 2
    y = (height - nh) // 2
    strip_rgb.paste(vert, (x, y))
    return cv2.cvtColor(np.asarray(strip_rgb), cv2.COLOR_RGB2BGR)


def compose_paper_grid_tight(
    row_specs: List[Tuple[str, List[np.ndarray]]],
    cell_w: int,
    cell_h: int,
    label_strip_w: int = 64,
    label_font_size: int = 26,
) -> np.ndarray:
    """
    多行拼图：左侧竖排行名 + 紧密拼接的格子（无顶栏、格内无字）。
    row_specs: [(left_label, [cell_bgr,...]), ...]
    """
    rows_out: List[np.ndarray] = []
    for left_label, cells in row_specs:
        resized = [cv2.resize(c, (cell_w, cell_h)) for c in cells]
        body = cv2.hconcat(resized) if resized else np.zeros((cell_h, 1, 3), np.uint8)
        h = body.shape[0]
        strip = _paper_left_label_strip_bgr(h, label_strip_w, left_label, font_size=label_font_size)
        rows_out.append(cv2.hconcat([strip, body]))
    return cv2.vconcat(rows_out) if rows_out else np.zeros((1, 1, 3), np.uint8)


def pick_column_indices_even(sorted_sel: List[int], n: int) -> List[int]:
    """在有序下标上均匀取 n 列（时间跨度）。"""
    u = len(sorted_sel)
    if n <= 0 or u == 0:
        return []
    if u <= n:
        return list(sorted_sel)
    if n == 1:
        return [sorted_sel[u // 2]]
    return [sorted_sel[int(round(i * (u - 1) / (n - 1)))] for i in range(n)]


def _build_legend_lines(
    gt_stats: FrameMatchStats,
    pred_stats: FrameMatchStats,
    model_name: str,
) -> List[Tuple[str, Tuple[int, int, int]]]:
    """
    返回 (text, color_bgr) 列表，供天空区域集中绘制。
    """
    lines: List[Tuple[str, Tuple[int, int, int]]] = []
    # GT
    for i, cat in enumerate(gt_stats.gt_cats):
        is_tp = gt_stats.gt_is_tp[i] if gt_stats.gt_is_tp else False
        tag = "TP" if is_tp else "FN"
        lines.append(
            (f"GT {tag}  类别{int(cat)}:{_lane_label_for_cat(int(cat))}", GT_COLOR_BGR)
        )

    # Pred（按 conf 从高到低）
    order = list(range(len(pred_stats.pred_cats)))
    if pred_stats.pred_confs:
        order.sort(key=lambda i: float(pred_stats.pred_confs[i]), reverse=True)
    for i in order:
        cat = int(pred_stats.pred_cats[i])
        is_tp = pred_stats.pred_is_tp[i] if pred_stats.pred_is_tp else False
        if not is_tp:
            tag = "FP"
        elif i in (pred_stats.pred_cat_mismatch_indices or []):
            tag = "MIS"
        else:
            tag = "TP"
        conf = (
            f"  conf={float(pred_stats.pred_confs[i]):.2f}"
            if pred_stats.pred_confs
            else ""
        )
        lines.append(
            (
                f"{model_name} {tag}  类别{cat}:{_lane_label_for_cat(cat)}{conf}",
                MODEL1_COLOR_BGR,
            )
        )
    return lines


def _draw_label_at_lane(
    img_bgr: np.ndarray,
    pts_xy: np.ndarray,
    text: str,
    color_bgr: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    pts = np.asarray(pts_xy, dtype=np.int32)
    if pts.shape[0] < 1:
        return
    # 放到“最上端”(最小 y)附近，尽量避免挡住车辆
    top_idx = int(np.argmin(pts[:, 1]))
    x, y = int(pts[top_idx, 0]), int(pts[top_idx, 1])
    x = max(0, min(img_bgr.shape[1] - 1, x))
    y = max(0, min(img_bgr.shape[0] - 1, y))

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # 背景框
    x0 = max(0, x - tw // 2 - 2)
    y0 = max(0, y - th - 4)
    x1 = min(img_bgr.shape[1] - 1, x0 + tw + 4)
    y1 = min(img_bgr.shape[0] - 1, y0 + th + baseline + 4)
    cv2.rectangle(img_bgr, (x0, y0), (x1, y1), color_bgr, thickness=-1)
    # 文本用白色提升对比度
    cv2.putText(
        img_bgr,
        text,
        (x0 + 2, y1 - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_failure_comparison(
    out_path: str,
    img_bgr: np.ndarray,
    gt_stats: FrameMatchStats,  # shared gt from either model
    m1_stats: FrameMatchStats,
    m2_stats: FrameMatchStats,
    m1_name: str,
    m2_name: str,
) -> None:
    """
    生成单张叠加图：
      - GT: 绿色
      - Model1: 红色
      - Model2: 蓝色
    """
    crop = int(CROP_TOP_PX)
    canvas, gt_pts, m1_pts, m2_pts = _crop_and_shift_lanes(
        img_bgr, crop, gt_stats, m1_stats, m2_stats
    )

    # 1) 顶部标题条（放大 + 中文）
    header_h = 44
    _alpha_box(canvas, 0, 0, canvas.shape[1], header_h, (0, 0, 0), alpha=0.85)
    header_text = (
        f"GT(绿) | {m1_name}(红) | {m2_name}(蓝) | IoU阈值={IOU_THRESHOLD}, 宽度={IOU_WIDTH}"
    )
    canvas = _pil_draw_text(canvas, (12, 8), header_text, HEADER_FONT_SIZE, (255, 255, 255))

    # 2) 画线
    for i, lane_pts_np in enumerate(gt_pts):
        if lane_pts_np.shape[0] < 2:
            continue
        is_tp = gt_stats.gt_is_tp[i] if gt_stats.gt_is_tp else False
        thickness = GT_LINE_WIDTH if (is_tp or not HIGHLIGHT_FP_FN) else GT_LINE_WIDTH + 2
        _draw_lane_polyline(canvas, lane_pts_np, GT_COLOR_BGR, thickness)

    for i, lane_pts_np in enumerate(m1_pts):
        if lane_pts_np.shape[0] < 2:
            continue
        is_tp = m1_stats.pred_is_tp[i] if m1_stats.pred_is_tp else False
        thickness = MODEL1_LINE_WIDTH + (1 if (HIGHLIGHT_FP_FN and (not is_tp)) else 0)
        _draw_lane_polyline(canvas, lane_pts_np, MODEL1_COLOR_BGR, thickness)

    for i, lane_pts_np in enumerate(m2_pts):
        if lane_pts_np.shape[0] < 2:
            continue
        is_tp = m2_stats.pred_is_tp[i] if m2_stats.pred_is_tp else False
        thickness = MODEL2_LINE_WIDTH + (1 if (HIGHLIGHT_FP_FN and (not is_tp)) else 0)
        _draw_lane_polyline(canvas, lane_pts_np, MODEL2_COLOR_BGR, thickness)

    # 3) 将类别标注贴在各自车道线旁（放大 + 避免重叠）
    occupied: List[Tuple[int, int, int, int]] = []
    # 避免遮挡顶部标题条
    occupied.append((0, 0, canvas.shape[1], header_h + 2))

    def draw_one_label(anchor: Tuple[int, int], text: str, color_bgr: Tuple[int, int, int]) -> None:
        placed = _place_label_box(
            anchor_xy=anchor,
            text=text,
            font_size=LANE_LABEL_FONT_SIZE,
            img_w=canvas.shape[1],
            img_h=canvas.shape[0],
            occupied=occupied,
        )
        if placed is None:
            return
        x0, y0, x1, y1, tx, ty = placed
        _alpha_box(canvas, x0, y0, x1, y1, color_bgr, alpha=float(LANE_LABEL_BOX_ALPHA))
        # 文本统一白色，保证对比度
        nonlocal_canvas = _pil_draw_text(canvas, (tx, ty), text, LANE_LABEL_FONT_SIZE, (255, 255, 255))
        canvas[:] = nonlocal_canvas

    # GT 标签（按 y 从小到大：先标更靠上的，减少遮挡）
    gt_order = sorted(range(len(gt_pts)), key=lambda i: float(np.min(gt_pts[i][:, 1])) if gt_pts[i].size else 1e9)
    for j, i in enumerate(gt_order[: int(LANE_LABEL_MAX_PER_GROUP)]):
        pts = gt_pts[i]
        if pts.shape[0] < 2:
            continue
        anchor = _lane_anchor_point(pts)
        if anchor is None:
            continue
        cat = int(gt_stats.gt_cats[i]) if gt_stats.gt_cats else 0
        draw_one_label(anchor, f"GT 类别{cat}:{_lane_label_for_cat(cat)}", GT_COLOR_BGR)

    # Model1（红）
    m1_order = list(range(len(m1_pts)))
    if m1_stats.pred_confs:
        m1_order.sort(key=lambda i: float(m1_stats.pred_confs[i]), reverse=True)
    for i in m1_order[: int(LANE_LABEL_MAX_PER_GROUP)]:
        pts = m1_pts[i]
        if pts.shape[0] < 2:
            continue
        anchor = _lane_anchor_point(pts)
        if anchor is None:
            continue
        cat = int(m1_stats.pred_cats[i]) if m1_stats.pred_cats else -1
        is_tp = m1_stats.pred_is_tp[i] if m1_stats.pred_is_tp else False
        tag = "FP" if not is_tp else ("MIS" if i in (m1_stats.pred_cat_mismatch_indices or []) else "TP")
        draw_one_label(anchor, f"{m1_name} {tag} 类别{cat}:{_lane_label_for_cat(cat)}", MODEL1_COLOR_BGR)

    # Model2（蓝）
    m2_order = list(range(len(m2_pts)))
    if m2_stats.pred_confs:
        m2_order.sort(key=lambda i: float(m2_stats.pred_confs[i]), reverse=True)
    for i in m2_order[: int(LANE_LABEL_MAX_PER_GROUP)]:
        pts = m2_pts[i]
        if pts.shape[0] < 2:
            continue
        anchor = _lane_anchor_point(pts)
        if anchor is None:
            continue
        cat = int(m2_stats.pred_cats[i]) if m2_stats.pred_cats else -1
        is_tp = m2_stats.pred_is_tp[i] if m2_stats.pred_is_tp else False
        tag = "FP" if not is_tp else ("MIS" if i in (m2_stats.pred_cat_mismatch_indices or []) else "TP")
        draw_one_label(anchor, f"{m2_name} {tag} 类别{cat}:{_lane_label_for_cat(cat)}", MODEL2_COLOR_BGR)

    _safe_mkdir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)


def _run_inference_on_subset(
    model,
    dataloader,
    device: str,
) -> List[List[Any]]:
    model.eval()
    pred_all: List[List[Any]] = []
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs)
            if hasattr(model, "get_lanes"):
                lanes_list = model.get_lanes(outputs)
            else:
                lanes_list = outputs
            pred_all.extend(lanes_list)
    return pred_all


def _failure_sort_key(stats: FrameMatchStats) -> Tuple[int, int, int, float]:
    # 越“差”越靠前，因此 key 用 (fn, fp, cat_wrong_tp, -cat_acc)
    return (stats.fn, stats.fp, stats.cat_wrong_tp, -stats.cat_acc)


def main() -> None:
    repo_root = _get_repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # ----------- 0. 加载 config / dataset -----------
    cfg = LazyConfig.load(CFG_PATH)
    param_cfg = getattr(cfg, "param_config", None)
    if param_cfg is None:
        raise ValueError(f"cfg 中找不到 param_config，请检查 {CFG_PATH}")

    # 数据根目录（用于加载子场景列表）
    data_root = getattr(param_cfg, "data_root", None)
    if not data_root:
        # fallback：从 dataset.data_root 取
        try:
            test_dataset = instantiate(cfg.dataloader.test).dataset
            data_root = test_dataset.data_root
        except Exception:
            data_root = None
    if not data_root:
        raise ValueError("无法推断 data_root，请确保 config 正确配置。")

    # ----------- 1. 构造“要看的子集帧”索引 -----------
    # 只实例化 dataset（不直接用原始 cfg.dataloader.test，因为要对子集抽样）
    test_dataset = instantiate(cfg.dataloader.test.dataset)

    indices_per_seq: Dict[str, List[int]] = {}
    for seq_name in SEQUENCE_NAMES:
        sel = _match_indices_for_sequence(test_dataset, seq_name, data_root=data_root)
        sel = sorted(set(sel))
        sel = _filter_indices_by_path_substr(test_dataset, sel, SEGMENT_PATH_CONTAINS)
        sel = _subsample_evenly(sel, MAX_FRAMES_PER_SEQUENCE)
        indices_per_seq[seq_name] = sel
        print(f"[SEQ] {seq_name}: selected {len(sel)} frames (after subsample).")

    union_indices = sorted(set(i for v in indices_per_seq.values() for i in v))
    if not union_indices:
        raise RuntimeError("没有匹配到任何帧：请检查 SEQUENCE_NAMES 或 data_root/list/test_split/test 中的文件名。")
    print(f"[SUBSET] union frames: {len(union_indices)}")

    # 子集 dataloader
    subset_dataset = torch.utils.data.Subset(test_dataset, union_indices)
    subset_dataloader = build_batch_data_loader(
        dataset=subset_dataset,
        total_batch_size=SUBSET_TOTAL_BATCH_SIZE,
        num_workers=SUBSET_NUM_WORKERS,
        drop_last=False,
        shuffle=False,
    )

    # ----------- 2. 分别加载两模型并推理 -----------
    m1_name = os.path.splitext(os.path.basename(MODEL1_CKPT_PATH))[0] or "model1"
    m2_name = os.path.splitext(os.path.basename(MODEL2_CKPT_PATH))[0] or "model2"

    print("[MODEL] Loading model1 ...")
    model1 = instantiate(cfg.model)
    model1.to(DEVICE)
    model1.eval()
    Checkpointer(model1).load(MODEL1_CKPT_PATH)

    print("[MODEL] Running inference for model1 ...")
    pred1 = _run_inference_on_subset(model1, subset_dataloader, device=DEVICE)

    cfg2_path = MODEL2_CFG_PATH or CFG_PATH
    cfg2 = LazyConfig.load(cfg2_path) if cfg2_path != CFG_PATH else cfg
    pc2 = getattr(cfg2, "param_config", None)
    if pc2 is not None and getattr(pc2, "data_root", None) != data_root:
        print(
            f"[WARN] 模型2 config 的 data_root 与模型1 不一致: {getattr(pc2, 'data_root', None)} vs {data_root}"
        )

    print("[MODEL] Loading model2 ...")
    model2 = instantiate(cfg2.model)
    model2.to(DEVICE)
    model2.eval()
    Checkpointer(model2).load(MODEL2_CKPT_PATH)

    print("[MODEL] Running inference for model2 ...")
    pred2 = _run_inference_on_subset(model2, subset_dataloader, device=DEVICE)

    if len(pred1) != len(pred2) or len(pred1) != len(union_indices):
        raise RuntimeError(
            f"pred length mismatch: len(pred1)={len(pred1)}, len(pred2)={len(pred2)}, union={len(union_indices)}"
        )

    # ----------- 3. 逐帧计算客观失败原因（TP/FP/FN + 分类错误） -----------
    # local_idx -> FrameMatchStats
    pred_stats_m1: List[FrameMatchStats] = []
    pred_stats_m2: List[FrameMatchStats] = []

    for local_idx, orig_idx in enumerate(union_indices):
        data_info = test_dataset.data_infos[orig_idx]
        gt_lanes = data_info.get("lanes", [])
        gt_cats = data_info.get("lane_categories", [-1] * len(gt_lanes))
        pred_stats_m1.append(
            _compute_frame_match_stats(
                pred_lanes=pred1[local_idx],
                gt_lanes=gt_lanes,
                param_cfg=param_cfg,
                gt_cats=gt_cats,
                iou_threshold=IOU_THRESHOLD,
                iou_width=IOU_WIDTH,
            )
        )
        # 几何解码以各自训练 param_config 为准（通常 ori/sample_y 一致；不一致时避免画线偏移）
        param_for_m2 = getattr(cfg2, "param_config", None) or param_cfg
        pred_stats_m2.append(
            _compute_frame_match_stats(
                pred_lanes=pred2[local_idx],
                gt_lanes=gt_lanes,
                param_cfg=param_for_m2,
                gt_cats=gt_cats,
                iou_threshold=IOU_THRESHOLD,
                iou_width=IOU_WIDTH,
            )
        )

    # ----------- 4. 按“序列/场景”筛选失败案例并绘制对比图 -----------
    _safe_mkdir(OUT_DIR)
    orig_to_local = _build_index_mapping(union_indices)

    summary: Dict[str, Any] = {
        "config": {
            "CFG_PATH": CFG_PATH,
            "MODEL2_CFG_PATH": MODEL2_CFG_PATH,
            "MODEL1_CKPT_PATH": MODEL1_CKPT_PATH,
            "MODEL2_CKPT_PATH": MODEL2_CKPT_PATH,
            "SEGMENT_PATH_CONTAINS": SEGMENT_PATH_CONTAINS,
            "SEQUENCE_NAMES": SEQUENCE_NAMES,
            "IOU_THRESHOLD": IOU_THRESHOLD,
            "IOU_WIDTH": IOU_WIDTH,
            "MAX_FRAMES_PER_SEQUENCE": MAX_FRAMES_PER_SEQUENCE,
            "TOPK_FAILURES_TO_DRAW_PER_SEQUENCE": TOPK_FAILURES_TO_DRAW_PER_SEQUENCE,
        },
        "per_sequence": {},
    }

    for seq_name in SEQUENCE_NAMES:
        seq_orig_indices = indices_per_seq.get(seq_name, [])
        seq_local_indices = [orig_to_local[i] for i in seq_orig_indices if i in orig_to_local]
        if not seq_local_indices:
            continue

        # 失败排序（分别针对两个模型）
        worst_m1 = sorted(
            seq_local_indices,
            key=lambda li: _failure_sort_key(pred_stats_m1[li]),
            reverse=True,
        )[:TOPK_FAILURES_TO_DRAW_PER_SEQUENCE]
        worst_m2 = sorted(
            seq_local_indices,
            key=lambda li: _failure_sort_key(pred_stats_m2[li]),
            reverse=True,
        )[:TOPK_FAILURES_TO_DRAW_PER_SEQUENCE]

        # 合并要绘制的帧：失败帧优先 + 随机少量对照
        selected_local: Set[int] = set(worst_m1) | set(worst_m2)
        random.seed(0)
        if ALSO_DRAW_RANDOM > 0:
            remaining = [li for li in seq_local_indices if li not in selected_local]
            random.shuffle(remaining)
            selected_local.update(remaining[:ALSO_DRAW_RANDOM])

        selected_local_sorted = sorted(selected_local)

        seq_out_dir = (
            os.path.join(OUT_DIR, seq_name) if APPEND_SCENARIO_SUBDIR else OUT_DIR
        )
        _safe_mkdir(seq_out_dir)

        # 导出客观失败榜单（CSV/JSON 不用额外依赖，这里直接 JSON）
        ranked = []
        for li in sorted(
            seq_local_indices,
            key=lambda x: _failure_sort_key(pred_stats_m1[x]),
            reverse=True,
        ):
            orig_idx = union_indices[li]
            data_info = test_dataset.data_infos[orig_idx]
            img_path = _img_path_from_data_info(data_info) or ""
            ranked.append(
                {
                    "local_idx": int(li),
                    "dataset_idx": int(orig_idx),
                    "img_path": img_path,
                    "m1": {
                        "tp": pred_stats_m1[li].tp,
                        "fp": pred_stats_m1[li].fp,
                        "fn": pred_stats_m1[li].fn,
                        "cat_acc": pred_stats_m1[li].cat_acc,
                        "cat_wrong_tp": pred_stats_m1[li].cat_wrong_tp,
                    },
                    "m2": {
                        "tp": pred_stats_m2[li].tp,
                        "fp": pred_stats_m2[li].fp,
                        "fn": pred_stats_m2[li].fn,
                        "cat_acc": pred_stats_m2[li].cat_acc,
                        "cat_wrong_tp": pred_stats_m2[li].cat_wrong_tp,
                    },
                }
            )

        with open(os.path.join(seq_out_dir, "failure_ranked.json"), "w") as f:
            json.dump(ranked, f, indent=2, ensure_ascii=False)

        # 绘制
        drawn = 0
        for li in selected_local_sorted:
            orig_idx = union_indices[li]
            data_info = test_dataset.data_infos[orig_idx]
            img_path = _img_path_from_data_info(data_info)
            if not img_path or not os.path.exists(img_path):
                continue
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            # 用 m1 的 gt_stats 作为共享真值统计（gt 对于两个模型应该一致）
            gt_stats = pred_stats_m1[li]
            m1_stats = pred_stats_m1[li]
            m2_stats = pred_stats_m2[li]

            img_name = _img_name_from_data_info(data_info)
            safe_name = img_name.replace("\\", "_").replace("/", "_")
            out_path = os.path.join(seq_out_dir, f"{safe_name}__compare.png")

            _draw_failure_comparison(
                out_path=out_path,
                img_bgr=img_bgr,
                gt_stats=gt_stats,
                m1_stats=m1_stats,
                m2_stats=m2_stats,
                m1_name=m1_name,
                m2_name=m2_name,
            )
            drawn += 1
        print(f"[DRAW] {seq_name}: drew {drawn} images to {seq_out_dir}")

        # 简单统计：在“最坏的 K 帧”里，统计类别错分情况（仅对 matched TP）
        mismatch_pair_counts: Dict[str, int] = {}
        top_local = selected_local_sorted[:TOPK_FAILURES_TO_DRAW_PER_SEQUENCE]
        for li in top_local:
            # 对 m1
            st1 = pred_stats_m1[li]
            for pr_lane_idx in st1.pred_cat_mismatch_indices:
                pr_cat = st1.pred_cats[pr_lane_idx]
                # 反查 matched gt：需要比较 cat 是否一致，所以我们用离散 IoU 的配对信息不保留
                # 这里为了“客观但轻量”，只统计“被错判的预测类别”频次。
                key = f"m1_pred_cat={int(pr_cat)}->{_lane_label_for_cat(int(pr_cat))}"
                mismatch_pair_counts[key] = mismatch_pair_counts.get(key, 0) + 1
            # 对 m2
            st2 = pred_stats_m2[li]
            for pr_lane_idx in st2.pred_cat_mismatch_indices:
                pr_cat = st2.pred_cats[pr_lane_idx]
                key = f"m2_pred_cat={int(pr_cat)}->{_lane_label_for_cat(int(pr_cat))}"
                mismatch_pair_counts[key] = mismatch_pair_counts.get(key, 0) + 1

        summary["per_sequence"][seq_name] = {
            "selected_frames_after_subsample": len(seq_orig_indices),
            "drawn_images": drawn,
            "top_mismatch_pred_cat_counts": dict(
                sorted(mismatch_pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
            ),
        }

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] 输出已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()

