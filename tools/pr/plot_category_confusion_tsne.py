import os
import sys
import json
import math
from typing import List, Optional, Tuple, Dict, Any

# 允许直接用 `python tools/pr/xxx.py` 运行（否则 sys.path[0]=tools/pr，找不到顶层的 config/ 与 unlanedet/）
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.matplotlib_font import *  # noqa: F401,F403
import matplotlib.pyplot as plt
import numpy as np

# 避免 import `unlanedet`（会触发编译扩展/模型依赖），这里直接内置 OpenLane 官方类别映射（并改为中文）
LANE_CATEGORIES = {
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


# ====== 配置区（只改这里即可运行） ======
PR_RESULT_DIR = (
    "output/llanetv1/openlane1000/category/clrnet_combined_resnet34/pr_curves"
)


OUT_DIR = os.path.join(PR_RESULT_DIR, "category")
os.makedirs(OUT_DIR, exist_ok=True)

# ====== t-SNE 绘图参数（可按需调整） ======
# 每个类别的文字标注字体大小（点）
TSNE_LABEL_FONTSIZE = 14
# 每个点的大小
TSNE_MARKER_SIZE = 70
# 给绘图区预留的边距比例，防止文字标注被裁剪/跑出边界
TSNE_PADDING_RATIO = 0.18
# 标注背景框（降低文字与点重叠时的可读性）
TSNE_LABEL_BBOX_ALPHA = 0.72
# 仅对哪些场景做 t-SNE（None 表示全部：overall/curve/...）
# 例如：TSNE_SCOPES = ["overall"]
TSNE_SCOPES = None

# 若不同类别的 t-SNE 坐标过于接近，会导致“点看起来像共用/缺失”
# 对这种情况给很小的抖动，帮助区分
TSNE_OVERLAP_EPS_RATIO = 3e-3  # 与坐标范围的比例（越小越不容易触发抖动）
TSNE_JITTER_RATIO = 3e-3  # 抖动幅度（与坐标范围比例）

# 全量跑：混淆矩阵 + t-SNE
PLOT_CONFUSION_MATRICES = True

# t-SNE 输出文件名里的目标 IoU/Width（即使不画混淆矩阵也固定命名）
TSNE_IOU_TARGET = 0.5
TSNE_WIDTH_TARGET = 30


def _class_label(cid: int) -> str:
    return LANE_CATEGORIES.get(int(cid), str(cid))


def _load_pr_metrics(pr_result_dir: str):
    path = os.path.join(pr_result_dir, "pr_metrics.json")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected pr_metrics.json format: {type(data)}")
    return data


def _pick_row(data, iou_target=0.5, width_target=30):
    """
    优先取完全匹配 (iou_target, width_target) 的条目；否则取距离最近的条目。
    """
    best = None
    best_score = None
    for d in data:
        try:
            iou = float(d.get("IoU_Threshold"))
            w = int(d.get("Width"))
        except Exception:
            continue
        score = abs(iou - float(iou_target)) + abs(w - int(width_target)) * 1e-3
        if (best is None) or (score < best_score):
            best = d
            best_score = score
    return best


def _pick_row_for_key(data, key: str, iou_target=0.5, width_target=30):
    """
    为指定 category_eval key 选择最接近 (iou_target, width_target) 的那行，
    并要求该行中 key 的 confusion_matrix 可用。
    """
    best = None
    best_score = None
    for d in data:
        blk = _get_eval_block(d, key)
        if blk is None:
            continue
        try:
            iou = float(d.get("IoU_Threshold"))
            w = int(d.get("Width"))
        except Exception:
            continue
        score = abs(iou - float(iou_target)) + abs(w - int(width_target)) * 1e-3
        if (best is None) or (score < best_score):
            best = d
            best_score = score
    if best is not None:
        return best
    return _pick_row(data, iou_target=iou_target, width_target=width_target)


def _get_eval_block(row: dict, key: str):
    blk = row.get(key)
    if not blk or not isinstance(blk, dict):
        return None
    if "confusion_matrix" not in blk or "num_classes" not in blk:
        return None
    return blk


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels,
    title: str,
    out_path: str,
    normalize: Optional[str],
):
    """
    normalize:
      - None: 原始计数
      - "true": 按 GT 行归一化
      - "pred": 按预测列归一化
    """
    cm = cm.astype(np.float64)
    disp = cm
    if normalize == "true":
        denom = disp.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        disp = disp / denom
    elif normalize == "pred":
        denom = disp.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        disp = disp / denom

    fig_w = max(8, 0.45 * len(labels))
    fig_h = max(7, 0.40 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(disp, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_xlabel("预测类别", fontname="AR PL UMing CN")
    ax.set_ylabel("GT 类别", fontname="AR PL UMing CN")
    ax.set_title(title, fontname="AR PL UMing CN")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 只在矩阵较小时标数值，避免太挤
    if len(labels) <= 20:
        thresh = (disp.max() if disp.size else 0) * 0.6
        fmt = ".2f" if normalize is not None else "d"
        for i in range(disp.shape[0]):
            for j in range(disp.shape[1]):
                val = disp[i, j]
                txt = format(val, fmt) if normalize is not None else str(int(cm[i, j]))
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=7,
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def _per_class_metrics_by_id(category_eval: dict):
    if not category_eval or "per_class" not in category_eval:
        return {}
    out = {}
    for item in category_eval.get("per_class", []):
        cid = item.get("class_id", None)
        if cid is None:
            continue
        out[int(cid)] = item
    return out


def _infer_class_ids_from_any(data, category_key="Category_Eval", drop_unknown=True):
    for d in data:
        blk = d.get(category_key)
        if not blk or "per_class" not in blk:
            continue
        ids = []
        for x in blk.get("per_class", []):
            try:
                cid = int(x.get("class_id"))
            except Exception:
                continue
            if drop_unknown and cid <= 0:
                continue
            ids.append(cid)
        if ids:
            return sorted(set(ids))
    # fallback：OpenLane 默认 0..14，其中 0 为 unknown
    return list(range(1, 15)) if drop_unknown else list(range(0, 15))


def _collect_tsne_features(data, category_key: str, class_ids: List[int]):
    """
    为每个类别拼接特征向量：
      - Width=30, IoU 扫描：按 IoU 升序，拼接 [P,R,F1]
      - IoU=0.5, Width 扫描：按 Width 升序，拼接 [P,R,F1]
    返回:
      X: (C, D) float
      x_meta: dict 记录使用的 IoU/Width 序列长度
    """
    # IoU sweep @ width=30
    iou_rows = []
    for d in data:
        if int(d.get("Width", -1)) != 30:
            continue
        if category_key not in d:
            continue
        try:
            iou = float(d.get("IoU_Threshold"))
        except Exception:
            continue
        iou_rows.append((iou, d))
    iou_rows.sort(key=lambda x: x[0])

    # Width sweep @ iou=0.5
    width_rows = []
    for d in data:
        try:
            iou = float(d.get("IoU_Threshold"))
        except Exception:
            continue
        if abs(iou - 0.5) > 1e-6:
            continue
        if category_key not in d:
            continue
        try:
            w = int(d.get("Width"))
        except Exception:
            continue
        width_rows.append((w, d))
    width_rows.sort(key=lambda x: x[0])

    if not iou_rows and not width_rows:
        return None, None

    feats = []
    for cid in class_ids:
        v = []
        for _, row in iou_rows:
            blk = row.get(category_key, {})
            pm = _per_class_metrics_by_id(blk).get(cid, {})
            v.extend(
                [
                    float(pm.get("precision", 0.0)),
                    float(pm.get("recall", 0.0)),
                    float(pm.get("f1", 0.0)),
                ]
            )
        for _, row in width_rows:
            blk = row.get(category_key, {})
            pm = _per_class_metrics_by_id(blk).get(cid, {})
            v.extend(
                [
                    float(pm.get("precision", 0.0)),
                    float(pm.get("recall", 0.0)),
                    float(pm.get("f1", 0.0)),
                ]
            )
        feats.append(v)

    X = np.asarray(feats, dtype=np.float32)
    meta = {
        "num_iou_points": len(iou_rows),
        "num_width_points": len(width_rows),
        "dim": int(X.shape[1]) if X.ndim == 2 else 0,
    }
    return X, meta


def _run_tsne(X: np.ndarray, random_state=0):
    """
    优先 t-SNE；若 sklearn 不可用或样本数过少，退化为 PCA(2D)。
    """
    n = int(X.shape[0])
    if n < 3:
        # 无法做 t-SNE，直接投影到前两维（或补零）
        if X.shape[1] >= 2:
            return X[:, :2]
        out = np.zeros((n, 2), dtype=np.float32)
        if X.shape[1] == 1:
            out[:, 0] = X[:, 0]
        return out

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.manifold import TSNE

        Xn = StandardScaler().fit_transform(X)
        perplexity = min(30, max(2, n - 1))
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        )
        Y = tsne.fit_transform(Xn)
        return Y.astype(np.float32)
    except Exception:
        # PCA fallback
        Xc = X - X.mean(axis=0, keepdims=True)
        # SVD: X = U S V^T -> PCA = U[:, :2] * S[:2]
        U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
        Y = (U[:, :2] * S[:2]).astype(np.float32)
        return Y


def _plot_tsne(Y: np.ndarray, class_ids: List[int], title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 7))
    x = Y[:, 0].astype(np.float32)
    y = Y[:, 1].astype(np.float32)

    # 1) 抖动分离：避免多个类别落在几乎同一坐标，导致“看起来少点/公用一个点”
    x_range = float(np.max(x) - np.min(x))
    y_range = float(np.max(y) - np.min(y))
    base_range = max(x_range, y_range, 1e-6)
    eps = TSNE_OVERLAP_EPS_RATIO * base_range
    jitter = TSNE_JITTER_RATIO * base_range
    if jitter > 0:
        n = max(1, len(class_ids))
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)

        # 全量轻微角度抖动：保证每个点都有“独特方向”，肉眼上不会共用同一位置
        for i in range(n):
            ang = float(angles[i])
            x[i] = x[i] + math.cos(ang) * jitter
            y[i] = y[i] + math.sin(ang) * jitter

        # 叠加额外抖动：对仍然极近的点再加强一次
        for i in range(n):
            for j in range(i):
                dx = float(x[i] - x[j])
                dy = float(y[i] - y[j])
                if dx * dx + dy * dy < eps * eps:
                    ang = float(angles[i] + angles[j])
                    x[i] = x[i] + math.cos(ang) * jitter
                    y[i] = y[i] + math.sin(ang) * jitter

    # 为每个类别分配不同颜色（按类别顺序稳定）
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.linspace(0, 1, max(1, len(class_ids))))

    ax.scatter(
        x,
        y,
        s=TSNE_MARKER_SIZE,
        c=colors,
        marker="o",
        alpha=0.95,
        edgecolors="k",
        linewidths=0.25,
        zorder=5,
    )

    # 为近距离点的文字标注分配不同偏移，避免文字完全重合
    x_range = float(np.max(x) - np.min(x))
    y_range = float(np.max(y) - np.min(y))
    x_range = x_range if x_range > 1e-12 else 1.0
    y_range = y_range if y_range > 1e-12 else 1.0

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    pad_x = TSNE_PADDING_RATIO * x_range
    pad_y = TSNE_PADDING_RATIO * y_range
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

    fig_w, fig_h = fig.get_size_inches()
    # offset points -> data units (近似：points按72dpi换算)
    def _offset_points_to_data(dx_points: float, dy_points: float) -> Tuple[float, float]:
        dx_data = dx_points * x_range / (72.0 * fig_w)
        dy_data = dy_points * y_range / (72.0 * fig_h)
        return dx_data, dy_data

    # 候选偏移（单位：offset points），数量足够覆盖 14 类
    offset_candidates = [
        (10, 10),
        (10, -10),
        (-10, 10),
        (-10, -10),
        (14, 0),
        (-14, 0),
        (0, 14),
        (0, -14),
        (18, 8),
        (-18, 8),
        (18, -8),
        (-18, -8),
        (8, 18),
        (-8, 18),
    ]
    placed_label_positions: List[Tuple[float, float]] = []

    for i, cid in enumerate(class_ids):
        best_offset = offset_candidates[i % len(offset_candidates)]
        best_score = -1e30

        for dx_pt, dy_pt in offset_candidates:
            dx_data, dy_data = _offset_points_to_data(dx_pt, dy_pt)
            cand_pos = (float(x[i] + dx_data), float(y[i] + dy_data))
            out = (
                cand_pos[0] < x_min - pad_x
                or cand_pos[0] > x_max + pad_x
                or cand_pos[1] < y_min - pad_y
                or cand_pos[1] > y_max + pad_y
            )
            if out:
                score = -1e20
            else:
                if placed_label_positions:
                    dists = [
                        math.sqrt((cand_pos[0] - px) ** 2 + (cand_pos[1] - py) ** 2)
                        for (px, py) in placed_label_positions
                    ]
                    min_dist = float(min(dists)) if dists else 0.0
                    score = min_dist
                else:
                    # 第一个标注时也要参与“在边界内”的选择
                    score = 0.0

            if score > best_score:
                best_score = score
                best_offset = (dx_pt, dy_pt)

        dx_pt, dy_pt = best_offset
        dx_data, dy_data = _offset_points_to_data(dx_pt, dy_pt)
        placed_label_positions.append((float(x[i] + dx_data), float(y[i] + dy_data)))

        ann = ax.annotate(
            _class_label(cid),
            xy=(float(x[i]), float(y[i])),
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            fontsize=TSNE_LABEL_FONTSIZE,
            ha="left",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=TSNE_LABEL_BBOX_ALPHA,
            ),
            zorder=10,
            annotation_clip=True,
            clip_on=True,
        )
        # 强制文本裁剪到坐标轴范围内（避免视觉上“跑出坐标系”）
        try:
            ann.set_clip_path(ax.patch)
        except Exception:
            pass
    ax.set_title(title, fontname="AR PL UMing CN")
    ax.set_xlabel("t-SNE 维度1", fontname="AR PL UMing CN")
    ax.set_ylabel("t-SNE 维度2", fontname="AR PL UMing CN")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def main():
    data = _load_pr_metrics(PR_RESULT_DIR)
    # 从全量数据中收集所有可能的 category_eval key
    keys_set = set()
    for d in data:
        for k in d.keys():
            if k == "Category_Eval" or k.endswith("_Category_Eval"):
                keys_set.add(k)
    keys = sorted(keys_set, key=lambda x: (x != "Category_Eval", x))

    # 1) 混淆矩阵：对每个 key 选择最接近 IoU=0.5 / Width=30 且 block 可用的那行
    if PLOT_CONFUSION_MATRICES:
        for k in keys:
            row_k = _pick_row_for_key(data, k, iou_target=0.5, width_target=30)
            if row_k is None:
                continue

            iou = float(row_k.get("IoU_Threshold", 0.0))
            w = int(row_k.get("Width", 0))

            blk = _get_eval_block(row_k, k)
            if blk is None:
                continue
            cm = np.asarray(blk.get("confusion_matrix", []), dtype=np.int64)
            num_classes = int(
                blk.get("num_classes", cm.shape[0] if cm.ndim == 2 else 0)
            )
            if (
                cm.ndim != 2
                or cm.shape[0] != cm.shape[1]
                or cm.shape[0] != num_classes
            ):
                continue

            # label：默认去掉 unknown(0)
            drop_unknown = True
            if drop_unknown and num_classes >= 2:
                cm_use = cm[1:, 1:]
                labels = [_class_label(i) for i in range(1, num_classes)]
            else:
                cm_use = cm
                labels = [_class_label(i) for i in range(num_classes)]

            scope = (
                "overall"
                if k == "Category_Eval"
                else k.replace("_Category_Eval", "")
            )
            scope_dir = os.path.join(OUT_DIR, scope)
            os.makedirs(scope_dir, exist_ok=True)
            base = os.path.join(scope_dir, f"confusion_matrix_iou{iou:.2f}_w{w}")
            _plot_confusion_matrix(
                cm_use,
                labels,
                title=f"{scope} 类别混淆矩阵 (IoU={iou:.2f}, Width={w})",
                out_path=base + ".png",
                normalize=None,
            )
            _plot_confusion_matrix(
                cm_use,
                labels,
                title=f"{scope} 类别混淆矩阵-按GT归一化 (IoU={iou:.2f}, Width={w})",
                out_path=base + "_norm_true.png",
                normalize="true",
            )

    # 2) t-SNE：基于 PR 曲线扫描下 per-class 的 (P/R/F1) 序列特征
    # overall + 每个分场景各画一张
    for scope_key in keys:
        scope = (
            "overall"
            if scope_key == "Category_Eval"
            else scope_key.replace("_Category_Eval", "")
        )
        if TSNE_SCOPES is not None and scope not in TSNE_SCOPES:
            continue

        class_ids = _infer_class_ids_from_any(data, scope_key, drop_unknown=True)
        X, meta = _collect_tsne_features(data, scope_key, class_ids)
        if X is None or meta is None or X.size == 0:
            continue
        # 若某个类别全 0，稍微加一点噪声避免完全重合
        if np.allclose(X, 0):
            rng = np.random.default_rng(0)
            X = X + rng.normal(0, 1e-6, size=X.shape).astype(np.float32)

        Y = _run_tsne(X, random_state=0)
        title = (
            f"{scope} 类别 t-SNE（特征=PR曲线 per-class P/R/F1 序列；"
            f"IoU点={meta['num_iou_points']}, Width点={meta['num_width_points']}）"
        )
        scope_dir = os.path.join(OUT_DIR, scope)
        os.makedirs(scope_dir, exist_ok=True)
        out_path = os.path.join(
            scope_dir,
            f"tsne_iou{TSNE_IOU_TARGET:.2f}_w{TSNE_WIDTH_TARGET}.png",
        )
        _plot_tsne(Y, class_ids, title=title, out_path=out_path)

    print(f"可视化结果已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
