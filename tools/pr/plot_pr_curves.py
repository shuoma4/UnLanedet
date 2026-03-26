import os
import json
import re
import numpy as np
from config.matplotlib_font import *
import matplotlib.pyplot as plt

from unlanedet.data.openlane import LANE_CATEGORIES

# 与 OpenLane 一致：0 为 unknown；绘图默认画除 unknown 外所有正类 ID（一般为 1–14）
DEFAULT_PER_CLASS_PLOT_IDS = list(range(1, 15))


def _infer_per_class_plot_ids(data, category_key="Category_Eval"):
    for d in data:
        block = d.get(category_key)
        if not block or "per_class" not in block:
            continue
        ids = [int(x["class_id"]) for x in block["per_class"] if int(x["class_id"]) > 0]
        return ids if ids else DEFAULT_PER_CLASS_PLOT_IDS
    return DEFAULT_PER_CLASS_PLOT_IDS


# ====== 配置区 ======
# 结果目录
PR_RESULT_DIR = (
    "output/llanetv1/openlane1000/category/clrnet_linear_resnet34/pr_curves"
)
IMG_DIR = os.path.join(PR_RESULT_DIR, "image")
os.makedirs(IMG_DIR, exist_ok=True)


# 读取数据
with open(os.path.join(PR_RESULT_DIR, "pr_metrics.json"), "r") as f:
    data = json.load(f)

# 提取所有分场景名
scene_keys = [
    k for k in data[0].keys() if k.endswith("_F1") and not k.startswith("Total")
]


# 1. 整体PR曲线（随IoU变化，Width固定为30）
def plot_overall_pr_curve_iou():
    ious, f1s, precs, recalls = [], [], [], []
    for d in data:
        if d["Width"] == 30:
            ious.append(d["IoU_Threshold"])
            f1s.append(d["Total_F1"])
            precs.append(d["Total_Precision"])
            recalls.append(d["Total_Recall"])
    plt.figure()
    plt.plot(ious, f1s, label="F1", marker="o")
    plt.plot(ious, precs, label="Precision", marker="s")
    plt.plot(ious, recalls, label="Recall", marker="^")
    plt.xlabel("IoU 阈值", fontname="AR PL UMing CN")
    plt.ylabel("指标值", fontname="AR PL UMing CN")
    plt.title("整体PR曲线（随IoU变化, Width=30）", fontname="AR PL UMing CN")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, "overall_pr_curve_by_iou.png"), dpi=600)
    plt.close()


# 2. 整体PR曲线（随Width变化，IoU固定为0.5）
def plot_overall_pr_curve_width():
    widths, f1s, precs, recalls = [], [], [], []
    for d in data:
        if abs(d["IoU_Threshold"] - 0.5) < 1e-6:
            widths.append(d["Width"])
            f1s.append(d["Total_F1"])
            precs.append(d["Total_Precision"])
            recalls.append(d["Total_Recall"])
    # 排序，防止乱序导致直线
    pairs = sorted(zip(widths, f1s, precs, recalls))
    if not pairs:
        return
    sorted_widths, sorted_f1s, sorted_precs, sorted_recalls = zip(*pairs)
    plt.figure()
    plt.plot(sorted_widths, sorted_f1s, label="F1", marker="o")
    plt.plot(sorted_widths, sorted_precs, label="Precision", marker="s")
    plt.plot(sorted_widths, sorted_recalls, label="Recall", marker="^")
    plt.xlabel("扩展宽度", fontname="AR PL UMing CN")
    plt.ylabel("指标值", fontname="AR PL UMing CN")
    plt.title("整体PR曲线（随Width变化, IoU=0.5）", fontname="AR PL UMing CN")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, "overall_pr_curve_by_width.png"), dpi=600)
    plt.close()


# 3. 分场景PR曲线（随IoU变化，Width固定为30）
def plot_scene_pr_curve_iou():
    for scene in scene_keys:
        ious, f1s = [], []
        for d in data:
            if d["Width"] == 30 and scene in d:
                ious.append(d["IoU_Threshold"])
                f1s.append(d[scene])
        plt.figure()
        plt.plot(ious, f1s, marker="o")
        plt.xlabel("IoU 阈值", fontname="AR PL UMing CN")
        plt.ylabel("F1", fontname="AR PL UMing CN")
        plt.title(
            f"{scene.replace('_F1','')}分场景PR曲线（随IoU变化, Width=30）",
            fontname="AR PL UMing CN",
        )
        plt.grid(True)
        plt.savefig(os.path.join(IMG_DIR, f"{scene}_pr_curve_by_iou.png"), dpi=600)
        plt.close()


# 4. 分场景PR曲线（随Width变化，IoU固定为0.5）
def plot_scene_pr_curve_width():
    for scene in scene_keys:
        widths, f1s = [], []
        for d in data:
            if abs(d["IoU_Threshold"] - 0.5) < 1e-6 and scene in d:
                widths.append(d["Width"])
                f1s.append(d[scene])
        # 排序，防止乱序导致直线
        pairs = sorted(zip(widths, f1s))
        if not pairs:
            continue
        sorted_widths, sorted_f1s = zip(*pairs)
        plt.figure()
        plt.plot(sorted_widths, sorted_f1s, marker="o")
        plt.xlabel("扩展宽度", fontname="AR PL UMing CN")
        plt.ylabel("F1", fontname="AR PL UMing CN")
        plt.title(
            f"{scene.replace('_F1','')}分场景PR曲线（随Width变化, IoU=0.5）",
            fontname="AR PL UMing CN",
        )
        plt.grid(True)
        plt.savefig(os.path.join(IMG_DIR, f"{scene}_pr_curve_by_width.png"), dpi=600)
        plt.close()


# 5. 整体类别PR曲线（宏/加权平均）
def plot_overall_cat_pr_curves():
    # 检查是否有分类功能（只要有一项不为0即可）
    has_macro = any(d.get("Cat_F1_Macro", 0) > 0 for d in data)
    has_weighted = any(d.get("Cat_F1_Weighted", 0) > 0 for d in data)
    if not (has_macro or has_weighted):
        return
    # IoU sweep
    ious, macro, weighted = [], [], []
    for d in data:
        if d["Width"] == 30:
            ious.append(d["IoU_Threshold"])
            macro.append(d.get("Cat_F1_Macro", 0))
            weighted.append(d.get("Cat_F1_Weighted", 0))
    plt.figure()
    if has_macro:
        plt.plot(ious, macro, label="Cat_F1_Macro", marker="o")
    if has_weighted:
        plt.plot(ious, weighted, label="Cat_F1_Weighted", marker="s")
    plt.xlabel("IoU 阈值", fontname="AR PL UMing CN")
    plt.ylabel("F1", fontname="AR PL UMing CN")
    plt.title("整体类别PR曲线（随IoU变化, Width=30）", fontname="AR PL UMing CN")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, "overall_cat_pr_curve_by_iou.png"), dpi=600)
    plt.close()
    # Width sweep
    widths, macro, weighted = [], [], []
    for d in data:
        if abs(d["IoU_Threshold"] - 0.5) < 1e-6:
            widths.append(d["Width"])
            macro.append(d.get("Cat_F1_Macro", 0))
            weighted.append(d.get("Cat_F1_Weighted", 0))
    pairs = sorted(zip(widths, macro, weighted))
    if not pairs:
        return
    sorted_widths, sorted_macro, sorted_weighted = zip(*pairs)
    plt.figure()
    if has_macro:
        plt.plot(sorted_widths, sorted_macro, label="Cat_F1_Macro", marker="o")
    if has_weighted:
        plt.plot(sorted_widths, sorted_weighted, label="Cat_F1_Weighted", marker="s")
    plt.xlabel("扩展宽度", fontname="AR PL UMing CN")
    plt.ylabel("F1", fontname="AR PL UMing CN")
    plt.title("整体类别PR曲线（随Width变化, IoU=0.5）", fontname="AR PL UMing CN")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, "overall_cat_pr_curve_by_width.png"), dpi=600)
    plt.close()


# 6. 分场景类别PR曲线（宏/加权平均）
def plot_scene_cat_pr_curves():
    # 自动检测所有分场景名
    scene_prefixes = set()
    for k in data[0].keys():
        if "_Cat_F1_Macro" in k:
            scene_prefixes.add(k.replace("_Cat_F1_Macro", ""))
    for scene in scene_prefixes:
        # 检查该场景是否有分类功能
        has_macro = any(d.get(f"{scene}_Cat_F1_Macro", 0) > 0 for d in data)
        has_weighted = any(d.get(f"{scene}_Cat_F1_Weighted", 0) > 0 for d in data)
        if not (has_macro or has_weighted):
            continue
        # IoU sweep
        ious, macro, weighted = [], [], []
        for d in data:
            if d["Width"] == 30:
                ious.append(d["IoU_Threshold"])
                macro.append(d.get(f"{scene}_Cat_F1_Macro", 0))
                weighted.append(d.get(f"{scene}_Cat_F1_Weighted", 0))
        plt.figure()
        if has_macro:
            plt.plot(ious, macro, label="Cat_F1_Macro", marker="o")
        if has_weighted:
            plt.plot(ious, weighted, label="Cat_F1_Weighted", marker="s")
        plt.xlabel("IoU 阈值", fontname="AR PL UMing CN")
        plt.ylabel("F1", fontname="AR PL UMing CN")
        plt.title(
            f"{scene}分场景类别PR曲线（随IoU变化, Width=30）", fontname="AR PL UMing CN"
        )
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(IMG_DIR, f"{scene}_cat_pr_curve_by_iou.png"), dpi=600)
        plt.close()
        # Width sweep
        widths, macro, weighted = [], [], []
        for d in data:
            if abs(d["IoU_Threshold"] - 0.5) < 1e-6:
                widths.append(d["Width"])
                macro.append(d.get(f"{scene}_Cat_F1_Macro", 0))
                weighted.append(d.get(f"{scene}_Cat_F1_Weighted", 0))
        pairs = sorted(zip(widths, macro, weighted))
        if not pairs:
            continue
        sorted_widths, sorted_macro, sorted_weighted = zip(*pairs)
        plt.figure()
        if has_macro:
            plt.plot(sorted_widths, sorted_macro, label="Cat_F1_Macro", marker="o")
        if has_weighted:
            plt.plot(
                sorted_widths, sorted_weighted, label="Cat_F1_Weighted", marker="s"
            )
        plt.xlabel("扩展宽度", fontname="AR PL UMing CN")
        plt.ylabel("F1", fontname="AR PL UMing CN")
        plt.title(
            f"{scene}分场景类别PR曲线（随Width变化, IoU=0.5）",
            fontname="AR PL UMing CN",
        )
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(IMG_DIR, f"{scene}_cat_pr_curve_by_width.png"), dpi=600
        )
        plt.close()


# 7. 分场景十四类车道线类别PR曲线（宏/加权平均，保存到 image/category/{scene}/）
def plot_scene_category14_cat_pr_curves():
    """
    对每个分场景输出 2 张“总类别”曲线图：
      - IoU sweep（Width=30）：单张图含 3 个子图（Recall/Precision/F1），每个子图两条线（m=macro, w=weighted）
      - Width sweep（IoU=0.5）：同上

    每个分场景目录：PR_RESULT_DIR/image/category/{scene}/
    生成文件：
      - {scene}_cat14_macro_weighted_recall_precision_f1_by_iou.png
      - {scene}_cat14_macro_weighted_recall_precision_f1_by_width.png
    """

    def _compute_macro_weighted_from_category_eval(category_eval: dict, class_ids):
        """
        从 pr_metrics.json 里的 {scene}_Category_Eval.confusion_matrix/per_class 推出：
          - macro: 平均 precision/recall/f1
          - weighted: 按 GT support=tp+fn 加权 precision/recall/f1
        """
        per_class = category_eval.get("per_class", []) if category_eval else []
        pm = {int(it.get("class_id")): it for it in per_class if it.get("class_id") is not None}

        # 只对需要的类计算；通常为 1..14（drop unknown）
        rows = [pm.get(int(cid), None) for cid in class_ids]
        # 如果某些类缺失，直接当作全 0（更稳）
        vals = []
        supports = []
        for it in rows:
            if it is None:
                vals.append((0.0, 0.0, 0.0))
                supports.append(0.0)
            else:
                prec = float(it.get("precision", 0.0))
                rec = float(it.get("recall", 0.0))
                f1 = float(it.get("f1", 0.0))
                tp = float(it.get("tp", 0.0))
                fn = float(it.get("fn", 0.0))
                support = tp + fn
                vals.append((prec, rec, f1))
                supports.append(support)

        precs = [v[0] for v in vals]
        recs = [v[1] for v in vals]
        f1s = [v[2] for v in vals]

        macro_prec = float(np.mean(precs)) if precs else 0.0
        macro_rec = float(np.mean(recs)) if recs else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        s_sum = float(np.sum(supports))
        if s_sum <= 0:
            weighted_prec = 0.0
            weighted_rec = 0.0
            weighted_f1 = 0.0
        else:
            weighted_prec = float(np.sum([p * s for p, s in zip(precs, supports)]) / s_sum)
            weighted_rec = float(np.sum([r * s for r, s in zip(recs, supports)]) / s_sum)
            weighted_f1 = float(np.sum([f * s for f, s in zip(f1s, supports)]) / s_sum)

        return macro_prec, macro_rec, macro_f1, weighted_prec, weighted_rec, weighted_f1

    # 自动检测所有分场景名（例如 curve / night / ...）
    scene_prefixes = set()
    for k in data[0].keys():
        if "_Category_Eval" in k:
            scene_prefixes.add(k.replace("_Category_Eval", ""))
    # overall 对应顶层的 "Category_Eval"
    scene_prefixes.add("overall")
    class_ids = DEFAULT_PER_CLASS_PLOT_IDS

    def _plot_one(scope_key: str, scene: str, x_mode: str, out_path: str):
        """
        x_mode:
          - "iou": sweep IoU at Width=30，x=IoU
          - "width": sweep Width at IoU=0.5，x=Width
        """
        xs = []
        macro_prec = []
        macro_rec = []
        macro_f1 = []
        weighted_prec = []
        weighted_rec = []
        weighted_f1 = []

        for d in data:
            if x_mode == "iou":
                if int(d.get("Width", -1)) != 30:
                    continue
            else:
                if abs(float(d.get("IoU_Threshold", 0.0)) - 0.5) > 1e-6:
                    continue

            if scope_key not in d:
                continue
            blk = d.get(scope_key)
            if not isinstance(blk, dict) or "per_class" not in blk:
                continue

            x = float(d.get("IoU_Threshold", 0.0)) if x_mode == "iou" else int(d.get("Width", 0))
            vals = _compute_macro_weighted_from_category_eval(blk, class_ids)
            xs.append(x)
            macro_prec.append(vals[0])
            macro_rec.append(vals[1])
            macro_f1.append(vals[2])
            weighted_prec.append(vals[3])
            weighted_rec.append(vals[4])
            weighted_f1.append(vals[5])

        if not xs:
            return

        # 排序，避免乱序导致曲线折线异常
        pairs = sorted(
            zip(xs, macro_rec, macro_prec, macro_f1, weighted_rec, weighted_prec, weighted_f1),
            key=lambda t: t[0],
        )
        (
            xs_sorted,
            macro_rec_sorted,
            macro_prec_sorted,
            macro_f1_sorted,
            weighted_rec_sorted,
            weighted_prec_sorted,
            weighted_f1_sorted,
        ) = zip(*pairs)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(xs_sorted, macro_rec_sorted, label="mRecall", marker="o")
        axes[0].plot(xs_sorted, weighted_rec_sorted, label="wRecall", marker="s")
        axes[0].set_title(f"{scene} Recall", fontname="AR PL UMing CN")
        axes[0].set_xlabel("IoU 阈值" if x_mode == "iou" else "扩展宽度", fontname="AR PL UMing CN")
        axes[0].set_ylabel("Recall", fontname="AR PL UMing CN")
        axes[0].grid(True)
        axes[0].legend(fontsize=7, loc="best")

        axes[1].plot(xs_sorted, macro_prec_sorted, label="mPrecision", marker="o")
        axes[1].plot(xs_sorted, weighted_prec_sorted, label="wPrecision", marker="s")
        axes[1].set_title(f"{scene} Precision", fontname="AR PL UMing CN")
        axes[1].set_xlabel("IoU 阈值" if x_mode == "iou" else "扩展宽度", fontname="AR PL UMing CN")
        axes[1].set_ylabel("Precision", fontname="AR PL UMing CN")
        axes[1].grid(True)
        axes[1].legend(fontsize=7, loc="best")

        axes[2].plot(xs_sorted, macro_f1_sorted, label="mF1", marker="o")
        axes[2].plot(xs_sorted, weighted_f1_sorted, label="wF1", marker="s")
        axes[2].set_title(f"{scene} F1", fontname="AR PL UMing CN")
        axes[2].set_xlabel("IoU 阈值" if x_mode == "iou" else "扩展宽度", fontname="AR PL UMing CN")
        axes[2].set_ylabel("F1", fontname="AR PL UMing CN")
        axes[2].grid(True)
        axes[2].legend(fontsize=7, loc="best")

        fig.suptitle(
            f"{scene} 14类车道线类别（m/w Recall/Precision/F1）",
            fontname="AR PL UMing CN",
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=600)
        plt.close(fig)

    for scene in sorted(scene_prefixes):
        scope_key = (
            "Category_Eval" if scene == "overall" else f"{scene}_Category_Eval"
        )
        scene_img_dir = os.path.join(PR_RESULT_DIR, "image", "category", scene)
        os.makedirs(scene_img_dir, exist_ok=True)

        _plot_one(
            scope_key=scope_key,
            scene=scene,
            x_mode="iou",
            out_path=os.path.join(
                scene_img_dir,
                f"{scene}_cat14_macro_weighted_recall_precision_f1_by_iou.png",
            ),
        )
        _plot_one(
            scope_key=scope_key,
            scene=scene,
            x_mode="width",
            out_path=os.path.join(
                scene_img_dir,
                f"{scene}_cat14_macro_weighted_recall_precision_f1_by_width.png",
            ),
        )


def _per_class_metrics_by_id(category_eval):
    """category_eval: dict with 'per_class' list -> {class_id: {precision, recall, f1, ...}}"""
    if not category_eval or "per_class" not in category_eval:
        return {}
    out = {}
    for item in category_eval["per_class"]:
        cid = item.get("class_id")
        if cid is not None:
            out[int(cid)] = item
    return out


def _class_label(cid):
    name = LANE_CATEGORIES.get(cid, str(cid))
    return f"{cid}:{name}"


def _collect_per_class_series(data, category_key, class_ids, x_key, x_filter):
    """
    x_filter: callable(d) -> bool，例如固定 Width==30 或 IoU==0.5。
    返回: xs, series_r, series_p, series_f1 每个为 {class_id: [y0,y1,...]}（与 xs 对齐）
    """
    xs = []
    by_cid_r = {c: [] for c in class_ids}
    by_cid_p = {c: [] for c in class_ids}
    by_cid_f = {c: [] for c in class_ids}
    for d in data:
        if not x_filter(d):
            continue
        block = d.get(category_key)
        if not block:
            continue
        pm = _per_class_metrics_by_id(block)
        if not pm:
            continue
        xs.append(d[x_key])
        for c in class_ids:
            row = pm.get(c, {})
            by_cid_r[c].append(float(row.get("recall", 0)))
            by_cid_p[c].append(float(row.get("precision", 0)))
            by_cid_f[c].append(float(row.get("f1", 0)))
    return xs, by_cid_r, by_cid_p, by_cid_f


def _plot_per_class_bundle(
    xs, by_r, by_p, by_f, class_ids, xlabel, title_prefix, out_path
):
    if not xs:
        return
    pairs = sorted(zip(xs, range(len(xs))))
    order = [i for _, i in pairs]
    xs_sorted = [xs[i] for i in order]

    def reorder(series_dict):
        return {c: [series_dict[c][i] for i in order] for c in class_ids}

    by_r = reorder(by_r)
    by_p = reorder(by_p)
    by_f = reorder(by_f)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, series, ylab in zip(
        axes, (by_r, by_p, by_f), ("Recall", "Precision", "F1")
    ):
        for c in class_ids:
            ys = series[c]
            if not ys:
                continue
            ax.plot(
                xs_sorted,
                ys,
                label=_class_label(c),
                marker="o",
                markersize=2,
                linewidth=1,
            )
        ax.set_xlabel(xlabel, fontname="AR PL UMing CN")
        ax.set_ylabel(ylab, fontname="AR PL UMing CN")
        ax.set_title(f"{title_prefix} — {ylab}", fontname="AR PL UMing CN")
        ax.grid(True)
        ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def plot_category_curve_dir_per_class14():
    """
    在 pr_curves/image/category/curve/ 下输出整体与各分场景的 per-class（默认 14 类，不含 unknown）
    曲线（Recall / Precision / F1 随 IoU 或 Width 变化）。
    """
    category_root = os.path.join(PR_RESULT_DIR, "image", "category")
    os.makedirs(category_root, exist_ok=True)

    scene_keys = set()
    for d in data:
        for k in d:
            m = re.match(r"^(.+)_Category_Eval$", k)
            if m:
                scene_keys.add(m.group(1))

    def plot_one_scope(category_key, scene_name: str):
        """
        将 per-class 输出改为：image/category/{scene_name}/curve/
        """
        class_ids = _infer_per_class_plot_ids(data, category_key)
        out_dir = os.path.join(category_root, scene_name, "curve")
        os.makedirs(out_dir, exist_ok=True)
        # IoU 扫描，Width=30
        xs, br, bp, bf = _collect_per_class_series(
            data,
            category_key,
            class_ids,
            "IoU_Threshold",
            lambda d: d.get("Width") == 30,
        )
        if xs:
            _plot_per_class_bundle(
                xs,
                br,
                bp,
                bf,
                class_ids,
                "IoU 阈值",
                f"{scene_name}（Width=30）",
                os.path.join(out_dir, "per_class14_recall_precision_f1_by_iou.png"),
            )
        # Width 扫描，IoU=0.5
        xs2, br2, bp2, bf2 = _collect_per_class_series(
            data,
            category_key,
            class_ids,
            "Width",
            lambda d: abs(float(d.get("IoU_Threshold", 0)) - 0.5) < 1e-6,
        )
        if xs2:
            _plot_per_class_bundle(
                xs2,
                br2,
                bp2,
                bf2,
                class_ids,
                "扩展宽度",
                f"{scene_name}（IoU=0.5）",
                os.path.join(out_dir, "per_class14_recall_precision_f1_by_width.png"),
            )

    if any("Category_Eval" in d for d in data):
        plot_one_scope("Category_Eval", "overall")
    for scene in sorted(scene_keys):
        key = f"{scene}_Category_Eval"
        if any(key in d for d in data):
            plot_one_scope(key, scene)


if __name__ == "__main__":
    plot_overall_pr_curve_iou()
    plot_overall_pr_curve_width()
    plot_scene_pr_curve_iou()
    plot_scene_pr_curve_width()
    plot_overall_cat_pr_curves()
    plot_scene_cat_pr_curves()
    plot_scene_category14_cat_pr_curves()
    plot_category_curve_dir_per_class14()
    print(f"所有PR曲线图片已保存到: {IMG_DIR}")
