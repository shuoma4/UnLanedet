import os
import json
import re
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
    "output/llanetv1/openlane1000/category/clrnet_combined_resnet34/pr_curves"
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
        # 创建保存目录
        scene_img_dir = os.path.join(PR_RESULT_DIR, "image", "category", scene)
        os.makedirs(scene_img_dir, exist_ok=True)
        # IoU sweep
        ious, macro, weighted = [], [], []
        for d in data:
            if d["Width"] == 30:
                ious.append(d["IoU_Threshold"])
                macro.append(d.get(f"{scene}_Cat_F1_Macro", 0))
                weighted.append(d.get(f"{scene}_Cat_F1_Weighted", 0))
        plt.figure()
        if has_macro:
            plt.plot(ious, macro, label="mF1 (Cat_F1_Macro)", marker="o")
        if has_weighted:
            plt.plot(ious, weighted, label="wF1 (Cat_F1_Weighted)", marker="s")
        plt.xlabel("IoU 阈值", fontname="AR PL UMing CN")
        plt.ylabel("F1", fontname="AR PL UMing CN")
        plt.title(
            f"{scene} 14类车道线类别PR曲线（随IoU变化, Width=30）",
            fontname="AR PL UMing CN",
        )
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(scene_img_dir, f"{scene}_cat14_pr_curve_by_iou.png"), dpi=600
        )
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
            plt.plot(
                sorted_widths, sorted_macro, label="mF1 (Cat_F1_Macro)", marker="o"
            )
        if has_weighted:
            plt.plot(
                sorted_widths,
                sorted_weighted,
                label="wF1 (Cat_F1_Weighted)",
                marker="s",
            )
        plt.xlabel("扩展宽度", fontname="AR PL UMing CN")
        plt.ylabel("F1", fontname="AR PL UMing CN")
        plt.title(
            f"{scene} 14类车道线类别PR曲线（随Width变化, IoU=0.5）",
            fontname="AR PL UMing CN",
        )
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(scene_img_dir, f"{scene}_cat14_pr_curve_by_width.png"), dpi=600
        )
        plt.close()


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
    curve_root = os.path.join(PR_RESULT_DIR, "image", "category", "curve")
    os.makedirs(curve_root, exist_ok=True)

    scene_keys = set()
    for d in data:
        for k in d:
            m = re.match(r"^(.+)_Category_Eval$", k)
            if m:
                scene_keys.add(m.group(1))

    def plot_one_scope(category_key, subdir_name):
        class_ids = _infer_per_class_plot_ids(data, category_key)
        out_dir = os.path.join(curve_root, subdir_name)
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
                f"{subdir_name}（Width=30）",
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
                f"{subdir_name}（IoU=0.5）",
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
