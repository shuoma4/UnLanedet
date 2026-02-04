import pickle
import cv2
import os
import numpy as np
import random
from tqdm import tqdm

# OpenLane 类别映射 (用于显示文字)
CATEGORY_MAP = {
    0: "unknown",
    1: "white-dash",
    2: "white-solid",
    3: "double-white-dash",
    4: "double-white-solid",
    5: "white-ldash-rsolid",
    6: "white-lsolid-rdash",
    7: "yellow-dash",
    8: "yellow-solid",
    9: "double-yellow-dash",
    10: "double-yellow-solid",
    11: "yellow-ldash-rsolid",
    12: "yellow-lsolid-rdash",
    13: "left-curbside",
    14: "right-curbside",
}

# 属性映射 (1:left-left, 2:left, 3:right, 4:right-right)
ATTR_MAP = {1: "L-L", 2: "L", 3: "R", 4: "R-R"}


def verify_and_visualize(cache_path, save_dir="vis_debug_detailed", num_samples=20):
    if not os.path.exists(cache_path):
        print(f"错误：找不到缓存文件 {cache_path}")
        return

    os.makedirs(save_dir, exist_ok=True)

    print(f"正在加载缓存: {cache_path} ...")
    with open(cache_path, "rb") as f:
        data_infos = pickle.load(f)

    # --- 筛选逻辑：优先选择“不可见点”比例较高的图片 ---
    print("正在筛选具备代表性（含有遮挡）的样本...")
    info_with_occlusion = []
    for info in data_infos:
        all_vis = (
            np.concatenate(info["lane_vis"])
            if len(info["lane_vis"]) > 0
            else np.array([1])
        )
        # 如果存在 visibility < 0.5 的点，标记为代表性样本
        if np.any(all_vis < 0.5):
            info_with_occlusion.append(info)

    print(f"发现包含遮挡点的样本数: {len(info_with_occlusion)}")

    # 如果遮挡样本多，从遮挡样本选；否则从全集选
    if len(info_with_occlusion) >= num_samples:
        samples = random.sample(info_with_occlusion, num_samples)
    else:
        samples = random.sample(data_infos, min(num_samples, len(data_infos)))

    for i, sample in enumerate(samples):
        img_path = sample["img_path"]
        img = cv2.imread(img_path)
        if img is None:
            continue

        lanes = sample["lanes"]
        lane_vis = sample["lane_vis"]
        lane_cats = sample.get("lane_categories", [])
        lane_attrs = sample.get("lane_attributes", [])

        for idx, (lane, vis) in enumerate(zip(lanes, lane_vis)):
            # 获取类别和属性字符串
            cat_id = lane_cats[idx] if idx < len(lane_cats) else 0
            attr_id = lane_attrs[idx] if idx < len(lane_attrs) else 0
            label_text = f"ID:{idx} {CATEGORY_MAP.get(cat_id, 'unk')}"
            if attr_id in ATTR_MAP:
                label_text += f" [{ATTR_MAP[attr_id]}]"

            # 绘制车道线上的点
            for j in range(len(lane)):
                u, v = lane[j]
                v_status = vis[j]
                pt = (int(round(u)), int(round(v)))

                # 颜色逻辑：绿色(可见)，红色(不可见/被遮挡)
                if v_status > 0.5:
                    cv2.circle(img, pt, 2, (0, 255, 0), -1)  # 可见点：绿色小圆
                else:
                    cv2.circle(img, pt, 5, (0, 0, 255), -1)  # 遮挡点：红色大圆（显眼）

                # 在每条线的中间位置打上标签文字
                if j == len(lane) // 2:
                    cv2.putText(
                        img,
                        label_text,
                        (pt[0] + 5, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

                # 连线
                if j > 0:
                    prev_pt = (int(round(lane[j - 1][0])), int(round(lane[j - 1][1])))
                    # 线段颜色：如果当前段有遮挡点，用红色细线
                    line_color = (0, 255, 0) if v_status > 0.5 else (0, 0, 255)
                    cv2.line(img, prev_pt, pt, line_color, 1)

        # 标注图片路径信息
        save_name = f"sample_{i}_{os.path.basename(img_path)}"
        cv2.putText(
            img,
            f"File: {os.path.basename(img_path)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
        cv2.putText(
            img,
            "RED BIG DOT: Occluded Lane Point (visibility=0)",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        cv2.imwrite(os.path.join(save_dir, save_name), img)

    print(f"\n可视化完成！已保存至: {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    # 填入你的 .pkl 路径
    CACHE_PATH = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/openlane_lane3d_1000_train_cuth-270_800x320_cache_v1.pkl"
    SAVE_DIR = "./vis/vis_visibility_points/"
    verify_and_visualize(CACHE_PATH, save_dir=SAVE_DIR, num_samples=30)
