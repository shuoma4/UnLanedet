import os
import pickle
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

DATA_ROOT = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw"


def generate_lane_mask_binary(image_shape, lane_points_list, line_width=15):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for points in lane_points_list:
        if points is None or len(points) < 2:
            continue
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], False, color=1, thickness=line_width)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def process_sample(args):
    data_info, mask_dir, cut_height, img_shape = args

    img_path = data_info["img_path"]

    # 提取相对路径: segment-.../xxx.jpg
    segment_idx = img_path.find("segment-")
    if segment_idx == -1:
        parts = img_path.split(os.sep)
        rel_path = os.path.join(parts[-2], parts[-1])
    else:
        rel_path = img_path[segment_idx:]

    # 后缀改为 png
    rel_path = rel_path.replace(".jpg", ".png")

    save_path = os.path.join(mask_dir, rel_path)
    if os.path.exists(save_path):
        return True

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    lanes_for_mask = []
    for lane in data_info["lanes"]:
        lane_shifted = lane.copy()
        lane_shifted[:, 1] -= cut_height
        lanes_for_mask.append(lane_shifted)

    mask = generate_lane_mask_binary(img_shape, lanes_for_mask)

    cv2.imwrite(save_path, mask)
    return True


def main():
    pkl_files = [
        "output/.cache/openlane_lane3d_300_train_cuth-270.pkl",
        "output/.cache/openlane_lane3d_300_val_cuth-270.pkl",
        "output/.cache/openlane_lane3d_1000_train_cuth-270.pkl",
        "output/.cache/openlane_lane3d_1000_val_cuth-270.pkl",
    ]

    mask_dir = os.path.join(DATA_ROOT, "mask")
    cut_height = 270
    img_shape = (1280 - cut_height, 1920)

    print("Masks will be saved to:", mask_dir)

    for pkl_file in pkl_files:
        if not os.path.exists(pkl_file):
            continue

        print(f"Loading {pkl_file}...")
        with open(pkl_file, "rb") as f:
            data_infos = pickle.load(f)

        args_list = [(info, mask_dir, cut_height, img_shape) for info in data_infos]

        print(f"Processing {len(args_list)} samples...")
        with Pool(16) as p:
            list(tqdm(p.imap(process_sample, args_list), total=len(args_list)))

    print("All masks generated completed!")


if __name__ == "__main__":
    main()
