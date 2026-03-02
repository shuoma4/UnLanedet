import argparse
import json
import os
import random

import cv2
import numpy as np


def load_lanes(json_path):
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        data = json.load(f)
    lanes = []
    for lane in data.get("lane_lines", []):
        uv = lane.get("uv")
        if not uv or len(uv) != 2:
            continue
        xs, ys = uv
        points = []
        for x, y in zip(xs, ys):
            if x is None or y is None:
                continue
            points.append((float(x), float(y)))
        if len(points) >= 2:
            lanes.append(points)
    return lanes


def draw_lanes(img, lanes, color, thickness):
    for pts in lanes:
        if len(pts) < 2:
            continue
        pts_np = np.array(pts, dtype=np.int32)
        for i in range(len(pts_np) - 1):
            p1 = tuple(pts_np[i])
            p2 = tuple(pts_np[i + 1])
            cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--lane-anno-dir", default="lane3d_1000")
    parser.add_argument("--list-file", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--line-width", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    list_file = args.list_file or os.path.join(args.pred_root, "test_list.txt")
    with open(list_file, "r") as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    random.seed(args.seed)
    if args.num_samples > 0:
        rel_paths = random.sample(rel_paths, min(args.num_samples, len(rel_paths)))

    for rel_path in rel_paths:
        pred_json = os.path.join(args.pred_root, rel_path.replace(".jpg", ".json"))
        gt_json = os.path.join(
            args.data_root, args.lane_anno_dir, rel_path.replace(".jpg", ".json")
        )
        img_path = os.path.join(args.data_root, rel_path)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        pred_lanes = load_lanes(pred_json)
        gt_lanes = load_lanes(gt_json)
        draw_lanes(img, gt_lanes, (0, 255, 0), args.line_width)
        draw_lanes(img, pred_lanes, (0, 0, 255), args.line_width)
        out_name = rel_path.replace("/", "_").replace(".jpg", ".png")
        out_path = os.path.join(args.out_dir, out_name)
        cv2.imwrite(out_path, img)


if __name__ == "__main__":
    main()
