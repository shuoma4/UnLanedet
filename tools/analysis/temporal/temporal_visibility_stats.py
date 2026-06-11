import argparse
import csv
import json
import os
import pickle
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

from unlanedet.data.openlane_temporal import load_segment_pkl


def find_cache_dir(data_root, split, seq_len, cut_height):
    exact = os.path.join(
        data_root, f"openlane_temporal_cache_dir_{split}_{seq_len}_cuth-{cut_height}"
    )
    if os.path.isdir(exact):
        return exact
    raise FileNotFoundError(f"Cache dir not found: {exact}")


def draw_visibility_overlay(frame, out_path):
    img_path = frame.get("img_path", "")
    img = cv2.imread(img_path)
    if img is None:
        return False

    lanes = frame.get("lanes", [])
    vis_all = frame.get("visibility", [])

    for lane_idx, lane in enumerate(lanes):
        if lane_idx >= len(vis_all):
            continue
        vis = vis_all[lane_idx]
        n = min(len(lane), len(vis))
        if n <= 0:
            continue
        for i in range(n):
            u, v = lane[i]
            x = int(round(float(u)))
            y = int(round(float(v)))
            color = (0, 255, 0) if float(vis[i]) > 0.5 else (0, 0, 255)
            cv2.circle(img, (x, y), 3, color, -1, lineType=cv2.LINE_AA)

    cv2.imwrite(out_path, img)
    return True


def main():
    parser = argparse.ArgumentParser(description="Temporal visibility/invisible-point analysis")
    parser.add_argument(
        "--data-root",
        default="/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/lane3d_1000",
    )
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--cut-height", type=int, default=600)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--output-root",
        default="/data1/lxy_log/workspace/ms/UnLanedet/output/analysis/time_series",
    )
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"visibility_stats_{args.split}_{run_id}")
    vis_dir = os.path.join(out_dir, "topk_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    cache_dir = find_cache_dir(args.data_root, args.split, args.seq_len, args.cut_height)
    with open(os.path.join(cache_dir, "index.pkl"), "rb") as f:
        index_data = pickle.load(f)
    segments = index_data["segments"]

    frame_rows = []
    lane_rows = []
    total_frames = 0
    frames_with_invisible = 0

    for seg in tqdm(segments, desc="Segments"):
        seg_data = load_segment_pkl(cache_dir, seg)
        for frame_idx, frame in enumerate(seg_data):
            total_frames += 1
            lanes = frame.get("lanes", [])
            vis_all = frame.get("visibility", [])

            frame_inv_points = 0
            frame_total_points = 0
            frame_inv_lanes = 0

            for lane_idx, lane in enumerate(lanes):
                if lane_idx >= len(vis_all):
                    continue
                vis = vis_all[lane_idx]
                n = min(len(lane), len(vis))
                if n <= 0:
                    continue
                inv_count = int(np.sum(np.asarray(vis[:n], dtype=np.float32) <= 0.5))
                total_count = int(n)
                frame_inv_points += inv_count
                frame_total_points += total_count
                if inv_count > 0:
                    frame_inv_lanes += 1

                lane_rows.append(
                    {
                        "segment_id": seg,
                        "frame_idx": int(frame_idx),
                        "image_id": frame.get("img_name", os.path.basename(frame.get("img_path", ""))),
                        "lane_idx": int(lane_idx),
                        "lane_track_id": int(frame.get("lane_track_ids", [-1] * len(lanes))[lane_idx])
                        if lane_idx < len(frame.get("lane_track_ids", []))
                        else -1,
                        "invisible_points": inv_count,
                        "total_points": total_count,
                        "invisible_ratio": float(inv_count / max(total_count, 1)),
                    }
                )

            if frame_inv_points > 0:
                frames_with_invisible += 1
            frame_rows.append(
                {
                    "segment_id": seg,
                    "frame_idx": int(frame_idx),
                    "image_id": frame.get("img_name", os.path.basename(frame.get("img_path", ""))),
                    "invisible_points": int(frame_inv_points),
                    "total_points": int(frame_total_points),
                    "invisible_ratio": float(frame_inv_points / max(frame_total_points, 1)),
                    "lanes_with_invisible": int(frame_inv_lanes),
                    "num_lanes": int(len(lanes)),
                }
            )

    # Top-K severe occlusion frames by invisible points, then ratio
    frame_rows_sorted = sorted(
        frame_rows,
        key=lambda x: (x["invisible_points"], x["invisible_ratio"], x["lanes_with_invisible"]),
        reverse=True,
    )
    topk = frame_rows_sorted[: args.topk]

    topk_meta = []
    seg_map = {}
    for seg in segments:
        seg_map[seg] = load_segment_pkl(cache_dir, seg)

    for rank, row in enumerate(topk, start=1):
        seg = row["segment_id"]
        idx = row["frame_idx"]
        frame = seg_map[seg][idx]
        out_name = f"rank_{rank:03d}__{seg}__f{idx:04d}.jpg".replace("/", "_")
        out_path = os.path.join(vis_dir, out_name)
        ok = draw_visibility_overlay(frame, out_path)
        if ok:
            item = dict(row)
            item["rank"] = rank
            item["visualization_path"] = out_path
            topk_meta.append(item)

    def write_csv(path, rows):
        if not rows:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    os.makedirs(out_dir, exist_ok=True)
    frame_csv = os.path.join(out_dir, "frame_visibility_stats.csv")
    lane_csv = os.path.join(out_dir, "lane_visibility_stats.csv")
    topk_csv = os.path.join(out_dir, "topk_invisible_frames.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    report_md = os.path.join(out_dir, "report.md")

    write_csv(frame_csv, frame_rows)
    write_csv(lane_csv, lane_rows)
    write_csv(topk_csv, topk_meta)

    summary = {
        "split": args.split,
        "cache_dir": cache_dir,
        "output_dir": out_dir,
        "total_frames": int(total_frames),
        "frames_with_invisible": int(frames_with_invisible),
        "frames_with_invisible_ratio": float(frames_with_invisible / max(total_frames, 1)),
        "topk": int(args.topk),
        "topk_visualizations_dir": vis_dir,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(
            "\n".join(
                [
                    "# 时序不可见性标记分析报告",
                    "",
                    f"- total_frames: `{summary['total_frames']}`",
                    f"- frames_with_invisible: `{summary['frames_with_invisible']}`",
                    f"- frames_with_invisible_ratio: `{summary['frames_with_invisible_ratio']}`",
                    f"- topk_visualizations_dir: `{vis_dir}`",
                    "",
                    "## 输出文件",
                    f"- `{frame_csv}`",
                    f"- `{lane_csv}`",
                    f"- `{topk_csv}`",
                    f"- `{summary_json}`",
                ]
            )
        )

    print(f"[DONE] output dir: {out_dir}")
    print(f"[DONE] topk vis dir: {vis_dir}")
    print(f"[DONE] summary: {summary_json}")


if __name__ == "__main__":
    main()
