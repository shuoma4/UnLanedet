import argparse
import csv
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
from tqdm import tqdm


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from unlanedet.data.openlane_temporal import load_segment_pkl  # noqa: E402


def _find_cache_dir(data_root, split, seq_len, cut_height):
    exact = os.path.join(
        data_root, f"openlane_temporal_cache_dir_{split}_{seq_len}_cuth-{cut_height}"
    )
    if os.path.isdir(exact):
        return exact

    prefix = f"openlane_temporal_cache_dir_{split}_{seq_len}"
    candidates = sorted(
        [
            os.path.join(data_root, x)
            for x in os.listdir(data_root)
            if x.startswith(prefix) and os.path.isdir(os.path.join(data_root, x))
        ]
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Cannot find temporal cache dir in {data_root} with prefix {prefix}"
    )


def _safe_array(x):
    if x is None:
        return None
    arr = np.asarray(x)
    return arr if arr.size > 0 else None


def _build_track_map(track_ids):
    out = {}
    for idx, tid in enumerate(track_ids):
        try:
            tid_int = int(tid)
        except Exception:
            continue
        if tid_int < 0:
            continue
        if tid_int not in out:
            out[tid_int] = idx
    return out


def _load_pose_from_json(data_root, frame):
    img_name = frame.get("img_name", "")
    img_path = frame.get("img_path", "")

    candidates = []
    if img_name:
        if img_name.endswith(".jpg"):
            candidates.append(os.path.join(data_root, img_name.replace(".jpg", ".json")))
    if img_path and img_path.endswith(".jpg"):
        candidates.append(img_path.replace(".jpg", ".json"))

    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    j = json.load(f)
                pose = j.get("pose", None)
                if pose is not None:
                    arr = np.asarray(pose, dtype=np.float64)
                    if arr.shape == (4, 4):
                        return arr
            except Exception:
                continue
    return None


def _interp_u_on_y(lane_uv, y_samples):
    if lane_uv is None or len(lane_uv) < 2:
        return None
    lane = np.asarray(lane_uv, dtype=np.float64)
    if lane.ndim != 2 or lane.shape[1] != 2:
        return None
    u = lane[:, 0]
    y = lane[:, 1]

    order = np.argsort(y)
    y = y[order]
    u = u[order]

    uniq_y, uniq_idx = np.unique(y, return_index=True)
    if uniq_y.shape[0] < 2:
        return None
    uniq_u = u[uniq_idx]

    low, high = uniq_y[0], uniq_y[-1]
    mask = (y_samples >= low) & (y_samples <= high)
    if not np.any(mask):
        return None

    y_valid = y_samples[mask]
    u_valid = np.interp(y_valid, uniq_y, uniq_u)
    return y_valid, u_valid


def _project_prev_xyz_to_curr_uv(prev_xyz, T_rel, K):
    pts = np.asarray(prev_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        return None

    pts_h = np.concatenate([pts.T, np.ones((1, pts.shape[0]), dtype=np.float64)], axis=0)
    pts_t = T_rel @ pts_h
    X = pts_t[0, :]
    Y = pts_t[1, :]
    Z = pts_t[2, :]

    valid = X > 0.1
    if not np.any(valid):
        return None
    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = (-Y / X) * fx + cx
    v = (-Z / X) * fy + cy
    return v, u


def main():
    parser = argparse.ArgumentParser(description="Temporal reprojection stats for OpenLane")
    parser.add_argument(
        "--data-root",
        default="/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/lane3d_1000",
    )
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--cut-height", type=int, default=600)
    parser.add_argument("--y-step", type=float, default=10.0)
    parser.add_argument(
        "--trusted-min-samples",
        type=int,
        default=10,
        help="可信统计最小重叠采样点数阈值",
    )
    parser.add_argument(
        "--trusted-max-error",
        type=float,
        default=300.0,
        help="可信统计中 lane max_abs_error_px 的上限阈值（像素，原图尺度）",
    )
    parser.add_argument(
        "--output-root",
        default="/data1/lxy_log/workspace/ms/UnLanedet/output/analysis/time_series",
    )
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, f"reproj_stats_{args.split}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    cache_dir = _find_cache_dir(args.data_root, args.split, args.seq_len, args.cut_height)
    index_file = os.path.join(cache_dir, "index.pkl")
    with open(index_file, "rb") as f:
        index_data = pickle.load(f)
    segments = index_data["segments"]

    scanline_rows = []
    lane_rows = []
    change_rows = []

    total_pairs = 0
    total_common_tracks = 0
    total_valid_lanes = 0

    y_grid = np.arange(float(args.cut_height), 1280.0, float(args.y_step), dtype=np.float64)

    for seg in tqdm(segments, desc="Segments"):
        seg_data = load_segment_pkl(cache_dir, seg)
        if len(seg_data) < 2:
            continue

        for i in range(1, len(seg_data)):
            prev_f = seg_data[i - 1]
            curr_f = seg_data[i]
            total_pairs += 1

            prev_name = prev_f.get("img_name", os.path.basename(prev_f.get("img_path", "")))
            curr_name = curr_f.get("img_name", os.path.basename(curr_f.get("img_path", "")))

            prev_map = _build_track_map(prev_f.get("lane_track_ids", []))
            curr_map = _build_track_map(curr_f.get("lane_track_ids", []))
            prev_tracks = set(prev_map.keys())
            curr_tracks = set(curr_map.keys())
            common = sorted(prev_tracks & curr_tracks)
            total_common_tracks += len(common)

            appeared = sorted(curr_tracks - prev_tracks)
            disappeared = sorted(prev_tracks - curr_tracks)
            for tid in appeared:
                change_rows.append(
                    {
                        "segment_id": seg,
                        "frame_idx_curr": i,
                        "prev_image": prev_name,
                        "curr_image": curr_name,
                        "event_type": "appeared",
                        "lane_track_id": int(tid),
                    }
                )
            for tid in disappeared:
                change_rows.append(
                    {
                        "segment_id": seg,
                        "frame_idx_curr": i,
                        "prev_image": prev_name,
                        "curr_image": curr_name,
                        "event_type": "disappeared",
                        "lane_track_id": int(tid),
                    }
                )

            E_t1 = _safe_array(prev_f.get("extrinsic"))
            E_t = _safe_array(curr_f.get("extrinsic"))
            P_t1 = _safe_array(prev_f.get("pose"))
            P_t = _safe_array(curr_f.get("pose"))
            if P_t1 is None:
                P_t1 = _load_pose_from_json(args.data_root, prev_f)
            if P_t is None:
                P_t = _load_pose_from_json(args.data_root, curr_f)
            K_t = _safe_array(curr_f.get("intrinsic"))
            if (
                E_t1 is None
                or E_t is None
                or P_t1 is None
                or P_t is None
                or K_t is None
                or E_t1.shape != (4, 4)
                or E_t.shape != (4, 4)
                or P_t1.shape != (4, 4)
                or P_t.shape != (4, 4)
                or K_t.shape != (3, 3)
            ):
                continue

            try:
                T_rel = np.linalg.inv(E_t) @ np.linalg.inv(P_t) @ P_t1 @ E_t1
            except np.linalg.LinAlgError:
                continue

            for tid in common:
                prev_idx = prev_map[tid]
                curr_idx = curr_map[tid]

                prev_xyz_all = prev_f.get("xyz", [])
                curr_lanes_all = curr_f.get("lanes", [])
                if prev_idx >= len(prev_xyz_all) or curr_idx >= len(curr_lanes_all):
                    continue

                proj = _project_prev_xyz_to_curr_uv(prev_xyz_all[prev_idx], T_rel, K_t)
                if proj is None:
                    continue
                v_proj, u_proj = proj

                v_order = np.argsort(v_proj)
                v_proj = v_proj[v_order]
                u_proj = u_proj[v_order]
                uv_uniq, uniq_idx = np.unique(v_proj, return_index=True)
                if uv_uniq.shape[0] < 2:
                    continue
                uu_uniq = u_proj[uniq_idx]

                curr_interp = _interp_u_on_y(curr_lanes_all[curr_idx], y_grid)
                if curr_interp is None:
                    continue
                y_curr, u_curr = curr_interp

                low = max(float(y_curr.min()), float(uv_uniq.min()))
                high = min(float(y_curr.max()), float(uv_uniq.max()))
                overlap = (y_curr >= low) & (y_curr <= high)
                if not np.any(overlap):
                    continue

                y_eval = y_curr[overlap]
                u_gt = u_curr[overlap]
                u_reproj = np.interp(y_eval, uv_uniq, uu_uniq)
                err = np.abs(u_reproj - u_gt)
                if err.size == 0:
                    continue

                total_valid_lanes += 1
                lane_rows.append(
                    {
                        "segment_id": seg,
                        "frame_idx_curr": i,
                        "prev_image": prev_name,
                        "curr_image": curr_name,
                        "lane_track_id": int(tid),
                        "num_samples": int(err.size),
                        "mean_abs_error_px": float(np.mean(err)),
                        "median_abs_error_px": float(np.median(err)),
                        "p90_abs_error_px": float(np.percentile(err, 90)),
                        "max_abs_error_px": float(np.max(err)),
                    }
                )

                for yi, ui_r, ui_g, ei in zip(y_eval, u_reproj, u_gt, err):
                    scanline_rows.append(
                        {
                            "segment_id": seg,
                            "frame_idx_curr": i,
                            "prev_image": prev_name,
                            "curr_image": curr_name,
                            "lane_track_id": int(tid),
                            "sample_y": float(yi),
                            "u_reprojected": float(ui_r),
                            "u_current": float(ui_g),
                            "abs_error_px": float(ei),
                        }
                    )

    lane_mean = [x["mean_abs_error_px"] for x in lane_rows]
    scanline_err = [x["abs_error_px"] for x in scanline_rows]
    appeared_cnt = sum(1 for x in change_rows if x["event_type"] == "appeared")
    disappeared_cnt = sum(1 for x in change_rows if x["event_type"] == "disappeared")

    trusted_lane_rows = [
        x
        for x in lane_rows
        if int(x.get("num_samples", 0)) >= int(args.trusted_min_samples)
        and float(x.get("max_abs_error_px", 1e9)) <= float(args.trusted_max_error)
    ]

    def _lane_stats(rows):
        if not rows:
            return {
                "count": 0,
                "mean_abs_error_px_mean": None,
                "mean_abs_error_px_median": None,
                "p90_abs_error_px_median": None,
                "max_abs_error_px_median": None,
            }
        mean_list = [float(r["mean_abs_error_px"]) for r in rows]
        p90_list = [float(r["p90_abs_error_px"]) for r in rows]
        max_list = [float(r["max_abs_error_px"]) for r in rows]
        return {
            "count": int(len(rows)),
            "mean_abs_error_px_mean": float(np.mean(mean_list)),
            "mean_abs_error_px_median": float(np.median(mean_list)),
            "p90_abs_error_px_median": float(np.median(p90_list)),
            "max_abs_error_px_median": float(np.median(max_list)),
        }

    lane_stats_all = _lane_stats(lane_rows)
    lane_stats_trusted = _lane_stats(trusted_lane_rows)

    summary = {
        "split": args.split,
        "cache_dir": cache_dir,
        "output_dir": out_dir,
        "num_segments": int(len(segments)),
        "num_frame_pairs": int(total_pairs),
        "num_common_tracks": int(total_common_tracks),
        "num_valid_lanes_with_error": int(total_valid_lanes),
        "num_scanline_samples": int(len(scanline_rows)),
        "scanline_mean_abs_offset_px": float(np.mean(scanline_err)) if scanline_err else None,
        "scanline_median_abs_offset_px": float(np.median(scanline_err)) if scanline_err else None,
        "lane_mean_abs_error_px_mean": lane_stats_all["mean_abs_error_px_mean"],
        "lane_mean_abs_error_px_median": lane_stats_all["mean_abs_error_px_median"],
        "trusted_filter": {
            "min_samples": int(args.trusted_min_samples),
            "max_lane_error_px": float(args.trusted_max_error),
        },
        "trusted_lane_count": lane_stats_trusted["count"],
        "trusted_lane_ratio": float(lane_stats_trusted["count"] / len(lane_rows)) if lane_rows else 0.0,
        "trusted_lane_mean_abs_error_px_mean": lane_stats_trusted["mean_abs_error_px_mean"],
        "trusted_lane_mean_abs_error_px_median": lane_stats_trusted["mean_abs_error_px_median"],
        "trusted_lane_p90_abs_error_px_median": lane_stats_trusted["p90_abs_error_px_median"],
        "trusted_lane_max_abs_error_px_median": lane_stats_trusted["max_abs_error_px_median"],
        "appeared_count": int(appeared_cnt),
        "disappeared_count": int(disappeared_cnt),
        "total_change_events": int(len(change_rows)),
    }

    def _write_csv(path, rows):
        if not rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write("")
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    lane_csv = os.path.join(out_dir, "lane_reprojection_error.csv")
    trusted_lane_csv = os.path.join(out_dir, "lane_reprojection_error_trusted.csv")
    scanline_csv = os.path.join(out_dir, "scanline_reprojection_offset.csv")
    change_csv = os.path.join(out_dir, "lane_change_events.csv")
    summary_json = os.path.join(out_dir, "summary.json")
    report_md = os.path.join(out_dir, "report.md")

    _write_csv(lane_csv, lane_rows)
    _write_csv(trusted_lane_csv, trusted_lane_rows)
    _write_csv(scanline_csv, scanline_rows)
    _write_csv(change_csv, change_rows)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# OpenLane lane3d_1000 连续帧重投影统计报告",
        "",
        f"- split: `{args.split}`",
        f"- cache_dir: `{cache_dir}`",
        f"- frame pairs: `{summary['num_frame_pairs']}`",
        f"- common tracks: `{summary['num_common_tracks']}`",
        f"- valid lanes: `{summary['num_valid_lanes_with_error']}`",
        f"- scanline samples: `{summary['num_scanline_samples']}`",
        f"- scanline mean abs offset(px): `{summary['scanline_mean_abs_offset_px']}`",
        f"- scanline median abs offset(px): `{summary['scanline_median_abs_offset_px']}`",
        f"- lane mean(abs error) mean(px): `{summary['lane_mean_abs_error_px_mean']}`",
        f"- lane mean(abs error) median(px): `{summary['lane_mean_abs_error_px_median']}`",
        "",
        "## 可信子集统计（建议用于趋势分析与损失门控）",
        f"- trusted filter: `num_samples >= {int(args.trusted_min_samples)}` and "
        f"`lane max_abs_error_px <= {float(args.trusted_max_error)}`",
        f"- trusted lanes: `{summary['trusted_lane_count']}`",
        f"- trusted ratio: `{summary['trusted_lane_ratio']}`",
        f"- trusted lane mean(abs error) mean(px): `{summary['trusted_lane_mean_abs_error_px_mean']}`",
        f"- trusted lane mean(abs error) median(px): `{summary['trusted_lane_mean_abs_error_px_median']}`",
        f"- trusted lane p90(abs error) median(px): `{summary['trusted_lane_p90_abs_error_px_median']}`",
        f"- trusted lane max(abs error) median(px): `{summary['trusted_lane_max_abs_error_px_median']}`",
        f"- appeared: `{summary['appeared_count']}`",
        f"- disappeared: `{summary['disappeared_count']}`",
        "",
        "## 输出文件",
        f"- `{lane_csv}`",
        f"- `{trusted_lane_csv}`",
        f"- `{scanline_csv}`",
        f"- `{change_csv}`",
        f"- `{summary_json}`",
    ]
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[DONE] output dir: {out_dir}")
    print(f"[DONE] lane csv: {lane_csv}")
    print(f"[DONE] scanline csv: {scanline_csv}")
    print(f"[DONE] change csv: {change_csv}")
    print(f"[DONE] summary: {summary_json}")
    print(f"[DONE] report: {report_md}")


if __name__ == "__main__":
    main()
