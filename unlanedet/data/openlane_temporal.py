import os
import json
import logging
import cv2
import numpy as np
import torch
from tqdm import tqdm
from .base_dataset import BaseDataset
import glob
import pickle
import gc
import functools

def load_segment_pkl(cache_dir, seg):
    with open(os.path.join(cache_dir, f"{seg}.pkl"), "rb") as f:
        return pickle.load(f)

class OpenLaneTemporal(BaseDataset):
    def __init__(self, data_root, split, cut_height, processes=None, cfg=None, seq_len=3):
        self.seq_len = seq_len
        self.split = split
        self.max_lanes = getattr(cfg, 'max_lanes', 24)
        self.cache_dir = os.path.join(data_root, f"openlane_temporal_cache_dir_{split}_{self.seq_len}_cuth-{cut_height}")
        
        # Instance-local cache to avoid threading deadlocks in DataLoader forks
        self._segment_cache = {}
        self._cache_queue = []
        self._max_cache_size = 8
        
        super().__init__(data_root, split, cut_height, processes=processes, cfg=cfg)
        self.load_annotations()

    def get_dataset_info(self, data_root, split):
        index_file = os.path.join(self.cache_dir, "index.pkl")
        
        if os.path.exists(index_file):
            self.logger.info(f"Loading cached temporal dataset index from {index_file}")
            with open(index_file, "rb") as f:
                index_data = pickle.load(f)
            return index_data['clips']

        os.makedirs(self.cache_dir, exist_ok=True)
        sub_dir = "training" if split == "train" else "validation"
        anno_dir = os.path.join(data_root, sub_dir)
        json_files = glob.glob(os.path.join(anno_dir, "**/*.json"), recursive=True)
        
        segment_groups = {}
        # Pre-parse all JSONs
        for json_path in tqdm(json_files, desc="Parsing all JSONs once"):
            rel_path = os.path.relpath(json_path, data_root)
            img_rel_path = rel_path.replace('.json', '.jpg')
            
            with open(json_path, 'r') as f:
                anno = json.load(f)
                
            from .openlane import VALID_LANE_CATEGORIES
            
            lanes = []
            lane_categories = []
            lane_attributes = []
            lane_track_ids = []
            lane_vis = []
            xyz = []

            for lane_id, ln in enumerate(anno.get('lane_lines', [])):
                uv = ln.get('uv', [])
                if len(uv) == 2 and len(uv[0]) > 1:
                    u = uv[0]
                    v = uv[1]
                    min_len = min(len(u), len(v))
                    
                    z_pts = ln.get('xyz', [[0]*min_len, [0]*min_len, [0]*min_len])
                    z_valid = len(z_pts) == 3 and len(z_pts[0]) >= min_len
                    vis_arr = ln.get('visibility', [1.0] * min_len)

                    if min_len >= 2:
                        points = []
                        vis_pts = []
                        xyz_pts = []
                        for i in range(min_len):
                            if v[i] >= self.cut_height:
                                points.append((u[i], v[i]))
                                vis_pts.append(vis_arr[i] if i < len(vis_arr) else 1.0)
                                if z_valid:
                                    xyz_pts.append((z_pts[0][i], z_pts[1][i], z_pts[2][i]))
                                else:
                                    xyz_pts.append((0.0, 0.0, 0.0))

                        if len(points) >= 2:
                            # sort by v descending (bottom to top)
                            combined = list(zip(points, vis_pts, xyz_pts))
                            combined.sort(key=lambda x: x[0][1], reverse=True)
                            
                            lanes.append([x[0] for x in combined])
                            lane_vis.append([x[1] for x in combined])
                            xyz.append([x[2] for x in combined])
                            
                            lane_categories.append(VALID_LANE_CATEGORIES.get(ln.get("category", 0), 0))
                            lane_attributes.append(ln.get("attribute", 0))
                            lane_track_ids.append(ln.get("track_id", -1))
            
            abs_img_path = os.path.join(os.path.dirname(data_root), img_rel_path)
            
            frame_info = {
                'img_path': abs_img_path,
                'img_name': img_rel_path,
                'img_rel_path': abs_img_path,
                'lanes': lanes,
                'lane_categories': lane_categories,
                'lane_attributes': lane_attributes,
                'lane_track_ids': lane_track_ids,
                'visibility': lane_vis,
                'xyz': xyz,
                'cut_height': self.cut_height,
                'extrinsic': np.array(anno.get('extrinsic', [])), 
                'intrinsic': np.array(anno.get('intrinsic', []))
            }
            
            parts = img_rel_path.split('/')
            if len(parts) >= 2:
                segment_name = parts[-2]
                if segment_name not in segment_groups:
                    segment_groups[segment_name] = []
                segment_groups[segment_name].append(frame_info)
            
        seq_infos = []
        segments = list(segment_groups.keys())
        for seg in tqdm(segments, desc="Saving segmented caches"):
            paths = segment_groups[seg]
            paths.sort(key=lambda x: x['img_name'])
            
            with open(os.path.join(self.cache_dir, f"{seg}.pkl"), "wb") as f:
                pickle.dump(paths, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            for i in range(max(1, len(paths) - self.seq_len + 1)):
                if len(paths) >= self.seq_len:
                    clip_indices = list(range(i, i + self.seq_len))
                else: 
                    clip_indices = [0] * (self.seq_len - len(paths)) + list(range(len(paths)))
                
                seq_infos.append((seg, clip_indices))
                
        with open(index_file, "wb") as f:
            pickle.dump({'clips': seq_infos, 'segments': segments}, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        return seq_infos

    def load_annotations(self):
        self.logger.info(f"Loading OpenLane Temporal Dataset (T={self.seq_len})...")
        self.clip_cache = self.get_dataset_info(self.data_root, self.split)
        
        self.data_infos = []
        if self.split != "train":
            seg_frames_needed = {}
            for idx, (seg, indices) in enumerate(self.clip_cache):
                if seg not in seg_frames_needed:
                    seg_frames_needed[seg] = {}
                seg_frames_needed[seg][idx] = indices[-1]

            self.data_infos = [None] * len(self.clip_cache)
            for seg, idx_map in seg_frames_needed.items():
                seg_data = load_segment_pkl(self.cache_dir, seg)
                for global_idx, frame_idx in idx_map.items():
                    self.data_infos[global_idx] = seg_data[frame_idx]

    def __len__(self):
        return len(self.clip_cache)

    def _process_single_frame(self, frame_info):
        frame_c = frame_info.copy()
        use_offline = getattr(self.cfg, "use_offline_resized", False)
        img_path = frame_c["img_path"]
        
        segment_idx = img_path.find("segment-")
        if segment_idx == -1:
            parts = img_path.split(os.sep)
            rel_path = os.path.join(parts[-2], parts[-1])
        else:
            rel_path = img_path[segment_idx:]

        if use_offline:
            base_dir = os.path.dirname(self.data_root)
            if self.cut_height == 600:
                train_dir = "training_cut_600_resized_800_320"
                val_dir = "validation_cut_600_resized_800_320"
                mask_dir = "mask_cut_600_resized_800_320"
            else:
                train_dir = "training_resized_800_320"
                val_dir = "validation_resized_800_320"
                mask_dir = "mask_resized_800_320"

            if "training" in img_path:
                resized_img_path = os.path.join(base_dir, train_dir, rel_path)
            else:
                resized_img_path = os.path.join(base_dir, val_dir, rel_path)
                
            img = cv2.imread(resized_img_path)
            
            mask_path = os.path.join(base_dir, mask_dir, rel_path).replace(".jpg", ".png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = None

            if mask is None:
                from .openlane import generate_lane_mask_binary
                lanes_for_mask = []
                for lane in frame_c["lanes"]:
                    lane_shifted = np.array(lane, dtype=np.float32)
                    lane_shifted[:, 1] -= self.cut_height
                    lanes_for_mask.append(lane_shifted)
                dummy_img = np.zeros((1280, 1920, 3), dtype=np.uint8)
                mask_org = generate_lane_mask_binary(dummy_img, lanes_for_mask)
                mask = cv2.resize(mask_org, (800, 320), interpolation=cv2.INTER_NEAREST)

            lanes = []
            for lane in frame_c["lanes"]:
                lane_scaled = np.array(lane, dtype=np.float32)
                lane_scaled[:, 0] *= (800.0 / 1920.0)
                lane_scaled[:, 1] = (lane_scaled[:, 1] - self.cut_height) * (320.0 / (1280.0 - self.cut_height))
                lanes.append(lane_scaled)
                
            frame_c["img"] = img
            frame_c["lanes"] = lanes
            frame_c["mask"] = mask
            frame_c["cut_height"] = 0
        else:
            img = cv2.imread(img_path)
            img = img[self.cut_height :, :, :]
            
            mask_dir_name = "mask_cut_600" if self.cut_height == 600 else "mask"
            mask_path = os.path.join(os.path.dirname(self.data_root), mask_dir_name, rel_path).replace(".jpg", ".png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = None
            
            if mask is None:
                from .openlane import generate_lane_mask_binary
                lanes_for_mask = []
                for lane in frame_c["lanes"]:
                    lane_shifted = np.array(lane, dtype=np.float32)
                    lane_shifted[:, 1] -= self.cut_height
                    lanes_for_mask.append(lane_shifted)
                mask = generate_lane_mask_binary(img, lanes_for_mask)
            
            frame_c["img"] = img
            frame_c["mask"] = mask
            
        return frame_c

    def _get_segment_data(self, seg):
        if seg in self._segment_cache:
            return self._segment_cache[seg]
            
        seg_data = load_segment_pkl(self.cache_dir, seg)
        self._segment_cache[seg] = seg_data
        self._cache_queue.append(seg)
        
        # Evict oldest segment to save memory
        if len(self._cache_queue) > self._max_cache_size:
            oldest_seg = self._cache_queue.pop(0)
            if oldest_seg in self._segment_cache:
                del self._segment_cache[oldest_seg]
                
        return seg_data

    def __getitem__(self, idx):
        seg, indices = self.clip_cache[idx]
        seg_data = self._get_segment_data(seg)
        
        processed_clip = []
        for i in indices:
            processed_clip.append(self._process_single_frame(seg_data[i]))
            
        return self.processes(processed_clip)
