import os
import json
import logging
import cv2
import numpy as np
import torch
from tqdm import tqdm
from .base_dataset import BaseDataset

class OpenLaneTemporal(BaseDataset):
    def __init__(self, data_root, split, cut_height, processes=None, cfg=None, seq_len=3):
        self.seq_len = seq_len
        self.split = split
        self.max_lanes = getattr(cfg, 'max_lanes', 24)
        super().__init__(data_root, split, cut_height, processes=processes, cfg=cfg)
        self.load_annotations()


    def get_dataset_info(self, data_root, split):
        import glob
        import pickle
        import os
        
        cache_file = os.path.join(data_root, f"openlane_temporal_cache_{split}_{self.seq_len}.pkl")
        if os.path.exists(cache_file):
            import gc
            self.logger.info(f"Loading cached temporal dataset from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        seq_infos = []
        sub_dir = "training" if split == "train" else "validation"
        anno_dir = os.path.join(data_root, sub_dir)
        json_files = glob.glob(os.path.join(anno_dir, "**/*.json"), recursive=True)
        
        segment_groups = {}
        # Pre-parse all JSONs
        frame_anno_dict = {}
        for json_path in tqdm(json_files, desc="Parsing all JSONs once"):
            rel_path = os.path.relpath(json_path, data_root)
            img_rel_path = rel_path.replace('.json', '.jpg')
            
            with open(json_path, 'r') as f:
                import json
                anno = json.load(f)
                

            lanes = []
            for lane_id, ln in enumerate(anno.get('lane_lines', [])):
                uv = ln.get('uv', [])
                if len(uv) == 2 and len(uv[0]) > 1:
                    u = uv[0]
                    v = uv[1]
                    min_len = min(len(u), len(v))
                    if min_len >= 2:
                        points = [(u[i], v[i]) for i in range(min_len) if v[i] >= self.cut_height]
                        if len(points) >= 2:
                            # sort by v descending
                            points.sort(key=lambda p: p[1], reverse=True)
                            lanes.append(points)

            
            abs_img_path = os.path.join(os.path.dirname(data_root), img_rel_path)
            
            frame_anno_dict[img_rel_path] = {
                'img_path': abs_img_path,
                'img_name': img_rel_path,
                'img_rel_path': abs_img_path, # fix __getitem__ issue as well
                'lanes': lanes,
                'cut_height': self.cut_height,
                'extrinsic': np.array(anno.get('extrinsic', [])), 
                'intrinsic': np.array(anno.get('intrinsic', []))
            }
            
            parts = img_rel_path.split('/')
            if len(parts) >= 2:
                segment_name = parts[-2]
                if segment_name not in segment_groups:
                    segment_groups[segment_name] = []
                segment_groups[segment_name].append(img_rel_path)
            
        for seg, paths in segment_groups.items():
            paths.sort() 
            for i in range(max(1, len(paths) - self.seq_len + 1)):
                if len(paths) >= self.seq_len:
                    clip_paths = paths[i : i + self.seq_len]
                else: 
                    clip_paths = [paths[0]] * (self.seq_len - len(paths)) + paths
                
                clip_anno = [frame_anno_dict[p] for p in clip_paths]
                seq_infos.append(clip_anno)
                
        with open(cache_file, "wb") as f:
            pickle.dump(seq_infos, f, protocol=pickle.HIGHEST_PROTOCOL)
        return seq_infos

    def load_annotations(self):
        self.logger.info(f"Loading OpenLane Temporal Dataset (T={self.seq_len})...")
        self.clip_cache = self.get_dataset_info(self.data_root, self.split)
        # 为 Evaluator 兼容提供 data_infos (提供当前帧 t 的真值信息)
        self.data_infos = [clip[-1] for clip in self.clip_cache]

            
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
            if "training" in img_path:
                resized_img_path = os.path.join(base_dir, "training_resized_800_320", rel_path)
            else:
                resized_img_path = os.path.join(base_dir, "validation_resized_800_320", rel_path)
                
            img = cv2.imread(resized_img_path)
            
            mask_path = os.path.join(base_dir, "mask_resized_800_320", rel_path).replace(".jpg", ".png")
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
                # Note: original image size is 1920x1280, generate_lane_mask_binary requires original image shape.
                # It accepts img just for its shape. We can pass a dummy image of 1280x1920 to get the 1280x1920 mask.
                dummy_img = np.zeros((1280, 1920, 3), dtype=np.uint8)
                mask_org = generate_lane_mask_binary(dummy_img, lanes_for_mask)
                # Then resize to 800x320
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
            
            # We need the original OpenLane tool to build a mask if offline not available
            mask_path = os.path.join(os.path.dirname(self.data_root), "mask", rel_path).replace(".jpg", ".png")
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

    def __getitem__(self, idx):
        clip_data = self.clip_cache[idx]
        
        processed_clip = []
        for frame in clip_data:
            processed_clip.append(self._process_single_frame(frame))
            
        return self.processes(processed_clip) 
