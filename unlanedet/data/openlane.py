import os
import os.path as osp
import json
import numpy as np
from .base_dataset import BaseDataset
from tqdm import tqdm
import logging

# OpenLane lane category mapping
# Including curbside lanes which are important for driving scenes
LANE_CATEGORIES = {
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
    13: "left-curbside",  # Remapped from 20 to 13
    14: "right-curbside",  # Remapped from 21 to 14
}

# Valid lane categories including curbside (remapped to 13, 14 for contiguous indices)
# Original categories: 0-12 (lanes), 20-21 (curbsides) → Remapped to: 0-14
VALID_LANE_CATEGORIES = {
    0: 0,  # unknown
    1: 1,  # white-dash
    2: 2,  # white-solid
    3: 3,  # double-white-dash
    4: 4,  # double-white-solid
    5: 5,  # white-ldash-rsolid
    6: 6,  # white-lsolid-rdash
    7: 7,  # yellow-dash
    8: 8,  # yellow-solid
    9: 9,  # double-yellow-dash
    10: 10,  # double-yellow-solid
    11: 11,  # yellow-ldash-rsolid
    12: 12,  # yellow-lsolid-rdash
    20: 13,  # left-curbside → 13
    21: 14,  # right-curbside → 14
}
NUM_LANE_CATEGORIES = 15  # 0-14 (13 regular lanes + 2 curbsides)

LEFT_RIGHT_ATTRIBUTES = {1: "left-left", 2: "left", 3: "right", 4: "right-right"}


def _load_json_file(json_path):
    """Helper function for parallel JSON loading (must be module-level for pickling)."""
    try:
        with open(json_path, "r") as f:
            anno_data = json.load(f)
            if "file_path" not in anno_data:
                return None
            return json_path, anno_data
    except (json.JSONDecodeError, IOError, OSError):
        return None


LIST_FILE = {
    "train": "list/train.txt",
    "val": "list/val.txt",
    "test": "list/test.txt",
}


class OpenLane(BaseDataset):
    def __init__(self, data_root, split, cut_height, processes=None, cfg=None):
        super().__init__(data_root, split, cut_height, processes=processes, cfg=cfg)
        self.data_infos = []

        # 🔧 修复：初始化 cfg
        self.cfg = cfg if cfg is not None else {}

        if cfg is not None:
            self.use_2d_only = cfg.get("use_2d_only", True)
            self.enable_attributes = cfg.get("enable_attributes", True)
            self.enable_tracking = cfg.get("enable_tracking", False)
        else:
            self.use_2d_only = True
            self.enable_attributes = True
            self.enable_tracking = False

        self.list_path = self._get_list_path(split)
        self.load_annotations()

    def _get_list_path(self, split):
        """OpenLane数据集直接扫描目录,不使用list文件"""
        return None

    def load_annotations(self):
        self.logger.info("Loading OpenLane annotations...")
        # OpenLane直接扫描目录加载标注文件
        self._scan_annotations()
        self.logger.info(f"Loaded {len(self.data_infos)} samples")

    def _scan_annotations(self):
        from multiprocessing import Pool, cpu_count

        anno_dirs = [
            osp.join(self.data_root, "lane3d_300", "training"),
            osp.join(self.data_root, "lane3d_300", "validation"),
            # osp.join(self.data_root, "lane3d_1000", "training"),
            # osp.join(self.data_root, "lane3d_1000", "validation"),
        ]

        # Step 1: Fast path collection using scandir (much faster than os.walk)
        json_paths = []
        for anno_dir in anno_dirs:
            if not osp.exists(anno_dir):
                continue
            self.logger.info(f"Scanning {anno_dir}...")
            try:
                with os.scandir(anno_dir) as entries:
                    for entry in entries:
                        if entry.is_dir():
                            try:
                                with os.scandir(entry.path) as sub_entries:
                                    for sub_entry in sub_entries:
                                        if (
                                            sub_entry.is_file()
                                            and sub_entry.name.endswith(".json")
                                        ):
                                            json_paths.append(sub_entry.path)
                            except (PermissionError, OSError):
                                continue
            except (PermissionError, OSError):
                continue

        self.logger.info(f"Found {len(json_paths)} annotation files, loading...")

        # Step 2: Parallel loading
        num_workers = min(4, cpu_count())  # 限制最大进程数为4
        chunk_size = max(1, len(json_paths) // (num_workers * 8))

        with Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_load_json_file, json_paths, chunksize=chunk_size),
                    total=len(json_paths),
                    desc="Loading annotations",
                )
            )

        # Step 3: Process loaded data
        for result in results:
            if result is None:
                continue
            json_path, anno_data = result
            infos = self._process_annotation_data(json_path, anno_data)
            if infos:
                self.data_infos.append(infos)

        self.logger.info(f"Loaded {len(self.data_infos)} valid samples")

    def _process_annotation_data(self, json_path, anno_data):
        """Process annotation data after it's been loaded."""
        infos = {}

        file_path = anno_data.get("file_path", "")
        if not file_path:
            return None

        img_path = osp.join(self.data_root, file_path)

        if not osp.exists(img_path):
            possible_bases = [
                self.data_root,
                osp.join(self.data_root, "dataset", "raw"),
            ]
            for base in possible_bases:
                possible_path = osp.join(base, file_path)
                if osp.exists(possible_path):
                    img_path = possible_path
                    break

        if not osp.exists(img_path):
            return None

        infos["img_name"] = file_path
        infos["img_path"] = img_path

        # Extract lane information
        lanes_info = anno_data.get("lane_lines", [])
        lanes = []
        lane_categories = []
        lane_attributes = []
        lane_track_ids = []

        for lane_info in lanes_info:
            uv = lane_info.get("uv", [])
            if len(uv) < 2 or len(uv[0]) < 2:
                continue
            u_coords = uv[0]
            v_coords = uv[1]
            lane_points = [
                (u_coords[i], v_coords[i])
                for i in range(len(u_coords))
                if u_coords[i] > 0 and v_coords[i] > 0
            ]
            if len(lane_points) < 2:
                continue

            # Keep order and deduplicate
            seen = set()
            lane_points_unique = []
            for point in lane_points:
                if point not in seen:
                    seen.add(point)
                    lane_points_unique.append(point)
            lane_points = lane_points_unique

            lane_points = sorted(lane_points, key=lambda p: p[1])
            lane_points = [p for p in lane_points if p[1] >= self.cut_height]
            if len(lane_points) < 2:
                continue
            lanes.append(lane_points)

            if self.enable_attributes:
                category = lane_info.get("category", 0)
                # Remap categories including curbside (20→13, 21→14)
                if category in VALID_LANE_CATEGORIES:
                    category = VALID_LANE_CATEGORIES[category]
                else:
                    category = 0  # unknown for any other category
                lane_categories.append(category)
                attribute = lane_info.get("attribute", 0)
                lane_attributes.append(attribute)

            if self.enable_tracking:
                track_id = lane_info.get("track_id", -1)
                lane_track_ids.append(track_id)

        lane_exist = np.ones(len(lanes), dtype=np.float32)
        infos["lanes"] = lanes

        if self.enable_attributes:
            max_lanes = self.cfg.get("max_lanes", 4)
            # Always ensure exactly max_lanes entries
            if len(lane_categories) > max_lanes:
                # Truncate if too many lanes
                lane_categories = lane_categories[:max_lanes]
                lane_attributes = lane_attributes[:max_lanes]
            elif len(lane_categories) < max_lanes:
                # Pad if too few lanes
                pad_len = max_lanes - len(lane_categories)
                lane_categories = np.pad(
                    lane_categories, (0, pad_len), constant_values=0
                )
                lane_attributes = np.pad(
                    lane_attributes, (0, pad_len), constant_values=0
                )

            infos["lane_categories"] = np.array(lane_categories, dtype=np.int64)
            infos["lane_attributes"] = np.array(lane_attributes, dtype=np.int64)

        if self.enable_tracking:
            infos["lane_track_ids"] = np.array(lane_track_ids, dtype=np.int64)
            max_lanes = self.cfg.get("max_lanes", 4)
            if len(lane_track_ids) < max_lanes:
                pad_len = max_lanes - len(lane_track_ids)
                infos["lane_track_ids"] = np.pad(
                    infos["lane_track_ids"], (0, pad_len), constant_values=-1
                )

        infos["lane_exist"] = lane_exist
        return infos

    def load_annotation(self, line_or_path):
        """Load annotation from a list file entry."""
        infos = {}

        if osp.exists(line_or_path) and line_or_path.endswith(".json"):
            # Load JSON and process
            with open(line_or_path, "r") as f:
                anno_data = json.load(f)
            return self._process_annotation_data(line_or_path, anno_data)
        else:
            # Parse line from list file
            parts = line_or_path.split()
            if len(parts) < 1:
                return None
            img_line = parts[0]
            img_line = img_line[1 if img_line[0] == "/" else 0 :]
            img_path = osp.join(self.data_root, img_line)
            if not osp.exists(img_path):
                self.logger.warning(f"Cannot find image: {img_path}")
                return None

            # Find corresponding JSON
            json_path = img_path.rsplit(".", 1)[0] + ".json"
            if not osp.exists(json_path):
                json_path = img_path + ".json"

            if osp.exists(json_path):
                with open(json_path, "r") as f:
                    anno_data = json.load(f)
                return self._process_annotation_data(json_path, anno_data)
            else:
                self.logger.warning(f"Cannot find annotation for {img_path}")
                return None

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        """重写__getitem__方法,OpenLane数据集不使用分割mask"""
        data_info = self.data_infos[idx]
        if not osp.isfile(data_info["img_path"]):
            raise FileNotFoundError(
                "cannot find file: {}".format(data_info["img_path"])
            )

        import cv2
        from .transform import DataContainer as DC

        img = cv2.imread(data_info["img_path"])
        img = img[self.cut_height :, :, :]
        sample = data_info.copy()
        sample.update({"img": img})

        # 传递lane_categories和lane_attributes到sample
        if "lane_categories" in data_info:
            sample["lane_categories"] = data_info["lane_categories"]
        if "lane_attributes" in data_info:
            sample["lane_attributes"] = data_info["lane_attributes"]

        # OpenLane数据集不使用分割mask,跳过mask读取
        # 直接进行数据预处理
        sample = self.processes(sample)
        meta = {
            "full_img_path": data_info["img_path"],
            "img_name": data_info["img_name"],
        }
        meta = DC(meta, cpu_only=True)
        sample.update({"meta": meta})

        return sample

    def get_category_name(self, category_id):
        return LANE_CATEGORIES.get(category_id, "unknown")

    def get_attribute_name(self, attribute_id):
        return LEFT_RIGHT_ATTRIBUTES.get(attribute_id, "unknown")
