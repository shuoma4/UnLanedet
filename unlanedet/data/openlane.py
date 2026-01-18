import os
import os.path as osp
import json
import numpy as np
import cv2
import glob
import pickle
from tqdm import tqdm
from .base_dataset import BaseDataset

# OpenLane lane category mapping (保持不变)
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
    13: "left-curbside",
    14: "right-curbside",
}

VALID_LANE_CATEGORIES = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    20: 13,
    21: 14,
}

LEFT_RIGHT_ATTRIBUTES = {1: "left-left", 2: "left", 3: "right", 4: "right-right"}


class OpenLane(BaseDataset):
    def __init__(self, data_root, split, cut_height, processes=None, cfg=None):
        super().__init__(data_root, split, cut_height, processes=processes, cfg=cfg)

        self.cfg = cfg if cfg is not None else {}
        self.use_preprocessed = self.cfg.get("use_preprocessed", False)

        self.target_w = self.cfg.get("img_w", 800)
        self.target_h = self.cfg.get("img_h", 320)
        self.ori_w = 1920
        self.ori_h = 1280
        self.valid_ori_h = self.ori_h - self.cut_height

        if self.use_preprocessed:
            self.logger.info(
                f"[OpenLane] Preprocessed Mode ENABLED. Target: {self.target_w}x{self.target_h}"
            )
            self.h_scale = self.target_h / self.valid_ori_h
            self.w_scale = self.target_w / self.ori_w
        else:
            self.h_scale = 1.0
            self.w_scale = 1.0

        self.enable_attributes = self.cfg.get("enable_attributes", True)
        self.enable_tracking = self.cfg.get("enable_tracking", False)

        # 缓存文件名 (区分训练/验证集)
        self.cache_path = os.path.join(data_root, f"openlane_{split}_cache_compact.pkl")

        # 加载数据
        self.data_infos = self.load_annotations(split)

    def load_annotations(self, split):
        # 1. 尝试加载缓存
        if os.path.exists(self.cache_path):
            self.logger.info(f"Loading cached annotations from {self.cache_path}...")
            try:
                with open(self.cache_path, "rb") as f:
                    data_infos = pickle.load(f)
                self.logger.info(
                    f"Successfully loaded {len(data_infos)} samples from cache."
                )
                return data_infos
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Re-generating...")

        # 2. 如果缓存不存在，生成缓存
        self.logger.info("Generating annotation cache (This runs only once)...")
        data_infos = self._generate_cache(split)

        # 3. 保存缓存
        self.logger.info(f"Saving cache to {self.cache_path}...")
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(data_infos, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

        return data_infos

    def _generate_cache(self, split):
        """
        扫描 JSON 并提取核心数据，丢弃冗余信息以节省内存。
        """
        # 从配置中获取 lane_anno_dir，默认为 lane3d_300
        lane_anno_dir = self.cfg.get("lane_anno_dir", "lane3d_300/")
        
        if split == "train":
            anno_dirs = [osp.join(self.data_root, lane_anno_dir, "training")]
        elif split == "val" or split == "test":
            anno_dirs = [osp.join(self.data_root, lane_anno_dir, "validation")]
        else:
            # Fallback for generic split names
            anno_dirs = [
                osp.join(self.data_root, lane_anno_dir, "training"),
                osp.join(self.data_root, lane_anno_dir, "validation"),
            ]

        json_files = []
        for anno_dir in anno_dirs:
            if osp.exists(anno_dir):
                json_files.extend(
                    glob.glob(osp.join(anno_dir, "**/*.json"), recursive=True)
                )

        compact_infos = []

        for json_path in tqdm(json_files, desc="Parsing JSONs"):
            try:
                with open(json_path, "r") as f:
                    anno_data = json.load(f)

                # --- 提取核心数据 ---
                file_path = anno_data.get("file_path", "")
                if not file_path:
                    continue

                # 1. 预计算图片路径
                if self.use_preprocessed:
                    parts = file_path.split(os.sep)
                    if len(parts) > 1:
                        split_name = parts[0]
                        rest_path = os.sep.join(parts[1:])
                        new_folder = (
                            f"{split_name}_resized_{self.target_w}_{self.target_h}"
                        )
                        img_path = osp.join(self.data_root, new_folder, rest_path)
                    else:
                        img_path = osp.join(self.data_root, file_path)
                else:
                    img_path = osp.join(self.data_root, file_path)
                    if not osp.exists(img_path):
                        possible = osp.join(self.data_root, "dataset", "raw", file_path)
                        if osp.exists(possible):
                            img_path = possible

                # 2. 提取车道线坐标 (提前做完坐标转换，__getitem__里就不用做了)
                lanes_info = anno_data.get("lane_lines", [])
                lanes = []
                lane_categories = []
                lane_attributes = []

                for lane_info in lanes_info:
                    uv = lane_info.get("uv", [])
                    if len(uv) < 2 or len(uv[0]) < 2:
                        continue

                    # 转换为 numpy 提高处理速度，最后转回 list 或 array 存入
                    u_coords = np.array(uv[0], dtype=np.float32)
                    v_coords = np.array(uv[1], dtype=np.float32)

                    # 坐标转换 (Pre-calculation)
                    if self.use_preprocessed:
                        v_coords = v_coords - self.cut_height
                        u_coords = u_coords * self.w_scale
                        v_coords = v_coords * self.h_scale

                        # Vectorized boundary check
                        valid_mask = (
                            (u_coords >= 0)
                            & (u_coords < self.target_w)
                            & (v_coords >= 0)
                            & (v_coords < self.target_h)
                        )
                    else:
                        valid_mask = (v_coords >= self.cut_height) & (u_coords > 0)

                    u_valid = u_coords[valid_mask]
                    v_valid = v_coords[valid_mask]

                    if len(u_valid) < 2:
                        continue

                    # Stack points
                    points = np.stack([u_valid, v_valid], axis=1)

                    # 去重和排序 (Unique & Sort by Y)
                    # Lexsort to sort by Y (column 1)
                    # Note: Unique might reorder, so we sort after
                    _, idx = np.unique(points, axis=0, return_index=True)
                    points = points[idx]
                    points = points[points[:, 1].argsort()]

                    if len(points) < 2:
                        continue

                    # 存入列表 (转为 list 以便 pickle 序列化更通用，或者保持 numpy)
                    # 这里保持 numpy float32 极其节省内存
                    lanes.append(points.astype(np.float32))

                    if self.enable_attributes:
                        cat = lane_info.get("category", 0)
                        lane_categories.append(VALID_LANE_CATEGORIES.get(cat, 0))
                        lane_attributes.append(lane_info.get("attribute", 0))

                if len(lanes) == 0:
                    continue

                # 3. 构建紧凑字典
                # 只保留训练必须的字段
                sample = {
                    "img_path": img_path,
                    "img_name": file_path,
                    "lanes": lanes,  # List of numpy arrays
                }

                if self.enable_attributes:
                    # Pad attributes here to save time in getitem
                    max_lanes = self.cfg.get("max_lanes", 24)
                    l_cats = np.array(lane_categories, dtype=np.int64)
                    l_attrs = np.array(lane_attributes, dtype=np.int64)

                    if len(l_cats) > max_lanes:
                        l_cats = l_cats[:max_lanes]
                        l_attrs = l_attrs[:max_lanes]
                    elif len(l_cats) < max_lanes:
                        pad = max_lanes - len(l_cats)
                        l_cats = np.pad(l_cats, (0, pad), constant_values=0)
                        l_attrs = np.pad(l_attrs, (0, pad), constant_values=0)

                    sample["lane_categories"] = l_cats
                    sample["lane_attributes"] = l_attrs

                compact_infos.append(sample)

            except Exception as e:
                continue

        return compact_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        # 直接从内存获取数据，无需 IO 和 解析
        data_info = self.data_infos[idx]

        # 图片读取
        if not osp.isfile(data_info["img_path"]):
            return self.__getitem__((idx + 1) % len(self))

        import cv2
        from .transform import DataContainer as DC

        img = cv2.imread(data_info["img_path"])
        if img is None:
            return self.__getitem__((idx + 1) % len(self))

        # 裁剪 (仅非预处理模式)
        if not self.use_preprocessed:
            img = img[self.cut_height :, :, :]

        # 构造返回样本 (浅拷贝即可)
        sample = data_info.copy()
        sample["img"] = img

        # 生成 existence flag
        sample["lane_exist"] = np.ones(len(data_info["lanes"]), dtype=np.float32)

        # Transform
        sample = self.processes(sample)

        meta = {
            "full_img_path": data_info["img_path"],
            "img_name": data_info["img_name"],
        }
        sample.update({"meta": DC(meta, cpu_only=True)})

        return sample

    def get_category_name(self, category_id):
        return LANE_CATEGORIES.get(category_id, "unknown")

    def get_attribute_name(self, attribute_id):
        return LEFT_RIGHT_ATTRIBUTES.get(attribute_id, "unknown")
