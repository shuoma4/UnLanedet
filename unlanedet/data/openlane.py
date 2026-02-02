import os.path as osp
import json
import numpy as np
import cv2
import glob
import pickle
from tqdm import tqdm
from .base_dataset import BaseDataset
import logging

# OpenLane 官方类别映射
LANE_CATEGORIES = {
    0: "unknown",  # 未知
    1: "white-dash",  # 白色虚线
    2: "white-solid",  # 白色实线
    3: "double-white-dash",  # 双白虚线
    4: "double-white-solid",  # 双白实线
    5: "white-ldash-rsolid",  # 左白虚右白实
    6: "white-lsolid-rdash",  # 左白实右白虚
    7: "yellow-dash",  # 黄色虚线
    8: "yellow-solid",  # 黄色实线
    9: "double-yellow-dash",  # 双黄虚线
    10: "double-yellow-solid",  # 双黄实线
    11: "yellow-ldash-rsolid",  # 左黄虚右黄实
    12: "yellow-lsolid-rdash",  # 左黄实右黄虚
    13: "left-curbside",  # 左侧路缘
    14: "right-curbside",  # 右侧路缘
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


class OpenLane(BaseDataset):
    """
    完整的OpenLane数据集类，修复了维度不匹配的错误
    """

    def __init__(self, data_root, split, cut_height, processes=None, cfg=None):
        super().__init__(data_root, split, cut_height, processes=processes, cfg=cfg)
        self.cfg = cfg if cfg is not None else {}
        # 基本参数
        self.use_preprocessed = self.cfg.get("use_preprocessed", False)
        self.img_w = self.cfg.get("img_w", 800)
        self.img_h = self.cfg.get("img_h", 320)
        self.max_lanes = self.cfg.get("max_lanes", 24)
        self.enable_3d = self.cfg.get("enable_3d", True)
        self.enable_attributes = self.cfg.get("enable_attributes", True)
        # 图像尺寸
        self.ori_w, self.ori_h = 1920, 1280
        self.cut_height = cut_height
        self.valid_ori_h = self.ori_h - self.cut_height
        # 缩放比例
        self.h_scale = self.img_h / self.valid_ori_h
        self.w_scale = self.img_w / self.ori_w
        # 缓存文件路径
        lane_anno_dir = self.cfg.get("lane_anno_dir", "lane3d_300/")
        if "300" in lane_anno_dir:
            name_part = "lane3d_300"
        elif "1000" in lane_anno_dir:
            name_part = "lane3d_1000"
        else:
            name_part = "unknown"
        self.cache_path = osp.join(
            data_root,
            f"openlane_{name_part}_{split}_cuth-{self.cut_height}_{self.img_w}x{self.img_h}_cache_v1.pkl",
        )
        self.data_infos = self.load_annotations(split)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"加载 {split} 数据集完成: {len(self.data_infos)} 个样本")

    def load_annotations(self, split):
        """加载数据标注"""
        if osp.exists(self.cache_path):
            self.logger.info(f"从缓存加载: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                data_infos = pickle.load(f)
            return data_infos
        self.logger.info(f"生成新的缓存: {split}")
        data_infos = self._generate_cache(split)
        with open(self.cache_path, "wb") as f:
            pickle.dump(data_infos, f)
        self.logger.info(f"缓存已保存: {self.cache_path}")
        return data_infos

    def _generate_cache(self, split):
        lane_anno_dir = self.cfg.get("lane_anno_dir", "lane3d_300/")
        sub_dir = "training" if split == "train" else "validation"
        anno_dir = osp.join(self.data_root, lane_anno_dir, sub_dir)
        # 查找所有JSON文件
        json_files = glob.glob(osp.join(anno_dir, "**/*.json"), recursive=True)
        self.logger.info(f"找到 {len(json_files)} 个JSON文件")
        data_infos = []
        processed_count = 0
        img_invalid_count = 0
        no_lanes_count = 0
        error_count = 0
        for json_path in tqdm(json_files, desc=f"处理 {split} 数据"):
            try:
                with open(json_path, "r") as f:
                    anno = json.load(f)
                # 获取文件路径
                rel_file_path = anno.get("file_path", "")
                if not rel_file_path:
                    continue
                # 构建图像路径
                if self.use_preprocessed:
                    parts = rel_file_path.split("/")
                    if len(parts) >= 2:
                        split_name = parts[0]  # training 或 validation
                        rest_path = "/".join(parts[1:])
                        img_path = osp.join(
                            self.data_root,
                            f"{split_name}_resized_{self.img_w}_{self.img_h}",
                            rest_path,
                        )
                    else:
                        img_path = osp.join(self.data_root, rel_file_path)
                else:
                    img_path = osp.join(self.data_root, rel_file_path)
                img = cv2.imread(img_path)
                if img is None:
                    img_invalid_count += 1
                    continue
                # 处理车道线数据
                lanes = []
                lanes_3d = []
                visibilities = []
                categories = []
                attributes = []
                track_ids = []
                for lane_info in anno.get("lane_lines", []):
                    uv = lane_info.get("uv", [])
                    if len(uv) != 2 or len(uv[0]) == 0:
                        continue
                    # 处理2D车道线点
                    lane_points, visibility = self.process_lane_points(
                        uv[0], uv[1], lane_info.get("visibility")
                    )
                    if lane_points is not None and len(lane_points) >= 2:
                        lanes.append(lane_points)
                        visibilities.append(visibility)
                        categories.append(
                            VALID_LANE_CATEGORIES.get(lane_info.get("category", 0), 0)
                        )
                        attributes.append(lane_info.get("attribute", 0))
                        track_ids.append(lane_info.get("track_id", -1))

                        xyz = lane_info.get("xyz", [])
                        if len(xyz) == 3 and len(xyz[0]) > 0:
                            lane_3d_points = self.process_3d_points(
                                xyz[0], xyz[1], xyz[2], uv[1]  # 使用2D的v坐标过滤
                            )
                            lanes_3d.append(lane_3d_points)
                        else:
                            lanes_3d.append(np.zeros((0, 3), dtype=np.float32))

                if len(lanes) == 0:
                    no_lanes_count += 1
                    continue  # 跳过没有有效车道线的样本

                # 构建样本
                sample = {
                    "img_path": img_path,  # 用于训练的图片完整路径(无论是否经过预处理)
                    "img_name": rel_file_path,  # 原图路径(相对于data_root)
                    "lanes": lanes,  # 2D车道线坐标
                    "lanes_3d": lanes_3d,  # 3D车道线坐标
                    "lane_vis": visibilities,  # 可见性信息
                    "lane_categories": categories,  # 车道线类别
                    "lane_attributes": attributes,  # 车道线属性
                    "lane_track_ids": track_ids,  # 时序tracking ID
                    "json_path": json_path,  # 标注文件路径
                }

                # 添加相机参数
                intrinsic = anno.get("intrinsic")
                extrinsic = anno.get("extrinsic")
                if intrinsic is not None:
                    sample["intrinsic"] = np.array(intrinsic, dtype=np.float32)
                else:
                    sample["intrinsic"] = None
                if extrinsic is not None:
                    sample["extrinsic"] = np.array(extrinsic, dtype=np.float32)
                else:
                    sample["extrinsic"] = None
                # 添加位姿
                pose = anno.get("pose")
                if pose is not None:
                    sample["pose"] = np.array(pose, dtype=np.float32)
                else:
                    sample["pose"] = None
                data_infos.append(sample)
                processed_count += 1

            except Exception as e:
                error_count += 1
                if error_count <= 10:  # 只打印前10个错误
                    self.logger.info(f"处理 {json_path} 时出错: {str(e)[:200]}")
                continue

        self.logger.info(f"成功处理 {processed_count} 个样本, 错误 {error_count} 个")
        self.logger.info(f"{img_invalid_count} 个图像无效样本")
        self.logger.info(
            f"{no_lanes_count} 个无车道线或者无有效车道线样本(车道线点数目少于两个)"
        )
        return data_infos

    def process_lane_points(self, u_coords, v_coords, visibility=None):
        """处理2D车道线点坐标和可见性"""
        # 确保u和v坐标长度一致
        if len(u_coords) != len(v_coords):
            min_len = min(len(u_coords), len(v_coords))
            u_coords = u_coords[:min_len]
            v_coords = v_coords[:min_len]
        u = np.array(u_coords, dtype=np.float32)
        v = np.array(v_coords, dtype=np.float32)
        # 处理可见性
        if (
            visibility is None
            or not isinstance(visibility, list)
            or len(visibility) != len(u)
        ):
            vis = np.ones_like(u, dtype=np.float32)
        else:
            # 确保可见性数组长度与坐标一致
            if len(visibility) != len(u):
                min_len = min(len(visibility), len(u))
                visibility = visibility[:min_len]
                u = u[:min_len]
                v = v[:min_len]
            vis = np.array(visibility, dtype=np.float32)
        # 过滤天空区域
        valid_mask = v >= self.cut_height
        u_valid, v_valid, vis_valid = u[valid_mask], v[valid_mask], vis[valid_mask]
        if len(u_valid) < 2:  # 有效车道线点数少于两个
            return None, None
        # 坐标转换
        if self.use_preprocessed:
            # 预处理图像：已经缩放，只需要调整坐标
            v_transformed = (v_valid - self.cut_height) * self.h_scale
            u_transformed = u_valid * self.w_scale
        else:
            # 原始图像：需要裁剪和缩放
            v_transformed = v_valid - self.cut_height
            u_transformed = u_valid
        points = np.stack([u_transformed, v_transformed], axis=1)
        sort_idx = points[:, 1].argsort()[::-1]
        return points[sort_idx], vis_valid[sort_idx]

    def process_3d_points(self, x_coords, y_coords, z_coords, v_coords_2d):
        """处理3D车道线点坐标 - 修复维度不匹配错误"""
        # 确保所有数组长度一致
        min_len = min(len(x_coords), len(y_coords), len(z_coords), len(v_coords_2d))
        if min_len == 0:
            return np.zeros((0, 3), dtype=np.float32)
        # 截取到相同长度
        x = np.array(x_coords[:min_len], dtype=np.float32)
        y = np.array(y_coords[:min_len], dtype=np.float32)
        z = np.array(z_coords[:min_len], dtype=np.float32)
        v_2d = np.array(v_coords_2d[:min_len], dtype=np.float32)
        # 使用2D坐标的v值来过滤3D点（确保与2D点对应）
        valid_mask = v_2d >= self.cut_height
        x_valid, y_valid, z_valid = x[valid_mask], y[valid_mask], z[valid_mask]
        if len(x_valid) < 2:
            return np.zeros((0, 3), dtype=np.float32)
        points_3d = np.stack([x_valid, y_valid, z_valid], axis=1)
        # 按y坐标排序（与2D保持一致）
        sort_idx = points_3d[:, 1].argsort()
        return points_3d[sort_idx]

    def __getitem__(self, idx):
        """
        获取单个样本

        Args:
            idx (int): 样本索引

        Returns:
            dict: 包含所有属性的样本字典
        """
        if idx >= len(self.data_infos):
            idx = idx % len(self.data_infos)
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info["img_path"])
        # 裁剪天空区域（如果是原始图像），此处无需再resize
        if not self.use_preprocessed:
            img = img[self.cut_height :, :, :]
        # 调整图像尺寸（如果是原始图像）
        if not self.use_preprocessed and (
            img.shape[0] != self.img_h or img.shape[1] != self.img_w
        ):
            img = cv2.resize(img, (self.img_w, self.img_h))

        # 构建样本
        sample = {
            "img": img,
            "lanes": data_info["lanes"],  # 2D车道线坐标
            "lane_vis": data_info["lane_vis"],  # 可见性信息
            "lane_categories": data_info["lane_categories"],  # 车道线类别
            "lane_attributes": data_info["lane_attributes"],  # 车道线属性
            "lane_track_ids": data_info["lane_track_ids"],  # 时序tracking ID
            "img_path": data_info["img_path"],  # 图片完整路径(无论是否经过预处理)
            "img_name": data_info["img_name"],  # 原图路径(相对于data_root)
            "intrinsic": data_info["intrinsic"],  # 相机内参
            "extrinsic": data_info["extrinsic"],  # 相机外参
            "pose": data_info["pose"],  # 相机位姿
        }
        # 添加3D车道线（如果存在）
        if self.enable_3d and "lanes_3d" in data_info:
            sample["lanes_3d"] = data_info["lanes_3d"]
        # 添加segment信息（时序部分有用）
        img_name = data_info["img_name"]
        parts = img_name.split("/")
        if len(parts) >= 2:
            sample["segment_name"] = parts[1]
        # 应用数据增强/预处理
        if self.processes is not None:
            sample = self.processes(sample)

        return sample

    def __len__(self):
        """返回数据集大小"""
        return len(self.data_infos)
