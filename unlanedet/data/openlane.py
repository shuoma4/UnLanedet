import glob
import json
import logging
import os
import os.path as osp
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from .base_dataset import BaseDataset
from .transform import generate_lane_mask_binary

# OpenLane 官方类别映射
LANE_CATEGORIES = {
    0: 'unknown',
    1: 'white-dash',
    2: 'white-solid',
    3: 'double-white-dash',
    4: 'double-white-solid',
    5: 'white-ldash-rsolid',
    6: 'white-lsolid-rdash',
    7: 'yellow-dash',
    8: 'yellow-solid',
    9: 'double-yellow-dash',
    10: 'double-yellow-solid',
    11: 'yellow-ldash-rsolid',
    12: 'yellow-lsolid-rdash',
    13: 'left-curbside',
    14: 'right-curbside',
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

LANE_ATTRIBUTES = {
    0: 'unknown',
    1: 'left-left',
    2: 'left',
    3: 'right',
    4: 'right-right',
}


class OpenLane(BaseDataset):
    """
    OpenLane数据集类
    func: load openlane dataset, generate anno pkl cache file to speed up loading and training
    input:
        data_root: root directory of the dataset
        split: dataset split, e.g. train, val, test
        cut_height: height to cut the image from top, default is 270
        max_lanes: maximum number of lanes to keep, default is 24
        enable_3d: whether to enable 3d lane points, default is False

    getitem return to a dict:
        {
            "img": image after cut_height crop (H-cut_height, W, 3)
            "img_path": full path of the image
            "origin_img_path": original image path
            "lanes": 2d lane points list, each lane is (N, 2) [u, v] in original image coords,
                     v >= cut_height guaranteed; sorted by v descending (bottom to top)
            "mask": lane seg mask (H-cut_height, W), dtype=uint8, aligned with cropped img
            "lane_vis": visibility of each lane (N,), 1 means visible, 0 means invisible
            "lane_categories": lane category of each lane, mapped from VALID_LANE_CATEGORIES
            "lane_attributes": lane attributes of each lane, mapped from LANE_ATTRIBUTES
            "lane_track_ids": track id of each lane, -1 means no track id
            "intrinsic": camera intrinsic matrix (3, 3)
            "extrinsic": camera extrinsic matrix (4, 4)
            "pose": camera pose matrix (4, 4)
            "lanes_3d": 3d lane points (N, 3), sorted by v descending
            "segment_name": segment name
        }

    Note:
        lanes 中的 v 坐标保留原始图像坐标系（未减 cut_height）。
        下游 GenerateLaneLine / OpenLaneGenerate 的 __call__ 会统一减去 cut_height 与裁剪后的 img 对齐。
        mask 在此处生成时已减去 cut_height，与裁剪后的 img 对齐，供 transform 同步增强使用。
    """

    def __init__(self, data_root, split, cut_height, processes=None, cfg=None):
        super().__init__(data_root, split, cut_height, processes=processes, cfg=cfg)
        self.cfg = cfg if cfg is not None else {}
        # 基本参数
        self.max_lanes = self.cfg.get('max_lanes', 24)
        self.enable_3d = self.cfg.get('enable_3d', False)
        # 图像尺寸
        self.ori_w, self.ori_h = 1920, 1280
        self.cut_height = cut_height

        # 缓存文件路径：只包含数据集规模和cut_height
        if '300' in data_root:
            name_part = 'lane3d_300'
        elif '1000' in data_root:
            name_part = 'lane3d_1000'
        else:
            name_part = 'unknown'

        local_cache_dir = '/data1/lxy_log/workspace/ms/UnLanedet/output/.cache/'
        os.makedirs(local_cache_dir, exist_ok=True)
        self.cache_path = osp.join(
            local_cache_dir,
            f'openlane_{name_part}_{split}_cuth-{self.cut_height}.pkl',
        )

        self.logger = logging.getLogger(__name__)
        self.data_infos = self.load_annotations(split)
        self.logger.info(f'加载 {split} 数据集完成: {len(self.data_infos)} 个样本')

    def load_annotations(self, split):
        """加载数据标注"""
        if osp.exists(self.cache_path):
            self.logger.info(f'从缓存加载: {self.cache_path}')
            with open(self.cache_path, 'rb') as f:
                data_infos = pickle.load(f)
            return data_infos
        self.logger.info(f'生成新的缓存: {split}')
        data_infos = self._generate_cache(split)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data_infos, f)
        self.logger.info(f'缓存已保存: {self.cache_path}')
        return data_infos

    def _generate_cache(self, split):
        sub_dir = 'training' if split == 'train' else 'validation'
        anno_dir = osp.join(self.data_root, sub_dir)
        json_files = glob.glob(osp.join(anno_dir, '**/*.json'), recursive=True)
        self.logger.info(f'找到 {len(json_files)} 个JSON文件')

        data_infos = []
        processed_count = 0
        img_invalid_count = 0
        no_lanes_count = 0
        error_count = 0

        for json_path in tqdm(json_files, desc=f'处理 {split} 数据'):
            try:
                with open(json_path, 'r') as f:
                    anno = json.load(f)

                rel_file_path = anno.get('file_path', '')
                if not rel_file_path:
                    continue

                img_path = osp.join(osp.dirname(self.data_root), rel_file_path)
                if not osp.exists(img_path):
                    img_invalid_count += 1
                    continue

                lanes = []
                lanes_3d = []
                visibilities = []
                categories = []
                attributes = []
                track_ids = []

                for lane_info in anno.get('lane_lines', []):
                    uv = lane_info.get('uv', [])
                    if len(uv) != 2 or len(uv[0]) == 0:
                        continue
                    lane_points, visibility = self.process_lane_points(uv[0], uv[1], lane_info.get('visibility'))
                    if lane_points is not None and len(lane_points) >= 2:
                        lanes.append(lane_points)
                        visibilities.append(visibility)
                        categories.append(VALID_LANE_CATEGORIES.get(lane_info.get('category', 0), 0))
                        attributes.append(lane_info.get('attribute', 0))
                        track_ids.append(lane_info.get('track_id', -1))

                        xyz = lane_info.get('xyz', [])
                        if len(xyz) == 3 and len(xyz[0]) > 0:
                            lane_3d_points = self.process_3d_points(xyz[0], xyz[1], xyz[2], uv[1])
                            lanes_3d.append(lane_3d_points)
                        else:
                            lanes_3d.append(np.zeros((0, 3), dtype=np.float32))

                if len(lanes) == 0:
                    no_lanes_count += 1
                    continue

                sample = {
                    'img_path': img_path,
                    'origin_img_path': osp.join(osp.dirname(self.data_root), rel_file_path),
                    'lanes': lanes,
                    'lanes_3d': lanes_3d,
                    'lane_vis': visibilities,
                    'lane_categories': categories,
                    'lane_attributes': attributes,
                    'lane_track_ids': track_ids,
                    'json_path': json_path,
                }

                intrinsic = anno.get('intrinsic')
                extrinsic = anno.get('extrinsic')
                sample['intrinsic'] = np.array(intrinsic, dtype=np.float32) if intrinsic is not None else None
                sample['extrinsic'] = np.array(extrinsic, dtype=np.float32) if extrinsic is not None else None

                pose = anno.get('pose')
                sample['pose'] = np.array(pose, dtype=np.float32) if pose is not None else None

                parts = rel_file_path.split('/')
                if len(parts) >= 2:
                    sample['segment_name'] = parts[1]

                data_infos.append(sample)
                processed_count += 1

            except Exception as e:
                error_count += 1
                if error_count <= 10:
                    self.logger.info(f'处理 {json_path} 时出错: {str(e)[:200]}')
                continue

        self.logger.info(f'成功处理 {processed_count} 个样本, 错误 {error_count} 个')
        self.logger.info(f'{img_invalid_count} 个图像无效样本')
        self.logger.info(f'{no_lanes_count} 个无车道线或者无有效车道线样本(车道线点数目少于两个)')
        return data_infos

    def process_lane_points(self, u_coords, v_coords, visibility=None):
        """处理2D车道线点坐标和可见性，保留原始图像坐标系（v 未减 cut_height）"""
        if len(u_coords) != len(v_coords):
            min_len = min(len(u_coords), len(v_coords))
            u_coords = u_coords[:min_len]
            v_coords = v_coords[:min_len]
        u = np.array(u_coords, dtype=np.float32)
        v = np.array(v_coords, dtype=np.float32)

        if visibility is None or not isinstance(visibility, list) or len(visibility) != len(u):
            vis = np.ones_like(u, dtype=np.float32)
        else:
            if len(visibility) != len(u):
                min_len = min(len(visibility), len(u))
                visibility = visibility[:min_len]
                u = u[:min_len]
                v = v[:min_len]
            vis = np.array(visibility, dtype=np.float32)

        # 过滤天空区域（v < cut_height 的点）
        valid_mask = v >= self.cut_height
        u_valid, v_valid, vis_valid = u[valid_mask], v[valid_mask], vis[valid_mask]
        if len(u_valid) < 2:
            return None, None

        points = np.stack([u_valid, v_valid], axis=1)
        sort_idx = points[:, 1].argsort()[::-1]  # 按 v 降序（图像底部到顶部）
        return points[sort_idx], vis_valid[sort_idx]

    def process_3d_points(self, x_coords, y_coords, z_coords, v_coords_2d):
        min_len = min(len(x_coords), len(y_coords), len(z_coords), len(v_coords_2d))
        if min_len == 0:
            return np.zeros((0, 3), dtype=np.float32)
        x = np.array(x_coords[:min_len], dtype=np.float32)
        y = np.array(y_coords[:min_len], dtype=np.float32)
        z = np.array(z_coords[:min_len], dtype=np.float32)
        v_2d = np.array(v_coords_2d[:min_len], dtype=np.float32)

        valid_mask = v_2d >= self.cut_height
        x_valid, y_valid, z_valid = x[valid_mask], y[valid_mask], z[valid_mask]
        if len(x_valid) < 2:
            return np.zeros((0, 3), dtype=np.float32)

        points_3d = np.stack([x_valid, y_valid, z_valid], axis=1)
        sort_idx = points_3d[:, 1].argsort()[::-1]
        return points_3d[sort_idx]

    def __getitem__(self, idx):
        if idx >= len(self.data_infos):
            idx = idx % len(self.data_infos)
        data_info = self.data_infos[idx]

        img = cv2.imread(data_info['img_path'])
        img = img[self.cut_height :, :, :]

        # mask 生成时坐标需减去 cut_height，与裁剪后的 img 对齐
        lanes_for_mask = []
        for lane in data_info['lanes']:
            lane_shifted = lane.copy()
            lane_shifted[:, 1] -= self.cut_height
            lanes_for_mask.append(lane_shifted)
        mask = generate_lane_mask_binary(img, lanes_for_mask)

        sample = {
            'img': img,
            'lanes': data_info['lanes'],  # v 坐标保留原始坐标系，下游 transform 统一减 cut_height
            'mask': mask,  # 已与裁剪后 img 对齐，供 transform 同步增强
            'lane_vis': data_info['lane_vis'],
            'lane_categories': data_info['lane_categories'],
            'lane_attributes': data_info['lane_attributes'],
            'lane_track_ids': data_info['lane_track_ids'],
            'img_path': data_info['img_path'],
            'origin_img_path': data_info['origin_img_path'],
            'intrinsic': data_info['intrinsic'],
            'extrinsic': data_info['extrinsic'],
            'pose': data_info['pose'],
        }

        if self.enable_3d and 'lanes_3d' in data_info:
            sample['lanes_3d'] = data_info['lanes_3d']

        img_name = data_info['origin_img_path']
        parts = img_name.split('/')
        if len(parts) >= 2:
            sample['segment_name'] = parts[1]

        if self.processes is not None:
            sample = self.processes(sample)

        return sample

    def __len__(self):
        return len(self.data_infos)
