import os.path as osp
import numpy as np
import math
import imgaug.augmenters as iaa
import cv2
import torch
import logging
from scipy.interpolate import InterpolatedUnivariateSpline

# 强制关闭 OpenCV 多线程
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from omegaconf import DictConfig


def convert_dictconfig_to_dict(config):
    if isinstance(config, DictConfig):
        new_dict = {}
        for key, value in config.items():
            new_dict[key] = convert_dictconfig_to_dict(value)
        return new_dict
    else:
        return config


class GenerateLaneLineOpenLane(object):
    def __init__(self, transforms=None, cfg=None, training=True):
        self.transform_dict = transforms
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        # Y轴坐标：从底部(img_h)到顶部(0)
        self.offsets_ys = np.linspace(self.img_h, 0, self.num_points)
        self.training = training
        self.cut_height = getattr(cfg, "cut_height", 270)
        self.enable_3d = getattr(cfg, "enable_3d", False)
        self.mean = np.array(cfg.img_norm["mean"], dtype=np.float32)
        self.std = np.array(cfg.img_norm["std"], dtype=np.float32)
        self.transform = self._build_transform_pipeline(transforms)

    def _build_transform_pipeline(self, transforms):
        """构建imgaug数据增强管道

        Args:
            transforms: 数据增强配置列表，可为None

        Returns:
            iaa.Sequential: 数据增强序列
        """
        if transforms is None or not transforms:
            return iaa.Sequential([])

        # 将OmegaConf的DictConfig转换为普通字典
        transforms_dicts = []
        for aug in transforms:
            if aug is None:
                continue
            transforms_dicts.append(convert_dictconfig_to_dict(aug))

        if not transforms_dicts:
            return iaa.Sequential([])

        img_transforms = []

        for aug_config in transforms_dicts:
            # 验证增强配置
            if not isinstance(aug_config, dict):
                self.logger.warning(f"无效的增强配置类型: {type(aug_config)}")
                continue
            p = aug_config.get("p", 1.0)  # 默认概率为1.0
            if aug_config["name"] == "OneOf":
                # 处理OneOf增强（从多个中随机选一个）
                oneof_transforms = []
                for sub_aug in aug_config.get("transforms", []):
                    if "name" not in sub_aug or "parameters" not in sub_aug:
                        self.logger.warning(f"OneOf中的增强配置无效: {sub_aug}")
                        continue
                    try:
                        transform_cls = getattr(iaa, sub_aug["name"])
                        transform_instance = transform_cls(**sub_aug["parameters"])
                        oneof_transforms.append(transform_instance)
                    except (AttributeError, TypeError) as e:
                        self.logger.warning(f"无法创建增强 {sub_aug['name']}: {e}")
                        continue

                if oneof_transforms:
                    oneof_transform = iaa.OneOf(oneof_transforms)
                    img_transforms.append(iaa.Sometimes(p, oneof_transform))
            else:
                # 处理普通增强
                try:
                    transform_cls = getattr(iaa, aug_config["name"])
                    transform_instance = transform_cls(
                        **aug_config.get("parameters", {})
                    )
                    img_transforms.append(iaa.Sometimes(p, transform_instance))
                except (AttributeError, TypeError) as e:
                    self.logger.warning(f"无法创建增强 {aug_config['name']}: {e}")
                    continue
        return iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            pts = np.array(lane)[:, :2]
            lines.append(LineString(pts))
        return lines

    def linestrings_to_lanes(self, line_strings):
        lanes = []
        for ls in line_strings:
            lanes.append(ls.coords)
        return lanes

    def sample_lane_fixed(self, points, num_samples):
        """固定点弧长均匀采样"""
        points = np.array(points, dtype=np.float32)
        if len(points) < 2:
            return np.array([]), np.array([])
        sort_idx = np.argsort(points[:, 1])[::-1]
        points = points[sort_idx]
        _, unique_indices = np.unique(points[:, 1], return_index=True)
        valid_points = [points[0]]
        for i in range(1, len(points)):
            if abs(points[i, 1] - points[i - 1, 1]) > 1e-4:
                valid_points.append(points[i])
        points = np.array(valid_points)
        if len(points) < 2:
            return np.array([]), np.array([])
        diffs = points[1:] - points[:-1]
        seg_lens = np.sqrt((diffs**2).sum(axis=1))
        arc_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = arc_len[-1]
        if total_len < 1e-6:
            return np.array([]), np.array([])
        target_lens = np.linspace(0, total_len, num_samples)
        sampled = []
        for t in target_lens:
            idx = np.searchsorted(arc_len, t) - 1
            idx = np.clip(idx, 0, len(points) - 2)
            l0, l1 = arc_len[idx], arc_len[idx + 1]
            ratio = (t - l0) / (l1 - l0 + 1e-6)
            p = points[idx] + ratio * (points[idx + 1] - points[idx])
            sampled.append(p)
        sampled = np.array(sampled)
        return sampled[:, 0], sampled[:, 1]

    def sample_lane(self, points, sample_ys, visibility):
        dense_x, dense_y = self.sample_lane_fixed(points, num_samples=72)
        if len(dense_x) < 2:
            return np.zeros_like(sample_ys) - 1e5, np.zeros_like(sample_ys)
        interp_xs = np.interp(
            sample_ys, dense_y[::-1], dense_x[::-1], left=-1e5, right=-1e5
        )
        if sample_ys[0] > dense_y[0]:
            p1_x, p1_y = dense_x[0], dense_y[0]
            p2_x, p2_y = dense_x[1], dense_y[1]
            slope = (p2_x - p1_x) / (p2_y - p1_y + 1e-6)
            extrap_mask = sample_ys > p1_y
            interp_xs[extrap_mask] = p1_x + (sample_ys[extrap_mask] - p1_y) * slope
        all_vis = np.where(
            (interp_xs >= 0) & (interp_xs < self.img_w) & (interp_xs > -1e4),
            1.0,
            0.0,
        )
        return interp_xs, all_vis

    def transform_annotation(self, anno):
        old_lanes = anno["lanes"]
        old_visibilities = anno.get("lane_vis", [])
        old_categories = anno.get("lane_categories", [0] * len(old_lanes))
        old_attributes = anno.get("lane_attributes", [0] * len(old_lanes))
        if not old_visibilities or len(old_visibilities) != len(old_lanes):
            old_visibilities = [None] * len(old_lanes)
        if len(old_lanes) > 0:
            combined = list(
                zip(old_lanes, old_visibilities, old_categories, old_attributes)
            )
            combined.sort(key=lambda x: -x[0][0][1])  # y坐标从大到小排列
            old_lanes, old_visibilities, old_categories, old_attributes = zip(*combined)
        lanes = (
            np.ones(
                (self.max_lanes, 2 + 1 + 1 + 1 + 1 + self.n_offsets), dtype=np.float32
            )
            * -1e5
        )
        lanes_endpoints = np.ones((self.max_lanes, 2))
        lanes_vis_interpolated = np.zeros(
            (self.max_lanes, self.n_offsets), dtype=np.float32
        )
        padded_categories = np.zeros(self.max_lanes, dtype=np.int64)
        padded_attributes = np.zeros(self.max_lanes, dtype=np.int64)
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break
            visibility = old_visibilities[lane_idx]
            try:
                all_xs, all_vis = self.sample_lane(lane, self.offsets_ys, visibility)
            except Exception as e:
                self.logger.error(f"catch error : {e}")
                continue
            valid_mask = (all_xs >= 0) & (all_xs < self.img_w)
            if valid_mask.sum() < 2:
                continue
            valid_indices = np.where(valid_mask)[0]
            start_index = valid_indices[0]
            xs_inside = all_xs[valid_mask]
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            y_start_phys = self.offsets_ys[start_index]
            lanes[lane_idx, 2] = y_start_phys / self.img_h
            lanes[lane_idx, 3] = xs_inside[0] / self.img_w
            ys_inside = self.offsets_ys[valid_indices]
            if len(xs_inside) > 1:
                p = np.polyfit(ys_inside, xs_inside, 1)
                k = p[0]  # dx/dy (pixel space)
                theta = 0.5 + math.atan(-k) / math.pi
            else:
                theta = 0.5
            lanes[lane_idx, 4] = theta
            lanes[lane_idx, 5] = len(xs_inside) / self.n_strips
            target_indices_slice = lanes[lane_idx, 6:]
            target_indices_slice[valid_mask] = xs_inside / self.img_w
            lanes[lane_idx, 6:] = target_indices_slice
            lanes_vis_interpolated[lane_idx, :] = all_vis
            lanes_endpoints[lane_idx, 0] = (
                start_index + len(xs_inside) - 1
            ) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside[-1] / self.img_w
            padded_categories[lane_idx] = old_categories[lane_idx]
            padded_attributes[lane_idx] = old_attributes[lane_idx]

        new_anno = {
            "label": lanes,
            "lane_endpoints": lanes_endpoints,
            "lane_vis_interpolated": lanes_vis_interpolated,
            "lanes": old_lanes,
            "padded_categories": padded_categories,
            "padded_attributes": padded_attributes,
        }
        return new_anno

    def __call__(self, sample):
        img_org = sample["img"]
        img_h_curr, img_w_curr = img_org.shape[:2]
        line_strings_org = self.lane_to_linestrings(sample["lanes"])
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        dummy_mask_arr = np.zeros((img_h_curr, img_w_curr, 1), dtype=np.uint8)
        mask_org = SegmentationMapsOnImage(dummy_mask_arr, shape=img_org.shape)
        # Perform augmentation with retries
        cnt = 0
        while True:
            cnt += 1
            try:
                if self.training:
                    img, line_strings, _ = self.transform(
                        image=img_org.copy().astype(np.uint8),
                        line_strings=line_strings_org,
                        segmentation_maps=mask_org,
                    )
                else:
                    img, line_strings = self.transform(
                        image=img_org.copy().astype(np.uint8),
                        line_strings=line_strings_org,
                    )
                aug_lanes = self.linestrings_to_lanes(line_strings)
                current_vis = sample.get("lane_vis", [])
                new_anno_input = {
                    "lanes": aug_lanes,
                    "lane_vis": current_vis,
                    "lane_categories": sample.get("lane_categories", []),
                    "lane_attributes": sample.get("lane_attributes", []),
                }
                annos = self.transform_annotation(new_anno_input)
                label = annos["label"]
                lane_endpoints = annos["lane_endpoints"]
                lane_vis_interpolated = annos["lane_vis_interpolated"]

                # Check validity: if valid lanes exist or we've tried enough
                # If training, we prefer to have at least one lane if possible
                if not self.training or np.sum(label[:, 1] == 1) > 0 or cnt >= 10:
                    break
            except Exception as e:
                if cnt >= 10:
                    self.logger.warning(f"Augmentation failed 10 times: {e}")
                    raise e
                continue

        padded_categories = annos["padded_categories"]
        padded_attributes = annos["padded_attributes"]
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        sample["img"] = img
        sample["lane_line"] = label
        sample["lanes_endpoints"] = lane_endpoints
        sample["lane_vis_interpolated"] = lane_vis_interpolated
        sample["gt_points"] = new_anno_input["lanes"]
        sample["lane_categories"] = padded_categories
        sample["lane_attributes"] = padded_attributes
        mask_scale = 8
        mask_h, mask_w = int(self.img_h // mask_scale), int(self.img_w // mask_scale)
        seg_map = np.zeros((mask_h, mask_w), dtype=np.uint8)
        try:
            lanes_points = new_anno_input["lanes"]
            for lane in lanes_points:
                if len(lane) < 2:
                    continue
                pts = np.array([lane], dtype=np.float32)
                pts = pts / mask_scale
                pts = pts.astype(np.int32)
                cv2.polylines(seg_map, pts, isClosed=False, color=1, thickness=1)
        except:
            pass
        sample["seg"] = seg_map.astype(np.int64)
        return sample
