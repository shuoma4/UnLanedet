import os.path as osp
import numpy as np
import math
import imgaug.augmenters as iaa
import cv2
import torch

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


def CLRTransformsOpenLane(img_h, img_w):
    return [
        dict(
            name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0
        ),
        dict(name="HorizontalFlip", parameters=dict(p=1.0), p=0.5),
        dict(
            name="Affine",
            parameters=dict(
                translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)),
                rotate=(-10, 10),
                scale=(0.8, 1.2),
            ),
            p=0.7,
        ),
        dict(
            name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0
        ),
    ]


class GenerateLaneLineOpenLane(object):
    def __init__(self, transforms=None, cfg=None, training=True):
        self.transforms = transforms
        self.cfg = cfg
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        
        # Y轴坐标：从底部(img_h)到顶部(0)
        self.offsets_ys = np.linspace(self.img_h, 0, self.num_points)
        self.training = training

        self.use_preprocessed = getattr(cfg, "use_preprocessed", False)
        self.cut_height = getattr(cfg, "cut_height", 270)
        self.enable_3d = getattr(cfg, "enable_3d", False)

        self.mean = np.array(cfg.img_norm["mean"], dtype=np.float32)
        self.std = np.array(cfg.img_norm["std"], dtype=np.float32)

        if transforms is None:
            transforms = CLRTransformsOpenLane(self.img_h, self.img_w)

        if transforms is not None:
            transforms = [convert_dictconfig_to_dict(aug) for aug in transforms]
            img_transforms = []
            for aug in transforms:
                p = aug["p"]
                if aug["name"] != "OneOf":
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=getattr(iaa, aug["name"])(**aug["parameters"]),
                        )
                    )
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf(
                                [
                                    getattr(iaa, aug_["name"])(**aug_["parameters"])
                                    for aug_ in aug["transforms"]
                                ]
                            ),
                        )
                    )
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

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

    def sample_lane(self, points, sample_ys, visibility):
        points = np.array(points)
        if len(points) < 2:
            raise Exception("Too few points")
            
        x, y = points[:, 0], points[:, 1]
        
        if visibility is not None:
            vis = np.array(visibility)
            if len(vis) != len(x):
                min_len = min(len(vis), len(x))
                vis = vis[:min_len]
                x = x[:min_len]
                y = y[:min_len]
        else:
            vis = np.ones_like(x)

        sort_idx = np.argsort(y)
        x = x[sort_idx]
        y = y[sort_idx]
        vis = vis[sort_idx]

        unique_y, unique_indices = np.unique(y, return_index=True)
        unique_x = x[unique_indices]
        unique_vis = vis[unique_indices]

        if len(unique_y) < 2:
            raise Exception("Too few points after dedup")

        all_xs = np.interp(sample_ys, unique_y, unique_x, left=-1e5, right=-1e5)
        all_vis = np.interp(sample_ys, unique_y, unique_vis, left=0.0, right=0.0)
        
        return all_xs, all_vis

    def transform_annotation(self, anno, img_wh):
        old_lanes = anno["lanes"]
        old_visibilities = anno.get("lane_vis", [])
        
        old_categories = anno.get("lane_categories", [0] * len(old_lanes))
        old_attributes = anno.get("lane_attributes", [0] * len(old_lanes))
        
        if not old_visibilities or len(old_visibilities) != len(old_lanes):
            old_visibilities = [None] * len(old_lanes)

        valid_indices = [i for i, x in enumerate(old_lanes) if len(x) > 1]
        old_lanes = [old_lanes[i] for i in valid_indices]
        old_visibilities = [old_visibilities[i] for i in valid_indices]
        old_categories = [old_categories[i] for i in valid_indices]
        old_attributes = [old_attributes[i] for i in valid_indices]

        if len(old_lanes) > 0:
            combined = list(zip(old_lanes, old_visibilities, old_categories, old_attributes))
            combined.sort(key=lambda x: -x[0][0][1] if len(x[0]) > 0 else 0)
            old_lanes, old_visibilities, old_categories, old_attributes = zip(*combined)

        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + 1 + self.n_offsets), dtype=np.float32) * -1e5
        lanes_endpoints = np.ones((self.max_lanes, 2))
        lanes_vis_interpolated = np.zeros((self.max_lanes, self.n_offsets), dtype=np.float32)
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
            except Exception:
                continue
            
            valid_mask = (all_xs >= 0) & (all_xs < self.img_w)
            
            if valid_mask.sum() < 2:
                continue
                
            valid_indices = np.where(valid_mask)[0]
            start_index = valid_indices[0] 
            xs_inside = all_xs[valid_mask]
            
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1

            # 1. Start Y
            lanes[lane_idx, 2] = self.offsets_ys[start_index] / self.img_h
            
            # 2. Start X
            lanes[lane_idx, 3] = xs_inside[0] / self.img_w

            # 3. Theta (全局计算，更稳健)
            # 使用起点和终点计算整体斜率
            x_start = xs_inside[0]
            x_end = xs_inside[-1]
            y_start = self.offsets_ys[start_index]
            y_end = self.offsets_ys[start_index + len(xs_inside) - 1]
            
            dx = x_end - x_start
            dy = y_end - y_start # 注意：OpenLane中y向上是减小，但这里我们用物理距离
            # 实际上我们希望 Theta 0.5 对应垂直。
            # 如果 dy 用 y_index 的差值 (正数)，dx (正为右)。
            # dy_phys = len * strip_size
            dy_phys = (len(xs_inside) - 1) * self.strip_size
            
            theta = 0.5 + math.atan2(dx, dy_phys) / math.pi
            lanes[lane_idx, 4] = theta

            # 4. Length
            lanes[lane_idx, 5] = len(xs_inside) / self.n_strips

            # 5. Points
            target_indices_slice = lanes[lane_idx, 6:]
            target_indices_slice[valid_mask] = xs_inside / self.img_w
            lanes[lane_idx, 6:] = target_indices_slice
            
            lanes_vis_interpolated[lane_idx, :] = all_vis

            lanes_endpoints[lane_idx, 0] = (start_index + len(xs_inside) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside[-1] / self.img_w
            
            padded_categories[lane_idx] = old_categories[lane_idx]
            padded_attributes[lane_idx] = old_attributes[lane_idx]

        new_anno = {
            "label": lanes,
            "lane_endpoints": lanes_endpoints,
            "lane_vis_interpolated": lanes_vis_interpolated,
            "lanes": old_lanes,
            "padded_categories": padded_categories,
            "padded_attributes": padded_attributes
        }
        return new_anno

    def __call__(self, sample):
        img_org = sample["img"]
        img_h_curr, img_w_curr = img_org.shape[:2]
        
        line_strings_org = self.lane_to_linestrings(sample["lanes"])
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)

        dummy_mask_arr = np.zeros((img_h_curr, img_w_curr, 1), dtype=np.uint8)
        mask_org = SegmentationMapsOnImage(dummy_mask_arr, shape=img_org.shape)

        success = False
        
        for i in range(30):
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
                    "lane_attributes": sample.get("lane_attributes", [])
                }
                
                annos = self.transform_annotation(
                    new_anno_input, img_wh=(self.img_w, self.img_h)
                )
                
                label = annos["label"]
                lane_endpoints = annos["lane_endpoints"]
                lane_vis_interpolated = annos["lane_vis_interpolated"]
                
                if np.sum(label[:, 1] == 1) > 0:
                     success = True
                     break
                     
                if i > 10: 
                    success = True
                    break
                    
            except Exception:
                continue

        if not success:
            label = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + 1 + self.n_offsets), dtype=np.float32) * -1e5
            label[:, 0] = 1; label[:, 1] = 0
            lane_endpoints = np.zeros((self.max_lanes, 2))
            lane_vis_interpolated = np.zeros((self.max_lanes, self.n_offsets), dtype=np.float32)
            padded_categories = np.zeros(self.max_lanes, dtype=np.int64)
            padded_attributes = np.zeros(self.max_lanes, dtype=np.int64)
            new_anno_input = {"lanes": sample["lanes"]}
        else:
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

        if self.enable_3d and "lanes_3d" in sample:
            sample["lanes_3d"] = sample["lanes_3d"]
            if "intrinsic" in sample: sample["intrinsic"] = sample["intrinsic"]
            if "extrinsic" in sample: sample["extrinsic"] = sample["extrinsic"]

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