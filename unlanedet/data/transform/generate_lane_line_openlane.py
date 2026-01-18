import os.path as osp
import numpy as np
import math
import imgaug.augmenters as iaa
import cv2

# === 【核心优化】强制关闭 OpenCV 多线程 ===
# 在 DataLoader 多进程模式下，OpenCV 的多线程会导致严重的 CPU 争抢和上下文切换
# 设置为 0 表示只使用当前线程，这是多进程数据加载的最佳实践
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# =======================================

from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline
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
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.cfg = cfg
        self.training = training

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
        return [LineString(lane) for lane in lanes]

    def linestrings_to_lanes(self, line_strings):
        return [line.coords for line in line_strings]

    def sample_lane(self, points, sample_ys):
        points = np.array(points)
        if len(points) < 2:
            raise Exception("Annotaion points have to be sorted")
        x, y = points[:, 0], points[:, 1]
        interp = InterpolatedUnivariateSpline(
            y[::-1], x[::-1], k=min(3, len(points) - 1)
        )
        all_xs = interp(sample_ys)
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]
        return xs_outside_image, xs_inside_image

    def transform_annotation(self, anno, img_wh):
        old_lanes = anno["lanes"]
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        lanes = (
            np.ones(
                (self.max_lanes, 2 + 1 + 1 + 1 + 1 + self.n_offsets), dtype=np.float32
            )
            * -1e5
        )
        lanes_endpoints = np.ones((self.max_lanes, 2))
        lanes[:, 0] = 1
        lanes[:, 1] = 0

        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(
                    lane, self.offsets_ys
                )
            except Exception:
                continue
            if len(xs_inside_image) <= 1:
                continue

            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0] / self.img_w

            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = (
                    math.atan(
                        i
                        * self.strip_size
                        / (xs_inside_image[i] - xs_inside_image[0] + 1e-5)
                    )
                    / math.pi
                )
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)
            theta_far = sum(thetas) / len(thetas) if thetas else 0.0

            lanes[lane_idx, 4] = theta_far
            lanes[lane_idx, 5] = len(xs_inside_image)
            lanes[lane_idx, 6 : 6 + len(all_xs)] = all_xs / self.img_w
            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1] / self.img_w

        new_anno = {"label": lanes, "lane_endpoints": lanes_endpoints}
        return new_anno

    def __call__(self, sample):
        img_org = sample["img"]
        img_h_curr, img_w_curr = img_org.shape[:2]
        is_preprocessed_img = img_h_curr == self.img_h
        global_cut_height = self.cfg.cut_height

        if not is_preprocessed_img and global_cut_height > 0:
            new_lanes = []
            for i in sample["lanes"]:
                lanes = []
                for p in i:
                    lanes.append((p[0], p[1] - global_cut_height))
                new_lanes.append(lanes)
            sample.update({"lanes": new_lanes})

        line_strings_org = self.lane_to_linestrings(sample["lanes"])
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)

        # Use a dummy mask because imgaug expects one if we pass segmentation_maps
        # but we handle segmentation manually later for speed
        dummy_mask_arr = np.zeros((img_h_curr, img_w_curr, 1), dtype=np.uint8)
        mask_org = SegmentationMapsOnImage(dummy_mask_arr, shape=img_org.shape)

        for i in range(30):
            try:
                if self.training:
                    # Pass dummy mask to keep imgaug happy if transforms require it
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
                line_strings.clip_out_of_image_()
                new_anno = {"lanes": self.linestrings_to_lanes(line_strings)}
                annos = self.transform_annotation(
                    new_anno, img_wh=(self.img_w, self.img_h)
                )
                label = annos["label"]
                lane_endpoints = annos["lane_endpoints"]
                break
            except Exception:
                if (i + 1) == 30:
                    exit()

        # === Normalization (ImageNet) ===
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        sample["img"] = img
        sample["lane_line"] = label
        sample["lanes_endpoints"] = lane_endpoints
        sample["gt_points"] = new_anno["lanes"]

        # === Optimized Seg Mask (Downsample 8x, uint8) ===
        # Reduce memory usage significantly
        mask_scale = 8
        mask_h, mask_w = int(self.img_h // mask_scale), int(self.img_w // mask_scale)
        seg_map = np.zeros((mask_h, mask_w), dtype=np.uint8)

        lanes_points = new_anno["lanes"]
        for lane in lanes_points:
            if len(lane) < 2:
                continue
            pts = np.array([lane], dtype=np.float32) / mask_scale
            pts = pts.astype(np.int32)
            cv2.polylines(seg_map, pts, isClosed=False, color=1, thickness=1)

        sample["seg"] = seg_map.astype(np.int64)
        # ===============================================

        return sample
