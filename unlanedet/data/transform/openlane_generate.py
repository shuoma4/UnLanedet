import logging

import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from omegaconf import DictConfig

from unlanedet.config import instantiate


def convert_dictconfig_to_dict(config):
    if isinstance(config, DictConfig):
        new_dict = {}
        for key, value in config.items():
            new_dict[key] = convert_dictconfig_to_dict(value)
        return new_dict
    else:
        return config


class OpenLaneGenerate(object):
    """
    数据增强 + GT 结构化编码
    """

    def __init__(self, transforms=None, cfg=None, training=True):
        self.transform_dict = transforms
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.max_lanes = cfg.max_lanes
        self.training = training
        self.mean = np.array(cfg.img_norm['mean'], dtype=np.float32)
        self.std = np.array(cfg.img_norm['std'], dtype=np.float32)
        self.encoder = instantiate(cfg.encoder)
        self.transform = self._build_transform_pipeline(transforms)

    # ============================================================
    # Augmentation
    # ============================================================
    def _build_transform_pipeline(self, transforms):
        if transforms is None or not transforms:
            return iaa.Sequential([])

        transforms_dicts = []
        for aug in transforms:
            if aug is None:
                continue
            transforms_dicts.append(convert_dictconfig_to_dict(aug))

        img_transforms = []
        for aug_config in transforms_dicts:
            if not isinstance(aug_config, dict):
                continue
            p = aug_config.get('p', 1.0)

            if aug_config['name'] == 'OneOf':
                oneof_transforms = []
                for sub_aug in aug_config.get('transforms', []):
                    try:
                        transform_cls = getattr(iaa, sub_aug['name'])
                        transform_instance = transform_cls(**sub_aug['parameters'])
                        oneof_transforms.append(transform_instance)
                    except Exception:
                        continue
                if oneof_transforms:
                    img_transforms.append(iaa.Sometimes(p, iaa.OneOf(oneof_transforms)))
            else:
                try:
                    transform_cls = getattr(iaa, aug_config['name'])
                    transform_instance = transform_cls(**aug_config.get('parameters', {}))
                    img_transforms.append(iaa.Sometimes(p, transform_instance))
                except Exception:
                    continue

        return iaa.Sequential(img_transforms)

    # ============================================================
    # Annotation Transform
    # ============================================================
    def lane_to_linestrings(self, lanes):
        return [LineString(np.array(lane)[:, :2]) for lane in lanes]

    def linestrings_to_lanes(self, line_strings):
        return [ls.coords for ls in line_strings]

    # ============================================================
    # GT 编码
    # ============================================================
    def transform_annotation(self, anno):
        old_lanes = anno['lanes']
        old_categories = anno.get('lane_categories', [0] * len(old_lanes))
        old_attributes = anno.get('lane_attributes', [0] * len(old_lanes))

        if len(old_lanes) > 0:
            combined = list(zip(old_lanes, old_categories, old_attributes))
            combined.sort(key=lambda x: -x[0][0][1])  # y 从大到小
            old_lanes, old_categories, old_attributes = zip(*combined)

        reg_dim = 6 + self.num_points
        lanes = np.ones((self.max_lanes, reg_dim), dtype=np.float32) * -1e5
        sample_xs = np.ones((self.max_lanes, self.num_points), dtype=np.float32) * -1e5
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        lane_endpoints = np.ones((self.max_lanes, 2), dtype=np.float32) * -1e5
        padded_categories = np.zeros(self.max_lanes, dtype=np.uint8)
        padded_attributes = np.zeros(self.max_lanes, dtype=np.uint8)

        for i, lane_pts in enumerate(old_lanes):
            if i >= self.max_lanes:
                break
            try:
                reg, end_pt, xs, ys = self.encoder.encode(lane_pts)
            except Exception as e:
                self.logger.warning(f'Encode lane failed: {e}')
                continue
            lanes[i, 0] = 0
            lanes[i, 1] = 1
            lanes[i, 2:] = reg
            sample_xs[i, :] = xs
            lane_endpoints[i, 0] = end_pt[0]
            lane_endpoints[i, 1] = end_pt[1]
            padded_categories[i] = old_categories[i]
            padded_attributes[i] = old_attributes[i]

        new_anno = {
            'lane_line': lanes,  # (max_lanes, 6 + N)
            'sample_xs': sample_xs,  # (max_lanes, N)
            'lane_endpoints': lane_endpoints,  # (max_lanes, 2)
            'padded_categories': padded_categories,
            'padded_attributes': padded_attributes,
            'gt_points': old_lanes,
        }
        return new_anno

    # ============================================================
    # Main Call
    # ============================================================
    def __call__(self, sample):
        img_org = sample['img']
        line_strings_org = self.lane_to_linestrings(sample['lanes'])
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        mask_org = SegmentationMapsOnImage(sample['mask'], shape=img_org.shape)

        cnt = 0
        while True:
            cnt += 1
            if cnt > 10:
                raise RuntimeError('Augmentation failed 10 times')

            try:
                if self.training:
                    img, line_strings, seg = self.transform(
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
                aug_lanes = self.linestrings_to_lanes(line_strings)

                new_anno_input = {
                    'lanes': aug_lanes,
                    'lane_categories': sample.get('lane_categories', []),
                    'lane_attributes': sample.get('lane_attributes', []),
                }

                annos = self.transform_annotation(new_anno_input)
                break
            except Exception:
                pass

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        sample['img'] = img
        sample['lane_line'] = annos['lane_line']
        sample['sample_xs'] = annos['sample_xs']
        sample['lane_endpoints'] = annos['lane_endpoints']
        sample['lane_categories'] = annos['padded_categories']
        sample['lane_attributes'] = annos['padded_attributes']
        sample['gt_points'] = annos['gt_points']
        sample['seg'] = seg.get_arr() if self.training else None

        return sample
