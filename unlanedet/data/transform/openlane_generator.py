import logging

import numpy as np
from imgaug.augmentables.lines import LineStringsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from .generate_lane_line import GenerateLaneLine


class OpenLaneGenerator(GenerateLaneLine):
    """OpenLane transform based on `GenerateLaneLine` with metadata passthrough.

    This keeps the fCLRNet-compatible lane-line encoding, while additionally
    carrying OpenLane-specific fields such as `lane_categories`,
    `lane_attributes`, and `lane_track_ids` through augmentation, filtering,
    sorting, and padding.
    """

    def __init__(self, transforms=None, cfg=None, training=True):
        super().__init__(transforms=transforms, cfg=cfg, training=training)
        self.logger = logging.getLogger(__name__)
        self.num_lane_categories = int(getattr(cfg, 'num_lane_categories', 15))
        self.num_lr_attributes = int(getattr(cfg, 'num_lr_attributes', 5))

    def _normalize_track_id(self, value):
        if value is None:
            return -1
        try:
            return int(value)
        except Exception:
            return -1

    def transform_annotation(self, anno, img_wh=None):
        img_w, img_h = self.img_w, self.img_h

        old_lanes = anno.get('lanes', [])
        old_categories = anno.get('lane_categories', [0] * len(old_lanes))
        old_attributes = anno.get('lane_attributes', [0] * len(old_lanes))
        old_track_ids = anno.get('lane_track_ids', [-1] * len(old_lanes))

        lane_infos = []
        for idx, lane in enumerate(old_lanes):
            if len(lane) <= 1:
                continue
            category = int(old_categories[idx]) if idx < len(old_categories) else 0
            attribute = int(old_attributes[idx]) if idx < len(old_attributes) else 0
            track_id = self._normalize_track_id(old_track_ids[idx] if idx < len(old_track_ids) else -1)
            lane_infos.append(
                {
                    'lane': lane,
                    'category': category,
                    'attribute': attribute,
                    'track_id': track_id,
                }
            )

        for lane_info in lane_infos:
            lane_info['lane'] = sorted(lane_info['lane'], key=lambda x: -x[1])
            lane_info['lane'] = self.filter_lane(lane_info['lane'])
            lane_info['lane'] = [
                [x * self.img_w / float(img_w), y * self.img_h / float(img_h)] for x, y in lane_info['lane']
            ]

        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32) * -1e5
        lanes_endpoints = np.ones((self.max_lanes, 2), dtype=np.float32) * -1e5
        padded_categories = np.zeros((self.max_lanes,), dtype=np.int64)
        padded_attributes = np.zeros((self.max_lanes,), dtype=np.int64)
        padded_track_ids = np.full((self.max_lanes,), -1, dtype=np.int64)
        padded_vis = np.zeros((self.max_lanes, self.n_offsets), dtype=np.float32)

        lanes[:, 0] = 1
        lanes[:, 1] = 0
        kept_lanes = []

        for lane_idx, lane_info in enumerate(lane_infos):
            if lane_idx >= self.max_lanes:
                break

            lane = lane_info['lane']
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
            except AssertionError:
                continue

            if len(xs_inside_image) <= 1:
                continue

            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
            lanes[lane_idx, 3] = xs_inside_image[0]

            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = np.arctan(i * self.strip_size / (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / np.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)
            lanes[lane_idx, 4] = float(sum(thetas) / len(thetas))
            lanes[lane_idx, 5] = len(xs_inside_image)
            lanes[lane_idx, 6 : 6 + len(all_xs)] = all_xs
            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]

            category = lane_info['category']
            attribute = lane_info['attribute']
            if category < 0 or category >= self.num_lane_categories:
                category = 0
            if attribute < 0 or attribute >= self.num_lr_attributes:
                attribute = 0

            padded_categories[lane_idx] = category
            padded_attributes[lane_idx] = attribute
            padded_track_ids[lane_idx] = lane_info['track_id']
            padded_vis[lane_idx, :len(all_xs)] = 1.0
            kept_lanes.append(lane)

        return {
            'label': lanes,
            'old_anno': anno,
            'lane_endpoints': lanes_endpoints,
            'lane_categories': padded_categories,
            'lane_attributes': padded_attributes,
            'lane_track_ids': padded_track_ids,
            'lane_vis': padded_vis,
            'gt_points': kept_lanes,
        }

    def __call__(self, sample):
        img_org = sample['img']
        if 'cut_height' in sample:
            self.cfg.cut_height = sample['cut_height']
        if self.cfg.cut_height != 0:
            new_lanes = []
            for lane in sample['lanes']:
                new_lanes.append([(p[0], p[1] - self.cfg.cut_height) for p in lane])
            sample.update({'lanes': new_lanes})

        line_strings_org = self.lane_to_linestrings(sample['lanes'])
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)

        seg = None
        for i in range(30):
            if self.training:
                mask_org = SegmentationMapsOnImage(sample['mask'], shape=img_org.shape)
                img, line_strings, seg = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org,
                    segmentation_maps=mask_org,
                )
            else:
                img, line_strings = self.transform(image=img_org.copy().astype(np.uint8), line_strings=line_strings_org)

            line_strings.clip_out_of_image_()
            new_anno = {
                'lanes': self.linestrings_to_lanes(line_strings),
                'lane_categories': sample.get('lane_categories', []),
                'lane_attributes': sample.get('lane_attributes', []),
                'lane_track_ids': sample.get('lane_track_ids', []),
            }
            try:
                annos = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))
                label = annos['label']
                lane_endpoints = annos['lane_endpoints']
                break
            except Exception:
                if (i + 1) == 30:
                    self.logger.critical('Transform annotation failed 30 times :(')
                    raise

        sample['img'] = img.astype(np.float32) / 255.0
        sample['lane_line'] = label
        sample['lanes_endpoints'] = lane_endpoints
        sample['gt_points'] = annos['gt_points']
        sample['lane_categories'] = annos['lane_categories']
        sample['lane_attributes'] = annos['lane_attributes']
        sample['lane_track_ids'] = annos['lane_track_ids']
        sample['lane_vis'] = annos['lane_vis']
        sample['seg'] = seg.get_arr() if self.training else np.zeros(img_org.shape[:2], dtype=np.uint8)

        return sample
