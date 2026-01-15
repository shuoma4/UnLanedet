import os.path as osp
import numpy as np
import math
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage
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


class GenerateLaneLineOpenLane(object):
    """
    OpenLane专用的GenerateLaneLine类
    OpenLane数据集不使用分割mask,只使用车道线坐标标注
    """
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

        if transforms is None:
            transforms = CLRTransformsOpenLane(self.img_h, self.img_w)

        if transforms is not None:
            transforms = [convert_dictconfig_to_dict(aug) for aug in transforms]
            img_transforms = []
            for aug in transforms:
                p = aug['p']
                if aug['name'] != 'OneOf':
                    img_transforms.append(
                        iaa.Sometimes(p=p,
                                      then_list=getattr(
                                          iaa,
                                          aug['name'])(**aug['parameters'])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa,
                                        aug_['name'])(**aug_['parameters'])
                                for aug_ in aug['transforms']
                            ])))
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def sample_lane(self, points, sample_ys):
        """
        Sample lane coordinates at given y positions.
        Returns xs_outside_image and xs_inside_image.
        """
        points = np.array(points)
        if len(points) < 2:
            raise Exception('Annotaion points have to be sorted')

        x, y = points[:, 0], points[:, 1]

        # Interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1],
                                              x[::-1],
                                              k=min(3,
                                                    len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()

        # Sample points
        all_xs = interp(sample_ys)

        # Split into inside and outside image
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def linestrings_to_lanes(self, line_strings):
        lanes = []
        for line in line_strings:
            lanes.append(line.coords)

        return lanes

    def transform_annotation(self, anno, img_wh):
        """Transform annotation to CLRNet format."""
        img_w, img_h = self.img_w, self.img_h

        old_lanes = anno['lanes']
        if len(old_lanes) > self.max_lanes:
            old_lanes = old_lanes[:self.max_lanes]

        # Remove lanes with less than 2 points and sort by Y (bottom to top)
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]

        # Create annotations in CLRNet format: (max_lanes, 78)
        # 78 = 2 scores + 1 start_y + 1 start_x + 1 theta + 1 length + 72 coordinates
        lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + 1 + self.n_offsets),
                      dtype=np.float32) * -1e5
        lanes_endpoints = np.ones((self.max_lanes, 2))

        # Lanes are invalid by default
        lanes[:, 0] = 1  # negative score = 1 (invalid)
        lanes[:, 1] = 0  # positive score = 0 (invalid)

        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break

            try:
                xs_outside_image, xs_inside_image = self.sample_lane(
                    lane, self.offsets_ys)
            except Exception:
                continue

            if len(xs_inside_image) <= 1:
                continue

            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0  # negative score = 0
            lanes[lane_idx, 1] = 1  # positive score = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips  # start_y
            lanes[lane_idx, 3] = xs_inside_image[0]  # start_x

            # Calculate theta
            thetas = []
            for i in range(1, len(xs_inside_image)):
                theta = math.atan(
                    i * self.strip_size /
                    (xs_inside_image[i] - xs_inside_image[0] + 1e-5)) / math.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)

            theta_far = sum(thetas) / len(thetas) if thetas else 0.0
            lanes[lane_idx, 4] = theta_far
            lanes[lane_idx, 5] = len(xs_inside_image)  # length
            lanes[lane_idx, 6:6 + len(all_xs)] = all_xs
            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]

        new_anno = {
            'label': lanes,
            'lane_endpoints': lanes_endpoints
        }

        return new_anno

    def __call__(self, sample):
        """处理OpenLane数据样本,不使用分割mask"""
        import logging
        self.logger = logging.getLogger(__name__)

        img_org = sample['img']
        if 'cut_height' in sample.keys():
            self.cfg.cut_height = sample['cut_height']
        if self.cfg.cut_height != 0:
            new_lanes = []
            for i in sample['lanes']:
                lanes = []
                for p in i:
                    lanes.append((p[0], p[1] - self.cfg.cut_height))
                new_lanes.append(lanes)
            sample.update({'lanes': new_lanes})

        line_strings_org = self.lane_to_linestrings(sample['lanes'])
        line_strings_org = LineStringsOnImage(line_strings_org,
                                              shape=img_org.shape)

        # 不需要分割mask,只对图像和车道线进行变换
        for i in range(30):
            img, line_strings = self.transform(
                image=img_org.copy().astype(np.uint8),
                line_strings=line_strings_org)

            line_strings.clip_out_of_image_()
            new_anno = {'lanes': self.linestrings_to_lanes(line_strings)}
            try:
                annos = self.transform_annotation(new_anno,
                                                  img_wh=(self.img_w,
                                                          self.img_h))
                label = annos['label']
                lane_endpoints = annos['lane_endpoints']
                break
            except Exception as e:
                self.logger.error(f'Transform annotation failed (attempt {i+1}/30): {e}')
                import traceback
                self.logger.error(traceback.format_exc())
                if (i + 1) == 30:
                    self.logger.critical(
                        'Transform annotation failed 30 times :(')
                    exit()

        sample['img'] = img.astype(np.float32) / 255.
        sample['lane_line'] = label
        sample['lanes_endpoints'] = lane_endpoints
        sample['gt_points'] = new_anno['lanes']
        # OpenLane不使用分割标签,创建全0分割图(背景类别)
        sample['seg'] = np.zeros((self.img_h, self.img_w), dtype=np.int64)

        return sample


def CLRTransformsOpenLane(img_h, img_w):
    return [
        dict(name='Resize',
             parameters=dict(size=dict(height=img_h, width=img_w)),
             p=1.0),
        dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
        dict(name='Affine',
             parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                    y=(-0.1, 0.1)),
                             rotate=(-10, 10),
                             scale=(0.8, 1.2)),
             p=0.7),
        dict(name='Resize',
             parameters=dict(size=dict(height=img_h, width=img_w)),
             p=1.0),
    ]
