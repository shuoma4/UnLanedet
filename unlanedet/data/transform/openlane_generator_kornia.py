import logging
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F
from scipy.interpolate import InterpolatedUnivariateSpline


def get_kornia_transforms(cfg, training=True):
    if not training:
        return transforms.Compose(
            [
                transforms.Resize((cfg.img_h, cfg.img_w), antialias=True),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((cfg.img_h, cfg.img_w), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=(0.85, 1.15), contrast=(0.9, 1.1), saturation=0.1, hue=0.1
            ),
            transforms.RandomAffine(
                degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.2), fill=0
            ),
        ]
    )


class OpenLaneGeneratorKornia:
    """A replacement for OpenLaneTemporalGenerator using torchvision.transforms.v2"""

    def __init__(self, transforms_cfg=None, cfg=None, training=True):
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.training = training

        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)

        self.num_lane_categories = int(getattr(cfg, "num_lane_categories", 15))
        self.num_lr_attributes = int(getattr(cfg, "num_lr_attributes", 5))

        self.transform = get_kornia_transforms(cfg, training)

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])
        return filtered_lane

    def sample_lane(self, points, sample_ys):
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception("Annotaion points have to be sorted")
        x, y = points[:, 0], points[:, 1]

        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(
            y[::-1], x[::-1], k=min(3, len(points) - 1)
        )
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[
            (sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)
        ]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)

        # Keep behavior aligned with GenerateLaneLine:
        # extrapolate to image bottom using two closest points and define "outside"
        # by x-range (not by y-domain), so start_y/length semantics stay consistent.
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def _normalize_track_id(self, value):
        if value is None:
            return -1
        try:
            return int(value)
        except Exception:
            return -1

    def transform_annotation(self, anno, img_wh=None):
        img_w, img_h = self.img_w, self.img_h

        old_lanes = anno.get("lanes", [])
        old_categories = anno.get("lane_categories", [0] * len(old_lanes))
        old_attributes = anno.get("lane_attributes", [0] * len(old_lanes))
        old_track_ids = anno.get("lane_track_ids", [-1] * len(old_lanes))

        lane_infos = []
        for idx, lane in enumerate(old_lanes):
            if len(lane) <= 1:
                continue
            category = int(old_categories[idx]) if idx < len(old_categories) else 0
            attribute = int(old_attributes[idx]) if idx < len(old_attributes) else 0
            track_id = self._normalize_track_id(
                old_track_ids[idx] if idx < len(old_track_ids) else -1
            )
            lane_infos.append(
                {
                    "lane": lane,
                    "category": category,
                    "attribute": attribute,
                    "track_id": track_id,
                }
            )

        for lane_info in lane_infos:
            lane_info["lane"] = sorted(lane_info["lane"], key=lambda x: -x[1])
            lane_info["lane"] = self.filter_lane(lane_info["lane"])

        lanes = (
            np.ones((self.max_lanes, 2 + 1 + 1 + 2 + self.n_offsets), dtype=np.float32)
            * -1e5
        )
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

            lane = lane_info["lane"]
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(
                    lane, self.offsets_ys
                )
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
                # Use np.clip to prevent zero division or extreme tangent values in head later
                dx = np.clip(
                    xs_inside_image[i] - xs_inside_image[0], -self.img_w, self.img_w
                )
                if abs(dx) < 1e-5:
                    dx = 1e-5 if dx >= 0 else -1e-5
                theta = np.arctan(i * self.strip_size / dx) / np.pi
                theta = theta if theta > 0 else 1 - abs(theta)
                thetas.append(theta)
            lanes[lane_idx, 4] = float(sum(thetas) / len(thetas))
            lanes[lane_idx, 5] = len(xs_inside_image)
            lanes[lane_idx, 6 : 6 + len(all_xs)] = all_xs
            lanes_endpoints[lane_idx, 0] = (len(all_xs) - 1) / self.n_strips
            lanes_endpoints[lane_idx, 1] = xs_inside_image[-1]

            category = lane_info["category"]
            attribute = lane_info["attribute"]
            if category < 0 or category >= self.num_lane_categories:
                category = 0
            if attribute < 0 or attribute >= self.num_lr_attributes:
                attribute = 0

            padded_categories[lane_idx] = category
            padded_attributes[lane_idx] = attribute
            padded_track_ids[lane_idx] = lane_info["track_id"]
            padded_vis[lane_idx, : len(all_xs)] = 1.0
            kept_lanes.append(lane)

        return {
            "label": lanes,
            "old_anno": anno,
            "lane_endpoints": lanes_endpoints,
            "lane_categories": padded_categories,
            "lane_attributes": padded_attributes,
            "lane_track_ids": padded_track_ids,
            "lane_vis": padded_vis,
            "gt_points": kept_lanes,
        }

    def __call__(self, clip_data):
        if not isinstance(clip_data, list):
            return self._process_single(clip_data)

        T = len(clip_data)
        transformed_frames = []
        for i in range(T):
            transformed_frames.append(self._process_single(clip_data[i]))

        return self._stack_clip(transformed_frames)

    def _stack_clip(self, minibatches):
        keys = minibatches[0].keys()
        batched_dict = {}
        last_frame = minibatches[-1]

        stackable_keys = [
            "img",
            "extrinsic",
            "intrinsic",
            "pose",
            "visibility",
            "xyz",
            "track_id",
        ]

        for k in keys:
            if k in stackable_keys:
                try:
                    batched_items = [b[k] for b in minibatches]
                    batched_dict[k] = (
                        np.stack(batched_items, axis=0)
                        if isinstance(batched_items[0], np.ndarray)
                        else batched_items
                    )
                except Exception:
                    batched_dict[k] = last_frame[k]
            else:
                batched_dict[k] = last_frame[k]

        # Handle some nested lists properly
        for k in ["visibility", "xyz", "track_id"]:
            if k in keys:
                batched_items = [b[k] for b in minibatches]
                batched_dict[f"seq_{k}"] = batched_items

        return batched_dict

    def _process_single(self, sample):
        img_org = sample["img"]  # numpy array (H, W, 3)
        cut_height = sample.get("cut_height", getattr(self.cfg, "cut_height", 0))

        # Crop height first
        if cut_height != 0:
            new_lanes = []
            for lane in sample["lanes"]:
                new_lanes.append([(p[0], p[1] - cut_height) for p in lane])
            sample.update({"lanes": new_lanes})

        # Convert to tv_tensors
        from torchvision import tv_tensors

        # To Torch, float32, channels first.
        # Keep image in [0, 1] for torchvision ColorJitter/affine numerical stability.
        img_tensor = torch.from_numpy(img_org).permute(2, 0, 1).float() / 255.0
        h_orig, w_orig = img_tensor.shape[1], img_tensor.shape[2]
        img_tv = tv_tensors.Image(img_tensor)

        pts_list = []
        pts_lens = []
        for lane in sample["lanes"]:
            pts_lens.append(len(lane))
            for p in lane:
                # Add tiny jitter to box size to avoid degenerate boxes which cause NaNs in torchvision
                pts_list.append([p[0], p[1], p[0] + 0.01, p[1] + 0.01])

        # Ensure we only use torch and properly detach/numpy for points mapping, while protecting against gradients tracking if any
        with torch.no_grad():
            if len(pts_list) > 0:
                boxes_tv = tv_tensors.BoundingBoxes(
                    pts_list,
                    format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=img_tensor.shape[1:],
                )

                # Temporary fix for NaNs in torchvision.transforms.v2 Affine with small bounding boxes:
                # Sometime extreme rotations throw NaNs on coordinate limits.
                # We explicitly check the boxes_out tensor for NaNs.
                if self.training and "mask" in sample:
                    mask_tv = tv_tensors.Mask(
                        torch.from_numpy(sample["mask"]).unsqueeze(0)
                    )
                    # Apply joint transform
                    img_out, boxes_out, mask_out = self.transform(
                        img_tv, boxes_tv, mask_tv
                    )
                    seg_out = mask_out.squeeze(0).numpy()
                else:
                    img_out, boxes_out = self.transform(img_tv, boxes_tv)
                    seg_out = None

                # Extract points back
                out_pts = boxes_out.numpy()[:, :2]  # (N, 2)

                out_lanes = []
                idx = 0
                for l in pts_lens:
                    lane_pts = out_pts[idx : idx + l]

                    # Check for NaNs and transformed-image bounds.
                    # IMPORTANT: boxes_out is in transformed canvas coordinates (img_h/img_w),
                    # not in the original 1920x1280 canvas.
                    valid_pts = []
                    h_out, w_out = self.img_h, self.img_w
                    for pt in lane_pts:
                        x, y = pt[0], pt[1]
                        if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                            continue
                        # Keep a small margin in the transformed image to avoid boundary artifacts.
                        if 2 <= x < w_out - 2 and 2 <= y < h_out - 2:
                            valid_pts.append([float(x), float(y)])

                    if len(valid_pts) > 1:
                        # Sort to ensure monotonic y for interpolation later
                        valid_pts = sorted(valid_pts, key=lambda p: -p[1])
                        out_lanes.append(valid_pts)
                    idx += l
            else:
                if self.training and "mask" in sample:
                    mask_tv = tv_tensors.Mask(
                        torch.from_numpy(sample["mask"]).unsqueeze(0)
                    )
                    img_out, mask_out = self.transform(img_tv, mask_tv)
                    seg_out = mask_out.squeeze(0).numpy()
                else:
                    img_out = self.transform(img_tv)
                    seg_out = None
                out_lanes = []

        new_anno = {
            "lanes": out_lanes,
            "lane_categories": sample.get("lane_categories", []),
            "lane_attributes": sample.get("lane_attributes", []),
            "lane_track_ids": sample.get("lane_track_ids", []),
        }

        # Handle annotations
        try:
            annos = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))
        except Exception as e:
            self.logger.critical(f"Transform annotation failed: {e}")
            # fallback to empty
            annos = self.transform_annotation(
                {"lanes": []}, img_wh=(self.img_w, self.img_h)
            )

        # Format final output
        # img needs to be float32 in [0, 1]; additionally sanitize any NaN/Inf from aggressive aug ops.
        img_np = img_out.numpy().transpose(1, 2, 0)
        img_np = np.nan_to_num(img_np, nan=0.0, posinf=1.0, neginf=0.0)
        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        sample["img"] = img_np  # back to HWC, float32
        if seg_out is not None:
            sample["seg"] = seg_out

        sample["lane_line"] = annos["label"]
        sample["lane_categories"] = annos["lane_categories"]
        sample["lane_attributes"] = annos["lane_attributes"]
        sample["lane_track_ids"] = annos["lane_track_ids"]
        # Keep a stable key for temporal geo loss path (expects seq_track_id built from track_id)
        sample["track_id"] = annos["lane_track_ids"].tolist()
        sample["lane_vis"] = annos["lane_vis"]

        return sample
