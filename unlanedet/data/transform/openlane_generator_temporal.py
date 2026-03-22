import numpy as np
import collections
from .openlane_generator import OpenLaneGenerator

class OpenLaneTemporalGenerator(OpenLaneGenerator):
    def __init__(self, transforms=None, cfg=None, training=True):
        super().__init__(transforms=transforms, cfg=cfg, training=training)
        
    def __call__(self, clip_data):
        if not isinstance(clip_data, list):
            # fallback to single frame generator if data isn't temporal
            return super().__call__(clip_data)
            
        T = len(clip_data)
        
        # Determine deterministic transformations to ensure consistent spatial warping across T frames
        seq_det = self.transform.to_deterministic() if self.training else self.transform

        transformed_frames = []
        for i in range(T):
            sample = clip_data[i]
            img_org = sample['img']
            
            if 'cut_height' in sample and sample['cut_height'] != 0:
                new_lanes = []
                for lane in sample['lanes']:
                    new_lanes.append([(p[0], p[1] - sample['cut_height']) for p in lane])
                sample['lanes'] = new_lanes

            line_strings_org = self.lane_to_linestrings(sample['lanes'])
            from imgaug.augmentables.segmaps import SegmentationMapsOnImage
            
            # Use seq_det
            if self.training:
                # Fallback if mask is somehow still None
                if sample.get('mask') is None:
                    print(f"WARNING: mask is None for sample {sample.get('img_path', 'unknown')}. Creating dummy mask.", flush=True)
                    sample['mask'] = np.zeros(img_org.shape[:2], dtype=np.uint8)
                
                mask_org = SegmentationMapsOnImage(sample['mask'], shape=img_org.shape)
                img, line_strings, seg = seq_det(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org,
                    segmentation_maps=mask_org
                )
            else:
                img, line_strings = seq_det(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org
                )

            # Generate target based on generic generator (cls, pts, etc..)
            out_sample = sample.copy()
            out_sample['img'] = img.astype(np.float32) / 255.0
            out_sample['lanes'] = [list(ls.coords) for ls in line_strings]
            out_sample['seg'] = seg.get_arr() if self.training else np.zeros(img_org.shape[:2], dtype=np.uint8)
            annos = self.transform_annotation(out_sample)
            out_sample.update(annos)
            out_sample['lane_line'] = annos['label']
            out_sample['lanes_endpoints'] = annos.get('lane_endpoints', None)
            gt_dict = out_sample
            
            # Need to retain physical tracking and pose structures

            gt_dict['extrinsic'] = sample['extrinsic']
            gt_dict['intrinsic'] = sample['intrinsic']
            gt_dict['visibility'] = sample.get('visibility', [])
            gt_dict['xyz'] = sample.get('xyz', [])
            gt_dict['track_id'] = sample.get('track_id', [])
            
            transformed_frames.append(gt_dict)
            
        batched_dict = self._stack_clip(transformed_frames)
        return batched_dict
        
    def _stack_clip(self, minibatches):
        keys = minibatches[0].keys()
        batched_dict = {}
        last_frame = minibatches[-1]
        for k in keys:
            if k == 'img':
                batched_items = [b[k] for b in minibatches]
                batched_dict[k] = np.stack(batched_items, axis=0)
            else:
                batched_dict[k] = last_frame[k]
        return batched_dict

    def generate_targets(self, sample):
        # A simple pass-through utilizing parents methods because we override __call__ entirely
        # OpenLaneGenerator (or GenerateLaneLine) populates standard things like max_lanes, etc.
        from imgaug.augmentables.lines import LineStringsOnImage
        sample['img'] = sample['img'].astype(np.float32) / 255.0
        # ... logic specific to converting self.lane_to_linestrings to reg format ...
        # (This avoids rewriting 100 lines of OpenLane points target parsing)
        
        # We manually construct a pseudo-dict that looks like __call__ output
        # For simplicity in this demo, parent relies heavily on the original dict.
        return sample

class TemporalToTensor(object):
    def __init__(self, keys=['img', 'mask'], collect_keys=[], cfg=None):
        self.keys = keys
        self.collect_keys = collect_keys

    def __call__(self, sample):
        from .transforms import to_tensor
        data = {}
        for key in sample.keys():
            if key in self.keys:
                if key == 'img' and len(sample['img'].shape) == 4:
                    # [T, H, W, C]
                    data[key] = to_tensor(sample[key])
                    # Permute to [T, C, H, W]
                    data[key] = data[key].permute(0, 3, 1, 2)
                else:
                    data[key] = to_tensor(sample[key])
            if key in self.collect_keys:
                data[key] = sample[key]
        return data

class TemporalNormalize(object):
    def __init__(self, img_norm, cfg=None):
        self.mean = np.array(img_norm['mean'], dtype=np.float32)
        self.std = np.array(img_norm['std'], dtype=np.float32)

    def __call__(self, sample):
        m = self.mean
        s = self.std
        img = sample['img'] # [T, C, H, W] or [T, H, W, C] depending on order
        
        # If it's applied before ToTensor, it's [T, H, W, C]
        if isinstance(img, np.ndarray):
            if len(m) == 1:
                img = (img - m) / s
            else:
                img = (img - m[np.newaxis, np.newaxis, np.newaxis, :]) / s[np.newaxis, np.newaxis, np.newaxis, :]
            sample['img'] = img
        else:
            # If after ToTensor, it's [T, C, H, W] torch tensor
            m_t = torch.tensor(m, device=img.device).view(1, len(m), 1, 1)
            s_t = torch.tensor(s, device=img.device).view(1, len(s), 1, 1)
            sample['img'] = (img - m_t) / s_t
            
        return sample
