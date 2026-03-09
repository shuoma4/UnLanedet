import cv2


class BGR2RGB(object):
    def __init__(self, cfg=None):
        pass

    def __call__(self, sample):
        if 'img' in sample:
            sample['img'] = cv2.cvtColor(sample['img'], cv2.COLOR_BGR2RGB)
        return sample
