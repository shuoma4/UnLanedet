class LanePostProcessor(object):
    """
    任务逻辑层：置信度过滤、NMS、格式转换
    """

    def __init__(self, cfg):
        self.conf_threshold = getattr(cfg, 'conf_threshold', 0.4)

    def get_lanes(self, xs, ys, valid_mask, scores=None):
        """
        xs, ys, valid_mask: 来自 decoder 的 Tensor
        scores: (B, M) 置信度（可选）

        return:
            lanes_all: List[List[List[(x, y)]]]
        """

        if scores is not None and self.conf_threshold > 0:
            keep = scores >= self.conf_threshold
            valid_mask = valid_mask & keep.unsqueeze(-1)

        xs_np = xs.detach().cpu().numpy()
        ys_np = ys.detach().cpu().numpy()
        mask_np = valid_mask.detach().cpu().numpy()

        B, M, N = xs_np.shape
        lanes_all = []

        for b in range(B):
            lanes_b = []
            for m in range(M):
                mask = mask_np[b, m]
                if mask.sum() < 2:
                    lanes_b.append([])
                    continue
                pts = [(float(x), float(y)) for x, y in zip(xs_np[b, m][mask], ys_np[b, m][mask])]
                lanes_b.append(pts)
            lanes_all.append(lanes_b)

        return lanes_all
