import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# ======================
# 配置（按你的cfg改）
# ======================
IMG_W = 800
IMG_H = 320
ORI_IMG = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)  # 或换成真实图像路径
# ORI_IMG = cv2.imread("test.jpg")

NUM_POINTS = 72
N_STRIPS = NUM_POINTS - 1
PRIOR_YS = np.linspace(1, 0, NUM_POINTS, dtype=np.float32)

# ======================
# 加载 npz
# ======================
data = np.load("source/clrnet/priors.npz")
print(data.files)
priors = np.load("source/clrnet/priors.npz")["priors"]  # (num_priors, 2+2+2+n_offsets)
priors_on_featmap = np.load("source/clrnet/priors_on_featmap.npz")["priors_on_featmap"]

print("priors shape:", priors.shape)
print("priors_on_featmap shape:", priors_on_featmap.shape)

# ======================
# 1. 可视化车道线先验
# ======================
img_vis = ORI_IMG.copy()

for i in range(priors.shape[0]):
    lane = priors[i]
    xs_norm = lane[6:]  # 归一化 [0,1]
    xs = xs_norm * (IMG_W - 1)
    ys = PRIOR_YS * IMG_H

    pts = []
    for x, y in zip(xs, ys):
        if 0 <= x < IMG_W:
            pts.append((int(x), int(y)))

    for j in range(len(pts) - 1):
        cv2.line(img_vis, pts[j], pts[j + 1], (0, 255, 0), 1)

# 保存图像
cv2.imwrite("source/clrnet/vis_priors_lanes.png", img_vis)
print("Saved: vis_priors_lanes.png")

# ======================
# 2. 可视化特征图上的先验采样点
# ======================
# 设特征图大小 (按你的 backbone 输出)
FEAT_W = 100  # 例如 img_w/8
FEAT_H = 40  # 例如 img_h/8

feat_vis = np.zeros((FEAT_H, FEAT_W, 3), dtype=np.uint8)

for i in range(priors_on_featmap.shape[0]):
    xs_norm = priors_on_featmap[i]  # [0,1] normalized
    xs = xs_norm * (FEAT_W - 1)
    ys = np.linspace(0, FEAT_H - 1, xs.shape[0])

    for x, y in zip(xs, ys):
        if 0 <= x < FEAT_W:
            cv2.circle(feat_vis, (int(x), int(y)), 1, (255, 255, 255), -1)

cv2.imwrite(
    "source/clrnet/vis_priors_featmap.png",
    cv2.resize(feat_vis, (FEAT_W * 4, FEAT_H * 4)),
)
print("Saved: vis_priors_featmap.png")
