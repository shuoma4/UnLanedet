import os

import cv2
import numpy as np
import torch


def visualize_training_progress(
    current_iter,
    batch,
    targets_list,
    batch_lane_categories,
    batch_lane_attributes,
    pos_preds,
    batch_idx,
    prior_idx,
    outputs,
    ious,
    loss_y,
    loss_x,
    loss_theta,
    loss_len,
    target_idx,
    img_h,
    img_w,
    prior_ys,
    enable_category=True,
    enable_attribute=True,
    save_dir='/data1/lxy_log/workspace/ms/UnLanedet/debug_vis',
):
    """
    Visualize training progress including ground truth and matched predictions.

    Args:
        current_iter (int): Current iteration number.
        batch (dict): Batch data containing input images.
        targets_list (list): List of ground truth targets.
        batch_lane_categories (Tensor): Ground truth lane categories.
        batch_lane_attributes (Tensor): Ground truth lane attributes.
        pos_preds (Tensor): Positive predictions (matched).
        batch_idx (Tensor): Batch indices for positive predictions.
        prior_idx (Tensor): Prior indices for positive predictions.
        outputs (dict): Model outputs containing category/attribute logits.
        ious (Tensor): IoU values for matched predictions.
        loss_y (Tensor): Y-coordinate regression loss.
        loss_x (Tensor): X-coordinate regression loss.
        loss_theta (Tensor): Theta regression loss.
        loss_len (Tensor): Length regression loss.
        target_idx (Tensor): Target indices for matched predictions.
        img_h (int): Image height.
        img_w (int): Image width.
        prior_ys (Tensor): Y-coordinates of priors (normalized).
        enable_category (bool): Whether category prediction is enabled.
        enable_attribute (bool): Whether attribute prediction is enabled.
        save_dir (str): Directory to save visualization results.
    """
    os.makedirs(save_dir, exist_ok=True)
    vis_txt_path = os.path.join(save_dir, f'iter_{current_iter}.txt')

    # 1. 图像预处理
    pad = 100
    device = batch['img'].device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    img_tensor = batch['img'][0]
    img_tensor = img_tensor * std[0] + mean[0]
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_vis = cv2.copyMakeBorder(
        img_vis,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    # 2. 绘制 GT (绿色)
    # 优先使用 batch['sample_xs'] (Absolute Pixel Coordinates) 如果可用
    gt_xs_pixel = None
    if 'sample_xs' in batch:
        gt_xs_pixel = batch['sample_xs'][0]  # (max_lanes, N_points)

    if len(targets_list) > 0:
        gt_lanes = targets_list[0]
        gt_cats = batch_lane_categories[0] if batch_lane_categories is not None else None
        gt_attrs = batch_lane_attributes[0] if batch_lane_attributes is not None else None

        for i_gt, lane in enumerate(gt_lanes):
            if lane[1] == 1:  # Valid lane
                points = []

                # 如果有 sample_xs (绝对坐标)，直接使用
                if gt_xs_pixel is not None and i_gt < len(gt_xs_pixel):
                    pts_x = gt_xs_pixel[i_gt]

                    # 额外过滤无效值: OpenLane 可能使用 -1e5 或 -1 作为无效值
                    # 同时需要过滤掉 x <= 0 的点 (图像左边界外)
                    # 以及 x >= img_w 的点 (图像右边界外)

                    for idx_pt, x in enumerate(pts_x):
                        # x is absolute pixel coordinate
                        # Invalid values are typically negative or very large negative
                        if x > 0 and x < img_w:
                            x_pixel = int(x) + pad
                            y_norm = prior_ys[idx_pt].item()
                            y = int((img_h - 1) * y_norm) + pad
                            points.append((x_pixel, y))
                else:
                    # Fallback: Try to use lane[6:] (Offsets or Normalized X) - Unreliable if offsets
                    # Assuming lane[6:] are normalized X for fallback (which they are NOT in LLANet)
                    pass

                if len(points) > 1:
                    cv2.polylines(
                        img_vis,
                        [np.array(points)],
                        False,
                        (0, 255, 0),
                        2,
                    )
                    start_pt = points[0]
                    gt_c = gt_cats[i_gt].item() if gt_cats is not None else -1
                    gt_a = gt_attrs[i_gt].item() if gt_attrs is not None else -1
                    cv2.putText(
                        img_vis,
                        f'GT_{i_gt}|C{gt_c}|A{gt_a}',
                        (start_pt[0], start_pt[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

    # 3. 绘制 Matched Preds & 写 Log
    b0_mask = batch_idx == 0
    with open(vis_txt_path, 'w') as f:
        f.write(f'Iteration {current_iter} - Batch 0 Analysis\n')
        f.write('=' * 60 + '\n')

        if b0_mask.sum() > 0:
            b0_pos_preds = pos_preds[b0_mask]
            b0_priors_idx = prior_idx[b0_mask]

            b0_cat_logits = (
                outputs['category'][0][b0_priors_idx] if (enable_category and 'category' in outputs) else None
            )
            b0_attr_logits = (
                outputs['attribute'][0][b0_priors_idx] if (enable_attribute and 'attribute' in outputs) else None
            )

            b0_iou = ious[b0_mask]
            b0_ly = loss_y[b0_mask]
            b0_lx = loss_x[b0_mask]
            b0_lt = loss_theta[b0_mask]
            b0_ll = loss_len[b0_mask]

            b0_tgt_idx = target_idx[b0_mask]

            for k in range(len(b0_pos_preds)):
                lane = b0_pos_preds[k].detach().cpu().numpy()
                score = lane[1]
                prob = 1 / (1 + np.exp(-score))

                cat_id = torch.argmax(b0_cat_logits[k]).item() if b0_cat_logits is not None else -1
                attr_id = torch.argmax(b0_attr_logits[k]).item() if b0_attr_logits is not None else -1

                color = (
                    np.random.randint(50, 255),
                    np.random.randint(50, 100),
                    np.random.randint(100, 255),
                )

                points_x = lane[6:]

                # Reconstruct Prediction using Prior + Delta
                start_y = lane[2].item()
                start_x = lane[3].item()
                theta = lane[4].item()
                # length = lane[5].item()
                deltas = lane[6:]

                # Denormalize Prior
                start_y_pixel = start_y * (img_h - 1)
                start_x_pixel = start_x * (img_w - 1)

                # Theta: [-1, 1] -> [-90, 90] deg -> rad
                theta_deg = theta * 90
                theta_rad = np.deg2rad(theta_deg)

                points = []
                for idx_pt, delta_norm in enumerate(deltas):
                    y_norm = prior_ys[idx_pt].item()
                    y_pixel = y_norm * (img_h - 1)

                    # Prior X
                    prior_x = start_x_pixel + (start_y_pixel - y_pixel) * np.tan(theta_rad)

                    # Delta X
                    delta_pixel = delta_norm.item() * (img_w - 1)

                    x_final = prior_x + delta_pixel

                    # Filter invalid points for PREDICTION as well
                    if x_final > 0 and x_final < img_w:
                        # Additionally check if y is within reasonable bounds (0, H)
                        if y_pixel >= 0 and y_pixel < img_h:
                            x_vis = int(x_final) + pad
                            y_vis = int(y_pixel) + pad
                            points.append((x_vis, y_vis))

                if len(points) > 1:
                    cv2.polylines(img_vis, [np.array(points)], False, color, 2)
                    start_pt = points[0]
                    cv2.circle(img_vis, start_pt, 4, color, -1)
                    info = f'P{k}|{prob:.2f}|C{cat_id}'
                    cv2.putText(
                        img_vis,
                        info,
                        (start_pt[0] + 10, start_pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        2,
                    )
                    cv2.putText(
                        img_vis,
                        info,
                        (start_pt[0] + 10, start_pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

                gt_id_val = b0_tgt_idx[k].item()
                f.write(f'Pred #{k} (Matched GT_{gt_id_val}) | Conf: {prob:.4f} | Cat: {cat_id} | Attr: {attr_id}\n')
                f.write(f'  IoU: {b0_iou[k]:.4f}\n')
                f.write(
                    f'  Reg Loss -> Y: {b0_ly[k]:.2f}, X: {b0_lx[k]:.2f}, Theta: {b0_lt[k]:.2f}, Len: {b0_ll[k]:.2f}\n'
                )
                f.write('-' * 40 + '\n')
        else:
            f.write('No matched predictions in Batch 0.\n')

    # 保存图片
    vis_img_path = os.path.join(save_dir, f'iter_{current_iter}.jpg')
    cv2.imwrite(vis_img_path, img_vis)
