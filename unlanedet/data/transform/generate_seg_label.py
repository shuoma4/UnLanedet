import cv2
import numpy as np
import pickle
import random
import os


def generate_lane_mask(image, lane_points_list, category_list, line_width=5):
    """
    Generate a lane line segmentation mask with Train IDs.

    Args:
        image (np.ndarray): Input image or image shape tuple (H, W, C) or (H, W).
        lane_points_list (list): List of lane points.
        category_list (list): List of category IDs.
        line_width (int): Width of the drawn lane lines.

    Returns:
        np.ndarray: The generated segmentation mask (H, W), dtype=uint8.
                    Values: 0 (Background), 1-24 (Lane Categories).
                    Mapping: Train ID = Category + 1 (assuming Category 0..21)
    """

    # Determine image shape
    if isinstance(image, tuple) or isinstance(image, list):
        if len(image) == 2:
            h, w = image
        else:
            h, w = image[:2]
    else:
        h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for points, category in zip(lane_points_list, category_list):
        if points is None or len(points) < 2:
            continue
        pts = np.array(points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=line_width)
    return mask


def main():
    pkl_path = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/openlane_lane3d_1000_train_cuth-270_800x320_cache_v1.pkl"
    vis_dir = "vis/seg"
    num_samples = 10

    if not os.path.exists(pkl_path):
        print(f"Error: pkl file not found at {pkl_path}")
        exit(1)

    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data_infos = pickle.load(f)

    print(f"Loaded {len(data_infos)} samples.")

    os.makedirs(vis_dir, exist_ok=True)

    indices = random.sample(range(len(data_infos)), min(num_samples, len(data_infos)))

    for i, idx in enumerate(indices):
        sample = data_infos[idx]
        img_path = sample.get("img_path", "")
        lanes = sample.get("lanes", [])
        categories = sample.get("lane_categories", [])

        print(f"[{i+1}/{len(indices)}] Processing sample {idx}, Img: {img_path}")

        # Try to read image to get shape, otherwise use default
        img = None
        if os.path.exists(img_path):
            img = cv2.imread(img_path)

        if img is not None:
            h, w = img.shape[:2]
            image_input = img
        else:
            # Fallback shape if image not found
            h, w = 320, 800
            image_input = (h, w)
            print(
                f"Warning: Image not found at {img_path}, using default shape {h}x{w}"
            )

        mask = generate_lane_mask(image_input, lanes, categories)

        filename = os.path.basename(img_path) if img_path else f"sample_{idx}.jpg"
        # Ensure filename has an extension
        if "." not in filename:
            filename += ".jpg"

        save_path = os.path.join(vis_dir, f"vis_{idx}_{filename}")

        if img is not None:
            # Resize mask to match image if needed
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(
                    mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
                )

            # Save mask
            cv2.imwrite(save_path, mask)

            # Save overlay
            overlay_path = os.path.join(vis_dir, f"overlay_{idx}_{filename}")
            # 将mask转换为3通道用于叠加
            mask_colored = cv2.applyColorMap(
                mask * 10, cv2.COLORMAP_JET
            )  # 放大以便观察
            vis_img = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
            cv2.imwrite(overlay_path, vis_img)
            print(f"Saved mask to {save_path} and overlay to {overlay_path}")
        else:
            cv2.imwrite(save_path, mask)
            print(f"Saved mask to {save_path}")


if __name__ == "__main__":
    main()
