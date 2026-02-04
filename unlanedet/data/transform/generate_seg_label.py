import cv2
import numpy as np


def generate_lane_mask(image, lane_points_list, category_list, line_width=5):
    """
    Generate a lane line segmentation mask with different colors for different categories.

    Args:
        image (np.ndarray): Input image or image shape tuple (H, W, C) or (H, W).
        lane_points_list (list): List of lane points, where each element is a list of (x, y) tuples or a numpy array of shape (N, 2).
        category_list (list): List of category IDs corresponding to each lane.
        line_width (int): Width of the drawn lane lines.

    Returns:
        np.ndarray: The generated segmentation mask (uint8).
                    If visualization is intended, it returns an RGB image (H, W, 3).
    """

    # Determine image shape
    if isinstance(image, tuple) or isinstance(image, list):
        if len(image) == 2:
            h, w = image
        else:
            h, w = image[:2]
    else:
        h, w = image.shape[:2]

    # Initialize a black mask (H, W, 3) for colored segmentation
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Define color map for OpenLane categories (B, G, R)
    # 0: 'unknown'
    # 1: 'white-dash'
    # 2: 'white-solid'
    # 3: 'double-white-dash'
    # 4: 'double-white-solid'
    # 5: 'white-ldash-rsolid'
    # 6: 'white-lsolid-rdash'
    # 7: 'yellow-dash'
    # 8: 'yellow-solid'
    # 9: 'double-yellow-dash'
    # 10: 'double-yellow-solid'
    # 11: 'yellow-ldash-rsolid'
    # 12: 'yellow-lsolid-rdash'
    # 20: 'left-curbside'
    # 21: 'right-curbside'

    category_colors = {
        0: (128, 128, 128),  # Gray
        1: (255, 0, 0),  # Blue
        2: (0, 255, 0),  # Green
        3: (0, 0, 255),  # Red
        4: (255, 255, 0),  # Cyan
        5: (255, 0, 255),  # Magenta
        6: (0, 255, 255),  # Yellow
        7: (128, 0, 0),  # Dark Blue
        8: (0, 128, 0),  # Dark Green
        9: (0, 0, 128),  # Dark Red
        10: (128, 128, 0),  # Teal
        11: (128, 0, 128),  # Purple
        12: (0, 128, 128),  # Olive
        20: (255, 128, 0),  # Orange (Curbside)
        21: (0, 128, 255),  # Light Orange/Brown (Curbside)
    }

    # Fallback color for undefined categories
    default_color = (255, 255, 255)

    for points, category in zip(lane_points_list, category_list):
        if points is None or len(points) < 2:
            continue

        # Ensure points are in the correct format for cv2.polylines
        # cv2.polylines expects a list of numpy arrays with shape (N, 1, 2) and type int32
        pts = np.array(points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))

        color = category_colors.get(category, default_color)

        # Draw the polyline
        # isClosed=False because lane lines are usually open curves
        cv2.polylines(mask, [pts], isClosed=False, color=color, thickness=line_width)

    return mask


import argparse
import pickle
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize lane segmentation labels from pkl file"
    )
    parser.add_argument("pkl_path", help="Path to the dataset cache pkl file")
    parser.add_argument(
        "--vis_dir", default="vis/seg", help="Directory to save visualization results"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to visualize"
    )
    args = parser.parse_args()

    if not os.path.exists(args.pkl_path):
        print(f"Error: pkl file not found at {args.pkl_path}")
        exit(1)

    print(f"Loading {args.pkl_path}...")
    with open(args.pkl_path, "rb") as f:
        data_infos = pickle.load(f)

    print(f"Loaded {len(data_infos)} samples.")

    os.makedirs(args.vis_dir, exist_ok=True)

    indices = random.sample(
        range(len(data_infos)), min(args.num_samples, len(data_infos))
    )

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

        mask = generate_lane_mask(image_input, lanes, categories, line_width=10)

        filename = os.path.basename(img_path) if img_path else f"sample_{idx}.jpg"
        # Ensure filename has an extension
        if "." not in filename:
            filename += ".jpg"

        save_path = os.path.join(args.vis_dir, f"vis_{idx}_{filename}")

        if img is not None:
            # Resize mask to match image if needed
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(
                    mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
                )

            # Save mask
            cv2.imwrite(save_path, mask)

            # Save overlay
            overlay_path = os.path.join(args.vis_dir, f"overlay_{idx}_{filename}")
            vis_img = cv2.addWeighted(img, 0.7, mask, 0.3, 0)
            cv2.imwrite(overlay_path, vis_img)
            print(f"Saved mask to {save_path} and overlay to {overlay_path}")
        else:
            cv2.imwrite(save_path, mask)
            print(f"Saved mask to {save_path}")
