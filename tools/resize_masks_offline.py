import os
import cv2
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

MASK_SRC = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/mask"
MASK_DST = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/mask_resized_800_320"
TARGET_W, TARGET_H = 800, 320

def process_mask(mask_path):
    rel_path = os.path.relpath(mask_path, MASK_SRC)
    save_path = os.path.join(MASK_DST, rel_path)
    
    if os.path.exists(save_path):
        return True
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
        
    # mask 已经是 1010x1920（因为之前生成时切掉了270）
    # 所以不需要 crop，直接 resize
    # 使用 INTER_NEAREST 防止产生非0/1的插值碎点
    mask_resized = cv2.resize(mask, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(save_path, mask_resized)
    return True

def main():
    img_files = glob.glob(os.path.join(MASK_SRC, "**/*.png"), recursive=True)
    print(f"Found {len(img_files)} masks in {MASK_SRC}")
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        list(tqdm(executor.map(process_mask, img_files), total=len(img_files), desc="Resizing Masks"))
        
if __name__ == '__main__':
    main()
