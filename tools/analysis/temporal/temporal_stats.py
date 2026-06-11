import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from unlanedet.data.openlane_temporal import load_segment_pkl

def main():
    data_root = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/lane3d_1000"
    cut_height = 600
    # Let's check config/llanetv1/openlane1000/tsm/llanet_tsm_v3.py 
    # It says: cut_height=dataloader.train.dataset.cut_height, so 270 probably.
    # Actually llanet_tsm_v3 uses openlane1000 base config.
    
    # We can just read the cache directory
    split = "train"
    seq_len = 3
    # let's find the cache dir
    cache_dirs = [d for d in os.listdir(data_root) if d.startswith(f"openlane_temporal_cache_dir_{split}_{seq_len}")]
    if not cache_dirs:
        print("Cache dir not found. Searching for any cache dir...")
        cache_dirs = [d for d in os.listdir(data_root) if d.startswith(f"openlane_temporal_cache_dir_")]
        if not cache_dirs:
            print("No cache dir found at all.")
            return
    cache_dir = os.path.join(data_root, cache_dirs[0])
    print(f"Using cache dir: {cache_dir}")
    
    index_file = os.path.join(cache_dir, "index.pkl")
    with open(index_file, "rb") as f:
        index_data = pickle.load(f)
        
    clips = index_data['clips']
    segments = index_data['segments']
    
    # Stats
    total_error = 0.0
    total_points = 0
    
    lane_change_events = []
    
    # We can process segment by segment to avoid redundant loading
    for seg in tqdm(segments, desc="Processing segments"):
        seg_data = load_segment_pkl(cache_dir, seg)
        
        # We process pairs of consecutive frames in the segment
        for i in range(1, len(seg_data)):
            prev_frame = seg_data[i-1]
            curr_frame = seg_data[i]
            
            prev_tracks = set(prev_frame['lane_track_ids'])
            curr_tracks = set(curr_frame['lane_track_ids'])
            
            # Remove -1 (untracked)
            prev_tracks.discard(-1)
            curr_tracks.discard(-1)
            
            # 2. Count lane changes
            if len(prev_tracks) != len(curr_tracks) or prev_tracks != curr_tracks:
                appeared = curr_tracks - prev_tracks
                disappeared = prev_tracks - curr_tracks
                event = {
                    'segment': seg,
                    'frame_idx': i,
                    'appeared': list(appeared),
                    'disappeared': list(disappeared)
                }
                lane_change_events.append(event)
                
            # 1. Calculate projection error for tracked lanes
            common_tracks = prev_tracks.intersection(curr_tracks)
            if not common_tracks:
                continue
                
            E_t1 = prev_frame['extrinsic']
            E_t = curr_frame['extrinsic']
            if len(E_t1) == 0 or len(E_t) == 0:
                continue
                
            E_t1 = np.array(E_t1)
            E_t = np.array(E_t)
            K_t = np.array(curr_frame['intrinsic'])
            
            # Read pose from json
            import json
            try:
                json_t1 = prev_frame['img_path'].replace('.jpg', '.json').replace('training_resized_800_320', 'training').replace('training_cut_600_resized_800_320', 'training').replace('images', 'training')
                # Actually img_path might be absolute path to jpg. Let's reconstruct json path
                # The original img_rel_path is in frame_info
                img_rel_path = prev_frame.get('img_name', prev_frame['img_path'])
                # If it's absolute, try to find 'training' or 'validation'
                if 'training/' in img_rel_path:
                    json_t1 = img_rel_path.replace('.jpg', '.json')
                else:
                    json_t1 = prev_frame['img_path'].replace('.jpg', '.json')
                    
                if not os.path.exists(json_t1):
                    # try to map from image path to json path
                    parts = prev_frame['img_path'].split('images/')
                    if len(parts) > 1:
                        json_t1 = parts[0] + 'training/' + parts[1].replace('.jpg', '.json')
                    else:
                        # Let's use data_root
                        seg_name = prev_frame['img_path'].split('/')[-2]
                        img_name = prev_frame['img_path'].split('/')[-1].replace('.jpg', '.json')
                        json_t1 = os.path.join(data_root, 'training', seg_name, img_name)
                
                with open(json_t1, 'r') as f:
                    pose_t1 = np.array(json.load(f)['pose'])
                    
                json_t = curr_frame['img_path'].replace('.jpg', '.json')
                if not os.path.exists(json_t):
                    seg_name = curr_frame['img_path'].split('/')[-2]
                    img_name = curr_frame['img_path'].split('/')[-1].replace('.jpg', '.json')
                    json_t = os.path.join(data_root, 'training', seg_name, img_name)
                    
                with open(json_t, 'r') as f:
                    pose_t = np.array(json.load(f)['pose'])
            except Exception as e:
                continue
            
            # Correct T_rel = inv(Ext) @ inv(Pose_t) @ Pose_t1 @ Ext
            # Ext maps camera to ego. Pose maps ego to world.
            # So P_world = Pose @ Ext @ P_cam
            try:
                # E_t1 is extrinsic (Camera to Ego)
                E_t1_cam2ego = np.array(E_t1)
                E_t_cam2ego = np.array(E_t)
                
                # Transform from cam_t1 to cam_t:
                # cam_t1 -> ego_t1 -> world -> ego_t -> cam_t
                T_rel = np.linalg.inv(E_t_cam2ego) @ np.linalg.inv(pose_t) @ pose_t1 @ E_t1_cam2ego
            except np.linalg.LinAlgError:
                continue
                
            for track_id in common_tracks:
                prev_idx = prev_frame['lane_track_ids'].index(track_id)
                curr_idx = curr_frame['lane_track_ids'].index(track_id)
                
                xyz_t1 = prev_frame['xyz'][prev_idx]
                if len(xyz_t1) == 0:
                    continue
                    
                # Project xyz_t1 to current frame
                pts_t1 = np.array(xyz_t1).T # 3 x N
                
                # Filter out invalid 3D points (e.g. all 0s)
                valid_3d = np.linalg.norm(pts_t1, axis=0) > 1e-3
                if not np.any(valid_3d):
                    continue
                    
                pts_t1 = pts_t1[:, valid_3d]
                pts_t1_h = np.vstack((pts_t1, np.ones((1, pts_t1.shape[1])))) # 4 x N
                
                pts_t = T_rel @ pts_t1_h # 4 x N
                
                # In OpenLane, xyz is in Camera coordinates but with X=forward, Y=left, Z=up.
                # To project: u = (-Y/X)*fx + cx, v = (-Z/X)*fy + cy
                X = pts_t[0, :]
                Y = pts_t[1, :]
                Z = pts_t[2, :]
                
                # Filter points behind camera
                valid_mask = X > 0.1
                if not np.any(valid_mask):
                    continue
                    
                X = X[valid_mask]
                Y = Y[valid_mask]
                Z = Z[valid_mask]
                
                fx, fy = K_t[0, 0], K_t[1, 1]
                cx, cy = K_t[0, 2], K_t[1, 2]
                
                u_proj = (-Y / X) * fx + cx
                v_proj = (-Z / X) * fy + cy
                
                # Get current frame's 2D lane
                curr_lane = np.array(curr_frame['lanes'][curr_idx]) # N x 2 (u, v)
                
                # We need to compute error. Since v might not exactly match, we interpolate
                # Current lane v is strictly decreasing (bottom to top usually), let's sort it
                sort_idx = np.argsort(curr_lane[:, 1])
                curr_u = curr_lane[sort_idx, 0]
                curr_v = curr_lane[sort_idx, 1]
                
                # Interpolate u_proj at curr_v
                # wait, better to interpolate curr_u at v_proj, because v_proj comes from valid 3D points
                # v_proj might be out of bounds, filter first
                in_bound = (v_proj >= np.min(curr_v)) & (v_proj <= np.max(curr_v))
                if not np.any(in_bound):
                    continue
                    
                v_eval = v_proj[in_bound]
                u_eval = u_proj[in_bound]
                
                # numpy interp expects x-coordinates to be increasing.
                # curr_v is increasing because of sort_idx
                u_gt = np.interp(v_eval, curr_v, curr_u)
                
                error = np.abs(u_eval - u_gt)
                total_error += np.sum(error)
                total_points += len(error)

    print("=== Task 1: Projection Error ===")
    if total_points > 0:
        mean_error = total_error / total_points
        if mean_error > 100:
            print("Debug info:")
            # Just print the first valid track
            pass
        print(f"Mean projection error across {total_points} points: {mean_error:.4f} pixels")
    else:
        print("No valid points found for projection.")
        
    print("\n=== Task 2: Lane Change Events ===")
    print(f"Total lane change events (appearance/disappearance): {len(lane_change_events)}")
    
    # Save events to a file or print a summary
    with open("lane_change_events.txt", "w") as f:
        for event in lane_change_events:
            f.write(f"Segment: {event['segment']}, Frame: {event['frame_idx']}, "
                    f"Appeared: {event['appeared']}, Disappeared: {event['disappeared']}\n")
    print("Detailed lane change events saved to lane_change_events.txt")

if __name__ == "__main__":
    main()
