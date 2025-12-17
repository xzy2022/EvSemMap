import os
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# ================= 配置区域 =================
# PRED_MAP_PATH = "/home/xzy/deployTest/00004_optimized_ell_sf2.map"
PRED_MAP_PATH = "/home/xzy/deployTest/rellis_map.map"
GT_ROOT = "/home/xzy/datasets/Rellis-3D/00004"
POSES_FILE = os.path.join(GT_ROOT, "poses.txt")
LIDAR_DIR = os.path.join(GT_ROOT, "os1_cloud_node_kitti_bin")
LABEL_DIR = os.path.join(GT_ROOT, "os1_cloud_node_semantickitti_label_id")

# 评估参数
EVAL_STRIDE = 10     
SEARCH_RADIUS = 0.5  
ANCHOR_FRAME_IDX = 95 # 保持之前的自动对齐结果

# ================= 标签映射 (核心修复) =================
# 基于 v3_v1.txt 和 v3_v1_colormap_paper_unified.txt
# 将 RELLIS 原始 ID 映射到 9 类系统
RELLIS_TO_9CLASS = {
    0: 0,   # void -> void
    1: 6,   # dirt -> brown
    2: 5,   # sand -> unpaved
    3: 7,   # grass -> green
    4: 8,   # tree -> vegetation
    5: 3,   # pole -> object
    6: 2,   # water -> water
    7: 1,   # sky -> sky
    8: 3,   # vehicle -> object
    9: 3,   # object -> object
    10: 4,  # asphalt -> paved
    11: 5,  # gravel -> unpaved
    12: 3,  # building -> object
    13: 6,  # mulch -> brown
    14: 5,  # Rock-bed -> unpaved
    15: 6,  # log -> brown
    16: 3,  # bicycle -> object
    17: 3,  # person -> object
    18: 0,  # fence -> void
    19: 8,  # bush -> vegetation
    20: 3,  # sign -> object
    21: 0,  # Rock -> void
    22: 3,  # bridge -> object
    23: 4,  # concrete -> paved
    24: 3,  # Picnic-table -> object
    27: 0,  # barrier -> void
    31: 2,  # puddle -> water
    33: 6,  # mud -> brown
    34: 0   # rubble -> void
}

# 9 类系统的名称
CLASS_NAMES_9 = {
    0: "void",
    1: "sky",
    2: "water",
    3: "object",
    4: "paved",
    5: "unpaved",
    6: "brown",
    7: "green",
    8: "vegetation"
}
NUM_CLASSES = 9

def load_poses(pose_file):
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.eye(4)
            pose[:3, :] = np.array(values).reshape(3, 4)
            poses.append(pose)
    return poses

def load_map_file(map_path):
    print(f"Loading map: {map_path} ...")
    points = []
    preds = []
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Map file not found: {map_path}")
    with open(map_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) < 5: continue
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            preds.append(int(parts[4]))
    return np.array(points), np.array(preds)

def load_gt_frame(frame_idx, poses, Tr):
    bin_path = os.path.join(LIDAR_DIR, f"{frame_idx:06d}.bin")
    label_path = os.path.join(LABEL_DIR, f"{frame_idx:06d}.label")
    if not os.path.exists(bin_path): return None, None
    
    scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]
    
    label_raw = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
    
    # [核心修改] 使用 RELLIS_TO_9CLASS 进行映射
    label_mapped = np.array([RELLIS_TO_9CLASS.get(l, 0) for l in label_raw])
    
    ones = np.ones((points.shape[0], 1))
    points_homo = np.hstack((points, ones))
    points_global = (poses[frame_idx] @ Tr @ points_homo.T).T
    
    return points_global[:, :3], label_mapped

def main():
    # 1. 加载预测地图
    map_points, map_preds = load_map_file(PRED_MAP_PATH)
    print(f"Map Loaded. Points: {len(map_points)}")
    
    # 2. 坐标转换
    print("Loading Poses and Aligning Coordinates...")
    poses = load_poses(POSES_FILE)
    anchor_pose = poses[ANCHOR_FRAME_IDX]
    
    ones = np.ones((map_points.shape[0], 1))
    map_points_homo = np.hstack((map_points, ones))
    # 将地图从局部坐标转换到全局坐标
    map_points = (anchor_pose @ map_points_homo.T).T[:, :3]
    
    # 3. 累积 GT
    print("Accumulating Full Ground Truth...")
    gt_points_all = []
    gt_labels_all = []
    
    for i in tqdm(range(0, len(poses), EVAL_STRIDE)):
        pts, lbls = load_gt_frame(i, poses, np.eye(4))
        if pts is not None:
            gt_points_all.append(pts[::5]) 
            gt_labels_all.append(lbls[::5])
            
    gt_points_all = np.concatenate(gt_points_all, axis=0)
    gt_labels_all = np.concatenate(gt_labels_all, axis=0)
    
    # 4. KDTree 匹配
    print("Building KDTree & Querying...")
    tree = cKDTree(gt_points_all)
    dists, indices = tree.query(map_points, k=1, distance_upper_bound=SEARCH_RADIUS)
    
    valid_mask = dists <= SEARCH_RADIUS
    match_rate = np.sum(valid_mask) / len(map_points) * 100
    print(f"Valid Matched Points: {np.sum(valid_mask)} / {len(map_points)} ({match_rate:.2f}%)")
    
    if match_rate < 10.0:
        print("[Error] 匹配率过低。")
        return

    # 5. 计算 mIoU
    gt_l_matched = gt_labels_all[indices[valid_mask]]
    pred_l_matched = map_preds[valid_mask]
    
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES))
    np.add.at(confusion, (gt_l_matched, pred_l_matched), 1)
    
    print("\n=== Class-wise IoU (9 Classes) ===")
    ious = []
    for i in range(1, NUM_CLASSES): # 跳过 Class 0 (void)
        tp = confusion[i, i]
        denom = np.sum(confusion[i, :]) + np.sum(confusion[:, i]) - tp
        name = CLASS_NAMES_9.get(i, str(i))
        
        if denom == 0:
            print(f"Class {i:<2} ({name:<10}): NaN (No GT)")
            continue
            
        iou = tp / denom
        ious.append(iou)
        print(f"Class {i:<2} ({name:<10}): {iou:.4f}")
            
    print(f"\nFinal mIoU (excluding void): {np.mean(ious):.4f}")

if __name__ == "__main__":
    main()