import os
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 待评估的地图路径
PRED_MAP_PATH = "/home/xzy/deployTest/00004_optimized_ell_sf2.map"

# 2. Ground Truth 路径
GT_ROOT = "/home/xzy/datasets/Rellis-3D/00004"
POSES_FILE = os.path.join(GT_ROOT, "poses.txt")
LIDAR_DIR = os.path.join(GT_ROOT, "os1_cloud_node_kitti_bin")
LABEL_DIR = os.path.join(GT_ROOT, "os1_cloud_node_semantickitti_label_id")

# 3. 评估参数
EVAL_STRIDE = 10     
SEARCH_RADIUS = 0.5  
ANCHOR_FRAME_IDX = 95 # [重要] 假设建图是以第0帧为原点，如果不对请修改此值

# RELLIS-3D 类别映射
RELLIS_MAPPING = {
    0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    12: 10, 15: 11, 17: 12, 18: 13, 19: 14, 23: 15, 27: 16, 31: 17, 33: 18, 34: 19
}
CLASS_NAMES = {
    0: "void", 1: "dirt", 2: "grass", 3: "tree", 4: "pole", 5: "water", 
    6: "sky", 7: "vehicle", 8: "object", 9: "asphalt", 10: "building", 
    11: "log", 12: "person", 13: "fence", 14: "bush", 15: "concrete", 
    16: "barrier", 17: "puddle", 18: "mud", 19: "rubble"
}
NUM_CLASSES = 20

# ================= 工具函数 =================

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
    label_mapped = np.array([RELLIS_MAPPING.get(l, 0) for l in label_raw])
    
    ones = np.ones((points.shape[0], 1))
    points_homo = np.hstack((points, ones))
    
    # 转换为 Global 坐标
    points_global = (poses[frame_idx] @ Tr @ points_homo.T).T
    
    return points_global[:, :3], label_mapped

# ================= 主逻辑 =================

def main():
    # 1. 加载预测地图
    map_points, map_preds = load_map_file(PRED_MAP_PATH)
    print(f"Map Loaded. Points: {len(map_points)}")
    
    # 2. 加载 Poses 并进行坐标系修复
    print("Loading Poses and Aligning Coordinates...")
    poses = load_poses(POSES_FILE)
    
    # [核心修复] 将地图点云从 Local (Pose 0 Frame) 转换到 Global
    anchor_pose = poses[ANCHOR_FRAME_IDX]
    print(f"Applying Anchor Pose Transform (Frame {ANCHOR_FRAME_IDX})...")
    print(f"Translation: {anchor_pose[:3, 3]}")
    
    ones = np.ones((map_points.shape[0], 1))
    map_points_homo = np.hstack((map_points, ones))
    # Transform: P_global = T_anchor * P_local
    map_points_global = (anchor_pose @ map_points_homo.T).T[:, :3]
    
    # 更新地图点云为全局坐标
    map_points = map_points_global
    
    # 3. 累积全量 GT
    Tr = np.eye(4) 
    print("Accumulating Full Ground Truth...")
    gt_points_all = []
    gt_labels_all = []
    
    for i in tqdm(range(0, len(poses), EVAL_STRIDE)):
        pts, lbls = load_gt_frame(i, poses, Tr)
        if pts is not None:
            gt_points_all.append(pts[::5]) # 降采样以节省内存
            gt_labels_all.append(lbls[::5])
            
    gt_points_all = np.concatenate(gt_points_all, axis=0)
    gt_labels_all = np.concatenate(gt_labels_all, axis=0)
    
    # 4. 再次检查对齐
    map_center = np.mean(map_points, axis=0)
    gt_center = np.mean(gt_points_all, axis=0)
    dist_diff = np.linalg.norm(map_center - gt_center)
    print(f"[对齐检查] Map中心: {map_center.astype(int)}, GT中心: {gt_center.astype(int)}")
    print(f"[对齐检查] 距离差异: {dist_diff:.2f} m")
    
    if dist_diff > 10.0:
        print("[严重警告] 对齐仍然失败，请检查 ANCHOR_FRAME_IDX 是否正确！")
        # 可以尝试在这里自动搜索最近的 Pose
        # return

    # 5. KDTree 匹配与评估
    print("Building KDTree & Querying...")
    tree = cKDTree(gt_points_all)
    dists, indices = tree.query(map_points, k=1, distance_upper_bound=SEARCH_RADIUS)
    
    valid_mask = dists <= SEARCH_RADIUS
    match_rate = np.sum(valid_mask) / len(map_points) * 100
    print(f"Valid Matched Points: {np.sum(valid_mask)} / {len(map_points)} ({match_rate:.2f}%)")
    
    if match_rate < 10.0:
        print("匹配率依然过低。")
        return

    # 6. 计算 mIoU
    gt_l_matched = gt_labels_all[indices[valid_mask]]
    pred_l_matched = map_preds[valid_mask]
    
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES))
    np.add.at(confusion, (gt_l_matched, pred_l_matched), 1)
    
    print("\n=== Class-wise IoU ===")
    ious = []
    for i in range(NUM_CLASSES):
        tp = confusion[i, i]
        denom = np.sum(confusion[i, :]) + np.sum(confusion[:, i]) - tp
        
        name = CLASS_NAMES.get(i, str(i))
        if denom == 0:
            print(f"Class {i:<2} ({name:<10}): NaN")
        else:
            iou = tp / denom
            ious.append(iou)
            print(f"Class {i:<2} ({name:<10}): {iou:.4f}")
            
    print(f"\n==============================")
    print(f"Final mIoU: {np.mean(ious):.4f}")
    print(f"==============================")

if __name__ == "__main__":
    main()