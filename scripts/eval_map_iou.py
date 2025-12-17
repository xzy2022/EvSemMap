import os
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 待评估的地图路径 (Mode 参数生成的地图)
PRED_MAP_PATH = "/home/xzy/deployTest/00004_optimized_ell_sf2.map" 

# 2. Ground Truth 路径
GT_ROOT = "/home/xzy/datasets/Rellis-3D/00004"
POSES_FILE = os.path.join(GT_ROOT, "poses.txt")
CALIB_FILE = os.path.join(GT_ROOT, "calib.txt")
LIDAR_DIR = os.path.join(GT_ROOT, "os1_cloud_node_kitti_bin")
LABEL_DIR = os.path.join(GT_ROOT, "os1_cloud_node_semantickitti_label_id")

# 评估参数
EVAL_STRIDE = 10     # 抽样间隔
SEARCH_RADIUS = 1.0  # [调试] 放大搜索半径到 1米，看看是不是误差太大

# RELLIS-3D 类别映射 (Original -> Train ID)
RELLIS_MAPPING = {
    0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    12: 10, 15: 11, 17: 12, 18: 13, 19: 14, 23: 15, 27: 16, 31: 17, 33: 18, 34: 19
}
NUM_CLASSES = 20

# ================= 调试工具函数 =================

def print_stats(name, points, labels):
    print(f"\n[{name} 统计信息]")
    print(f"  点数: {len(points)}")
    if len(points) > 0:
        print(f"  空间范围 (XYZ Min): {np.min(points, axis=0)}")
        print(f"  空间范围 (XYZ Max): {np.max(points, axis=0)}")
        print(f"  空间中心 (Mean):    {np.mean(points, axis=0)}")
        
        unique_ids, counts = np.unique(labels, return_counts=True)
        print(f"  包含标签 IDs: {unique_ids}")
        # print(f"  标签分布: {dict(zip(unique_ids, counts))}")
        
        if np.max(unique_ids) > 19:
            print(f"  [警报] 发现大于 19 的 ID，说明 {name} 使用的是原始 ID (0-34)，可能需要映射！")
        else:
            print(f"  [正常] ID 范围在 0-19 之间，看起来是映射过的 Train ID。")

def load_calib_debug(calib_file):
    """更健壮的标定文件加载与调试"""
    Tr = np.eye(4)
    if not os.path.exists(calib_file):
        print(f"[Error] 标定文件不存在: {calib_file}")
        return Tr
    
    print(f"\n正在读取标定文件: {calib_file}")
    with open(calib_file, 'r') as f:
        content = f.readlines()
        for line in content:
            print(f"  文件内容: {line.strip()}") # 打印出来看看格式
            parts = line.strip().split()
            # 尝试匹配 'Tr:', 'Tr_velo_to_cam:', 或者没有前缀直接是数字的情况
            vals = []
            try:
                # 尝试查找这一行里是否有连续 12 个浮点数
                nums = [float(x) for x in parts if x.replace('.','',1).replace('e','',1).replace('-','',1).isdigit()]
                if len(nums) == 12:
                    vals = nums
            except:
                pass

            if len(vals) == 12:
                Tr[:3, :] = np.array(vals).reshape(3, 4)
                print("  -> 成功提取到 3x4 变换矩阵。")
                return Tr

    print("  [Warning] 未能在文件中找到 12 个变换参数，将使用 Identity 矩阵。这会导致点云无法对齐！")
    return Tr

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
    
    # 加载数据
    scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]
    label_raw = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
    
    # 标签映射 (Raw -> TrainID)
    label_mapped = np.array([RELLIS_MAPPING.get(l, 0) for l in label_raw])
    
    # 坐标变换
    ones = np.ones((points.shape[0], 1))
    points_homo = np.hstack((points, ones))
    
    # Tr: LiDAR -> Camera/Base
    points_ref = (Tr @ points_homo.T).T 
    
    # Pose: Camera/Base -> Global
    pose = poses[frame_idx]
    points_global = (pose @ points_ref.T).T
    
    return points_global[:, :3], label_mapped

# ================= 主逻辑 =================

def main():
    # 1. 加载预测地图
    map_points, map_preds = load_map_file(PRED_MAP_PATH)
    print_stats("预测地图 (Map)", map_points, map_preds)
    
    # 2. 加载标定 (Debug)
    Tr = load_calib_debug(CALIB_FILE)
    print(f"使用的标定矩阵 (Tr):\n{Tr}")
    
    # 3. 加载 GT
    print("\nAccumulating Ground Truth (Sampled)...")
    poses = load_poses(POSES_FILE)
    gt_points_all = []
    gt_labels_all = []
    
    # 为了快速 Debug，只取前 50 帧看看位置对不对
    for i in tqdm(range(0, min(len(poses), 100), EVAL_STRIDE)):
        pts, lbls = load_gt_frame(i, poses, Tr)
        if pts is not None:
            gt_points_all.append(pts[::10]) 
            gt_labels_all.append(lbls[::10])
            
    gt_points_all = np.concatenate(gt_points_all, axis=0)
    gt_labels_all = np.concatenate(gt_labels_all, axis=0)
    
    print_stats("真值点云 (GT)", gt_points_all, gt_labels_all)
    
    # 4. 检查中心点偏移
    map_center = np.mean(map_points, axis=0)
    gt_center = np.mean(gt_points_all, axis=0)
    dist_diff = np.linalg.norm(map_center - gt_center)
    print(f"\n[对齐检查] Map中心: {map_center}, GT中心: {gt_center}")
    print(f"[对齐检查] 中心点距离差异: {dist_diff:.2f} 米")
    
    if dist_diff > 5.0:
        print("!!! [严重警告] 地图和GT的空间位置差异巨大 (>5米)！")
        print("原因可能是: 1. 标定矩阵(Tr)错误  2. Poses 坐标系定义不同  3. GT 帧数范围和 Map 不匹配")
    
    # 5. KDTree 匹配 (Debug Only)
    tree = cKDTree(gt_points_all)
    dists, indices = tree.query(map_points, k=1, distance_upper_bound=SEARCH_RADIUS)
    valid_mask = dists <= SEARCH_RADIUS
    
    match_rate = np.sum(valid_mask) / len(map_points) * 100
    print(f"\n[匹配结果] 半径 {SEARCH_RADIUS}m 内匹配率: {match_rate:.2f}%")

    if match_rate < 5.0:
        print("-> 匹配率极低，无需计算 mIoU。请先解决坐标对齐问题。")
        return

    # 简单计算 mIoU
    valid_indices = indices[valid_mask]
    gt_l_matched = gt_labels_all[valid_indices]
    pred_l_matched = map_preds[valid_mask]
    
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES))
    np.add.at(confusion, (gt_l_matched, pred_l_matched), 1)
    
    print("\nClass-wise IoU (Debug):")
    ious = []
    for i in range(NUM_CLASSES):
        tp = confusion[i, i]
        denom = np.sum(confusion[i, :]) + np.sum(confusion[:, i]) - tp
        if denom > 0:
            iou = tp / denom
            ious.append(iou)
            print(f"Class {i}: {iou:.4f}")
    print(f"Mean IoU: {np.mean(ious):.4f}")

if __name__ == "__main__":
    main()