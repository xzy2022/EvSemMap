import os
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# ================= 配置区域 =================
PRED_MAP_PATH = "/home/xzy/deployTest/00004_optimized_ell_sf2.map"
GT_ROOT = "/home/xzy/datasets/Rellis-3D/00004"
POSES_FILE = os.path.join(GT_ROOT, "poses.txt")
LIDAR_DIR = os.path.join(GT_ROOT, "os1_cloud_node_kitti_bin")

# 采样参数
MAP_SAMPLE_SIZE = 2000   # 地图采样点数
GT_SKIP = 5              # GT 帧采样间隔
GT_POINT_SKIP = 10       # GT 单帧点采样间隔
POSE_SKIP = 1            # 遍历 Pose 的间隔 (1表示逐帧搜索)

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

def load_map_sampled(map_path, n_sample):
    print(f"Loading map: {map_path} ...")
    points = []
    with open(map_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) < 3: continue
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    points = np.array(points)
    if len(points) > n_sample:
        indices = np.random.choice(len(points), n_sample, replace=False)
        return points[indices]
    return points

def load_gt_cloud(poses, lidar_dir):
    print("Loading GT Point Cloud (this may take a while)...")
    gt_points = []
    
    # 只加载前 1000 帧左右的 GT 即可，因为我们知道 Map 肯定在前面
    # 如果不确定，可以加载全部
    n_frames = len(poses)
    
    for i in tqdm(range(0, n_frames, GT_SKIP)):
        bin_path = os.path.join(lidar_dir, f"{i:06d}.bin")
        if not os.path.exists(bin_path): continue
        
        scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
        pts = scan[::GT_POINT_SKIP, :3] # 降采样
        
        # Local -> Global
        ones = np.ones((pts.shape[0], 1))
        pts_homo = np.hstack((pts, ones))
        pts_global = (poses[i] @ pts_homo.T).T[:, :3]
        
        gt_points.append(pts_global)
        
    return np.concatenate(gt_points, axis=0)

# ================= 主程序 =================

def main():
    if not os.path.exists(PRED_MAP_PATH):
        print(f"File not found: {PRED_MAP_PATH}")
        return

    # 1. 准备数据
    map_points = load_map_sampled(PRED_MAP_PATH, MAP_SAMPLE_SIZE)
    poses = load_poses(POSES_FILE)
    gt_points = load_gt_cloud(poses, LIDAR_DIR)
    
    print(f"Building GT KDTree ({len(gt_points)} points)...")
    tree = cKDTree(gt_points)
    
    # 2. 暴力搜索最佳 Anchor
    print(f"Searching for best anchor among {len(poses)} poses...")
    
    results = []
    
    map_homo = np.hstack((map_points, np.ones((len(map_points), 1)))) # N x 4
    
    # 遍历每一个 Pose 作为假设的 Anchor
    for i in tqdm(range(0, len(poses), POSE_SKIP)):
        anchor_pose = poses[i]
        
        # 变换: P_global = T_anchor * P_local
        # 注意：这里我们只变换采样的 2000 个点，速度很快
        map_global = (anchor_pose @ map_homo.T).T[:, :3]
        
        # 查询最近邻距离
        dists, _ = tree.query(map_global, k=1)
        mean_dist = np.mean(dists)
        
        results.append((i, mean_dist))
        
    # 3. 排序并输出
    results.sort(key=lambda x: x[1])
    
    print("\n=== Top 5 Best Fitting Anchors ===")
    for idx, dist in results[:5]:
        print(f"Frame {idx:06d}: Mean Error = {dist:.4f} m")
        
    best_frame, best_err = results[0]
    
    if best_err > 1.0:
        print("\n[警告] 即使是最佳匹配，误差依然很大 (>1.0m)。")
        print("可能原因：1. 坐标轴定义不同 (如 Z轴反转)；2. 需要外参变换。")
    else:
        print(f"\n[成功] 建议在 eval_map_iou.py 中设置 ANCHOR_FRAME_IDX = {best_frame}")

if __name__ == "__main__":
    main()