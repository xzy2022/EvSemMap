import os
import numpy as np
import struct

# ================= 配置路径 =================
# 根据 loader_rellis.py 中的配置，您的数据根目录可能在这里
RELLIS_ROOT = '/home/xzy/datasets/Rellis-3D' 
SEQUENCE = '00004'

# RELLIS-3D 官方目录结构通常如下：
# 点云目录
LIDAR_DIR = os.path.join(RELLIS_ROOT, SEQUENCE, 'os1_cloud_node_kitti_bin')
# 官方 3D 标签目录 (SemanticKITTI 格式)
# 注意：如果此文件夹不存在，说明您可能没有下载官方的 3D 标签数据
LABEL_DIR = os.path.join(RELLIS_ROOT, SEQUENCE, 'os1_cloud_node_semantickitti_label_id')

def check_paths():
    if not os.path.exists(LIDAR_DIR):
        print(f"[Error] 点云目录不存在: {LIDAR_DIR}")
        return False
    if not os.path.exists(LABEL_DIR):
        print(f"[Warning] 3D 标签目录不存在: {LABEL_DIR}")
        print("  -> 请检查是否已下载 RELLIS-3D 的 'Annotations' 或 'Labels' 部分。")
        print("  -> 对于 HMC 参数搜索，您必须拥有这些官方标签作为 Ground Truth。")
        return False
    return True

def load_kitti_bin(bin_path):
    """加载二进制点云文件 (x, y, z, intensity)"""
    # float32, reshaped to (N, 4)
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan[:, :3] # 返回 x, y, z

def load_semantic_label(label_path):
    """加载二进制标签文件 (SemanticKITTI 格式)"""
    # uint32, 每个点一个标签
    # 低 16 位是语义标签，高 16 位是实例 ID
    label = np.fromfile(label_path, dtype=np.uint32)
    semantic_label = label & 0xFFFF  # 取低 16 位
    return semantic_label

def main():
    print(f"Checking Sequence: {SEQUENCE} in {RELLIS_ROOT}...")
    
    if not check_paths():
        return

    # 获取文件列表并排序，确保一一对应
    scan_files = sorted(os.listdir(LIDAR_DIR))
    label_files = sorted(os.listdir(LABEL_DIR))
    
    # 简单的对齐检查
    if len(scan_files) == 0:
        print("目录下无点云文件。")
        return
        
    print(f"Found {len(scan_files)} scans and {len(label_files)} labels.")
    
    # 取中间一帧进行验证 (避免开头结尾可能存在的空数据)
    idx = min(50, len(scan_files) - 1)
    sample_bin = scan_files[idx]
    # 标签文件名通常与点云文件名一致，只是后缀不同 (.label vs .bin)
    sample_label_name = sample_bin.replace('.bin', '.label')
    
    bin_path = os.path.join(LIDAR_DIR, sample_bin)
    label_path = os.path.join(LABEL_DIR, sample_label_name)
    
    if not os.path.exists(label_path):
        print(f"[Error] 找不到对应的标签文件: {label_path}")
        return

    # 加载数据
    points = load_kitti_bin(bin_path)
    labels = load_semantic_label(label_path)
    
    # 验证
    print(f"\n--- Frame {sample_bin} Verification ---")
    print(f"Point Cloud Shape: {points.shape}")
    print(f"Label Shape:       {labels.shape}")
    
    if points.shape[0] == labels.shape[0]:
        print("[Success] 点云数量与标签数量一致。")
        
        # 打印一些存在的类别 ID，确认数据是否合理
        unique_labels = np.unique(labels)
        print(f"包含的语义类别 ID (前10个): {unique_labels[:10]}")
        
        # 简单保存一个小样本用于 HMC 调试 (可选)
        # save_path = 'mini_sample_00004.npy'
        # np.save(save_path, {'points': points, 'labels': labels})
        # print(f"已保存测试样本到: {save_path}")
    else:
        print(f"[Fail] 数量不匹配! Points: {points.shape[0]}, Labels: {labels.shape[0]}")

if __name__ == "__main__":
    main()