import os
import numpy as np
from tqdm import tqdm # 进度条库，如果没有请 pip install tqdm

# ================= 配置路径 =================
RELLIS_ROOT = '/home/xzy/datasets/Rellis-3D'
SEQUENCE = '00004'
START_FRAME = 50
END_FRAME = 100
DOWNSAMPLE_RATE = 10  # 每10个点取1个，防止数据量爆炸

LIDAR_DIR = os.path.join(RELLIS_ROOT, SEQUENCE, 'os1_cloud_node_kitti_bin')
LABEL_DIR = os.path.join(RELLIS_ROOT, SEQUENCE, 'os1_cloud_node_semantickitti_label_id')

def load_kitti_bin(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    return scan[:, :3]

def load_semantic_label(label_path):
    label = np.fromfile(label_path, dtype=np.uint32)
    semantic_label = label & 0xFFFF 
    return semantic_label

def main():
    print(f"Extracting frames {START_FRAME} to {END_FRAME} from {SEQUENCE}...")
    
    scan_files = sorted(os.listdir(LIDAR_DIR))
    
    all_points = []
    all_labels = []
    
    # 确保索引不越界
    end_idx = min(END_FRAME, len(scan_files))
    
    for i in tqdm(range(START_FRAME, end_idx)):
        bin_file = scan_files[i]
        label_file = bin_file.replace('.bin', '.label')
        
        bin_path = os.path.join(LIDAR_DIR, bin_file)
        label_path = os.path.join(LABEL_DIR, label_file)
        
        if not os.path.exists(label_path):
            continue
            
        pts = load_kitti_bin(bin_path)
        lbls = load_semantic_label(label_path)
        
        # 降采样
        pts = pts[::DOWNSAMPLE_RATE]
        lbls = lbls[::DOWNSAMPLE_RATE]
        
        all_points.append(pts)
        all_labels.append(lbls)
        
    # 合并数据
    final_points = np.concatenate(all_points, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nExtraction Done.")
    print(f"Total Points: {final_points.shape[0]}")
    
    # 保存
    save_name = f'rellis_{SEQUENCE}_frames_{START_FRAME}_to_{END_FRAME}.npy'
    save_path = os.path.join(RELLIS_ROOT, SEQUENCE, save_name)
    np.save(save_path, {'points': final_points, 'labels': final_labels})
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    main()