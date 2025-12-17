import os
import numpy as np

# ================= 配置 =================
PRED_MAP_PATH = "/home/xzy/deployTest/00004_optimized_ell_sf2.map"

# GT 相关 (保持和 eval_map_iou 一致)
GT_ROOT = "/home/xzy/datasets/Rellis-3D/00004"
LABEL_DIR = os.path.join(GT_ROOT, "os1_cloud_node_semantickitti_label_id")

# RELLIS 原始 ID 到 Train ID 的映射 (eval_map_iou.py 中使用的)
RELLIS_MAPPING = {
    0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    12: 10, 15: 11, 17: 12, 18: 13, 19: 14, 23: 15, 27: 16, 31: 17, 33: 18, 34: 19
}

def load_map_labels(map_path):
    preds = []
    with open(map_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) < 5: continue
            preds.append(int(parts[4])) # 第5列是 Label
    return np.array(preds)

def load_gt_labels_sample(label_dir):
    # 随机读几个文件看看 GT 转换后的样子
    files = sorted(os.listdir(label_dir))[100:105] 
    all_mapped = []
    all_raw = []
    for f in files:
        path = os.path.join(label_dir, f)
        label_raw = np.fromfile(path, dtype=np.uint32) & 0xFFFF
        label_mapped = np.array([RELLIS_MAPPING.get(l, 0) for l in label_raw])
        all_mapped.extend(label_mapped)
        all_raw.extend(label_raw)
    return np.unique(all_raw), np.unique(all_mapped)

def main():
    print("--- 1. 分析预测地图 (Map) ---")
    if os.path.exists(PRED_MAP_PATH):
        map_labels = load_map_labels(PRED_MAP_PATH)
        u_map, c_map = np.unique(map_labels, return_counts=True)
        print(f"地图中出现的唯一 ID: {u_map}")
        print(f"Top 5 高频 ID: {u_map[np.argsort(-c_map)][:5]}")
    else:
        print("地图文件未找到")

    print("\n--- 2. 分析真值 (GT) ---")
    if os.path.exists(LABEL_DIR):
        u_raw, u_mapped = load_gt_labels_sample(LABEL_DIR)
        print(f"GT 原始文件中的 ID: {u_raw}")
        print(f"GT 经过 RELLIS_MAPPING 转换后的 ID (eval脚本用的): {u_mapped}")
    
    print("\n--- 3. 结论 ---")
    print("请对比上面 [地图 ID] 和 [GT 转换后 ID] 是否有交集？")

if __name__ == "__main__":
    main()