from pathlib import Path
import numpy as np
from PIL import Image

label_dir = Path("/home/xzy/datasets/Rellis-3D/00000/pylon_camera_node_label_id")
png = sorted(label_dir.glob("*.png"))[0]   # 取第一张
print("File:", png)

im = Image.open(png)
print("PIL mode:", im.mode)               # 关键：L / P / RGB / I;16 等
arr = np.array(im)
print("np shape:", arr.shape)
print("dtype:", arr.dtype)
print("min/max:", arr.min(), arr.max())

# 看看有哪些类别ID（只采样统计，避免太慢）
uniq = np.unique(arr)
print("unique count:", len(uniq))
print("first 30 unique values:", uniq[:30])
print("last 30 unique values:", uniq[-30:])

# 如果是调色板图(P模式)，把 palette 打出来看看
if im.mode == "P":
    pal = im.getpalette()
    print("palette len:", len(pal) if pal is not None else None)
