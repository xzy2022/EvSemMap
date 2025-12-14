#!/bin/bash
cd /root/autodl-fs
target_dir="/root/autodl-tmp/Rellis-3D"

# 创建目标目录
mkdir -p "$target_dir"

# 解压所有分卷压缩文件
for prefix in 00000 00001 00002 00003 00004; do
# for prefix in 00000 00001 00002 00003; do
# for prefix in 00004; do
    first_part="${prefix}.7z.001"
    if [ -f "$first_part" ]; then
        echo "正在解压 $prefix 系列文件..."
        7z x "$first_part" -o"$target_dir"
    fi
done

# 解压预训练模型
# if [ -f "pretrained.zip" ]; then
#     echo "正在解压预训练模型..."
#     unzip pretrained.zip -d /root/autodl-tmp/
# fi

echo "解压完成！"