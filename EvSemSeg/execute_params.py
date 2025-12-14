import os
import argparse
import sys
PYTHON_EXE = sys.executable


# --log_dir ./ckpts (default)
# --save_freq 10 (default)
common_params_for_train = f'--evd_type edl \
   --unc_act exp \
   --unc_type log \
   --kl_strength 0.5 \
   --ohem -1.0 '
############################################# TRAIN #############################################
# 本项目使用DP进行并行训练，batch_size会自动被平均分到各个卡上
# -- load指向你要恢复的 checkpoint，例如
# --load ./ckpts/rellisv3_edl_train-4_temp/50.pth
# --load \'$NONE$\'  表示重新训练
# --save_freq 1 表示每个epoch都保存权重，否则每10轮保存一次（追加而非覆盖）
rellisv3_train = f'CUDA_VISIBLE_DEVICES=0,1 {PYTHON_EXE} main.py \
   --n_epoch 100 \
   --batch_size 16 \
   --l_rate 2e-4 \
   --model evidential \
   --dataset rellis_-4 \
   --remap_version 3 \
   --phase train \
   --remark rellisv3_edl_train-4_temp \
   --load \'$NONE$\' \
   {common_params_for_train}\
   --save_freq 5\
   --with_void False'
########################################### VALIDATION ###########################################
# [作用]：在训练集划分上跑验证（通常意义不大，主要是检查代码是否跑通）
rellisv3_val = f'CUDA_VISIBLE_DEVICES=2 {PYTHON_EXE} main.py \
   --n_epoch 100 \
   --batch_size 24 \
   --l_rate 2e-4 \
   --model evidential \
   --dataset rellis_-4 \
   --remap_version 3 \
   --phase val \
   --remark rellisv3_edl_train-4 \
   --load /kjyoung/EvSemMapCode/EvSemSeg/ckpts/rellisv3_edl_train-4/100.pth \
   --partial_val 100 \
   {common_params_for_train}\
   --with_void False'

# [作用]：【核心命令】在“留出集”（Sequence 4）上跑验证。
# 有图像处理
# dataset rellis_4 意味着只读取 00004 序列。而训练中只用了 00000 00001 00002 00003 四个序列。
rellisv3_val_holdout = f'CUDA_VISIBLE_DEVICES=0 {PYTHON_EXE} main.py \
   --n_epoch 100 \
   --batch_size 8 \
   --l_rate 2e-4 \
   --model evidential \
   --dataset rellis_4 \
   --remap_version 3 \
   --phase val \
   --remark rellisv3_edl_train-4 \
   --load /root/autodl-tmp/ckpts/rellisv3_edl_train-4_temp/15.pth \
   {common_params_for_train}\
   --with_void False'

#    --partial_val 100 \ 只选择100个样本

# 测试跨数据集泛化能力
# 代码试图加载一个非 RELLIS 的数据集（这里写的是 DIFFERENT_DATASET，实际可能指向 RUGD 或私有数据集）。
rellisv3_val_cross = f'CUDA_VISIBLE_DEVICES=2 {PYTHON_EXE} main.py \
   --n_epoch 100 \
   --batch_size 24 \
   --l_rate 2e-4 \
   --model evidential \
   --dataset rellis_4 \
   --cross_inference DIFFERENT_DATASET(Offroad dataset would not be published) \
   --remap_version 3 \
   --phase val \
   --remark rellisv3_edl_train-4 \
   --load /kjyoung/EvSemMapCode/EvSemSeg/ckpts/rellisv3_edl_train-4/100.pth \
   --partial_val 100 \
   {common_params_for_train}\
   --with_void False'

########################################### TEST ###########################################
# 标准的论文跑分命令。它会在 RELLIS 的测试集（序列 4）上跑完全部数据，计算最终的 mIoU
# 通常也会保存彩色的分割结果图，用于写论文贴图或人工检查。
# 似乎这个才是没有处理图像
rellisv3_test_holdout = f'CUDA_VISIBLE_DEVICES=0 {PYTHON_EXE} main.py \
   --n_epoch 100 \
   --batch_size 40 \
   --l_rate 2e-4 \
   --model evidential \
   --dataset rellis_4 \
   --remap_version 3 \
   --phase test \
   --remark rellisv3_edl_train-4 \
   --load /root/autodl-tmp/ckpts/rellisv3_edl_train-4_temp/5.pth \
   {common_params_for_train}\
   --with_void False'

# batch = 40: 19557MB

########################################### PREP ###########################################
# EvSemMap 项目特有的关键步骤。它的目的不是为了给人看，而是为了给机器（3D 建图模块）看。
# 它会把推理出的 原始证据（Evidence/Logits）保存下来。
rellisv3_prep = f'CUDA_VISIBLE_DEVICES=0 {PYTHON_EXE} main.py \
   --batch_size 8 \
   --model evidential \
   --dataset rellis_4 \
   --cross_inference rellis_4 \
   --remap_version 3 \
   --phase prep \
   --remark rellisv3_edl_train-4 \
   {common_params_for_train}\
   --load /root/autodl-tmp/ckpts/rellisv3_edl_train-4_temp/15.pth \
   --with_void False'

def main():
    parser = argparse.ArgumentParser(description="Execute Commands")
    parser.add_argument('--mode', type=str, help='Mode (Pipeline) to execute')
    
    args = parser.parse_args()
    
    if args.mode == 'ex-train':
        os.system(rellisv3_train)
    
    elif args.mode == 'ex-val':
        os.system(rellisv3_val)
    elif args.mode == 'ex-val-holdout':
        os.system(rellisv3_val_holdout)
    elif args.mode == 'ex-val-cross':
        os.system(rellisv3_val_cross)

    elif args.mode == 'ex-test-holdout':
        os.system(rellisv3_test_holdout)

    elif args.mode == 'ex-prep-rellis':
        os.system(rellisv3_prep)
    
    else:
        raise Exception('MODE - Not Implemented')

if __name__ == "__main__":
    main()