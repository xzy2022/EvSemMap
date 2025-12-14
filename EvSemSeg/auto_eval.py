import os
import re

# 配置
ckpt_dir = "./ckpts/rellisv3_edl_train-4_temp"
log_file = "./results/eval_results.txt"

# 获取所有 .pth 文件并按数字排序
files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
# 提取数字进行排序
files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

print(f"Found checkpoints: {files}")

for f in files:
    ckpt_path = os.path.join(ckpt_dir, f)
    print(f"Evaluating {ckpt_path}...")
    
    # 构造命令 (使用 ex-val-holdout 模式)
    # 注意：这里我们直接调用 main.py 方便传参，或者你修改 execute_params.py 每次接受参数
    # 这里偷懒直接拼 shell 命令
    cmd = (
        f"CUDA_VISIBLE_DEVICES=0 {sys.executable} main.py "  # 指定单卡0验证即可
        f"--batch_size 16 "           # 显存够大可以设大点，跑得快
        f"--model evidential "
        f"--dataset rellis_4 "        # 关键：指定验证集序列 00004
        f"--remap_version 3 "
        f"--phase val "               # 关键：验证模式
        f"--remark eval_result "      # 给个名字，方便看日志
        f"--load {ckpt_path} "        # 循环传入的权重路径
        f"--evd_type edl --unc_act exp --unc_type log --kl_strength 0.5 --ohem -1.0 " # 保持模型结构参数一致
        f"--with_void False"
        # 注意：绝对不要加 --partial_val 100
    )
    
    # 执行并把结果追加到日志
    os.system(f"{cmd} >> {log_file}")

print("Evaluation finished. Check results/eval_results.txt")