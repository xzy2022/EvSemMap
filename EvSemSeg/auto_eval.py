import os
import re
import sys
PYTHON_EXE = sys.executable

# 配置
ckpt_dir = "/root/autodl-tmp/ckpts/rellisv3_edl_train-4_temp"

# 当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 项目根目录（脚本在子目录中，因此需要再取一次父目录）
project_root = os.path.dirname(current_dir)

# results 目录下的日志文件
log_file = os.path.join(project_root, "results", "eval_results.txt")



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
        f"CUDA_VISIBLE_DEVICES=0 {PYTHON_EXE} main.py "  # 指定单卡0验证即可
        f"--batch_size 40 "           # 显存够大可以设大点，跑得快
        f"--model evidential "
        f"--dataset rellis_4 "        # 关键：指定验证集序列 00004
        f"--remap_version 3 "
        f"--phase test "               # 关键：测试模式
        f"--remark eval_result "      # 给个名字，方便看日志
        f"--load {ckpt_path} "        # 循环传入的权重路径
        f"--evd_type edl --unc_act exp --unc_type log --kl_strength 0.5 --ohem -1.0 " # 保持模型结构参数一致
        f"--with_void False "
        
        # 注意：绝对不要加 --partial_val 100
    )
    
    # 执行并把结果追加到日志
    os.system(f"{cmd} >> {log_file}")

print("Evaluation finished. Check results/eval_results.txt")