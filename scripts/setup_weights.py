# 新文件名建议: setup_weights.py
import os
import torch
from torchvision import models

def setup_local_weights():
    # 路径设定
    current_dir = os.path.dirname(os.path.abspath(__file__)) # scripts/
    project_root = os.path.dirname(current_dir)              # EvSemMap/
    pretrained_dir = os.path.join(project_root, "pretrained")
    
    # 1. 刚才 wget 下载下来的源文件
    source_file = os.path.join(pretrained_dir, "resnet50-0676ba61.pth")
    
    # 2. 最终你项目想要保存的目标文件名 (如果不改名，这一步其实只是原地复制)
    # 假设你项目代码里读取的是 "resnet50.pth"
    target_file = os.path.join(pretrained_dir, "resnet50-0676ba61.pth")

    if not os.path.exists(source_file):
        print(f"❌ 错误: 找不到文件 {source_file}")
        print("请先执行 wget 命令下载！")
        return

    print(f"正在从本地文件加载: {source_file} ...")
    
    try:
        # 关键修改：weights=None (不下载)，然后手动 load_state_dict
        model = models.resnet50(weights=None)
        
        # 加载刚才 wget 的权重
        state_dict = torch.load(source_file)
        model.load_state_dict(state_dict)
        
        # 保存为你项目需要的最终格式/名字
        torch.save(model.state_dict(), target_file)
        
        print(f"✅ 处理完成！最终权重已保存至:\n{target_file}")
        
    except Exception as e:
        print(f"❌ 处理出错: {e}")

if __name__ == "__main__":
    setup_local_weights()