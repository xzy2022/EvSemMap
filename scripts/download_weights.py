import os
import torch
from torchvision import models

def download_resnet50():
    # 1. 以“脚本所在路径”为基准
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 项目根目录 = scripts 的上一级
    project_root = os.path.dirname(script_dir)

    # 3. 目标路径：EvSemMap/pretrained
    target_dir = os.path.join(project_root, "pretrained")

    os.makedirs(target_dir, exist_ok=True)

    save_path = os.path.join(target_dir, "resnet50-19c8e357.pth")
    # save_path = os.path.join(target_dir, "resnet50-0676ba61.pth")


    print("Downloading ResNet50 weights (IMAGENET1K_V1)...")
    try:
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        torch.save(model.state_dict(), save_path)

        print(f"✅ Success! Weights saved to:\n{save_path}")
        print("You can now specify this path in your config files.")

    except Exception as e:
        print(f"❌ Error downloading weights: {e}")

if __name__ == "__main__":
    download_resnet50()
