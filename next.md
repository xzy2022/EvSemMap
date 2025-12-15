既然你已经完成了 `ex-prep-rellis` 步骤，并生成了用于下游任务的 `.npy` 证据文件（在 `01_inferenced_npy` 目录下），接下来的复现流程主要包含三个阶段：**投影 (Projection)** -\> **格式转换 (Conversion)** -\> **建图 (Mapping)**。

这是从“2D 图像理解”迈向“3D 空间建图”的关键跨越。请按照以下步骤操作：

### 第一阶段：2D 语义投影到 3D 点云 (Projection)

**目的**：将你刚刚生成的 2D 预测结果和不确定性信息，结合相机内参和外参，投影到 3D 激光雷达点云上。

**1. 修改路径配置**
打开文件 `Projection/rellis_acc_inference.py`，你需要根据你的服务器环境修改硬编码的路径。

找到如下代码块（大约在 118-125 行左右），并进行修改：

```python
# 修改前（原作者路径）：
# RELLIS_ROOT = '/data/Rellis-3D'
# RELLIS_CAMERA_INFO = '/data/Rellis_3D_cam_intrinsic/Rellis-3D'
# RELLIS_TRANSFORM = '/data/Rellis_3D_cam2lidar_20210224/Rellis_3D'

# 修改后（你的环境）：
RELLIS_ROOT = '/root/autodl-tmp/Rellis-3D'  # 你的数据集根目录
# 注意：你需要确认是否有 camera_info.txt 和 transforms.yaml 文件
# 如果它们在数据集子文件夹里，你需要相应调整路径，或者把它们复制出来
RELLIS_CAMERA_INFO = '/root/autodl-tmp/Rellis-3D' 
RELLIS_TRANSFORM = '/root/autodl-tmp/Rellis-3D'

# ... (往下翻) ...

# 修改前：
# MY_DIR_ROOT = '/kjyoung/convertedRellis'

# 修改后：
MY_DIR_ROOT = '/root/convertedRellis' # 这必须与你在 prep.py 中设置的 OUTPUT_ROOT 一致
```

**2. 执行投影脚本**
在 `Projection` 目录下运行以下命令。这会将序列 `00004` 的每一帧进行投影并融合（binning）。

```bash
cd Projection
# 语法: python rellis_acc_inference.py {remark} {sequence} {binning_num}
# binning_num=30 意味着每30帧合并成一个文件，用于加速建图
python rellis_acc_inference.py rellisv3_edl_train-4 00004 30
```

  * **成功标志**：运行结束后，你的 `/root/convertedRellis/rellisv3_edl_train-4/` 下会出现 `03_nby3pK_npy` 文件夹，里面包含融合后的 `.npy` 文件。

-----

### 第二阶段：ROS 编译与数据格式转换

**目的**：EvSemMap 的建图模块是基于 C++ 和 ROS 的，它无法直接读取 Python 的 `.npy`，需要转换为 `.pcd` (Point Cloud Data) 格式。

**1. 编译 ROS 工作空间**
首先确保你安装了所有依赖（如 `pcl_ros`, `octomap_ros` 等）。

```bash
# 假设你的代码目录结构是 EvSemMap/SemanticMap/src/SemanticMap
cd ~/EvSemMap/SemanticMap  # 进入包含 src 的父目录
catkin_make                # 编译 ROS 包
source devel/setup.bash    # 刷新环境变量
```

**2. 修改转换配置**
打开 `SemanticMap/src/SemanticMap/launch/pcd_conversion.launch`，修改输入输出路径：

```xml
<launch>
    <arg name="pkg" default="$(find evsemmap)" />
    <node pkg="evsemmap" type="pcd_conversion" name="pcd_conversion" output="screen">        
        <param name="convert_mode_to_my_extension" value="true" />

        <param name="scan_start" value="1" />   
        <param name="scan_num" value="30" />   

        <param name="out_path" value="/home/xzy/Downloads/convertedRellis/rellisv3_edl_train-4/05_pVec_pcd/00004/" />
        <param name="dir" value="/home/xzy/Downloads/convertedRellis/rellisv3_edl_train-4/03_nby3pK_npy/00004/" />

    </node>
</launch>
```

**3. 执行转换**

```bash
roslaunch evsemmap pcd_conversion.launch
```

  * **成功标志**：终端打印 `Saved ... data points`，并且 `05_pVec_pcd` 目录下出现 `.pcd` 文件。

-----

### 第三阶段：构建语义地图 (Mapping)

**目的**：使用贝叶斯核推理 (BKI) 将离散的点云融合成连续的体素地图。

**1. 修改建图配置**
打开 `SemanticMap/src/SemanticMap/config/datasets/deploy_rellisv3_4_1-30.yaml`，这是核心配置文件。

```yaml
# 修改输入目录，指向你刚刚生成的 pcd 文件夹
dir: /root/convertedRellis/rellisv3_edl_train-4/05_pVec_pcd/00004/

# 确保 scan_num 与转换阶段一致
scan_num: 30 
```

**2. 执行建图命令**
这里我们使用论文推荐的 `ebs` (Evidential Bayesian Semantic Mapping) 方法或 `dempster` 方法。

```bash
# dataset 对应 config/datasets/ 下的文件名
# method 对应 config/methods/ 下的文件名
# result_name 是输出地图的路径和前缀
roslaunch evsemmap mapping.launch \
    dataset:=deploy_rellisv3_4_1-30 \
    method:=ebs \
    result_name:=/root/deployTest/rellis_map
```

  * **结果**：这将生成一个包含语义和不确定性信息的全局地图文件（通常是 `.map` 或自定义文本格式）。

-----

### 总结：你的“任务清单”

1.  **Run Python**: 修改并运行 `Projection/rellis_acc_inference.py`。
2.  **Build ROS**: 在 `SemanticMap` 目录运行 `catkin_make`。
3.  **Run ROS (Convert)**: 修改并运行 `pcd_conversion.launch`。
4.  **Run ROS (Map)**: 修改 `deploy_rellisv3_4_1-30.yaml` 并运行 `mapping.launch`。

如果你在第一步 Projection 找不到 `camera_info` 或 `transforms` 文件，请检查 Rellis-3D 数据集解压后的目录结构，这些标定文件通常需要单独下载或位于特定子目录中。