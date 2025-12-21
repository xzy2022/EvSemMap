根据您提供的 `rellis_acc_inference.py` 和 `universal_utils.py`，以及您之前的“投影”进度，我建议您将后续工作拆分为以下 **4 个独立的 Jupyter Notebook**。

这种分步方式可以帮助您将 **数据关联（Data Association）**、**空间变换（Spatial Transformation）** 和 **不确定性计算（Uncertainty Calculation）** 解耦，便于调试和理解。

---

### **步骤一：单帧语义融合与深度校验**

**文件名建议：** `01_Frame_Fusion_and_Sampling.ipynb`

**目标：** 实现单帧 LiDAR 点云与 2D 概率图（Probability Map）的对应，并解决“遮挡”问题。

**核心内容：**

1. **数据读取**：加载一帧 LiDAR `.bin` 和对应的 2D 语义分割推理结果 `.npy`（即 `rellis_acc_inference.py` 中的 `lbl_path`）。
2. **投影索引**：利用您已经写好的投影代码，获取每个 3D 点在图像上的  坐标。
3. **概率采样（关键）**：
* 复现 `process_each_frame` 中的逻辑：根据  索引，从 `.npy` (维度 ) 中取出对应的概率向量 (维度 )。
* **改进点**：原代码直接采样（`label[:, v, u]`），您应在此步骤增加 **深度检查（Z-buffer check）** 或 **遮挡剔除**，避免将前景的概率赋给被遮挡的背景点。


4. **可视化验证**：
* 输出这一帧的点云，根据最大概率类别（Argmax）上色，检查物体边缘是否清晰（验证投影和采样是否正确）。



**涉及原代码函数：** `process_each_frame`

---

### **步骤二：多帧点云的时空对齐（拼接）**

**文件名建议：** `02_PointCloud_Accumulation.ipynb`

**目标：** 利用位姿信息，将连续多帧的点云统一到同一个坐标系下。

**核心内容：**

1. **位姿读取**：读取 `poses.txt`，解析出  变换矩阵。
2. **相对变换计算**：
* 复现 `accumulate` 函数的逻辑。
* 计算当前帧  到基准帧（如第 0 帧）的变换矩阵：。


3. **坐标变换**：将第  帧的点云  变换为 。
4. **堆叠（Stacking）**：将多帧变换后的几何坐标  和步骤一中采样得到的概率向量  进行 `np.concatenate` 拼接。
5. **可视化验证**：
* 使用 Matplotlib 或 Open3D 显示拼接后的点云，检查是否有明显的“分层”或“漂移”（验证位姿变换是否正确）。
* **注意**：观察移动物体（如车）是否留下了长长的“鬼影”，这是原代码的痛点。



**涉及原代码函数：** `accumulate`, `parse_poses_slam`

---

### **步骤三：证据理论（Evidential）与不确定性计算**

**文件名建议：** `03_Uncertainty_Calculation.ipynb`

**目标：** 处理拼接后的概率向量，计算不确定性指标（EvSemMap 的核心）。

**核心内容：**

1. **概率融合（可选）**：
* 原代码似乎只是简单的拼接点（points queue），没有对同一个空间位置的点的概率进行贝叶斯更新或 DS 证据理论融合。
* **创新点/进阶**：如果您想做体素化（Voxelization）融合，可以在这里尝试将空间相近的点的概率向量合并。


2. **不确定性度量**：
* 基于每个点的概率向量 ，计算不确定性。
* **熵 (Entropy)**: 
* **置信度 (Confidence)**: 
* 如果您的 `.npy` 输出的是 logits 而非 softmax 后的概率，或者对应 Evidential Deep Learning 的  参数，这里需要应用相应的 Dempster-Shafer 理论公式（如 Vacuity, Dissonance）。


3. **分析**：统计地图中哪些区域（如物体边缘、远处）不确定性最高。

**涉及原代码逻辑：** 原代码在 `rellis_acc_inference.py` 中主要是存储和传递 `labels`（即概率），具体计算可能在后续的 C++ 代码或 `universal_utils` 外部。您需要在 Python 中显式实现这部分。

---

### **步骤四：VTK 文件生成与最终可视化**

**文件名建议：** `04_Export_and_Visualization.ipynb`

**目标：** 将处理好的数据导出为标准格式，以便在 Paraview 中查看（复现 `universal_utils.py` 的功能）。

**核心内容：**

1. **数据整理**：准备好  的坐标数组，以及  的概率数组，或者  的不确定性数组。
2. **VTK 写入**：
* 调用 `universal_utils.py` 中的 `write_geoatt_vtk3_prob`。
* 该函数会将概率向量拆解为多个标量场（Scalar Fields），例如 `class1`, `class2`... 和 `predictedClass`。


3. **属性扩展**：
* 修改/扩展该函数，将步骤三计算的 `Uncertainty` 也作为一个 Scalar Field 写入 `.vtu` 文件。


4. **最终检查**：使用 Paraview 打开生成的 `.vtu` 文件，查看点云地图，并切换 Color Map 查看“预测类别”和“不确定性”的分布。

**涉及原代码函数：** `write_geoatt_vtk3`, `write_geoatt_vtk3_prob`

---

### **总结建议**

* **Step 1** 是最难的（涉及坐标系转换和遮挡处理），也是最重要的基础。
* **Step 2** 主要是矩阵运算，验证位姿是否准确。
* **Step 3** 是您项目的核心理论部分（EvSemMap 的 "Ev"）。
* **Step 4** 是为了好看的结果展示。

您可以先从 **Step 1** 开始，确保单帧的颜色（语义）打在点云上是准确的。