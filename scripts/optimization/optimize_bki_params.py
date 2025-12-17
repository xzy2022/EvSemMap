import os
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

# ================= 配置区域 =================
# 1. 数据路径 (指向您提取的一小段测试数据)
# [修改] 改为相对路径或您指定的绝对路径，确保文件存在
DATA_PATH = '/home/xzy/datasets/Rellis-3D/00004/mini_sample_00004.npy' 

# 2. C++ 逻辑对齐 (请务必核对 vbki.h 中的公式)
# 假设 C++ 中的 distAdaptive 是高斯核: k(d) = sf2 * exp(-0.5 * d^2 / ell^2)
# 如果您的 C++ 是其他公式（如多项式），请在下方 `bki_kernel_tensor` 中修改
CUTOFF_RATIO = 3.0  # 稀疏截断: dist > 3*ell 时 k=0
# [修改] 实际类别数由数据动态决定，防止 hardcode 导致的越界
NUM_CLASSES = 20    # RELLIS-3D 类别数

# ================= 辅助函数 =================

def load_data(path):
    """加载预处理好的小批量数据"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到数据文件: {path}\n请先检查路径或运行提取脚本生成 .npy 文件。")
    data = np.load(path, allow_pickle=True).item()
    return data['points'], data['labels']

def create_distance_matrix(X_train, X_query):
    """预计算距离矩阵 (N_train, N_query)"""
    # 注意：如果数据量太大，这里会爆内存。建议 N_train < 10000, N_query < 1000
    dists = cdist(X_train, X_query, metric='euclidean')
    return dists

# ================= PyMC 模型 =================

def run_hmc_optimization(points, labels, n_query=500):
    """
    运行 HMC 搜索最优 ell 和 sf2
    points: (N, 3) 训练点 (作为地图先验)
    labels: (N,) 训练点标签
    n_query: 从 points 中随机选一部分作为 query 点来验证预测能力
    """
    
    # 1. 准备数据：构建 Training Set (地图) 和 Query Set (当前观测)
    N = len(points)
    # 打乱数据
    idx = np.random.permutation(N)
    
    # 选一部分作为"已知地图点"(Train)，一部分作为"待预测点"(Query)
    # 实际 BKI 中，Train 是历史帧，Query 是当前帧。这里简化为空间插值问题。
    n_train = min(2000, N - n_query) # 限制训练点数量以保证速度
    
    X_train = points[idx[:n_train]]
    y_train_raw = labels[idx[:n_train]]
    
    X_query = points[idx[n_train:n_train+n_query]]
    y_query_raw = labels[idx[n_train:n_train+n_query]] 
    
    # [核心新增] === 标签映射 (Label Mapping) ===
    # 原因：RELLIS-3D 原始标签是不连续的(如 0, 1, 3, 4...)。
    # 如果直接用作 One-Hot 的索引，会导致 IndexError 或 矩阵维度过大。
    # 这里将出现的标签重新映射为 0, 1, 2... 紧凑格式。
    unique_labels = np.unique(np.concatenate((y_train_raw, y_query_raw)))
    n_active_classes = len(unique_labels)
    
    print(f"检测到 {n_active_classes} 个有效类别，正在建立映射...")
    label_map = {original: mapped for mapped, original in enumerate(unique_labels)}
    
    # 应用映射
    y_train = np.array([label_map[y] for y in y_train_raw])
    y_query_gt = np.array([label_map[y] for y in y_query_raw])
    
    print(f"Building Model | Train: {n_train}, Query: {n_query}, Classes: {n_active_classes}")
    
    # 将 y_train 转为 One-Hot (N_train, n_active_classes)
    y_train_oh = np.zeros((n_train, n_active_classes))
    y_train_oh[np.arange(n_train), y_train] = 1.0

    # 2. 预计算距离矩阵 (Constant in the model)
    dist_matrix = create_distance_matrix(X_train, X_query)
    
    with pm.Model() as bki_model:
        # --- A. 定义随机变量 (Priors) ---
        # ell (长度尺度): 必须 > 0。根据 vbki.h 默认 0.1-0.5，我们给一个 Gamma 分布
        ell = pm.Gamma("ell", alpha=2.0, beta=10.0) # 均值 0.2
        
        # sf2 (信号方差): 必须 > 0。默认 1.0
        sf2 = pm.Gamma("sf2", alpha=2.0, beta=2.0)  # 均值 1.0
        
        # --- B. 定义 BKI 核函数 (Deterministic Logic) ---
        # 对应 C++: covSparse / distAdaptive
        # k = sf2 * exp( -0.5 * dist^2 / ell^2 )
        # 注意：需要处理截断 (Sparse Cutoff)
        
        # 1. 计算高斯部分
        kernel_val = sf2 * pm.math.exp(-0.5 * (dist_matrix**2) / (ell**2))
        
        # 2. 应用稀疏截断 (模拟 C++ if dist > 3*ell return 0)
        # 为了 HMC 梯度连续性，这里暂不使用硬截断(switch)，依靠高斯函数的快速衰减
        
        # --- C. BKI 更新 (Bayesian Update) ---
        # Alphas = Sum( Kernel * Labels ) + Prior
        # (N_train, N_query).T @ (N_train, n_classes) -> (N_query, n_classes)
        # 这是一个加权求和过程
        
        # PyTensor 的 dot 操作
        # kernel_val shape: (N_train, N_query)
        # y_train_oh shape: (N_train, n_classes)
        # output alphas: (N_query, n_classes)
        
        # PyTensor 的 dot 操作
        alphas_pred = pt.dot(kernel_val.T, y_train_oh)
        
        # 添加极小值防止除零 (Dirichlet Prior typically 0 or 0.001)
        alphas_pred = alphas_pred + 1e-3
        
        # --- D. 定义 Likelihood ---
        # 我们希望预测的 alphas 能最大化真实标签 y_query_gt 的概率
        # BKI 输出的是 Dirichlet 分布参数，其期望概率为 alpha_i / sum(alphas)
        
        probs = alphas_pred / pt.sum(alphas_pred, axis=1, keepdims=True)
        
        # 观测值：真实的类别 ID (Mapped IDs)
        obs = pm.Categorical("obs", p=probs, observed=y_query_gt)
        
        # --- E. 采样 ---
        print("Starting HMC Sampling...")
        trace = pm.sample(draws=1000, tune=1000, chains=2, target_accept=0.9)
        
        return trace, bki_model

# ================= 主程序 =================

if __name__ == "__main__":
    # 1. 加载数据
    try:
        points, labels = load_data(DATA_PATH)
        print(f"Data Loaded: {len(points)} points")
        
        # 2. 运行优化
        trace, model = run_hmc_optimization(points, labels, n_query=200)
        
        # 3. 分析结果
        print("\n=== Optimization Results ===")
        pm.summary(trace)
        
        # 绘制后验分布
        # ------------------------------------------------------------------
        # 图表解释：
        # 这是一个 2x2 的图表，行代表参数 (ell, sf2)，列代表分析维度。
        #
        # 【左列 - 密度图 (KDE)】：代表参数的"最终推荐分布"。
        #   - X轴：参数的值。
        #   - 曲线最高点：概率最大的值（即最推荐回填到 C++ 的值）。
        #   - 虚线/实线：代表不同的采样链(Chains)。如果重合得好，说明结果可信。
        #
        # 【右列 - 轨迹图 (Trace)】：代表采样的"寻找过程"。
        #   - X轴：迭代次数。
        #   - Y轴：尝试的值。
        #   - 理想状态：应该像一条水平的"毛毛虫"（Fuzzy Caterpillar），
        #     上下快速震荡，没有明显的上升/下降趋势。这代表算法已收敛。
        #      
        # 实线与虚线：完全独立等价的两次探索， chains=2 设置
        #
        # 参数含义：
        # ell (Length Scale - 长度尺度)：
        #   含义：“一个点的影响范围有多大？”
        #   直观理解：如果 ell = 0.5m，意味着当你观测到一个点是“草地”时，它周围 0.5 米范围内的空白区域也大概率是“草地”。
        #   数值影响：
        #       ell 太大：地图会变得过于平滑，细节丢失（墙角变圆）。
        #       ell 太小：地图会变得像“麻子”一样，点与点之间没有联系，全是孤立的噪点。
        #   你的图：峰值在 0.45 左右，说明对于 RELLIS-3D 数据，0.45米是一个既能保持连贯性又能保留细节的最佳半径。
        #
        # sf2 (Signal Variance - 信号方差)：
        #   含义：“当前这次观测有多‘强’？” 或者 “置信度的积累速度”。
        #   直观理解：在 BKI 的连续计数模型（Continuous Counting）中，它相当于是一个学习率或权重乘子。
        #   数值影响：
        #   sf2 很大：观测一次就非常确信（地图更新很快，但也容易被噪声带偏）。
        #   sf2 很小：需要反复观测很多次，该区域的状态才会从“未知”变成“确信”（抗噪能力强，但建图慢，有拖影）。
        # ------------------------------------------------------------------
        try:
            pm.plot_trace(trace)
            plt.savefig("hmc_trace_plot.png")
            print("Trace plot saved to hmc_trace_plot.png")
        except Exception as e:
            print(f"绘图警告: {e}")
        
        # 提取建议参数
        best_ell = trace.posterior["ell"].mean().item()
        best_sf2 = trace.posterior["sf2"].mean().item()
        
        print(f"\n[结果] 推荐回填 C++ (vbki.h) 的参数:")
        print(f"float ell = {best_ell:.4f};")
        print(f"float sf2 = {best_sf2:.4f};")
        
    except Exception as e:
        print(f"\nCritical Error: {e}")