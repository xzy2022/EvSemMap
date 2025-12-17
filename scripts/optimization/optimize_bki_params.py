import os
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy.stats as stats

# ================= 配置区域 =================
# 1. 数据路径
DATA_PATH = '/home/xzy/datasets/Rellis-3D/00004/mini_sample_00004.npy' 
np.random.seed(42) 

# 2. 优化原理
# 利用贝叶斯推断（HMC）结合稀疏余弦核函数，从真实点云数据中自动学习出
# 最优的核函数长度尺度（ell）和信号方差（sf2），以最大化地图构建的语义预测准确性。

# ================= 辅助函数 =================

def load_data(path):
    """加载预处理好的小批量数据"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到数据文件: {path}\n请先检查路径或运行提取脚本生成 .npy 文件。")
    data = np.load(path, allow_pickle=True).item()
    return data['points'], data['labels']

def create_distance_matrix(X_train, X_query):
    """预计算距离矩阵 (N_train, N_query)"""
    return cdist(X_train, X_query, metric='euclidean')

def get_mode(trace_data):
    """通过 KDE 计算分布的峰值（众数/Mode）"""
    data = trace_data.values.flatten()
    kde = stats.gaussian_kde(data)
    x_grid = np.linspace(data.min(), data.max(), 1000)
    density = kde(x_grid)
    peak_idx = np.argmax(density)
    return x_grid[peak_idx]

# ================= PyMC 模型 =================

def run_hmc_optimization(points, labels, n_query=500):
    """
    运行 HMC 搜索最优 ell 和 sf2
    """
    N = len(points)
    idx = np.random.permutation(N)
    
    # 限制训练点数量以保证速度
    n_train = min(2000, N - n_query)
    
    X_train = points[idx[:n_train]]
    y_train_raw = labels[idx[:n_train]]
    
    X_query = points[idx[n_train:n_train+n_query]]
    y_query_raw = labels[idx[n_train:n_train+n_query]] 
    
    # === 标签映射 (Label Mapping) ===
    unique_labels = np.unique(np.concatenate((y_train_raw, y_query_raw)))
    n_active_classes = len(unique_labels)
    
    print(f"检测到 {n_active_classes} 个有效类别，正在建立映射...")
    label_map = {original: mapped for mapped, original in enumerate(unique_labels)}
    
    y_train = np.array([label_map[y] for y in y_train_raw])
    y_query_gt = np.array([label_map[y] for y in y_query_raw])
    
    print(f"Building Model | Train: {n_train}, Query: {n_query}, Classes: {n_active_classes}")
    
    y_train_oh = np.zeros((n_train, n_active_classes))
    y_train_oh[np.arange(n_train), y_train] = 1.0

    # 预计算距离矩阵
    dist_matrix = create_distance_matrix(X_train, X_query)
    
    with pm.Model() as bki_model:
        # --- A. 定义随机变量 (Priors) ---
        # ell: 长度尺度，影响范围。根据 vbki.h 默认设定，我们给一个宽泛的 Gamma 分布
        ell = pm.Gamma("ell", alpha=2.0, beta=10.0) 
        
        # sf2: 信号方差，置信度权重。
        sf2 = pm.Gamma("sf2", alpha=2.0, beta=2.0)
        
        # --- B. 定义 BKI 核函数 (复现 C++ vbki.h 逻辑: Sparse Cosine Kernel) ---
        # 1. 计算归一化距离 r = d / ell
        r = dist_matrix / ell
        
        # 2. 定义常量
        pi = np.pi
        two_pi_r = 2.0 * pi * r
        
        # 3. 计算 Sparse Kernel 公式
        # k = ( (2+cos(2pir))*(1-r)/3 + sin(2pir)/2pi ) * sf2
        term1 = 2.0 + pm.math.cos(two_pi_r)
        term2 = 1.0 - r
        term3 = pm.math.sin(two_pi_r)
        
        k_raw = ( (term1 * term2 / 3.0) + (term3 / (2.0 * pi)) ) * sf2
        
        # 4. 截断 (C++逻辑: 小于0则置为0)
        kernel_val = pm.math.switch(k_raw < 0, 0.0, k_raw)
        
        # --- C. BKI 更新 (Bayesian Update) ---
        # Alphas = Sum( Kernel * Labels ) + Prior
        alphas_pred = pt.dot(kernel_val.T, y_train_oh)
        
        # 添加极小值防止除零
        alphas_pred = alphas_pred + 1e-3
        
        # --- D. 定义 Likelihood ---
        # BKI 输出的是 Dirichlet 分布参数，其期望概率为 alpha_i / sum(alphas)
        probs = alphas_pred / pt.sum(alphas_pred, axis=1, keepdims=True)
        
        # 观测值
        obs = pm.Categorical("obs", p=probs, observed=y_query_gt)
        
        # --- E. 采样 ---
        print("Starting HMC Sampling...")
        trace = pm.sample(draws=1000, tune=1000, chains=2, target_accept=0.9)
        
        return trace

# ================= 主程序 =================

if __name__ == "__main__":
    try:
        # 1. 加载数据
        points, labels = load_data(DATA_PATH)
        print(f"Data Loaded: {len(points)} points")
        
        # 2. 运行优化
        trace = run_hmc_optimization(points, labels, n_query=200)
        
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
        
        # 4. 提取建议参数 (Mean vs Mode)
        ell_samples = trace.posterior["ell"]
        sf2_samples = trace.posterior["sf2"]
        
        # 计算均值 (Mean) - 统计学期望
        mean_ell = ell_samples.mean().item()
        mean_sf2 = sf2_samples.mean().item()
        
        # 计算众数 (Mode/Peak) - 概率密度最大的点 (最推荐)
        mode_ell = get_mode(ell_samples)
        mode_sf2 = get_mode(sf2_samples)
        
        print(f"\n[结果比较]")
        print(f"Mean (均值) -> ell: {mean_ell:.4f}, sf2: {mean_sf2:.4f}")
        print(f"Mode (众数) -> ell: {mode_ell:.4f}, sf2: {mode_sf2:.4f}")
        
        print(f"\n[最终推荐回填 C++ (vbki.h) 的参数] (建议优先使用 Mode)")
        print(f"float ell = {mode_ell:.4f};")
        print(f"float sf2 = {mode_sf2:.4f};")
        
    except Exception as e:
        print(f"\nCritical Error: {e}")