import os
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

# ================= 配置区域 =================
# 1. 数据路径 (指向您提取的一小段测试数据)
# 建议先用之前提到的脚本提取50-100帧保存为 'mini_batch_data.npy'
DATA_PATH = '/home/xzy/datasets/Rellis-3D/00004/mini_sample_00004.npy' 

# 2. C++ 逻辑对齐 (请务必核对 vbki.h 中的公式)
# 假设 C++ 中的 distAdaptive 是高斯核: k(d) = sf2 * exp(-0.5 * d^2 / ell^2)
# 如果您的 C++ 是其他公式（如多项式），请在下方 `bki_kernel_tensor` 中修改
CUTOFF_RATIO = 3.0  # 稀疏截断: dist > 3*ell 时 k=0
NUM_CLASSES = 20    # RELLIS-3D 类别数

# ================= 辅助函数 =================

def load_data(path):
    """加载预处理好的小批量数据"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"请先生成测试数据: {path}")
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
    # 为了简化，我们从所有点中随机切分
    N = len(points)
    idx = np.random.permutation(N)
    
    # 选一部分作为"已知地图点"(Train)，一部分作为"待预测点"(Query)
    # 实际 BKI 中，Train 是历史帧，Query 是当前帧。这里简化为空间插值问题。
    n_train = min(2000, N - n_query) # 限制训练点数量以保证速度
    
    X_train = points[idx[:n_train]]
    y_train = labels[idx[:n_train]]
    
    X_query = points[idx[n_train:n_train+n_query]]
    y_query_gt = labels[idx[n_train:n_train+n_query]] # Ground Truth for Likelihood
    
    # 将 y_train 转为 One-Hot (N_train, n_classes)
    y_train_oh = np.zeros((n_train, NUM_CLASSES))
    y_train_oh[np.arange(n_train), y_train] = 1.0

    print(f"Building Model | Train: {n_train}, Query: {n_query}")
    
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
        # 使用 switch 函数保证可导性，或者用 soft 截断
        # 这里为了 HMC 稳定性，暂时忽略硬截断，或者用极小值代替
        # 如果必须硬截断：
        # mask = pm.math.switch(dist_matrix > (CUTOFF_RATIO * ell), 0.0, 1.0)
        # kernel_val = kernel_val * mask
        
        # --- C. BKI 更新 (Bayesian Update) ---
        # Alphas = Sum( Kernel * Labels ) + Prior
        # (N_train, N_query).T @ (N_train, n_classes) -> (N_query, n_classes)
        # 这是一个加权求和过程
        
        # PyTensor 的 dot 操作
        # kernel_val shape: (N_train, N_query)
        # y_train_oh shape: (N_train, n_classes)
        # output alphas: (N_query, n_classes)
        
        alphas_pred = pt.dot(kernel_val.T, y_train_oh)
        
        # 添加极小值防止除零 (Dirichlet Prior typically 0 or 0.001)
        alphas_pred = alphas_pred + 1e-3
        
        # --- D. 定义 Likelihood ---
        # 我们希望预测的 alphas 能最大化真实标签 y_query_gt 的概率
        # BKI 输出的是分类分布的参数。
        # Prob(class_i) = alpha_i / sum(alphas)
        
        probs = alphas_pred / pt.sum(alphas_pred, axis=1, keepdims=True)
        
        # 观测值：真实的类别 ID
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
        pm.plot_trace(trace)
        plt.savefig("hmc_trace_plot.png")
        print("Trace plot saved to hmc_trace_plot.png")
        
        # 提取建议参数
        best_ell = trace.posterior["ell"].mean().item()
        best_sf2 = trace.posterior["sf2"].mean().item()
        
        print(f"\n推荐回填 C++ 的参数:")
        print(f"float ell = {best_ell:.4f};")
        print(f"float sf2 = {best_sf2:.4f};")
        
    except Exception as e:
        print(f"Error: {e}")
        print("提示：请先确保 'mini_sample_00004.npy' 存在 (参考上一步的脚本生成)")