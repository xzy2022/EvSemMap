import numpy as np

def check_consistency():
    # 模拟参数（与您刚才 C++ 测试的一致）
    # tests/test_bki_math.cpp --> ./test_bki_simple


    # params
    # float test_ell = 0.35f;
    # float test_sf2 = 0.95f;

    # input 
    # test_dists = {0.0f, 0.1f, 0.35f, 0.5f, 0.7f, 1.0f}

    # output
    # [Kernel Check]
    # Dist: 0.000000 -> Kernel: 0.950000
    # Dist: 0.100000 -> Kernel: 0.549455
    # Dist: 0.350000 -> Kernel: 0.000000
    # Dist: 0.500000 -> Kernel: 0.000000
    # Dist: 0.700000 -> Kernel: 0.000000
    # Dist: 1.000000 -> Kernel: 0.000000
    
    ell = 0.35
    sf2 = 0.95
    dist = 0.1
    
    # 稀疏余弦核公式 (Sparse Cosine Kernel)
    r = dist / ell
    if r >= 1.0:
        k = 0.0
    else:
        pi = np.pi
        two_pi_r = 2.0 * pi * r
        term1 = 2.0 + np.cos(two_pi_r)
        term2 = 1.0 - r
        term3 = np.sin(two_pi_r)
        k = ((term1 * term2 / 3.0) + (term3 / (2.0 * pi))) * sf2
        
    print(f"Python Kernel Result: {k:.6f}")
    print(f"C++ Test Expectation: 0.549455")

check_consistency()