// test_bki_math.cpp
#include <iostream>
#include <vector>
#include <iomanip>
#include "semantic_bki/mapping/vbki.h"
#include "semantic_bki/common/vpoint3f.h"

int main()
{
    // 1. 设置与 Python 完全一致的参数
    float test_ell = 0.3f;
    float test_sf2 = 1.0f;

    vsemantic_bki::VBKI bki(test_ell, test_sf2);

    std::cout << "=== C++ BKI Math Verification ===" << std::endl;
    std::cout << "Params: ell=" << test_ell << ", sf2=" << test_sf2 << std::endl;

    // 2. 测试 Kernel 函数 (covSparse / distAdaptive)
    std::vector<float> test_dists = {0.0f, 0.1f, 0.3f, 0.5f, 0.9f, 1.0f};

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n[Kernel Check]" << std::endl;
    for (float d : test_dists)
    {
        float k = bki.covSparse(d);
        std::cout << "Dist: " << d << " -> Kernel: " << k << std::endl;
    }

    // 3. 验证这一步的输出
    // 将这些输出值与 Python 中运行 sparse_kernel_numpy(d, ...) 的结果对比
    // 如果两边一致，说明你的概率模型建立是准确的。

    return 0;
}