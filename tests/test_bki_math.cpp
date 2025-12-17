#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

// 1. 直接复现 vbki.h 中的数学逻辑
// 对应源码中的 covSparse 函数逻辑
float kernel_function_cpp_logic(float dist, float ell, float sf2)
{
    // 源码逻辑: dist(x/ell, z/ell) => r = dist / ell
    float r = dist / ell;

    // 源码公式:
    // Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) +
    //       (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f)).matrix() * sf2;

    float pi = 3.1415926f;
    float two_pi_r = 2.0f * pi * r;

    // 分解公式
    float term1 = 2.0f + std::cos(two_pi_r);
    float term2 = 1.0f - r;
    float term3 = std::sin(two_pi_r);

    float k = ((term1 * term2 / 3.0f) + (term3 / (2.0f * pi))) * sf2;

    // 源码逻辑: if (Kxz(i,j) < 0.0) Kxz(i,j) = 0.0f;
    if (k < 0.0f)
    {
        k = 0.0f;
    }

    return k;
}

int main()
{
    // 2. 设置测试参数
    float test_ell = 0.35f;
    float test_sf2 = 0.95f;

    std::cout << "=== C++ Kernel Logic Verification (Standalone) ===" << std::endl;
    std::cout << "Params: ell=" << test_ell << ", sf2=" << test_sf2 << std::endl;

    std::vector<float> test_dists = {0.0f, 0.1f, 0.35f, 0.5f, 0.7f, 1.0f};

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n[Kernel Check]" << std::endl;
    for (float d : test_dists)
    {
        float k = kernel_function_cpp_logic(d, test_ell, test_sf2);
        std::cout << "Dist: " << d << " -> Kernel: " << k << std::endl;
    }

    return 0;
}