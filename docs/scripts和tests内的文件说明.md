
`scripts/optimization/optimize_bki_params.py`：使用00004点云数据获取最佳的`ell`和`sf2`，并生成`hmc_trace_plot.png`。关于图片的详细说明见`results/hmc_trace_plot的说明.md`。
```c++
// include/semantic_bki/mapping/vbki.h 手动指定的参数 (优化前)。
// 对应 src/mapping.cpp 和 deploy_rellisv3_4_1-30.yaml
float ell = 0.3;  // 默认配置
float sf2 = 1.0;  // 硬编码默认值

// [结果] 依据均值 mean() 推荐回填 C++ (vbki.h) 的参数:
float ell = 1.2067;
float sf2 = 0.6711;

// 依据众数（峰值对应的横坐标）推荐回填 C++ (vbki.h) 的参数:
float ell = 1.1489;;
float sf2 = 0.4188;;

// 注意，在不固定随机种子的情况下，由于选取的数据不同，可能导致双峰分布。
// Data Loaded: 131072 points
// Train: 2000, Query: 200, Classes: 7
```


`tests/test_00004.py`：用于检测00004序列是否有点云的真实标签。

`tests/test_bki_math.cpp`：实现`vbki.h`的稀疏余弦核（Sparse Cosine Kernel）的逻辑。并进行一个测试。
```bash
g++ test_bki_math.cpp -o test_bki_simple
./test_bki_simple

```
实验结果
```c++
    // params
    float test_ell = 0.35f;
    float test_sf2 = 0.95f;

    // input 
    test_dists = {0.0f, 0.1f, 0.35f, 0.5f, 0.7f, 1.0f}

    // output
    // [Kernel Check]
    // Dist: 0.000000 -> Kernel: 0.950000
    // Dist: 0.100000 -> Kernel: 0.549455
    // Dist: 0.350000 -> Kernel: 0.000000
    // Dist: 0.500000 -> Kernel: 0.000000
    // Dist: 0.700000 -> Kernel: 0.000000
    // Dist: 1.000000 -> Kernel: 0.000000
```

`tests/test_consistency.py`：比较`tests/test_bki_math.cpp`的c++执行逻辑与本代码使用python实现的余弦稀疏核结果是否相同。
