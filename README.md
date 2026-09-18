# kernels：CuTe C++ kernel 练习仓库

这个仓库只放 kernel 实现与 benchmark。两个练习算子共用固定接口 `run(a, b, out)`：输出由 benchmark 预先分配，计时范围内不分配张量。

```text
kernels/
  5090/
    baseline/
      elementwise_add.py # torch.add，FP32
      gemm_bf16.py       # torch.mm，BF16
    cute/
      elementwise_add.cu # 可运行的 CuTe Layout/Tensor 入门实现
      gemm_bf16.cu       # 可运行的标量 CuTe GEMM，FP32 累加
      bindings.cpp       # PyTorch 扩展入口、输入约束与当前 CUDA stream
      launchers.h
      ops.py             # Python 调用包装
benchmarks/
  bench_elementwise.py
  bench_gemm.py
  common.py
setup.py
```

设备 kernel 使用 `.cu`，因为 PyTorch 扩展会交给 `nvcc` 编译；`.cpp` 负责主机侧绑定。`gemm_bf16.cu` 是**正确性与 CuTe 索引参考起点**，每个线程计算一个输出元素，未使用共享内存或 Tensor Core，性能预计远低于 `torch.mm`。你可以在同一文件里逐步改成分块、共享内存复用、MMA 和流水线版本。保持 BF16 输入输出、FP32 累加、任意正整数 M/N/K 的边界处理。Benchmark 会比较结果和耗时。

## 环境和编译

在该机器上复用已有的 `harrix` conda 环境、CUDA Toolkit 与 CUTLASS 头文件：

```bash
conda activate harrix
cd /home/miyaa/work/learn/kernels
export CUDA_HOME=/usr/local/cuda
export CUTLASS_ROOT=/home/miyaa/work/src/xcompute/third/cutlass
MAX_JOBS=2 python setup.py build_ext --inplace
```

只跑 torch baseline 时不用编译扩展。

## Benchmark

```bash
python -m benchmarks.bench_elementwise --numel 1048576 --impl torch
python -m benchmarks.bench_elementwise --numel 1048576 --impl all --mode graph
python -m benchmarks.bench_gemm --m 256 --n 256 --k 512 --impl torch
python -m benchmarks.bench_gemm --m 256 --n 256 --k 512 --impl all --mode graph
```

`--kernel-device` 默认选择 `kernels/5090/`，`--device` 选择 CUDA 设备（如 `cuda:0`）。以后新增设备时，按 `kernels/<设备名>/{baseline,cute}/` 放实现，用 `KERNEL_DEVICE=<设备名> python setup.py build_ext --inplace` 构建，并给 benchmark 加 `--kernel-device <设备名>`。`--impl` 可选 `torch`、`cute`、`all`；`--mode` 可选 `eager`、`graph`。CUDA Graph 模式重复执行预先捕获的调用，用于减少短 kernel 测量中的主机发射间隙；它与 eager 模式的数字应分开比较。输出包括延迟以及 elementwise 的名义有效 GB/s、GEMM 的 TFLOP/s。`--no-check` 可以跳过计时前的结果比较，调试时建议保留默认校验。GEMM 默认用 `atol=0.1, rtol=0.02` 比较 BF16 输出，因为不同 FP32 累加顺序可能舍入到相邻 BF16 值；可用 `--atol`、`--rtol` 覆盖。使用 `--help` 查看预热、重复次数等参数。

## 建议练习顺序

1. 对 elementwise add 改变线程块大小和每线程元素数，测 1K、1M、64M 元素，解释启动开销与带宽差异。
2. 对 GEMM 固定两个维度、分别扫描 M、N、K。先数输出 tile，再加入共享内存分块与复用。
3. 每次只改一个调度或数据流因素；记录 shape、预计瓶颈、正确性、延迟和资源占用。只有观察到对应瓶颈时，再尝试 Split-K、persistent 或异步流水线。

CuTe API 入门可对照 [CuTe Tensor 教程](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html) 与 [GEMM 教程](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)。
