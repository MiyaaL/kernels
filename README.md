# kernels��CuTe C++ kernel ��ϰ�ֿ�

����ֿ�ֻ�� kernel ʵ���� benchmark��������ϰ���ӹ��ù̶��ӿ� `run(a, b, out)`������� benchmark Ԥ�ȷ��䣬��ʱ��Χ�ڲ�����������

```text
kernels/
  5090/
    baseline/
      elementwise_add.py # torch.add��FP32
      gemm_bf16.py       # torch.mm��BF16
    cute/
      elementwise_add.cu # �����е� CuTe Layout/Tensor ����ʵ��
      gemm_bf16.cu       # �����еı��� CuTe GEMM��FP32 �ۼ�
      bindings.cpp       # PyTorch ��չ��ڡ�����Լ���뵱ǰ CUDA stream
      launchers.h
      ops.py             # Python ���ð�װ
benchmarks/
  bench_elementwise.py
  bench_gemm.py
  common.py
setup.py
```

�豸 kernel ʹ�� `.cu`����Ϊ PyTorch ��չ�ύ�� `nvcc` ���룻`.cpp` ����������󶨡�`gemm_bf16.cu` ��**��ȷ���� CuTe �����ο����**��ÿ���̼߳���һ�����Ԫ�أ�δʹ�ù����ڴ�� Tensor Core������Ԥ��Զ���� `torch.mm`���������ͬһ�ļ����𲽸ĳɷֿ顢�����ڴ渴�á�MMA ����ˮ�߰汾������ BF16 ���������FP32 �ۼӡ����������� M/N/K �ı߽紦����Benchmark ��ȽϽ���ͺ�ʱ��

## �����ͱ���

�ڸû����ϸ������е� `harrix` conda ������CUDA Toolkit �� CUTLASS ͷ�ļ���

```bash
conda activate harrix
cd /home/miyaa/work/learn/kernels
export CUDA_HOME=/usr/local/cuda
export CUTLASS_ROOT=/home/miyaa/work/src/xcompute/third/cutlass
MAX_JOBS=2 python setup.py build_ext --inplace
```

ֻ�� torch baseline ʱ���ñ�����չ��

## Benchmark

```bash
python -m benchmarks.bench_elementwise --numel 1048576 --impl torch
python -m benchmarks.bench_elementwise --numel 1048576 --impl all --mode graph
python -m benchmarks.bench_gemm --m 256 --n 256 --k 512 --impl torch
python -m benchmarks.bench_gemm --m 256 --n 256 --k 512 --impl all --mode graph
```

`--kernel-device` Ĭ��ѡ�� `kernels/5090/`��`--device` ѡ�� CUDA �豸���� `cuda:0`�����Ժ������豸ʱ���� `kernels/<�豸��>/{baseline,cute}/` ��ʵ�֣��� `KERNEL_DEVICE=<�豸��> python setup.py build_ext --inplace` ���������� benchmark �� `--kernel-device <�豸��>`��`--impl` ��ѡ `torch`��`cute`��`all`��`--mode` ��ѡ `eager`��`graph`��CUDA Graph ģʽ�ظ�ִ��Ԥ�Ȳ���ĵ��ã����ڼ��ٶ� kernel �����е����������϶������ eager ģʽ������Ӧ�ֿ��Ƚϡ���������ӳ��Լ� elementwise ��������Ч GB/s��GEMM �� TFLOP/s��`--no-check` ����������ʱǰ�Ľ���Ƚϣ�����ʱ���鱣��Ĭ��У�顣GEMM Ĭ���� `atol=0.1, rtol=0.02` �Ƚ� BF16 �������Ϊ��ͬ FP32 �ۼ�˳��������뵽���� BF16 ֵ������ `--atol`��`--rtol` ���ǡ�ʹ�� `--help` �鿴Ԥ�ȡ��ظ������Ȳ�����

## ������ϰ˳��

1. �� elementwise add �ı��߳̿��С��ÿ�߳�Ԫ�������� 1K��1M��64M Ԫ�أ���������������������졣
2. �� GEMM �̶�����ά�ȡ��ֱ�ɨ�� M��N��K��������� tile���ټ��빲���ڴ�ֿ��븴�á�
3. ÿ��ֻ��һ�����Ȼ����������أ���¼ shape��Ԥ��ƿ������ȷ�ԡ��ӳٺ���Դռ�á�ֻ�й۲쵽��Ӧƿ��ʱ���ٳ��� Split-K��persistent ���첽��ˮ�ߡ�

CuTe API ���ſɶ��� [CuTe Tensor �̳�](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/03_tensor.html) �� [GEMM �̳�](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0x_gemm_tutorial.html)��
