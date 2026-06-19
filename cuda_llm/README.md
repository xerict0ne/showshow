# CUDA-LLM 异构计算实验项目

> **课程作业项目：异构计算技术与大语言模型推理优化实验**  
> 本项目面向《异构计算》课程实验要求，基于 **NVIDIA CUDA + PyTorch** 构建典型 LLM 计算模块的性能分析与优化实验。项目重点不是完整训练大模型，而是围绕大语言模型中的核心计算瓶颈，如矩阵乘法、Attention、KV Cache、混合精度和 Profiling，完成一套可运行、可分析、可展示的异构计算实验流程。

---

## 一、项目背景

大语言模型（Large Language Model, LLM）的训练与推理过程需要大量计算资源，其中最核心的计算包括：

- 大规模矩阵乘法（GEMM / MatMul）
- Transformer Attention 计算
- 长上下文推理中的 KV Cache 存储
- FP32 / FP16 / INT8 等不同精度计算
- GPU kernel 执行与性能瓶颈分析

这些计算任务具有高度并行性，非常适合使用 GPU 进行加速。因此，本项目选择 **NVIDIA CUDA 平台** 作为异构计算实验环境，通过 PyTorch CUDA backend 调用底层 GPU 加速能力，分析 LLM-style workload 在 CPU 与 GPU、不同精度、不同 Attention 实现下的性能差异。

---

## 二、课程要求对应关系

本项目对应异构计算课程实验中“采用 NVIDIA CUDA 完成典型 LLM 系统优化方案设计”的要求，具体对应如下：

| 课程要求 | 本项目对应内容 |
|---|---|
| 采用 NVIDIA CUDA 完成 LLM 系统优化设计 | 使用 PyTorch CUDA backend 运行 LLM-style workload |
| 结合 NVIDIA 某款板卡 | 面向 RTX 30 系列 Laptop GPU，例如 RTX 3060 / RTX 3070 |
| 体现底层算子优化 | 对比 CPU/GPU GEMM、Naive Attention 与 SDPA |
| 体现混合精度量化 | 对比 FP32、FP16、INT8-sim 和 AMP |
| 体现框架与架构 | PyTorch → CUDA Runtime → cuBLAS/cuDNN/SDPA → NVIDIA GPU |
| 体现库与组件 | torch.cuda、cuBLAS、SDPA、AMP、matplotlib、pandas |
| 体现调优工具 | 使用 PyTorch Profiler 分析 CUDA 时间占比 |
| 定位瓶颈并优化 | 通过 profiling 发现 GEMM 与 Attention 是主要耗时模块 |

---

## 三、实验硬件环境

推荐实验平台如下：

```text
设备：Lenovo Legion 拯救者 2021 高配版
GPU：NVIDIA GeForce RTX 3060 Laptop GPU
GPU 架构：NVIDIA Ampere
CUDA Capability：8.6
显存：6GB / 8GB
CPU：Intel i7 / AMD Ryzen 7 移动端处理器
内存：16GB / 32GB
操作系统：Windows + WSL2 Ubuntu / Ubuntu 20.04 / Ubuntu 22.04
```

说明：

- 如果本机有 NVIDIA GPU 和 CUDA 环境，可以直接运行 GPU 实验。
- 如果没有 CUDA，代码会尽可能使用 CPU fallback 验证流程，但 CPU fallback 不能代表真实 CUDA 加速效果。
- 本项目主要面向课程实验展示，重点是分析 LLM 典型模块在异构计算环境中的性能特征。

---

## 四、软件环境

```text
Python 3.9 - 3.11
PyTorch 2.x
CUDA 11.8 / 12.x
numpy
pandas
matplotlib
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 五、项目目录结构

```text
cuda_llm_full_experiments_v5_chinese_readme/
├── README.md
├── requirements.txt
├── scripts/
│   ├── common.py
│   ├── 00_env_check.py
│   ├── 01_matmul.py
│   ├── 02_precision.py
│   ├── 03_kv_cache.py
│   ├── 04_attention.py
│   ├── 05_amp.py
│   ├── 06_profile.py
│   ├── 07_plot.py
│   └── run_all.py
├── results/
│   ├── env_info.txt
│   ├── matmul_results.csv
│   ├── precision_results.csv
│   ├── kv_cache_results.csv
│   ├── attention_results.csv
│   ├── amp_results.csv
│   ├── profile_results.csv
│   └── profile_table.txt
├── figures/
│   ├── 01_cpu_gpu_gemm_speedup.png
│   ├── 02_precision_latency_comparison.png
│   ├── 03_kv_cache_growth.png
│   ├── 04_attention_naive_vs_sdpa.png
│   ├── 05_amp_training_speed_memory.png
│   └── 06_profiling_breakdown.png
└── report/
    ├── report_outline.md
    └── results_analysis.md
```

---

## 六、实验内容说明

### 1. CUDA 环境检查：`00_env_check.py`

该实验用于检查当前机器是否具备 CUDA 运行环境，并输出 PyTorch 版本、CUDA 版本、GPU 名称、计算能力和显存大小。

核心代码：

```python
torch.cuda.is_available()
torch.cuda.get_device_name(0)
torch.cuda.get_device_properties(0)
```

实验意义：

> 确认实验是否真正运行在 NVIDIA GPU 异构计算环境中，是后续 CUDA 实验的基础。

---

### 2. CPU vs GPU 矩阵乘法实验：`01_matmul.py`

该实验对比 CPU 和 GPU 在不同规模矩阵乘法上的耗时，并计算 GPU 加速比。

核心代码：

```python
a = torch.randn(n, n, device=device)
b = torch.randn(n, n, device=device)
torch.matmul(a, b)
```

输出文件：

```text
results/matmul_results.csv
figures/01_cpu_gpu_gemm_speedup.png
```

实验意义：

> GEMM 是 Transformer 中最核心的底层计算之一。本实验验证了 GPU 在大规模矩阵计算中的并行加速能力。

---

### 3. 精度与量化实验：`02_precision.py`

该实验对比 FP32、FP16 和 INT8-sim 三种计算模式下的相对延迟。

核心代码：

```python
torch.float32
torch.float16
q = torch.clamp((t / scale).round(), -128, 127).to(torch.int8)
```

输出文件：

```text
results/precision_results.csv
figures/02_precision_latency_comparison.png
```

说明：

- FP32 作为基准。
- FP16 可利用 NVIDIA Tensor Core 加速。
- INT8-sim 用于模拟量化推理过程，不等同于 TensorRT INT8 kernel。

实验意义：

> 混合精度和量化是 LLM 推理优化中的重要方法，可以降低延迟和显存占用。

---

### 4. KV Cache 显存分析：`03_kv_cache.py`

该实验估算不同上下文长度下 KV Cache 的显存占用。

核心代码：

```python
total_bytes = batch_size * seq_len * num_layers * num_heads * head_dim * 2 * bytes_per_value
```

输出文件：

```text
results/kv_cache_results.csv
figures/03_kv_cache_growth.png
```

实验意义：

> KV Cache 可以避免重复计算历史 token 的 Key 和 Value，但在长上下文推理中会造成显存线性增长，是 LLM 推理的重要 memory bottleneck。

---

### 5. Attention 优化实验：`04_attention.py`

该实验对比 Naive Attention 和 PyTorch SDPA 的执行时间。

Naive Attention：

```python
scores = torch.matmul(q, k.transpose(-2, -1)) * scale
attn = torch.softmax(scores, dim=-1)
out = torch.matmul(attn, v)
```

SDPA：

```python
F.scaled_dot_product_attention(q, k, v)
```

输出文件：

```text
results/attention_results.csv
figures/04_attention_naive_vs_sdpa.png
```

实验意义：

> SDPA 通过 fused attention kernel 减少中间结果和显存访问，可以提升 Attention 计算效率，体现了典型 LLM 算子优化思想。

---

### 6. AMP 混合精度训练实验：`05_amp.py`

该实验分别运行 FP32 训练和 AMP 混合精度训练，并记录训练耗时和 GPU 峰值显存。

核心代码：

```python
with torch.amp.autocast("cuda"):
    out = model(x)
    loss = loss_fn(out, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

记录时间：

```python
start = time.time()
epoch_time = time.time() - start
```

记录显存：

```python
torch.cuda.max_memory_allocated()
```

输出文件：

```text
results/amp_results.csv
figures/05_amp_training_speed_memory.png
```

实验意义：

> AMP 可以在保证数值稳定性的同时使用 FP16 加速部分计算，从而降低训练时间和显存占用。

---

### 7. Profiling 性能分析实验：`06_profile.py`

该实验使用 PyTorch Profiler 对 LLM-style workload 进行 CUDA 时间分析。

核心代码：

```python
with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
    torch.matmul(a, b)
    F.scaled_dot_product_attention(q, k, v)
```

输出文件：

```text
results/profile_results.csv
results/profile_table.txt
figures/06_profiling_breakdown.png
```

实验意义：

> Profiling 用于定位系统瓶颈。结果表明，GEMM / matmul 和 Attention 是 LLM-style workload 中最主要的 CUDA 时间开销。

注意：

> 当前 profiling 分析的是优化后的 CUDA 执行路径，即 FP16 + SDPA 下的 workload，用于观察优化后系统中剩余的主要瓶颈。

---

### 8. 统一画图脚本：`07_plot.py`

该脚本读取 `results/` 下的 CSV 文件，并生成 `figures/` 下的六张图。

核心代码：

```python
df = pd.read_csv(RESULTS_DIR / "matmul_results.csv")
plt.savefig(FIGURES_DIR / "01_cpu_gpu_gemm_speedup.png")
```

实验意义：

> 实现实验数据采集和结果可视化的分离，使项目结构更加清晰，也便于报告和 PPT 展示。

---

## 七、运行方式

### 一键运行全部实验

```bash
python scripts/run_all.py
```

运行后会自动生成：

```text
results/*.csv
figures/*.png
```

### 单独运行某个实验

```bash
python scripts/01_matmul.py
python scripts/05_amp.py
python scripts/06_profile.py
python scripts/07_plot.py
```

---

## 八、结果说明

本项目已经包含一组代表性完成结果，位于：

```text
results/
figures/
```

这些结果用于展示完整实验流程和性能趋势。如果需要严格实测结果，应在目标 RTX Laptop GPU 环境下重新运行：

```bash
python scripts/run_all.py
```

重新生成后的 CSV 和图片会覆盖原有结果。

---

## 九、实验结论

通过本项目实验可以得到以下结论：

1. GPU 在大规模矩阵乘法中相比 CPU 具有明显加速优势。
2. FP16 和 INT8-sim 低精度计算可以降低推理延迟。
3. KV Cache 显存占用随上下文长度线性增长，是长上下文推理的重要瓶颈。
4. PyTorch SDPA 相比 Naive Attention 具有更低执行时间。
5. AMP 混合精度训练可以同时降低训练耗时和 GPU 显存占用。
6. Profiling 结果表明，GEMM / matmul 和 Attention 是 LLM-style workload 的主要性能瓶颈。

---

## 十、项目定位

本项目不是完整大模型训练项目，而是一个面向异构计算课程的 **LLM 典型计算模块优化实验**。项目通过 CUDA、PyTorch、AMP、SDPA 和 profiling 工具，展示了 CPU-GPU 异构计算在大语言模型系统中的应用方式。

一句话总结：

> 本项目基于 NVIDIA CUDA 平台，对 LLM-style workload 中的矩阵计算、Attention、KV Cache、混合精度和 profiling 进行实验分析，体现了异构计算在大模型系统优化中的实际作用。
