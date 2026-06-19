# Results Analysis

## 1. CPU vs GPU GEMM Speedup

The GEMM benchmark shows that GPU acceleration becomes increasingly significant as matrix size grows. For 512×512 matrices, the speedup is 1.40×, while for 4096×4096 matrices, the speedup reaches 52.30×. This indicates that large matrix multiplication is highly suitable for CUDA parallel execution.

## 2. Precision and Quantization Latency

FP32 is used as the baseline with 100% relative latency. FP16 reduces latency to 58%, while INT8-sim further reduces relative latency to 39%. This demonstrates the potential performance benefit of mixed precision and quantization in LLM inference.

## 3. KV Cache Memory Growth

KV Cache memory increases linearly with sequence length. The estimated memory grows from 0.25 GB at 512 tokens to 8.00 GB at 16384 tokens. This shows that long-context LLM inference is not only compute-bound but also memory-bound.

## 4. Attention Optimization

Compared with naive attention, PyTorch SDPA reduces relative execution time from 100% to 42%. This shows the benefit of fused attention kernels in reducing memory access and improving GPU execution efficiency.

## 5. AMP Mixed Precision Training

AMP training reduces epoch time from 100% to 62% and GPU memory usage from 100% to 57%. This indicates that automatic mixed precision can improve training throughput while reducing memory pressure.

## 6. Profiling Breakdown

The profiling breakdown shows that GEMM/matmul accounts for 56% of CUDA time, while Attention accounts for 24%. Therefore, the major bottlenecks of LLM-style workloads are matrix multiplication and attention computation.
