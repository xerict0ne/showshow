# Report Outline

## 1. Experimental Environment

- Lenovo Legion 2021 high-end model
- NVIDIA RTX 3060 / RTX 3070 Laptop GPU
- CUDA + PyTorch 2.x
- Python 3.9-3.11

## 2. Experiment Modules

### 2.1 CUDA Environment Check

Purpose: verify CUDA availability and GPU information.

### 2.2 CPU vs GPU GEMM Benchmark

Purpose: compare CPU and GPU performance on matrix multiplication.

### 2.3 Precision and Quantization

Purpose: compare FP32, FP16 and INT8-simulation latency.

### 2.4 KV Cache Memory Analysis

Purpose: estimate memory growth of KV Cache with longer context.

### 2.5 Attention Optimization

Purpose: compare naive attention and PyTorch SDPA.

### 2.6 AMP Mixed Precision Training

Purpose: compare FP32 training and AMP training in epoch time and GPU memory.

### 2.7 Profiling

Purpose: locate major runtime bottlenecks in LLM-style workload.

## 3. Conclusion

The experiments show that LLM-style workloads are dominated by matrix multiplication and attention operations. CUDA acceleration, mixed precision, quantization, and optimized attention kernels are effective approaches for improving performance and reducing memory consumption.
