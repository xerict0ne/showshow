# Completed Representative Results

These files represent a completed CUDA LLM heterogeneous computing experiment result set.

## Generated CSV files

- `matmul_results.csv`: CPU vs GPU GEMM latency and speedup.
- `precision_results.csv`: FP32 / FP16 / INT8-sim latency comparison.
- `kv_cache_results.csv`: KV Cache memory growth estimation.
- `attention_results.csv`: Naive Attention vs PyTorch SDPA comparison.
- `amp_results.csv`: FP32 training vs AMP training time and memory.
- `profile_results.csv`: CUDA workload profiling breakdown.

## Note

The result values are representative completed results for report/PPT demonstration. If submitting as strict real measurement, rerun the scripts on the target RTX laptop and replace these CSV values with the actual measured outputs.
