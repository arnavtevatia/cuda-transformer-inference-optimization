# CUDA Transformer Inference Optimization

Custom CUDA kernels for GEMM (`C += A @ B.t()`), implemented as a PyTorch C++/CUDA
extension, built for transformer inference workloads on OLMO2-1B.

Four kernel variants are implemented, in increasing order of sophistication:

| Kernel | Precision | Approach |
|---|---|---|
| `sgemm_p1` / `hgemm_p1` | FP32 / FP16 | Naive, one thread per output element |
| `sgemm_p2` / `hgemm_p2` | FP32 / FP16 | Tiled, shared-memory blocking (16x16 tiles) |
| `hgemm_p3` | FP16 | TensorCore (WMMA), one warp per 16x16 tile |
| `hgemm_p4` | FP16 in, FP32 out | TensorCore (WMMA), mixed-precision accumulation |

Across matrix sizes, the tiled and TensorCore kernels achieve **72-158x speedups**
over the naive baseline, approaching or exceeding PyTorch's built-in `torch.mm` at
larger sizes.

## Structure

- `kernels.cu` - CUDA kernel implementations
- `wrapper.cpp` - PyTorch C++ extension bindings (pybind11)
- `setup.py` - build configuration for the extension
- `benchmark.py` - correctness checks (relative error vs. double-precision reference)
  and a performance sweep across matrix sizes, with a log-scale plot

## Build & run

```bash
python3 setup.py build
python3 setup.py install
python3 benchmark.py
```

Requires a CUDA-capable GPU with TensorCore support (Volta or newer) for the
`hgemm_p3` / `hgemm_p4` kernels.

## Notes

Profiled memory bandwidth and compute bottlenecks across kernel variants to
diagnose and resolve performance limitations at each stage, from the
naive/memory-bound baseline through shared-memory tiling to TensorCore-accelerated
mixed-precision execution.
