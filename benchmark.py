"""
Correctness check and timing sweep for the custom CUDA GEMM kernels
(C = A @ B.T) against PyTorch's torch.mm, in FP32 and FP16, including
the TensorCore (WMMA) kernels.

Timing uses CUDA events, warmup runs, and the median of several
repetitions, so one-time costs (CUDA context setup, cuBLAS handle
creation, first-launch overhead) do not distort small-size results.

Usage:
    python3 setup.py build
    python3 setup.py install
    python3 benchmark.py
"""

import os
import statistics

import torch
import matplotlib.pyplot as plt

import cuda_gemm

torch.backends.cuda.matmul.allow_tf32 = False

WARMUP = 3
REPS = 10
SIZES = [256, 512, 1024, 2048, 3072, 4096]


def time_ms(fn, warmup=WARMUP, reps=REPS):
    """Median runtime of fn() in milliseconds, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def rel_error(C, Cd):
    """Relative squared error against a double-precision reference."""
    return ((C.double() - Cd).square().sum() / Cd.square().sum()).item()


def kernels_for(A, B, A16, B16, s):
    """Each entry: name -> (callable, output tensor)."""
    C32 = torch.zeros(s, s, device="cuda")
    C16 = torch.zeros(s, s, device="cuda", dtype=torch.float16)
    out = {}
    out["torch_f32"] = (lambda C=C32.clone(): torch.mm(A, B.t(), out=C))
    out["sgemm_p1"] = (lambda C=C32.clone(): cuda_gemm.sgemm_p1(A, B, C))
    out["sgemm_p2"] = (lambda C=C32.clone(): cuda_gemm.sgemm_p2(A, B, C))
    out["torch_f16"] = (lambda C=C16.clone(): torch.mm(A16, B16.t(), out=C))
    out["hgemm_p1"] = (lambda C=C16.clone(): cuda_gemm.hgemm_p1(A16, B16, C))
    out["hgemm_p2"] = (lambda C=C16.clone(): cuda_gemm.hgemm_p2(A16, B16, C))
    out["hgemm_p3"] = (lambda C=C16.clone(): cuda_gemm.hgemm_p3(A16, B16, C))
    out["hgemm_p4"] = (lambda C=C32.clone(): cuda_gemm.hgemm_p4(A16, B16, C))
    return out


def correctness_check():
    """Run every kernel once and report its error vs. a float64 reference."""
    M, N, K = 3 * 1024, 4 * 1024, 5 * 1024
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(N, K, device="cuda")
    A16, B16 = A.half(), B.half()

    ref32 = A.double() @ B.double().t()
    ref16 = A16.double() @ B16.double().t()

    cases = [
        ("torch_f32", lambda C: torch.mm(A, B.t(), out=C), torch.float32, ref32),
        ("sgemm_p1", lambda C: cuda_gemm.sgemm_p1(A, B, C), torch.float32, ref32),
        ("sgemm_p2", lambda C: cuda_gemm.sgemm_p2(A, B, C), torch.float32, ref32),
        ("torch_f16", lambda C: torch.mm(A16, B16.t(), out=C), torch.float16, ref16),
        ("hgemm_p1", lambda C: cuda_gemm.hgemm_p1(A16, B16, C), torch.float16, ref16),
        ("hgemm_p2", lambda C: cuda_gemm.hgemm_p2(A16, B16, C), torch.float16, ref16),
        ("hgemm_p3", lambda C: cuda_gemm.hgemm_p3(A16, B16, C), torch.float16, ref16),
        ("hgemm_p4", lambda C: cuda_gemm.hgemm_p4(A16, B16, C), torch.float32, ref16),
    ]

    print(f"Correctness (M={M}, N={N}, K={K}), relative squared error vs. float64:")
    errors = {}
    for name, fn, dtype, ref in cases:
        C = torch.zeros(M, N, device="cuda", dtype=dtype)
        fn(C)
        torch.cuda.synchronize()
        errors[name] = rel_error(C, ref)
        print(f"  {name:10s} {errors[name]:.3e}")
    return errors


def sweep():
    """Time every kernel across square sizes. Returns {name: [ms per size]}."""
    results = {}
    for s in SIZES:
        print(f"Timing size {s}...")
        A = torch.randn(s, s, device="cuda")
        B = torch.randn(s, s, device="cuda")
        A16, B16 = A.half(), B.half()
        for name, fn in kernels_for(A, B, A16, B16, s).items():
            # The naive kernels are slow at large sizes, so use fewer reps there.
            reps = 3 if name.endswith("_p1") and s >= 2048 else REPS
            results.setdefault(name, []).append(time_ms(fn, reps=reps))
    return results


def print_table(results):
    """Print a markdown table of runtimes plus key speedup ratios."""
    print("\n| Size | " + " | ".join(results) + " |")
    print("|---|" + "---|" * len(results))
    for i, s in enumerate(SIZES):
        row = " | ".join(f"{results[k][i]:.3f}" for k in results)
        print(f"| {s} | {row} |")
    print("\n(times in ms, median of repetitions)\n")

    print("| Size | p2 vs p1 (fp16) | p3 vs p1 (fp16) | p3 vs torch_f16 | p4 vs torch_f16 |")
    print("|---|---|---|---|---|")
    for i, s in enumerate(SIZES):
        r = {k: v[i] for k, v in results.items()}
        print(
            f"| {s} | {r['hgemm_p1'] / r['hgemm_p2']:.1f}x faster"
            f" | {r['hgemm_p1'] / r['hgemm_p3']:.1f}x faster"
            f" | {r['torch_f16'] / r['hgemm_p3']:.2f}x"
            f" | {r['torch_f16'] / r['hgemm_p4']:.2f}x |"
        )
    print("\n(last two columns: >1 means the custom kernel is faster than torch.mm)")


def plot(results, path=os.path.join("assets", "benchmark.png")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for k, v in results.items():
        style = "--" if k.startswith("torch") else "o-"
        plt.plot(SIZES, v, style, marker="o", label=k)
    plt.xlabel("Matrix size (M = N = K)")
    plt.ylabel("Median time (ms)")
    plt.yscale("log")
    plt.legend()
    plt.title(f"GEMM kernel runtime vs. matrix size ({torch.cuda.get_device_name()})")
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()}\n")
    correctness_check()
    print()
    results = sweep()
    print_table(results)
    plot(results)
