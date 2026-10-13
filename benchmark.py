"""
Benchmark and correctness check for the custom CUDA GEMM kernels
against PyTorch's built-in torch.mm, across FP32 and FP16, including
a TensorCore (WMMA) kernel.

Usage:
    python3 setup.py build
    python3 setup.py install
    python3 benchmark.py
"""

import torch
import cuda_gemm
import time
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = False


def correctness_check():
    M, N, K = 3 * 1024, 4 * 1024, 5 * 1024

    print("float32")
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(N, K, device="cuda")
    C = torch.zeros(M, N, device="cuda")
    C1 = torch.zeros(M, N, device="cuda")
    C2 = torch.zeros(M, N, device="cuda")
    torch.cuda.synchronize()

    start = time.time()
    torch.mm(A, B.t(), out=C)
    torch.cuda.synchronize()
    print(f"elapsed for torch.mm: {time.time() - start:.4f} s")

    start = time.time()
    cuda_gemm.sgemm_p1(A, B, C1)
    torch.cuda.synchronize()
    print(f"elapsed for sgemm_p1: {time.time() - start:.4f} s")

    start = time.time()
    cuda_gemm.sgemm_p2(A, B, C2)
    torch.cuda.synchronize()
    print(f"elapsed for sgemm_p2: {time.time() - start:.4f} s")

    Cd = A.double() @ B.double().t()
    print(f"relative square error torch: {(C.double() - Cd).square().sum() / Cd.square().sum()}")
    print(f"relative square error    p1: {(C1.double() - Cd).square().sum() / Cd.square().sum()}")
    print(f"relative square error    p2: {(C2.double() - Cd).square().sum() / Cd.square().sum()}")

    print("\nfloat16")
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    C1 = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    C2 = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    C3 = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    C4 = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()

    start = time.time()
    torch.mm(A, B.t(), out=C)
    torch.cuda.synchronize()
    print(f"elapsed for torch.mm: {time.time() - start:.4f} s")

    start = time.time()
    cuda_gemm.hgemm_p1(A, B, C1)
    torch.cuda.synchronize()
    print(f"elapsed for hgemm_p1: {time.time() - start:.4f} s")

    start = time.time()
    cuda_gemm.hgemm_p2(A, B, C2)
    torch.cuda.synchronize()
    print(f"elapsed for hgemm_p2: {time.time() - start:.4f} s")

    start = time.time()
    cuda_gemm.hgemm_p3(A, B, C3)
    torch.cuda.synchronize()
    print(f"elapsed for hgemm_p3: {time.time() - start:.4f} s")

    start = time.time()
    cuda_gemm.hgemm_p4(A, B, C4)
    torch.cuda.synchronize()
    print(f"elapsed for hgemm_p4: {time.time() - start:.4f} s")

    Cd = A.double() @ B.double().t()
    print(f"relative square error torch: {(C.double() - Cd).square().sum() / Cd.square().sum()}")
    print(f"relative square error    p1: {(C1.double() - Cd).square().sum() / Cd.square().sum()}")
    print(f"relative square error    p2: {(C2.double() - Cd).square().sum() / Cd.square().sum()}")
    print(f"relative square error    p3: {(C3.double() - Cd).square().sum() / Cd.square().sum()}")
    print(f"relative square error    p4: {(C4.double() - Cd).square().sum() / Cd.square().sum()}")


def sweep_and_plot():
    sizes = [256, 512, 1024, 2048, 3072, 4096]
    results = {k: [] for k in [
        "torch_f32", "sgemm_p1", "sgemm_p2",
        "torch_f16", "hgemm_p1", "hgemm_p2", "hgemm_p3", "hgemm_p4",
    ]}

    for s in sizes:
        print(f"Size {s}...")

        A = torch.randn(s, s, device="cuda")
        B = torch.randn(s, s, device="cuda")
        C = torch.zeros(s, s, device="cuda")

        torch.cuda.synchronize(); t = time.time(); torch.mm(A, B.t(), out=C); torch.cuda.synchronize()
        results["torch_f32"].append(time.time() - t)

        C.zero_(); torch.cuda.synchronize(); t = time.time(); cuda_gemm.sgemm_p1(A, B, C); torch.cuda.synchronize()
        results["sgemm_p1"].append(time.time() - t)

        C.zero_(); torch.cuda.synchronize(); t = time.time(); cuda_gemm.sgemm_p2(A, B, C); torch.cuda.synchronize()
        results["sgemm_p2"].append(time.time() - t)

        A = torch.randn(s, s, device="cuda", dtype=torch.float16)
        B = torch.randn(s, s, device="cuda", dtype=torch.float16)
        C = torch.zeros(s, s, device="cuda", dtype=torch.float16)
        C4 = torch.zeros(s, s, device="cuda", dtype=torch.float32)

        torch.cuda.synchronize(); t = time.time(); torch.mm(A, B.t(), out=C); torch.cuda.synchronize()
        results["torch_f16"].append(time.time() - t)

        C.zero_(); torch.cuda.synchronize(); t = time.time(); cuda_gemm.hgemm_p1(A, B, C); torch.cuda.synchronize()
        results["hgemm_p1"].append(time.time() - t)

        C.zero_(); torch.cuda.synchronize(); t = time.time(); cuda_gemm.hgemm_p2(A, B, C); torch.cuda.synchronize()
        results["hgemm_p2"].append(time.time() - t)

        C.zero_(); torch.cuda.synchronize(); t = time.time(); cuda_gemm.hgemm_p3(A, B, C); torch.cuda.synchronize()
        results["hgemm_p3"].append(time.time() - t)

        torch.cuda.synchronize(); t = time.time(); cuda_gemm.hgemm_p4(A, B, C4); torch.cuda.synchronize()
        results["hgemm_p4"].append(time.time() - t)

    plt.figure(figsize=(10, 6))
    for k, v in results.items():
        plt.plot(sizes, v, "o-", label=k)
    plt.xlabel("Matrix Size (M=N=K)")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.legend()
    plt.title("Kernel Performance vs Matrix Size")
    plt.grid(True, alpha=0.3)
    plt.savefig("benchmark.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    correctness_check()
    print()
    sweep_and_plot()
