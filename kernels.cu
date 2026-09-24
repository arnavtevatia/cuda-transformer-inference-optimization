
#include <iostream>
#include <cassert>
#include <vector>
#include <utility>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <mma.h>

using namespace torch::indexing;
using namespace nvcuda;

#define FULL_MASK 0xffffffff
#define HALF_MASK 0x0000ffff

#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) 	        do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(false)
#define gpuErrchk(ans)          do { gpuAssert((ans), __FILE__, __LINE__); } while (false)


__global__ void sgemm_p1_kernel(
  const float* A,
  const float* B,
  float* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  size_t iin = tid % N;
  size_t iim = tid / N;

  if (iim > M) {
    // no work for this thread to do
    return;
  }

  //extra threads do no work
  if (tid >= M * N) {
    return;
  }

  float acc = 0.0f;
  const size_t a_offset = iim * K;
  const size_t b_offset = iin * K;
  for (size_t k = 0; k < K; ++k){
    acc += A[a_offset + k] * B[b_offset + k];
  }
  C[iim * N + iin] += acc;
}

// C += A @ B.t()
void sgemm_p1(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat32);
  assert(B.dtype() == torch::kFloat32);
  assert(C.dtype() == torch::kFloat32);

  const size_t THREADS_PER_BLOCK = 256;
  const size_t TOTAL_THREADS = M * N;
  const size_t TOTAL_BLOCKS = (TOTAL_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  const dim3 threads(THREADS_PER_BLOCK);
  const dim3 blocks(TOTAL_BLOCKS);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  sgemm_p1_kernel<<<blocks, threads, 0, stream>>>(
    A.data_ptr<float>(),
    B.data_ptr<float>(),
    C.data_ptr<float>(),
    M,
    N,
    K
  );
}


__global__ void hgemm_p1_kernel(
  const __half* A,
  const __half* B,
  __half* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  size_t iin = tid % N;
  size_t iim = tid / N;

  if (iim > M) {
    // no work for this thread to do
    return;
  }

  if (iin >= N || iim >= M) {
    return;
  }

  size_t c_ind = iim * N + iin;
  __half acc = C[c_ind];

  for (size_t k = 0; k < K; ++k) {
    size_t a_ind = iim * K + k;
    size_t b_ind = iin * K + k;
    __half a = A[a_ind];
    __half b = B[b_ind];

    acc = __hfma(a, b, acc);
  }

  C[c_ind] = acc;
}

// C += A @ B.t()
void hgemm_p1(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat16);
  assert(B.dtype() == torch::kFloat16);
  assert(C.dtype() == torch::kFloat16);

  const size_t THREADS_PER_BLOCK = 256;
  const size_t TOTAL_THREADS = M * N;
  const size_t TOTAL_BLOCKS = (TOTAL_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  const dim3 threads(THREADS_PER_BLOCK);
  const dim3 blocks(TOTAL_BLOCKS);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  hgemm_p1_kernel<<<blocks, threads, 0, stream>>>(
    (const __half*)A.data_ptr<at::Half>(),
    (const __half*)B.data_ptr<at::Half>(),
    (__half*)C.data_ptr<at::Half>(),
    M,
    N,
    K
  );
}


__global__ void sgemm_p2_kernel(
  const float* A,
  const float* B,
  float* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
    int row = blockIdx.x * 16 + threadIdx.x;
    int col = blockIdx.y * 16 + threadIdx.y;

    __shared__ float Anew[16][16];
    __shared__ float Bnew[16][16];

    float sum = 0.0f;

    for (size_t k0 = 0; k0 < K; k0 += 16) {
      Anew[threadIdx.x][threadIdx.y] = A[row * K + (k0 + threadIdx.y)];
      Bnew[threadIdx.x][threadIdx.y] = B[col * K + (k0 + threadIdx.x)];

      __syncthreads();

      // multiply the two tiles
      for (int k = 0; k < 16; k++) {
        sum += Anew[threadIdx.x][k] * Bnew[k][threadIdx.y];
      }

      __syncthreads();
    }
    C[row * N + col] = sum;
}

// C += A @ B.t()
void sgemm_p2(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat32);
  assert(B.dtype() == torch::kFloat32);
  assert(C.dtype() == torch::kFloat32);

  // for simplicity, restrict the matrices we support to ones that are a multiple of the block size
  assert(M % 16 == 0);
  assert(N % 16 == 0);
  assert(K % 16 == 0);

  const size_t MM_BLOCK_SIZE = 16; // split matrix into 16x16 blocks
  const size_t m_blocks = M / MM_BLOCK_SIZE;
  const size_t n_blocks = N / MM_BLOCK_SIZE;

  const dim3 threads(16,16);
  const dim3 blocks(m_blocks,n_blocks);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  sgemm_p2_kernel<<<blocks, threads, 0, stream>>>(
    A.data_ptr<float>(),
    B.data_ptr<float>(),
    C.data_ptr<float>(),
    M,
    N,
    K
  );
}


__global__ void hgemm_p2_kernel(
  const __half* A,
  const __half* B,
  __half* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
    int row = blockIdx.x * 16 + threadIdx.x;
    int col = blockIdx.y * 16 + threadIdx.y;

    __shared__ __half Anew[16][16];
    __shared__ __half Bnew[16][16];

    __half sum = __float2half(0.0f);

    for (size_t k0 = 0; k0 < K; k0 += 16) {
      Anew[threadIdx.x][threadIdx.y] = A[row * K + (k0 + threadIdx.y)];
      Bnew[threadIdx.x][threadIdx.y] = B[col * K + (k0 + threadIdx.x)];

      __syncthreads();

      //multiply the two tiles
      for (int k = 0; k < 16; k++) {
        sum = __hfma(Anew[threadIdx.x][k], Bnew[k][threadIdx.y], sum);
      }

      __syncthreads();
    }

    C[row * N + col] = sum;
}

// C += A @ B.t()
void hgemm_p2(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat16);
  assert(B.dtype() == torch::kFloat16);
  assert(C.dtype() == torch::kFloat16);

  // for simplicity, restrict the matrices we support to ones that are a multiple of the block size
  assert(M % 16 == 0);
  assert(N % 16 == 0);
  assert(K % 16 == 0);

  const size_t MM_BLOCK_SIZE = 16; // split matrix into 16x16 blocks
  const size_t m_blocks = M / MM_BLOCK_SIZE;
  const size_t n_blocks = N / MM_BLOCK_SIZE;

  const dim3 threads(16,16);
  const dim3 blocks(m_blocks,n_blocks);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  hgemm_p2_kernel<<<blocks, threads, 0, stream>>>(
    (const __half*)A.data_ptr<at::Half>(),
    (const __half*)B.data_ptr<at::Half>(),
    (__half*)C.data_ptr<at::Half>(),
    M,
    N,
    K
  );
}

__global__ void hgemm_p3_kernel(
  const __half* A,
  const __half* B,
  __half* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
  // each block is a single warp that computes one 16x16 tile of C
  const int tile_m = blockIdx.x; // which 16-row block of C (the M dim)
  const int tile_n = blockIdx.y; // which 16-col block of C (the N dim)

  const int row = tile_m * 16; // starting row index in C (and A)
  const int col = tile_n * 16; // starting column index in C (and B^T)

  // WMMA frags
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

  // init accumulator fragment to zero
  wmma::fill_fragment(c_frag, __float2half(0.0f));

  // Loop over K dim in blocks of 16
  for (int k0 = 0; k0 < (int)K; k0 += 16) {
    // ptr to the top-left element of the current 16x16 tile of A (rows [row .. row+15], cols [k0 .. k0+15])
    const half* tile_A = reinterpret_cast<const half*>(A + row * K + k0);

    // the ptr to the top-left element of the current 16x16 tile of B^T:
    // B is NxK row-major matrix (n,k) -> B[n*K + k], so we reinterpret it as a KxN col-major matrix B^T (k,n), B^T(k,n) == B[n*K + k], with leading dimension K
    // (rows (k) [k0 .. k0+15], cols (n) [col .. col+15])
    const half* tile_B = reinterpret_cast<const half*>(B + col * K + k0);

    // load the tiles into WMMA frags
    wmma::load_matrix_sync(a_frag, tile_A, K); // row-major, ld = K
    wmma::load_matrix_sync(b_frag, tile_B, K); // col-major, ld = K

    // tensorCore MMA, c_frag = a_frag * b_frag + c_frag
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // store the resulting 16x16 tile back to C in row-major order
  half* tile_C = reinterpret_cast<half*>(C + row * N + col);
  wmma::store_matrix_sync(tile_C, c_frag, N, wmma::mem_row_major);
}

// C += A @ B.t()
void hgemm_p3(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat16);
  assert(B.dtype() == torch::kFloat16);
  assert(C.dtype() == torch::kFloat16);

  // for simplicity, restrict the matrices we support to ones that are a multiple of the block size
  assert(M % 16 == 0);
  assert(N % 16 == 0);
  assert(K % 16 == 0);

  const size_t MM_BLOCK_SIZE = 16; // split matrix into 16x16 blocks
  const size_t m_blocks = M / MM_BLOCK_SIZE;
  const size_t n_blocks = N / MM_BLOCK_SIZE;

  const dim3 threads(32);
  const dim3 blocks(m_blocks,n_blocks);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  hgemm_p3_kernel<<<blocks, threads, 0, stream>>>(
    (const __half*)A.data_ptr<at::Half>(),
    (const __half*)B.data_ptr<at::Half>(),
    (__half*)C.data_ptr<at::Half>(),
    M,
    N,
    K
  );
}


__global__ void hgemm_p4_kernel(
  const __half* A,
  const __half* B,
  float* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
  // each block is exactly one warp (32 threads), so each block computes one 16x16 tile of C
  const int tile_m = (int)blockIdx.x; // along M
  const int tile_n = (int)blockIdx.y; // along N

  const int row_base = tile_m * 16;
  const int col_base = tile_n * 16;

  // WMMA frags
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  // loop over K tiles
  for (int k0 = 0; k0 < (int)K; k0 += 16) {
    const __half* A_tile = A + row_base * (int)K + k0; // A is MxK row-major

    // C = A * B^T, so we want B stored as NxK row-major, and we interpret B as a (KxN) col-major matrix,
    // element (k, n) in col-major is stored at k + n*K, so we use B[n*K + k]
    const __half* B_tile = B + col_base * (int)K + k0;

    wmma::load_matrix_sync(a_frag, A_tile, (int)K);
    wmma::load_matrix_sync(b_frag, B_tile, (int)K);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  wmma::store_matrix_sync(
    C + row_base * (int)N + col_base,
    c_frag,
    (int)N,
    nvcuda::wmma::mem_row_major
  );
}

// C += A @ B.t()
void hgemm_p4(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat16);
  assert(B.dtype() == torch::kFloat16);
  assert(C.dtype() == torch::kFloat32);

  // for simplicity, restrict the matrices we support to ones that are a multiple of the block size
  assert(M % 16 == 0);
  assert(N % 16 == 0);
  assert(K % 16 == 0);

  const size_t MM_BLOCK_SIZE = 16; // split matrix into 16x16 blocks
  const size_t m_blocks = M / MM_BLOCK_SIZE;
  const size_t n_blocks = N / MM_BLOCK_SIZE;

  const dim3 threads(32);
  const dim3 blocks(m_blocks,n_blocks);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  hgemm_p4_kernel<<<blocks, threads, 0, stream>>>(
    (const __half*)A.data_ptr<at::Half>(),
    (const __half*)B.data_ptr<at::Half>(),
    (float*)C.data_ptr<float>(),
    M,
    N,
    K
  );
}


