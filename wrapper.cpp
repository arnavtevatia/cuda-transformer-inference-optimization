
#include <torch/extension.h>

// C += A @ B.t()
// all tensors must be contiguous
void sgemm_p1(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
);

// C += A @ B.t()
// all tensors must be contiguous
void hgemm_p1(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
);

// C += A @ B.t()
// all tensors must be contiguous
void sgemm_p2(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
);

// C += A @ B.t()
// all tensors must be contiguous
void hgemm_p2(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
);

// C += A @ B.t()
// all tensors must be contiguous
void hgemm_p3(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
);

// C += A @ B.t()
// all tensors must be contiguous
void hgemm_p4(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_p1", &sgemm_p1, "sgemm_p1");
    m.def("hgemm_p1", &hgemm_p1, "hgemm_p1");
    m.def("sgemm_p2", &sgemm_p2, "sgemm_p2");
    m.def("hgemm_p2", &hgemm_p2, "hgemm_p2");
    m.def("hgemm_p3", &hgemm_p3, "hgemm_p3");
    m.def("hgemm_p4", &hgemm_p4, "hgemm_p4");
}

