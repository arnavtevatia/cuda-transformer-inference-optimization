from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="cuda_gemm",
  ext_modules=[
    cpp_extension.CUDAExtension(
      "cuda_gemm",
      ["wrapper.cpp","kernels.cu"]
    )],
  cmdclass={'build_ext': cpp_extension.BuildExtension}
)

