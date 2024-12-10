from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='marlin-reproduction',
    version='0.1.1',
    author='Carl Guo',
    author_email='carlguo@mit.edu',
    description='Highly optimized FP16xINT4 CUDA matmul kernel.',
    install_requires=['numpy', 'torch'],
    packages=['marlin_reproduction'],
    ext_modules=[cpp_extension.CUDAExtension(
        # 'marlin_cuda', ['marlin_reproduction/marlin_cuda.cpp', 'marlin_reproduction/marlin_cuda_kernel_orig.cu']
        # 'marlin_cuda', ['marlin_reproduction/marlin_cuda.cpp', 'marlin_reproduction/marlin_cuda_kernel.cu']
        'marlin_cuda', ['marlin_reproduction/marlin_cuda.cpp', 'marlin_reproduction/marlin_hadamard.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
