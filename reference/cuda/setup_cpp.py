from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Default to T4 (7.5) and A100 (8.0); override for the target GPU.
# Honor a caller-supplied TORCH_CUDA_ARCH_LIST so on-prem GPUs (e.g. Blackwell
# GB10 = sm_100) can pass their arch via the docker-compose build arg without
# editing this file. Always include +PTX so kernels can JIT-fallback if the
# runtime GPU is newer than every compiled SASS arch.
_default_arches = '7.5;8.0'
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', _default_arches)
if '+PTX' not in os.environ['TORCH_CUDA_ARCH_LIST']:
    # Tack PTX onto the highest listed arch so the extension is JIT-portable.
    archs = os.environ['TORCH_CUDA_ARCH_LIST'].split(';')
    archs[-1] = archs[-1] + '+PTX'
    os.environ['TORCH_CUDA_ARCH_LIST'] = ';'.join(archs)

setup(
    name='toxtransformer_cuda',
    ext_modules=[
        CUDAExtension(
            name='toxtransformer_cuda',
            sources=['tensor_builder.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
