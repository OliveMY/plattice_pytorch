from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='PLOO',
    ext_modules=[
        CUDAExtension('PLOO', [
            'permuto.cpp',
            'permutohedral.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-std=c++17',
                '-gencode=arch=compute_86,code=sm_86',
            ],
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
