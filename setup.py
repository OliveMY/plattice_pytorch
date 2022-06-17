from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='PLOO',
    ext_modules=[
        CUDAExtension('PLOO', [
            'permuto.cpp',
            'permutohedral.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })