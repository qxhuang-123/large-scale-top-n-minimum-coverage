import os
from setuptools import Extension, setup
import pybind11

flags = ['/O2', '/DNDEBUG', '/std:c++17', '/EHsc'] if os.name == 'nt' else ['-O3', '-DNDEBUG', '-std=c++17']

def extension(name, source, depends=()):
    return Extension(name, [source], depends=list(depends),
                     include_dirs=[pybind11.get_include()], language='c++',
                     extra_compile_args=flags)

setup(name='dgca',version='1.0.0',ext_modules=[extension('dgca_stage2','stage2.cpp',['vendor/reference_core.cpp']),extension('bgcr_d_bench','vendor/reference_core.cpp')])
