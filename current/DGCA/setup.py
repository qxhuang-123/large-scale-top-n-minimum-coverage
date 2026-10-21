from setuptools import setup, Extension
import pybind11
import sys

flags=['/O2','/DNDEBUG','/std:c++17','/EHsc'] if sys.platform=='win32' else ['-O3','-DNDEBUG','-std=c++17']
setup(name='bgcr-d-local-benchmark',version='1.0.0',ext_modules=[
    Extension('bgcr_d_bench',['native/bgcr_d_bench.cpp'],include_dirs=[pybind11.get_include()],language='c++',extra_compile_args=flags),
    Extension('bgcr_variants',['supplied/BGCR_OP_Yelp_comparison/src/bgcr_variants.cpp'],include_dirs=[pybind11.get_include()],language='c++',extra_compile_args=flags)])
