from setuptools import setup,Extension
import sys,pybind11
setup(name='dgca',ext_modules=[Extension('bgcr_d_bench',['native/bgcr_d_bench.cpp'],include_dirs=[pybind11.get_include()],language='c++',extra_compile_args=['/O2','/DNDEBUG','/std:c++17','/EHsc'] if sys.platform=='win32' else ['-O3','-DNDEBUG','-std=c++17'])])
