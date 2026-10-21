from setuptools import Extension, setup
import pybind11
import sys

setup(name='trmnl-tpcar-progressive', version='1.0.0',
      ext_modules=[Extension('state_variants_progressive', ['native_progressive/state_variants.cpp'],
          include_dirs=[pybind11.get_include(), 'native_progressive'], language='c++',
          extra_compile_args=['/O2', '/DNDEBUG', '/std:c++17', '/EHsc'] if sys.platform == 'win32'
          else ['-O3', '-DNDEBUG', '-std=c++17'])])
