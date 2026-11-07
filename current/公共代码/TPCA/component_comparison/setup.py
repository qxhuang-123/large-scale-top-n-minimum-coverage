from setuptools import setup, Extension
import pybind11,sys
setup(name='tpca-dis-core',version='1.0',ext_modules=[Extension(
    'tpca_dis_index_core',['native/state_variants.cpp'],
    include_dirs=[pybind11.get_include(),'native'],language='c++',
    extra_compile_args=['/O2','/DNDEBUG','/std:c++17','/EHsc'] if sys.platform=='win32' else ['-O3','-DNDEBUG','-std=c++17'])])
