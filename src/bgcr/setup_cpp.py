from setuptools import Extension, setup
import pybind11

setup(
    name="yelp_greedy_core",
    ext_modules=[Extension(
        "yelp_greedy_core", ["Yelp贪心算法_cpp.cpp"], language="c++",
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["/O2", "/Ob2", "/Oi", "/Ot", "/GL", "/std:c++17", "/DNDEBUG"],
        extra_link_args=["/LTCG"],
    )],
)
