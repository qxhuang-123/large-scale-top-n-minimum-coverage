"""Build F with GCC/Clang or native Windows MSVC (compatibility header only)."""
import os
import shutil
import sys
from setuptools import Extension, setup
import pybind11

if sys.platform == "win32":
    compile_args = ["/O2", "/DNDEBUG", "/std:c++17", "/EHsc"]
else:
    compile_args = ["-O3", "-DNDEBUG", "-std=c++17"]
    for env, choices in (("CC", ("gcc", "clang")), ("CXX", ("g++", "clang++"))):
        if env not in os.environ:
            found = next((shutil.which(c) for c in choices if shutil.which(c)), None)
            if found:
                os.environ[env] = found

setup(
    name="minimum-coverage-f", version="1.0.0",
    py_modules=["f_algorithm"],
    ext_modules=[Extension(
        "state_variants", ["src/state_variants.cpp"],
        include_dirs=[pybind11.get_include(), "src"], language="c++",
        extra_compile_args=compile_args,
    )],
)
