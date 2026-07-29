from setuptools import Extension, setup
import pybind11

ext_modules = [
    Extension(
        "tpcar_core_fast",
        ["tpcar_core.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=[
            "/O2",
            "/Ob2",
            "/Oi",
            "/Ot",
            "/GL",
            "/std:c++17",
            "/DNDEBUG",
            "/DTPCAR_MODULE_NAME=tpcar_core_fast",
        ],
        extra_link_args=["/LTCG"],
    )
]

setup(name="tpcar_core_fast", version="0.2", ext_modules=ext_modules)
