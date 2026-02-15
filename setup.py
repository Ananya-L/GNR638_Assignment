from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cpp_backend",
        ["framework/cpp_backend.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/openmp"],
    ),
]

setup(
    name="cpp_backend",
    ext_modules=ext_modules,
)
