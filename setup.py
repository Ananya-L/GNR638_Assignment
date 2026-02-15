from setuptools import setup, Extension
import pybind11
import platform

compile_args = []
link_args = []

if platform.system() != "Windows":
    compile_args.append("-fopenmp")
    link_args.append("-fopenmp")

ext_modules = [
    Extension(
        "cpp_backend",
        ["framework/cpp_backend.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

setup(
    name="cpp_backend",
    ext_modules=ext_modules,
    zip_safe=False,
)
