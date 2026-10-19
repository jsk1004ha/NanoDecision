import os
import sys
from setuptools import setup, Extension

# The default kernel has no external libraries. Opt into system BLAS explicitly.
blas = os.environ.get("ND2_BLAS", "0") == "1"
if os.name == "nt":
    args = ["/std:c++17", "/O2", "/EHsc"]
else:
    args = ["-std=c++17", "-O3", "-ffp-contract=off"]
setup(ext_modules=[Extension(
    "nanodecision2._native", ["native/python_module.cpp"],
    include_dirs=["native"], language="c++", extra_compile_args=args,
    define_macros=[("ND2_USE_BLAS", "1")] if blas else [],
    libraries=["blas"] if blas else [],
)])
