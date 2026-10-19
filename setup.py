import os
import sys
from setuptools import setup, Extension

windows = sys.platform == "win32"
use_blas = os.environ.get("ND2_BLAS") == "1"
args = ["/std:c++17", "/O2"] if windows else ["-std=c++17", "-O3", "-ffp-contract=off", "-Wno-misleading-indentation"]
macros = [("ND2_USE_BLAS", "1")] if use_blas else []
libraries = ["blas"] if use_blas else []
setup(ext_modules=[Extension(
    "nanodecision2._native", ["native/python_module.cpp"],
    include_dirs=["native"], define_macros=macros, libraries=libraries,
    language="c++", extra_compile_args=args,
)])
