from setuptools import setup, Extension
from Cython.Build import cythonize
import sys


extensions = [Extension("Cython_LL", ["Cython_LL.pyx"]),
              Extension("CythonParallel_LL", ["CythonParallel_LL.pyx"],
                        extra_compile_args=["/openmp"])]

setup(ext_modules=cythonize(extensions, annotate=True))

