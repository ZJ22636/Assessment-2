from setuptools import setup, Extension
from Cython.Build import cythonize
import sys

if sys.platform == "win32":
    openmp_compile, openmp_link = ["/openmp"], []  
else:
    openmp_compile, openmp_link = ["-fopenmp"], ["-fopenmp"] #for marker if on linux or mac

extensions = [Extension("Cython_LL", ["Cython_LL.pyx"]),
              Extension("CythonParallel_LL", ["CythonParallel_LL.pyx"],
                        extra_compile_args=["/openmp"], 
                        extra_link_args=openmp_link),
              Extension("Optimised_LL", ["Optimised_LL.pyx"])]

setup(ext_modules=cythonize(extensions, annotate=True))

