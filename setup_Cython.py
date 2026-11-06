from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [Extension("Cython_LL", ["Cython_LL.pyx"])]

setup(ext_modules=cythonize(extensions, annotate=True))

