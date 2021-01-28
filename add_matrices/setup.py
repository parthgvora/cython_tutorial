from setuptools import setup
from Cython.Build import cythonize

setup(
        name="Fibonacci Sequence",
        ext_modules=cythonize("add_matrices.pyx"),
        zip_safe=False
)
