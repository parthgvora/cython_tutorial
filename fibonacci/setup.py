from setuptools import setup
from Cython.Build import cythonize

setup(
        name="Fibonacci Sequence",
        ext_modules=cythonize("fib.pyx"),
        zip_safe=False
)
