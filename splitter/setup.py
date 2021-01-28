from setuptools import setup
from Cython.Build import cythonize

setup(
        name="Splitter",
        ext_modules=cythonize("split.pyx"),
        zip_safe=False
)
