from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
        Extension(
            "split",
            ["split.pyx"],
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp"],
            language="c++"
        
        )
]

setup(
        name="Csplitter",
        ext_modules=cythonize(ext_modules),
)


"""
setup(ext_modules = cythonize(

    "split.pyx",
    language="c++"
    
    
))
"""
