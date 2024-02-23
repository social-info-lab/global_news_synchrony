from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# setup the Cython implementation
# python3 setup.py build_ext --inplace

setup(ext_modules = cythonize(Extension(
    'cython_jaccard_sim',
    sources=['cython_jaccard_sim.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))