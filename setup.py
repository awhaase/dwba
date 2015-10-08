from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'DWBA',
  ext_modules = cythonize("dwba_c.pyx"),
)