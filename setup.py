from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("dwba_c", ["dwba.py"])]

setup(
  name = 'DWBA Method C',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)