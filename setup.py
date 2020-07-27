from distutils.core import setup, Extension
import os

setup(name='HMFunc',
      version='0.1',
      description='Simple Halo mass function tool with a comparison wrapper to Colossus',
      url='https://github.com/nbatta/HMFunc',
      author='Nicholas Battaglia',
      author_email='nicholas.battaglia@gmail.com',
      license='BSD-2-Clause',
      packages=['HMFunc'.'HMFunc.src'],
      package_dir={'HMFunc':'src'},
      zip_safe=False)
