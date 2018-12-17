#!/usr/bin/env python
from setuptools import setup, find_packages, Extension  # This setup relies on setuptools since distutils is insufficient and badly hacked code
import numpy as np
from distutils.command.build_ext import build_ext
from Cython.Build import cythonize
import os

version = '0.0.1'

copt = {'msvc': ['-Itestbeam_analysis/cpp/external', '/EHsc']}  # set additional include path and EHsc exception handling for VS
lopt = {}


class build_ext_opt(build_ext):
    def initialize_options(self):
        build_ext.initialize_options(self)
        self.compiler = 'msvc' if os.name == 'nt' else None  # in Miniconda the libpython package includes the MinGW import libraries and a file (Lib/distutils/distutils.cfg) which sets the default compiler to mingw32. Alternatively try conda remove libpython.

    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in copt:
            for e in self.extensions:
                e.extra_compile_args = copt[c]
        if c in lopt:
            for e in self.extensions:
                e.extra_link_args = lopt[c]
        build_ext.build_extensions(self)

cpp_extension = cythonize([
    Extension('testbeam_analysis.analysis_functions', ['testbeam_analysis/cpp/analysis_functions.pyx'])
])

author = 'David-Leon Pohl, Jens Janssen, Yannick Dieter, Christian Bespin, Luigi Vigani'
author_email = 'pohl@physik.uni-bonn.de'

# requirements for core functionality from requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='testbeam_analysis',
    version=version,
    description='A light weight test beam analysis in Python and C++.',
    url='https://github.com/SiLab-Bonn/testbeam_analysis',
    license='BSD 3-Clause ("BSD New" or "BSD Simplified") License',
    long_description='A simple analysis of pixel-sensor data from testbeams. All steps of a full analysis are included in one file in < 1500 lines of Python code. If you you want to do simple straight line fits without a Kalman filter or you want to understand the basics of telescope reconstruction this code might help. If you want to have something fancy to account for thick devices in combination with low energetic beams use e.g. EUTelescope. Depending on the setup a resolution that is only ~ 15% worse can be archieved with this code. For a quick first impression check the example plots in the wiki and run the examples.',
    author=author,
    maintainer=author,
    author_email=author_email,
    maintainer_email=author_email,
    install_requires=install_requires,
    packages=find_packages(),  # exclude=['*.tests', '*.test']),
    include_package_data=True,  # accept all data files and directories matched by MANIFEST.in or found in source control
    package_data={'': ['README.*', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
    ext_modules=cpp_extension,
    include_dirs=[np.get_include()],
    cmdclass={'build_ext': build_ext_opt},
    keywords=['testbeam', 'particle', 'reconstruction', 'pixel', 'detector'],
    platforms='any',
    entry_points={
          'console_scripts': [
              'tba = testbeam_analysis.gui.main:main',
              'testbeam_analysis = testbeam_analysis.gui.main:main'
          ]
      },
)
