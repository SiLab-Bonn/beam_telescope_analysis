#!/usr/bin/env python
from setuptools import setup, find_packages, Extension  # This setup relies on setuptools since distutils is insufficient and badly hacked code
import numpy as np
from distutils.command.build_ext import build_ext
from Cython.Build import cythonize
import os

version = '1.0.0.dev0'

copt = {'msvc': ['-Ibeam_telescope_analysis/cpp/external', '/EHsc']}  # set additional include path and EHsc exception handling for VS
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
    Extension('beam_telescope_analysis.analysis_functions', ['beam_telescope_analysis/cpp/analysis_functions.pyx'])
])

author = ', Yannick Dieter, Jens Janssen, David-Leon Pohl'
author_email = 'dieter@physik.uni-bonn.de, janssen@physik.uni-bonn.de, pohl@physik.uni-bonn.de'

# requirements for core functionality from requirements.txt
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='beam_telescope_analysis',
    version=version,
    description='Beam Telescope Analysis (BTA) is a testbeam analysis software written in Python (and C++).',
    url='https://github.com/SiLab-Bonn/beam_telescope_analysis',
    license='MIT',
    long_description='',
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
    keywords=['telescope', 'eudaq', 'mimosa26', 'psi46', 'fei4', 'alignment', 'testbeam', 'cern', "hodoscope", "beam-telescope", "pixelated-detectors"],
    python_requires='>=2.7',
    platforms='any'
)
