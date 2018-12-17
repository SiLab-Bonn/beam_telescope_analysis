===============================================
Introduction
===============================================

|travis-status|  |appveyor-status|  |coverage|  |doc|

The testbeam analysis package is for the analysis of pixel detector data together with data from a beam telescope in a particle beam.
All steps of from raw data conversion to telescope alignment and finally efficiency calculation and plotting are provided and implemented in separate functions.
For a first impression, please check the examples in the documentation and inside the examples folder.

Installation
============
The following modules are required:
  - cython
  - dill
  - future
  - matplotlib
  - numba
  - numexpr
  - numpy
  - numpydoc
  - pixel_clusterizer
  - progressbar-latest
  - pylandau
  - pytables
  - pyyaml
  - scipy

If you are new to Python, please have a look at the installation guide in the wiki.
We recommend to use to use `Anaconda/Miniconda <https://conda.io/docs/user-guide/install/download.html>`_ Python to ease the installation of the dependencies.

Clone the testbeam analysis repository and run the following commands inside the beam_telescope_analysis folder:

.. code-block:: bash

   conda install cython dill future matplotlib numba numexpr numpy numpydoc pytables pyyaml scipy
   pip install pixel_clusterizer progressbar-latest pylandau
   python setup.py develop

This does not copy the code to a new location, but just links to it.
Uninstall with:

.. code-block:: bash

   pip uninstall beam_telescope_analysis


Example Usage
=============
Check the examples folder with data and examples of a Mimosa26 and a FE-I4 telescope analysis.
Run eutelescope_example.py or fei4_telescope_example.py in the example folder and check the text output to
the console as well as the plots and data files that are created to understand what is going on.
In the examples folder type e.g.:

.. code-block:: bash

   python fei4_telescope_example.py

.. |travis-status| image:: https://travis-ci.org/SiLab-Bonn/beam_telescope_analysis.svg?branch=master
    :target: https://travis-ci.org/SiLab-Bonn/beam_telescope_analysis
    :alt: Build status

.. |appveyor-status| image:: https://ci.appveyor.com/api/projects/status/github/SiLab-Bonn/beam_telescope_analysis/branch/master
    :target: https://ci.appveyor.com/project/DavidLP/testbeam-analysis/branch/master
    :alt: Build status

.. |doc| image:: https://img.shields.io/badge/documentation--blue.svg
    :target: http://silab-bonn.github.io/beam_telescope_analysis
    :alt: Documentation

.. |coverage| image:: https://coveralls.io/repos/SiLab-Bonn/beam_telescope_analysis/badge.svg?branch=master
    :target: https://coveralls.io/github/SiLab-Bonn/beam_telescope_analysis?branch=master
    :alt: Coverage


