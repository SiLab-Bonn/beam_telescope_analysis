===============================================
Introduction
===============================================

|travis-status|  |appveyor-status|  |coverage|  |doc|

Testbeam analysis is a simple to use software to analyse pixel-sensor data taken in a particle-beam telescope-setup.
All steps of a complete analysis are implemented with a few independent python functions.
For a quick first impression check the examples in the documentation.

In a future release it is forseen to enhance the alignment to work more reliable.

Installation
============
You have to have Python 2/3 with the following modules installed:
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

If you are new to Python please look at the installation guide in the wiki.
Since it is recommended to change example files according to your needs you should install the module with

.. code-block:: bash

   conda install python=2.7 cython dill future matplotlib numba numexpr numpy numpydoc pytables pyyaml scipy
   pip install pixel_clusterizer progressbar-latest pylandau
   python setup.py develop

This does not copy the code to a new location, but just links to it.
Uninstall:

.. code-block:: bash

   pip uninstall testbeam_analysis


Example usage
==============
Check the examples folder with data and examples of a Mimosa26 and a FE-I4 telescope analysis.
Run eutelescope_example.py or fei4_telescope_example.py in the example folder and check the text output to
the console as well as the plot and data files that are created to understand what is going on.
In the examples folder type e.g.:

.. code-block:: bash

   python fei4_telescope_example.py

.. |travis-status| image:: https://travis-ci.org/SiLab-Bonn/testbeam_analysis.svg?branch=gui
    :target: https://travis-ci.org/SiLab-Bonn/testbeam_analysis
    :alt: Build status

.. |appveyor-status| image:: https://ci.appveyor.com/api/projects/status/github/SiLab-Bonn/testbeam_analysis/branch/gui
    :target: https://ci.appveyor.com/project/DavidLP/testbeam-analysis/branch/gui
    :alt: Build status

.. |doc| image:: https://img.shields.io/badge/documentation--blue.svg
    :target: http://silab-bonn.github.io/testbeam_analysis
    :alt: Documentation

.. |coverage| image:: https://coveralls.io/repos/SiLab-Bonn/testbeam_analysis/badge.svg?branch=gui
    :target: https://coveralls.io/github/SiLab-Bonn/testbeam_analysis?branch=gui
    :alt: Coverage


