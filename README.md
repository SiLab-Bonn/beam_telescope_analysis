# Beam Telescope Analysis
Beam Telescope Analysis (BTA) software written in Python (and C++).

## Intended Use
BTA is intended for use with data from multiple particle detectors in a particle beam.
One or more detectors may be the device under test (DUT).
Any detectors can part of a beam telescope (or hodoscope) or can be a trigger plane (e.g., region of interest trigger) or timing reference (i.e., reference plane, usually of the same type as the DUT).
The software provides detailed analysis of any detector in sub-micrometer range if the resolution of the telescope allows.

## Installation

### Prerequisites for installation:

Python 2.7 or Python 3 must be used.

The following packages are required and can be obtained from PyPI (via `pip install`) and/or from Anaconda (via `conada install`):
- Cython
- PyTables
- NumPy
- SciPy
- Numba
- NumExpr
- Matplotlib
- PyYAML
- dill
- tqdm
- [pixel_clusterizer](https://github.com/SiLab-Bonn/pixel_clusterizer)
- [PyLandau](https://github.com/SiLab-Bonn/pylandau)

*Note:*
Only the development branch of [pixel_clusterizer](https://github.com/SiLab-Bonn/pixel_clusterizer) is currently supported.
For that, install pixel_clusterizer with the following command:
```
pip install git+https://github.com/SiLab-Bonn/pixel_clusterizer@development
```

### Installation of BTA

Clone the BTA repository:
```
git clone https://github.com/SiLab-Bonn/beam_telescope_analysis.git
```

If you want to modify the code without having to re-install the package every time, use the following command (from inside the repository):
```
pip install -e .
```

If you just want to use BTA, run the following command:
```
pip install .
```

## Usage

1. Provide input data files (PyTables/HDF5) specified below.
2. Write a BTA script which is specific to your telescope setup (see examples in [`./beam_telescope_analysis/examples/`](https://github.com/SiLab-Bonn/beam_telescope_analysis/tree/master/beam_telescope_analysis/examples) folder).
3. Run BTA script and wait for the output plots (PDF) to appear.
4. Check output plots for validity and in case of failure modify the BTA script.

Optional:
1. Modify the BTA source code according to your needs.
2. Under some circumstances it might be necessary to add your detector specifications to BTA. For that, add your specification to [`./beam_telescope_analysis/telescope/dut.py`](https://github.com/SiLab-Bonn/beam_telescope_analysis/blob/master/beam_telescope_analysis/telescope/dut.py).

## Input Data File

### File Format

The input data for BTA must be using the PyTables/HDF5 file format.
No other file format is currently supported.
A single file must be provided for each detector (i.e., telescope plane, DUT, reference DUT, etc.).

###  Hit Data Table
Each input data file must contain a data table with the node name `Hits`.
The hit data table must at least contain the following columns:
- `event_number` (int): unique event number (increasing) for all events accross all detectors
- `column` (int): column index starting from 1
- `row` (int): row index starting from 1
- `frame` (int): timing bin (only applicable to some detectors, can be used for cluster building); if not available, set to 0.
- `charge` (float): charge seen by the detector (only applicable to some detectors, can be used for charge weighted clustering); if not available, set to 0.

Additional columns can be provided.

*Note:*
The columns `column` and `row` can be provided as float data type if x/y coordinates instead of indices are available.

## Contributing to BTA

### Bug Report / Feature Request / Question

Please use GitHub's [issue tracker](https://github.com/SiLab-Bonn/beam_telescope_analysis/issues).

### Contributing Code to BTA

1. Fork the project.
2. Clone your fork and/or get the latest changes from upstream.
2. Create a topic branch.
3. Modify the code and commit your changes in logical chunks.
4. Locally rebase the upstream branch into your topic branch.
5. Push your topic branch to your fork.
6. Open a [Pull Request (PR)](https://help.github.com/en/articles/about-pull-requests) with clear title and description about the modifications.

### Predecessor of BTA

[Testbeam Analysis](https://github.com/SiLab-Bonn/testbeam_analysis) (TBA) is the predecessor of Beam Telescope Analysis (BTA).
