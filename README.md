# Beam Telescope Analysis
Beam Telescope Analysis (BTA) software written in Python (and C++).

## Intended Use

BTA is intended for use with data from multiple particle detectors in a particle beam.
One or more detectors may be the device under test (DUT).
Any detectors can part of a beam telescope (or hodoscope) or can be a trigger plane (e.g., region of interest trigger) or timing reference (i.e., reference plane, usually of the same type as the DUT).
The software provides detailed analysis of any detector in sub-micrometer range if the resolution of the telescope allows.

### Features

1. Supporting any pixelated detectors, e.g., detectors with hexagonal pixels have been successfully investigated.
2. Works even under harsh beam environments (i.e., high track densities, strong beam background) and delivers precise efficiencies.
3. Alignment works even with limited information about the location of each detector (e.g., only z-position from the first and last telescope plane necessary).
4. Providing Kalman-based estimates of particle tracks, especially for low-energy particle beams.

BTA uses some novel approaches which have not yet been applied to data from beam telescopes:
1. SVD-based method for suppressing beam background in pre-alignment.
2. SVD-based method for alignment of the detector planes.

## Installation

### Prerequisites for installation:

Python 2.7 or Python 3 must be used.

The following packages are required and can be obtained from PyPI (via `pip install`) and/or from Anaconda (via `conda install`):
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
A single file must be provided for each detector (i.e., telescope plane, timing reference, DUT, etc.).

###  Hit Data Table
Each input data file must contain a data table with the node name `Hits`.
The hit data table must at least contain the following columns:
- `event_number` (int): unique event number (increasing) for all events accross all detectors; starting from 0
- `column` (int): pixel column index; starting from 1
- `row` (int): pixel row index; starting from 1
- `frame` (int): timing bin (only applicable to some detectors, can be used for cluster building); if not available, set to 0.
- `charge` (float): charge seen by the detector (only applicable to some detectors, can be used for charge weighted clustering); if not available, set to 0.

Additional columns can be provided.

*Note:*
The columns `column` and `row` can be provided as float data type if x/y coordinates instead of indices are available.

## Publications

### Proceedings and Papers
1. J. Janssen (on behalf of the [RD42 collaboration](https://rd42.web.cern.ch)), *Test Beam Results of ATLAS DBM pCVD Diamond Detectors Using a Novel Threshold Tuning Method*, JINST. DOI: [10.1088/1748-0221/12/03/C03072](https://dx.doi.org/10.1088/1748-0221/12/03/C03072)
2. N. Wermes, *Pixel detectors ... where do we stand?*, NIMA. DOI: [10.1016/j.nima.2018.07.003](https://dx.doi.org/10.1016/j.nima.2018.07.003)
3. D.-L. Pohl et al., *Radiation Hard Pixel Sensors Using High-Resistive Wafers in a 150nm CMOS Processing Line*, JINST. DOI: [10.1088/1748-0221/12/06/P06020](https://dx.doi.org/10.1088/1748-0221/12/06/P06020)
4. M. Reichmann (on behalf of the [RD42 collaboration](https://rd42.web.cern.ch)), *New Test Beam Results of 3D and Pad Detectors Constructed with Poly-Crystalline CVD Diamond*, preprint submitted.

...more to come.

### Conferences

1. J. Janssen (on behalf of the [RD42 collaboration](https://rd42.web.cern.ch)), *Test Beam Results of ATLAS Diamond Beam Monitor poly-crystalline CVD Diamond Modules*, [Topical Seminar on Innovative Particle and Radiation Detectors (IPRD)](http://www.bo.infn.it/sminiato/siena16.html), 2016, Siena, Italy.
2. D.-L. Pohl et al., *Radiation-hard passive CMOS-sensors*, [13th Trento Workshop on Advanced Silicon Radiation Detectors](https://indico.cern.ch/event/666427/), 2018, Munich, Germany. https://indico.cern.ch/event/666427/contributions/2881132/
3. H. Kagan (on behalf of the [RD42 collaboration](https://rd42.web.cern.ch)), *Beam Test Results of 3D Pixel Detectors Constructed with pCVD Diamond*, [International Workshop on Semiconductor Pixel Detectors for Particles and Imaging (Pixel)](https://indico.cern.ch/event/669866/), 2018, Taipei, Taiwan. https://indico.cern.ch/event/669866/contributions/3245181/
4. A. Oh (on behalf of the [RD42 collaboration](https://rd42.web.cern.ch)), *Latest Results on Radiation Tolerance of Diamond Detectors & Status of 3D Diamond*, [Advanced Diamond Assemblies (ADAMAS)](http://www-adamas.gsi.de/workshops), 2018, Vienna, Austria.
5. M. Reichmann (on behalf of the [RD42 collaboration](https://rd42.web.cern.ch)), *New Test Beam Results of 3D and Pad Detectors Constructed with Poly-Crystalline CVD Diamond*, [15th Vienna Conference on Instrumentation (VCI)](https://vci2019.hephy.at/home/), 2019, Vienna, Austria.
6. D. Sanz (on behalf of the [RD42 collaboration](https://rd42.web.cern.ch)), *Development of Polycrystalline Chemical Vapor Deposition Diamond Detectors for Radiation Monitoring*, [7th International Conference on Radiation in Various Fields of Research (RAD)](https://rad2019.rad-conference.org/welcome.php), 2019, Herceg Novi, Montenegro.
7. M. Reichmann (on behalf of the [RD42 collaboration](https://rd42.web.cern.ch)), *New Beam Test Results of 3D Pixel Detectors Constructed with Poly-Crystalline CVD Diamond*, [29th International Symposium on Lepton Photon Interactions at High Energies](https://indico.cern.ch/event/688643/), 2019, Toronto, Canada. https://indico.cern.ch/event/688643/contributions/3427856/

...more to come.

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
