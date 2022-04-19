''' This is an example how to use a kalman filter to improve tracking by taking into account multiple scattering.
In the example a telescope consisting of six Mimosa26 sensors (18.5 um x 18.5 um) and one ATLAS FE-I4 plane (50 um x 250 um) is considered.
It is assumed, that only at the Mimosa26 sensor multiple scattering occurs. The discrete scattering plane consists in this case of the 50 um thick
Mimosa26 sensor and one 25 um thick kapton foil before and after the sensor. For the RMS scattering angle the standard formula

.. math:: \\theta_\\mathrm{rms} = \\frac{13.6}{\\beta c p} z \\sqrt{\\frac{L}{X_0}} [ 1 + 0.038 \\ln ( \\frac{L}{X_0} ) ]

is used.
Based on this assumption the transition covariance matrix for the Kalman Filter is calculated (for a full
derivation see `here <http://web-docs.gsi.de/~ikisel/reco/Methods/CovarianceMatrices-NIMA329-1993.pdf>`_).

The example plot below shows the true position of the measured hits (green), the smoothed estimated positions determined with the Kalman Filter (blue)
and a simple straight line fit (orange). The hits treated as missing measurements are marked as red. Obviously the straight line fit
describes the track insufficuent since multiple scattering is present. In this case the assumption that the track can be described by a single line
is wrong since the measurements are correlated with each other. Within the Kalman Filter a correlation of the neighboring measurements is taken into account
which then describes the track correctly.

    .. NOTE::
       Missing measurements are supported to obtain unbiased residuals. The respective DUT hit can be masked in the measurement array and
       will then be excluded in the filtering process. In addition to that the Kalman Filter can handle rotated planes (especially around x and y axes)
       by updating the transition matrix before each prediction step in order to take different z-positions of the measurements into account.

Within the Kalman Filter the state vector
:math:`\\boldsymbol{x}_k= \\begin{pmatrix} x_k, y_k, \\tan\\theta_x, \\tan\\theta_y\\end{pmatrix}^{\\mathrm{T}}`,
with :math:`(x_k, y_k)` being the hit position and :math:`(\\tan\\theta_x, \\tan\\theta_y)` being the slopes of the scattered tracks, will be extrapolated linearly from plane :math:`k` to
from plane :math:`k + 1` through

.. math:: \\boldsymbol{x}_k = \\boldsymbol{F}_{k - 1} \\boldsymbol{x}_{k - 1} + \\boldsymbol{w}_{k -1}.

:math:`\\boldsymbol{F}_{k - 1}` denotes the matrix which transports the state vector from plane :math:`k - 1` to plane :math:`k` and
:math:`\\boldsymbol{w}_{k -1}` is the process noise which is in this case due to multiple scattering. Its covariance matrix is denoted by
:math:`\\boldsymbol{Q}_{k}`.
The actual measurements :math:`m_k = \\begin{pmatrix} x_k, y_k \\end{pmatrix}^{\\mathrm{T}}`
are through

.. math:: \\boldsymbol{m}_k = \\boldsymbol{x}_k \\boldsymbol{H}_{k} + \\boldsymbol{\\epsilon}_{k}

connected with state vector. Here, :math:`\\boldsymbol{H}_{k}` connects the measurements with the state vector and :math:`\\boldsymbol{\\epsilon}_{k}`
respresents the measurement noise with its covariance matrix :math:`\\boldsymbol{V}_{k}`.

Further, one can express the predicted estimate :math:`\\boldsymbol{x}_k^{k -1}` of the state vector
which includes all measurements up to but not the :math:`k^\mathrm{th}` plane as follows

.. math:: \\boldsymbol{x}_k^{k -1} = \\boldsymbol{F}_{k -1} \\boldsymbol{x}_{k- 1}^{k -1}.

In case that the expectation value of :math:`(\\boldsymbol{x}_k^k - \\boldsymbol{x}_k)` is zero, the filtered estimate :math:`\\boldsymbol{x}_k^k`
which is the best estimate of the state vector including the measurement at the :math:`k^\mathrm{th}` plane can be written as

.. math:: \\boldsymbol{x}_k^k = \\boldsymbol{x}_k^{k -1} + \\boldsymbol{K}_k  (\\boldsymbol{m}_k - \\boldsymbol{H}_k \\boldsymbol{x}_k^{k-1}).

The matrix :math:`\\boldsymbol{K}_k` is called the Kalman Gain.
Requiring that :math:`\\boldsymbol{K}_k` minimizes the sum of the squares of the std. deviation of the estimated track parameters

.. math:: \\frac{\\partial \mathrm{Tr}(\\boldsymbol{C}_k)}{\\partial \\boldsymbol{K}_k} = 0,

with :math:`\\boldsymbol{C}_k` being the covariance matrix of the estimated track parameters at plane :math:`k`, yields

.. math:: \\boldsymbol{K}_k = \\boldsymbol{C}_k^{k - 1} \\boldsymbol{H}_k^{\\mathrm{T}}(\\boldsymbol{V}_k + \\boldsymbol{H}_k \\boldsymbol{C}_k^{k - 1} \\boldsymbol{H}_k^{\\mathrm{T}})^{-1}.

In analogy to the state vector

.. math:: \\boldsymbol{C}_k^{k} = (\\mathbb{1} - \\boldsymbol{K}_k \\boldsymbol{H}_k) \\boldsymbol{C}_k^{k - 1}

is the covariance of the filtered estimate :math:`\\boldsymbol{x}_k^k` and

.. math:: \\boldsymbol{C}_k^{k - 1} = \\boldsymbol{F}_{k - 1} \\boldsymbol{C}_{k - 1}^{k - 1}\\boldsymbol{F}^\\mathrm{T}_{k -1} + \\boldsymbol{Q}_{k - 1}

is the covariance of the predicted estimate :math:`\\boldsymbol{x}_k^{k- 1}`.

In order to take all measuremnts into account, and not only the past measurements
a Kalman Smoother is applied (in reverse order) to the filtered estimates. The smoothed estimate :math:`\\boldsymbol{x}_k^{n}` (:math:`n > k`)
is given by

.. math:: \\boldsymbol{x}_k^{n} = \\boldsymbol{x}_k^{k} + \\boldsymbol{A}_k ( \\boldsymbol{x}^n_{k + 1} - \\boldsymbol{x}^{k}_{k + 1}),

where

.. math:: \\boldsymbol{A}_k = \\boldsymbol{C}_k^k \\boldsymbol{F}_{k}^\\mathrm{T} (\\boldsymbol{C}^k_{k + 1})^{-1}

is the smoothed Kalman Gain matrix. The covariance matrix of the smooted estimate is finally given by

.. math:: \\boldsymbol{C}^n_k = \\boldsymbol{C}_k^k + \\boldsymbol{A}_k (\\boldsymbol{C}^n_{k + 1} - \\boldsymbol{C}^k_{k + 1}) \\boldsymbol{A}_k^\\mathrm{T}.
'''

import os
import inspect

import numpy as np
from scipy.optimize import curve_fit
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure

from beam_telescope_analysis import track_analysis
from beam_telescope_analysis.telescope.telescope import Telescope


def straight_line(x, slope, offset):
    return slope * x + offset


if __name__ == '__main__':  # Main entry point is needed for multiprocessing under windows

    # Get the absolute path of example data
    tests_data_folder = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')

    # Create output subfolder where all output data and plots are stored
    output_folder = os.path.join(tests_data_folder, 'output_kalman_filter')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    z_positions = np.array([0.0, 29900.0, 60300.0, 82100.0, 118700.0, 160700.0, 197800.0])
    material_budget = [100.0 / 125390.0, 100.0 / 125390.0, 100.0 / 125390.0, 100.0 / 125390.0, 100.0 / 125390.0, 100.0 / 125390.0, 250.0 / 93700]
    telescope = Telescope()
    telescope.add_dut(dut_type="Mimosa26", dut_id=0, translation_x=0, translation_y=0, translation_z=z_positions[0], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[0], name="Telescope 1")
    telescope.add_dut(dut_type="Mimosa26", dut_id=1, translation_x=0, translation_y=0, translation_z=z_positions[1], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[1], name="Telescope 2")
    telescope.add_dut(dut_type="Mimosa26", dut_id=2, translation_x=0, translation_y=0, translation_z=z_positions[2], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[2], name="Telescope 3")
    telescope.add_dut(dut_type="Mimosa26", dut_id=3, translation_x=0, translation_y=0, translation_z=z_positions[3], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[3], name="Telescope 4")
    telescope.add_dut(dut_type="Mimosa26", dut_id=4, translation_x=0, translation_y=0, translation_z=z_positions[4], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[4], name="Telescope 5")
    telescope.add_dut(dut_type="Mimosa26", dut_id=5, translation_x=0, translation_y=0, translation_z=z_positions[5], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[5], name="Telescope 6")
    telescope.add_dut(dut_type="FEI4", dut_id=6, translation_x=0, translation_y=0, translation_z=z_positions[6], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[6], name="FEI4 Reference")

    # pixel resolution: need error on measurements. For real data, the error comes from
    # the cluster position errors which depends on cluster size
    pixel_size = np.array([(18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (250.0, 50.)])
    pixel_resolution = pixel_size / np.sqrt(12)

    # measurements: (x, y, z, x_err, y_err, z_err), data is taken from measurement
    measurements = np.array([[[-1229.22372954, 2828.19616302, 0.0, pixel_resolution[0][0], pixel_resolution[0][1], 0.0],
                              [-1254.51224282, 2827.4291421, 0.0, pixel_resolution[1][0], pixel_resolution[1][1], 0.0],
                              [-1285.6117892, 2822.34536687, 0.0, pixel_resolution[2][0], pixel_resolution[2][1], 0.0],
                              [-1311.31083616, 2823.56121414, 0.0, pixel_resolution[3][0], pixel_resolution[3][1], 0.0],
                              [-1335.8529645, 2828.43359043, 0.0, pixel_resolution[4][0], pixel_resolution[4][1], 0.0],
                              [-1357.81872222, 2840.86947964, 0.0, pixel_resolution[5][0], pixel_resolution[5][1], 0.0],
                              [-1396.35698339, 2843.76799577, 0.0, pixel_resolution[6][0], pixel_resolution[6][1], 0.0]]])

    # select fit DUTs
    select_fit_duts = 126  # E.g. 61 corresponds to 0b111101 which means that DUT 1 and DUT 6 are treated as missing measurement.

    dut_list = np.full(shape=(measurements.shape[1]), fill_value=np.nan)
    for index in range(measurements.shape[1]):
        dut_n = index
        if np.bitwise_and(1 << index, select_fit_duts) == 2 ** index:
            dut_list[dut_n] = dut_n
    # DUTs which are used for fit
    fit_selection = dut_list[~np.isnan(dut_list)].astype(int)
    # DUTs which are not used for fit
    no_fit_selection = list(set(range(len(telescope))) - set(fit_selection))
    offsets, slopes, chi2, _, _, x_errs, y_errs, _, _, _ = track_analysis._fit_tracks_kalman_loop(
        track_hits=measurements,
        telescope=telescope,
        select_fit_duts=fit_selection,
        beam_energy=2500.0,
        particle_mass=0.511,
        scattering_planes=None,
        alpha=np.array([1.0] * len(telescope)))

    # offsets = track_estimates_chunk[:, :len(telescope), :3]
    # slopes = track_estimates_chunk[:, :len(telescope), 3:]

    # interpolate hits with straight line
    fit_x, _ = curve_fit(straight_line, measurements[0, :, 2][fit_selection] / 1000.0, measurements[0, :, 0][fit_selection])
    fit_y, _ = curve_fit(straight_line, measurements[0, :, 2][fit_selection] / 1000.0, measurements[0, :, 1][fit_selection])
    z_range = np.linspace(min(measurements[0, :, 2]), max(measurements[0, :, 2]), 2) / 1000.0

    # plot tracks in x-direction
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_title('Track projection on xz-plane')
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('x [$\mathrm{\mu}$m]')
    ax.grid()

    ax.errorbar(
        z_positions / 1000.0,
        offsets[0, :, 0],
        yerr=x_errs[0],
        marker='o',
        linestyle='-',
        label='Smoothed estimates',
        color='green',
        zorder=3)
    ax.errorbar(
        z_positions[no_fit_selection] / 1000.0,
        offsets[0, :, 0][no_fit_selection],
        yerr=x_errs[0][no_fit_selection],
        marker='o',
        label='Smoothed estimates (not fit)',
        color='indianred',
        zorder=4)
    ax.plot(
        z_positions / 1000.0,
        measurements[0, :, 0],
        marker='o',
        linestyle='-',
        label='Hit positions',
        color='steelblue',
        zorder=2)
    ax.plot(
        z_range,
        straight_line(z_range, fit_x[0], fit_x[1]),
        label='Straight Line Fit',
        color='darkorange',
        zorder=1)
    ax.legend()
    fig.savefig(os.path.join(output_folder, 'kalman_track_x.pdf'))

    # plot tracks in x-direction
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_title('Track projection on yz-plane')
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('y [$\mathrm{\mu}$m]')
    ax.grid()

    ax.errorbar(
        z_positions / 1000.0,
        offsets[0, :, 1],
        yerr=y_errs[0],
        marker='o',
        linestyle='-',
        label='Smoothed estimates',
        color='green',
        zorder=3)
    ax.errorbar(
        z_positions[no_fit_selection] / 1000.0,
        offsets[0, :, 1][no_fit_selection],
        yerr=y_errs[0][no_fit_selection],
        marker='o',
        label='Smoothed estimates (not fit)',
        color='indianred',
        zorder=4)
    ax.plot(
        z_positions / 1000.0,
        measurements[0, :, 1],
        marker='o',
        linestyle='-',
        label='Hit positions',
        color='steelblue',
        zorder=2)
    ax.plot(
        z_range,
        straight_line(z_range, fit_y[0], fit_y[1]),
        label='Straight Line Fit',
        color='darkorange',
        zorder=1)
    ax.legend()
    fig.savefig(os.path.join(output_folder, 'kalman_track_y.pdf'))
