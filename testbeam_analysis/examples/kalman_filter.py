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

import numpy as np
import matplotlib.pyplot as plt
import os
import inspect

from scipy.optimize import curve_fit
from testbeam_analysis import track_analysis


def straight_line(x, slope, offset):
    return slope * x + offset


if __name__ == '__main__':  # Main entry point is needed for multiprocessing under windows

    # Get the absolute path of example data
    tests_data_folder = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')

    # Create output subfolder where all output data and plots are stored
    output_folder = os.path.join(tests_data_folder, 'output_kalman')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # pixel size of sensor
    pixel_size = np.array([(18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (250., 50.)])
    # number of pixels of each sensor
    n_pixels = np.array([(576, 1152), (576, 1152), (576, 1152), (576, 1152), (576, 1152), (576, 1152), (80, 336)])

    # pixel resolution: need error on measurements. For real data, the error comes from
    # the cluster position errors which depends on cluster size
    pixel_resolution = pixel_size / np.sqrt(12)

    # measurements: (x, y, z, xerr, yerr), data is taken from measurement
    measurements = np.array([[[-1229.22372954, 2828.19616302, 0., pixel_resolution[0][0], pixel_resolution[0][1]],
                              [-1254.51224282, 2827.4291421, 29900., pixel_resolution[1][0], pixel_resolution[1][1]],
                              [-1285.6117892, 2822.34536687, 60300., pixel_resolution[2][0], pixel_resolution[2][1]],
                              [-1311.31083616, 2823.56121414, 82100., pixel_resolution[3][0], pixel_resolution[3][1]],
                              [-1335.8529645, 2828.43359043, 118700., pixel_resolution[4][0], pixel_resolution[4][1]],
                              [-1357.81872222, 2840.86947964, 160700., pixel_resolution[5][0], pixel_resolution[5][1]],
                              [-1396.35698339, 2843.76799577, 197800., pixel_resolution[6][0], pixel_resolution[6][1]]]])

    # copy of measurements for plotting, needed if actual measurements contains NaNs
    measurements_plot = np.array([[[-1229.22372954, 2828.19616302, 0.],
                                   [-1254.51224282, 2827.4291421, 29900.],
                                   [-1285.6117892, 2822.34536687, 60300.],
                                   [-1311.31083616, 2823.56121414, 82100.],
                                   [-1335.8529645, 2828.43359043, 118700.],
                                   [-1357.81872222, 2840.86947964, 160700.],
                                   [-1396.35698339, 2843.76799577, 197800.]]])

    # duts to fit tracks for
    fit_duts = np.array([0])
    # dut_fit_selection
    dut_fit_selection = 126  # E.g. 61 corresponds to 0b111101 which means that DUT 1 and DUT 6 are treated as missing meas..
    # intialize arrays for storing data
    track_estimates_chunk_all = np.zeros((fit_duts.shape[0], measurements.shape[1], 4))
    chi2s = np.zeros((fit_duts.shape[0], 1))
    fit_results_x = np.zeros((fit_duts.shape[0], 2))
    fit_results_y = np.zeros((fit_duts.shape[0], 2))
    x_errs_all = np.zeros((fit_duts.shape[0], measurements.shape[1],))
    y_errs_all = np.zeros((fit_duts.shape[0], measurements.shape[1],))
    # select dut tracks to plot
    plot_dut = np.array([0])

    if plot_dut not in fit_duts:
        raise ValueError('Plot DUT is not in fit duts. Please select another DUT.')

    dut_list = np.full(shape=(measurements.shape[1]), fill_value=np.nan)
    for index in range(measurements.shape[1]):
        dut_n = index
        if np.bitwise_and(1 << index, dut_fit_selection) == 2 ** index:
            dut_list[dut_n] = dut_n
    fit_selection = dut_list[~np.isnan(dut_list)].astype(int)

    for fit_dut_index, actual_fit_dut in enumerate(fit_duts):
        track_estimates_chunk, chi2, x_errs, y_errs = track_analysis._fit_tracks_kalman_loop(
            measurements, dut_fit_selection,
            pixel_size, n_pixels, measurements_plot[0, :, -1],
            alignment=None,
            beam_energy=2500.,
            material_budget=[100. / 125390., 100. / 125390., 100. / 125390., 100. / 125390., 100. / 125390., 100. / 125390., 250. / 93700],
            add_scattering_plane=False)
        # interpolate hits with straight line
        fit_x, _ = curve_fit(straight_line, measurements_plot[0, :, -1][fit_selection] / 1000., measurements[0, :, 0][fit_selection])
        fit_y, _ = curve_fit(straight_line, measurements_plot[0, :, -1][fit_selection] / 1000., measurements[0, :, 1][fit_selection])

        # store
        track_estimates_chunk_all[fit_dut_index] = track_estimates_chunk
        chi2s[fit_dut_index] = chi2
        x_errs_all[fit_dut_index] = x_errs
        y_errs_all[fit_dut_index] = y_errs
        fit_results_x[fit_dut_index] = fit_x
        fit_results_y[fit_dut_index] = fit_y

        if actual_fit_dut in plot_dut:  # plot only for selected DUT

            # duts which are unused for fit
            unused_duts = np.setdiff1d(range(measurements_plot.shape[1]), fit_selection)
            x_fit = np.arange(measurements_plot[0, :, -1][0], measurements_plot[0, :, -1][-1], 1.) / 1000.

            # plot tracks in x-direction
            plt.title('Tracks in x-direction for DUT_%d' % plot_dut)
            plt.xlabel('z / mm')
            plt.ylabel('x / $\mathrm{\mu}$m')
            plt.grid()

            plt.errorbar(measurements_plot[0, :, -1] / 1000.,
                         track_estimates_chunk_all[np.where(fit_duts == plot_dut)[0], :, 0].reshape(measurements.shape[1],),
                         yerr=x_errs_all[np.where(fit_duts == plot_dut)[0]].reshape(measurements.shape[1],),
                         marker='o', linestyle='-', label='Smoothed estimates', zorder=2)
            plt.plot(measurements[0, :, -1][unused_duts] / 1000.,
                     track_estimates_chunk_all[np.where(fit_duts == plot_dut)[0], unused_duts, 0].reshape(len(unused_duts),),
                     'o', color='indianred', zorder=4)
            plt.plot(measurements[0, :, -1] / 1000.,
                     measurements_plot[0, :, 0],
                     marker='o', linestyle='-', label='Hit positions', color='green', zorder=3)
            plt.plot(x_fit,
                     straight_line(x_fit, fit_results_x[np.where(fit_duts == plot_dut)[0], 0], fit_results_x[np.where(fit_duts == plot_dut)[0], 1]),
                     label='Straight Line Fit', zorder=1)
            plt.legend()
            plt.savefig(os.path.join(output_folder, 'kalman_tracks_x_DUT%d.pdf' % actual_fit_dut))
            plt.close()

            # plot tracks in y-direction
            plt.title('Tracks in y-direction for DUT_%d' % plot_dut)
            plt.xlabel('z / mm')
            plt.ylabel('y / $\mathrm{\mu}$m')
            plt.grid()
            plt.errorbar(measurements_plot[0, :, -1] / 1000.,
                         track_estimates_chunk_all[np.where(fit_duts == plot_dut)[0], :, 1].reshape(measurements.shape[1],),
                         yerr=y_errs_all[np.where(fit_duts == plot_dut)[0]].reshape(measurements.shape[1],),
                         marker='o', linestyle='-', label='Smoothed estimates', zorder=2)
            plt.plot(measurements_plot[0, :, -1][unused_duts] / 1000.,
                     track_estimates_chunk_all[np.where(fit_duts == plot_dut)[0], unused_duts, 1].reshape(len(unused_duts), ),
                     'o', color='indianred', zorder=4)
            plt.plot(measurements_plot[0, :, -1] / 1000.,
                     measurements_plot[0, :, 1],
                     marker='o', linestyle='-', label='Hit positions', zorder=3)
            plt.plot(x_fit,
                     straight_line(x_fit, fit_results_y[np.where(fit_duts == plot_dut)[0], 0], fit_results_y[np.where(fit_duts == plot_dut)[0], 1]),
                     label='Straight Line Fit', zorder=1)
            plt.legend()
            plt.savefig(os.path.join(output_folder, 'kalman_tracks_y_DUT%d.pdf' % actual_fit_dut))
            plt.close()
