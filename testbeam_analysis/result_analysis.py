''' All functions creating results (e.g. efficiency, residuals, track density) from fitted tracks are listed here.'''
from __future__ import division

import logging
import re
from collections import Iterable
import os.path

import tables as tb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binned_statistic_2d
from scipy.optimize import curve_fit

from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.tools import analysis_utils


def calculate_residuals(input_tracks_file, input_alignment_file, n_pixels, pixel_size, output_residuals_file=None, dut_names=None, use_duts=None, max_chi2=None, nbins_per_pixel=None, npixels_per_bin=None, force_prealignment=False, use_fit_limits=False, cluster_size_selection=None, plot=True, gui=False, chunk_size=1000000):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    input_alignment_file : string
        Filename of the input aligment file.
    n_pixels : iterable of tuples
        One tuple per DUT describing the number of pixels in column, row direction
        e.g. for 2 DUTs: n_pixels = [(80, 336), (80, 336)]
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension in um in column, row direction
        e.g. for 2 DUTs: pixel_size = [(250, 50), (250, 50)]
    output_residuals_file : string
        Filename of the output residuals file. If None, the filename will be derived from the input hits file.
    dut_names : iterable
        Name of the DUTs. If None, DUT numbers will be used.
    use_duts : iterable
        The duts to calculate residuals for. If None all duts in the input_tracks_file are used
    max_chi2 : uint, iterable
        Use only not heavily scattered tracks to increase track pointing resolution (cut on chi2).
        Cut can be a number and is used then for all DUTS or a list with a chi 2 cut for each DUT.
        If None, no cut is applied.
    nbins_per_pixel : int
        Number of bins per pixel along the residual axis. Number is a positive integer or None to automatically set the binning.
    npixels_per_bin : int
        Number of pixels per bin along the position axis. Number is a positive integer or None to automatically set the binning.
    force_prealignment : bool
        Take the prealignment, although if a coarse alignment is availale.
    use_fit_limits : bool
        If True, use fit limits from pre-alignment for selecting fit range for the alignment.
    cluster_size_selection : uint
        Select which cluster sizes should be included for residual calculation. If None all cluster sizes are taken.
    plot : bool
        If True, create additional output plots.
    gui : bool
        If True, use GUI for plotting.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Calculating residuals ===')

    use_prealignment = True if force_prealignment else False

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        if use_prealignment:
            logging.info('Use pre-alignment data')
            prealignment = in_file_h5.root.PreAlignment[:]
            n_duts = prealignment.shape[0]
        else:
            logging.info('Use alignment data')
            alignment = in_file_h5.root.Alignment[:]
            n_duts = alignment.shape[0]
        if use_fit_limits:
            fit_limits = in_file_h5.root.PreAlignment.attrs.fit_limits

    if output_residuals_file is None:
        output_residuals_file = os.path.splitext(input_tracks_file)[0] + '_residuals.h5'

    if plot is True and not gui:
        output_pdf = PdfPages(os.path.splitext(output_residuals_file)[0] + '.pdf', keep_empty=False)
    else:
        output_pdf = None

    figs = [] if gui else None

    if not isinstance(max_chi2, Iterable):
        max_chi2 = [max_chi2] * n_duts

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_residuals_file, mode='w') as out_file_h5:
            for node in in_file_h5.root:
                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.debug('Calculate residuals for DUT%d', actual_dut)

                if use_fit_limits:
                    fit_limit_x_local, fit_limit_y_local = fit_limits[actual_dut][0], fit_limits[actual_dut][1]
                else:
                    fit_limit_x_local = None
                    fit_limit_y_local = None

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    # select good hits and tracks
                    selection = np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut]), ~np.isnan(tracks_chunk['track_chi2']))
                    tracks_chunk = tracks_chunk[selection]  # Take only tracks where actual dut has a hit, otherwise residual wrong
                    if cluster_size_selection is not None:
                        tracks_chunk = tracks_chunk[tracks_chunk['n_hits_dut_%d' % actual_dut] == cluster_size_selection]
                    if max_chi2[actual_dut] is not None:
                        tracks_chunk = tracks_chunk[tracks_chunk['track_chi2'] <= max_chi2[actual_dut]]

                    # Coordinates in global coordinate system (x, y, z)
                    hit_x, hit_y, hit_z = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
                    intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']

                    # Transform to local coordinate system
                    if use_prealignment:
                        hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
                                                                                               dut_index=actual_dut,
                                                                                               prealignment=prealignment,
                                                                                               inverse=True)
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          prealignment=prealignment,
                                                                                                                          inverse=True)
                    else:  # Apply transformation from fine alignment information
                        hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
                                                                                               dut_index=actual_dut,
                                                                                               alignment=alignment,
                                                                                               inverse=True)
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          alignment=alignment,
                                                                                                                          inverse=True)

                    if not np.allclose(hit_z_local, 0.0) or not np.allclose(intersection_z_local, 0.0):
                        logging.error('Hit z position = %s and z intersection %s', str(hit_z_local[:3]), str(intersection_z_local[:3]))
                        raise RuntimeError('The transformation to the local coordinate system did not give all z = 0. Wrong alignment used?')

                    limit_x_selection = np.ones_like(hit_x_local, dtype=np.bool)
                    if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local.min()):
                        limit_x_selection &= hit_x_local >= fit_limit_x_local.min()
                    if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local.max()):
                        limit_x_selection &= hit_x_local <= fit_limit_x_local.max()

                    limit_y_selection = np.ones_like(hit_x_local, dtype=np.bool)
                    if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local.min()):
                        limit_y_selection &= hit_y_local >= fit_limit_y_local.min()
                    if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local.max()):
                        limit_y_selection &= hit_y_local <= fit_limit_y_local.max()

                    limit_xy_selection = np.ones_like(hit_x_local, dtype=np.bool)
                    if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local.min()):
                        limit_xy_selection &= hit_x_local >= fit_limit_x_local.min()
                    if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local.max()):
                        limit_xy_selection &= hit_x_local <= fit_limit_x_local.max()
                    if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local.min()):
                        limit_xy_selection &= hit_y_local >= fit_limit_y_local.min()
                    if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local.max()):
                        limit_xy_selection &= hit_y_local <= fit_limit_y_local.max()

                    hit_x_local_limit_x = hit_x_local[limit_x_selection]
                    hit_y_local_limit_x = hit_y_local[limit_x_selection]
                    hit_z_local_limit_x = hit_z_local[limit_x_selection]
                    intersection_x_local_limit_x = intersection_x_local[limit_x_selection]
                    intersection_y_local_limit_x = intersection_y_local[limit_x_selection]
                    intersection_z_local_limit_x = intersection_z_local[limit_x_selection]

                    hit_x_local_limit_y = hit_x_local[limit_y_selection]
                    hit_y_local_limit_y = hit_y_local[limit_y_selection]
                    hit_z_local_limit_y = hit_z_local[limit_y_selection]
                    intersection_x_local_limit_y = intersection_x_local[limit_y_selection]
                    intersection_y_local_limit_y = intersection_y_local[limit_y_selection]
                    intersection_z_local_limit_y = intersection_z_local[limit_y_selection]

                    hit_x_local_limit_xy = hit_x_local[limit_xy_selection]
                    hit_y_local_limit_xy = hit_y_local[limit_xy_selection]
                    hit_z_local_limit_xy = hit_z_local[limit_xy_selection]
                    intersection_x_local_limit_xy = intersection_x_local[limit_xy_selection]
                    intersection_y_local_limit_xy = intersection_y_local[limit_xy_selection]
                    intersection_z_local_limit_xy = intersection_z_local[limit_xy_selection]

                    difference_local_limit_x = np.column_stack((hit_x_local_limit_x, hit_y_local_limit_x, hit_z_local_limit_x)) - np.column_stack((intersection_x_local_limit_x, intersection_y_local_limit_x, intersection_z_local_limit_x))

                    difference_local_limit_y = np.column_stack((hit_x_local_limit_y, hit_y_local_limit_y, hit_z_local_limit_y)) - np.column_stack((intersection_x_local_limit_y, intersection_y_local_limit_y, intersection_z_local_limit_y))

                    difference_local_limit_xy = np.column_stack((hit_x_local_limit_xy, hit_y_local_limit_xy, hit_z_local_limit_xy)) - np.column_stack((intersection_x_local_limit_xy, intersection_y_local_limit_xy, intersection_z_local_limit_xy))

                    difference = np.column_stack((hit_x, hit_y, hit_z)) - np.column_stack((intersection_x, intersection_y, intersection_z))
                    difference_local = np.column_stack((hit_x_local, hit_y_local, hit_z_local)) - np.column_stack((intersection_x_local, intersection_y_local, intersection_z_local))

                    # Histogram residuals in different ways
                    if initialize:  # Only true for the first iteration, calculate the binning for the histograms
                        initialize = False
                        plot_n_pixels = 6.0

                        # detect peaks and calculate width to estimate the size of the histograms
                        if nbins_per_pixel is not None:
                            min_difference, max_difference = np.min(difference[:, 0]), np.max(difference[:, 0])
                            nbins = np.arange(min_difference - (pixel_size[actual_dut][0] / nbins_per_pixel), max_difference + 2 * (pixel_size[actual_dut][0] / nbins_per_pixel), pixel_size[actual_dut][0] / nbins_per_pixel)
                        else:
                            nbins = "auto"
                        hist, edges = np.histogram(difference[:, 0], bins=nbins)
                        edge_center = (edges[1:] + edges[:-1]) / 2.0
                        try:
                            _, center_x, fwhm_x, _ = analysis_utils.peak_detect(edge_center, hist)
                        except RuntimeError:
                            # do some simple FWHM with numpy array
                            try:
                                _, center_x, fwhm_x, _ = analysis_utils.simple_peak_detect(edge_center, hist)
                            except RuntimeError:
                                center_x, fwhm_x = 0.0, pixel_size[actual_dut][0] * plot_n_pixels

                        if nbins_per_pixel is not None:
                            min_difference, max_difference = np.min(difference[:, 1]), np.max(difference[:, 1])
                            nbins = np.arange(min_difference - (pixel_size[actual_dut][1] / nbins_per_pixel), max_difference + 2 * (pixel_size[actual_dut][1] / nbins_per_pixel), pixel_size[actual_dut][1] / nbins_per_pixel)
                        else:
                            nbins = "auto"
                        hist, edges = np.histogram(difference[:, 1], bins=nbins)
                        edge_center = (edges[1:] + edges[:-1]) / 2.0
                        try:
                            _, center_y, fwhm_y, _ = analysis_utils.peak_detect(edge_center, hist)
                        except RuntimeError:
                            # do some simple FWHM with numpy array
                            try:
                                _, center_y, fwhm_y, _ = analysis_utils.simple_peak_detect(edge_center, hist)
                            except RuntimeError:
                                center_y, fwhm_y = 0.0, pixel_size[actual_dut][1] * plot_n_pixels

                        if nbins_per_pixel is not None:
                            min_difference, max_difference = np.min(difference_local[:, 0]), np.max(difference_local[:, 0])
                            nbins = np.arange(min_difference - (pixel_size[actual_dut][0] / nbins_per_pixel), max_difference + 2 * (pixel_size[actual_dut][0] / nbins_per_pixel), pixel_size[actual_dut][0] / nbins_per_pixel)
                        else:
                            nbins = "auto"
                        hist, edges = np.histogram(difference_local[:, 0], bins=nbins)
                        edge_center = (edges[1:] + edges[:-1]) / 2.0
                        try:
                            _, center_col, fwhm_col, _ = analysis_utils.peak_detect(edge_center, hist)
                        except RuntimeError:
                            # do some simple FWHM with numpy array
                            try:
                                _, center_col, fwhm_col, _ = analysis_utils.simple_peak_detect(edge_center, hist)
                            except RuntimeError:
                                center_col, fwhm_col = 0.0, pixel_size[actual_dut][0] * plot_n_pixels

                        if nbins_per_pixel is not None:
                            min_difference, max_difference = np.min(difference_local[:, 1]), np.max(difference_local[:, 1])
                            nbins = np.arange(min_difference - (pixel_size[actual_dut][1] / nbins_per_pixel), max_difference + 2 * (pixel_size[actual_dut][1] / nbins_per_pixel), pixel_size[actual_dut][1] / nbins_per_pixel)
                        else:
                            nbins = "auto"
                        hist, edges = np.histogram(difference_local[:, 1], bins=nbins)
                        edge_center = (edges[1:] + edges[:-1]) / 2.0
                        try:
                            _, center_row, fwhm_row, _ = analysis_utils.peak_detect(edge_center, hist)
                        except RuntimeError:
                            # do some simple FWHM with numpy array
                            try:
                                _, center_row, fwhm_row, _ = analysis_utils.simple_peak_detect(edge_center, hist)
                            except RuntimeError:
                                center_row, fwhm_row = 0.0, pixel_size[actual_dut][1] * plot_n_pixels

                        # calculate the binning of the histograms, the minimum size is given by plot_n_pixels, otherwise FWHM is taken into account
                        if nbins_per_pixel is not None:
                            width = max(plot_n_pixels * pixel_size[actual_dut][0], pixel_size[actual_dut][0] * np.ceil(plot_n_pixels * fwhm_x / pixel_size[actual_dut][0]))
                            if np.mod(width / pixel_size[actual_dut][0], 2) != 0:
                                width += pixel_size[actual_dut][0]
                            nbins = int(nbins_per_pixel * width / pixel_size[actual_dut][0])
                            x_range = (center_x - 0.5 * width, center_x + 0.5 * width)
                        else:
                            nbins = "auto"
                            width = pixel_size[actual_dut][0] * np.ceil(plot_n_pixels * fwhm_x / pixel_size[actual_dut][0])
                            x_range = (center_x - width, center_x + width)
                        hist_residual_x_hist, hist_residual_x_xedges = np.histogram(difference[:, 0], range=x_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_x), np.max(intersection_x)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][0], npixels_per_bin * pixel_size[actual_dut][0])
                        else:
                            nbins = "auto"
                        _, hist_residual_x_yedges = np.histogram(intersection_x, bins=nbins)

                        if nbins_per_pixel is not None:
                            width = max(plot_n_pixels * pixel_size[actual_dut][1], pixel_size[actual_dut][1] * np.ceil(plot_n_pixels * fwhm_y / pixel_size[actual_dut][1]))
                            if np.mod(width / pixel_size[actual_dut][1], 2) != 0:
                                width += pixel_size[actual_dut][1]
                            nbins = int(nbins_per_pixel * width / pixel_size[actual_dut][1])
                            y_range = (center_y - 0.5 * width, center_y + 0.5 * width)
                        else:
                            nbins = "auto"
                            width = pixel_size[actual_dut][1] * np.ceil(plot_n_pixels * fwhm_y / pixel_size[actual_dut][1])
                            y_range = (center_y - width, center_y + width)
                        hist_residual_y_hist, hist_residual_y_yedges = np.histogram(difference[:, 1], range=y_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_y), np.max(intersection_y)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][1], npixels_per_bin * pixel_size[actual_dut][1])
                        else:
                            nbins = "auto"
                        _, hist_residual_y_xedges = np.histogram(intersection_y, bins=nbins)

                        if nbins_per_pixel is not None:
                            width = max(plot_n_pixels * pixel_size[actual_dut][0], pixel_size[actual_dut][0] * np.ceil(plot_n_pixels * fwhm_col / pixel_size[actual_dut][0]))
                            if np.mod(width / pixel_size[actual_dut][0], 2) != 0:
                                width += pixel_size[actual_dut][0]
                            nbins = int(nbins_per_pixel * width / pixel_size[actual_dut][0])
                            col_range = (center_col - 0.5 * width, center_col + 0.5 * width)
                        else:
                            nbins = "auto"
                            width = pixel_size[actual_dut][0] * np.ceil(plot_n_pixels * fwhm_col / pixel_size[actual_dut][0])
                            col_range = (center_col - width, center_col + width)
                        hist_residual_col_hist, hist_residual_col_xedges = np.histogram(difference_local_limit_xy[:, 0], range=col_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_x_local), np.max(intersection_x_local)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][0], npixels_per_bin * pixel_size[actual_dut][0])
                        else:
                            nbins = "auto"
                        _, hist_residual_col_yedges = np.histogram(intersection_x_local, bins=nbins)

                        if nbins_per_pixel is not None:
                            width = max(plot_n_pixels * pixel_size[actual_dut][1], pixel_size[actual_dut][1] * np.ceil(plot_n_pixels * fwhm_row / pixel_size[actual_dut][1]))
                            if np.mod(width / pixel_size[actual_dut][1], 2) != 0:
                                width += pixel_size[actual_dut][1]
                            nbins = int(nbins_per_pixel * width / pixel_size[actual_dut][1])
                            row_range = (center_row - 0.5 * width, center_row + 0.5 * width)
                        else:
                            nbins = "auto"
                            width = pixel_size[actual_dut][1] * np.ceil(plot_n_pixels * fwhm_row / pixel_size[actual_dut][1])
                            row_range = (center_row - width, center_row + width)
                        hist_residual_row_hist, hist_residual_row_yedges = np.histogram(difference_local_limit_xy[:, 1], range=row_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_y_local), np.max(intersection_y_local)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][1], npixels_per_bin * pixel_size[actual_dut][1])
                        else:
                            nbins = "auto"
                        _, hist_residual_row_xedges = np.histogram(intersection_y_local, bins=nbins)

                        # global x residual against x position
                        hist_x_residual_x_hist, hist_x_residual_x_xedges, hist_x_residual_x_yedges = np.histogram2d(
                            intersection_x,
                            difference[:, 0],
                            bins=(hist_residual_x_yedges, hist_residual_x_xedges))

                        # global y residual against y position
                        hist_y_residual_y_hist, hist_y_residual_y_xedges, hist_y_residual_y_yedges = np.histogram2d(
                            intersection_y,
                            difference[:, 1],
                            bins=(hist_residual_y_xedges, hist_residual_y_yedges))

                        # global y residual against x position
                        hist_x_residual_y_hist, hist_x_residual_y_xedges, hist_x_residual_y_yedges = np.histogram2d(
                            intersection_x,
                            difference[:, 1],
                            bins=(hist_residual_x_yedges, hist_residual_y_yedges))

                        # global x residual against y position
                        hist_y_residual_x_hist, hist_y_residual_x_xedges, hist_y_residual_x_yedges = np.histogram2d(
                            intersection_y,
                            difference[:, 0],
                            bins=(hist_residual_y_xedges, hist_residual_x_xedges))

                        # local column residual against column position
                        hist_col_residual_col_hist, hist_col_residual_col_xedges, hist_col_residual_col_yedges = np.histogram2d(
                            intersection_x_local,
                            difference_local[:, 0],
                            bins=(hist_residual_col_yedges, hist_residual_col_xedges))

                        # local row residual against row position
                        hist_row_residual_row_hist, hist_row_residual_row_xedges, hist_row_residual_row_yedges = np.histogram2d(
                            intersection_y_local,
                            difference_local[:, 1],
                            bins=(hist_residual_row_xedges, hist_residual_row_yedges))

                        # local row residual against column position
                        hist_col_residual_row_hist, hist_col_residual_row_xedges, hist_col_residual_row_yedges = np.histogram2d(
                            intersection_x_local_limit_y,
                            difference_local_limit_y[:, 1],
                            bins=(hist_residual_col_yedges, hist_residual_row_yedges))

                        # local column residual against row position
                        hist_row_residual_col_hist, hist_row_residual_col_xedges, hist_row_residual_col_yedges = np.histogram2d(
                            intersection_y_local_limit_x,
                            difference_local_limit_x[:, 0],
                            bins=(hist_residual_row_xedges, hist_residual_col_xedges))

                    else:  # adding data to existing histograms
                        hist_residual_x_hist += np.histogram(difference[:, 0], bins=hist_residual_x_xedges)[0]
                        hist_residual_y_hist += np.histogram(difference[:, 1], bins=hist_residual_y_yedges)[0]
                        hist_residual_col_hist += np.histogram(difference_local_limit_xy[:, 0], bins=hist_residual_col_xedges)[0]
                        hist_residual_row_hist += np.histogram(difference_local_limit_xy[:, 1], bins=hist_residual_row_yedges)[0]

                        # global x residual against x position
                        hist_x_residual_x_hist += np.histogram2d(
                            intersection_x,
                            difference[:, 0],
                            bins=(hist_x_residual_x_xedges, hist_x_residual_x_yedges))[0]

                        # global y residual against y position
                        hist_y_residual_y_hist += np.histogram2d(
                            intersection_y,
                            difference[:, 1],
                            bins=(hist_y_residual_y_xedges, hist_y_residual_y_yedges))[0]

                        # global y residual against x position
                        hist_x_residual_y_hist += np.histogram2d(
                            intersection_x,
                            difference[:, 1],
                            bins=(hist_x_residual_y_xedges, hist_x_residual_y_yedges))[0]

                        # global x residual against y position
                        hist_y_residual_x_hist += np.histogram2d(
                            intersection_y,
                            difference[:, 0],
                            bins=(hist_y_residual_x_xedges, hist_y_residual_x_yedges))[0]

                        # local column residual against column position
                        hist_col_residual_col_hist += np.histogram2d(
                            intersection_x_local,
                            difference_local[:, 0],
                            bins=(hist_col_residual_col_xedges, hist_col_residual_col_yedges))[0]

                        # local row residual against row position
                        hist_row_residual_row_hist += np.histogram2d(
                            intersection_y_local,
                            difference_local[:, 1],
                            bins=(hist_row_residual_row_xedges, hist_row_residual_row_yedges))[0]

                        # local row residual against column position
                        hist_col_residual_row_hist += np.histogram2d(
                            intersection_x_local_limit_y,
                            difference_local_limit_y[:, 1],
                            bins=(hist_col_residual_row_xedges, hist_col_residual_row_yedges))[0]

                        # local column residual against row position
                        hist_row_residual_col_hist += np.histogram2d(
                            intersection_y_local_limit_x,
                            difference_local_limit_x[:, 0],
                            bins=(hist_row_residual_col_xedges, hist_row_residual_col_yedges))[0]

                logging.debug('Storing residual histograms...')

                dut_name = dut_names[actual_dut] if dut_names else ("DUT" + str(actual_dut))
                # Global residuals
                fit_residual_x, cov_residual_x = analysis_utils.fit_residuals(
                    hist=hist_residual_x_hist,
                    edges=hist_residual_x_xedges,
                    label='X residual [um]',
                    title='X residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_res_x = out_file_h5.create_carray(out_file_h5.root,
                                                      name='ResidualsX_DUT%d' % (actual_dut),
                                                      title='Residual distribution in x direction for %s' % (dut_name),
                                                      atom=tb.Atom.from_dtype(hist_residual_x_hist.dtype),
                                                      shape=hist_residual_x_hist.shape,
                                                      filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_x.attrs.xedges = hist_residual_x_xedges
                out_res_x.attrs.fit_coeff = fit_residual_x
                out_res_x.attrs.fit_cov = cov_residual_x
                out_res_x[:] = hist_residual_x_hist

                fit_residual_y, cov_residual_y = analysis_utils.fit_residuals(
                    hist=hist_residual_y_hist,
                    edges=hist_residual_y_yedges,
                    label='Y residual [um]',
                    title='Y residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_res_y = out_file_h5.create_carray(out_file_h5.root,
                                                      name='ResidualsY_DUT%d' % (actual_dut),
                                                      title='Residual distribution in y direction for %s' % (dut_name),
                                                      atom=tb.Atom.from_dtype(hist_residual_y_hist.dtype),
                                                      shape=hist_residual_y_hist.shape,
                                                      filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_y.attrs.yedges = hist_residual_y_yedges
                out_res_y.attrs.fit_coeff = fit_residual_y
                out_res_y.attrs.fit_cov = cov_residual_y
                out_res_y[:] = hist_residual_y_hist

                fit_x_residual_x, cov_x_residual_x = analysis_utils.fit_residuals_vs_position(
                    hist=hist_x_residual_x_hist,
                    xedges=hist_x_residual_x_xedges,
                    yedges=hist_x_residual_x_yedges,
                    xlabel='X position [um]',
                    ylabel='X residual [um]',
                    title='X residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_x_res_x = out_file_h5.create_carray(out_file_h5.root,
                                                        name='XResidualsX_DUT%d' % (actual_dut),
                                                        title='Residual distribution in x direction as a function of the x position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_x_residual_x_hist.dtype),
                                                        shape=hist_x_residual_x_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_x_res_x.attrs.xedges = hist_x_residual_x_xedges
                out_x_res_x.attrs.yedges = hist_x_residual_x_yedges
                out_x_res_x.attrs.fit_coeff = fit_x_residual_x
                out_x_res_x.attrs.fit_cov = cov_x_residual_x
                out_x_res_x[:] = hist_x_residual_x_hist

                fit_y_residual_y, cov_y_residual_y = analysis_utils.fit_residuals_vs_position(
                    hist=hist_y_residual_y_hist,
                    xedges=hist_y_residual_y_xedges,
                    yedges=hist_y_residual_y_yedges,
                    xlabel='Y position [um]',
                    ylabel='Y residual [um]',
                    title='Y residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_y_res_y = out_file_h5.create_carray(out_file_h5.root,
                                                        name='YResidualsY_DUT%d' % (actual_dut),
                                                        title='Residual distribution in y direction as a function of the y position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_y_residual_y_hist.dtype),
                                                        shape=hist_y_residual_y_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_y_res_y.attrs.xedges = hist_y_residual_y_xedges
                out_y_res_y.attrs.yedges = hist_y_residual_y_yedges
                out_y_res_y.attrs.fit_coeff = fit_y_residual_y
                out_y_res_y.attrs.fit_cov = cov_y_residual_y
                out_y_res_y[:] = hist_y_residual_y_hist

                fit_x_residual_y, cov_x_residual_y = analysis_utils.fit_residuals_vs_position(
                    hist=hist_x_residual_y_hist,
                    xedges=hist_x_residual_y_xedges,
                    yedges=hist_x_residual_y_yedges,
                    xlabel='X position [um]',
                    ylabel='Y residual [um]',
                    title='Y residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_x_res_y = out_file_h5.create_carray(out_file_h5.root,
                                                        name='XResidualsY_DUT%d' % (actual_dut),
                                                        title='Residual distribution in y direction as a function of the x position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_x_residual_y_hist.dtype),
                                                        shape=hist_x_residual_y_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_x_res_y.attrs.xedges = hist_x_residual_y_xedges
                out_x_res_y.attrs.yedges = hist_x_residual_y_yedges
                out_x_res_y.attrs.fit_coeff = fit_x_residual_y
                out_x_res_y.attrs.fit_cov = cov_x_residual_y
                out_x_res_y[:] = hist_x_residual_y_hist

                fit_y_residual_x, cov_y_residual_x = analysis_utils.fit_residuals_vs_position(
                    hist=hist_y_residual_x_hist,
                    xedges=hist_y_residual_x_xedges,
                    yedges=hist_y_residual_x_yedges,
                    xlabel='Y position [um]',
                    ylabel='X residual [um]',
                    title='X residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_y_res_x = out_file_h5.create_carray(out_file_h5.root,
                                                        name='YResidualsX_DUT%d' % (actual_dut),
                                                        title='Residual distribution in x direction as a function of the y position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_y_residual_x_hist.dtype),
                                                        shape=hist_y_residual_x_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_y_res_x.attrs.xedges = hist_y_residual_x_xedges
                out_y_res_x.attrs.yedges = hist_y_residual_x_yedges
                out_y_res_x.attrs.fit_coeff = fit_y_residual_x
                out_y_res_x.attrs.fit_cov = cov_y_residual_x
                out_y_res_x[:] = hist_y_residual_x_hist

                # Local residuals
                fit_residual_col, cov_residual_col = analysis_utils.fit_residuals(
                    hist=hist_residual_col_hist,
                    edges=hist_residual_col_xedges,
                    label='Column residual [um]',
                    title='Column residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_res_col = out_file_h5.create_carray(out_file_h5.root,
                                                        name='ResidualsCol_DUT%d' % (actual_dut),
                                                        title='Residual distribution in column direction for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_residual_col_hist.dtype),
                                                        shape=hist_residual_col_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_col.attrs.xedges = hist_residual_col_xedges
                out_res_col.attrs.fit_coeff = fit_residual_col
                out_res_col.attrs.fit_cov = cov_residual_col
                out_res_col[:] = hist_residual_col_hist

                fit_residual_row, cov_residual_row = analysis_utils.fit_residuals(
                    hist=hist_residual_row_hist,
                    edges=hist_residual_row_yedges,
                    label='Row residual [um]',
                    title='Row residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_res_row = out_file_h5.create_carray(out_file_h5.root,
                                                        name='ResidualsRow_DUT%d' % (actual_dut),
                                                        title='Residual distribution in row direction for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(hist_residual_row_hist.dtype),
                                                        shape=hist_residual_row_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_res_row.attrs.yedges = hist_residual_row_yedges
                out_res_row.attrs.fit_coeff = fit_residual_row
                out_res_row.attrs.fit_cov = cov_residual_row
                out_res_row[:] = hist_residual_row_hist

                fit_col_residual_col, cov_col_residual_col = analysis_utils.fit_residuals_vs_position(
                    hist=hist_col_residual_col_hist,
                    xedges=hist_col_residual_col_xedges,
                    yedges=hist_col_residual_col_yedges,
                    xlabel='Column position [um]',
                    ylabel='Column residual [um]',
                    title='Column residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    fit_limit=fit_limit_x_local,
                    gui=gui,
                    figs=figs
                )
                out_col_res_col = out_file_h5.create_carray(out_file_h5.root,
                                                            name='ColResidualsCol_DUT%d' % (actual_dut),
                                                            title='Residual distribution in column direction as a function of the column position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(hist_col_residual_col_hist.dtype),
                                                            shape=hist_col_residual_col_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col_res_col.attrs.xedges = hist_col_residual_col_xedges
                out_col_res_col.attrs.yedges = hist_col_residual_col_yedges
                out_col_res_col.attrs.fit_coeff = fit_col_residual_col
                out_col_res_col.attrs.fit_cov = cov_col_residual_col
                out_col_res_col[:] = hist_col_residual_col_hist

                fit_row_residual_row, cov_row_residual_row = analysis_utils.fit_residuals_vs_position(
                    hist=hist_row_residual_row_hist,
                    xedges=hist_row_residual_row_xedges,
                    yedges=hist_row_residual_row_yedges,
                    xlabel='Row position [um]',
                    ylabel='Row residual [um]',
                    title='Row residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    fit_limit=fit_limit_y_local,
                    gui=gui,
                    figs=figs
                )
                out_row_res_row = out_file_h5.create_carray(out_file_h5.root,
                                                            name='RowResidualsRow_DUT%d' % (actual_dut),
                                                            title='Residual distribution in row direction as a function of the row position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(hist_row_residual_row_hist.dtype),
                                                            shape=hist_row_residual_row_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_row_res_row.attrs.xedges = hist_row_residual_row_xedges
                out_row_res_row.attrs.yedges = hist_row_residual_row_yedges
                out_row_res_row.attrs.fit_coeff = fit_row_residual_row
                out_row_res_row.attrs.fit_cov = cov_row_residual_row
                out_row_res_row[:] = hist_row_residual_row_hist

                fit_col_residual_row, cov_col_residual_row = analysis_utils.fit_residuals_vs_position(
                    hist=hist_col_residual_row_hist,
                    xedges=hist_col_residual_row_xedges,
                    yedges=hist_col_residual_row_yedges,
                    xlabel='Column position [um]',
                    ylabel='Row residual [um]',
                    title='Row residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    fit_limit=fit_limit_x_local,
                    gui=gui,
                    figs=figs
                )
                out_col_res_row = out_file_h5.create_carray(out_file_h5.root,
                                                            name='ColResidualsRow_DUT%d' % (actual_dut),
                                                            title='Residual distribution in row direction as a function of the column position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(hist_col_residual_row_hist.dtype),
                                                            shape=hist_col_residual_row_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col_res_row.attrs.xedges = hist_col_residual_row_xedges
                out_col_res_row.attrs.yedges = hist_col_residual_row_yedges
                out_col_res_row.attrs.fit_coeff = fit_col_residual_row
                out_col_res_row.attrs.fit_cov = cov_col_residual_row
                out_col_res_row[:] = hist_col_residual_row_hist

                fit_row_residual_col, cov_row_residual_col = analysis_utils.fit_residuals_vs_position(
                    hist=hist_row_residual_col_hist,
                    xedges=hist_row_residual_col_xedges,
                    yedges=hist_row_residual_col_yedges,
                    xlabel='Row position [um]',
                    ylabel='Column residual [um]',
                    title='Column residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    fit_limit=fit_limit_y_local,
                    gui=gui,
                    figs=figs
                )
                out_row_res_col = out_file_h5.create_carray(out_file_h5.root,
                                                            name='RowResidualsCol_DUT%d' % (actual_dut),
                                                            title='Residual distribution in column direction as a function of the row position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(hist_row_residual_col_hist.dtype),
                                                            shape=hist_row_residual_col_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_row_res_col.attrs.xedges = hist_row_residual_col_xedges
                out_row_res_col.attrs.yedges = hist_row_residual_col_yedges
                out_row_res_col.attrs.fit_coeff = fit_row_residual_col
                out_row_res_col.attrs.fit_cov = cov_row_residual_col
                out_row_res_col[:] = hist_row_residual_col_hist

    if output_pdf is not None:
        output_pdf.close()

    if gui:
        return figs


def calculate_efficiency(input_tracks_file, input_alignment_file, bin_size, sensor_size, output_efficiency_file=None, pixel_size=None, n_pixels=None, minimum_track_density=1, max_distance=500, use_duts=None, max_chi2=None, force_prealignment=False, cut_distance=None, col_range=None, row_range=None, show_inefficient_events=False, plot=True, gui=False, chunk_size=1000000):
    '''Takes the tracks and calculates the hit efficiency and hit/track hit distance for selected DUTs.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    input_alignment_file : string
        Filename of the input alignment file.
    bin_size : iterable
        Sizes of bins (i.e. (virtual) pixel size). Give one tuple (x, y) for every plane or list of tuples for different planes.
    sensor_size : Tuple or list of tuples
        Describes the sensor size for each DUT. If one tuple is given it is (size x, size y)
        If several tuples are given it is [(DUT0 size x, DUT0 size y), (DUT1 size x, DUT1 size y), ...]
    output_efficiency_file : string
        Filename of the output efficiency file. If None, the filename will be derived from the input hits file.
    minimum_track_density : int
        Minimum track density required to consider bin for efficiency calculation.
    use_duts : iterable
        Calculate the efficiency for selected DUTs. If None, all duts are selected.
    max_chi2 : uint
        Only use tracks with a chi2 <= max_chi2.
    force_prealignment : bool
        Take the prealignment, although if a coarse alignment is availale.
    cut_distance : int
        Use only distances (between DUT hit and track hit) smaller than cut_distance.
    max_distance : int
        Defines binnig of distance values.
    col_range : iterable
        Column value to calculate efficiency for (to neglect noisy edge pixels for efficiency calculation).
    row_range : iterable
        Row value to calculate efficiency for (to neglect noisy edge pixels for efficiency calculation).
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.
    pixel_size : iterable
        tuple or list of col/row pixel dimension
    n_pixels : iterable
        tuple or list of amount of pixel in col/row dimension
    show_inefficient_events : bool
        Whether to log inefficient events
    gui : bool
        If True, use GUI for plotting.
    '''
    logging.info('=== Calculating efficiency ===')

    if output_efficiency_file is None:
        output_efficiency_file = os.path.splitext(input_tracks_file)[0] + '_efficiency.h5'

    if plot is True and not gui:
        output_pdf = PdfPages(os.path.splitext(output_efficiency_file)[0] + '.pdf', keep_empty=False)
    else:
        output_pdf = None

    use_prealignment = True if force_prealignment else False

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        if use_prealignment:
            logging.info('Use pre-alignment data')
            prealignment = in_file_h5.root.PreAlignment[:]
            n_duts = prealignment.shape[0]
        else:
            logging.info('Use alignment data')
            alignment = in_file_h5.root.Alignment[:]
            n_duts = alignment.shape[0]

    use_duts = use_duts if use_duts is not None else range(n_duts)  # standard setting: fit tracks for all DUTs

    if not isinstance(max_chi2, Iterable):
        max_chi2 = [max_chi2] * len(use_duts)

    efficiencies = []
    pass_tracks = []
    total_tracks = []
    figs = [] if gui else None
    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_efficiency_file, 'w') as out_file_h5:
            for index, node in enumerate(in_file_h5.root):
                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                if actual_dut not in use_duts:
                    continue
                dut_index = np.where(np.array(use_duts) == actual_dut)[0][0]
                logging.info('Calculate efficiency for DUT%d', actual_dut)

                # Calculate histogram properties (bins size and number of bins)
                bin_size = [bin_size, ] if not isinstance(bin_size, Iterable) else bin_size
                if len(bin_size) == 1:
                    actual_bin_size_x = bin_size[0][0]
                    actual_bin_size_y = bin_size[0][1]
                else:
                    actual_bin_size_x = bin_size[dut_index][0]
                    actual_bin_size_y = bin_size[dut_index][1]

                dimensions = [sensor_size, ] if not isinstance(sensor_size, Iterable) else sensor_size  # Sensor dimensions for each DUT
                if len(dimensions) == 1:
                    dimensions = dimensions[0]
                else:
                    dimensions = dimensions[dut_index]

                n_bin_x = int(dimensions[0] / actual_bin_size_x)
                n_bin_y = int(dimensions[1] / actual_bin_size_y)

                # Define result histograms, these are filled for each hit chunk
#                 total_distance_array = np.zeros(shape=(n_bin_x, n_bin_y, max_distance))
                total_hit_hist = np.zeros(shape=(n_bin_x, n_bin_y), dtype=np.uint32)
                total_track_density = np.zeros(shape=(n_bin_x, n_bin_y))
                total_track_density_with_DUT_hit = np.zeros(shape=(n_bin_x, n_bin_y))

                actual_max_chi2 = max_chi2[dut_index]

                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    # Cut in Chi 2 of the track fit
                    if actual_max_chi2:
                        tracks_chunk = tracks_chunk[tracks_chunk['track_chi2'] <= max_chi2]

                    # Transform the hits and track intersections into the local coordinate system
                    # Coordinates in global coordinate system (x, y, z)
                    hit_x, hit_y, hit_z = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
                    intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']

                    # Transform to local coordinate system
                    if use_prealignment:
                        hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
                                                                                               dut_index=actual_dut,
                                                                                               prealignment=prealignment,
                                                                                               inverse=True)
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          prealignment=prealignment,
                                                                                                                          inverse=True)
                    else:  # Apply transformation from alignment information
                        hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
                                                                                               dut_index=actual_dut,
                                                                                               alignment=alignment,
                                                                                               inverse=True)
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          alignment=alignment,
                                                                                                                          inverse=True)

                    # Quickfix that center of sensor is local system is in the center and not at the edge
                    hit_x_local, hit_y_local = hit_x_local + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], hit_y_local + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]
                    intersection_x_local, intersection_y_local = intersection_x_local + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], intersection_y_local + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]

                    intersections_local = np.column_stack((intersection_x_local, intersection_y_local, intersection_z_local))
                    hits_local = np.column_stack((hit_x_local, hit_y_local, hit_z_local))

                    if not np.allclose(hits_local[np.isfinite(hits_local[:, 2]), 2], 0.0) or not np.allclose(intersection_z_local, 0.0):
                        raise RuntimeError('The transformation to the local coordinate system did not give all z = 0. Wrong alignment used?')

                    # Usefull for debugging, print some inefficient events that can be cross checked
                    # Select virtual hits
                    sel_virtual = np.isnan(tracks_chunk['x_dut_%d' % actual_dut])
                    if show_inefficient_events:
                        logging.info('These events are inefficient: %s', str(tracks_chunk['event_number'][sel_virtual]))

                    # Select hits from column, row range (e.g. to supress edge pixels)
                    col_range = [col_range, ] if not isinstance(col_range, Iterable) else col_range
                    if len(col_range) == 1:
                        curr_col_range = col_range[0]
                    else:
                        curr_col_range = col_range[dut_index]
                    if curr_col_range is not None:
                        selection = np.logical_and(intersections_local[:, 0] >= curr_col_range[0], intersections_local[:, 0] <= curr_col_range[1])  # Select real hits
                        hits_local, intersections_local = hits_local[selection], intersections_local[selection]

                    row_range = [row_range, ] if not isinstance(row_range, Iterable) else row_range
                    if len(row_range) == 1:
                        curr_row_range = row_range[0]
                    else:
                        curr_row_range = row_range[dut_index]
                    if curr_row_range is not None:
                        selection = np.logical_and(intersections_local[:, 1] >= curr_row_range[0], intersections_local[:, 1] <= curr_row_range[1])  # Select real hits
                        hits_local, intersections_local = hits_local[selection], intersections_local[selection]

                    # Calculate distance between track hit and DUT hit
                    scale = np.square(np.array((1, 1, 0)))  # Regard pixel size for calculating distances
                    distance = np.sqrt(np.dot(np.square(intersections_local - hits_local), scale))  # Array with distances between DUT hit and track hit for each event. Values in um

                    col_row_distance = np.column_stack((hits_local[:, 0], hits_local[:, 1], distance))

#                     total_distance_array += np.histogramdd(col_row_distance, bins=(n_bin_x, n_bin_y, max_distance), range=[[0, dimensions[0]], [0, dimensions[1]], [0, max_distance]])[0]
                    total_hit_hist += (np.histogram2d(hits_local[:, 0], hits_local[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]).astype(np.uint32)
#                     total_hit_hist += (np.histogram2d(hits_local[:, 0], hits_local[:, 1], bins=(n_bin_x, n_bin_y), range=[[-dimensions[0] / 2., dimensions[0] / 2.], [-dimensions[1] / 2., dimensions[1] / 2.]])[0]).astype(np.uint32)

                    # Calculate efficiency
                    selection = ~np.isnan(hits_local[:, 0])
                    if cut_distance:  # Select intersections where hit is in given distance around track intersection
                        intersection_valid_hit = intersections_local[np.logical_and(selection, distance < cut_distance)]
                    else:
                        intersection_valid_hit = intersections_local[selection]

                    total_track_density += np.histogram2d(intersections_local[:, 0], intersections_local[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]
                    total_track_density_with_DUT_hit += np.histogram2d(intersection_valid_hit[:, 0], intersection_valid_hit[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]

                    if np.all(total_track_density == 0):
                        logging.warning('No tracks on DUT%d, cannot calculate efficiency', actual_dut)
                        continue

                efficiency = np.zeros_like(total_track_density_with_DUT_hit)
                efficiency[total_track_density != 0] = total_track_density_with_DUT_hit[total_track_density != 0].astype(np.float) / total_track_density[total_track_density != 0].astype(np.float) * 100.

                efficiency = np.ma.array(efficiency, mask=total_track_density < minimum_track_density)

                if not np.any(efficiency):
                    raise RuntimeError('All efficiencies for DUT%d are zero, consider changing cut values!', actual_dut)

                # Calculate distances between hit and intersection
#                 distance_mean_array = np.average(total_distance_array, axis=2, weights=range(0, max_distance)) * sum(range(0, max_distance)) / total_hit_hist.astype(np.float)
#                 distance_mean_array = np.ma.masked_invalid(distance_mean_array)
#                 distance_max_array = np.amax(total_distance_array, axis=2) * sum(range(0, max_distance)) / total_hit_hist.astype(np.float)
#                 distance_min_array = np.amin(total_distance_array, axis=2) * sum(range(0, max_distance)) / total_hit_hist.astype(np.float)
#                 distance_max_array = np.ma.masked_invalid(distance_max_array)
#                 distance_min_array = np.ma.masked_invalid(distance_min_array)
#                 plot_utils.plot_track_distances(distance_min_array, distance_max_array, distance_mean_array)
                plot_utils.efficiency_plots(total_hit_hist, total_track_density, total_track_density_with_DUT_hit, efficiency, actual_dut, minimum_track_density, plot_range=dimensions, cut_distance=cut_distance, output_pdf=output_pdf, gui=gui, figs=figs)

                # Calculate mean efficiency without any binning
                eff, eff_err_min, eff_err_pl = analysis_utils.get_mean_efficiency(array_pass=total_track_density_with_DUT_hit,
                                                                                  array_total=total_track_density)

                logging.info('Efficiency =  %1.4f - %1.4f + %1.4f', eff, eff_err_min, eff_err_pl)
                efficiencies.append(np.ma.mean(efficiency))

                dut_group = out_file_h5.create_group(out_file_h5.root, 'DUT_%d' % actual_dut)

                out_efficiency = out_file_h5.create_carray(dut_group, name='Efficiency', title='Efficiency map of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(efficiency.dtype), shape=efficiency.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_efficiency_mask = out_file_h5.create_carray(dut_group, name='Efficiency_mask', title='Masked pixel map of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(efficiency.mask.dtype), shape=efficiency.mask.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                # For correct statistical error calculation the number of detected tracks over total tracks is needed
                out_pass = out_file_h5.create_carray(dut_group, name='Passing_tracks', title='Passing events of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(total_track_density_with_DUT_hit.dtype), shape=total_track_density_with_DUT_hit.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_total = out_file_h5.create_carray(dut_group, name='Total_tracks', title='Total events of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(total_track_density.dtype), shape=total_track_density.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                pass_tracks.append(total_track_density_with_DUT_hit.sum())
                total_tracks.append(total_track_density.sum())
                logging.info('Passing / total tracks: %d / %d', total_track_density_with_DUT_hit.sum(), total_track_density.sum())

                # Store parameters used for efficiency calculation
                out_efficiency.attrs.bin_size = bin_size
                out_efficiency.attrs.minimum_track_density = minimum_track_density
                out_efficiency.attrs.sensor_size = sensor_size
                out_efficiency.attrs.use_duts = use_duts
                out_efficiency.attrs.max_chi2 = max_chi2
                out_efficiency.attrs.cut_distance = cut_distance
                out_efficiency.attrs.max_distance = max_distance
                out_efficiency.attrs.col_range = col_range
                out_efficiency.attrs.row_range = row_range
                out_efficiency[:] = efficiency.T
                out_efficiency_mask[:] = efficiency.mask.T
                out_pass[:] = total_track_density_with_DUT_hit.T
                out_total[:] = total_track_density.T

    if output_pdf is not None:
        output_pdf.close()

    if gui:
        return figs

    return efficiencies, pass_tracks, total_tracks


def calculate_purity(input_tracks_file, input_alignment_file, bin_size, sensor_size, output_purity_file=None, pixel_size=None, n_pixels=None, minimum_hit_density=10, max_distance=500, use_duts=None, max_chi2=None, force_prealignment=False, cut_distance=None, col_range=None, row_range=None, show_inefficient_events=False, output_file=None, plot=True, chunk_size=1000000):
    '''Takes the tracks and calculates the hit purity and hit/track hit distance for selected DUTs.
    Parameters
    ----------
    input_tracks_file : string
        Filename with the tracks table.
    input_alignment_file : pytables file
        Filename of the input aligment data.
    bin_size : iterable
        Bins sizes (i.e. (virtual) pixel size). Give one tuple (x, y) for every plane or list of tuples for different planes.
    sensor_size : Tuple or list of tuples
        Describes the sensor size for each DUT. If one tuple is given it is (size x, size y).
        If several tuples are given it is [(DUT0 size x, DUT0 size y), (DUT1 size x, DUT1 size y), ...].
    output_purity_file : string
        Filename of the output purity file. If None, the filename will be derived from the input hits file.
    minimum_hit_density : int
        Minimum hit density required to consider bin for purity calculation.
    use_duts : iterable
        The DUTs to calculate purity for. If None all duts are used.
    max_chi2 : int
        Only use track with a chi2 <= max_chi2.
    force_prealignment : bool
        Take the prealignment, although if a coarse alignment is availale.
    cut_distance : int
        Hit - track intersection <= cut_distance = pure hit (hit assigned to track).
        Hit - track intersection > cut_distance = inpure hit (hit without a track).
    max_distance : int
        Defines binnig of distance values.
    col_range, row_range : iterable
        Column / row value to calculate purity for (to neglect noisy edge pixels for purity calculation).
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Calculate purity ===')

    if output_purity_file is None:
        output_purity_file = os.path.splitext(input_tracks_file)[0] + '_purity.h5'

    if plot is True:
        output_pdf = PdfPages(os.path.splitext(output_purity_file)[0] + '.pdf', keep_empty=False)
    else:
        output_pdf = None

    use_prealignment = True if force_prealignment else False

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        prealignment = in_file_h5.root.PreAlignment[:]
        n_duts = prealignment.shape[0]
        if not use_prealignment:
            try:
                alignment = in_file_h5.root.Alignment[:]
                logging.info('Use alignment data')
            except tb.exceptions.NodeError:
                use_prealignment = True
                logging.info('Use prealignment data')

    if not isinstance(max_chi2, Iterable):
        max_chi2 = [max_chi2] * n_duts

    purities = []
    pure_hits = []
    total_hits = []
    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_purity_file, 'w') as out_file_h5:
            for index, node in enumerate(in_file_h5.root):
                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.info('Calculate purity for DUT %d', actual_dut)

                # Calculate histogram properties (bins size and number of bins)
                bin_size = [bin_size, ] if not isinstance(bin_size, Iterable) else bin_size
                if len(bin_size) != 1:
                    actual_bin_size_x = bin_size[index][0]
                    actual_bin_size_y = bin_size[index][1]
                else:
                    actual_bin_size_x = bin_size[0][0]
                    actual_bin_size_y = bin_size[0][1]
                dimensions = [sensor_size, ] if not isinstance(sensor_size, Iterable) else sensor_size  # Sensor dimensions for each DUT
                if len(dimensions) == 1:
                    dimensions = dimensions[0]
                else:
                    dimensions = dimensions[index]
                n_bin_x = int(dimensions[0] / actual_bin_size_x)
                n_bin_y = int(dimensions[1] / actual_bin_size_y)

                # Define result histograms, these are filled for each hit chunk
                total_hit_hist = np.zeros(shape=(n_bin_x, n_bin_y), dtype=np.uint32)
                total_pure_hit_hist = np.zeros(shape=(n_bin_x, n_bin_y), dtype=np.uint32)

                actual_max_chi2 = max_chi2[index]

                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    # Take only tracks where actual dut has a hit, otherwise residual wrong
                    selection = np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut]), ~np.isnan(tracks_chunk['track_chi2']))
                    selection_hit = ~np.isnan(tracks_chunk['x_dut_%d' % actual_dut])
                    # Cut in Chi 2 of the track fit
                    if actual_max_chi2:
                        tracks_chunk = tracks_chunk[tracks_chunk['track_chi2'] <= max_chi2]

                    # Transform the hits and track intersections into the local coordinate system
                    # Coordinates in global coordinate system (x, y, z)
                    hit_x_dut, hit_y_dut, hit_z_dut = tracks_chunk['x_dut_%d' % actual_dut][selection_hit], tracks_chunk['y_dut_%d' % actual_dut][selection_hit], tracks_chunk['z_dut_%d' % actual_dut][selection_hit]
                    hit_x, hit_y, hit_z = tracks_chunk['x_dut_%d' % actual_dut][selection], tracks_chunk['y_dut_%d' % actual_dut][selection], tracks_chunk['z_dut_%d' % actual_dut][selection]
                    intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'][selection], tracks_chunk['offset_1'][selection], tracks_chunk['offset_2'][selection]

                    # Transform to local coordinate system
                    if use_prealignment:
                        hit_x_local_dut, hit_y_local_dut, hit_z_local_dut = geometry_utils.apply_alignment(hit_x_dut, hit_y_dut, hit_z_dut,
                                                                                                           dut_index=actual_dut,
                                                                                                           prealignment=prealignment,
                                                                                                           inverse=True)
                        hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
                                                                                               dut_index=actual_dut,
                                                                                               prealignment=prealignment,
                                                                                               inverse=True)
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          prealignment=prealignment,
                                                                                                                          inverse=True)
                    else:  # Apply transformation from alignment information
                        hit_x_local_dut, hit_y_local_dut, hit_z_local_dut = geometry_utils.apply_alignment(hit_x_dut, hit_y_dut, hit_z_dut,
                                                                                                           dut_index=actual_dut,
                                                                                                           alignment=alignment,
                                                                                                           inverse=True)
                        hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
                                                                                               dut_index=actual_dut,
                                                                                               alignment=alignment,
                                                                                               inverse=True)
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          alignment=alignment,
                                                                                                                          inverse=True)
                    # Quickfix that center of sensor is local system is in the center and not at the edge
                    hit_x_local_dut, hit_y_local_dut = hit_x_local_dut + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], hit_y_local_dut + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]
                    hit_x_local, hit_y_local = hit_x_local + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], hit_y_local + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]
                    intersection_x_local, intersection_y_local = intersection_x_local + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], intersection_y_local + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]

                    intersections_local = np.column_stack((intersection_x_local, intersection_y_local, intersection_z_local))
                    hits_local = np.column_stack((hit_x_local, hit_y_local, hit_z_local))
                    hits_local_dut = np.column_stack((hit_x_local_dut, hit_y_local_dut, hit_z_local_dut))

                    if not np.allclose(hits_local[np.isfinite(hits_local[:, 2]), 2], 0.0) or not np.allclose(intersection_z_local, 0.0):
                        raise RuntimeError('The transformation to the local coordinate system did not give all z = 0. Wrong alignment used?')

                    # Usefull for debugging, print some inpure events that can be cross checked
                    # Select virtual hits
                    sel_virtual = np.isnan(tracks_chunk['x_dut_%d' % actual_dut])
                    if show_inefficient_events:
                        logging.info('These events are unpure: %s', str(tracks_chunk['event_number'][sel_virtual]))

                    # Select hits from column, row range (e.g. to supress edge pixels)
                    col_range = [col_range, ] if not isinstance(col_range, Iterable) else col_range
                    row_range = [row_range, ] if not isinstance(row_range, Iterable) else row_range
                    if len(col_range) == 1:
                        index = 0
                    if len(row_range) == 1:
                        index = 0

                    if col_range[index] is not None:
                        selection = np.logical_and(intersections_local[:, 0] >= col_range[index][0], intersections_local[:, 0] <= col_range[index][1])  # Select real hits
                        hits_local, intersections_local = hits_local[selection], intersections_local[selection]
                    if row_range[index] is not None:
                        selection = np.logical_and(intersections_local[:, 1] >= row_range[index][0], intersections_local[:, 1] <= row_range[index][1])  # Select real hits
                        hits_local, intersections_local = hits_local[selection], intersections_local[selection]

                    # Calculate distance between track hit and DUT hit
                    scale = np.square(np.array((1, 1, 0)))  # Regard pixel size for calculating distances
                    distance = np.sqrt(np.dot(np.square(intersections_local - hits_local), scale))  # Array with distances between DUT hit and track hit for each event. Values in um

                    total_hit_hist += (np.histogram2d(hits_local_dut[:, 0], hits_local_dut[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]).astype(np.uint32)

                    # Calculate purity
                    pure_hits_local = hits_local[distance < cut_distance]

                    if not np.any(pure_hits_local):
                        logging.warning('No pure hits in DUT %d, cannot calculate purity', actual_dut)
                        continue
                    total_pure_hit_hist += (np.histogram2d(pure_hits_local[:, 0], pure_hits_local[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]).astype(np.uint32)

                purity = np.zeros_like(total_hit_hist)
                purity[total_hit_hist != 0] = total_pure_hit_hist[total_hit_hist != 0].astype(np.float) / total_hit_hist[total_hit_hist != 0].astype(np.float) * 100.
                purity = np.ma.array(purity, mask=total_hit_hist < minimum_hit_density)

                if not np.any(purity):
                    raise RuntimeError('No pure hit for DUT%d, consider changing cut values or check track building!', actual_dut)

                # Calculate distances between hit and intersection
#                 distance_mean_array = np.average(total_distance_array, axis=2, weights=range(0, max_distance)) * sum(range(0, max_distance)) / total_hit_hist.astype(np.float)
#                 distance_mean_array = np.ma.masked_invalid(distance_mean_array)
#                 distance_max_array = np.amax(total_distance_array, axis=2) * sum(range(0, max_distance)) / total_hit_hist.astype(np.float)
#                 distance_min_array = np.amin(total_distance_array, axis=2) * sum(range(0, max_distance)) / total_hit_hist.astype(np.float)
#                 distance_max_array = np.ma.masked_invalid(distance_max_array)
#                 distance_min_array = np.ma.masked_invalid(distance_min_array)

#                 plot_utils.plot_track_distances(distance_min_array, distance_max_array, distance_mean_array)
                plot_utils.purity_plots(total_pure_hit_hist, total_hit_hist, purity, actual_dut, minimum_hit_density, plot_range=dimensions, cut_distance=cut_distance, output_pdf=output_pdf)

                logging.info('Purity =  %1.4f +- %1.4f', np.ma.mean(purity), np.ma.std(purity))
                purities.append(np.ma.mean(purity))

                dut_group = out_file_h5.create_group(out_file_h5.root, 'DUT_%d' % actual_dut)

                out_purity = out_file_h5.create_carray(dut_group, name='Purity', title='Purity map of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(purity.dtype), shape=purity.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_purity_mask = out_file_h5.create_carray(dut_group, name='Purity_mask', title='Masked pixel map of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(purity.mask.dtype), shape=purity.mask.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                # For correct statistical error calculation the number of pure hits over total hits is needed
                out_pure_hits = out_file_h5.create_carray(dut_group, name='Pure_hits', title='Passing events of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(total_pure_hit_hist.dtype), shape=total_pure_hit_hist.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_total_total = out_file_h5.create_carray(dut_group, name='Total_hits', title='Total events of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(total_hit_hist.dtype), shape=total_hit_hist.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                pure_hits.append(total_pure_hit_hist.sum())
                total_hits.append(total_hit_hist.sum())
                logging.info('Pure hits / total hits: %d / %d, Purity = %.2f', total_pure_hit_hist.sum(), total_hit_hist.sum(), total_pure_hit_hist.sum() / total_hit_hist.sum() * 100)

                # Store parameters used for purity calculation
                out_purity.attrs.bin_size = bin_size
                out_purity.attrs.minimum_hit_density = minimum_hit_density
                out_purity.attrs.sensor_size = sensor_size
                out_purity.attrs.use_duts = use_duts
                out_purity.attrs.max_chi2 = max_chi2
                out_purity.attrs.cut_distance = cut_distance
                out_purity.attrs.max_distance = max_distance
                out_purity.attrs.col_range = col_range
                out_purity.attrs.row_range = row_range
                out_purity[:] = purity.T
                out_purity_mask[:] = purity.mask.T
                out_pure_hits[:] = total_pure_hit_hist.T
                out_total_total[:] = total_hit_hist.T

    if output_pdf is not None:
        output_pdf.close()

    return purities, pure_hits, total_hits


def histogram_track_angle(input_tracks_file, input_alignment_file=None, output_track_angle_file=None, n_bins="auto", plot_range=(None, None), use_duts=None, dut_names=None, plot=True, chunk_size=499999):
    '''Calculates and histograms the track angle of the fitted tracks for selected DUTs.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    input_alignment_file : string
        Filename of the input alignment file.
        If None, the DUT planes are assumed to be perpendicular to the z axis.
    output_track_angle_file: string
        Filename of the output track angle file with track angle histogram and fitted means and sigmas of track angles for selected DUTs.
        If None, deduce filename from input tracks file.
    n_bins : uint
        Number of bins for the histogram.
        If "auto", automatic binning is used.
    plot_range : iterable of tuples
        Tuple of the plot range in rad for alpha and beta angular distribution, e.g. ((-0.01, +0.01), -0.01, +0.01)).
        If (None, None), plotting from minimum to maximum.
    use_duts : iterable
        Calculate the track angle for given DUTs. If None, all duts are used.
    dut_names : iterable
        Name of the DUTs. If None, DUT numbers will be used.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Calculating track angles ===')

    if input_alignment_file:
        with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
            logging.info('Use alignment data')
            alignment = in_file_h5.root.Alignment[:]
    else:
        alignment = None

    if output_track_angle_file is None:
        output_track_angle_file = os.path.splitext(input_tracks_file)[0] + '_track_angles.h5'

    with tb.open_file(input_tracks_file, 'r') as in_file_h5:
        with tb.open_file(output_track_angle_file, mode="w") as out_file_h5:
            nodes = in_file_h5.list_nodes("/")
            if not nodes:
                return
            extended_nodes = nodes[:1]
            extended_nodes.extend(nodes)
            for index, node in enumerate(extended_nodes):  # loop through all DUTs in track table
                initialize = True
                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                if index == 0:
                    dut_name = None
                else:
                    dut_name = "DUT%d" % actual_dut
                if use_duts is not None and actual_dut not in use_duts:
                    continue

                if alignment is not None and index != 0:
                    rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[actual_dut]['alpha'],
                                                                     beta=alignment[actual_dut]['beta'],
                                                                     gamma=alignment[actual_dut]['gamma'])
                    basis_global = rotation_matrix.T.dot(np.eye(3))
                    dut_plane_normal = basis_global[2]
                    if dut_plane_normal[2] < 0:
                        dut_plane_normal = -dut_plane_normal
                else:
                    dut_plane_normal = np.array([0.0, 0.0, 1.0])
                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):  # only store track slopes of selected DUTs
                    track_slopes = np.column_stack((tracks_chunk['slope_0'],
                                                    tracks_chunk['slope_1'],
                                                    tracks_chunk['slope_2']))

                    # TODO: alpha/beta wrt DUT col / row
                    total_angles = np.arccos(np.inner(dut_plane_normal, track_slopes))
                    alpha_angles = 0.5 * np.pi - np.arccos(np.inner(track_slopes, np.cross(dut_plane_normal, np.array([1.0, 0.0, 0.0]))))
                    beta_angles = 0.5 * np.pi - np.arccos(np.inner(track_slopes, np.cross(dut_plane_normal, np.array([0.0, 1.0, 0.0]))))

                    if initialize:
                        total_angle_hist, total_angle_hist_edges = np.histogram(total_angles, bins=n_bins, range=None)
                        alpha_angle_hist, alpha_angle_hist_edges = np.histogram(alpha_angles, bins=n_bins, range=plot_range[1])
                        beta_angle_hist, beta_angle_hist_edges = np.histogram(beta_angles, bins=n_bins, range=plot_range[0])
                        initialize = False
                    else:
                        total_angle_hist += np.histogram(total_angles, bins=total_angle_hist_edges)[0]
                        alpha_angle_hist += np.histogram(alpha_angles, bins=alpha_angle_hist_edges)[0]
                        beta_angle_hist += np.histogram(beta_angles, bins=beta_angle_hist_edges)[0]

                # write results
                track_angle_total = out_file_h5.create_carray(where=out_file_h5.root,
                                                              name='Total_Track_Angle_Hist%s' % (("_%s" % dut_name) if dut_name else ""),
                                                              title='Total track angle distribution%s' % (("_for_%s" % dut_name) if dut_name else ""),
                                                              atom=tb.Atom.from_dtype(total_angle_hist.dtype),
                                                              shape=total_angle_hist.shape,
                                                              filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                track_angle_beta = out_file_h5.create_carray(where=out_file_h5.root,
                                                             name='Beta_Track_Angle_Hist%s' % (("_%s" % dut_name) if dut_name else ""),
                                                             title='Beta track angle distribution%s' % (("_for_%s" % dut_name) if dut_name else ""),
                                                             atom=tb.Atom.from_dtype(beta_angle_hist.dtype),
                                                             shape=beta_angle_hist.shape,
                                                             filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                track_angle_alpha = out_file_h5.create_carray(where=out_file_h5.root,
                                                              name='Alpha_Track_Angle_Hist%s' % (("_%s" % dut_name) if dut_name else ""),
                                                              title='Alpha track angle distribution%s' % (("_for_%s" % dut_name) if dut_name else ""),
                                                              atom=tb.Atom.from_dtype(alpha_angle_hist.dtype),
                                                              shape=alpha_angle_hist.shape,
                                                              filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                # fit histograms for x and y direction
                bin_center = (total_angle_hist_edges[1:] + total_angle_hist_edges[:-1]) / 2.0
                mean = analysis_utils.get_mean_from_histogram(total_angle_hist, bin_center)
                rms = analysis_utils.get_rms_from_histogram(total_angle_hist, bin_center)
                fit_total, cov = curve_fit(analysis_utils.gauss, bin_center, total_angle_hist, p0=[np.amax(total_angle_hist), mean, rms])

                bin_center = (beta_angle_hist_edges[1:] + beta_angle_hist_edges[:-1]) / 2.0
                mean = analysis_utils.get_mean_from_histogram(beta_angle_hist, bin_center)
                rms = analysis_utils.get_rms_from_histogram(beta_angle_hist, bin_center)
                fit_beta, cov = curve_fit(analysis_utils.gauss, bin_center, beta_angle_hist, p0=[np.amax(beta_angle_hist), mean, rms])

                bin_center = (alpha_angle_hist_edges[1:] + alpha_angle_hist_edges[:-1]) / 2.0
                mean = analysis_utils.get_mean_from_histogram(alpha_angle_hist, bin_center)
                rms = analysis_utils.get_rms_from_histogram(alpha_angle_hist, bin_center)
                fit_alpha, cov = curve_fit(analysis_utils.gauss, bin_center, alpha_angle_hist, p0=[np.amax(alpha_angle_hist), mean, rms])

                # total
                track_angle_total.attrs.edges = total_angle_hist_edges
                track_angle_total.attrs.edges = total_angle_hist_edges
                track_angle_total.attrs.amp = fit_total[0]
                track_angle_total.attrs.mean = fit_total[1]
                track_angle_total.attrs.sigma = fit_total[2]
                track_angle_total[:] = total_angle_hist
                # x
                track_angle_beta.attrs.edges = beta_angle_hist_edges
                track_angle_beta.attrs.amp = fit_beta[0]
                track_angle_beta.attrs.mean = fit_beta[1]
                track_angle_beta.attrs.sigma = fit_beta[2]
                track_angle_beta[:] = beta_angle_hist
                # y
                track_angle_alpha.attrs.edges = alpha_angle_hist_edges
                track_angle_alpha.attrs.amp = fit_alpha[0]
                track_angle_alpha.attrs.mean = fit_alpha[1]
                track_angle_alpha.attrs.sigma = fit_alpha[2]
                track_angle_alpha[:] = alpha_angle_hist

    if plot:
        plot_utils.plot_track_angle(input_track_angle_file=output_track_angle_file, output_pdf_file=None, dut_names=dut_names)
