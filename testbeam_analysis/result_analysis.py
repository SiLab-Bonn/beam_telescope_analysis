''' All functions creating results (e.g. efficiency, residuals, track density) from fitted tracks are listed here.'''
from __future__ import division

import logging
import re
from collections import Iterable
import os.path

import progressbar
import tables as tb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from scipy import stats
from scipy.signal import hilbert

from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.tools import analysis_utils
import testbeam_analysis.dut_alignment


def calculate_residuals(input_tracks_file, input_alignment_file, use_prealignment, select_duts, n_pixels, pixel_size, output_residuals_file=None, dut_names=None, nbins_per_pixel=None, npixels_per_bin=None, use_fit_limits=False, plot=True, gui=False, chunk_size=1000000):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    input_alignment_file : string
        Filename of the input aligment file.
    use_prealignment : bool
        If True, use pre-alignment from correlation data; if False, use alignment.
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
    select_duts : iterable
        Selecting DUTs that will be processed.
    nbins_per_pixel : int
        Number of bins per pixel along the residual axis. Number is a positive integer or None to automatically set the binning.
    npixels_per_bin : int
        Number of pixels per bin along the position axis. Number is a positive integer or None to automatically set the binning.
    use_fit_limits : bool
        If True, use fit limits from pre-alignment for selecting fit range for the alignment.
    plot : bool
        If True, create additional output plots.
    gui : bool
        If True, use GUI for plotting.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Calculating residuals ===')

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

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_residuals_file, mode='w') as out_file_h5:
            for actual_dut in select_duts:
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT_%d' % actual_dut)
                logging.info('Calculating residuals for DUT%d', actual_dut)

                if use_fit_limits:
                    fit_limit_x_local, fit_limit_y_local = fit_limits[actual_dut][0], fit_limits[actual_dut][1]
                else:
                    fit_limit_x_local = None
                    fit_limit_y_local = None

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    # Select data with hits and tracks
                    selection = np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut]), ~np.isnan(tracks_chunk['track_chi2']))
                    tracks_chunk = tracks_chunk[selection]  # Take only tracks where actual dut has a hit, otherwise residual wrong

                    # Coordinates in global coordinate system (x, y, z)
                    hit_x_local, hit_y_local, hit_z_local = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
                    hit_local = np.column_stack([hit_x_local, hit_y_local])
                    intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']
                    slopes = np.column_stack([tracks_chunk['slope_0'], tracks_chunk['slope_1'], tracks_chunk['slope_2']])

                    if use_prealignment:
                        hit_x, hit_y, hit_z = geometry_utils.apply_alignment(hit_x_local, hit_y_local, hit_z_local,
                                                                             dut_index=actual_dut,
                                                                             prealignment=prealignment,
                                                                             inverse=False)

#                         dut_position = np.array([0., 0., prealignment['z'][actual_dut]])
#                         dut_plane_normal = np.array([0., 0., 1.])
#
#                         # Set the offset to the track intersection with the tilted plane
#                         dut_intersection = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
#                                                                                             line_directions=slopes,
#                                                                                             position_plane=dut_position,
#                                                                                             normal_plane=dut_plane_normal)
#                         intersection_x, intersection_y, intersection_z = dut_intersection[:, 0], dut_intersection[:, 1], dut_intersection[:, 2]

                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          prealignment=prealignment,
                                                                                                                          inverse=True)
                    else:
                        hit_x, hit_y, hit_z = geometry_utils.apply_alignment(hit_x_local, hit_y_local, hit_z_local,
                                                                             dut_index=actual_dut,
                                                                             alignment=alignment,
                                                                             inverse=False)

#                         dut_position = np.array([alignment[actual_dut]['translation_x'], alignment[actual_dut]['translation_y'], alignment[actual_dut]['translation_z']])
#                         rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[actual_dut]['alpha'],
#                                                                          beta=alignment[actual_dut]['beta'],
#                                                                          gamma=alignment[actual_dut]['gamma'])
#                         basis_global = rotation_matrix.T.dot(np.eye(3))
#                         dut_plane_normal = basis_global[2]
#                         if dut_plane_normal[2] < 0:
#                             dut_plane_normal = -dut_plane_normal
#
#                         # Set the offset to the track intersection with the tilted plane
#                         dut_intersection = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
#                                                                                             line_directions=slopes,
#                                                                                             position_plane=dut_position,
#                                                                                             normal_plane=dut_plane_normal)
#                         intersection_x, intersection_y, intersection_z = dut_intersection[:, 0], dut_intersection[:, 1], dut_intersection[:, 2]

                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          alignment=alignment,
                                                                                                                          inverse=True)

                    intersection_local = np.column_stack([intersection_x_local, intersection_y_local])

                    if not np.allclose(hit_z_local[np.isfinite(hit_z_local)], 0.0) or not np.allclose(intersection_z_local[np.isfinite(intersection_z_local)], 0.0):
                        raise RuntimeError("Transformation into local coordinate system gives z != 0")

                    difference = np.column_stack((hit_x, hit_y)) - np.column_stack((intersection_x, intersection_y))
                    difference_local = np.column_stack((hit_x_local, hit_y_local)) - np.column_stack((intersection_x_local, intersection_y_local))

                    limit_x_local_sel = np.ones_like(hit_x_local, dtype=np.bool)
                    if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local[0]):
                        limit_x_local_sel &= hit_x_local >= fit_limit_x_local[0]
                    if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local[1]):
                        limit_x_local_sel &= hit_x_local <= fit_limit_x_local[1]

                    limit_y_local_sel = np.ones_like(hit_x_local, dtype=np.bool)
                    if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local[0]):
                        limit_y_local_sel &= hit_y_local >= fit_limit_y_local[0]
                    if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local[1]):
                        limit_y_local_sel &= hit_y_local <= fit_limit_y_local[1]

                    limit_xy_local_sel = np.logical_and(limit_x_local_sel, limit_y_local_sel)

                    hit_x_local_limit_x = hit_x_local[limit_x_local_sel]
                    hit_y_local_limit_x = hit_y_local[limit_x_local_sel]
                    intersection_x_local_limit_x = intersection_x_local[limit_x_local_sel]
                    intersection_y_local_limit_x = intersection_y_local[limit_x_local_sel]

                    hit_x_local_limit_y = hit_x_local[limit_y_local_sel]
                    hit_y_local_limit_y = hit_y_local[limit_y_local_sel]
                    intersection_x_local_limit_y = intersection_x_local[limit_y_local_sel]
                    intersection_y_local_limit_y = intersection_y_local[limit_y_local_sel]

                    hit_x_local_limit_xy = hit_x_local[limit_xy_local_sel]
                    hit_y_local_limit_xy = hit_y_local[limit_xy_local_sel]
                    intersection_x_local_limit_xy = intersection_x_local[limit_xy_local_sel]
                    intersection_y_local_limit_xy = intersection_y_local[limit_xy_local_sel]

                    difference_local_limit_x = np.column_stack((hit_x_local_limit_x, hit_y_local_limit_x)) - np.column_stack((intersection_x_local_limit_x, intersection_y_local_limit_x))
                    difference_local_limit_y = np.column_stack((hit_x_local_limit_y, hit_y_local_limit_y)) - np.column_stack((intersection_x_local_limit_y, intersection_y_local_limit_y))
                    difference_local_limit_xy = np.column_stack((hit_x_local_limit_xy, hit_y_local_limit_xy)) - np.column_stack((intersection_x_local_limit_xy, intersection_y_local_limit_xy))
                    distance = np.sqrt(np.einsum('ij,ij->i', intersection_local - hit_local, intersection_local - hit_local))

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
                            min_difference, max_difference = np.min(difference_local_limit_xy[:, 0]), np.max(difference_local_limit_xy[:, 0])
                            nbins = np.arange(min_difference - (pixel_size[actual_dut][0] / nbins_per_pixel), max_difference + 2 * (pixel_size[actual_dut][0] / nbins_per_pixel), pixel_size[actual_dut][0] / nbins_per_pixel)
                        else:
                            nbins = "auto"
                        hist, edges = np.histogram(difference_local_limit_xy[:, 0], bins=nbins)
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
                            min_difference, max_difference = np.min(difference_local_limit_xy[:, 1]), np.max(difference_local_limit_xy[:, 1])
                            nbins = np.arange(min_difference - (pixel_size[actual_dut][1] / nbins_per_pixel), max_difference + 2 * (pixel_size[actual_dut][1] / nbins_per_pixel), pixel_size[actual_dut][1] / nbins_per_pixel)
                        else:
                            nbins = "auto"
                        hist, edges = np.histogram(difference_local_limit_xy[:, 1], bins=nbins)
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
                            if fwhm_x < 0.01:
                                nbins = 1000
                            else:
                                nbins = "auto"
                            width = pixel_size[actual_dut][0] * np.ceil(plot_n_pixels * fwhm_x / pixel_size[actual_dut][0])
                            x_range = (center_x - width, center_x + width)
                        x_res_hist, x_res_hist_edges = np.histogram(difference[:, 0], range=x_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_x), np.max(intersection_x)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][0], npixels_per_bin * pixel_size[actual_dut][0])
                        else:
                            nbins = "auto"
                        _, x_pos_hist_edges = np.histogram(intersection_x, bins=nbins)

                        if nbins_per_pixel is not None:
                            width = max(plot_n_pixels * pixel_size[actual_dut][1], pixel_size[actual_dut][1] * np.ceil(plot_n_pixels * fwhm_y / pixel_size[actual_dut][1]))
                            if np.mod(width / pixel_size[actual_dut][1], 2) != 0:
                                width += pixel_size[actual_dut][1]
                            nbins = int(nbins_per_pixel * width / pixel_size[actual_dut][1])
                            y_range = (center_y - 0.5 * width, center_y + 0.5 * width)
                        else:
                            if fwhm_y < 0.01:
                                nbins = 1000
                            else:
                                nbins = "auto"
                            width = pixel_size[actual_dut][1] * np.ceil(plot_n_pixels * fwhm_y / pixel_size[actual_dut][1])
                            y_range = (center_y - width, center_y + width)
                        y_res_hist, y_res_hist_edges = np.histogram(difference[:, 1], range=y_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_y), np.max(intersection_y)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][1], npixels_per_bin * pixel_size[actual_dut][1])
                        else:
                            nbins = "auto"
                        _, y_pos_hist_edges = np.histogram(intersection_y, bins=nbins)

                        if nbins_per_pixel is not None:
                            width = max(plot_n_pixels * pixel_size[actual_dut][0], pixel_size[actual_dut][0] * np.ceil(plot_n_pixels * fwhm_col / pixel_size[actual_dut][0]))
                            if np.mod(width / pixel_size[actual_dut][0], 2) != 0:
                                width += pixel_size[actual_dut][0]
                            nbins = int(nbins_per_pixel * width / pixel_size[actual_dut][0])
                            col_range = (center_col - 0.5 * width, center_col + 0.5 * width)
                        else:
                            if fwhm_col < 0.01:
                                nbins = 1000
                            else:
                                nbins = "auto"
                            width = pixel_size[actual_dut][0] * np.ceil(plot_n_pixels * fwhm_col / pixel_size[actual_dut][0])
                            col_range = (center_col - width, center_col + width)
                        col_res_hist, col_res_hist_edges = np.histogram(difference_local_limit_xy[:, 0], range=col_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_x_local), np.max(intersection_x_local)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][0], npixels_per_bin * pixel_size[actual_dut][0])
                        else:
                            nbins = "auto"
                        _, col_pos_hist_edges = np.histogram(intersection_x_local, bins=nbins)

                        if nbins_per_pixel is not None:
                            width = max(plot_n_pixels * pixel_size[actual_dut][1], pixel_size[actual_dut][1] * np.ceil(plot_n_pixels * fwhm_row / pixel_size[actual_dut][1]))
                            if np.mod(width / pixel_size[actual_dut][1], 2) != 0:
                                width += pixel_size[actual_dut][1]
                            nbins = int(nbins_per_pixel * width / pixel_size[actual_dut][1])
                            row_range = (center_row - 0.5 * width, center_row + 0.5 * width)
                        else:
                            if fwhm_row < 0.01:
                                nbins = 1000
                            else:
                                nbins = "auto"
                            width = pixel_size[actual_dut][1] * np.ceil(plot_n_pixels * fwhm_row / pixel_size[actual_dut][1])
                            row_range = (center_row - width, center_row + width)
                        row_res_hist, row_res_hist_edges = np.histogram(difference_local_limit_xy[:, 1], range=row_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_y_local), np.max(intersection_y_local)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * pixel_size[actual_dut][1], npixels_per_bin * pixel_size[actual_dut][1])
                        else:
                            nbins = "auto"
                        _, row_pos_hist_edges = np.histogram(intersection_y_local, bins=nbins)

                        dut_x_size = n_pixels[actual_dut][0] * pixel_size[actual_dut][0]
                        dut_y_size = n_pixels[actual_dut][1] * pixel_size[actual_dut][1]
                        hist_2d_res_x_edges = np.linspace(-dut_x_size / 2.0, dut_x_size / 2.0, n_pixels[actual_dut][0] + 1, endpoint=True)
                        hist_2d_res_y_edges = np.linspace(-dut_y_size / 2.0, dut_y_size / 2.0, n_pixels[actual_dut][1] + 1, endpoint=True)
                        hist_2d_edges = [hist_2d_res_x_edges, hist_2d_res_y_edges]

                        # global x residual against x position
                        x_res_x_pos_hist, _, _ = np.histogram2d(
                            intersection_x,
                            difference[:, 0],
                            bins=(x_pos_hist_edges, x_res_hist_edges))
                        stat_x_res_x_pos_hist, _, _ = stats.binned_statistic(x=intersection_x, values=difference[:, 0], statistic='mean', bins=x_pos_hist_edges)
                        stat_x_res_x_pos_hist = np.nan_to_num(stat_x_res_x_pos_hist)
                        count_x_res_x_pos_hist, _, _ = stats.binned_statistic(x=intersection_x, values=difference[:, 0], statistic='count', bins=x_pos_hist_edges)

                        # global y residual against y position
                        y_res_y_pos_hist, _, _ = np.histogram2d(
                            intersection_y,
                            difference[:, 1],
                            bins=(y_pos_hist_edges, y_res_hist_edges))
                        stat_y_res_y_pos_hist, _, _ = stats.binned_statistic(x=intersection_y, values=difference[:, 1], statistic='mean', bins=y_pos_hist_edges)
                        stat_y_res_y_pos_hist = np.nan_to_num(stat_y_res_y_pos_hist)
                        count_y_res_y_pos_hist, _, _ = stats.binned_statistic(x=intersection_y, values=difference[:, 1], statistic='count', bins=y_pos_hist_edges)

                        # global y residual against x position
                        y_res_x_pos_hist, _, _ = np.histogram2d(
                            intersection_x,
                            difference[:, 1],
                            bins=(x_pos_hist_edges, y_res_hist_edges))
                        stat_y_res_x_pos_hist, _, _ = stats.binned_statistic(x=intersection_x, values=difference[:, 1], statistic='mean', bins=x_pos_hist_edges)
                        stat_y_res_x_pos_hist = np.nan_to_num(stat_y_res_x_pos_hist)
                        count_y_res_x_pos_hist, _, _ = stats.binned_statistic(x=intersection_x, values=difference[:, 1], statistic='count', bins=x_pos_hist_edges)

                        # global x residual against y position
                        x_res_y_pos_hist, _, _ = np.histogram2d(
                            intersection_y,
                            difference[:, 0],
                            bins=(y_pos_hist_edges, x_res_hist_edges))
                        stat_x_res_y_pos_hist, _, _ = stats.binned_statistic(x=intersection_y, values=difference[:, 0], statistic='mean', bins=y_pos_hist_edges)
                        stat_x_res_y_pos_hist = np.nan_to_num(stat_x_res_y_pos_hist)
                        count_x_res_y_pos_hist, _, _ = stats.binned_statistic(x=intersection_y, values=difference[:, 0], statistic='count', bins=y_pos_hist_edges)

                        # local column residual against column position
                        col_res_col_pos_hist, _, _ = np.histogram2d(
                            intersection_x_local_limit_y,
                            difference_local_limit_y[:, 0],
                            bins=(col_pos_hist_edges, col_res_hist_edges))
                        stat_col_res_col_pos_hist, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 0], statistic='mean', bins=col_pos_hist_edges)
                        stat_col_res_col_pos_hist = np.nan_to_num(stat_col_res_col_pos_hist)
                        count_col_res_col_pos_hist, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 0], statistic='count', bins=col_pos_hist_edges)

                        # local row residual against row position
                        row_res_row_pos_hist, _, _ = np.histogram2d(
                            intersection_y_local_limit_x,
                            difference_local_limit_x[:, 1],
                            bins=(row_pos_hist_edges, row_res_hist_edges))
                        stat_row_res_row_pos_hist, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 1], statistic='mean', bins=row_pos_hist_edges)
                        stat_row_res_row_pos_hist = np.nan_to_num(stat_row_res_row_pos_hist)
                        count_row_res_row_pos_hist, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 1], statistic='count', bins=row_pos_hist_edges)

                        # local row residual against column position
                        row_res_col_pos_hist, _, _ = np.histogram2d(
                            intersection_x_local_limit_y,
                            difference_local_limit_y[:, 1],
                            bins=(col_pos_hist_edges, row_res_hist_edges))
                        stat_row_res_col_pos_hist, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 1], statistic='mean', bins=col_pos_hist_edges)
                        stat_row_res_col_pos_hist = np.nan_to_num(stat_row_res_col_pos_hist)
                        count_row_res_col_pos_hist, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 1], statistic='count', bins=col_pos_hist_edges)

                        # local column residual against row position
                        col_res_row_pos_hist, _, _ = np.histogram2d(
                            intersection_y_local_limit_x,
                            difference_local_limit_x[:, 0],
                            bins=(row_pos_hist_edges, col_res_hist_edges))
                        stat_col_res_row_pos_hist, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 0], statistic='mean', bins=row_pos_hist_edges)
                        stat_col_res_row_pos_hist = np.nan_to_num(stat_col_res_row_pos_hist)
                        count_col_res_row_pos_hist, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 0], statistic='count', bins=row_pos_hist_edges)

                        # 2D residuals
                        stat_2d_res_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=distance, statistic='mean', bins=hist_2d_edges)
                        stat_2d_res_hist = np.nan_to_num(stat_2d_res_hist)
                        count_2d_res_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=distance, statistic='count', bins=hist_2d_edges)

                        # 2D hits
                        count_2d_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=None, statistic='count', bins=hist_2d_edges)


                    else:  # adding data to existing histograms
                        x_res_hist += np.histogram(difference[:, 0], bins=x_res_hist_edges)[0]
                        y_res_hist += np.histogram(difference[:, 1], bins=y_res_hist_edges)[0]
                        col_res_hist += np.histogram(difference_local_limit_xy[:, 0], bins=col_res_hist_edges)[0]
                        row_res_hist += np.histogram(difference_local_limit_xy[:, 1], bins=row_res_hist_edges)[0]

                        # global x residual against x position
                        x_res_x_pos_hist += np.histogram2d(
                            intersection_x,
                            difference[:, 0],
                            bins=(x_pos_hist_edges, x_res_hist_edges))[0]
                        stat_x_res_x_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_x, values=difference[:, 0], statistic='mean', bins=x_pos_hist_edges)
                        stat_x_res_x_pos_hist_tmp = np.nan_to_num(stat_x_res_x_pos_hist_tmp)
                        count_x_res_x_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_x, values=difference[:, 0], statistic='count', bins=x_pos_hist_edges)
                        stat_x_res_x_pos_hist, count_x_res_x_pos_hist = np.ma.average(a=np.stack([stat_x_res_x_pos_hist, stat_x_res_x_pos_hist_tmp]), axis=0, weights=np.stack([count_x_res_x_pos_hist, count_x_res_x_pos_hist_tmp]), returned=True)

                        # global y residual against y position
                        y_res_y_pos_hist += np.histogram2d(
                            intersection_y,
                            difference[:, 1],
                            bins=(y_pos_hist_edges, y_res_hist_edges))[0]
                        stat_y_res_y_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_y, values=difference[:, 1], statistic='mean', bins=y_pos_hist_edges)
                        stat_y_res_y_pos_hist_tmp = np.nan_to_num(stat_y_res_y_pos_hist_tmp)
                        count_y_res_y_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_y, values=difference[:, 1], statistic='count', bins=y_pos_hist_edges)
                        stat_y_res_y_pos_hist, count_y_res_y_pos_hist = np.ma.average(a=np.stack([stat_y_res_y_pos_hist, stat_y_res_y_pos_hist_tmp]), axis=0, weights=np.stack([count_y_res_y_pos_hist, count_y_res_y_pos_hist_tmp]), returned=True)


                        # global y residual against x position
                        y_res_x_pos_hist += np.histogram2d(
                            intersection_x,
                            difference[:, 1],
                            bins=(x_pos_hist_edges, y_res_hist_edges))[0]
                        stat_y_res_x_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_x, values=difference[:, 1], statistic='mean', bins=x_pos_hist_edges)
                        stat_y_res_x_pos_hist_tmp = np.nan_to_num(stat_y_res_x_pos_hist_tmp)
                        count_y_res_x_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_x, values=difference[:, 1], statistic='count', bins=x_pos_hist_edges)
                        stat_y_res_x_pos_hist, count_y_res_x_pos_hist = np.ma.average(a=np.stack([stat_y_res_x_pos_hist, stat_y_res_x_pos_hist_tmp]), axis=0, weights=np.stack([count_y_res_x_pos_hist, count_y_res_x_pos_hist_tmp]), returned=True)

                        # global x residual against y position
                        x_res_y_pos_hist += np.histogram2d(
                            intersection_y,
                            difference[:, 0],
                            bins=(y_pos_hist_edges, x_res_hist_edges))[0]
                        stat_x_res_y_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_y, values=difference[:, 0], statistic='mean', bins=y_pos_hist_edges)
                        stat_x_res_y_pos_hist_tmp = np.nan_to_num(stat_x_res_y_pos_hist_tmp)
                        count_x_res_y_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_y, values=difference[:, 0], statistic='count', bins=y_pos_hist_edges)
                        stat_x_res_y_pos_hist, count_x_res_y_pos_hist = np.ma.average(a=np.stack([stat_x_res_y_pos_hist, stat_x_res_y_pos_hist_tmp]), axis=0, weights=np.stack([count_x_res_y_pos_hist, count_x_res_y_pos_hist_tmp]), returned=True)

                        # local column residual against column position
                        col_res_col_pos_hist += np.histogram2d(
                            intersection_x_local_limit_y,
                            difference_local_limit_y[:, 0],
                            bins=(col_pos_hist_edges, col_res_hist_edges))[0]
                        stat_col_res_col_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 0], statistic='mean', bins=col_pos_hist_edges)
                        stat_col_res_col_pos_hist_tmp = np.nan_to_num(stat_col_res_col_pos_hist_tmp)
                        count_col_res_col_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 0], statistic='count', bins=col_pos_hist_edges)
                        stat_col_res_col_pos_hist, count_col_res_col_pos_hist = np.ma.average(a=np.stack([stat_col_res_col_pos_hist, stat_col_res_col_pos_hist_tmp]), axis=0, weights=np.stack([count_col_res_col_pos_hist, count_col_res_col_pos_hist_tmp]), returned=True)

                        # local row residual against row position
                        row_res_row_pos_hist += np.histogram2d(
                            intersection_y_local_limit_x,
                            difference_local_limit_x[:, 1],
                            bins=(row_pos_hist_edges, row_res_hist_edges))[0]
                        stat_row_res_row_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 1], statistic='mean', bins=row_pos_hist_edges)
                        stat_row_res_row_pos_hist_tmp = np.nan_to_num(stat_row_res_row_pos_hist_tmp)
                        count_row_res_row_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 1], statistic='count', bins=row_pos_hist_edges)
                        stat_row_res_row_pos_hist, count_row_res_row_pos_hist = np.ma.average(a=np.stack([stat_row_res_row_pos_hist, stat_row_res_row_pos_hist_tmp]), axis=0, weights=np.stack([count_row_res_row_pos_hist, count_row_res_row_pos_hist_tmp]), returned=True)

                        # local row residual against column position
                        row_res_col_pos_hist += np.histogram2d(
                            intersection_x_local_limit_y,
                            difference_local_limit_y[:, 1],
                            bins=(col_pos_hist_edges, row_res_hist_edges))[0]
                        stat_row_res_col_pos_tmp, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 1], statistic='mean', bins=col_pos_hist_edges)
                        stat_row_res_col_pos_tmp = np.nan_to_num(stat_row_res_col_pos_tmp)
                        count_row_res_col_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 1], statistic='count', bins=col_pos_hist_edges)
                        stat_row_res_col_pos_hist, count_row_res_col_pos_hist = np.ma.average(a=np.stack([stat_row_res_col_pos_hist, stat_row_res_col_pos_tmp]), axis=0, weights=np.stack([count_row_res_col_pos_hist, count_row_res_col_pos_hist_tmp]), returned=True)

                        # local column residual against row position
                        col_res_row_pos_hist += np.histogram2d(
                            intersection_y_local_limit_x,
                            difference_local_limit_x[:, 0],
                            bins=(row_pos_hist_edges, col_res_hist_edges))[0]
                        stat_col_res_row_pos_tmp, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 0], statistic='mean', bins=row_pos_hist_edges)
                        stat_col_res_row_pos_tmp = np.nan_to_num(stat_col_res_row_pos_tmp)
                        count_col_res_row_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 0], statistic='count', bins=row_pos_hist_edges)
                        stat_col_res_row_pos_hist, count_col_res_row_pos_hist = np.ma.average(a=np.stack([stat_col_res_row_pos_hist, stat_col_res_row_pos_tmp]), axis=0, weights=np.stack([count_col_res_row_pos_hist, count_col_res_row_pos_hist_tmp]), returned=True)

                        # 2D residuals
                        stat_2d_res_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=distance, statistic='mean', bins=hist_2d_edges)
                        stat_2d_res_hist_tmp = np.nan_to_num(stat_2d_res_hist_tmp)
                        count_2d_res_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=distance, statistic='count', bins=hist_2d_edges)
                        stat_2d_res_hist, count_2d_res_hist = np.ma.average(a=np.stack([stat_2d_res_hist, stat_2d_res_hist_tmp]), axis=0, weights=np.stack([count_2d_res_hist, count_2d_res_hist_tmp]), returned=True)

                        # 2D hits
                        count_2d_hist += stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=None, statistic='count', bins=hist_2d_edges)[0]

                logging.debug('Storing residual histograms...')

                dut_name = dut_names[actual_dut] if dut_names else ("DUT" + str(actual_dut))

                stat_x_res_x_pos_hist[count_x_res_x_pos_hist == 0] = np.nan
                stat_y_res_y_pos_hist[count_y_res_y_pos_hist == 0] = np.nan
                stat_y_res_x_pos_hist[count_y_res_x_pos_hist == 0] = np.nan
                stat_x_res_y_pos_hist[count_x_res_y_pos_hist == 0] = np.nan
                stat_col_res_col_pos_hist[count_col_res_col_pos_hist == 0] = np.nan
                stat_row_res_row_pos_hist[count_row_res_row_pos_hist == 0] = np.nan
                stat_row_res_col_pos_hist[count_row_res_col_pos_hist == 0] = np.nan
                stat_col_res_row_pos_hist[count_col_res_row_pos_hist == 0] = np.nan

                # Local residuals
                fit_col_res, cov_col_res = analysis_utils.fit_residuals(
                    hist=col_res_hist,
                    edges=col_res_hist_edges,
                )
                plot_utils.plot_residuals(
                    histogram=col_res_hist,
                    edges=col_res_hist_edges,
                    fit=fit_col_res,
                    cov=cov_col_res,
                    xlabel='Column residual [$\mathrm{\mu}$m]',
                    title='Column residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_col_res = out_file_h5.create_carray(out_file_h5.root,
                                                        name='col_res_DUT%d' % (actual_dut),
                                                        title='Residual distribution in column direction for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(col_res_hist.dtype),
                                                        shape=col_res_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col_res.attrs.xedges = col_res_hist_edges
                out_col_res.attrs.fit_coeff = fit_col_res
                out_col_res.attrs.fit_cov = cov_col_res
                out_col_res[:] = col_res_hist

                fit_row_res, cov_row_res = analysis_utils.fit_residuals(
                    hist=row_res_hist,
                    edges=row_res_hist_edges,
                )
                plot_utils.plot_residuals(
                    histogram=row_res_hist,
                    edges=row_res_hist_edges,
                    fit=fit_row_res,
                    cov=cov_row_res,
                    xlabel='Row residual [$\mathrm{\mu}$m]',
                    title='Row residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_row_res = out_file_h5.create_carray(out_file_h5.root,
                                                        name='row_res_DUT%d' % (actual_dut),
                                                        title='Residual distribution in row direction for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(row_res_hist.dtype),
                                                        shape=row_res_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_row_res.attrs.yedges = row_res_hist_edges
                out_row_res.attrs.fit_coeff = fit_row_res
                out_row_res.attrs.fit_cov = cov_row_res
                out_row_res[:] = row_res_hist

                fit_col_res_col_pos, cov_col_res_col_pos, select = analysis_utils.fit_residuals_vs_position(
                    hist=col_res_col_pos_hist,
                    xedges=col_pos_hist_edges,
                    yedges=col_res_hist_edges,
                    mean=stat_col_res_col_pos_hist,
                    count=count_col_res_col_pos_hist,
                    fit_limit=fit_limit_x_local
                )
                plot_utils.plot_residuals_vs_position(
                    hist=col_res_col_pos_hist,
                    xedges=col_pos_hist_edges,
                    yedges=col_res_hist_edges,
                    xlabel='Column position [$\mathrm{\mu}$m]',
                    ylabel='Column residual [$\mathrm{\mu}$m]',
                    title='Column residuals vs. column positions for %s' % (dut_name,),
                    res_mean=stat_col_res_col_pos_hist,
                    select=select,
                    fit=fit_col_res_col_pos,
                    cov=cov_col_res_col_pos,
                    fit_limit=fit_limit_x_local,
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_col_res_col_pos = out_file_h5.create_carray(out_file_h5.root,
                                                            name='col_res_col_pos_DUT%d' % (actual_dut),
                                                            title='Residual distribution in column direction as a function of the column position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(col_res_col_pos_hist.dtype),
                                                            shape=col_res_col_pos_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col_res_col_pos.attrs.xedges = col_pos_hist_edges
                out_col_res_col_pos.attrs.yedges = col_res_hist_edges
                out_col_res_col_pos.attrs.fit_coeff = fit_col_res_col_pos
                out_col_res_col_pos.attrs.fit_cov = cov_col_res_col_pos
                out_col_res_col_pos[:] = col_res_col_pos_hist

                fit_row_res_row_pos, cov_row_res_row_pos, select = analysis_utils.fit_residuals_vs_position(
                    hist=row_res_row_pos_hist,
                    xedges=row_pos_hist_edges,
                    yedges=row_res_hist_edges,
                    mean=stat_row_res_row_pos_hist,
                    count=count_row_res_row_pos_hist,
                    fit_limit=fit_limit_y_local
                )
                plot_utils.plot_residuals_vs_position(
                    hist=row_res_row_pos_hist,
                    xedges=row_pos_hist_edges,
                    yedges=row_res_hist_edges,
                    xlabel='Row position [$\mathrm{\mu}$m]',
                    ylabel='Row residual [$\mathrm{\mu}$m]',
                    title='Row residuals vs. row positions for %s' % (dut_name,),
                    res_mean=stat_row_res_row_pos_hist,
                    select=select,
                    fit=fit_row_res_row_pos,
                    cov=cov_row_res_row_pos,
                    fit_limit=fit_limit_y_local,
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_row_res_row_pos = out_file_h5.create_carray(out_file_h5.root,
                                                            name='row_res_row_pos_DUT%d' % (actual_dut),
                                                            title='Residual distribution in row direction as a function of the row position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(row_res_row_pos_hist.dtype),
                                                            shape=row_res_row_pos_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_row_res_row_pos.attrs.xedges = row_pos_hist_edges
                out_row_res_row_pos.attrs.yedges = row_res_hist_edges
                out_row_res_row_pos.attrs.fit_coeff = fit_row_res_row_pos
                out_row_res_row_pos.attrs.fit_cov = cov_row_res_row_pos
                out_row_res_row_pos[:] = row_res_row_pos_hist

                fit_row_res_col_pos, cov_row_res_col_pos, select = analysis_utils.fit_residuals_vs_position(
                    hist=row_res_col_pos_hist,
                    xedges=col_pos_hist_edges,
                    yedges=row_res_hist_edges,
                    mean=stat_row_res_col_pos_hist,
                    count=count_row_res_col_pos_hist,
                    fit_limit=fit_limit_x_local
                )
                plot_utils.plot_residuals_vs_position(
                    hist=row_res_col_pos_hist,
                    xedges=col_pos_hist_edges,
                    yedges=row_res_hist_edges,
                    xlabel='Column position [$\mathrm{\mu}$m]',
                    ylabel='Row residual [$\mathrm{\mu}$m]',
                    title='Row residuals for vs. column positions %s' % (dut_name,),
                    res_mean=stat_row_res_col_pos_hist,
                    select=select,
                    fit=fit_row_res_col_pos,
                    cov=cov_row_res_col_pos,
                    fit_limit=fit_limit_x_local,
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_col_res_row_pos_pos = out_file_h5.create_carray(out_file_h5.root,
                                                            name='row_res_col_pos_DUT%d' % (actual_dut),
                                                            title='Residual distribution in row direction as a function of the column position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(row_res_col_pos_hist.dtype),
                                                            shape=row_res_col_pos_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col_res_row_pos_pos.attrs.xedges = col_pos_hist_edges
                out_col_res_row_pos_pos.attrs.yedges = row_res_hist_edges
                out_col_res_row_pos_pos.attrs.fit_coeff = fit_row_res_col_pos
                out_col_res_row_pos_pos.attrs.fit_cov = cov_row_res_col_pos
                out_col_res_row_pos_pos[:] = row_res_col_pos_hist

                fit_col_res_row_pos, cov_col_res_row_pos, select = analysis_utils.fit_residuals_vs_position(
                    hist=col_res_row_pos_hist,
                    xedges=row_pos_hist_edges,
                    yedges=col_res_hist_edges,
                    mean=stat_col_res_row_pos_hist,
                    count=count_col_res_row_pos_hist,
                    fit_limit=fit_limit_y_local
                )
                plot_utils.plot_residuals_vs_position(
                    hist=col_res_row_pos_hist,
                    xedges=row_pos_hist_edges,
                    yedges=col_res_hist_edges,
                    xlabel='Row position [$\mathrm{\mu}$m]',
                    ylabel='Column residual [$\mathrm{\mu}$m]',
                    title='Column residuals vs. row positions for %s' % (dut_name,),
                    res_mean=stat_col_res_row_pos_hist,
                    select=select,
                    fit=fit_col_res_row_pos,
                    cov=cov_col_res_row_pos,
                    fit_limit=fit_limit_y_local,
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_col_res_row_pos = out_file_h5.create_carray(out_file_h5.root,
                                                            name='col_res_row_pos_DUT%d' % (actual_dut),
                                                            title='Residual distribution in column direction as a function of the row position for %s' % (dut_name),
                                                            atom=tb.Atom.from_dtype(col_res_row_pos_hist.dtype),
                                                            shape=col_res_row_pos_hist.shape,
                                                            filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_col_res_row_pos.attrs.xedges = row_pos_hist_edges
                out_col_res_row_pos.attrs.yedges = col_res_hist_edges
                out_col_res_row_pos.attrs.fit_coeff = fit_col_res_row_pos
                out_col_res_row_pos.attrs.fit_cov = cov_col_res_row_pos
                out_col_res_row_pos[:] = col_res_row_pos_hist

                # 2D residual plot
                z_max = np.sqrt(fit_col_res[2] ** 2 + fit_row_res[2] ** 2)
                plot_utils.plot_2d_map(hist2d=stat_2d_res_hist.T,
                                       plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                                       title='2D average residuals for %s' % (dut_name,),
                                       x_axis_title='Column position [$\mathrm{\mu}$m]',
                                       y_axis_title='Row position [$\mathrm{\mu}$m]',
                                       z_min=0,
                                       z_max=z_max,
                                       output_pdf=output_pdf)

                # 2D hits plot
                count_2d_hist = np.ma.masked_equal(count_2d_hist, 0)
                plot_utils.plot_2d_map(hist2d=count_2d_hist.T,
                                       plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                                       title='2D occupancy for %s' % (dut_name,),
                                       x_axis_title='Column position [$\mathrm{\mu}$m]',
                                       y_axis_title='Row position [$\mathrm{\mu}$m]',
                                       z_min=0,
                                       z_max=None,
                                       output_pdf=output_pdf)

                # Global residuals
                fit_x_res, cov_x_res = analysis_utils.fit_residuals(
                    hist=x_res_hist,
                    edges=x_res_hist_edges,
                )
                plot_utils.plot_residuals(
                    histogram=x_res_hist,
                    edges=x_res_hist_edges,
                    fit=fit_x_res,
                    cov=cov_x_res,
                    xlabel='X residual [$\mathrm{\mu}$m]',
                    title='X residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_x_res = out_file_h5.create_carray(out_file_h5.root,
                                                      name='x_res_DUT%d' % (actual_dut),
                                                      title='Residual distribution in x direction for %s' % (dut_name),
                                                      atom=tb.Atom.from_dtype(x_res_hist.dtype),
                                                      shape=x_res_hist.shape,
                                                      filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_x_res.attrs.xedges = x_res_hist_edges
                out_x_res.attrs.fit_coeff = fit_x_res
                out_x_res.attrs.fit_cov = cov_x_res
                out_x_res[:] = x_res_hist

                fit_y_res, cov_y_res = analysis_utils.fit_residuals(
                    hist=y_res_hist,
                    edges=y_res_hist_edges,
                )
                plot_utils.plot_residuals(
                    histogram=y_res_hist,
                    edges=y_res_hist_edges,
                    fit=fit_y_res,
                    cov=cov_y_res,
                    xlabel='Y residual [$\mathrm{\mu}$m]',
                    title='Y residuals for %s' % (dut_name,),
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_y_res = out_file_h5.create_carray(out_file_h5.root,
                                                      name='y_res_DUT%d' % (actual_dut),
                                                      title='Residual distribution in y direction for %s' % (dut_name),
                                                      atom=tb.Atom.from_dtype(y_res_hist.dtype),
                                                      shape=y_res_hist.shape,
                                                      filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_y_res.attrs.yedges = y_res_hist_edges
                out_y_res.attrs.fit_coeff = fit_y_res
                out_y_res.attrs.fit_cov = cov_y_res
                out_y_res[:] = y_res_hist

                fit_x_res_x_pos, cov_x_res_x_pos, select = analysis_utils.fit_residuals_vs_position(
                    hist=x_res_x_pos_hist,
                    xedges=x_pos_hist_edges,
                    yedges=x_res_hist_edges,
                    mean=stat_x_res_x_pos_hist,
                    count=count_x_res_x_pos_hist
                )
                plot_utils.plot_residuals_vs_position(
                    hist=x_res_x_pos_hist,
                    xedges=x_pos_hist_edges,
                    yedges=x_res_hist_edges,
                    xlabel='X position [$\mathrm{\mu}$m]',
                    ylabel='X residual [$\mathrm{\mu}$m]',
                    title='X residuals vs. X positions for %s' % (dut_name,),
                    res_mean=stat_x_res_x_pos_hist,
                    select=select,
                    fit=fit_x_res_x_pos,
                    cov=cov_x_res_x_pos,
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_x_res_x_pos = out_file_h5.create_carray(out_file_h5.root,
                                                        name='x_res_x_pos_DUT%d' % (actual_dut),
                                                        title='Residual distribution in X direction as a function of the X position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(x_res_x_pos_hist.dtype),
                                                        shape=x_res_x_pos_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_x_res_x_pos.attrs.xedges = x_pos_hist_edges
                out_x_res_x_pos.attrs.yedges = x_res_hist_edges
                out_x_res_x_pos.attrs.fit_coeff = fit_x_res_x_pos
                out_x_res_x_pos.attrs.fit_cov = cov_x_res_x_pos
                out_x_res_x_pos[:] = x_res_x_pos_hist

                fit_y_res_y_pos, cov_y_res_y_pos, select = analysis_utils.fit_residuals_vs_position(
                    hist=y_res_y_pos_hist,
                    xedges=y_pos_hist_edges,
                    yedges=y_res_hist_edges,
                    mean=stat_y_res_y_pos_hist,
                    count=count_y_res_y_pos_hist
                )
                plot_utils.plot_residuals_vs_position(
                    hist=y_res_y_pos_hist,
                    xedges=y_pos_hist_edges,
                    yedges=y_res_hist_edges,
                    xlabel='Y position [$\mathrm{\mu}$m]',
                    ylabel='Y residual [$\mathrm{\mu}$m]',
                    title='Y residuals vs. Y positions for %s' % (dut_name,),
                    res_mean=stat_y_res_y_pos_hist,
                    select=select,
                    fit=fit_y_res_y_pos,
                    cov=cov_y_res_y_pos,
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_y_res_y_pos = out_file_h5.create_carray(out_file_h5.root,
                                                        name='y_res_y_pos_DUT%d' % (actual_dut),
                                                        title='Residual distribution in Y direction as a function of the Y position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(y_res_y_pos_hist.dtype),
                                                        shape=y_res_y_pos_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_y_res_y_pos.attrs.xedges = y_pos_hist_edges
                out_y_res_y_pos.attrs.yedges = y_res_hist_edges
                out_y_res_y_pos.attrs.fit_coeff = fit_y_res_y_pos
                out_y_res_y_pos.attrs.fit_cov = cov_y_res_y_pos
                out_y_res_y_pos[:] = y_res_y_pos_hist

                fit_y_res_x_pos, cov_y_res_x_pos, select = analysis_utils.fit_residuals_vs_position(
                    hist=y_res_x_pos_hist,
                    xedges=x_pos_hist_edges,
                    yedges=y_res_hist_edges,
                    mean=stat_y_res_x_pos_hist,
                    count=count_y_res_x_pos_hist
                )
                plot_utils.plot_residuals_vs_position(
                    hist=y_res_x_pos_hist,
                    xedges=x_pos_hist_edges,
                    yedges=y_res_hist_edges,
                    xlabel='X position [$\mathrm{\mu}$m]',
                    ylabel='Y residual [$\mathrm{\mu}$m]',
                    title='Y residuals vs. X positions for %s' % (dut_name,),
                    res_mean=stat_y_res_x_pos_hist,
                    select=select,
                    fit=fit_y_res_x_pos,
                    cov=cov_y_res_x_pos,
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_y_res_x_pos = out_file_h5.create_carray(out_file_h5.root,
                                                        name='y_res_x_pos_DUT%d' % (actual_dut),
                                                        title='Residual distribution in Y direction as a function of the X position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(y_res_x_pos_hist.dtype),
                                                        shape=y_res_x_pos_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_y_res_x_pos.attrs.xedges = x_pos_hist_edges
                out_y_res_x_pos.attrs.yedges = y_res_hist_edges
                out_y_res_x_pos.attrs.fit_coeff = fit_y_res_x_pos
                out_y_res_x_pos.attrs.fit_cov = cov_y_res_x_pos
                out_y_res_x_pos[:] = y_res_x_pos_hist

                fit_x_res_y_pos, cov_x_res_y_pos, select = analysis_utils.fit_residuals_vs_position(
                    hist=x_res_y_pos_hist,
                    xedges=y_pos_hist_edges,
                    yedges=x_res_hist_edges,
                    mean=stat_x_res_y_pos_hist,
                    count=count_x_res_y_pos_hist
                )
                plot_utils.plot_residuals_vs_position(
                    hist=x_res_y_pos_hist,
                    xedges=y_pos_hist_edges,
                    yedges=x_res_hist_edges,
                    xlabel='Y position [$\mathrm{\mu}$m]',
                    ylabel='X residual [$\mathrm{\mu}$m]',
                    title='X residuals vs. Y positions for %s' % (dut_name,),
                    res_mean=stat_x_res_y_pos_hist,
                    select=select,
                    fit=fit_x_res_y_pos,
                    cov=cov_x_res_y_pos,
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs
                )
                out_x_res_y_pos = out_file_h5.create_carray(out_file_h5.root,
                                                        name='x_res_y_pos_DUT%d' % (actual_dut),
                                                        title='Residual distribution in X direction as a function of the Y position for %s' % (dut_name),
                                                        atom=tb.Atom.from_dtype(x_res_y_pos_hist.dtype),
                                                        shape=x_res_y_pos_hist.shape,
                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_x_res_y_pos.attrs.xedges = y_pos_hist_edges
                out_x_res_y_pos.attrs.yedges = x_res_hist_edges
                out_x_res_y_pos.attrs.fit_coeff = fit_x_res_y_pos
                out_x_res_y_pos.attrs.fit_cov = cov_x_res_y_pos
                out_x_res_y_pos[:] = x_res_y_pos_hist

    if output_pdf is not None:
        output_pdf.close()

    if gui:
        return figs


# def calculate_efficiency(input_tracks_file, input_alignment_file, use_prealignment, bin_size, pixel_size, n_pixels, select_duts, output_efficiency_file=None, dut_names=None, sensor_sizes=None, minimum_tracks_per_bin=0, cut_distance=None, charge_bins=None, dut_masks=None, col_range=None, row_range=None, efficiency_range=None, show_inefficient_events=False, plot=True, chunk_size=1000000):
#     '''Takes the tracks and calculates the hit efficiency and hit/track hit distance for selected DUTs.
#
#     Parameters
#     ----------
#     input_tracks_file : string
#         Filename of the input tracks file.
#     input_alignment_file : string
#         Filename of the input alignment file.
#     use_prealignment : bool
#         If True, use pre-alignment from correlation data; if False, use alignment.
#     bin_size : iterable
#         Sizes of bins (i.e. (virtual) pixel size). Give one tuple (x, y) for every plane or list of tuples for different planes.
#     sensor_size : Tuple or list of tuples
#         Describes the sensor size for each DUT. If one tuple is given it is (size x, size y)
#         If several tuples are given it is [(DUT0 size x, DUT0 size y), (DUT1 size x, DUT1 size y), ...]
#     output_efficiency_file : string
#         Filename of the output efficiency file. If None, the filename will be derived from the input hits file.
#     minimum_track_density : int
#         Minimum track density required to consider bin for efficiency calculation.
#     select_duts : iterable
#         Selecting DUTs that will be processed.
#     cut_distance : int
#         Use only distances (between DUT hit and track hit) smaller than cut_distance.
#     col_range, row_range : iterable
#         Column / row value to calculate efficiency for (to neglect noisy edge pixels for efficiency calculation).
#     plot : bool
#         If True, create additional output plots.
#     chunk_size : int
#         Chunk size of the data when reading from file.
#     '''
#     logging.info('=== Calculating efficiency ===')
#
#     if output_efficiency_file is None:
#         output_efficiency_file = os.path.splitext(input_tracks_file)[0] + '_efficiency.h5'
#
#     if plot is True:
#         output_pdf = PdfPages(os.path.splitext(output_efficiency_file)[0] + '.pdf')
#     else:
#         output_pdf = None
#
#     with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
#         if use_prealignment:
#             logging.info('Use pre-alignment data')
#             prealignment = in_file_h5.root.PreAlignment[:]
#             n_duts = prealignment.shape[0]
#         else:
#             logging.info('Use alignment data')
#             alignment = in_file_h5.root.Alignment[:]
#             n_duts = alignment.shape[0]
#
#     select_duts = select_duts if select_duts is not None else range(n_duts)  # standard setting: fit tracks for all DUTs
#
#     sensor_sizes = [sensor_sizes, ] if not isinstance(sensor_sizes, Iterable) else sensor_size  # Sensor dimensions for each DUT
#
#     if not isinstance(cut_distance, Iterable):
#         cut_distance = [cut_distance] * len(select_duts)
#
#     if not isinstance(charge_bins, Iterable):
#         charge_bins = [charge_bins] * len(select_duts)
#
#     if not isinstance(dut_masks, Iterable):
#         dut_masks = [dut_masks] * len(select_duts)
#
#     if not isinstance(efficiency_range, Iterable):
#         efficiency_range = [efficiency_range] * len(select_duts)
#
#     output_pdf_file = os.path.splitext(output_efficiency_file)[0] + '.pdf'
#
#     efficiencies = []
#     pass_tracks = []
#     total_tracks = []
#     with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
#         with tb.open_file(output_efficiency_file, 'w') as out_file_h5:
#             for dut_index, actual_dut in enumerate(select_duts):
#                 node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT_%d' % actual_dut)
#                 print "actual_dut", actual_dut
#                 print "dut_index", dut_index
#                 dut_name = dut_names[actual_dut] if dut_names else ("DUT " + str(actual_dut))
#                 logging.info('Calculating efficiency for DUT%d', actual_dut)
#
#                 # Calculate histogram properties (bins size and number of bins)
#                 bin_size = [bin_size, ] if not isinstance(bin_size, Iterable) else bin_size
#                 if len(bin_size) == 1:
#                     actual_bin_size_x = bin_size[0][0]
#                     actual_bin_size_y = bin_size[0][1]
#                 else:
#                     actual_bin_size_x = bin_size[dut_index][0]
#                     actual_bin_size_y = bin_size[dut_index][1]
#
#                 if len(sensor_sizes) == 1:
#                     sensor_size = sensor_sizes[0]
#                 else:
#                     sensor_size = sensor_sizes[actual_dut]
#                 if sensor_size is None:
#                     sensor_size = np.array(pixel_size[actual_dut]) * n_pixels[actual_dut]
#                 print "sensor_size", sensor_size
#
#                 extend_bins = 0
#                 sensor_range_corr = [[-0.5 * pixel_size[actual_dut][0] * n_pixels[actual_dut][0] - extend_bins * actual_bin_size_x, 0.5 * pixel_size[actual_dut][0] * n_pixels[actual_dut][0] + extend_bins * actual_bin_size_x], [- 0.5 * pixel_size[actual_dut][1] * n_pixels[actual_dut][1] - extend_bins * actual_bin_size_y, 0.5 * pixel_size[actual_dut][1] * n_pixels[actual_dut][1] + extend_bins * actual_bin_size_y]]
#                 print "sensor_range_corr", sensor_range_corr
#
#                 sensor_range_corr_with_distance = sensor_range_corr[:]
#                 sensor_range_corr_with_distance.append([0, cut_distance[dut_index]])
#
#                 sensor_range_corr_with_charge = sensor_range_corr[:]
#                 sensor_range_corr_with_charge.append([0, charge_bins[dut_index]])
#
#                 print sensor_size[0], actual_bin_size_x, sensor_size[1], actual_bin_size_y
#                 n_bin_x = sensor_size[0] / actual_bin_size_x
#                 n_bin_y = sensor_size[1] / actual_bin_size_y
#                 if not n_bin_x.is_integer() or not n_bin_y.is_integer():
#                     raise ValueError("change bin_size: %f, %f" % (n_bin_x, n_bin_x))
#                 n_bin_x = int(n_bin_x)
#                 n_bin_y = int(n_bin_y)
#                 # has to be even
#                 print "bins", n_bin_x, n_bin_y
#
#
#                 # Define result histograms, these are filled for each hit chunk
# #                 total_distance_array = np.zeros(shape=(n_bin_x, n_bin_y, max_distance))
#                 total_hit_hist = None
#                 total_track_density = None
#                 total_track_density_with_dut_hit = None
#                 distance_array = None
#                 hit_hist = None
#                 charge_array = None
#                 average_charge_valid_hit = None
#
#                 for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
#                     # Transform the hits and track intersections into the local coordinate system
#                     # Coordinates in global coordinate system (x, y, z)
#                     hit_x, hit_y, hit_z = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
#                     charge = tracks_chunk['charge_dut_%d' % actual_dut]
#
#                     # track intersection at DUT
#                     intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']
#
#                     if use_prealignment:
#                         hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
#                                                                                                dut_index=actual_dut,
#                                                                                                prealignment=prealignment,
#                                                                                                inverse=True)
#                         intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
#                                                                                                                           dut_index=actual_dut,
#                                                                                                                           prealignment=prealignment,
#                                                                                                                           inverse=True)
#                     else:
#                         hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hit_x, hit_y, hit_z,
#                                                                                                dut_index=actual_dut,
#                                                                                                alignment=alignment,
#                                                                                                inverse=True)
#                         intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
#                                                                                                                           dut_index=actual_dut,
#                                                                                                                           alignment=alignment,
#                                                                                                                           inverse=True)
#
#                     intersections_local = np.column_stack((intersection_x_local, intersection_y_local, intersection_z_local))
#                     hits_local = np.column_stack((hit_x_local, hit_y_local, hit_z_local))
#
#                     # Select valid hits/tracks
#                     selection = np.logical_and(~np.isnan(hit_x), ~np.isnan(hit_y))
#
#                     if not np.allclose(hit_z_local[selection], 0.0) or not np.allclose(intersection_z_local[selection], 0.0):
#                         raise RuntimeError('The transformation to the local coordinate system did not give all z = 0. Wrong alignment used?')
# #                     if not np.allclose(hits_local[np.isfinite(hits_local[:, 2]), 2], 0.0) or not np.allclose(intersection_z_local, 0.0):
# #                         raise RuntimeError("Transformation into local coordinate system gives z != 0")
#
#                     # Usefull for debugging, print some inefficient events that can be cross checked
#                     if show_inefficient_events:
#                         logging.info('These events are inefficient: %s', str(tracks_chunk['event_number'][selection]))
#
#                     # Select hits from column, row range (e.g. to supress edge pixels)
# #                     col_range = [col_range, ] if not isinstance(col_range, Iterable) else col_range
# #                     if len(col_range) == 1:
# #                         curr_col_range = col_range[0]
# #                     else:
# #                         curr_col_range = col_range[dut_index]
# #                     if curr_col_range is not None:
# #                         selection = np.logical_and(intersections_local[:, 0] >= curr_col_range[0], intersections_local[:, 0] <= curr_col_range[1])  # Select real hits
# #                         hits_local, intersections_local = hits_local[selection], intersections_local[selection]
# #
# #                     row_range = [row_range, ] if not isinstance(row_range, Iterable) else row_range
# #                     if len(row_range) == 1:
# #                         curr_row_range = row_range[0]
# #                     else:
# #                         curr_row_range = row_range[dut_index]
# #                     if curr_row_range is not None:
# #                         selection = np.logical_and(intersections_local[:, 1] >= curr_row_range[0], intersections_local[:, 1] <= curr_row_range[1])  # Select real hits
# #                         hits_local, intersections_local = hits_local[selection], intersections_local[selection]
#
#                     # Calculate distance between track hit and DUT hit
#                     # TODO: scale correct? USE np.square(np.array((1, 1, 1)))
#                     scale = np.square(np.array((1, 1, 1)))  # Regard pixel size for calculating distances
#                     distance = np.sqrt(np.dot(np.square(intersections_local - hits_local), scale))  # Array with distances between DUT hit and track hit for each event. Values in um
#
#                     total_hit_hist_tmp = np.histogram2d(hits_local[:, 0], hits_local[:, 1], bins=(n_bin_x, n_bin_y), range=sensor_range_corr)[0]
#                     if total_hit_hist is None:
#                         total_hit_hist = total_hit_hist_tmp
#                     else:
#                         total_hit_hist += total_hit_hist_tmp
#
#                     total_track_density_tmp = np.histogram2d(intersections_local[:, 0], intersections_local[:, 1], bins=(n_bin_x, n_bin_y), range=sensor_range_corr)[0]
#                     if total_track_density is None:
#                         total_track_density = total_track_density_tmp
#                     else:
#                         total_track_density += total_track_density_tmp
#
#                         # Calculate efficiency
#                     if cut_distance[dut_index] is not None:  # Select intersections where hit is in given distance around track intersection
#                         selection = np.logical_and(selection, distance < cut_distance[dut_index])
#                     intersections_local_valid_hit = intersections_local[selection]
#                     hits_local_valid_hit = hits_local[selection]
#                     charge_valid_hit = charge[selection]
#
#                     total_track_density_with_dut_hit_tmp, xedges, yedges = np.histogram2d(intersections_local_valid_hit[:, 0], intersections_local_valid_hit[:, 1], bins=(n_bin_x, n_bin_y), range=sensor_range_corr)
#                     if total_track_density_with_dut_hit is None:
#                         total_track_density_with_dut_hit = total_track_density_with_dut_hit_tmp
#                     else:
#                         total_track_density_with_dut_hit += total_track_density_with_dut_hit_tmp
#
#                     intersections_distance = np.column_stack((intersections_local[:, 0], intersections_local[:, 1], distance))
#
#                     distance_array_tmp = np.histogramdd(intersections_distance, bins=(n_bin_x, n_bin_y, 100), range=sensor_range_corr_with_distance)[0]
#                     if distance_array is None:
#                         distance_array = distance_array_tmp
#                     else:
#                         distance_array += distance_array_tmp
#
#                     hit_hist_tmp = np.histogram2d(hits_local[:, 0], hits_local[:, 1], bins=(n_bin_x, n_bin_y), range=sensor_range_corr)[0]
#                     if hit_hist is None:
#                         hit_hist = hit_hist_tmp
#                     else:
#                         hit_hist += hit_hist_tmp
#
#                     if charge_bins[dut_index] is not None:
#                         average_charge_valid_hit_tmp, _, _, _ = stats.binned_statistic_2d(intersections_local_valid_hit[:, 0], intersections_local_valid_hit[:, 1], charge_valid_hit[:], statistic="mean", bins=(n_bin_x, n_bin_y), range=sensor_range_corr)
#                         average_charge_valid_hit_tmp = np.nan_to_num(average_charge_valid_hit_tmp)
#                         if average_charge_valid_hit is None:
#                             average_charge_valid_hit = average_charge_valid_hit_tmp
#                         else:
#                             average_charge_valid_hit[total_track_density_with_dut_hit != 0] = (((average_charge_valid_hit * total_track_density_with_dut_hit_previous) + (average_charge_valid_hit_tmp * total_track_density_with_dut_hit_tmp)) / total_track_density_with_dut_hit)[total_track_density_with_dut_hit != 0]
#                         total_track_density_with_dut_hit_previous = total_track_density_with_dut_hit.copy()
#
#                         intersection_charge_valid_hit = np.column_stack((intersections_local_valid_hit[:, 0], intersections_local_valid_hit[:, 1], charge_valid_hit[:]))
#                         charge_array_tmp = np.histogramdd(intersection_charge_valid_hit, bins=(n_bin_x, n_bin_y, charge_bins[dut_index]), range=sensor_range_corr_with_charge)[0]
#                         if charge_array is None:
#                             charge_array = charge_array_tmp
#                         else:
#                             charge_array += charge_array_tmp
#
#                     if np.all(total_track_density == 0):
#                         logging.warning('No tracks on DUT%d, cannot calculate efficiency', actual_dut)
#                         continue
#
#                 if charge_bins[dut_index] is not None:
#                     average_charge_valid_hit = np.ma.masked_where(total_track_density_with_dut_hit == 0, average_charge_valid_hit)
#                 # efficiency
#                 efficiency = np.full_like(total_track_density_with_dut_hit, fill_value=np.nan, dtype=np.float)
#                 efficiency[total_track_density != 0] = total_track_density_with_dut_hit[total_track_density != 0].astype(np.float) / total_track_density[total_track_density != 0].astype(np.float) * 100.0
#                 efficiency = np.ma.masked_invalid(efficiency)
#                 efficiency = np.ma.masked_where(total_track_density < minimum_tracks_per_bin, efficiency)
#
#                 distance_mean_array = np.average(distance_array, axis=2, weights=range(0, 100)) * sum(range(0, 100)) / np.sum(distance_array, axis=2)
#
#                 distance_mean_array = np.ma.masked_invalid(distance_mean_array)
#
#                 print "bins with tracks", np.ma.count(efficiency), "of", efficiency.shape[0] * efficiency.shape[1]
#                 print "tracks outside left / right", np.where(intersections_local_valid_hit[:, 0] < sensor_range_corr[0][0])[0].shape[0], np.where(intersections_local_valid_hit[:, 0] > sensor_range_corr[0][1])[0].shape[0]
#                 print "tracks outside below / above", np.where(intersections_local_valid_hit[:, 1] < sensor_range_corr[1][0])[0].shape[0], np.where(intersections_local_valid_hit[:, 1] > sensor_range_corr[1][1])[0].shape[0]
#
#                 # Calculate mean efficiency without any binning
#                 eff, eff_err_min, eff_err_pl = analysis_utils.get_mean_efficiency(array_pass=total_track_density_with_DUT_hit,
#                                                                                   array_total=total_track_density)
#
#                 logging.info('Efficiency =  %.4f (+%.4f/-%.4f)', eff, eff_err_pl, eff_err_min)
#                 efficiencies.append(np.ma.mean(efficiency))
#
#                 if pixel_size:
#                     aspect = pixel_size[actual_dut][1] / pixel_size[actual_dut][0]
#                 else:
#                     aspect = "auto"
#
#                 plot_utils.efficiency_plots(
#                     distance_mean_array=distance_mean_array,
#                     hit_hist=hit_hist,
#                     track_density=total_track_density,
#                     track_density_with_hit=total_track_density_with_dut_hit,
#                     efficiency=efficiency,
#                     charge_array=charge_array,
#                     average_charge=average_charge_valid_hit,
#                     dut_name=dut_name,
#                     plot_range=sensor_range_corr,
#                     efficiency_range=efficiency_range[dut_index],
#                     bin_size=bin_size[dut_index],
#                     xedges=xedges,
#                     yedges=yedges,
#                     n_pixels=n_pixels[actual_dut],
#                     charge_bins=charge_bins[dut_index],
#                     dut_mask=dut_masks[dut_index],
#                     aspect=aspect,
#                     output_pdf=output_pdf,
#                     gui=gui,
#                     figs=figs)
#
#                 dut_group = out_file_h5.create_group(out_file_h5.root, 'DUT_%d' % actual_dut)
#
#                 out_efficiency = out_file_h5.create_carray(dut_group, name='Efficiency', title='Efficiency per bin of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(efficiency.dtype), shape=efficiency.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
#                 out_tracks_per_bin = out_file_h5.create_carray(dut_group, name='Tracks_per_bin', title='Tracks per bin of DUT%d' % actual_dut, atom=tb.Atom.from_dtype(total_track_density.dtype), shape=total_track_density.T.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
#                 # Store parameters used for efficiency calculation
#                 # TODO: add attributes to DUT group
#                 # TODO: adding all attributes and histograms
#                 out_efficiency.attrs.bin_size = bin_size
#                 out_efficiency.attrs.minimum_tracks_per_bin = minimum_tracks_per_bin
#                 out_efficiency.attrs.sensor_size = sensor_size
#                 out_efficiency.attrs.cut_distance = cut_distance[dut_index]
#                 out_efficiency.attrs.charge_bins = charge_bins[dut_index]
#     #             out_efficiency.attrs.col_range = col_range
#     #             out_efficiency.attrs.row_range = row_range
#                 out_efficiency[:] = efficiency.T
#                 out_tracks_per_bin[:] = total_track_density.T
#                 efficiencies.append(np.ma.mean(efficiency))
#                 pass_tracks.append(total_track_density_with_dut_hit.sum())
#                 total_tracks.append(total_track_density.sum())
#
#     if output_pdf is not None:
#         output_pdf.close()
#
#     return efficiencies, pass_tracks, total_tracks


def calculate_efficiency(input_tracks_file, input_alignment_file, use_prealignment, select_duts, pixel_size, n_pixels, bin_size=None, n_bins_in_pixel=None, n_pixel_projection=None, output_efficiency_file=None, minimum_track_density=1, cut_distance=None, col_range=None, row_range=None, show_inefficient_events=False, plot=True, in_pixel=False, gui=False, chunk_size=1000000):
    '''Takes the tracks and calculates the hit efficiency and hit/track hit distance for selected DUTs.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    input_alignment_file : string
        Filename of the input alignment file.
    use_prealignment : bool
        If True, use pre-alignment from correlation data; if False, use alignment.
    select_duts : iterable
        Selecting DUTs that will be processed.
    pixel_size : iterable
        tuple or list of col/row pixel dimension
    n_pixels : iterable
        tuple or list of amount of pixel in col/row dimension
    bin_size : iterable
        Sizes of bins (i.e. (virtual) pixel size). Give one tuple (x, y) for every plane or list of tuples for different planes.
    n_bins_in_pixel : iterable
        Number of bins used for in-pixel efficiency calculation. Give one tuple (n_bins_x, n_bins_y) for every plane or list of tuples for different planes.
        Only needed if in_pixel is True.
    n_pixel_projection : int
        Number of pixels on which efficiency is projected. Only needed if in_pixel is True.
    output_efficiency_file : string
        Filename of the output efficiency file. If None, the filename will be derived from the input hits file.
    minimum_track_density : int
        Minimum track density required to consider bin for efficiency calculation.
    cut_distance : int
        Use only distances (between DUT hit and track hit) smaller than cut_distance.
    col_range : iterable
        Column value to calculate efficiency for (to neglect noisy edge pixels for efficiency calculation).
    row_range : iterable
        Row value to calculate efficiency for (to neglect noisy edge pixels for efficiency calculation).
    show_inefficient_events : bool
        Whether to log inefficient events
    plot : bool
        If True, create additional output plots.
    in_pixel : bool
        If True, calculate and plot in-pixel efficiency. Default is False.
    gui : bool
        If True, use GUI for plotting.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Calculating efficiency ===')

    if output_efficiency_file is None:
        output_efficiency_file = os.path.splitext(input_tracks_file)[0] + '_efficiency.h5'

    if plot is True and not gui:
        output_pdf = PdfPages(os.path.splitext(output_efficiency_file)[0] + '.pdf', keep_empty=False)
    else:
        output_pdf = None

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        if use_prealignment:
            logging.info('Use pre-alignment data')
            prealignment = in_file_h5.root.PreAlignment[:]
            n_duts = prealignment.shape[0]
        else:
            logging.info('Use alignment data')
            alignment = in_file_h5.root.Alignment[:]
            n_duts = alignment.shape[0]

    efficiencies = []
    pass_tracks = []
    total_tracks = []
    figs = [] if gui else None
    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_efficiency_file, 'w') as out_file_h5:
            for dut_index, actual_dut in enumerate(select_duts):
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT_%d' % actual_dut)
                logging.info('Calculating efficiency for DUT%d', actual_dut)

                # Calculate histogram properties (bins size and number of bins)
                bin_size = [bin_size, ] if not isinstance(bin_size, Iterable) else bin_size
                if len(bin_size) == 1:
                    actual_bin_size_x = bin_size[0][0]
                    actual_bin_size_y = bin_size[0][1]
                else:
                    actual_bin_size_x = bin_size[actual_dut][0]
                    actual_bin_size_y = bin_size[actual_dut][1]

                if in_pixel is True:
                    n_bins_in_pixel = [n_bins_in_pixel, ] if not isinstance(n_bins_in_pixel, Iterable) else n_bins_in_pixel
                    if len(n_bins_in_pixel) == 1:
                        actual_bin_size_in_pixel_x = n_bins_in_pixel[0][0]
                        actual_bin_size_in_pixel_y = n_bins_in_pixel[0][1]
                    else:
                        actual_bin_size_in_pixel_x = n_bins_in_pixel[actual_dut][0]
                        actual_bin_size_in_pixel_y = n_bins_in_pixel[actual_dut][1]

                sensor_size = np.array(n_pixels) * np.array(pixel_size)

                dimensions = [sensor_size, ] if not isinstance(sensor_size, Iterable) else sensor_size  # Sensor dimensions for each DUT
                if len(dimensions) == 1:
                    dimensions = dimensions[0]
                else:
                    dimensions = dimensions[actual_dut]
                n_bin_x = int(dimensions[0] / actual_bin_size_x)
                n_bin_y = int(dimensions[1] / actual_bin_size_y)

                # Define result histograms, these are filled for each hit chunk
#                 total_distance_array = np.zeros(shape=(n_bin_x, n_bin_y, max_distance))
                total_hit_hist = np.zeros(shape=(n_bin_x, n_bin_y), dtype=np.uint32)
                total_track_density = np.zeros(shape=(n_bin_x, n_bin_y))
                total_track_density_with_DUT_hit = np.zeros(shape=(n_bin_x, n_bin_y))

                if in_pixel is True:
                    total_hit_hist_projected = np.zeros(shape=(actual_bin_size_in_pixel_x, actual_bin_size_in_pixel_y), dtype=np.uint32)
                    total_track_density_projected = np.zeros(shape=(actual_bin_size_in_pixel_x, actual_bin_size_in_pixel_y))
                    total_track_density_with_DUT_hit_projected = np.zeros(shape=(actual_bin_size_in_pixel_x, actual_bin_size_in_pixel_y))

                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    # Transform the hits and track intersections into the local coordinate system
                    # Coordinates in global coordinate system (x, y, z)
                    hit_x_local, hit_y_local, hit_z_local = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
                    intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']

                    if use_prealignment:
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          prealignment=prealignment,
                                                                                                                          inverse=True)
                    else:
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          alignment=alignment,
                                                                                                                          inverse=True)

                    # Quickfix that center of sensor is local system is in the center and not at the edge
                    hit_x_local, hit_y_local = hit_x_local + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], hit_y_local + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]
                    intersection_x_local, intersection_y_local = intersection_x_local + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], intersection_y_local + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]

                    intersections_local = np.column_stack((intersection_x_local, intersection_y_local, intersection_z_local))
                    hits_local = np.column_stack((hit_x_local, hit_y_local, hit_z_local))

                    if not np.allclose(hit_z_local[np.isfinite(hit_z_local)], 0.0) or not np.allclose(intersection_z_local[np.isfinite(intersection_z_local)], 0.0):
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

                    selection = ~np.isnan(hits_local[:, 0])
                    if cut_distance:  # Select intersections where hit is in given distance around track intersection
                        intersection_valid_hit = intersections_local[np.logical_and(selection, distance < cut_distance)]
                    else:
                        intersection_valid_hit = intersections_local[selection]

                    total_track_density += np.histogram2d(intersections_local[:, 0], intersections_local[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]
                    total_track_density_with_DUT_hit += np.histogram2d(intersection_valid_hit[:, 0], intersection_valid_hit[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]

                    # project intersections and hits onto n x n pixel area
                    if in_pixel is True:
                        n = n_pixel_projection  # select pixel areas (n x n)
                        hit_x_local_projection = np.mod(hit_x_local, np.array([n * pixel_size[actual_dut][0]] * len(hit_x_local)))
                        hit_y_local_projection = np.mod(hit_y_local, np.array([n * pixel_size[actual_dut][1]] * len(hit_y_local)))
                        intersection_x_local_projection = np.mod(intersections_local[:, 0], np.array([n * pixel_size[actual_dut][0]] * len(intersections_local[:, 0])))
                        intersection_y_local_projection = np.mod(intersections_local[:, 1], np.array([n * pixel_size[actual_dut][1]] * len(intersections_local[:, 1])))
                        intersection_valid_hit_projection_x = np.mod(intersection_valid_hit[:, 0], np.array([n * pixel_size[actual_dut][0]] * len(intersection_valid_hit[:, 0])))
                        intersection_valid_hit_projection_y = np.mod(intersection_valid_hit[:, 1], np.array([n * pixel_size[actual_dut][1]] * len(intersection_valid_hit[:, 1])))

                        intersections_local_projection = np.column_stack((intersection_x_local_projection, intersection_y_local_projection))
                        hits_local_projection = np.column_stack((hit_x_local_projection, hit_y_local_projection))
                        intersections_valid_hit_projection = np.column_stack((intersection_valid_hit_projection_x, intersection_valid_hit_projection_y))

                        total_hit_hist_projected += (np.histogram2d(hits_local_projection[:, 0], hits_local_projection[:, 1],
                                                                    bins=(actual_bin_size_in_pixel_x, actual_bin_size_in_pixel_y),
                                                                    range=[[0, n * pixel_size[actual_dut][0]], [0, n * pixel_size[actual_dut][1]]])[0]).astype(np.uint32)
                        total_track_density_projected += np.histogram2d(intersections_local_projection[:, 0], intersections_local_projection[:, 1],
                                                                        bins=(actual_bin_size_in_pixel_x, actual_bin_size_in_pixel_y),
                                                                        range=[[0, n * pixel_size[actual_dut][0]], [0, n * pixel_size[actual_dut][1]]])[0]
                        total_track_density_with_DUT_hit_projected += np.histogram2d(intersections_valid_hit_projection[:, 0], intersections_valid_hit_projection[:, 1],
                                                                                     bins=(actual_bin_size_in_pixel_x, actual_bin_size_in_pixel_y),
                                                                                     range=[[0, n * pixel_size[actual_dut][0]], [0, n * pixel_size[actual_dut][1]]])[0]

                    if np.all(total_track_density == 0):
                        logging.warning('No tracks on DUT%d, cannot calculate efficiency', actual_dut)
                        continue

                # Calculate efficiency
                efficiency = np.zeros_like(total_track_density_with_DUT_hit)
                efficiency[total_track_density != 0] = total_track_density_with_DUT_hit[total_track_density != 0].astype(np.float) / total_track_density[total_track_density != 0].astype(np.float) * 100.
                efficiency = np.ma.array(efficiency, mask=total_track_density < minimum_track_density)

                # Calculate in-pixel-efficiency
                if in_pixel is True:
                    in_pixel_efficiency = np.zeros_like(total_track_density_with_DUT_hit_projected)
                    in_pixel_efficiency[total_track_density_projected != 0] = total_track_density_with_DUT_hit_projected[total_track_density_projected != 0].astype(np.float) / total_track_density_projected[total_track_density_projected != 0].astype(np.float) * 100.
                    in_pixel_efficiency = np.ma.array(in_pixel_efficiency, mask=total_track_density_projected < minimum_track_density)
                if not np.any(efficiency):
                    raise RuntimeError('All efficiencies for DUT%d are zero, consider changing cut values!', actual_dut)

                if in_pixel is True:
                    plot_utils.efficiency_plots(total_hit_hist,
                                                total_track_density,
                                                total_track_density_with_DUT_hit,
                                                efficiency,
                                                actual_dut,
                                                plot_range=[0.0, dimensions[0], dimensions[1], 0.0],
                                                in_pixel_efficiency=in_pixel_efficiency,
                                                plot_range_in_pixel=[0.0, n * pixel_size[actual_dut][0], n * pixel_size[actual_dut][1], 0.0],
                                                output_pdf=output_pdf,
                                                gui=gui,
                                                figs=figs)
                else:
                    plot_utils.efficiency_plots(total_hit_hist,
                                                total_track_density,
                                                total_track_density_with_DUT_hit,
                                                efficiency,
                                                actual_dut,
                                                plot_range=[0.0, dimensions[0], dimensions[1], 0.0],
                                                output_pdf=output_pdf,
                                                gui=gui,
                                                figs=figs)

                # Calculate mean efficiency without any binning
                eff, eff_err_min, eff_err_pl = analysis_utils.get_mean_efficiency(array_pass=total_track_density_with_DUT_hit,
                                                                                  array_total=total_track_density)

                logging.info('Efficiency =  %.4f (+%.4f/-%.4f)', eff, eff_err_pl, eff_err_min)
                efficiencies.append(np.ma.mean(efficiency))

                dut_group = out_file_h5.create_group(out_file_h5.root, 'DUT_%d' % actual_dut)

                out_efficiency = out_file_h5.create_carray(dut_group, name='Efficiency', title='Efficiency map of DUT%d' % actual_dut,
                                                           atom=tb.Atom.from_dtype(efficiency.dtype), shape=efficiency.T.shape,
                                                           filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_efficiency_mask = out_file_h5.create_carray(dut_group, name='Efficiency_mask', title='Masked pixel map of DUT%d' % actual_dut,
                                                                atom=tb.Atom.from_dtype(efficiency.mask.dtype), shape=efficiency.mask.T.shape,
                                                                filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                if in_pixel is True:
                    out_in_pixel_efficiency = out_file_h5.create_carray(dut_group, name='In_Pixel_Efficiency', title='In-Pixel-Efficiency map of DUT%d' % actual_dut,
                                                                        atom=tb.Atom.from_dtype(in_pixel_efficiency.dtype), shape=in_pixel_efficiency.T.shape,
                                                                        filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                # For correct statistical error calculation the number of detected tracks over total tracks is needed
                out_pass = out_file_h5.create_carray(dut_group, name='Passing_tracks', title='Passing events of DUT%d' % actual_dut,
                                                     atom=tb.Atom.from_dtype(total_track_density_with_DUT_hit.dtype), shape=total_track_density_with_DUT_hit.T.shape,
                                                     filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                out_total = out_file_h5.create_carray(dut_group, name='Total_tracks', title='Total events of DUT%d' % actual_dut,
                                                      atom=tb.Atom.from_dtype(total_track_density.dtype), shape=total_track_density.T.shape,
                                                      filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                pass_tracks.append(total_track_density_with_DUT_hit.sum())
                total_tracks.append(total_track_density.sum())
                logging.info('Passing / total tracks: %d / %d', total_track_density_with_DUT_hit.sum(), total_track_density.sum())

                # Store parameters used for efficiency calculation
                out_efficiency.attrs.bin_size = bin_size
                out_efficiency.attrs.minimum_track_density = minimum_track_density
                out_efficiency.attrs.sensor_size = sensor_size
                out_efficiency.attrs.cut_distance = cut_distance
                out_efficiency.attrs.col_range = col_range
                out_efficiency.attrs.row_range = row_range
                out_efficiency[:] = efficiency.T
                if in_pixel is True:
                    out_in_pixel_efficiency[:] = in_pixel_efficiency.T
                out_efficiency_mask[:] = efficiency.mask.T
                out_pass[:] = total_track_density_with_DUT_hit.T
                out_total[:] = total_track_density.T

    if output_pdf is not None:
        output_pdf.close()

    if gui:
        return figs

    return eff, eff_err_min, eff_err_pl


def calculate_purity(input_tracks_file, input_alignment_file, use_prealignment, bin_size, sensor_size, select_duts, output_purity_file=None, pixel_size=None, n_pixels=None, minimum_hit_density=10, cut_distance=None, col_range=None, row_range=None, show_inefficient_events=False, plot=True, chunk_size=1000000):
    '''Takes the tracks and calculates the hit purity and hit/track hit distance for selected DUTs.

    Parameters
    ----------
    input_tracks_file : string
        Filename with the tracks table.
    input_alignment_file : pytables file
        Filename of the input aligment data.
    use_prealignment : bool
        If True, use pre-alignment from correlation data; if False, use alignment.
    bin_size : iterable
        Bins sizes (i.e. (virtual) pixel size). Give one tuple (x, y) for every plane or list of tuples for different planes.
    sensor_size : Tuple or list of tuples
        Describes the sensor size for each DUT. If one tuple is given it is (size x, size y).
        If several tuples are given it is [(DUT0 size x, DUT0 size y), (DUT1 size x, DUT1 size y), ...].
    output_purity_file : string
        Filename of the output purity file. If None, the filename will be derived from the input hits file.
    minimum_hit_density : int
        Minimum hit density required to consider bin for purity calculation.
    select_duts : iterable
        Selecting DUTs that will be processed.
    cut_distance : int
        Hit - track intersection <= cut_distance = pure hit (hit assigned to track).
        Hit - track intersection > cut_distance = inpure hit (hit without a track).
    col_range, row_range : iterable
        Column / row value to calculate purity for (to neglect noisy edge pixels for purity calculation).
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Calculating purity ===')

    if output_purity_file is None:
        output_purity_file = os.path.splitext(input_tracks_file)[0] + '_purity.h5'

    if plot is True:
        output_pdf = PdfPages(os.path.splitext(output_purity_file)[0] + '.pdf', keep_empty=False)
    else:
        output_pdf = None

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        if use_prealignment:
            logging.info('Use pre-alignment data')
            prealignment = in_file_h5.root.PreAlignment[:]
            n_duts = prealignment.shape[0]
        else:
            logging.info('Use alignment data')
            alignment = in_file_h5.root.Alignment[:]
            n_duts = alignment.shape[0]

    purities = []
    purities_sensor_mean = []
    pure_hits = []
    total_hits = []
    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_purity_file, 'w') as out_file_h5:
            for dut_index, actual_dut in enumerate(select_duts):
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT_%d' % actual_dut)
                logging.info('Calculating purity for DUT%d', actual_dut)

                # Calculate histogram properties (bins size and number of bins)
                bin_size = [bin_size, ] if not isinstance(bin_size, Iterable) else bin_size
                if len(bin_size) != 1:
                    actual_bin_size_x = bin_size[dut_index][0]
                    actual_bin_size_y = bin_size[dut_index][1]
                else:
                    actual_bin_size_x = bin_size[0][0]
                    actual_bin_size_y = bin_size[0][1]
                dimensions = [sensor_size, ] if not isinstance(sensor_size, Iterable) else sensor_size  # Sensor dimensions for each DUT
                if len(dimensions) == 1:
                    dimensions = dimensions[0]
                else:
                    dimensions = dimensions[dut_index]
                n_bin_x = int(dimensions[0] / actual_bin_size_x)
                n_bin_y = int(dimensions[1] / actual_bin_size_y)

                # Define result histograms, these are filled for each hit chunk
                total_hit_hist = np.zeros(shape=(n_bin_x, n_bin_y), dtype=np.uint32)
                total_pure_hit_hist = np.zeros(shape=(n_bin_x, n_bin_y), dtype=np.uint32)

                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    # Take only tracks where actual dut has a hit, otherwise residual wrong
                    selection = np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut]), ~np.isnan(tracks_chunk['track_chi2']))
                    selection_hit = ~np.isnan(tracks_chunk['x_dut_%d' % actual_dut])

                    # Transform the hits and track intersections into the local coordinate system
                    # Coordinates in global coordinate system (x, y, z)
                    hit_x_local_dut, hit_y_local_dut, hit_z_local_dut = tracks_chunk['x_dut_%d' % actual_dut][selection_hit], tracks_chunk['y_dut_%d' % actual_dut][selection_hit], tracks_chunk['z_dut_%d' % actual_dut][selection_hit]
                    hit_x_local, hit_y_local, hit_z_local = tracks_chunk['x_dut_%d' % actual_dut][selection], tracks_chunk['y_dut_%d' % actual_dut][selection], tracks_chunk['z_dut_%d' % actual_dut][selection]

                    intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'][selection], tracks_chunk['offset_1'][selection], tracks_chunk['offset_2'][selection]

                    if use_prealignment:
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          prealignment=prealignment,
                                                                                                                          inverse=True)
                    else:
                        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                          dut_index=actual_dut,
                                                                                                                          alignment=alignment,
                                                                                                                          inverse=True)

                    if not np.allclose(hit_z_local[np.isfinite(hit_z_local)], 0.0) or not np.allclose(intersection_z_local[np.isfinite(intersection_z_local)], 0.0):
                        raise RuntimeError("Transformation into local coordinate system gives z != 0")

                    # Quickfix that center of sensor is local system is in the center and not at the edge
                    hit_x_local_dut, hit_y_local_dut = hit_x_local_dut + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], hit_y_local_dut + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]
                    hit_x_local, hit_y_local = hit_x_local + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], hit_y_local + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]
                    intersection_x_local, intersection_y_local = intersection_x_local + pixel_size[actual_dut][0] / 2. * n_pixels[actual_dut][0], intersection_y_local + pixel_size[actual_dut][1] / 2. * n_pixels[actual_dut][1]

                    intersections_local = np.column_stack((intersection_x_local, intersection_y_local, intersection_z_local))
                    hits_local = np.column_stack((hit_x_local, hit_y_local, hit_z_local))
                    hits_local_dut = np.column_stack((hit_x_local_dut, hit_y_local_dut, hit_z_local_dut))

                    # Usefull for debugging, print some inpure events that can be cross checked
                    # Select virtual hits
                    sel_virtual = np.isnan(tracks_chunk['x_dut_%d' % actual_dut])
                    if show_inefficient_events:
                        logging.info('These events are not pure: %s', str(tracks_chunk['event_number'][sel_virtual]))

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

                    total_hit_hist += (np.histogram2d(hits_local_dut[:, 0], hits_local_dut[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]).astype(np.uint32)

                    # Calculate purity
                    pure_hits_local = hits_local[distance < cut_distance]

                    if not np.any(pure_hits_local):
                        logging.warning('No pure hits in DUT%d, cannot calculate purity', actual_dut)
                        continue
                    total_pure_hit_hist += (np.histogram2d(pure_hits_local[:, 0], pure_hits_local[:, 1], bins=(n_bin_x, n_bin_y), range=[[0, dimensions[0]], [0, dimensions[1]]])[0]).astype(np.uint32)

                purity = np.zeros_like(total_hit_hist)
                purity[total_hit_hist != 0] = total_pure_hit_hist[total_hit_hist != 0].astype(np.float) / total_hit_hist[total_hit_hist != 0].astype(np.float) * 100.
                purity = np.ma.array(purity, mask=total_hit_hist < minimum_hit_density)
                # calculate sensor purity by weighting each pixel purity with total number of hits within the pixel
                purity_sensor = np.repeat(purity.ravel(), total_hit_hist.ravel())

                if not np.any(purity):
                    raise RuntimeError('No pure hit for DUT%d, consider changing cut values or check track building!', actual_dut)

                plot_utils.purity_plots(total_pure_hit_hist, total_hit_hist, purity, purity_sensor, actual_dut, minimum_hit_density, plot_range=[0.0, dimensions[0], dimensions[1], 0.0], cut_distance=cut_distance, output_pdf=output_pdf)
                logging.info('Purity =  %1.4f +- %1.4f', np.ma.mean(purity), np.ma.std(purity))
                purities.append(np.ma.mean(purity))
                purities_sensor_mean.append(np.ma.mean(purity_sensor))

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
                out_purity.attrs.use_duts = select_duts
                out_purity.attrs.cut_distance = cut_distance
                out_purity.attrs.col_range = col_range
                out_purity.attrs.row_range = row_range
                out_purity.attrs.purity_average = total_pure_hit_hist.sum()/total_hit_hist.sum() * 100
                out_purity[:] = purity.T
                out_purity_mask[:] = purity.mask.T
                out_pure_hits[:] = total_pure_hit_hist.T
                out_total_total[:] = total_hit_hist.T

    if output_pdf is not None:
        output_pdf.close()

    return purities, pure_hits, total_hits, purities_sensor_mean


def histogram_track_angle(input_tracks_file, select_duts, input_alignment_file, use_prealignment, output_track_angle_file=None, n_bins="auto", plot_range=(None, None), dut_names=None, plot=True, chunk_size=1000000):
    '''Calculates and histograms the track angle of the fitted tracks for selected DUTs.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    input_alignment_file : string
        Filename of the input alignment file.
    use_prealignment : bool
        If True, use pre-alignment from correlation data; if False, use alignment.
        If True, the DUT planes are assumed to be perpendicular to the z axis.
    output_track_angle_file: string
        Filename of the output track angle file with track angle histogram and fitted means and sigmas of track angles for selected DUTs.
        If None, deduce filename from input tracks file.
    n_bins : uint
        Number of bins for the histogram.
        If "auto", automatic binning is used.
    plot_range : iterable of tuples
        Tuple of the plot range in rad for alpha and beta angular distribution, e.g. ((-0.01, +0.01), -0.01, +0.01)).
        If (None, None), plotting from minimum to maximum.
    select_duts : iterable
        Selecting DUTs that will be processed.
    dut_names : iterable
        Name of the DUTs. If None, DUT numbers will be used.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Calculating track angles ===')


    def get_angles(track_slopes, xz_plane_normal, yz_plane_normal, dut_plane_normal):
        # track slopes need to be normalized to 1
        track_slopes_onto_xz_plane = track_slopes - np.matmul(xz_plane_normal, track_slopes.T).reshape(-1, 1) * xz_plane_normal
        track_slopes_onto_xz_plane /= np.sqrt(np.einsum('ij,ij->i', track_slopes_onto_xz_plane, track_slopes_onto_xz_plane)).reshape(-1, 1)
        track_slopes_onto_yz_plane = track_slopes - np.matmul(yz_plane_normal, track_slopes.T).reshape(-1, 1) * yz_plane_normal
        track_slopes_onto_yz_plane /= np.sqrt(np.einsum('ij,ij->i', track_slopes_onto_yz_plane, track_slopes_onto_yz_plane)).reshape(-1, 1)
        normal_onto_xz_plane = dut_plane_normal - np.inner(xz_plane_normal, dut_plane_normal) * xz_plane_normal
        normal_onto_yz_plane = dut_plane_normal - np.inner(yz_plane_normal, dut_plane_normal) * yz_plane_normal
        alpha_angles = np.arctan2(np.matmul(yz_plane_normal, np.cross(track_slopes_onto_yz_plane, normal_onto_yz_plane).T), np.matmul(normal_onto_yz_plane, track_slopes_onto_yz_plane.T))
        beta_angles = np.arctan2(np.matmul(xz_plane_normal, np.cross(track_slopes_onto_xz_plane, normal_onto_xz_plane).T), np.matmul(normal_onto_xz_plane, track_slopes_onto_xz_plane.T))
        return alpha_angles, beta_angles

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        if use_prealignment:
            logging.info('Use pre-alignment data')
            prealignment = in_file_h5.root.PreAlignment[:]
            n_duts = prealignment.shape[0]
        else:
            logging.info('Use alignment data')
            alignment = in_file_h5.root.Alignment[:]
            n_duts = alignment.shape[0]

    if output_track_angle_file is None:
        output_track_angle_file = os.path.splitext(input_tracks_file)[0] + '_track_angles.h5'

    with tb.open_file(input_tracks_file, 'r') as in_file_h5:
        with tb.open_file(output_track_angle_file, mode="w") as out_file_h5:
            # insert DUT with lowest index at the beginning of the nodes list for calculating telescope tracks
            select_duts_mod = select_duts[:]
            select_duts_mod.insert(0, min(select_duts))
            for index, actual_dut in enumerate(select_duts_mod):
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT_%d' % actual_dut)

                if index == 0:
                    logging.info('Calculating track angles for telescope')
                    dut_name = None
                else:
                    logging.info('Calculating track angles for DUT%d', actual_dut)
                    dut_name = "DUT%d" % actual_dut

                if use_prealignment or dut_name is None:
                    dut_plane_normal = np.array([0.0, 0.0, 1.0])
                else:
                    rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[actual_dut]['alpha'],
                                                                     beta=alignment[actual_dut]['beta'],
                                                                     gamma=alignment[actual_dut]['gamma'])
                    basis_global = rotation_matrix.T.dot(np.eye(3))
                    dut_plane_normal = basis_global[2]
                    if dut_plane_normal[2] < 0:
                        dut_plane_normal = -dut_plane_normal

                initialize = True
                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):  # only store track slopes of selected DUTs
                    track_slopes = np.column_stack((tracks_chunk['slope_0'],
                                                    tracks_chunk['slope_1'],
                                                    tracks_chunk['slope_2']))

                    # TODO: alpha/beta wrt DUT col / row
                    total_angles = np.arccos(np.inner(dut_plane_normal, track_slopes))
                    xz_plane_normal = np.array([0.0, 1.0, 0.0])
                    yz_plane_normal = np.array([1.0, 0.0, 0.0])
                    alpha_angles, beta_angles = get_angles(track_slopes=track_slopes,
                                                           xz_plane_normal=xz_plane_normal,
                                                           yz_plane_normal=yz_plane_normal,
                                                           dut_plane_normal=dut_plane_normal)
                    if dut_name is not None:
                        xz_plane_normal = np.dot(rotation_matrix, np.array([0.0, 1.0, 0.0]))
                        yz_plane_normal = np.dot(rotation_matrix, np.array([1.0, 0.0, 0.0]))
                        alpha_angles_local, beta_angles_local = get_angles(track_slopes=track_slopes,
                                                                           xz_plane_normal=xz_plane_normal,
                                                                           yz_plane_normal=yz_plane_normal,
                                                                           dut_plane_normal=dut_plane_normal)

                    if initialize:
                        total_angle_hist, total_angle_hist_edges = np.histogram(total_angles, bins=n_bins, range=None)
                        alpha_angle_hist, alpha_angle_hist_edges = np.histogram(alpha_angles, bins=n_bins, range=plot_range[0])
                        beta_angle_hist, beta_angle_hist_edges = np.histogram(beta_angles, bins=n_bins, range=plot_range[1])
                        if dut_name is not None:
                            alpha_angle_local_hist, alpha_angle_local_hist_edges = np.histogram(alpha_angles_local, bins=n_bins, range=plot_range[0])
                            beta_angle_local_hist, beta_angle_local_hist_edges = np.histogram(beta_angles_local, bins=n_bins, range=plot_range[1])
                        initialize = False
                    else:
                        total_angle_hist += np.histogram(total_angles, bins=total_angle_hist_edges)[0]
                        alpha_angle_hist += np.histogram(alpha_angles, bins=alpha_angle_hist_edges)[0]
                        beta_angle_hist += np.histogram(beta_angles, bins=beta_angle_hist_edges)[0]
                        if dut_name is not None:
                            alpha_angle_local_hist += np.histogram(alpha_angles_local, bins=alpha_angle_local_hist_edges)[0]
                            beta_angle_local_hist += np.histogram(beta_angles_local, bins=beta_angle_local_hist_edges)[0]

                # write results
                track_angle_total = out_file_h5.create_carray(where=out_file_h5.root,
                                                              name='Total_track_angle_hist%s' % (("_%s" % dut_name) if dut_name else ""),
                                                              title='Total track angle distribution%s' % ((" for %s" % dut_name) if dut_name else ""),
                                                              atom=tb.Atom.from_dtype(total_angle_hist.dtype),
                                                              shape=total_angle_hist.shape)
                track_angle_total_edges = out_file_h5.create_carray(where=out_file_h5.root,
                                                                    name='Total_track_angle_edges%s' % (("_%s" % dut_name) if dut_name else ""),
                                                                    title='Total track angle hist edges%s' % ((" for %s" % dut_name) if dut_name else ""),
                                                                    atom=tb.Atom.from_dtype(total_angle_hist_edges.dtype),
                                                                    shape=total_angle_hist_edges.shape)
                track_angle_alpha = out_file_h5.create_carray(where=out_file_h5.root,
                                                              name='Alpha_track_angle_hist%s' % (("_%s" % dut_name) if dut_name else ""),
                                                              title='Alpha track angle distribution%s' % ((" for %s" % dut_name) if dut_name else ""),
                                                              atom=tb.Atom.from_dtype(alpha_angle_hist.dtype),
                                                              shape=alpha_angle_hist.shape)
                track_angle_alpha_edges = out_file_h5.create_carray(where=out_file_h5.root,
                                                                    name='Alpha_track_angle_edges%s' % (("_%s" % dut_name) if dut_name else ""),
                                                                    title='Alpha track angle hist edges%s' % ((" for %s" % dut_name) if dut_name else ""),
                                                                    atom=tb.Atom.from_dtype(alpha_angle_hist_edges.dtype),
                                                                    shape=alpha_angle_hist_edges.shape)
                track_angle_beta = out_file_h5.create_carray(where=out_file_h5.root,
                                                             name='Beta_track_angle_hist%s' % (("_%s" % dut_name) if dut_name else ""),
                                                             title='Beta track angle distribution%s' % ((" for %s" % dut_name) if dut_name else ""),
                                                             atom=tb.Atom.from_dtype(beta_angle_hist.dtype),
                                                             shape=beta_angle_hist.shape)
                track_angle_beta_edges = out_file_h5.create_carray(where=out_file_h5.root,
                                                                   name='Beta_track_angle_edges%s' % (("_%s" % dut_name) if dut_name else ""),
                                                                   title='Beta track angle hist edges%s' % ((" for %s" % dut_name) if dut_name else ""),
                                                                   atom=tb.Atom.from_dtype(beta_angle_hist_edges.dtype),
                                                                   shape=beta_angle_hist_edges.shape)
                if dut_name is not None:
                    track_angle_alpha_local = out_file_h5.create_carray(where=out_file_h5.root,
                                                                  name='Local_alpha_track_angle_hist_%s' % (dut_name,),
                                                                  title='Local alpha track angle distribution for %s' % (dut_name,),
                                                                  atom=tb.Atom.from_dtype(alpha_angle_local_hist.dtype),
                                                                  shape=alpha_angle_local_hist.shape)
                    track_angle_alpha_edges_local = out_file_h5.create_carray(where=out_file_h5.root,
                                                                        name='Local_alpha_track_angle_edges_%s' % (dut_name,),
                                                                        title='Local alpha track angle hist edges for %s' % (dut_name,),
                                                                        atom=tb.Atom.from_dtype(alpha_angle_local_hist_edges.dtype),
                                                                        shape=alpha_angle_local_hist_edges.shape)
                    track_angle_beta_local = out_file_h5.create_carray(where=out_file_h5.root,
                                                                 name='Local_beta_track_angle_hist_%s' % (dut_name,),
                                                                 title='Local beta track angle distribution %s' % (dut_name,),
                                                                 atom=tb.Atom.from_dtype(beta_angle_local_hist.dtype),
                                                                 shape=beta_angle_local_hist.shape)
                    track_angle_beta_edges_local = out_file_h5.create_carray(where=out_file_h5.root,
                                                                       name='Local_beta_track_angle_edges_%s' % (dut_name,),
                                                                       title='Local beta track angle hist edges for %s' % (dut_name,),
                                                                       atom=tb.Atom.from_dtype(beta_angle_local_hist_edges.dtype),
                                                                       shape=beta_angle_local_hist_edges.shape)

                loop_over_data = [(total_angle_hist, total_angle_hist_edges, track_angle_total, track_angle_total_edges),
                                  (alpha_angle_hist, alpha_angle_hist_edges, track_angle_alpha, track_angle_alpha_edges),
                                  (beta_angle_hist, beta_angle_hist_edges, track_angle_beta, track_angle_beta_edges)]
                if dut_name is not None:
                    loop_over_data.extend([(alpha_angle_local_hist, alpha_angle_local_hist_edges, track_angle_alpha_local, track_angle_alpha_edges_local),
                                           (beta_angle_local_hist, beta_angle_local_hist_edges, track_angle_beta_local, track_angle_beta_edges_local)])

                for hist, edges, hist_carray, edges_carray in loop_over_data:
                    # fit histograms
                    bin_center = (edges[1:] + edges[:-1]) / 2.0
                    mean = analysis_utils.get_mean_from_histogram(hist, bin_center)
                    rms = analysis_utils.get_rms_from_histogram(hist, bin_center)
                    try:
                        fit, _ = curve_fit(analysis_utils.gauss, bin_center, hist, p0=[np.amax(hist), mean, rms])
                    except RuntimeError:
                        hilb = hilbert(hist)
                        hilb = np.absolute(hilb)
                        fit, _ = curve_fit(analysis_utils.gauss, bin_center, hilb, p0=[np.amax(hist), mean, rms])
#                         fit = [np.nan, np.nan, np.nan]

                    # store the data
                    hist_carray.attrs.amplitude = fit[0]
                    hist_carray.attrs.mean = fit[1]
                    hist_carray.attrs.sigma = fit[2]
                    hist_carray[:] = hist
                    edges_carray[:] = edges

    if plot:
        plot_utils.plot_track_angle(input_track_angle_file=output_track_angle_file, select_duts=select_duts, output_pdf_file=None, dut_names=dut_names)


def calculate_residual_correlation(input_tracks_file, input_alignment_file, use_prealignment, n_pixels, pixel_size, plot_n_pixels, plot_n_bins, correlate_n_tracks=10000, output_residual_correlation_file=None, dut_names=None, select_duts=None, use_fit_limits=False, plot=True, chunk_size=1000000):
    logging.info('=== Calculating residual correlation ===')

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

    if output_residual_correlation_file is None:
        output_residual_correlation_file = os.path.splitext(input_tracks_file)[0] + '_residual_correlation.h5'

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_residual_correlation_file, mode='w') as out_file_h5:
            for dut_index, actual_dut in enumerate(select_duts):
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT_%d' % actual_dut)
                logging.info('Calculating residual correlation for DUT%d', actual_dut)

                stop = [plot_n_pixels[dut_index][0] * pixel_size[actual_dut][0], plot_n_pixels[dut_index][1] * pixel_size[actual_dut][1]]
                x_edges = np.linspace(start=0, stop=stop[0], num=plot_n_bins[dut_index][0] + 1, endpoint=True)
                y_edges = np.linspace(start=0, stop=stop[1], num=plot_n_bins[dut_index][1] + 1, endpoint=True)

                ref_x_residuals_earray = out_file_h5.create_earray(out_file_h5.root,
                                                                   name='Column_residuals_reference_DUT_%d' % actual_dut,
                                                                   atom=tb.Float32Atom(),
                                                                   shape=(0,),
                                                                   title='Reference column residuals',
                                                                   filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                x_residuals_earray = out_file_h5.create_earray(out_file_h5.root,
                                                               name='Column_residuals_DUT_%d' % actual_dut,
                                                               atom=tb.Float32Atom(),
                                                               shape=(0, plot_n_bins[dut_index][0]),
                                                               title='Column residuals',
                                                               filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                x_residuals_earray.attrs.edges = x_edges
                ref_y_residuals_earray = out_file_h5.create_earray(out_file_h5.root,
                                                                   name='Row_residuals_reference_DUT_%d' % actual_dut,
                                                                   atom=tb.Float32Atom(),
                                                                   shape=(0,),
                                                                   title='Reference row residuals',
                                                                   filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                y_residuals_earray = out_file_h5.create_earray(out_file_h5.root,
                                                               name='Row_residuals_DUT_%d' % actual_dut,
                                                               atom=tb.Float32Atom(),
                                                               shape=(0, plot_n_bins[dut_index][1]),
                                                               title='Row residuals',
                                                               filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                y_residuals_earray.attrs.edges = y_edges
                if use_fit_limits:
                    fit_limit_x_local, fit_limit_y_local = fit_limits[actual_dut][0], fit_limits[actual_dut][1]
                else:
                    fit_limit_x_local = None
                    fit_limit_y_local = None

                correlate_n_tracks = min(correlate_n_tracks, node.nrows)
                widgets = ['', progressbar.Percentage(), ' ',
                           progressbar.Bar(marker='*', left='|', right='|'),
                           ' ', progressbar.AdaptiveETA()]
                progress_bar = progressbar.ProgressBar(widgets=widgets,
                                                       maxval=1.0,
                                                       term_width=80)
                progress_bar.start()

                correlate_n_tracks_total = 0

                initialize = True  # initialize the histograms
                ref_index_last = 0
                correlate_start_index = 0
                correlate_index_last = 0
                for ref_chunk, curr_ref_index in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    selection = np.logical_and(~np.isnan(ref_chunk['x_dut_%d' % actual_dut]), ~np.isnan(ref_chunk['track_chi2']))
                    ref_chunk = ref_chunk[selection]  # Take only tracks where actual dut has a hit, otherwise residual wrong

                    # Coordinates in global coordinate system (x, y, z)
                    ref_hit_x_local, ref_hit_y_local, ref_hit_z_local = ref_chunk['x_dut_%d' % actual_dut], ref_chunk['y_dut_%d' % actual_dut], ref_chunk['z_dut_%d' % actual_dut]
#                     ref_hit_local = np.column_stack([ref_hit_x_local, ref_hit_y_local])
                    ref_intersection_x, ref_intersection_y, ref_intersection_z = ref_chunk['offset_0'], ref_chunk['offset_1'], ref_chunk['offset_2']
#                     ref_offsets = np.column_stack([ref_chunk['offset_0'], ref_chunk['offset_1'], ref_chunk['offset_2']])
#                     ref_slopes = np.column_stack([ref_chunk['slope_0'], ref_chunk['slope_1'], ref_chunk['slope_2']])
                    ref_event_numbers = ref_chunk['event_number']

                    if use_prealignment:
                        ref_hit_x, ref_hit_y, ref_hit_z = geometry_utils.apply_alignment(ref_hit_x_local, ref_hit_y_local, ref_hit_z_local,
                                                                                         dut_index=actual_dut,
                                                                                         prealignment=prealignment,
                                                                                         inverse=False)

                        ref_intersection_x_local, ref_intersection_y_local, ref_intersection_z_local = geometry_utils.apply_alignment(ref_intersection_x, ref_intersection_y, ref_intersection_z,
                                                                                                                                      dut_index=actual_dut,
                                                                                                                                      prealignment=prealignment,
                                                                                                                                      inverse=True)
                    else:
                        ref_hit_x, ref_hit_y, ref_hit_z = geometry_utils.apply_alignment(ref_hit_x_local, ref_hit_y_local, ref_hit_z_local,
                                                                                         dut_index=actual_dut,
                                                                                         alignment=alignment,
                                                                                         inverse=False)

#                         dut_position = np.array([alignment[actual_dut]['translation_x'], alignment[actual_dut]['translation_y'], alignment[actual_dut]['translation_z']])
#                         rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[actual_dut]['alpha'],
#                                                                          beta=alignment[actual_dut]['beta'],
#                                                                          gamma=alignment[actual_dut]['gamma'])
#                         basis_global = rotation_matrix.T.dot(np.eye(3))
#                         dut_plane_normal = basis_global[2]
#                         if dut_plane_normal[2] < 0:
#                             dut_plane_normal = -dut_plane_normal
#
#                         # Set the offset to the track intersection with the tilted plane
#                         ref_dut_intersection = geometry_utils.get_line_intersections_with_plane(line_origins=ref_offsets,
#                                                                                                 line_directions=ref_slopes,
#                                                                                                 position_plane=dut_position,
#                                                                                                 normal_plane=dut_plane_normal)
#                         ref_intersection_x, ref_intersection_y, ref_intersection_z = ref_dut_intersection[:, 0], ref_dut_intersection[:, 1], ref_dut_intersection[:, 2]

                        ref_intersection_x_local, ref_intersection_y_local, ref_intersection_z_local = geometry_utils.apply_alignment(ref_intersection_x, ref_intersection_y, ref_intersection_z,
                                                                                                                                      dut_index=actual_dut,
                                                                                                                                      alignment=alignment,
                                                                                                                                      inverse=True)

                    ref_intersection_local = np.column_stack([ref_intersection_x_local, ref_intersection_y_local])

                    if not np.allclose(ref_hit_z_local[np.isfinite(ref_hit_z_local)], 0.0) or not np.allclose(ref_intersection_z_local[np.isfinite(ref_intersection_z_local)], 0.0):
                        raise RuntimeError("Transformation into local coordinate system gives z != 0")

                    ref_limit_x_local_sel = np.ones_like(ref_hit_x_local, dtype=np.bool)
                    if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local[0]):
                        ref_limit_x_local_sel &= ref_hit_x_local >= fit_limit_x_local[0]
                    if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local[1]):
                        ref_limit_x_local_sel &= ref_hit_x_local <= fit_limit_x_local[1]

                    ref_limit_y_local_sel = np.ones_like(ref_hit_x_local, dtype=np.bool)
                    if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local[0]):
                        ref_limit_y_local_sel &= ref_hit_y_local >= fit_limit_y_local[0]
                    if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local[1]):
                        ref_limit_y_local_sel &= ref_hit_y_local <= fit_limit_y_local[1]

                    ref_limit_xy_local_sel = np.logical_and(ref_limit_x_local_sel, ref_limit_y_local_sel)

                    ref_hit_x_local_limit_xy = ref_hit_x_local[ref_limit_xy_local_sel]
                    ref_hit_y_local_limit_xy = ref_hit_y_local[ref_limit_xy_local_sel]
                    ref_intersection_x_local_limit_xy = ref_intersection_x_local[ref_limit_xy_local_sel]
                    ref_intersection_y_local_limit_xy = ref_intersection_y_local[ref_limit_xy_local_sel]
                    ref_event_numbers = ref_event_numbers[ref_limit_xy_local_sel]

                    ref_difference_x_local_limit_xy = ref_hit_x_local_limit_xy - ref_intersection_x_local_limit_xy
                    ref_difference_y_local_limit_xy = ref_hit_y_local_limit_xy - ref_intersection_y_local_limit_xy
#                         distance = np.sqrt(np.einsum('ij,ij->i', intersection_local - hit_local, intersection_local - hit_local))

                    for tracks_chunk, curr_correlate_index in analysis_utils.data_aligned_at_events(node, start_index=correlate_start_index, chunk_size=chunk_size):
                        selection = np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut]), ~np.isnan(tracks_chunk['track_chi2']))
                        tracks_chunk = tracks_chunk[selection]  # Take only tracks where actual dut has a hit, otherwise residual wrong

                        # Coordinates in global coordinate system (x, y, z)
                        hit_x_local, hit_y_local, hit_z_local = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
#                         hit_local = np.column_stack([hit_x_local, hit_y_local])
                        intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']
#                         offsets = np.column_stack([tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']])
#                         slopes = np.column_stack([tracks_chunk['slope_0'], tracks_chunk['slope_1'], tracks_chunk['slope_2']])
                        event_numbers = tracks_chunk['event_number']

                        if use_prealignment:
                            hit_x, hit_y, hit_z = geometry_utils.apply_alignment(hit_x_local, hit_y_local, hit_z_local,
                                                                                 dut_index=actual_dut,
                                                                                 prealignment=prealignment,
                                                                                 inverse=False)

                            intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                              dut_index=actual_dut,
                                                                                                                              prealignment=prealignment,
                                                                                                                              inverse=True)
                        else:
                            hit_x, hit_y, hit_z = geometry_utils.apply_alignment(hit_x_local, hit_y_local, hit_z_local,
                                                                                 dut_index=actual_dut,
                                                                                 alignment=alignment,
                                                                                 inverse=False)

#                             dut_position = np.array([alignment[actual_dut]['translation_x'], alignment[actual_dut]['translation_y'], alignment[actual_dut]['translation_z']])
#                             rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[actual_dut]['alpha'],
#                                                                              beta=alignment[actual_dut]['beta'],
#                                                                              gamma=alignment[actual_dut]['gamma'])
#                             basis_global = rotation_matrix.T.dot(np.eye(3))
#                             dut_plane_normal = basis_global[2]
#                             if dut_plane_normal[2] < 0:
#                                 dut_plane_normal = -dut_plane_normal
#
#                             # Set the offset to the track intersection with the tilted plane
#                             dut_intersection = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
#                                                                                                 line_directions=slopes,
#                                                                                                 position_plane=dut_position,
#                                                                                                 normal_plane=dut_plane_normal)
#                             intersection_x, intersection_y, intersection_z = dut_intersection[:, 0], dut_intersection[:, 1], dut_intersection[:, 2]

                            intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                              dut_index=actual_dut,
                                                                                                                              alignment=alignment,
                                                                                                                              inverse=True)

                        intersection_local = np.column_stack([intersection_x_local, intersection_y_local])

                        if not np.allclose(hit_z_local[np.isfinite(hit_z_local)], 0.0) or not np.allclose(intersection_z_local[np.isfinite(intersection_z_local)], 0.0):
                            raise RuntimeError("Transformation into local coordinate system gives z != 0")

                        limit_x_local_sel = np.ones_like(hit_x_local, dtype=np.bool)
                        if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local[0]):
                            limit_x_local_sel &= hit_x_local >= fit_limit_x_local[0]
                        if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local[1]):
                            limit_x_local_sel &= hit_x_local <= fit_limit_x_local[1]

                        limit_y_local_sel = np.ones_like(hit_x_local, dtype=np.bool)
                        if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local[0]):
                            limit_y_local_sel &= hit_y_local >= fit_limit_y_local[0]
                        if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local[1]):
                            limit_y_local_sel &= hit_y_local <= fit_limit_y_local[1]

                        limit_xy_local_sel = np.logical_and(limit_x_local_sel, limit_y_local_sel)

                        hit_x_local_limit_xy = hit_x_local[limit_xy_local_sel]
                        hit_y_local_limit_xy = hit_y_local[limit_xy_local_sel]
                        intersection_x_local_limit_xy = intersection_x_local[limit_xy_local_sel]
                        intersection_y_local_limit_xy = intersection_y_local[limit_xy_local_sel]
                        event_numbers = event_numbers[limit_xy_local_sel]

                        difference_x_local_limit_xy = hit_x_local_limit_xy - intersection_x_local_limit_xy
                        difference_y_local_limit_xy = hit_y_local_limit_xy - intersection_y_local_limit_xy
#                         distance = np.sqrt(np.einsum('ij,ij->i', intersection_local - hit_local, intersection_local - hit_local))

                        iterate_n_ref_tracks = min(ref_intersection_x_local_limit_xy.shape[0], correlate_n_tracks - correlate_n_tracks_total)
                        for ref_index in range(iterate_n_ref_tracks):
                            # Histogram residuals in different ways
                            x_res = [None] * plot_n_bins[dut_index][0]
                            y_res = [None] * plot_n_bins[dut_index][1]
                            diff_x = ref_intersection_x_local_limit_xy[ref_index] - intersection_x_local_limit_xy
                            diff_y = ref_intersection_y_local_limit_xy[ref_index] - intersection_y_local_limit_xy
                            ref_event_number = ref_event_numbers[ref_index]
                            # in x direction
#                             for index, actual_range in enumerate(np.column_stack([x_edges[:-1], x_edges[1:]])):
#                                 sel_from = actual_range[0]
#                                 sel_to = actual_range[1]
#                                 intersection_select = ((np.absolute(diff_x) < sel_to) & (np.absolute(diff_x) >= sel_from) & (np.absolute(diff_y) < 5.0))
#                                 x_res[index] = np.array(difference_x_local_limit_xy[intersection_select])
                            def func(x):
                                return x
                            select_valid = (np.absolute(diff_x) <= stop[0]) & (np.absolute(diff_y) < 5.0)
                            # do not correlate events with event numbers smaller or equal than current event number
                            next_event_index = np.searchsorted(event_numbers, ref_event_number + 1, side="left")
                            select_valid[:next_event_index] = 0
                            # fast binned statistic
                            x_res = analysis_utils.binned_statistic(np.absolute(diff_x[select_valid]), difference_x_local_limit_xy[select_valid], func=func, nbins=plot_n_bins[dut_index][0], range=(0, stop[0]))
                            x_res_arr = np.full((np.max([arr.shape[0] for arr in x_res]), len(x_res)), fill_value=np.nan)
                            for index, arr in enumerate(x_res):
#                                 assert np.array_equal(x_res[index], x_res_alternative[index])
                                x_res_arr[:arr.shape[0], index] = arr
                            x_residuals_earray.append(x_res_arr)
                            x_residuals_earray.flush()
                            ref_x_res_arr = np.full(x_res_arr.shape[0], fill_value=ref_difference_x_local_limit_xy[ref_index])
                            ref_x_residuals_earray.append(ref_x_res_arr)
                            ref_x_residuals_earray.flush()
                            # in y direction
#                             for index, actual_range in enumerate(np.column_stack([y_edges[:-1], y_edges[1:]])):
#                                 sel_from = actual_range[0]
#                                 sel_to = actual_range[1]
#                                 intersection_select = ((np.absolute(diff_y) < sel_to) & (np.absolute(diff_y) >= sel_from) & (np.absolute(diff_x) < 5.0))
#                                 y_res[index] = difference_y_local_limit_xy[intersection_select]
                            select_valid = (np.absolute(diff_y) <= stop[1]) & (np.absolute(diff_x) < 5.0)
                            # do not correlate events with event numbers smaller or equal than current event number
                            select_valid[:next_event_index] = 0
                            # fast binned statistic
                            y_res = analysis_utils.binned_statistic(np.absolute(diff_y[select_valid]), difference_y_local_limit_xy[select_valid], func=func, nbins=plot_n_bins[dut_index][1], range=(0, stop[1]))
                            y_res_arr = np.full((np.max([arr.shape[0] for arr in y_res]), len(y_res)), fill_value=np.nan)
                            for index, arr in enumerate(y_res):
                                y_res_arr[:arr.shape[0], index] = arr
                            y_residuals_earray.append(y_res_arr)
                            y_residuals_earray.flush()
                            ref_y_res_arr = np.full(y_res_arr.shape[0], fill_value=ref_difference_y_local_limit_xy[ref_index])
                            ref_y_residuals_earray.append(ref_y_res_arr)
                            ref_y_residuals_earray.flush()
                            progress_bar.update(min(1.0, (((ref_index + 1) / iterate_n_ref_tracks * (curr_correlate_index - correlate_index_last) / node.nrows) * iterate_n_ref_tracks / correlate_n_tracks) + ((correlate_index_last / node.nrows) * iterate_n_ref_tracks / correlate_n_tracks)))
                        correlate_index_last = curr_correlate_index
                    correlate_n_tracks_total += iterate_n_ref_tracks

                    if correlate_n_tracks_total >= correlate_n_tracks:
                        break
                    correlate_start_index = curr_ref_index
                    ref_index_last = curr_ref_index

                progress_bar.finish()

    if plot:
        plot_utils.plot_residual_correlation(input_residual_correlation_file=output_residual_correlation_file, select_duts=select_duts, pixel_size=pixel_size, output_pdf_file=None, dut_names=dut_names, chunk_size=chunk_size)

