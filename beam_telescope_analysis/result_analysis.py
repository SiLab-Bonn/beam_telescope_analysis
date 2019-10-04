''' All functions creating results (e.g. efficiency, residuals, track density) from fitted tracks are listed here.'''
from __future__ import division

import logging
from collections import Iterable
import os.path
import math

import tables as tb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from scipy import stats
from scipy.signal import hilbert
from scipy.spatial import cKDTree

from tqdm import tqdm

from beam_telescope_analysis.telescope.telescope import Telescope
from beam_telescope_analysis.tools import plot_utils
from beam_telescope_analysis.tools import geometry_utils
from beam_telescope_analysis.tools import analysis_utils
from beam_telescope_analysis.tools.storage_utils import save_arguments


@save_arguments
def calculate_residuals(telescope_configuration, input_tracks_file, select_duts, output_residuals_file=None, nbins_per_pixel=None, npixels_per_bin=None, use_limits=True, plot=True, chunk_size=1000000):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_tracks_file : string
        Filename of the input tracks file.
    select_duts : list
        Selecting DUT indices that will be processed.
    output_residuals_file : string
        Filename of the output residuals file. If None, the filename will be derived from the input hits file.
    nbins_per_pixel : int
        Number of bins per pixel along the residual axis. Number is a positive integer or None to automatically set the binning.
    npixels_per_bin : int
        Number of pixels per bin along the position axis. Number is a positive integer or None to automatically set the binning.
    use_limits : bool
        If True, use column and row limits from pre-alignment for selecting the data.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_residuals_file : string
        Filename of the output residuals file.
    '''
    telescope = Telescope(telescope_configuration)
    logging.info('=== Calculating residuals for %d DUTs ===' % len(select_duts))

    if output_residuals_file is None:
        output_residuals_file = os.path.splitext(input_tracks_file)[0] + '_residuals.h5'

    if plot is True:
        output_pdf = PdfPages(os.path.splitext(output_residuals_file)[0] + '.pdf', keep_empty=False)
    else:
        output_pdf = None

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_residuals_file, mode='w') as out_file_h5:
            for actual_dut_index in select_duts:
                actual_dut = telescope[actual_dut_index]
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)
                logging.info('== Calculating residuals for %s ==', actual_dut.name)

                if use_limits:
                    limit_x_local = actual_dut.x_limit  # (lower limit, upper limit)
                    limit_y_local = actual_dut.y_limit  # (lower limit, upper limit)
                else:
                    limit_x_local = None
                    limit_y_local = None

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    # Select data with hits and tracks
                    selection = np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut_index]), ~np.isnan(tracks_chunk['track_chi2']))
                    tracks_chunk = tracks_chunk[selection]  # Take only tracks where actual dut has a hit, otherwise residual wrong

                    hit_x_local, hit_y_local = tracks_chunk['x_dut_%d' % actual_dut_index], tracks_chunk['y_dut_%d' % actual_dut_index]
                    hit_local = np.column_stack([hit_x_local, hit_y_local])

                    intersection_x_local, intersection_y_local = tracks_chunk['offset_x'], tracks_chunk['offset_y']
                    intersection_local = np.column_stack([intersection_x_local, intersection_y_local])
                    difference_local = hit_local - intersection_local

                    limit_x_local_sel = np.ones_like(hit_x_local, dtype=np.bool)
                    if limit_x_local is not None and np.isfinite(limit_x_local[0]):
                        limit_x_local_sel &= hit_x_local >= limit_x_local[0]
                    if limit_x_local is not None and np.isfinite(limit_x_local[1]):
                        limit_x_local_sel &= hit_x_local <= limit_x_local[1]

                    limit_y_local_sel = np.ones_like(hit_x_local, dtype=np.bool)
                    if limit_y_local is not None and np.isfinite(limit_y_local[0]):
                        limit_y_local_sel &= hit_y_local >= limit_y_local[0]
                    if limit_y_local is not None and np.isfinite(limit_y_local[1]):
                        limit_y_local_sel &= hit_y_local <= limit_y_local[1]

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
                    distance_local = np.sqrt(np.einsum('ij,ij->i', difference_local, difference_local))

                    # Histogram residuals in different ways
                    if initialize:  # Only true for the first iteration, calculate the binning for the histograms
                        initialize = False
                        plot_n_pixels = 6.0

                        # detect peaks and calculate width to estimate the size of the histograms
                        if nbins_per_pixel is not None:
                            min_difference, max_difference = np.min(difference_local_limit_xy[:, 0]), np.max(difference_local_limit_xy[:, 0])
                            nbins = np.arange(min_difference - (actual_dut.column_size / nbins_per_pixel), max_difference + 2 * (actual_dut.column_size / nbins_per_pixel), actual_dut.column_size / nbins_per_pixel)
                        else:
                            nbins = "sturges"
                        hist, edges = np.histogram(difference_local_limit_xy[:, 0], bins=nbins)
                        edge_center = (edges[1:] + edges[:-1]) / 2.0
                        try:
                            _, center_x_local, fwhm_x_local, _ = analysis_utils.peak_detect(edge_center, hist)
                        except RuntimeError:
                            # do some simple FWHM with numpy array
                            try:
                                _, center_x_local, fwhm_x_local, _ = analysis_utils.simple_peak_detect(edge_center, hist)
                            except RuntimeError:
                                center_x_local, fwhm_x_local = 0.0, actual_dut.column_size * plot_n_pixels

                        if nbins_per_pixel is not None:
                            min_difference, max_difference = np.min(difference_local_limit_xy[:, 1]), np.max(difference_local_limit_xy[:, 1])
                            nbins = np.arange(min_difference - (actual_dut.row_size / nbins_per_pixel), max_difference + 2 * (actual_dut.row_size / nbins_per_pixel), actual_dut.row_size / nbins_per_pixel)
                        else:
                            nbins = "sturges"
                        hist, edges = np.histogram(difference_local_limit_xy[:, 1], bins=nbins)
                        edge_center = (edges[1:] + edges[:-1]) / 2.0
                        try:
                            _, center_y_local, fwhm_y_local, _ = analysis_utils.peak_detect(edge_center, hist)
                        except RuntimeError:
                            # do some simple FWHM with numpy array
                            try:
                                _, center_y_local, fwhm_y_local, _ = analysis_utils.simple_peak_detect(edge_center, hist)
                            except RuntimeError:
                                center_y_local, fwhm_y_local = 0.0, actual_dut.row_size * plot_n_pixels

                        # calculate the binning of the histograms, the minimum size is given by plot_n_pixels, otherwise FWHM is taken into account
                        if nbins_per_pixel is not None:
                            width = max(plot_n_pixels * actual_dut.column_size, actual_dut.column_size * np.ceil(plot_n_pixels * fwhm_x_local / actual_dut.column_size))
                            if np.mod(width / actual_dut.column_size, 2) != 0:
                                width += actual_dut.column_size
                            nbins = int(nbins_per_pixel * width / actual_dut.column_size)
                            local_x_range = (center_x_local - 0.5 * width, center_x_local + 0.5 * width)
                        else:
                            # if fwhm_x_local < 0.01:

                            # else:
                            #     nbins = "sturges"
                            nbins = 200
                            width = actual_dut.column_size * np.ceil(plot_n_pixels * fwhm_x_local / actual_dut.column_size)
                            local_x_range = (center_x_local - width, center_x_local + width)
                        local_x_residuals_hist, local_x_residuals_hist_edges = np.histogram(difference_local_limit_xy[:, 0], range=local_x_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_x_local), np.max(intersection_x_local)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * actual_dut.column_size, npixels_per_bin * actual_dut.column_size)
                        else:
                            nbins = "sturges"
                        _, x_pos_hist_edges = np.histogram(intersection_x_local, bins=nbins)

                        if nbins_per_pixel is not None:
                            width = max(plot_n_pixels * actual_dut.row_size, actual_dut.row_size * np.ceil(plot_n_pixels * fwhm_y_local / actual_dut.row_size))
                            if np.mod(width / actual_dut.row_size, 2) != 0:
                                width += actual_dut.row_size
                            nbins = int(nbins_per_pixel * width / actual_dut.row_size)
                            local_y_range = (center_y_local - 0.5 * width, center_y_local + 0.5 * width)
                        else:
                            # if fwhm_y_local < 0.01:
                            #     nbins = 200
                            # else:
                            #     nbins = "sturges"
                            nbins = 200
                            width = actual_dut.row_size * np.ceil(plot_n_pixels * fwhm_y_local / actual_dut.row_size)
                            local_y_range = (center_y_local - width, center_y_local + width)
                        local_y_residuals_hist, local_y_residuals_hist_edges = np.histogram(difference_local_limit_xy[:, 1], range=local_y_range, bins=nbins)

                        if npixels_per_bin is not None:
                            min_intersection, max_intersection = np.min(intersection_y_local), np.max(intersection_y_local)
                            nbins = np.arange(min_intersection, max_intersection + npixels_per_bin * actual_dut.row_size, npixels_per_bin * actual_dut.row_size)
                        else:
                            nbins = "sturges"
                        _, y_pos_hist_edges = np.histogram(intersection_y_local, bins=nbins)

                        dut_x_size = actual_dut.x_extent()
                        dut_y_size = actual_dut.y_extent()
                        hist_2d_residuals_x_edges = np.linspace(dut_x_size[0], dut_x_size[1], actual_dut.n_columns + 1, endpoint=True)
                        hist_2d_residuals_y_edges = np.linspace(dut_y_size[0], dut_y_size[1], actual_dut.n_rows + 1, endpoint=True)
                        hist_2d_edges = [hist_2d_residuals_x_edges, hist_2d_residuals_y_edges]

                        # local X residual against X position
                        local_x_residuals_x_pos_hist, _, _ = np.histogram2d(
                            intersection_x_local_limit_y,
                            difference_local_limit_y[:, 0],
                            bins=(x_pos_hist_edges, local_x_residuals_hist_edges))
                        stat_local_x_residuals_x_pos_hist, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 0], statistic='mean', bins=x_pos_hist_edges)
                        stat_local_x_residuals_x_pos_hist = np.nan_to_num(stat_local_x_residuals_x_pos_hist)
                        count_local_x_residuals_x_pos_hist, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 0], statistic='count', bins=x_pos_hist_edges)

                        # local Y residual against Y position
                        local_y_residuals_y_pos_hist, _, _ = np.histogram2d(
                            intersection_y_local_limit_x,
                            difference_local_limit_x[:, 1],
                            bins=(y_pos_hist_edges, local_y_residuals_hist_edges))
                        stat_local_y_residuals_y_pos_hist, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 1], statistic='mean', bins=y_pos_hist_edges)
                        stat_local_y_residuals_y_pos_hist = np.nan_to_num(stat_local_y_residuals_y_pos_hist)
                        count_local_y_residuals_y_pos_hist, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 1], statistic='count', bins=y_pos_hist_edges)

                        # local Y residual against X position
                        local_y_residuals_x_pos_hist, _, _ = np.histogram2d(
                            intersection_x_local_limit_y,
                            difference_local_limit_y[:, 1],
                            bins=(x_pos_hist_edges, local_y_residuals_hist_edges))
                        stat_local_y_residuals_x_pos_hist, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 1], statistic='mean', bins=x_pos_hist_edges)
                        stat_local_y_residuals_x_pos_hist = np.nan_to_num(stat_local_y_residuals_x_pos_hist)
                        count_local_y_residuals_x_pos_hist, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 1], statistic='count', bins=x_pos_hist_edges)

                        # local X residual against Y position
                        local_x_residuals_y_pos_hist, _, _ = np.histogram2d(
                            intersection_y_local_limit_x,
                            difference_local_limit_x[:, 0],
                            bins=(y_pos_hist_edges, local_x_residuals_hist_edges))
                        stat_local_x_residuals_y_pos_hist, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 0], statistic='mean', bins=y_pos_hist_edges)
                        stat_local_x_residuals_y_pos_hist = np.nan_to_num(stat_local_x_residuals_y_pos_hist)
                        count_local_x_residuals_y_pos_hist, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 0], statistic='count', bins=y_pos_hist_edges)

                        # 2D residuals
                        stat_2d_residuals_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=distance_local, statistic='mean', bins=hist_2d_edges)
                        stat_2d_residuals_hist = np.nan_to_num(stat_2d_residuals_hist)
                        count_2d_residuals_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=distance_local, statistic='count', bins=hist_2d_edges)

                        # 2D hits
                        count_2d_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=None, statistic='count', bins=hist_2d_edges)

                    else:  # adding data to existing histograms
                        local_x_residuals_hist += np.histogram(difference_local_limit_xy[:, 0], bins=local_x_residuals_hist_edges)[0]
                        local_y_residuals_hist += np.histogram(difference_local_limit_xy[:, 1], bins=local_y_residuals_hist_edges)[0]

                        # local X residual against X position
                        local_x_residuals_x_pos_hist += np.histogram2d(
                            intersection_x_local_limit_y,
                            difference_local_limit_y[:, 0],
                            bins=(x_pos_hist_edges, local_x_residuals_hist_edges))[0]
                        stat_local_x_residuals_x_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 0], statistic='mean', bins=x_pos_hist_edges)
                        stat_local_x_residuals_x_pos_hist_tmp = np.nan_to_num(stat_local_x_residuals_x_pos_hist_tmp)
                        count_local_x_residuals_x_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 0], statistic='count', bins=x_pos_hist_edges)
                        stat_local_x_residuals_x_pos_hist, count_local_x_residuals_x_pos_hist = np.ma.average(a=np.stack([stat_local_x_residuals_x_pos_hist, stat_local_x_residuals_x_pos_hist_tmp]), axis=0, weights=np.stack([count_local_x_residuals_x_pos_hist, count_local_x_residuals_x_pos_hist_tmp]), returned=True)

                        # local Y residual against Y position
                        local_y_residuals_y_pos_hist += np.histogram2d(
                            intersection_y_local_limit_x,
                            difference_local_limit_x[:, 1],
                            bins=(y_pos_hist_edges, local_y_residuals_hist_edges))[0]
                        stat_local_y_residuals_y_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 1], statistic='mean', bins=y_pos_hist_edges)
                        stat_local_y_residuals_y_pos_hist_tmp = np.nan_to_num(stat_local_y_residuals_y_pos_hist_tmp)
                        count_local_y_residuals_y_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 1], statistic='count', bins=y_pos_hist_edges)
                        stat_local_y_residuals_y_pos_hist, count_local_y_residuals_y_pos_hist = np.ma.average(a=np.stack([stat_local_y_residuals_y_pos_hist, stat_local_y_residuals_y_pos_hist_tmp]), axis=0, weights=np.stack([count_local_y_residuals_y_pos_hist, count_local_y_residuals_y_pos_hist_tmp]), returned=True)

                        # local Y residual against X position
                        local_y_residuals_x_pos_hist += np.histogram2d(
                            intersection_x_local_limit_y,
                            difference_local_limit_y[:, 1],
                            bins=(x_pos_hist_edges, local_y_residuals_hist_edges))[0]
                        stat_local_y_residuals_x_pos_tmp, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 1], statistic='mean', bins=x_pos_hist_edges)
                        stat_local_y_residuals_x_pos_tmp = np.nan_to_num(stat_local_y_residuals_x_pos_tmp)
                        count_local_y_residuals_x_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_x_local_limit_y, values=difference_local_limit_y[:, 1], statistic='count', bins=x_pos_hist_edges)
                        stat_local_y_residuals_x_pos_hist, count_local_y_residuals_x_pos_hist = np.ma.average(a=np.stack([stat_local_y_residuals_x_pos_hist, stat_local_y_residuals_x_pos_tmp]), axis=0, weights=np.stack([count_local_y_residuals_x_pos_hist, count_local_y_residuals_x_pos_hist_tmp]), returned=True)

                        # local X residual against Y position
                        local_x_residuals_y_pos_hist += np.histogram2d(
                            intersection_y_local_limit_x,
                            difference_local_limit_x[:, 0],
                            bins=(y_pos_hist_edges, local_x_residuals_hist_edges))[0]
                        stat_local_x_residuals_y_pos_tmp, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 0], statistic='mean', bins=y_pos_hist_edges)
                        stat_local_x_residuals_y_pos_tmp = np.nan_to_num(stat_local_x_residuals_y_pos_tmp)
                        count_local_x_residuals_y_pos_hist_tmp, _, _ = stats.binned_statistic(x=intersection_y_local_limit_x, values=difference_local_limit_x[:, 0], statistic='count', bins=y_pos_hist_edges)
                        stat_local_x_residuals_y_pos_hist, count_local_x_residuals_y_pos_hist = np.ma.average(a=np.stack([stat_local_x_residuals_y_pos_hist, stat_local_x_residuals_y_pos_tmp]), axis=0, weights=np.stack([count_local_x_residuals_y_pos_hist, count_local_x_residuals_y_pos_hist_tmp]), returned=True)

                        # 2D residuals
                        stat_2d_residuals_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=distance_local, statistic='mean', bins=hist_2d_edges)
                        stat_2d_residuals_hist_tmp = np.nan_to_num(stat_2d_residuals_hist_tmp)
                        count_2d_residuals_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=distance_local, statistic='count', bins=hist_2d_edges)
                        stat_2d_residuals_hist, count_2d_residuals_hist = np.ma.average(a=np.stack([stat_2d_residuals_hist, stat_2d_residuals_hist_tmp]), axis=0, weights=np.stack([count_2d_residuals_hist, count_2d_residuals_hist_tmp]), returned=True)

                        # 2D hits
                        count_2d_hist += stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=None, statistic='count', bins=hist_2d_edges)[0]

                if initialize is True:
                    raise RuntimeError("No tracks were processed.")
                logging.debug('Storing residual histograms...')

                stat_local_x_residuals_x_pos_hist[count_local_x_residuals_x_pos_hist == 0] = np.nan
                stat_local_y_residuals_y_pos_hist[count_local_y_residuals_y_pos_hist == 0] = np.nan
                stat_local_y_residuals_x_pos_hist[count_local_y_residuals_x_pos_hist == 0] = np.nan
                stat_local_x_residuals_y_pos_hist[count_local_x_residuals_y_pos_hist == 0] = np.nan

                # Plotting local residuals
                fit_local_x_res, cov_local_x_res = analysis_utils.fit_residuals(
                    hist=local_x_residuals_hist,
                    edges=local_x_residuals_hist_edges,
                )
                plot_utils.plot_residuals(
                    histogram=local_x_residuals_hist,
                    edges=local_x_residuals_hist_edges,
                    fit=fit_local_x_res,
                    cov=cov_local_x_res,
                    xlabel='X residual [$\mathrm{\mu}$m]',
                    title='Local X residuals for %s' % actual_dut.name,
                    output_pdf=output_pdf)
                out_local_x_res = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='local_x_residuals_DUT%d' % (actual_dut_index),
                    title='Local residual distribution in X direction for %s' % (actual_dut.name),
                    atom=tb.Atom.from_dtype(local_x_residuals_hist.dtype),
                    shape=local_x_residuals_hist.shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_local_x_res.attrs.xedges = local_x_residuals_hist_edges
                out_local_x_res.attrs.fit_coeff = fit_local_x_res
                out_local_x_res.attrs.fit_cov = cov_local_x_res
                out_local_x_res[:] = local_x_residuals_hist

                fit_local_y_res, cov_local_y_res = analysis_utils.fit_residuals(
                    hist=local_y_residuals_hist,
                    edges=local_y_residuals_hist_edges,
                )
                plot_utils.plot_residuals(
                    histogram=local_y_residuals_hist,
                    edges=local_y_residuals_hist_edges,
                    fit=fit_local_y_res,
                    cov=cov_local_y_res,
                    xlabel='Y residual [$\mathrm{\mu}$m]',
                    title='Local Y residuals for %s' % actual_dut.name,
                    output_pdf=output_pdf)
                out_local_y_res = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='local_y_residuals_DUT%d' % (actual_dut_index),
                    title='Local residual distribution in Y direction for %s' % (actual_dut.name),
                    atom=tb.Atom.from_dtype(local_y_residuals_hist.dtype),
                    shape=local_y_residuals_hist.shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_local_y_res.attrs.yedges = local_y_residuals_hist_edges
                out_local_y_res.attrs.fit_coeff = fit_local_y_res
                out_local_y_res.attrs.fit_cov = cov_local_y_res
                out_local_y_res[:] = local_y_residuals_hist

                fit_local_x_residuals_x_pos, cov_local_x_residuals_x_pos, select, stat_local_x_residuals_x_pos_hist = analysis_utils.fit_residuals_vs_position(
                    hist=local_x_residuals_x_pos_hist,
                    xedges=x_pos_hist_edges,
                    yedges=local_x_residuals_hist_edges,
                    mean=stat_local_x_residuals_x_pos_hist,
                    count=count_local_x_residuals_x_pos_hist,
                    limit=limit_x_local
                )
                plot_utils.plot_residuals_vs_position(
                    hist=local_x_residuals_x_pos_hist,
                    xedges=x_pos_hist_edges,
                    yedges=local_x_residuals_hist_edges,
                    xlabel='X [$\mathrm{\mu}$m]',
                    ylabel='X residual [$\mathrm{\mu}$m]',
                    title='Local X residuals vs. X positions for %s' % actual_dut.name,
                    residuals_mean=stat_local_x_residuals_x_pos_hist,
                    select=select,
                    fit=fit_local_x_residuals_x_pos,
                    cov=cov_local_x_residuals_x_pos,
                    limit=limit_x_local,
                    output_pdf=output_pdf)
                out_local_x_residuals_x_pos = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='local_x_residuals_x_pos_DUT%d' % (actual_dut_index),
                    title='Local residual distribution in X direction as a function of the X position for %s' % (actual_dut.name),
                    atom=tb.Atom.from_dtype(local_x_residuals_x_pos_hist.dtype),
                    shape=local_x_residuals_x_pos_hist.shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_local_x_residuals_x_pos.attrs.xedges = x_pos_hist_edges
                out_local_x_residuals_x_pos.attrs.yedges = local_x_residuals_hist_edges
                out_local_x_residuals_x_pos.attrs.fit_coeff = fit_local_x_residuals_x_pos
                out_local_x_residuals_x_pos.attrs.fit_cov = cov_local_x_residuals_x_pos
                out_local_x_residuals_x_pos[:] = local_x_residuals_x_pos_hist

                fit_local_y_residuals_y_pos, cov_local_y_residuals_y_pos, select, stat_local_y_residuals_y_pos_hist = analysis_utils.fit_residuals_vs_position(
                    hist=local_y_residuals_y_pos_hist,
                    xedges=y_pos_hist_edges,
                    yedges=local_y_residuals_hist_edges,
                    mean=stat_local_y_residuals_y_pos_hist,
                    count=count_local_y_residuals_y_pos_hist,
                    limit=limit_y_local
                )
                plot_utils.plot_residuals_vs_position(
                    hist=local_y_residuals_y_pos_hist,
                    xedges=y_pos_hist_edges,
                    yedges=local_y_residuals_hist_edges,
                    xlabel='Y [$\mathrm{\mu}$m]',
                    ylabel='Y residual [$\mathrm{\mu}$m]',
                    title='Local Y residuals vs. Y positions for %s' % actual_dut.name,
                    residuals_mean=stat_local_y_residuals_y_pos_hist,
                    select=select,
                    fit=fit_local_y_residuals_y_pos,
                    cov=cov_local_y_residuals_y_pos,
                    limit=limit_y_local,
                    output_pdf=output_pdf)
                out_local_y_residuals_y_pos = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='local_y_residuals_y_pos_DUT%d' % (actual_dut_index),
                    title='Local residual distribution in Y direction as a function of the Y position for %s' % (actual_dut.name),
                    atom=tb.Atom.from_dtype(local_y_residuals_y_pos_hist.dtype),
                    shape=local_y_residuals_y_pos_hist.shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_local_y_residuals_y_pos.attrs.xedges = y_pos_hist_edges
                out_local_y_residuals_y_pos.attrs.yedges = local_y_residuals_hist_edges
                out_local_y_residuals_y_pos.attrs.fit_coeff = fit_local_y_residuals_y_pos
                out_local_y_residuals_y_pos.attrs.fit_cov = cov_local_y_residuals_y_pos
                out_local_y_residuals_y_pos[:] = local_y_residuals_y_pos_hist

                fit_local_y_residuals_x_pos, cov_local_y_residuals_x_pos, select, stat_local_y_residuals_x_pos_hist = analysis_utils.fit_residuals_vs_position(
                    hist=local_y_residuals_x_pos_hist,
                    xedges=x_pos_hist_edges,
                    yedges=local_y_residuals_hist_edges,
                    mean=stat_local_y_residuals_x_pos_hist,
                    count=count_local_y_residuals_x_pos_hist,
                    limit=limit_x_local
                )
                plot_utils.plot_residuals_vs_position(
                    hist=local_y_residuals_x_pos_hist,
                    xedges=x_pos_hist_edges,
                    yedges=local_y_residuals_hist_edges,
                    xlabel='X [$\mathrm{\mu}$m]',
                    ylabel='Y residual [$\mathrm{\mu}$m]',
                    title='Local Y residuals for vs. X positions %s' % actual_dut.name,
                    residuals_mean=stat_local_y_residuals_x_pos_hist,
                    select=select,
                    fit=fit_local_y_residuals_x_pos,
                    cov=cov_local_y_residuals_x_pos,
                    limit=limit_x_local,
                    output_pdf=output_pdf)
                out_local_x_residuals_y_pos_pos = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='local_y_residuals_x_pos_DUT%d' % (actual_dut_index),
                    title='Local residual distribution in X direction as a function of the Y position for %s' % (actual_dut.name),
                    atom=tb.Atom.from_dtype(local_y_residuals_x_pos_hist.dtype),
                    shape=local_y_residuals_x_pos_hist.shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_local_x_residuals_y_pos_pos.attrs.xedges = x_pos_hist_edges
                out_local_x_residuals_y_pos_pos.attrs.yedges = local_y_residuals_hist_edges
                out_local_x_residuals_y_pos_pos.attrs.fit_coeff = fit_local_y_residuals_x_pos
                out_local_x_residuals_y_pos_pos.attrs.fit_cov = cov_local_y_residuals_x_pos
                out_local_x_residuals_y_pos_pos[:] = local_y_residuals_x_pos_hist

                fit_local_x_residuals_y_pos, cov_local_x_residuals_y_pos, select, stat_local_x_residuals_y_pos_hist = analysis_utils.fit_residuals_vs_position(
                    hist=local_x_residuals_y_pos_hist,
                    xedges=y_pos_hist_edges,
                    yedges=local_x_residuals_hist_edges,
                    mean=stat_local_x_residuals_y_pos_hist,
                    count=count_local_x_residuals_y_pos_hist,
                    limit=limit_y_local
                )
                plot_utils.plot_residuals_vs_position(
                    hist=local_x_residuals_y_pos_hist,
                    xedges=y_pos_hist_edges,
                    yedges=local_x_residuals_hist_edges,
                    xlabel='Y [$\mathrm{\mu}$m]',
                    ylabel='X residual [$\mathrm{\mu}$m]',
                    title='Local X residuals vs. Y positions for %s' % actual_dut.name,
                    residuals_mean=stat_local_x_residuals_y_pos_hist,
                    select=select,
                    fit=fit_local_x_residuals_y_pos,
                    cov=cov_local_x_residuals_y_pos,
                    limit=limit_y_local,
                    output_pdf=output_pdf)
                out_local_x_residuals_y_pos = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='local_x_residuals_y_pos_DUT%d' % (actual_dut_index),
                    title='Local residual distribution in X direction as a function of the Y position for %s' % (actual_dut.name),
                    atom=tb.Atom.from_dtype(local_x_residuals_y_pos_hist.dtype),
                    shape=local_x_residuals_y_pos_hist.shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_local_x_residuals_y_pos.attrs.xedges = y_pos_hist_edges
                out_local_x_residuals_y_pos.attrs.yedges = local_x_residuals_hist_edges
                out_local_x_residuals_y_pos.attrs.fit_coeff = fit_local_x_residuals_y_pos
                out_local_x_residuals_y_pos.attrs.fit_cov = cov_local_x_residuals_y_pos
                out_local_x_residuals_y_pos[:] = local_x_residuals_y_pos_hist

                # 2D residual plot
                stat_2d_residuals_hist_masked = np.ma.masked_equal(stat_2d_residuals_hist, 0)
                z_min = max(0, np.ceil(np.ma.median(stat_2d_residuals_hist_masked) - 3 * np.ma.std(stat_2d_residuals_hist_masked)))
                z_max = np.ceil(np.ma.median(stat_2d_residuals_hist_masked) + 3 * np.ma.std(stat_2d_residuals_hist_masked) + 1)
                plot_utils.plot_2d_map(
                    hist2d=stat_2d_residuals_hist_masked.T,
                    plot_range=[dut_x_size[0], dut_x_size[1], dut_y_size[1], dut_y_size[0]],
                    title='2D average residuals for %s' % actual_dut.name,
                    x_axis_title='X [$\mathrm{\mu}$m]',
                    y_axis_title='Y [$\mathrm{\mu}$m]',
                    z_min=z_min,
                    z_max=z_max,
                    output_pdf=output_pdf)

                # 2D hits plot
                count_2d_hist_masked = np.ma.masked_equal(count_2d_hist, 0)
                z_min = max(0, np.ceil(np.ma.median(count_2d_hist_masked) - 3 * np.ma.std(count_2d_hist_masked)))
                z_max = np.ceil(np.ma.median(count_2d_hist_masked) + 3 * np.ma.std(count_2d_hist_masked) + 1)
                plot_utils.plot_2d_map(
                    hist2d=count_2d_hist_masked.T,
                    plot_range=[dut_x_size[0], dut_x_size[1], dut_y_size[1], dut_y_size[0]],
                    title='2D occupancy for %s' % actual_dut.name,
                    x_axis_title='X [$\mathrm{\mu}$m]',
                    y_axis_title='Y [$\mathrm{\mu}$m]',
                    z_min=z_min,
                    z_max=z_max,
                    output_pdf=output_pdf)

    if output_pdf is not None:
        output_pdf.close()

    return output_residuals_file


@save_arguments
def calculate_efficiency(telescope_configuration, input_tracks_file, select_duts, input_cluster_files=None, resolutions=None, in_pixel_resolutions=None, extend_areas=None, extend_in_pixel_areas=None, plot_ranges=None, n_bins_track_angle=100, efficiency_regions=None, efficiency_region_names=None, output_efficiency_file=None, minimum_track_density=1, cut_distances=(250.0, 250.0), plot=True, chunk_size=1000000):
    '''Takes the tracks and calculates the hit efficiency and hit/track hit distance for selected DUTs.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_tracks_file : string
        Filename of the input tracks file.
    select_duts : iterable
        Selecting DUTs that will be processed.
    input_cluster_files : list of strings
        Filenames of the input cluster files with ClusterHits table for each selected DUT.
    resolutions : 2-tuple or list of 2-tuples
        Resolutionin x and y direction (in um) of the histograms for each selected DUT.
        If None, the resolution will be set to the pixel size.
    in_pixel_resolutions : 2-tuple or list of 2-tuples
        Resolution in x and y direction (in um) of the in-pixel histograms for each selected DUT.
        If None, the resolution will be set to 1/10 of the pixel size or copied from resolutions (whatever is smaller).
    extend_areas : 2-tuple or list of 2-tuples
        Extending the area of the histograms in x and y direction (in um) for each selected DUT.
        If None, only the active are will be plotted.
    extend_in_pixel_areas : 2-tuple or list of 2-tuples
        Extending the area of the in-pixel histograms in x and y direction (in um) for each selected DUT.
        If None, only the primitive cell will be plotted.
    plot_ranges : 2-tuple of 2-tuples or list of 2-tuple of 2-tuples
        Plot range in x and y direction (in um) for each selected DUT.
        If None, use default values (i.e., positive direction of the x axis to the right and of y axis to the top, including extended area).
    n_bins_track_angle: int
        Number of bins for track angle histograms.
    efficiency_regions : tuple of tuples of 2-tuples or list of lists of tuples of 2-tuples
        Fiducial region in x and y direction (in um) for each selected DUT.
        The efficiency will be calculated plotted for each region individually.
        For each efficiency region, in-pixel plots are generated.
    efficiency_region_names : list of strings
        Optional names for efficiency regions.
    output_efficiency_file : string
        Filename of the output efficiency file. If None, the filename will be derived from the input hits file.
    minimum_track_density : uint
        Minimum track density required to consider bin for efficiency calculation.
    cut_distances : 2-tuple or list of 2-tuples
        X and y distance (in um) for each selected DUT to calculate the efficiency.
        Hits contribute to efficiency when the distance between track and hist is smaller than the cut_distance.
        If None, use infinite distance.
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    logging.info('=== Calculating efficiency for %d DUTs ===' % len(select_duts))

    # Create resolutions array
    if isinstance(resolutions, tuple) or resolutions is None:
        resolutions = [resolutions] * len(select_duts)
    # Check iterable and length
    if not isinstance(resolutions, Iterable):
        raise ValueError("Parameter resolutions is not an iterable.")
    elif not resolutions:  # empty iterable
        raise ValueError("Parameter resolutions has no items.")
    # Finally check length of all arrays
    if len(resolutions) != len(select_duts):  # empty iterable
        raise ValueError("Parameter resolutions has the wrong length.")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, resolutions)):
        raise ValueError("Not all items in parameter resolutions are iterable or None.")
    # Finally check length of all arrays
    for resolution in resolutions:
        if resolution is not None and len(resolution) != 2:  # check the length of the items
            raise ValueError("Item in parameter resolutions has length != 2.")

    # Create in-pixel resolutions array
    if isinstance(in_pixel_resolutions, tuple) or in_pixel_resolutions is None:
        in_pixel_resolutions = [in_pixel_resolutions] * len(select_duts)
    # Check iterable and length
    if not isinstance(in_pixel_resolutions, Iterable):
        raise ValueError("Parameter in_pixel_resolutions is not an iterable.")
    elif not in_pixel_resolutions:  # empty iterable
        raise ValueError("Parameter in_pixel_resolutions has no items.")
    # Finally check length of all arrays
    if len(in_pixel_resolutions) != len(select_duts):  # empty iterable
        raise ValueError("Parameter in_pixel_resolutions has the wrong length.")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, in_pixel_resolutions)):
        raise ValueError("Not all items in parameter in_pixel_resolutions are iterable or None.")
    # Finally check length of all arrays
    for resolution in in_pixel_resolutions:
        if resolution is not None and len(resolution) != 2:  # check the length of the items
            raise ValueError("Item in parameter in_pixel_resolutions has length != 2.")

    # Create extend_areas array
    if isinstance(extend_areas, tuple) or extend_areas is None:
        extend_areas = [extend_areas] * len(select_duts)
    # Check iterable and length
    if not isinstance(extend_areas, Iterable):
        raise ValueError("Parameter extend_areas is not an iterable.")
    elif not extend_areas:  # empty iterable
        raise ValueError("Parameter extend_areas has no items.")
    # Finally check length of all arrays
    if len(extend_areas) != len(select_duts):  # empty iterable
        raise ValueError("Parameter extend_areas has the wrong length.")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, extend_areas)):
        raise ValueError("Not all items in parameter extend_areas are iterable or None.")
    # Finally check length of all arrays
    for extend_area in extend_areas:
        if extend_area is not None and len(extend_area) != 2:  # check the length of the items
            raise ValueError("Item in parameter extend_areas has length != 2.")

    # Create extend_in_pixel_areas array
    if isinstance(extend_in_pixel_areas, tuple) or extend_in_pixel_areas is None:
        extend_in_pixel_areas = [extend_in_pixel_areas] * len(select_duts)
    # Check iterable and length
    if not isinstance(extend_in_pixel_areas, Iterable):
        raise ValueError("Parameter extend_in_pixel_areas is not an iterable.")
    elif not extend_in_pixel_areas:  # empty iterable
        raise ValueError("Parameter extend_in_pixel_areas has no items.")
    # Finally check length of all arrays
    if len(extend_in_pixel_areas) != len(select_duts):  # empty iterable
        raise ValueError("Parameter extend_in_pixel_areas has the wrong length.")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, extend_in_pixel_areas)):
        raise ValueError("Not all items in parameter extend_in_pixel_areas are iterable or None.")
    # Finally check length of all arrays
    for extend_area in extend_in_pixel_areas:
        if extend_area is not None and len(extend_area) != 2:  # check the length of the items
            raise ValueError("Item in parameter extend_in_pixel_areas has length != 2.")

    # Create plot_ranges array
    if isinstance(plot_ranges, tuple) or plot_ranges is None:
        plot_ranges = [plot_ranges] * len(select_duts)
    # Check iterable and length
    if not isinstance(plot_ranges, Iterable):
        raise ValueError("Parameter plot_ranges is not an iterable.")
    elif not plot_ranges:  # empty iterable
        raise ValueError("Parameter plot_ranges has no items.")
    # Finally check length of all arrays
    if len(plot_ranges) != len(select_duts):  # empty iterable
        raise ValueError("Parameter plot_ranges has the wrong length.")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, plot_ranges)):
        raise ValueError("Not all items in parameter plot_ranges are iterable or None.")
    # Finally check length of all arrays
    for plot_range in plot_ranges:
        if plot_range is not None:
            if len(plot_range) != 2:  # check the length of the items
                raise ValueError("Item in parameter plot_ranges has length != 2.")
            for plot_range_direction in plot_range:
                if len(plot_range_direction) != 2:  # check the length of the items
                    raise ValueError("Item in parameter plot_ranges is not 2-tuple of 2-tuples.")

    # Create efficiency_regions array
    if isinstance(efficiency_regions, tuple) or efficiency_regions is None:
        efficiency_regions = [efficiency_regions] * len(select_duts)
    # Check iterable and length
    if not isinstance(efficiency_regions, list):
        raise ValueError("Parameter efficiency_regions is not a list.")
    elif not efficiency_regions:  # empty iterable
        raise ValueError("Parameter efficiency_regions has no items.")
    # Finally check length of all arrays
    if len(efficiency_regions) != len(select_duts):  # empty iterable
        raise ValueError("Parameter efficiency_regions has the wrong length.")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, efficiency_regions)):
        raise ValueError("Not all items in parameter efficiency_regions are iterable or None.")
    # Finally check length of all arrays
    for regions in efficiency_regions:
        if regions is not None:
            for region in regions:
                if len(region) != 2:  # check the length of the items
                    raise ValueError("Item in parameter efficiency_regions has length != 2.")
                for region_direction in region:
                    if len(region_direction) != 2:  # check the length of the items
                        raise ValueError("Item in parameter efficiency_regions is not list of 2-tuples of 2-tuples.")

    # Create efficiency_region_names array
    if efficiency_region_names is None:
        efficiency_region_names = [None] * len(select_duts)  # expand to select_duts
    # Check iterable and length
    if not isinstance(efficiency_region_names, list):
        raise ValueError("Parameter efficiency_region_names is not a list.")
    elif not efficiency_region_names:  # empty iterable
        raise ValueError("Parameter efficiency_region_names has no items.")
    # Finally check length of all arrays
    if len(efficiency_region_names) != len(select_duts):  # empty iterable
        raise ValueError("Parameter efficiency_region_names has the wrong length.")
    # Finally check length of all arrays
    for index, (region_name, regions) in enumerate(zip(efficiency_region_names, efficiency_regions)):
        if regions and region_name is None:
            efficiency_region_names[index] = [None] * len(regions)  # expand to item in select_duts
        if regions and len(efficiency_region_names[index]) != len(regions):
            raise ValueError("Item in parameter efficiency_region_names has wrong length.")

    # Create cut distance
    if isinstance(cut_distances, tuple) or cut_distances is None:
        cut_distances = [cut_distances] * len(select_duts)
    # Check iterable and length
    if not isinstance(cut_distances, Iterable):
        raise ValueError("Parameter cut_distances is not an iterable.")
    elif not cut_distances:  # empty iterable
        raise ValueError("Parameter cut_distances has no items.")
    # Finally check length of all arrays
    if len(cut_distances) != len(select_duts):  # empty iterable
        raise ValueError("Parameter cut_distances has the wrong length.")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, cut_distances)):
        raise ValueError("Not all items in parameter cut_distances are iterable or None.")
    # Finally check length of all arrays
    for distance in cut_distances:
        if distance is not None and len(distance) != 2:  # check the length of the items
            raise ValueError("Item in parameter cut_distances has length != 2.")

    if output_efficiency_file is None:
        output_efficiency_file = os.path.splitext(input_tracks_file)[0] + '_efficiency.h5'

    if plot is True:
        output_pdf = PdfPages(os.path.splitext(output_efficiency_file)[0] + '.pdf', keep_empty=False)
    else:
        output_pdf = None

    if input_cluster_files is not None and len(select_duts) != len(input_cluster_files):
        raise ValueError('Parameter "input_cluster_files" has wrong length.')

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_efficiency_file, mode='w') as out_file_h5:
            for index, actual_dut_index in enumerate(select_duts):
                if input_cluster_files is not None and input_cluster_files[index] is not None:
                    in_cluster_file_h5 = tb.open_file(filename=input_cluster_files[index], mode='r')
                else:
                    in_cluster_file_h5 = None
                actual_dut = telescope[actual_dut_index]
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)

                logging.info('== Calculating efficiency for %s ==', actual_dut.name)

                # Calculate histogram properties (bins size and number of bins)
                resolution = resolutions[index]
                if resolution is None:
                    resolution = actual_dut.pixel_size
                if resolution[0] is None:
                    resolution[0] = actual_dut.pixel_size[0]
                if resolution[1] is None:
                    resolution[1] = actual_dut.pixel_size[1]
                extend_area = extend_areas[index]

                # select cluster shapes for analysis
                efficiency_regions_analyze_cluster_shapes = [1, 3, 5, 13, 14, 7, 11, 15]

                # DUT size
                dut_x_extent = actual_dut.x_extent(global_position=False)
                dut_x_size = dut_x_extent[1] - dut_x_extent[0]
                dut_y_extent = actual_dut.y_extent(global_position=False)
                dut_y_size = dut_y_extent[1] - dut_y_extent[0]
                dut_extent = [dut_x_extent[0], dut_x_extent[1], dut_y_extent[0], dut_y_extent[1]]

                # DUT hist size
                dut_x_center = 0.0
                dut_hist_x_size = math.ceil(dut_x_size / resolution[0]) * resolution[0]
                if extend_area is not None and extend_area[0] is not None:
                    dut_hist_x_extent_area = math.ceil(extend_area[0] / resolution[0]) * resolution[0] * 2.0
                    dut_hist_x_size += dut_hist_x_extent_area
                dut_hist_x_extent = [dut_x_center - dut_hist_x_size / 2.0, dut_x_center + dut_hist_x_size / 2.0]
                dut_y_center = 0.0
                dut_hist_y_size = math.ceil(dut_y_size / resolution[1]) * resolution[1]
                if extend_area is not None and extend_area[1] is not None:
                    dut_hist_y_extent_area = math.ceil(extend_area[1] / resolution[1]) * resolution[1] * 2.0
                    dut_hist_y_size += dut_hist_y_extent_area
                dut_hist_y_extent = [dut_y_center - dut_hist_y_size / 2.0, dut_y_center + dut_hist_y_size / 2.0]
                hist_extent = [dut_hist_x_extent[0], dut_hist_x_extent[1], dut_hist_y_extent[0], dut_hist_y_extent[1]]
                hist_2d_x_n_bins = int(dut_hist_x_size / resolution[0])
                hist_2d_y_n_bins = int(dut_hist_y_size / resolution[1])
                hist_2d_x_edges = np.linspace(dut_hist_x_extent[0], dut_hist_x_extent[1], hist_2d_x_n_bins + 1, endpoint=True)
                hist_2d_y_edges = np.linspace(dut_hist_y_extent[0], dut_hist_y_extent[1], hist_2d_y_n_bins + 1, endpoint=True)
                hist_2d_edges = [hist_2d_x_edges, hist_2d_y_edges]
                plot_range = plot_ranges[index]
                if plot_range is None:
                    plot_range = [dut_hist_x_extent, dut_hist_y_extent]
                efficiency_regions_dut = efficiency_regions[index]
                efficiency_regions_names_dut = efficiency_region_names[index]
                if efficiency_regions_dut is not None:
                    extend_in_pixel_area = extend_in_pixel_areas[index]
                    # Calculate in-pixel histogram properties (bins size and number of bins)
                    in_pixel_resolution = in_pixel_resolutions[index]
                    if in_pixel_resolution is None:
                        in_pixel_resolution = map(min, zip(resolution, [pitch / 10.0 for pitch in actual_dut.pixel_size]))
                    if in_pixel_resolution[0] is None:
                        in_pixel_resolution[0] = min(resolution[0], actual_dut.pixel_size[0] / 10.0)
                    if in_pixel_resolution[1] is None:
                        in_pixel_resolution[1] = min(resolution[1], actual_dut.pixel_size[1] / 10.0)
                    # generate hists for each region
                    efficiency_regions_efficiency = []
                    efficiency_regions_efficiency_chunk = []
                    efficiency_regions_stat = []
                    efficiency_regions_count_1d_charge_hist = []
                    efficiency_regions_count_1d_frame_hist = []
                    efficiency_regions_count_1d_cluster_size_hist = []
                    efficiency_regions_count_1d_cluster_shape_hist = []
                    efficiency_regions_count_1d_total_angle_hist = []
                    efficiency_regions_count_1d_total_angle_hist_edges = []
                    efficiency_regions_count_1d_alpha_angle_hist = []
                    efficiency_regions_count_1d_alpha_angle_hist_edges = []
                    efficiency_regions_count_1d_beta_angle_hist = []
                    efficiency_regions_count_1d_beta_angle_hist_edges = []
                    efficiency_regions_count_tracks_pixel_hist = []
                    efficiency_regions_count_tracks_with_hit_pixel_hist = []
                    efficiency_regions_stat_pixel_efficiency_hist = []
                    efficiency_regions_chunk_stats = []
                    for region in efficiency_regions_dut:
                        efficiency_regions_efficiency.append(0.0)
                        efficiency_regions_efficiency_chunk.append(0.0)
                        efficiency_regions_stat.append(0)
                        efficiency_regions_count_1d_charge_hist.append(None)
                        efficiency_regions_count_1d_frame_hist.append(None)
                        efficiency_regions_count_1d_cluster_size_hist.append(None)
                        efficiency_regions_count_1d_total_angle_hist.append(None)
                        efficiency_regions_count_1d_total_angle_hist_edges.append(None)
                        efficiency_regions_count_1d_alpha_angle_hist.append(None)
                        efficiency_regions_count_1d_alpha_angle_hist_edges.append(None)
                        efficiency_regions_count_1d_beta_angle_hist.append(None)
                        efficiency_regions_count_1d_beta_angle_hist_edges.append(None)
                        efficiency_regions_count_1d_cluster_shape_hist.append(None)
                        efficiency_regions_count_tracks_pixel_hist.append(None)
                        efficiency_regions_count_tracks_with_hit_pixel_hist.append(None)
                        efficiency_regions_stat_pixel_efficiency_hist.append(None)
                        efficiency_regions_chunk_stats.append([])

                    # for in-pixel statistics
                    # generate primitive cell of the size of a single pixel
                    dut_in_pixel_hist_x_size = math.ceil(actual_dut.column_size / in_pixel_resolution[0]) * in_pixel_resolution[0]
                    dut_in_pixel_hist_y_size = math.ceil(actual_dut.row_size / in_pixel_resolution[1]) * in_pixel_resolution[1]
                    hist_in_pixel_x_n_bins = int(dut_in_pixel_hist_x_size / in_pixel_resolution[0])
                    hist_in_pixel_y_n_bins = int(dut_in_pixel_hist_y_size / in_pixel_resolution[1])
                    hist_in_pixel_x_edges = np.linspace(0.0, dut_in_pixel_hist_x_size, hist_in_pixel_x_n_bins + 1, endpoint=True)
                    hist_in_pixel_y_edges = np.linspace(0.0, dut_in_pixel_hist_y_size, hist_in_pixel_y_n_bins + 1, endpoint=True)
                    hist_in_pixel_2d_edges = [hist_in_pixel_x_edges, hist_in_pixel_y_edges]
                    hist_in_pixel_cluster_shape_edges = [hist_in_pixel_x_edges, hist_in_pixel_y_edges, range(len(efficiency_regions_analyze_cluster_shapes) + 1)]
                    in_pixel_hist_extent = [0.0, dut_in_pixel_hist_x_size, 0.0, dut_in_pixel_hist_y_size]
                    in_pixel_plot_range = [[0.0, dut_in_pixel_hist_x_size], [0.0, dut_in_pixel_hist_y_size]]
                    # generate in-pixel hists for each region
                    count_in_pixel_hits_2d_hists = []
                    count_in_pixel_tracks_2d_hists = []
                    count_in_pixel_tracks_with_hit_2d_hists = []
                    stat_in_pixel_x_residuals_2d_hists = []
                    stat_in_pixel_y_residuals_2d_hists = []
                    stat_in_pixel_residuals_2d_hists = []
                    stat_in_pixel_charge_2d_hists = []
                    stat_in_pixel_frame_2d_hists = []
                    stat_in_pixel_cluster_size_2d_hists = []
                    count_in_pixel_cluster_shape_2d_hists = []
                    stat_in_pixel_cluster_shape_2d_hists = []
                    for region in efficiency_regions_dut:
                        # 2D in-pixel hits
                        count_in_pixel_hits_2d_hists.append(np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64))
                        # 2D in-pixel tracks
                        count_in_pixel_tracks_2d_hists.append(np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64))
                        # 2D in-pixel tracks with valid hit
                        count_in_pixel_tracks_with_hit_2d_hists.append(np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64))
                        # 2D in-pixel x residuals
                        stat_in_pixel_x_residuals_2d_hists.append(np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64))
                        # 2D in-pixel y residuals
                        stat_in_pixel_y_residuals_2d_hists.append(np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64))
                        # 2D in-pixel residuals
                        stat_in_pixel_residuals_2d_hists.append(np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64))
                        # 2D in-pixel charge
                        stat_in_pixel_charge_2d_hists.append(np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64))
                        # 2D in-pixel frame
                        stat_in_pixel_frame_2d_hists.append(np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64))
                        # 2D in-pixel mean cluster size
                        stat_in_pixel_cluster_size_2d_hists.append(np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64))
                        # 2D in-pixel cluster shape
                        count_in_pixel_cluster_shape_2d_hists.append(np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins, len(efficiency_regions_analyze_cluster_shapes)), dtype=np.float64))
                else:
                    efficiency_regions_efficiency = None
                    efficiency_regions_efficiency_chunk = None
                    efficiency_regions_stat = None
                    efficiency_regions_count_1d_charge_hist = None
                    efficiency_regions_count_1d_frame_hist = None
                    efficiency_regions_count_1d_cluster_size_hist = None
                    efficiency_regions_count_1d_total_angle_hist = None
                    efficiency_regions_count_1d_total_angle_hist_edges = None
                    efficiency_regions_count_1d_alpha_angle_hist = None
                    efficiency_regions_count_1d_alpha_angle_hist_edges = None
                    efficiency_regions_count_1d_beta_angle_hist = None
                    efficiency_regions_count_1d_beta_angle_hist_edges = None
                    efficiency_regions_count_1d_cluster_shape_hist = None
                    efficiency_regions_count_tracks_pixel_hist = None
                    efficiency_regions_count_tracks_with_hit_pixel_hist = None
                    efficiency_regions_stat_pixel_efficiency_hist = None

                pixel_indices = np.indices((actual_dut.n_columns, actual_dut.n_rows)).reshape(2, -1).T
                local_x, local_y, _ = actual_dut.index_to_local_position(
                    column=pixel_indices[:, 0] + 1,
                    row=pixel_indices[:, 1] + 1)
                pixel_center_data = np.column_stack((local_x, local_y))
                points, _, _, _ = analysis_utils.voronoi_finite_polygons_2d(points=pixel_center_data, dut_extent=dut_extent)
                # pixel_center_data is a subset of points
                # points includes additional image points
                pixel_center_extended_kd_tree = cKDTree(points)
                count_tracks_pixel_hist = np.zeros(shape=pixel_center_data.shape[0], dtype=np.int64)
                count_tracks_with_hit_pixel_hist = np.zeros(shape=pixel_center_data.shape[0], dtype=np.int64)
                if in_cluster_file_h5:
                    x_bin_centers = (hist_2d_edges[0][1:] + hist_2d_edges[0][:-1]) / 2
                    y_bin_centers = (hist_2d_edges[1][1:] + hist_2d_edges[1][:-1]) / 2
                    y_meshgrid, x_meshgrid = np.meshgrid(y_bin_centers, x_bin_centers)
                    bin_center_data = np.column_stack((np.ravel(x_meshgrid), np.ravel(y_meshgrid)))
                    _, bin_center_to_pixel_center = cKDTree(pixel_center_data).query(bin_center_data)
                    # estimate the size of the relative hit index array
                    # assume a region of 1mm x 1mm
                    shape_x = int(math.ceil(1000.0 / actual_dut.column_size))
                    if shape_x % 2 == 0:
                        shape_x += 1
                    else:
                        shape_x += 2
                    shape_y = int(math.ceil(1000.0 / actual_dut.row_size))
                    if shape_y % 2 == 0:
                        shape_y += 1
                    else:
                        shape_y += 2
                    count_pixel_hits_2d_hist = np.zeros(shape=(hist_2d_x_n_bins, hist_2d_y_n_bins, shape_x, shape_y), dtype=np.int32)
                    if (count_pixel_hits_2d_hist.shape[2] <= 1) or (count_pixel_hits_2d_hist.shape[2] % 2 != 1) or (count_pixel_hits_2d_hist.shape[3] <= 1) or (count_pixel_hits_2d_hist.shape[3] % 2 != 1):
                        raise RuntimeError("Invalid shape of histogram")
                else:
                    count_pixel_hits_2d_hist = None

                pbar = tqdm(total=node.shape[0], ncols=80)
                initialize = True
                start_index_cluster_hits = 0
                chunk_indices = []
                for tracks_chunk, index_chunk in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    # Transform the hits and track intersections into the local coordinate system
                    # Coordinates in global coordinate system (x, y, z)
                    hit_x_local, hit_y_local = tracks_chunk['x_dut_%d' % actual_dut_index], tracks_chunk['y_dut_%d' % actual_dut_index]
                    intersection_x_local, intersection_y_local = tracks_chunk['offset_x'], tracks_chunk['offset_y']
                    charge = tracks_chunk['charge_dut_%d' % actual_dut_index]
                    frame = tracks_chunk['frame_dut_%d' % actual_dut_index]
                    cluster_size = tracks_chunk['n_hits_dut_%d' % actual_dut_index]
                    cluster_shape = tracks_chunk['cluster_shape_dut_%d' % actual_dut_index]

                    # Calculate distance between track intersection and DUT hit location
                    x_residuals = hit_x_local - intersection_x_local
                    y_residuals = hit_y_local - intersection_y_local
                    distance_local = np.sqrt(np.square(x_residuals) + np.square(y_residuals))
                    select_finite_distance = np.isfinite(distance_local)

                    # Calculate track angles in column and row direction (local coordinate system)
                    track_slopes_local = np.column_stack([
                        tracks_chunk['slope_x'],
                        tracks_chunk['slope_y'],
                        tracks_chunk['slope_z']])
                    total_angles_local, alpha_angles_local, beta_angles_local = get_angles(
                        slopes=track_slopes_local,
                        xz_plane_normal=np.array([0.0, 1.0, 0.0]),
                        yz_plane_normal=np.array([1.0, 0.0, 0.0]),
                        dut_plane_normal=np.array([0.0, 0.0, 1.0]))

                    cut_distance = cut_distances[index]
                    if cut_distance is None:
                        cut_distance = (np.inf, np.inf)
                    if cut_distance[0] is None:
                        cut_distance[0] = np.inf
                    if cut_distance[1] is None:
                        cut_distance[1] = np.inf
                    # Select data where distance between the hit and track is smaller than the given value
                    select_valid_hit = np.isfinite(hit_x_local)
                    if cut_distance[0] >= 2.5 * actual_dut.pixel_size[0] and cut_distance[1] >= 2.5 * actual_dut.pixel_size[1]:  # use ellipse
                        select_valid_hit[select_finite_distance] &= ((np.square(x_residuals[select_finite_distance]) / cut_distance[0]**2) + (np.square(y_residuals[select_finite_distance]) / cut_distance[1]**2)) <= 1
                    else:  # use square
                        select_valid_hit[select_finite_distance] &= (np.abs(x_residuals[select_finite_distance]) <= cut_distance[0])
                        select_valid_hit[select_finite_distance] &= (np.abs(y_residuals[select_finite_distance]) <= cut_distance[1])
                    # select cluser shapes with cluster size <=4
                    select_small_cluster_sizes = np.zeros_like(select_valid_hit)
                    for cluster_shape_value in efficiency_regions_analyze_cluster_shapes:
                        select_small_cluster_sizes |= cluster_shape_value == cluster_shape_value
                    if efficiency_regions_dut is not None:
                        # for in-pixel statistics
                        # if neccessary increase size of primitive cell (may be larger than single pixel)
                        in_pixel_hit_x_local, in_pixel_hit_y_local = actual_dut.map_to_primitive_cell(hit_x_local, hit_y_local)
                        in_pixel_intersection_x_local, in_pixel_intersection_y_local = actual_dut.map_to_primitive_cell(intersection_x_local, intersection_y_local)
                        resize_in_pixel_hist = False
                        if np.any(in_pixel_intersection_x_local > dut_in_pixel_hist_x_size):
                            resize_in_pixel_hist = True
                            dut_in_pixel_hist_x_size = math.ceil(np.ceil(np.max(in_pixel_intersection_x_local) / actual_dut.column_size) * actual_dut.column_size / in_pixel_resolution[0]) * in_pixel_resolution[0]
                            hist_in_pixel_x_n_bins = int(dut_in_pixel_hist_x_size / in_pixel_resolution[0])
                            hist_in_pixel_x_edges = np.linspace(0.0, dut_in_pixel_hist_x_size, hist_in_pixel_x_n_bins + 1, endpoint=True)
                        if np.any(in_pixel_intersection_y_local > dut_in_pixel_hist_y_size):
                            resize_in_pixel_hist = True
                            dut_in_pixel_hist_y_size = math.ceil(np.ceil(np.max(in_pixel_intersection_y_local) / actual_dut.row_size) * actual_dut.row_size / in_pixel_resolution[1]) * in_pixel_resolution[1]
                            hist_in_pixel_y_n_bins = int(dut_in_pixel_hist_y_size / in_pixel_resolution[1])
                            hist_in_pixel_y_edges = np.linspace(0.0, dut_in_pixel_hist_y_size, hist_in_pixel_y_n_bins + 1, endpoint=True)
                        if resize_in_pixel_hist:
                            hist_in_pixel_2d_edges = [hist_in_pixel_x_edges, hist_in_pixel_y_edges]
                            hist_in_pixel_cluster_shape_edges = [hist_in_pixel_x_edges, hist_in_pixel_y_edges, range(len(efficiency_regions_analyze_cluster_shapes) + 1)]
                            in_pixel_hist_extent = [0.0, dut_in_pixel_hist_x_size, 0.0, dut_in_pixel_hist_y_size]
                            in_pixel_plot_range = [[0.0, dut_in_pixel_hist_x_size], [0.0, dut_in_pixel_hist_y_size]]
                            for region_index, region in enumerate(efficiency_regions_dut):
                                # extend in-pixel arrays
                                # 2D in-pixel hits
                                count_in_pixel_hits_2d_hist_tmp = count_in_pixel_hits_2d_hists[region_index]
                                count_in_pixel_hits_2d_hists[region_index] = np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64)
                                count_in_pixel_hits_2d_hists[region_index][:count_in_pixel_hits_2d_hist_tmp.shape[0], :count_in_pixel_hits_2d_hist_tmp.shape[1]] += count_in_pixel_hits_2d_hist_tmp
                                # 2D in-pixel tracks
                                count_in_pixel_tracks_2d_hist_tmp = count_in_pixel_tracks_2d_hists[region_index]
                                count_in_pixel_tracks_2d_hists[region_index] = np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64)
                                count_in_pixel_tracks_2d_hists[region_index][:count_in_pixel_tracks_2d_hist_tmp.shape[0], :count_in_pixel_tracks_2d_hist_tmp.shape[1]] += count_in_pixel_tracks_2d_hist_tmp
                                # 2D in-pixel tracks with valid hit
                                count_in_pixel_tracks_with_hit_2d_hist_tmp = count_in_pixel_tracks_with_hit_2d_hists[region_index]
                                count_in_pixel_tracks_with_hit_2d_hists[region_index] = np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64)
                                count_in_pixel_tracks_with_hit_2d_hists[region_index][:count_in_pixel_tracks_with_hit_2d_hist_tmp.shape[0], :count_in_pixel_tracks_with_hit_2d_hist_tmp.shape[1]] += count_in_pixel_tracks_with_hit_2d_hist_tmp
                                # 2D in-pixel x residuals
                                stat_in_pixel_x_residuals_2d_hist_tmp = stat_in_pixel_x_residuals_2d_hists[region_index]
                                stat_in_pixel_x_residuals_2d_hists[region_index] = np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64)
                                stat_in_pixel_x_residuals_2d_hists[region_index][:stat_in_pixel_x_residuals_2d_hist_tmp.shape[0], :stat_in_pixel_x_residuals_2d_hist_tmp.shape[1]] += stat_in_pixel_x_residuals_2d_hist_tmp
                                # 2D in-pixel y residuals
                                stat_in_pixel_y_residuals_2d_hist_tmp = stat_in_pixel_y_residuals_2d_hists[region_index]
                                stat_in_pixel_y_residuals_2d_hists[region_index] = np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64)
                                stat_in_pixel_y_residuals_2d_hists[region_index][:stat_in_pixel_y_residuals_2d_hist_tmp.shape[0], :stat_in_pixel_y_residuals_2d_hist_tmp.shape[1]] += stat_in_pixel_y_residuals_2d_hist_tmp
                                # 2D in-pixel residuals
                                stat_in_pixel_residuals_2d_hist_tmp = stat_in_pixel_residuals_2d_hists[region_index]
                                stat_in_pixel_residuals_2d_hists[region_index] = np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64)
                                stat_in_pixel_residuals_2d_hists[region_index][:stat_in_pixel_residuals_2d_hist_tmp.shape[0], :stat_in_pixel_residuals_2d_hist_tmp.shape[1]] += stat_in_pixel_residuals_2d_hist_tmp
                                # 2D in-pixel charge
                                stat_in_pixel_charge_2d_hist_tmp = stat_in_pixel_charge_2d_hists[region_index]
                                stat_in_pixel_charge_2d_hists[region_index] = np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64)
                                stat_in_pixel_charge_2d_hists[region_index][:stat_in_pixel_charge_2d_hist_tmp.shape[0], :stat_in_pixel_charge_2d_hist_tmp.shape[1]] += stat_in_pixel_charge_2d_hist_tmp
                                # 2D in-pixel frame
                                stat_in_pixel_frame_2d_hist_tmp = stat_in_pixel_frame_2d_hists[region_index]
                                stat_in_pixel_frame_2d_hists[region_index] = np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64)
                                stat_in_pixel_frame_2d_hists[region_index][:stat_in_pixel_frame_2d_hist_tmp.shape[0], :stat_in_pixel_frame_2d_hist_tmp.shape[1]] += stat_in_pixel_frame_2d_hist_tmp
                                # 2D in-pixel mean cluster size
                                stat_in_pixel_cluster_size_2d_hist_tmp = stat_in_pixel_cluster_size_2d_hists[region_index]
                                stat_in_pixel_cluster_size_2d_hists[region_index] = np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins), dtype=np.float64)
                                stat_in_pixel_cluster_size_2d_hists[region_index][:stat_in_pixel_cluster_size_2d_hist_tmp.shape[0], :stat_in_pixel_cluster_size_2d_hist_tmp.shape[1]] += stat_in_pixel_cluster_size_2d_hist_tmp
                                # 2D in-pixel cluster shape
                                count_in_pixel_cluster_shape_2d_hist_tmp = count_in_pixel_cluster_shape_2d_hists[region_index]
                                count_in_pixel_cluster_shape_2d_hists[region_index] = np.zeros(shape=(hist_in_pixel_x_n_bins, hist_in_pixel_y_n_bins, len(efficiency_regions_analyze_cluster_shapes)), dtype=np.float64)
                                count_in_pixel_cluster_shape_2d_hists[region_index][:count_in_pixel_cluster_shape_2d_hist_tmp.shape[0], :count_in_pixel_cluster_shape_2d_hist_tmp.shape[1], :] += count_in_pixel_cluster_shape_2d_hist_tmp

                        for region_index, region in enumerate(efficiency_regions_dut):
                            select_valid_tracks_efficiency_region = np.ones_like(select_valid_hit)
                            select_valid_tracks_efficiency_region &= intersection_x_local > min(region[0])
                            select_valid_tracks_efficiency_region &= intersection_x_local < max(region[0])
                            select_valid_tracks_efficiency_region &= intersection_y_local > min(region[1])
                            select_valid_tracks_efficiency_region &= intersection_y_local < max(region[1])
                            # Moving average
                            efficiency_regions_efficiency[region_index] = (efficiency_regions_efficiency[region_index] * efficiency_regions_stat[region_index] + np.count_nonzero(select_valid_hit[select_valid_tracks_efficiency_region])) / (efficiency_regions_stat[region_index] + np.count_nonzero(select_valid_tracks_efficiency_region))
                            efficiency_regions_stat[region_index] = efficiency_regions_stat[region_index] + np.count_nonzero(select_valid_tracks_efficiency_region)
                            # Per chunk
                            efficiency_regions_efficiency_chunk[region_index] = np.count_nonzero(select_valid_hit[select_valid_tracks_efficiency_region]) / np.count_nonzero(select_valid_tracks_efficiency_region)
                            if efficiency_regions_count_1d_charge_hist[region_index] is None:
                                efficiency_regions_count_1d_charge_hist[region_index] = np.bincount(charge[select_valid_tracks_efficiency_region & select_valid_hit].astype(np.int64))
                            else:
                                efficiency_region_count_1d_charge_hist_tmp = np.bincount(charge[select_valid_tracks_efficiency_region & select_valid_hit].astype(np.int64))
                                if efficiency_region_count_1d_charge_hist_tmp.size > efficiency_regions_count_1d_charge_hist[region_index].size:
                                    efficiency_regions_count_1d_charge_hist[region_index].resize(efficiency_region_count_1d_charge_hist_tmp.size)
                                else:
                                    efficiency_region_count_1d_charge_hist_tmp.resize(efficiency_regions_count_1d_charge_hist[region_index].size)
                                efficiency_regions_count_1d_charge_hist[region_index] += efficiency_region_count_1d_charge_hist_tmp
                            if efficiency_regions_count_1d_frame_hist[region_index] is None:
                                efficiency_regions_count_1d_frame_hist[region_index] = np.bincount(frame[select_valid_tracks_efficiency_region & select_valid_hit])
                            else:
                                efficiency_region_count_1d_frame_hist_tmp = np.bincount(frame[select_valid_tracks_efficiency_region & select_valid_hit])
                                if efficiency_region_count_1d_frame_hist_tmp.size > efficiency_regions_count_1d_frame_hist[region_index].size:
                                    efficiency_regions_count_1d_frame_hist[region_index].resize(efficiency_region_count_1d_frame_hist_tmp.size)
                                else:
                                    efficiency_region_count_1d_frame_hist_tmp.resize(efficiency_regions_count_1d_frame_hist[region_index].size)
                                efficiency_regions_count_1d_frame_hist[region_index] += efficiency_region_count_1d_frame_hist_tmp
                            if efficiency_regions_count_1d_cluster_size_hist[region_index] is None:
                                efficiency_regions_count_1d_cluster_size_hist[region_index] = np.bincount(cluster_size[select_valid_tracks_efficiency_region & select_valid_hit])
                            else:
                                efficiency_region_count_1d_cluster_size_hist_tmp = np.bincount(cluster_size[select_valid_tracks_efficiency_region & select_valid_hit])
                                if efficiency_region_count_1d_cluster_size_hist_tmp.size > efficiency_regions_count_1d_cluster_size_hist[region_index].size:
                                    efficiency_regions_count_1d_cluster_size_hist[region_index].resize(efficiency_region_count_1d_cluster_size_hist_tmp.size)
                                else:
                                    efficiency_region_count_1d_cluster_size_hist_tmp.resize(efficiency_regions_count_1d_cluster_size_hist[region_index].size)
                                efficiency_regions_count_1d_cluster_size_hist[region_index] += efficiency_region_count_1d_cluster_size_hist_tmp
                            if efficiency_regions_count_1d_total_angle_hist[region_index] is None:
                                local_total_mean = np.nanmean(total_angles_local[select_valid_tracks_efficiency_region & select_valid_hit])
                                local_total_std = np.nanstd(total_angles_local[select_valid_tracks_efficiency_region & select_valid_hit])
                                efficiency_regions_count_1d_total_angle_hist[region_index], efficiency_regions_count_1d_total_angle_hist_edges[region_index] = np.histogram(total_angles_local[select_valid_tracks_efficiency_region & select_valid_hit], bins=n_bins_track_angle, range=(local_total_mean - 5 * local_total_std, local_total_mean + 5 * local_total_std))
                            else:
                                efficiency_region_count_1d_total_angle_hist_tmp = np.histogram(total_angles_local[select_valid_tracks_efficiency_region & select_valid_hit], bins=efficiency_regions_count_1d_total_angle_hist_edges[region_index])[0]
                                if efficiency_region_count_1d_total_angle_hist_tmp.size > efficiency_regions_count_1d_total_angle_hist[region_index].size:
                                    efficiency_regions_count_1d_total_angle_hist[region_index].resize(efficiency_region_count_1d_total_angle_hist_tmp.size)
                                else:
                                    efficiency_region_count_1d_total_angle_hist_tmp.resize(efficiency_regions_count_1d_total_angle_hist[region_index].size)
                                efficiency_regions_count_1d_total_angle_hist[region_index] += efficiency_region_count_1d_total_angle_hist_tmp
                            if efficiency_regions_count_1d_alpha_angle_hist[region_index] is None:
                                local_alpha_mean = np.nanmean(alpha_angles_local[select_valid_tracks_efficiency_region & select_valid_hit])
                                local_alpha_std = np.nanstd(alpha_angles_local[select_valid_tracks_efficiency_region & select_valid_hit])
                                efficiency_regions_count_1d_alpha_angle_hist[region_index], efficiency_regions_count_1d_alpha_angle_hist_edges[region_index] = np.histogram(alpha_angles_local[select_valid_tracks_efficiency_region & select_valid_hit], bins=n_bins_track_angle, range=(local_alpha_mean - 5 * local_alpha_std, local_alpha_mean + 5 * local_alpha_std))
                            else:
                                efficiency_region_count_1d_alpha_angle_hist_tmp = np.histogram(alpha_angles_local[select_valid_tracks_efficiency_region & select_valid_hit], bins=efficiency_regions_count_1d_alpha_angle_hist_edges[region_index])[0]
                                if efficiency_region_count_1d_alpha_angle_hist_tmp.size > efficiency_regions_count_1d_alpha_angle_hist[region_index].size:
                                    efficiency_regions_count_1d_alpha_angle_hist[region_index].resize(efficiency_region_count_1d_alpha_angle_hist_tmp.size)
                                else:
                                    efficiency_region_count_1d_alpha_angle_hist_tmp.resize(efficiency_regions_count_1d_alpha_angle_hist[region_index].size)
                                efficiency_regions_count_1d_alpha_angle_hist[region_index] += efficiency_region_count_1d_alpha_angle_hist_tmp
                            if efficiency_regions_count_1d_beta_angle_hist[region_index] is None:
                                local_beta_mean = np.nanmean(beta_angles_local[select_valid_tracks_efficiency_region & select_valid_hit])
                                local_beta_std = np.nanstd(beta_angles_local[select_valid_tracks_efficiency_region & select_valid_hit])
                                efficiency_regions_count_1d_beta_angle_hist[region_index], efficiency_regions_count_1d_beta_angle_hist_edges[region_index] = np.histogram(beta_angles_local[select_valid_tracks_efficiency_region & select_valid_hit], bins=n_bins_track_angle, range=(local_beta_mean - 5 * local_beta_std, local_beta_mean + 5 * local_beta_std))
                            else:
                                efficiency_region_count_1d_beta_angle_hist_tmp = np.histogram(beta_angles_local[select_valid_tracks_efficiency_region & select_valid_hit], bins=efficiency_regions_count_1d_beta_angle_hist_edges[region_index])[0]
                                if efficiency_region_count_1d_beta_angle_hist_tmp.size > efficiency_regions_count_1d_beta_angle_hist[region_index].size:
                                    efficiency_regions_count_1d_beta_angle_hist[region_index].resize(efficiency_region_count_1d_beta_angle_hist_tmp.size)
                                else:
                                    efficiency_region_count_1d_beta_angle_hist_tmp.resize(efficiency_regions_count_1d_beta_angle_hist[region_index].size)
                                efficiency_regions_count_1d_beta_angle_hist[region_index] += efficiency_region_count_1d_beta_angle_hist_tmp
                            if efficiency_regions_count_1d_cluster_shape_hist[region_index] is None:
                                efficiency_regions_count_1d_cluster_shape_hist[region_index] = np.histogram(a=cluster_shape[select_valid_tracks_efficiency_region & select_valid_hit], bins=np.arange(2**(4 * 4)))[0]
                            else:
                                efficiency_regions_count_1d_cluster_shape_hist[region_index] += np.histogram(a=cluster_shape[select_valid_tracks_efficiency_region & select_valid_hit], bins=np.arange(2**(4 * 4)))[0]
                            # Pixel tracks
                            _, closest_indices = pixel_center_extended_kd_tree.query(np.column_stack((intersection_x_local[select_valid_tracks_efficiency_region], intersection_y_local[select_valid_tracks_efficiency_region])))
                            if efficiency_regions_count_tracks_pixel_hist[region_index] is None:
                                efficiency_regions_count_tracks_pixel_hist[region_index] = np.bincount(closest_indices, minlength=pixel_center_data.shape[0])[:pixel_center_data.shape[0]]
                            else:
                                efficiency_regions_count_tracks_pixel_hist[region_index] += np.bincount(closest_indices, minlength=pixel_center_data.shape[0])[:pixel_center_data.shape[0]]
                            # Pixel tracks with valid hit
                            _, closest_indices_with_hit = pixel_center_extended_kd_tree.query(np.column_stack((intersection_x_local[select_valid_tracks_efficiency_region & select_valid_hit], intersection_y_local[select_valid_tracks_efficiency_region & select_valid_hit])))
                            if efficiency_regions_count_tracks_with_hit_pixel_hist[region_index] is None:
                                efficiency_regions_count_tracks_with_hit_pixel_hist[region_index] = np.bincount(closest_indices_with_hit, minlength=pixel_center_data.shape[0])[:pixel_center_data.shape[0]]
                            else:
                                efficiency_regions_count_tracks_with_hit_pixel_hist[region_index] += np.bincount(closest_indices_with_hit, minlength=pixel_center_data.shape[0])[:pixel_center_data.shape[0]]

                            # for in-pixel statistics
                            # 2D in-pixel tracks
                            count_in_pixel_tracks_2d_hists[region_index] += stats.binned_statistic_2d(x=in_pixel_intersection_x_local[select_valid_tracks_efficiency_region], y=in_pixel_intersection_y_local[select_valid_tracks_efficiency_region], values=None, statistic='count', bins=hist_in_pixel_2d_edges)[0]
                            if np.any(select_valid_tracks_efficiency_region & select_valid_hit):  # Check for valid hits
                                count_in_pixel_tracks_with_hit_2d_hist_tmp = stats.binned_statistic_2d(x=in_pixel_intersection_x_local[select_valid_tracks_efficiency_region & select_valid_hit], y=in_pixel_intersection_y_local[select_valid_tracks_efficiency_region & select_valid_hit], values=None, statistic='count', bins=hist_in_pixel_2d_edges)[0]
                                # 2D in-pixel hits
                                count_in_pixel_hits_2d_hists[region_index] += stats.binned_statistic_2d(x=in_pixel_hit_x_local[select_valid_tracks_efficiency_region], y=in_pixel_hit_y_local[select_valid_tracks_efficiency_region], values=None, statistic='count', bins=hist_in_pixel_2d_edges)[0]
                                # 2D in-pixel x residuals
                                stat_in_pixel_x_residuals_2d_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=in_pixel_intersection_x_local[select_valid_tracks_efficiency_region & select_valid_hit], y=in_pixel_intersection_y_local[select_valid_tracks_efficiency_region & select_valid_hit], values=x_residuals[select_valid_tracks_efficiency_region & select_valid_hit], statistic='mean', bins=hist_in_pixel_2d_edges)
                                stat_in_pixel_x_residuals_2d_hist_tmp = np.nan_to_num(stat_in_pixel_x_residuals_2d_hist_tmp)
                                stat_in_pixel_x_residuals_2d_hists[region_index], _ = np.ma.average(a=np.stack([stat_in_pixel_x_residuals_2d_hists[region_index], stat_in_pixel_x_residuals_2d_hist_tmp]), axis=0, weights=np.stack([count_in_pixel_tracks_with_hit_2d_hists[region_index], count_in_pixel_tracks_with_hit_2d_hist_tmp]), returned=True)
                                # 2D in-pixel y residuals
                                stat_in_pixel_y_residuals_2d_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=in_pixel_intersection_x_local[select_valid_tracks_efficiency_region & select_valid_hit], y=in_pixel_intersection_y_local[select_valid_tracks_efficiency_region & select_valid_hit], values=y_residuals[select_valid_tracks_efficiency_region & select_valid_hit], statistic='mean', bins=hist_in_pixel_2d_edges)
                                stat_in_pixel_y_residuals_2d_hist_tmp = np.nan_to_num(stat_in_pixel_y_residuals_2d_hist_tmp)
                                stat_in_pixel_y_residuals_2d_hists[region_index], _ = np.ma.average(a=np.stack([stat_in_pixel_y_residuals_2d_hists[region_index], stat_in_pixel_y_residuals_2d_hist_tmp]), axis=0, weights=np.stack([count_in_pixel_tracks_with_hit_2d_hists[region_index], count_in_pixel_tracks_with_hit_2d_hist_tmp]), returned=True)
                                # 2D in-pixel residuals
                                stat_in_pixel_residuals_2d_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=in_pixel_intersection_x_local[select_valid_tracks_efficiency_region & select_valid_hit], y=in_pixel_intersection_y_local[select_valid_tracks_efficiency_region & select_valid_hit], values=distance_local[select_valid_tracks_efficiency_region & select_valid_hit], statistic='mean', bins=hist_in_pixel_2d_edges)
                                stat_in_pixel_residuals_2d_hist_tmp = np.nan_to_num(stat_in_pixel_residuals_2d_hist_tmp)
                                stat_in_pixel_residuals_2d_hists[region_index], _ = np.ma.average(a=np.stack([stat_in_pixel_residuals_2d_hists[region_index], stat_in_pixel_residuals_2d_hist_tmp]), axis=0, weights=np.stack([count_in_pixel_tracks_with_hit_2d_hists[region_index], count_in_pixel_tracks_with_hit_2d_hist_tmp]), returned=True)
                                # 2D in-pixel charge
                                stat_in_pixel_charge_2d_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=in_pixel_intersection_x_local[select_valid_tracks_efficiency_region & select_valid_hit], y=in_pixel_intersection_y_local[select_valid_tracks_efficiency_region & select_valid_hit], values=charge[select_valid_tracks_efficiency_region & select_valid_hit], statistic='mean', bins=hist_in_pixel_2d_edges)
                                stat_in_pixel_charge_2d_hist_tmp = np.nan_to_num(stat_in_pixel_charge_2d_hist_tmp)
                                stat_in_pixel_charge_2d_hists[region_index], _ = np.ma.average(a=np.stack([stat_in_pixel_charge_2d_hists[region_index], stat_in_pixel_charge_2d_hist_tmp]), axis=0, weights=np.stack([count_in_pixel_tracks_with_hit_2d_hists[region_index], count_in_pixel_tracks_with_hit_2d_hist_tmp]), returned=True)
                                # 2D in-pixel frame
                                stat_in_pixel_frame_2d_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=in_pixel_intersection_x_local[select_valid_tracks_efficiency_region & select_valid_hit], y=in_pixel_intersection_y_local[select_valid_tracks_efficiency_region & select_valid_hit], values=frame[select_valid_tracks_efficiency_region & select_valid_hit], statistic='mean', bins=hist_in_pixel_2d_edges)
                                stat_in_pixel_frame_2d_hist_tmp = np.nan_to_num(stat_in_pixel_frame_2d_hist_tmp)
                                stat_in_pixel_frame_2d_hists[region_index], _ = np.ma.average(a=np.stack([stat_in_pixel_frame_2d_hists[region_index], stat_in_pixel_frame_2d_hist_tmp]), axis=0, weights=np.stack([count_in_pixel_tracks_with_hit_2d_hists[region_index], count_in_pixel_tracks_with_hit_2d_hist_tmp]), returned=True)
                                # 2D in-pixel mean cluster size
                                stat_in_pixel_cluster_size_2d_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=in_pixel_intersection_x_local[select_valid_tracks_efficiency_region & select_valid_hit], y=in_pixel_intersection_y_local[select_valid_tracks_efficiency_region & select_valid_hit], values=cluster_size[select_valid_tracks_efficiency_region & select_valid_hit], statistic='mean', bins=hist_in_pixel_2d_edges)
                                stat_in_pixel_cluster_size_2d_hist_tmp = np.nan_to_num(stat_in_pixel_cluster_size_2d_hist_tmp)
                                stat_in_pixel_cluster_size_2d_hists[region_index], _ = np.ma.average(a=np.stack([stat_in_pixel_cluster_size_2d_hists[region_index], stat_in_pixel_cluster_size_2d_hist_tmp]), axis=0, weights=np.stack([count_in_pixel_tracks_with_hit_2d_hists[region_index], count_in_pixel_tracks_with_hit_2d_hist_tmp]), returned=True)
                                # 2D in-pixel cluster shape
                                cluster_shape_mod_tmp = cluster_shape[select_valid_tracks_efficiency_region & select_valid_hit & select_small_cluster_sizes].copy()
                                for i, cluster_shape_value in enumerate(efficiency_regions_analyze_cluster_shapes):
                                    cluster_shape_mod_tmp[cluster_shape_mod_tmp == cluster_shape_value] = i
                                count_in_pixel_cluster_shape_2d_hists[region_index] += stats.binned_statistic_dd(sample=[in_pixel_intersection_x_local[select_valid_tracks_efficiency_region & select_valid_hit & select_small_cluster_sizes], in_pixel_intersection_y_local[select_valid_tracks_efficiency_region & select_valid_hit & select_small_cluster_sizes], cluster_shape_mod_tmp], values=None, statistic='count', bins=hist_in_pixel_cluster_shape_edges)[0]
                                # updated last:
                                # 2D in-pixel tracks with valid hit
                                count_in_pixel_tracks_with_hit_2d_hists[region_index] += count_in_pixel_tracks_with_hit_2d_hist_tmp

                    # Histograms for per-pixel efficiency
                    # Pixel tracks
                    _, closest_indices = pixel_center_extended_kd_tree.query(np.column_stack((intersection_x_local, intersection_y_local)))
                    count_tracks_pixel_hist += np.bincount(closest_indices, minlength=pixel_center_data.shape[0])[:pixel_center_data.shape[0]]
                    # Pixel tracks with valid hit
                    _, closest_indices_with_hit = pixel_center_extended_kd_tree.query(np.column_stack((intersection_x_local[select_valid_hit], intersection_y_local[select_valid_hit])))
                    count_tracks_with_hit_pixel_hist += np.bincount(closest_indices_with_hit, minlength=pixel_center_data.shape[0])[:pixel_center_data.shape[0]]

                    # Check if DUT has valid hits
                    if initialize:
                        initialize = False
                        # 2D tracks
                        count_tracks_2d_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=None, statistic='count', bins=hist_2d_edges)
                        if np.any(select_valid_hit):  # Check for valid hits
                            # 2D hits
                            count_hits_2d_hist, _, _, _ = stats.binned_statistic_2d(x=hit_x_local[select_valid_hit], y=hit_y_local[select_valid_hit], values=None, statistic='count', bins=hist_2d_edges)
                            # 2D tracks with valid hit
                            count_tracks_with_hit_2d_hist = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=None, statistic='count', bins=hist_2d_edges)[0]
                            # 2D x residuals
                            stat_2d_x_residuals_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=x_residuals[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_x_residuals_hist = np.nan_to_num(stat_2d_x_residuals_hist)
                            # count_2d_x_residuals_hist = count_tracks_with_hit_2d_hist_tmp
                            # 2D y residuals
                            stat_2d_y_residuals_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=y_residuals[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_y_residuals_hist = np.nan_to_num(stat_2d_y_residuals_hist)
                            # count_2d_y_residuals_hist = count_tracks_with_hit_2d_hist_tmp
                            # 2D residuals
                            stat_2d_residuals_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=distance_local[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_residuals_hist = np.nan_to_num(stat_2d_residuals_hist)
                            # count_2d_residuals_hist = count_tracks_with_hit_2d_hist_tmp
                            # 2D charge
                            stat_2d_charge_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=charge[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_charge_hist = np.nan_to_num(stat_2d_charge_hist)
                            # count_2d_charge_hist = count_tracks_with_hit_2d_hist_tmp
                            # 1D charge
                            count_1d_charge_hist = np.bincount(charge[select_valid_hit].astype(np.int64))
                            # 2D frame
                            stat_2d_frame_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=frame[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_frame_hist = np.nan_to_num(stat_2d_frame_hist)
                            # count_2d_frame_hist = count_tracks_with_hit_2d_hist_tmp
                            # 1D frame
                            count_1d_frame_hist = np.bincount(frame[select_valid_hit])
                            # 2D mean cluster size
                            stat_2d_cluster_size_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=cluster_size[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_cluster_size_hist = np.nan_to_num(stat_2d_cluster_size_hist)
                            # 1D total track angle
                            local_total_mean = np.nanmean(total_angles_local[select_valid_hit])
                            local_total_std = np.nanstd(total_angles_local[select_valid_hit])
                            count_1d_total_angle_hist, count_1d_total_angle_hist_edges = np.histogram(total_angles_local[select_valid_hit], bins=n_bins_track_angle, range=(local_total_mean - 5 * local_total_std, local_total_mean + 5 * local_total_std))
                            # 2D mean total angle
                            stat_2d_total_angle_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=total_angles_local[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_total_angle_hist = np.nan_to_num(stat_2d_total_angle_hist)
                            # 1D alpha track angle
                            local_alpha_mean = np.nanmean(alpha_angles_local[select_valid_hit])
                            local_alpha_std = np.nanstd(alpha_angles_local[select_valid_hit])
                            count_1d_alpha_angle_hist, count_1d_alpha_angle_hist_edges = np.histogram(alpha_angles_local[select_valid_hit], bins=n_bins_track_angle, range=(local_alpha_mean - 5 * local_alpha_std, local_alpha_mean + 5 * local_alpha_std))
                            # 2D mean alpha track angle
                            stat_2d_alpha_angle_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=alpha_angles_local[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_alpha_angle_hist = np.nan_to_num(stat_2d_alpha_angle_hist)
                            # 1D beta track angle
                            local_beta_mean = np.nanmean(beta_angles_local[select_valid_hit])
                            local_beta_std = np.nanstd(beta_angles_local[select_valid_hit])
                            count_1d_beta_angle_hist, count_1d_beta_angle_hist_edges = np.histogram(beta_angles_local[select_valid_hit], bins=n_bins_track_angle, range=(local_beta_mean - 5 * local_beta_std, local_beta_mean + 5 * local_beta_std))
                            # 2D mean beta track angle
                            stat_2d_beta_angle_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=beta_angles_local[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_beta_angle_hist = np.nan_to_num(stat_2d_beta_angle_hist)
                    else:
                        # 2D tracks
                        count_tracks_2d_hist_tmp = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=None, statistic='count', bins=hist_2d_edges)[0]
                        if np.any(select_valid_hit):  # Check for valid hits
                            count_tracks_with_hit_2d_hist_tmp = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=None, statistic='count', bins=hist_2d_edges)[0]
                            # 2D hits
                            count_hits_2d_hist += stats.binned_statistic_2d(x=hit_x_local[select_valid_hit], y=hit_y_local[select_valid_hit], values=None, statistic='count', bins=hist_2d_edges)[0]
                            # 2D x residuals
                            stat_2d_x_residuals_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=x_residuals[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_x_residuals_hist_tmp = np.nan_to_num(stat_2d_x_residuals_hist_tmp)
                            stat_2d_x_residuals_hist, _ = np.ma.average(a=np.stack([stat_2d_x_residuals_hist, stat_2d_x_residuals_hist_tmp]), axis=0, weights=np.stack([count_tracks_with_hit_2d_hist, count_tracks_with_hit_2d_hist_tmp]), returned=True)
                            # 2D y residuals
                            stat_2d_y_residuals_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=y_residuals[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_y_residuals_hist_tmp = np.nan_to_num(stat_2d_y_residuals_hist_tmp)
                            stat_2d_y_residuals_hist, _ = np.ma.average(a=np.stack([stat_2d_y_residuals_hist, stat_2d_y_residuals_hist_tmp]), axis=0, weights=np.stack([count_tracks_with_hit_2d_hist, count_tracks_with_hit_2d_hist_tmp]), returned=True)
                            # 2D residuals
                            stat_2d_residuals_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=distance_local[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_residuals_hist_tmp = np.nan_to_num(stat_2d_residuals_hist_tmp)
                            stat_2d_residuals_hist, _ = np.ma.average(a=np.stack([stat_2d_residuals_hist, stat_2d_residuals_hist_tmp]), axis=0, weights=np.stack([count_tracks_with_hit_2d_hist, count_tracks_with_hit_2d_hist_tmp]), returned=True)
                            # 2D charge
                            stat_2d_charge_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=charge[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_charge_hist_tmp = np.nan_to_num(stat_2d_charge_hist_tmp)
                            stat_2d_charge_hist, _ = np.ma.average(a=np.stack([stat_2d_charge_hist, stat_2d_charge_hist_tmp]), axis=0, weights=np.stack([count_tracks_with_hit_2d_hist, count_tracks_with_hit_2d_hist_tmp]), returned=True)
                            # 1D charge
                            count_1d_charge_hist_tmp = np.bincount(charge[select_valid_hit].astype(np.int64))
                            if count_1d_charge_hist_tmp.size > count_1d_charge_hist.size:
                                count_1d_charge_hist.resize(count_1d_charge_hist_tmp.size)
                            else:
                                count_1d_charge_hist_tmp.resize(count_1d_charge_hist.size)
                            count_1d_charge_hist += count_1d_charge_hist_tmp
                            # 2D frame
                            stat_2d_frame_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=frame[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_frame_hist_tmp = np.nan_to_num(stat_2d_frame_hist_tmp)
                            stat_2d_frame_hist, count_2d_frame_hist = np.ma.average(a=np.stack([stat_2d_frame_hist, stat_2d_frame_hist_tmp]), axis=0, weights=np.stack([count_tracks_with_hit_2d_hist, count_tracks_with_hit_2d_hist_tmp]), returned=True)
                            # 1D frame
                            count_1d_frame_hist_tmp = np.bincount(frame[select_valid_hit])
                            if count_1d_frame_hist_tmp.size > count_1d_frame_hist.size:
                                count_1d_frame_hist.resize(count_1d_frame_hist_tmp.size)
                            else:
                                count_1d_frame_hist_tmp.resize(count_1d_frame_hist.size)
                            count_1d_frame_hist += count_1d_frame_hist_tmp
                            # 2D mean cluster size
                            stat_2d_cluster_size_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit & select_small_cluster_sizes], y=intersection_y_local[select_valid_hit & select_small_cluster_sizes], values=cluster_size[select_valid_hit & select_small_cluster_sizes], statistic='mean', bins=hist_2d_edges)
                            stat_2d_cluster_size_hist_tmp = np.nan_to_num(stat_2d_cluster_size_hist_tmp)
                            stat_2d_cluster_size_hist, _ = np.ma.average(a=np.stack([stat_2d_cluster_size_hist, stat_2d_cluster_size_hist_tmp]), axis=0, weights=np.stack([count_tracks_with_hit_2d_hist, count_tracks_with_hit_2d_hist_tmp]), returned=True)
                            # 1D total track anlge
                            count_1d_total_angle_hist_tmp = np.histogram(total_angles_local[select_valid_hit], bins=count_1d_total_angle_hist_edges)[0]
                            if count_1d_total_angle_hist_tmp.size > count_1d_total_angle_hist.size:
                                count_1d_total_angle_hist.resize(count_1d_total_angle_hist_tmp.size)
                            else:
                                count_1d_total_angle_hist_tmp.resize(count_1d_total_angle_hist.size)
                            count_1d_total_angle_hist += count_1d_total_angle_hist_tmp
                            # 2D total track angle
                            stat_2d_total_angle_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=total_angles_local[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_total_angle_hist_tmp = np.nan_to_num(stat_2d_total_angle_hist_tmp)
                            stat_2d_total_angle_hist, count_2d_total_angle_hist = np.ma.average(a=np.stack([stat_2d_total_angle_hist, stat_2d_total_angle_hist_tmp]), axis=0, weights=np.stack([count_tracks_with_hit_2d_hist, count_tracks_with_hit_2d_hist_tmp]), returned=True)
                            # 1D alpha track angle
                            count_1d_alpha_angle_hist_tmp = np.histogram(alpha_angles_local[select_valid_hit], bins=count_1d_alpha_angle_hist_edges)[0]
                            if count_1d_alpha_angle_hist_tmp.size > count_1d_alpha_angle_hist.size:
                                count_1d_alpha_angle_hist.resize(count_1d_alpha_angle_hist_tmp.size)
                            else:
                                count_1d_alpha_angle_hist_tmp.resize(count_1d_alpha_angle_hist.size)
                            count_1d_alpha_angle_hist += count_1d_alpha_angle_hist_tmp
                            # 2D alpha track angle
                            stat_2d_alpha_angle_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=alpha_angles_local[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_alpha_angle_hist_tmp = np.nan_to_num(stat_2d_alpha_angle_hist_tmp)
                            stat_2d_alpha_angle_hist, count_2d_alpha_angle_hist = np.ma.average(a=np.stack([stat_2d_alpha_angle_hist, stat_2d_alpha_angle_hist_tmp]), axis=0, weights=np.stack([count_tracks_with_hit_2d_hist, count_tracks_with_hit_2d_hist_tmp]), returned=True)
                            # 1D beta track angle
                            count_1d_beta_angle_hist_tmp = np.histogram(beta_angles_local[select_valid_hit], bins=count_1d_beta_angle_hist_edges)[0]
                            if count_1d_beta_angle_hist_tmp.size > count_1d_beta_angle_hist.size:
                                count_1d_beta_angle_hist.resize(count_1d_beta_angle_hist_tmp.size)
                            else:
                                count_1d_beta_angle_hist_tmp.resize(count_1d_beta_angle_hist.size)
                            count_1d_beta_angle_hist += count_1d_beta_angle_hist_tmp
                            # 2D beta track angle
                            stat_2d_beta_angle_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=beta_angles_local[select_valid_hit], statistic='mean', bins=hist_2d_edges)
                            stat_2d_beta_angle_hist_tmp = np.nan_to_num(stat_2d_beta_angle_hist_tmp)
                            stat_2d_beta_angle_hist, count_2d_beta_angle_hist = np.ma.average(a=np.stack([stat_2d_beta_angle_hist, stat_2d_beta_angle_hist_tmp]), axis=0, weights=np.stack([count_tracks_with_hit_2d_hist, count_tracks_with_hit_2d_hist_tmp]), returned=True)
                            # updated last:
                            # 2D tracks with valid hit
                            count_tracks_with_hit_2d_hist += count_tracks_with_hit_2d_hist_tmp
                        count_tracks_2d_hist += count_tracks_2d_hist_tmp

                    if in_cluster_file_h5:
                        # get 2D indices (without overflow bins)
                        binnumber = stats.binned_statistic_2d(x=intersection_x_local[select_valid_hit], y=intersection_y_local[select_valid_hit], values=None, statistic='count', bins=hist_2d_edges, expand_binnumbers=True).binnumber
                        # get 1D index, and remove 1st overflow bin
                        tracks_with_hits_to_bin_center_index = np.ravel_multi_index(binnumber - 1, (hist_2d_x_n_bins, hist_2d_y_n_bins))
                        # event_number_id_array = np.repeat(np.column_stack((tracks_chunk['event_number'][select_valid_hit], tracks_chunk['cluster_id_dut_%d' % actual_dut_index][select_valid_hit])), tracks_chunk['n_hits_dut_%d' % actual_dut_index][select_valid_hit])
                        for actual_cluster_hits_dut, start_index_cluster_hits in analysis_utils.data_aligned_at_events(in_cluster_file_h5.root.ClusterHits, start_index=start_index_cluster_hits, start_event_number=tracks_chunk['event_number'][0], stop_event_number=tracks_chunk['event_number'][-1] + 1, chunk_size=chunk_size, fail_on_missing_events=False):
                            cluster_hits_event_numbers_cluster_ids = actual_cluster_hits_dut[['event_number', 'cluster_ID']]
                            selected_event_numbers_clusters_ids = np.array(tracks_chunk[['event_number', 'cluster_ID_dut_%d' % actual_dut_index]][select_valid_hit], dtype=cluster_hits_event_numbers_cluster_ids.dtype)
                            selected_cluster_hits, selected_indices = analysis_utils.in1d_index(ar1=cluster_hits_event_numbers_cluster_ids, ar2=selected_event_numbers_clusters_ids, fill_invalid=None, assume_sorted=False)
                            # get the cluster hits
                            actual_cluster_hits = actual_cluster_hits_dut[selected_cluster_hits]
                            # actual_tracks = np.repeat(tracks_chunk[select_valid_hit][selected_indices], tracks_chunk['n_hits_dut_%d' % actual_dut_index][select_valid_hit][selected_indices])
                            # expand size of index array so that the length matches the cluster array
                            actual_bin_center_indices = np.repeat(tracks_with_hits_to_bin_center_index[selected_indices], tracks_chunk['n_hits_dut_%d' % actual_dut_index][select_valid_hit][selected_indices])
                            actual_bin_center_col_row_pair = np.column_stack(np.unravel_index(actual_bin_center_indices, dims=(count_pixel_hits_2d_hist.shape[0], count_pixel_hits_2d_hist.shape[1])))
                            # get the corresponding pixel center for each track
                            actual_pixel_center_indices = bin_center_to_pixel_center[actual_bin_center_indices]
                            actual_pixel_center_col_row_pair = np.column_stack(np.unravel_index(indices=actual_pixel_center_indices, dims=(actual_dut.n_columns, actual_dut.n_rows)))
                            # get the corresponding cluster hits
                            actual_cluster_hits_col_row_pair = np.column_stack((actual_cluster_hits['column'] - 1, actual_cluster_hits['row'] - 1))
                            # calculate the relative index
                            actual_col_index = actual_cluster_hits_col_row_pair[:, 0] - actual_pixel_center_col_row_pair[:, 0] + int(count_pixel_hits_2d_hist.shape[2] / 2)
                            actual_row_index = actual_cluster_hits_col_row_pair[:, 1] - actual_pixel_center_col_row_pair[:, 1] + int(count_pixel_hits_2d_hist.shape[3] / 2)
                            # select hits which are inside the array
                            hits_select = actual_col_index >= 0
                            hits_select &= actual_col_index < count_pixel_hits_2d_hist.shape[2]
                            hits_select &= actual_row_index >= 0
                            hits_select &= actual_row_index < count_pixel_hits_2d_hist.shape[3]
                            # check for significant number of hits outside of the array
                            if np.count_nonzero(~hits_select) / hits_select.shape[0] > 1e-3:
                                logging.warning('Consider increasing shape of count_pixel_hits_2d_hist')
                            ravel_indices = np.ravel_multi_index((actual_bin_center_col_row_pair[hits_select, 0], actual_bin_center_col_row_pair[hits_select, 1], actual_col_index[hits_select], actual_row_index[hits_select]), dims=count_pixel_hits_2d_hist.shape)
                            unique_indices, unique_indices_count = np.unique(ravel_indices, return_counts=True)
                            count_pixel_hits_2d_hist.reshape(-1)[unique_indices] += unique_indices_count

                    pbar.update(tracks_chunk.shape[0])
                pbar.close()

                if np.all(count_tracks_2d_hist == 0):
                    logging.warning('No tracks found for DUT%d, cannot calculate efficiency.', actual_dut_index)
                    continue

                # if np.any(select_valid_hit):
                #     # Calculate mean efficiency without any binning per chunk
                #     eff_chunk, _, _ = analysis_utils.get_mean_efficiency(
                #         array_pass=count_tracks_with_hit_2d_hist_chunk,
                #         array_total=count_tracks_2d_hist_chunk)
                # else:
                #     eff_chunk = 0.0

                # Append chunk stats
                chunk_indices.append(index_chunk)
                if efficiency_regions_dut is not None:
                    for region_index, region in enumerate(efficiency_regions_dut):
                        efficiency_regions_chunk_stats[region_index].append(efficiency_regions_efficiency_chunk[region_index])

                # Calculate efficiency
                stat_2d_efficiency_hist = np.full_like(count_tracks_2d_hist, fill_value=np.nan, dtype=np.float64)
                stat_2d_efficiency_hist[count_tracks_2d_hist != 0] = count_tracks_with_hit_2d_hist[count_tracks_2d_hist != 0].astype(np.float64) / count_tracks_2d_hist[count_tracks_2d_hist != 0].astype(np.float64) * 100.0
                # Masking low stat
                stat_2d_efficiency_hist = np.ma.array(stat_2d_efficiency_hist, mask=count_tracks_2d_hist < minimum_track_density)
                stat_2d_residuals_hist = np.ma.array(stat_2d_residuals_hist, mask=count_tracks_2d_hist < minimum_track_density)
                stat_2d_charge_hist = np.ma.array(stat_2d_charge_hist, mask=count_tracks_2d_hist < minimum_track_density)
                stat_2d_frame_hist = np.ma.array(stat_2d_frame_hist, mask=count_tracks_2d_hist < minimum_track_density)
                stat_2d_cluster_size_hist = np.ma.array(stat_2d_cluster_size_hist, mask=count_tracks_2d_hist < minimum_track_density)
                stat_2d_total_angle_hist = np.ma.array(stat_2d_total_angle_hist, mask=count_tracks_2d_hist < minimum_track_density)
                stat_2d_alpha_angle_hist = np.ma.array(stat_2d_alpha_angle_hist, mask=count_tracks_2d_hist < minimum_track_density)
                stat_2d_beta_angle_hist = np.ma.array(stat_2d_beta_angle_hist, mask=count_tracks_2d_hist < minimum_track_density)

                if efficiency_regions_dut is not None:
                    efficiency_regions_mask = []
                    efficiency_regions_stat_in_pixel_efficiency_2d_hist = []

                    for region_index, region in enumerate(efficiency_regions_dut):
                        efficiency_regions_stat_pixel_efficiency_hist[region_index] = np.full_like(efficiency_regions_count_tracks_pixel_hist[region_index], fill_value=np.nan, dtype=np.float64)
                        efficiency_regions_stat_pixel_efficiency_hist[region_index][efficiency_regions_count_tracks_pixel_hist[region_index] != 0] = efficiency_regions_count_tracks_with_hit_pixel_hist[region_index][efficiency_regions_count_tracks_pixel_hist[region_index] != 0].astype(np.float64) / efficiency_regions_count_tracks_pixel_hist[region_index][efficiency_regions_count_tracks_pixel_hist[region_index] != 0].astype(np.float64) * 100.0
                        efficiency_regions_stat_pixel_efficiency_hist[region_index] = np.ma.array(efficiency_regions_stat_pixel_efficiency_hist[region_index], mask=efficiency_regions_count_tracks_pixel_hist[region_index] < minimum_track_density)

                        # Calculate mask
                        efficiency_regions_mask.append(count_in_pixel_tracks_2d_hists[region_index] < minimum_track_density)
                        # Calculate in-pixel efficiency
                        stat_in_pixel_efficiency_2d_hist_tmp = np.full_like(count_in_pixel_tracks_2d_hists[region_index], fill_value=np.nan, dtype=np.float64)
                        stat_in_pixel_efficiency_2d_hist_tmp[count_in_pixel_tracks_2d_hists[region_index] != 0] = count_in_pixel_tracks_with_hit_2d_hists[region_index][count_in_pixel_tracks_2d_hists[region_index] != 0].astype(np.float64) / count_in_pixel_tracks_2d_hists[region_index][count_in_pixel_tracks_2d_hists[region_index] != 0].astype(np.float64) * 100.0
                        efficiency_regions_stat_in_pixel_efficiency_2d_hist.append(stat_in_pixel_efficiency_2d_hist_tmp)
                        # Calculate most probable cluster shape
                        stat_in_pixel_cluster_shape_2d_hists.append(np.argmax(count_in_pixel_cluster_shape_2d_hists[region_index], axis=2))

                    # extending plot area for in-pixel plots
                    if extend_in_pixel_area is not None and extend_in_pixel_area[0] is not None:
                        dut_hits_in_pixel_x_extend_n_bins = int(math.ceil(extend_in_pixel_area[0] / in_pixel_resolution[0]))
                        dut_hist_in_pixel_x_extent_area = dut_hits_in_pixel_x_extend_n_bins * in_pixel_resolution[0]
                        in_pixel_hist_extent[0] -= dut_hist_in_pixel_x_extent_area
                        in_pixel_hist_extent[1] += dut_hist_in_pixel_x_extent_area
                        in_pixel_plot_range[0][0] -= extend_in_pixel_area[0]
                        in_pixel_plot_range[0][1] += extend_in_pixel_area[0]
                        for region_index, region in enumerate(efficiency_regions_dut):
                            efficiency_regions_mask[region_index] = np.pad(efficiency_regions_mask[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                            count_in_pixel_hits_2d_hists[region_index] = np.pad(count_in_pixel_hits_2d_hists[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                            count_in_pixel_tracks_2d_hists[region_index] = np.pad(count_in_pixel_tracks_2d_hists[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                            count_in_pixel_tracks_with_hit_2d_hists[region_index] = np.pad(count_in_pixel_tracks_with_hit_2d_hists[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                            efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index] = np.pad(efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                            stat_in_pixel_x_residuals_2d_hists[region_index] = np.pad(stat_in_pixel_x_residuals_2d_hists[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                            stat_in_pixel_y_residuals_2d_hists[region_index] = np.pad(stat_in_pixel_y_residuals_2d_hists[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                            stat_in_pixel_residuals_2d_hists[region_index] = np.pad(stat_in_pixel_residuals_2d_hists[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                            stat_in_pixel_charge_2d_hists[region_index] = np.pad(stat_in_pixel_charge_2d_hists[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                            stat_in_pixel_frame_2d_hists[region_index] = np.pad(stat_in_pixel_frame_2d_hists[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                            stat_in_pixel_cluster_size_2d_hists[region_index] = np.pad(stat_in_pixel_cluster_size_2d_hists[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                            count_in_pixel_cluster_shape_2d_hists[region_index] = np.pad(count_in_pixel_cluster_shape_2d_hists[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0), (0, 0)), 'wrap')
                            stat_in_pixel_cluster_shape_2d_hists[region_index] = np.pad(stat_in_pixel_cluster_shape_2d_hists[region_index], ((dut_hits_in_pixel_x_extend_n_bins, dut_hits_in_pixel_x_extend_n_bins), (0, 0)), 'wrap')
                    if extend_in_pixel_area is not None and extend_in_pixel_area[1] is not None:
                        dut_hits_in_pixel_y_extend_n_bins = int(math.ceil(extend_in_pixel_area[1] / in_pixel_resolution[1]))
                        dut_hist_in_pixel_y_extent_area = dut_hits_in_pixel_y_extend_n_bins * in_pixel_resolution[1]
                        in_pixel_hist_extent[2] -= dut_hist_in_pixel_y_extent_area
                        in_pixel_hist_extent[3] += dut_hist_in_pixel_y_extent_area
                        in_pixel_plot_range[1][0] -= extend_in_pixel_area[1]
                        in_pixel_plot_range[1][1] += extend_in_pixel_area[1]
                        for region_index, region in enumerate(efficiency_regions_dut):
                            efficiency_regions_mask[region_index] = np.pad(efficiency_regions_mask[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                            count_in_pixel_hits_2d_hists[region_index] = np.pad(count_in_pixel_hits_2d_hists[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                            count_in_pixel_tracks_2d_hists[region_index] = np.pad(count_in_pixel_tracks_2d_hists[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                            count_in_pixel_tracks_with_hit_2d_hists[region_index] = np.pad(count_in_pixel_tracks_with_hit_2d_hists[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                            efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index] = np.pad(efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                            stat_in_pixel_x_residuals_2d_hists[region_index] = np.pad(stat_in_pixel_x_residuals_2d_hists[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                            stat_in_pixel_y_residuals_2d_hists[region_index] = np.pad(stat_in_pixel_y_residuals_2d_hists[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                            stat_in_pixel_residuals_2d_hists[region_index] = np.pad(stat_in_pixel_residuals_2d_hists[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                            stat_in_pixel_charge_2d_hists[region_index] = np.pad(stat_in_pixel_charge_2d_hists[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                            stat_in_pixel_frame_2d_hists[region_index] = np.pad(stat_in_pixel_frame_2d_hists[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                            stat_in_pixel_cluster_size_2d_hists[region_index] = np.pad(stat_in_pixel_cluster_size_2d_hists[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                            count_in_pixel_cluster_shape_2d_hists[region_index] = np.pad(count_in_pixel_cluster_shape_2d_hists[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins), (0, 0)), 'wrap')
                            stat_in_pixel_cluster_shape_2d_hists[region_index] = np.pad(stat_in_pixel_cluster_shape_2d_hists[region_index], ((0, 0), (dut_hits_in_pixel_y_extend_n_bins, dut_hits_in_pixel_y_extend_n_bins)), 'wrap')
                    # Masking low stat
                    for region_index, region in enumerate(efficiency_regions_dut):
                        efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index] = np.ma.array(efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index], mask=efficiency_regions_mask[region_index])
                        stat_in_pixel_x_residuals_2d_hists[region_index] = np.ma.array(stat_in_pixel_x_residuals_2d_hists[region_index], mask=efficiency_regions_mask[region_index])
                        stat_in_pixel_y_residuals_2d_hists[region_index] = np.ma.array(stat_in_pixel_y_residuals_2d_hists[region_index], mask=efficiency_regions_mask[region_index])
                        stat_in_pixel_residuals_2d_hists[region_index] = np.ma.array(stat_in_pixel_residuals_2d_hists[region_index], mask=efficiency_regions_mask[region_index])
                        stat_in_pixel_charge_2d_hists[region_index] = np.ma.array(stat_in_pixel_charge_2d_hists[region_index], mask=efficiency_regions_mask[region_index])
                        stat_in_pixel_frame_2d_hists[region_index] = np.ma.array(stat_in_pixel_frame_2d_hists[region_index], mask=efficiency_regions_mask[region_index])
                        stat_in_pixel_cluster_size_2d_hists[region_index] = np.ma.array(stat_in_pixel_cluster_size_2d_hists[region_index], mask=efficiency_regions_mask[region_index])
                        stat_in_pixel_cluster_shape_2d_hists[region_index] = np.ma.array(stat_in_pixel_cluster_shape_2d_hists[region_index], mask=efficiency_regions_mask[region_index])
                else:
                    count_in_pixel_hits_2d_hists = None
                    count_in_pixel_tracks_2d_hists = None
                    count_in_pixel_tracks_with_hit_2d_hists = None
                    efficiency_regions_stat_in_pixel_efficiency_2d_hist = None
                    stat_in_pixel_x_residuals_2d_hists = None
                    stat_in_pixel_y_residuals_2d_hists = None
                    stat_in_pixel_residuals_2d_hists = None
                    stat_in_pixel_charge_2d_hists = None
                    stat_in_pixel_frame_2d_hists = None
                    stat_in_pixel_cluster_size_2d_hists = None
                    count_in_pixel_cluster_shape_2d_hists = None
                    stat_in_pixel_cluster_shape_2d_hists = None
                    in_pixel_hist_extent = None
                    in_pixel_plot_range = None

                stat_pixel_efficiency_hist = np.full_like(count_tracks_pixel_hist, fill_value=np.nan, dtype=np.float64)
                stat_pixel_efficiency_hist[count_tracks_pixel_hist != 0] = count_tracks_with_hit_pixel_hist[count_tracks_pixel_hist != 0].astype(np.float64) / count_tracks_pixel_hist[count_tracks_pixel_hist != 0].astype(np.float64) * 100.0
                stat_pixel_efficiency_hist = np.ma.array(stat_pixel_efficiency_hist, mask=count_tracks_pixel_hist < minimum_track_density)

                # Calculate mean efficiency without any binning
                eff, eff_err_min, eff_err_pl = analysis_utils.get_mean_efficiency(
                    array_pass=count_tracks_with_hit_2d_hist,
                    array_total=count_tracks_2d_hist)
                logging.info('Selected tracks / total tracks: %d / %d', count_tracks_with_hit_2d_hist.sum(), count_tracks_2d_hist.sum())
                logging.info('Efficiency = %.2f (+%.2f / %.2f)%%' % (eff * 100.0, eff_err_pl * 100.0, eff_err_min * 100.0))
                if efficiency_regions_dut is not None:
                    for region_index, efficiency in enumerate(efficiency_regions_efficiency):
                        logging.info('Efficiency for region %d%s= %.2f%%' % (region_index + 1, (" (" + efficiency_regions_names_dut[region_index] + ")") if efficiency_regions_names_dut[region_index] else "", efficiency * 100.0))
                        # resize so that all histograms have the same size
                        if count_1d_charge_hist.size > efficiency_regions_count_1d_charge_hist[region_index].size:
                            efficiency_regions_count_1d_charge_hist[region_index].resize(count_1d_charge_hist.size)
                        if count_1d_frame_hist.size > efficiency_regions_count_1d_frame_hist[region_index].size:
                            efficiency_regions_count_1d_frame_hist[region_index].resize(count_1d_frame_hist.size)
                        mean_charge = analysis_utils.get_mean_from_histogram(efficiency_regions_count_1d_charge_hist[region_index], range(count_1d_charge_hist.size))
                        logging.info('Mean charge for region %d%s = %.2f' % (region_index + 1, (" (" + efficiency_regions_names_dut[region_index] + ")") if efficiency_regions_names_dut[region_index] else "", mean_charge))

                if not np.any(stat_2d_efficiency_hist):
                    raise RuntimeError('All efficiencies for DUT%d are zero, consider changing cut values!', actual_dut_index)

                plot_utils.efficiency_plots(
                    telescope=telescope,
                    hist_2d_edges=hist_2d_edges,
                    count_hits_2d_hist=count_hits_2d_hist,
                    count_tracks_2d_hist=count_tracks_2d_hist,
                    count_tracks_with_hit_2d_hist=count_tracks_with_hit_2d_hist,
                    stat_2d_x_residuals_hist=stat_2d_x_residuals_hist,
                    stat_2d_y_residuals_hist=stat_2d_y_residuals_hist,
                    stat_2d_residuals_hist=stat_2d_residuals_hist,
                    count_1d_charge_hist=count_1d_charge_hist,
                    stat_2d_charge_hist=stat_2d_charge_hist,
                    count_1d_frame_hist=count_1d_frame_hist,
                    stat_2d_frame_hist=stat_2d_frame_hist,
                    stat_2d_cluster_size_hist=stat_2d_cluster_size_hist,
                    count_1d_total_angle_hist=count_1d_total_angle_hist,
                    count_1d_total_angle_hist_edges=count_1d_total_angle_hist_edges,
                    stat_2d_total_angle_hist=stat_2d_total_angle_hist,
                    count_1d_alpha_angle_hist=count_1d_alpha_angle_hist,
                    count_1d_alpha_angle_hist_edges=count_1d_alpha_angle_hist_edges,
                    stat_2d_alpha_angle_hist=stat_2d_alpha_angle_hist,
                    count_1d_beta_angle_hist=count_1d_beta_angle_hist,
                    count_1d_beta_angle_hist_edges=count_1d_beta_angle_hist_edges,
                    stat_2d_beta_angle_hist=stat_2d_beta_angle_hist,
                    stat_2d_efficiency_hist=stat_2d_efficiency_hist,
                    stat_pixel_efficiency_hist=stat_pixel_efficiency_hist,
                    count_pixel_hits_2d_hist=count_pixel_hits_2d_hist,
                    efficiency=[eff, eff_err_pl, eff_err_min],
                    actual_dut_index=actual_dut_index,
                    dut_extent=dut_extent,
                    hist_extent=hist_extent,
                    plot_range=plot_range,
                    efficiency_regions=efficiency_regions_dut,
                    efficiency_regions_names=efficiency_regions_names_dut,
                    efficiency_regions_efficiency=efficiency_regions_efficiency,
                    efficiency_regions_count_1d_charge_hist=efficiency_regions_count_1d_charge_hist,
                    efficiency_regions_count_1d_frame_hist=efficiency_regions_count_1d_frame_hist,
                    efficiency_regions_count_1d_cluster_size_hist=efficiency_regions_count_1d_cluster_size_hist,
                    efficiency_regions_count_1d_total_angle_hist=efficiency_regions_count_1d_total_angle_hist,
                    efficiency_regions_count_1d_total_angle_hist_edges=efficiency_regions_count_1d_total_angle_hist_edges,
                    efficiency_regions_count_1d_alpha_angle_hist=efficiency_regions_count_1d_alpha_angle_hist,
                    efficiency_regions_count_1d_alpha_angle_hist_edges=efficiency_regions_count_1d_alpha_angle_hist_edges,
                    efficiency_regions_count_1d_beta_angle_hist=efficiency_regions_count_1d_beta_angle_hist,
                    efficiency_regions_count_1d_beta_angle_hist_edges=efficiency_regions_count_1d_beta_angle_hist_edges,
                    efficiency_regions_count_1d_cluster_shape_hist=efficiency_regions_count_1d_cluster_shape_hist,
                    efficiency_regions_stat_pixel_efficiency_hist=efficiency_regions_stat_pixel_efficiency_hist,
                    efficiency_regions_count_in_pixel_hits_2d_hist=count_in_pixel_hits_2d_hists,
                    efficiency_regions_count_in_pixel_tracks_2d_hist=count_in_pixel_tracks_2d_hists,
                    efficiency_regions_count_in_pixel_tracks_with_hit_2d_hist=count_in_pixel_tracks_with_hit_2d_hists,
                    efficiency_regions_stat_in_pixel_efficiency_2d_hist=efficiency_regions_stat_in_pixel_efficiency_2d_hist,
                    efficiency_regions_stat_in_pixel_x_residuals_2d_hist=stat_in_pixel_x_residuals_2d_hists,
                    efficiency_regions_stat_in_pixel_y_residuals_2d_hist=stat_in_pixel_y_residuals_2d_hists,
                    efficiency_regions_stat_in_pixel_residuals_2d_hist=stat_in_pixel_residuals_2d_hists,
                    efficiency_regions_stat_in_pixel_charge_2d_hist=stat_in_pixel_charge_2d_hists,
                    efficiency_regions_stat_in_pixel_frame_2d_hist=stat_in_pixel_frame_2d_hists,
                    efficiency_regions_stat_in_pixel_cluster_size_2d_hist=stat_in_pixel_cluster_size_2d_hists,
                    efficiency_regions_count_in_pixel_cluster_shape_2d_hist=count_in_pixel_cluster_shape_2d_hists,
                    efficiency_regions_stat_in_pixel_cluster_shape_2d_hist=stat_in_pixel_cluster_shape_2d_hists,
                    chunk_indices=chunk_indices,
                    efficiency_regions_chunk_stats=efficiency_regions_chunk_stats,
                    efficiency_regions_in_pixel_hist_extent=in_pixel_hist_extent,
                    efficiency_regions_in_pixel_plot_range=in_pixel_plot_range,
                    efficiency_regions_analyze_cluster_shapes=efficiency_regions_analyze_cluster_shapes,
                    mask_zero=True,
                    output_pdf=output_pdf)

                dut_group = out_file_h5.create_group("/", 'DUT%d' % actual_dut_index)

                out_hist_2d_x_edges = out_file_h5.create_carray(
                    where=dut_group,
                    name='hist_2d_x_edges',
                    title='hist_2d_x_edges for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(hist_2d_x_edges.dtype),
                    shape=hist_2d_x_edges.shape)
                out_hist_2d_x_edges[:] = hist_2d_x_edges

                out_hist_2d_y_edges = out_file_h5.create_carray(
                    where=dut_group,
                    name='hist_2d_y_edges',
                    title='hist_2d_y_edges for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(hist_2d_y_edges.dtype),
                    shape=hist_2d_y_edges.shape)
                out_hist_2d_y_edges[:] = hist_2d_y_edges

                out_count_hits_2d_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='count_hits_2d_hist',
                    title='count_hits_2d_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(count_hits_2d_hist.dtype),
                    shape=count_hits_2d_hist.shape)
                out_count_hits_2d_hist[:] = count_hits_2d_hist

                out_count_tracks_2d_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='count_tracks_2d_hist',
                    title='count_tracks_2d_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(count_tracks_2d_hist.dtype),
                    shape=count_tracks_2d_hist.shape)
                out_count_tracks_2d_hist[:] = count_tracks_2d_hist

                out_count_tracks_with_hit_2d_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='count_tracks_with_hit_2d_hist',
                    title='count_tracks_with_hit_2d_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(count_tracks_with_hit_2d_hist.dtype),
                    shape=count_tracks_with_hit_2d_hist.shape)
                out_count_tracks_with_hit_2d_hist[:] = count_tracks_with_hit_2d_hist

                out_stat_2d_x_residuals_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_2d_x_residuals_hist',
                    title='stat_2d_x_residuals_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_2d_x_residuals_hist.dtype),
                    shape=stat_2d_x_residuals_hist.shape)
                out_stat_2d_x_residuals_hist[:] = stat_2d_x_residuals_hist

                out_stat_2d_y_residuals_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_2d_y_residuals_hist',
                    title='stat_2d_y_residuals_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_2d_y_residuals_hist.dtype),
                    shape=stat_2d_y_residuals_hist.shape)
                out_stat_2d_y_residuals_hist[:] = stat_2d_y_residuals_hist

                out_stat_2d_residuals_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_2d_residuals_hist',
                    title='stat_2d_residuals_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_2d_residuals_hist.dtype),
                    shape=stat_2d_residuals_hist.shape)
                out_stat_2d_residuals_hist[:] = stat_2d_residuals_hist

                out_count_1d_charge_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='count_1d_charge_hist',
                    title='count_1d_charge_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(count_1d_charge_hist.dtype),
                    shape=count_1d_charge_hist.shape)
                out_count_1d_charge_hist[:] = count_1d_charge_hist

                out_stat_2d_charge_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_2d_charge_hist',
                    title='stat_2d_charge_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_2d_charge_hist.dtype),
                    shape=stat_2d_charge_hist.shape)
                out_stat_2d_charge_hist[:] = stat_2d_charge_hist

                out_count_1d_frame_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='count_1d_frame_hist',
                    title='count_1d_frame_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(count_1d_frame_hist.dtype),
                    shape=count_1d_frame_hist.shape)
                out_count_1d_frame_hist[:] = count_1d_frame_hist

                out_stat_2d_frame_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_2d_frame_hist',
                    title='stat_2d_frame_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_2d_frame_hist.dtype),
                    shape=stat_2d_frame_hist.shape)
                out_stat_2d_frame_hist[:] = stat_2d_frame_hist

                out_stat_2d_cluster_size_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_2d_cluster_size_hist',
                    title='stat_2d_cluster_size_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_2d_cluster_size_hist.dtype),
                    shape=stat_2d_cluster_size_hist.shape)
                out_stat_2d_cluster_size_hist[:] = stat_2d_cluster_size_hist

                out_count_1d_total_angle_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='count_1d_total_angle_hist',
                    title='count_1d_total_angle_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(count_1d_total_angle_hist.dtype),
                    shape=count_1d_total_angle_hist.shape)
                out_count_1d_total_angle_hist[:] = count_1d_total_angle_hist

                out_count_1d_total_angle_hist_edges = out_file_h5.create_carray(
                    where=dut_group,
                    name='count_1d_total_angle_hist_edges',
                    title='count_1d_total_angle_hist_edges for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(count_1d_total_angle_hist_edges.dtype),
                    shape=count_1d_total_angle_hist_edges.shape)
                out_count_1d_total_angle_hist_edges[:] = count_1d_total_angle_hist_edges

                out_stat_2d_total_angle_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_2d_total_angle_hist',
                    title='stat_2d_total_angle_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_2d_total_angle_hist.dtype),
                    shape=stat_2d_total_angle_hist.shape)
                out_stat_2d_total_angle_hist[:] = stat_2d_total_angle_hist

                out_count_1d_alpha_angle_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='count_1d_alpha_angle_hist',
                    title='count_1d_alpha_angle_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(count_1d_alpha_angle_hist.dtype),
                    shape=count_1d_alpha_angle_hist.shape)
                out_count_1d_alpha_angle_hist[:] = count_1d_alpha_angle_hist

                out_count_1d_alpha_angle_hist_edges = out_file_h5.create_carray(
                    where=dut_group,
                    name='count_1d_alpha_angle_hist_edges',
                    title='count_1d_alpha_angle_hist_edges for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(count_1d_alpha_angle_hist_edges.dtype),
                    shape=count_1d_alpha_angle_hist_edges.shape)
                out_count_1d_alpha_angle_hist_edges[:] = count_1d_alpha_angle_hist_edges

                out_stat_2d_alpha_angle_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_2d_alpha_angle_hist',
                    title='stat_2d_alpha_angle_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_2d_alpha_angle_hist.dtype),
                    shape=stat_2d_alpha_angle_hist.shape)
                out_stat_2d_alpha_angle_hist[:] = stat_2d_alpha_angle_hist

                out_count_1d_beta_angle_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='count_1d_beta_angle_hist',
                    title='count_1d_beta_angle_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(count_1d_beta_angle_hist.dtype),
                    shape=count_1d_beta_angle_hist.shape)
                out_count_1d_beta_angle_hist[:] = count_1d_beta_angle_hist

                out_count_1d_beta_angle_hist_edges = out_file_h5.create_carray(
                    where=dut_group,
                    name='count_1d_beta_angle_hist_edges',
                    title='count_1d_beta_angle_hist_edges for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(count_1d_beta_angle_hist_edges.dtype),
                    shape=count_1d_beta_angle_hist_edges.shape)
                out_count_1d_beta_angle_hist_edges[:] = count_1d_beta_angle_hist_edges

                out_stat_2d_beta_angle_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_2d_beta_angle_hist',
                    title='stat_2d_beta_angle_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_2d_beta_angle_hist.dtype),
                    shape=stat_2d_beta_angle_hist.shape)
                out_stat_2d_beta_angle_hist[:] = stat_2d_beta_angle_hist

                out_stat_2d_efficiency_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_2d_efficiency_hist',
                    title='stat_2d_efficiency_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_2d_efficiency_hist.dtype),
                    shape=stat_2d_efficiency_hist.shape)
                out_stat_2d_efficiency_hist[:] = stat_2d_efficiency_hist

                out_stat_pixel_efficiency_hist = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_pixel_efficiency_hist',
                    title='stat_pixel_efficiency_hist for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_pixel_efficiency_hist.dtype),
                    shape=stat_pixel_efficiency_hist.shape)
                out_stat_pixel_efficiency_hist[:] = stat_pixel_efficiency_hist

                out_stat_2d_efficiency_hist_mask = out_file_h5.create_carray(
                    where=dut_group,
                    name='stat_2d_efficiency_hist_mask',
                    title='stat_2d_efficiency_hist_mask for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(stat_2d_efficiency_hist.mask.dtype),
                    shape=stat_2d_efficiency_hist.mask.shape)
                out_stat_2d_efficiency_hist_mask[:] = stat_2d_efficiency_hist.mask

                if in_cluster_file_h5:
                    out_count_pixel_hits_2d_hist = out_file_h5.create_carray(
                        where=dut_group,
                        name='count_pixel_hits_2d_hist',
                        title='count_pixel_hits_2d_hist for DUT%d' % actual_dut_index,
                        atom=tb.Atom.from_dtype(count_pixel_hits_2d_hist.dtype),
                        shape=count_pixel_hits_2d_hist.shape,
                        filters=tb.Filters(
                            complib='blosc',
                            complevel=5,
                            fletcher32=False))
                    out_count_pixel_hits_2d_hist[:] = count_pixel_hits_2d_hist

                if efficiency_regions_dut:
                    for actual_region_index, _ in enumerate(efficiency_regions_dut):
                        region_group = out_file_h5.create_group("/DUT%d" % actual_dut_index, 'Region_%d' % actual_region_index)

                        region_group._v_attrs.efficiency_regions_efficiency = efficiency_regions_efficiency[actual_region_index]
                        region_group._v_attrs.efficiency_regions_in_pixel_hist_extent = in_pixel_hist_extent
                        region_group._v_attrs.efficiency_regions_in_pixel_plot_range = in_pixel_plot_range
                        region_group._v_attrs.efficiency_regions_analyze_cluster_shapes = efficiency_regions_analyze_cluster_shapes

                        out_efficiency_regions_count_1d_charge_hist = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_count_1d_charge_hist',
                            title='efficiency_regions_count_1d_charge_hist for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_count_1d_charge_hist[actual_region_index].dtype),
                            shape=efficiency_regions_count_1d_charge_hist[actual_region_index].shape)
                        out_efficiency_regions_count_1d_charge_hist[:] = efficiency_regions_count_1d_charge_hist[actual_region_index]

                        out_efficiency_regions_count_1d_frame_hist = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_count_1d_frame_hist',
                            title='efficiency_regions_count_1d_frame_hist for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_count_1d_frame_hist[actual_region_index].dtype),
                            shape=efficiency_regions_count_1d_frame_hist[actual_region_index].shape)
                        out_efficiency_regions_count_1d_frame_hist[:] = efficiency_regions_count_1d_frame_hist[actual_region_index]

                        out_efficiency_regions_count_1d_cluster_size_hist = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_count_1d_cluster_size_hist',
                            title='efficiency_regions_count_1d_cluster_size_hist for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_count_1d_cluster_size_hist[actual_region_index].dtype),
                            shape=efficiency_regions_count_1d_cluster_size_hist[actual_region_index].shape)
                        out_efficiency_regions_count_1d_cluster_size_hist[:] = efficiency_regions_count_1d_cluster_size_hist[actual_region_index]

                        out_efficiency_regions_count_1d_total_angle_hist = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_count_1d_total_angle_hist',
                            title='efficiency_regions_count_1d_total_angle_hist for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_count_1d_total_angle_hist[actual_region_index].dtype),
                            shape=efficiency_regions_count_1d_total_angle_hist[actual_region_index].shape)
                        out_efficiency_regions_count_1d_total_angle_hist[:] = efficiency_regions_count_1d_total_angle_hist[actual_region_index]

                        out_efficiency_regions_count_1d_total_angle_hist_edges = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_count_1d_total_angle_hist_edges',
                            title='efficiency_regions_count_1d_total_angle_hist_edges for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_count_1d_total_angle_hist_edges[actual_region_index].dtype),
                            shape=efficiency_regions_count_1d_total_angle_hist_edges[actual_region_index].shape)
                        out_efficiency_regions_count_1d_total_angle_hist_edges[:] = efficiency_regions_count_1d_total_angle_hist_edges[actual_region_index]

                        out_efficiency_regions_count_1d_alpha_angle_hist = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_count_1d_alpha_angle_hist',
                            title='efficiency_regions_count_1d_alpha_angle_hist for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_count_1d_alpha_angle_hist[actual_region_index].dtype),
                            shape=efficiency_regions_count_1d_alpha_angle_hist[actual_region_index].shape)
                        out_efficiency_regions_count_1d_alpha_angle_hist[:] = efficiency_regions_count_1d_alpha_angle_hist[actual_region_index]

                        out_efficiency_regions_count_1d_alpha_angle_hist_edges = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_count_1d_alpha_angle_hist_edges',
                            title='efficiency_regions_count_1d_alpha_angle_hist_edges for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_count_1d_alpha_angle_hist_edges[actual_region_index].dtype),
                            shape=efficiency_regions_count_1d_alpha_angle_hist_edges[actual_region_index].shape)
                        out_efficiency_regions_count_1d_alpha_angle_hist_edges[:] = efficiency_regions_count_1d_alpha_angle_hist_edges[actual_region_index]

                        out_efficiency_regions_count_1d_beta_angle_hist = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_count_1d_beta_angle_hist',
                            title='efficiency_regions_count_1d_beta_angle_hist for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_count_1d_beta_angle_hist[actual_region_index].dtype),
                            shape=efficiency_regions_count_1d_beta_angle_hist[actual_region_index].shape)
                        out_efficiency_regions_count_1d_beta_angle_hist[:] = efficiency_regions_count_1d_beta_angle_hist[actual_region_index]

                        out_efficiency_regions_count_1d_beta_angle_hist_edges = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_count_1d_beta_angle_hist_edges',
                            title='efficiency_regions_count_1d_beta_angle_hist_edges for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_count_1d_beta_angle_hist_edges[actual_region_index].dtype),
                            shape=efficiency_regions_count_1d_beta_angle_hist_edges[actual_region_index].shape)
                        out_efficiency_regions_count_1d_beta_angle_hist_edges[:] = efficiency_regions_count_1d_beta_angle_hist_edges[actual_region_index]

                        out_efficiency_regions_count_1d_cluster_shape_hist = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_count_1d_cluster_shape_hist',
                            title='efficiency_regions_count_1d_cluster_shape_hist for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_count_1d_cluster_shape_hist[actual_region_index].dtype),
                            shape=efficiency_regions_count_1d_cluster_shape_hist[actual_region_index].shape)
                        out_efficiency_regions_count_1d_cluster_shape_hist[:] = efficiency_regions_count_1d_cluster_shape_hist[actual_region_index]

                        out_efficiency_regions_stat_pixel_efficiency_hist = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_stat_pixel_efficiency_hist',
                            title='efficiency_regions_stat_pixel_efficiency_hist for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_stat_pixel_efficiency_hist[actual_region_index].dtype),
                            shape=efficiency_regions_stat_pixel_efficiency_hist[actual_region_index].shape)
                        out_efficiency_regions_stat_pixel_efficiency_hist[:] = efficiency_regions_stat_pixel_efficiency_hist[actual_region_index]

                        out_count_in_pixel_hits_2d_hists = out_file_h5.create_carray(
                            where=region_group,
                            name='count_in_pixel_hits_2d_hists',
                            title='count_in_pixel_hits_2d_hists for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(count_in_pixel_hits_2d_hists[actual_region_index].dtype),
                            shape=count_in_pixel_hits_2d_hists[actual_region_index].shape)
                        out_count_in_pixel_hits_2d_hists[:] = count_in_pixel_hits_2d_hists[actual_region_index]

                        out_count_in_pixel_tracks_2d_hists = out_file_h5.create_carray(
                            where=region_group,
                            name='count_in_pixel_tracks_2d_hists',
                            title='count_in_pixel_tracks_2d_hists for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(count_in_pixel_tracks_2d_hists[actual_region_index].dtype),
                            shape=count_in_pixel_tracks_2d_hists[actual_region_index].shape)
                        out_count_in_pixel_tracks_2d_hists[:] = count_in_pixel_tracks_2d_hists[actual_region_index]

                        out_count_in_pixel_tracks_with_hit_2d_hists = out_file_h5.create_carray(
                            where=region_group,
                            name='count_in_pixel_tracks_with_hit_2d_hists',
                            title='count_in_pixel_tracks_with_hit_2d_hists for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(count_in_pixel_tracks_with_hit_2d_hists[actual_region_index].dtype),
                            shape=count_in_pixel_tracks_with_hit_2d_hists[actual_region_index].shape)
                        out_count_in_pixel_tracks_with_hit_2d_hists[:] = count_in_pixel_tracks_with_hit_2d_hists[actual_region_index]

                        out_efficiency_regions_stat_in_pixel_efficiency_2d_hist = out_file_h5.create_carray(
                            where=region_group,
                            name='efficiency_regions_stat_in_pixel_efficiency_2d_hist',
                            title='efficiency_regions_stat_in_pixel_efficiency_2d_hist for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(efficiency_regions_stat_in_pixel_efficiency_2d_hist[actual_region_index].dtype),
                            shape=efficiency_regions_stat_in_pixel_efficiency_2d_hist[actual_region_index].shape)
                        out_efficiency_regions_stat_in_pixel_efficiency_2d_hist[:] = efficiency_regions_stat_in_pixel_efficiency_2d_hist[actual_region_index]

                        out_stat_in_pixel_x_residuals_2d_hists = out_file_h5.create_carray(
                            where=region_group,
                            name='stat_in_pixel_x_residuals_2d_hists',
                            title='stat_in_pixel_x_residuals_2d_hists for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(stat_in_pixel_x_residuals_2d_hists[actual_region_index].dtype),
                            shape=stat_in_pixel_x_residuals_2d_hists[actual_region_index].shape)
                        out_stat_in_pixel_x_residuals_2d_hists[:] = stat_in_pixel_x_residuals_2d_hists[actual_region_index]

                        out_stat_in_pixel_y_residuals_2d_hists = out_file_h5.create_carray(
                            where=region_group,
                            name='stat_in_pixel_y_residuals_2d_hists',
                            title='stat_in_pixel_y_residuals_2d_hists for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(stat_in_pixel_y_residuals_2d_hists[actual_region_index].dtype),
                            shape=stat_in_pixel_y_residuals_2d_hists[actual_region_index].shape)
                        out_stat_in_pixel_y_residuals_2d_hists[:] = stat_in_pixel_y_residuals_2d_hists[actual_region_index]

                        out_stat_in_pixel_residuals_2d_hists = out_file_h5.create_carray(
                            where=region_group,
                            name='stat_in_pixel_residuals_2d_hists',
                            title='stat_in_pixel_residuals_2d_hists for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(stat_in_pixel_residuals_2d_hists[actual_region_index].dtype),
                            shape=stat_in_pixel_residuals_2d_hists[actual_region_index].shape)
                        out_stat_in_pixel_residuals_2d_hists[:] = stat_in_pixel_residuals_2d_hists[actual_region_index]

                        out_stat_in_pixel_charge_2d_hists = out_file_h5.create_carray(
                            where=region_group,
                            name='stat_in_pixel_charge_2d_hists',
                            title='stat_in_pixel_charge_2d_hists for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(stat_in_pixel_charge_2d_hists[actual_region_index].dtype),
                            shape=stat_in_pixel_charge_2d_hists[actual_region_index].shape)
                        out_stat_in_pixel_charge_2d_hists[:] = stat_in_pixel_charge_2d_hists[actual_region_index]

                        out_stat_in_pixel_frame_2d_hists = out_file_h5.create_carray(
                            where=region_group,
                            name='stat_in_pixel_frame_2d_hists',
                            title='stat_in_pixel_frame_2d_hists for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(stat_in_pixel_frame_2d_hists[actual_region_index].dtype),
                            shape=stat_in_pixel_frame_2d_hists[actual_region_index].shape)
                        out_stat_in_pixel_frame_2d_hists[:] = stat_in_pixel_frame_2d_hists[actual_region_index]

                        out_stat_in_pixel_cluster_size_2d_hists = out_file_h5.create_carray(
                            where=region_group,
                            name='stat_in_pixel_cluster_size_2d_hists',
                            title='stat_in_pixel_cluster_size_2d_hists for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(stat_in_pixel_cluster_size_2d_hists[actual_region_index].dtype),
                            shape=stat_in_pixel_cluster_size_2d_hists[actual_region_index].shape)
                        out_stat_in_pixel_cluster_size_2d_hists[:] = stat_in_pixel_cluster_size_2d_hists[actual_region_index]

                        out_count_in_pixel_cluster_shape_2d_hists = out_file_h5.create_carray(
                            where=region_group,
                            name='count_in_pixel_cluster_shape_2d_hists',
                            title='count_in_pixel_cluster_shape_2d_hists for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(count_in_pixel_cluster_shape_2d_hists[actual_region_index].dtype),
                            shape=count_in_pixel_cluster_shape_2d_hists[actual_region_index].shape)
                        out_count_in_pixel_cluster_shape_2d_hists[:] = count_in_pixel_cluster_shape_2d_hists[actual_region_index]

                        out_stat_in_pixel_cluster_shape_2d_hists = out_file_h5.create_carray(
                            where=region_group,
                            name='stat_in_pixel_cluster_shape_2d_hists',
                            title='stat_in_pixel_cluster_shape_2d_hists for DUT%d' % actual_dut_index,
                            atom=tb.Atom.from_dtype(stat_in_pixel_cluster_shape_2d_hists[actual_region_index].dtype),
                            shape=stat_in_pixel_cluster_shape_2d_hists[actual_region_index].shape)
                        out_stat_in_pixel_cluster_shape_2d_hists[:] = stat_in_pixel_cluster_shape_2d_hists[actual_region_index]

                if in_cluster_file_h5:
                    try:
                        in_cluster_file_h5.close()
                    except Exception:
                        pass
                    finally:
                        in_cluster_file_h5 = None

    if output_pdf is not None:
        output_pdf.close()

    return output_efficiency_file


def calculate_purity(telescope_configuration, input_tracks_file, select_duts, bin_size, output_purity_file=None, minimum_hit_density=10, cut_distance=None, local_x_range=None, local_y_range=None, plot=True, chunk_size=1000000):
    '''Takes the tracks and calculates the hit purity and hit/track hit distance for selected DUTs.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_tracks_file : string
        Filename with the tracks table.
    select_duts : iterable
        Selecting DUTs that will be processed.
    bin_size : iterable
        Bins sizes (i.e. (virtual) pixel size). Give one tuple (x, y) for every plane or list of tuples for different planes.
    output_purity_file : string
        Filename of the output purity file. If None, the filename will be derived from the input hits file.
    minimum_hit_density : int
        Minimum hit density required to consider bin for purity calculation.
    cut_distance : int
        Hit - track intersection <= cut_distance = pure hit (hit assigned to track).
        Hit - track intersection > cut_distance = inpure hit (hit without a track).
    local_x_range, local_y_range : iterable
        Column / row value to calculate purity for (to neglect noisy edge pixels for purity calculation).
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    logging.info('=== Calculating purity for %d DUTs ===' % len(select_duts))

    if output_purity_file is None:
        output_purity_file = os.path.splitext(input_tracks_file)[0] + '_purity.h5'

    if plot is True:
        output_pdf = PdfPages(os.path.splitext(output_purity_file)[0] + '.pdf', keep_empty=False)
    else:
        output_pdf = None

    purities = []
    purities_sensor_mean = []
    pure_hits = []
    total_hits = []
    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_purity_file, mode='w') as out_file_h5:
            for index, actual_dut_index in enumerate(select_duts):
                actual_dut = telescope[actual_dut_index]
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)

                logging.info('Calculating purity for %s', actual_dut.name)

                # Calculate histogram properties (bins size and number of bins)
                bin_size = [bin_size, ] if not isinstance(bin_size, Iterable) else bin_size
                if len(bin_size) != 1:
                    actual_bin_size_x = bin_size[index][0]
                    actual_bin_size_y = bin_size[index][1]
                else:
                    actual_bin_size_x = bin_size[0][0]
                    actual_bin_size_y = bin_size[0][1]

                dut_x_size = np.abs(np.diff(actual_dut.x_extent()))[0]
                dut_y_size = np.abs(np.diff(actual_dut.y_extent()))[0]
                sensor_size = [dut_x_size, dut_y_size]
                dimensions = sensor_size
                n_bin_x = int(dut_x_size / actual_bin_size_x)
                n_bin_y = int(dut_y_size / actual_bin_size_y)

                # Define result histograms, these are filled for each hit chunk
                total_hit_hist = np.zeros(shape=(n_bin_x, n_bin_y), dtype=np.uint32)
                total_pure_hit_hist = np.zeros(shape=(n_bin_x, n_bin_y), dtype=np.uint32)

                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    # Take only tracks where actual dut has a hit, otherwise residual wrong
                    selection = np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut_index]), ~np.isnan(tracks_chunk['track_chi2']))
                    selection_hit = ~np.isnan(tracks_chunk['x_dut_%d' % actual_dut_index])

                    # Transform the hits and track intersections into the local coordinate system
                    # Coordinates in global coordinate system (x, y, z)
                    hit_x_local_dut, hit_y_local_dut = tracks_chunk['x_dut_%d' % actual_dut_index][selection_hit], tracks_chunk['y_dut_%d' % actual_dut_index][selection_hit]
                    hit_x_local, hit_y_local = tracks_chunk['x_dut_%d' % actual_dut_index][selection], tracks_chunk['y_dut_%d' % actual_dut_index][selection]

                    intersection_x_local, intersection_y_local = tracks_chunk['offset_x'][selection], tracks_chunk['offset_y'][selection]

                    intersections_local = np.column_stack((intersection_x_local, intersection_y_local))
                    hits_local = np.column_stack((hit_x_local, hit_y_local))
                    hits_local_dut = np.column_stack((hit_x_local_dut, hit_y_local_dut))

                    # Select hits from column, row range (e.g. to supress edge pixels)
                    local_x_range = [local_x_range, ] if not isinstance(local_x_range, Iterable) else local_x_range
                    if len(local_x_range) == 1:
                        curr_local_x_range = local_x_range[0]
                    else:
                        curr_local_x_range = local_x_range[index]
                    if curr_local_x_range is not None:
                        selection = np.logical_and(intersections_local[:, 0] >= curr_local_x_range[0], intersections_local[:, 0] <= curr_local_x_range[1])  # Select real hits
                        hits_local, intersections_local = hits_local[selection], intersections_local[selection]

                    local_y_range = [local_y_range, ] if not isinstance(local_y_range, Iterable) else local_y_range
                    if len(local_y_range) == 1:
                        curr_local_y_range = local_y_range[0]
                    else:
                        curr_local_y_range = local_y_range[index]
                    if curr_local_y_range is not None:
                        selection = np.logical_and(intersections_local[:, 1] >= curr_local_y_range[0], intersections_local[:, 1] <= curr_local_y_range[1])  # Select real hits
                        hits_local, intersections_local = hits_local[selection], intersections_local[selection]

                    # Calculate distance between track hit and DUT hit
                    scale = np.square(np.array((1, 1)))  # Regard pixel size for calculating distances
                    distance_local = np.sqrt(np.dot(np.square(intersections_local - hits_local), scale))  # Array with distances between DUT hit and track hit for each event. Values in um

                    total_hit_hist += (np.histogram2d(hits_local_dut[:, 0], hits_local_dut[:, 1], bins=(n_bin_x, n_bin_y), range=[actual_dut.x_extent(), actual_dut.y_extent()])[0]).astype(np.uint32)

                    # Calculate purity
                    pure_hits_local = hits_local[distance_local < cut_distance]

                    if not np.any(pure_hits_local):
                        logging.warning('No pure hits in DUT%d, cannot calculate purity', actual_dut_index)
                        continue
                    total_pure_hit_hist += (np.histogram2d(pure_hits_local[:, 0], pure_hits_local[:, 1], bins=(n_bin_x, n_bin_y), range=[actual_dut.x_extent(), actual_dut.y_extent()])[0]).astype(np.uint32)

                purity = np.zeros_like(total_hit_hist)
                purity[total_hit_hist != 0] = total_pure_hit_hist[total_hit_hist != 0].astype(np.float64) / total_hit_hist[total_hit_hist != 0].astype(np.float64) * 100.0
                purity = np.ma.array(purity, mask=total_hit_hist < minimum_hit_density)
                # calculate sensor purity by weighting each pixel purity with total number of hits within the pixel
                purity_sensor = np.repeat(purity.ravel(), total_hit_hist.ravel())

                if not np.any(purity):
                    raise RuntimeError('No pure hit for DUT%d, consider changing cut values or check track building!', actual_dut_index)

                plot_utils.purity_plots(
                    total_pure_hit_hist,
                    total_hit_hist, purity,
                    purity_sensor,
                    actual_dut_index,
                    minimum_hit_density,
                    hist_extent=[actual_dut.x_extent()[0], actual_dut.x_extent()[1], actual_dut.y_extent()[1], actual_dut.y_extent()[0]],
                    cut_distance=cut_distance,
                    output_pdf=output_pdf)

                logging.info('Purity =  %1.4f +- %1.4f%%' % np.ma.mean(purity), np.ma.std(purity))
                purities.append(np.ma.mean(purity))
                purities_sensor_mean.append(np.ma.mean(purity_sensor))

                dut_group = out_file_h5.create_group(out_file_h5.root, 'DUT%d' % actual_dut_index)

                out_purity = out_file_h5.create_carray(
                    where=dut_group,
                    name='Purity',
                    title='Purity map for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(purity.dtype),
                    shape=purity.T.shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_purity_mask = out_file_h5.create_carray(
                    where=dut_group,
                    name='Purity_mask',
                    title='Masked pixel map for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(purity.mask.dtype),
                    shape=purity.mask.T.shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))

                # For correct statistical error calculation the number of pure hits over total hits is needed
                out_pure_hits = out_file_h5.create_carray(
                    where=dut_group,
                    name='Pure_hits',
                    title='Passing events for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(total_pure_hit_hist.dtype),
                    shape=total_pure_hit_hist.T.shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_total_total = out_file_h5.create_carray(
                    where=dut_group,
                    name='Total_hits',
                    title='Total events for DUT%d' % actual_dut_index,
                    atom=tb.Atom.from_dtype(total_hit_hist.dtype),
                    shape=total_hit_hist.T.shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))

                pure_hits.append(total_pure_hit_hist.sum())
                total_hits.append(total_hit_hist.sum())
                logging.info('Pure hits / total hits: %d / %d, Purity = %.2f%%' % (total_pure_hit_hist.sum(), total_hit_hist.sum(), total_pure_hit_hist.sum() / total_hit_hist.sum() * 100))

                # Store parameters used for purity calculation
                out_purity.attrs.bin_size = bin_size
                out_purity.attrs.minimum_hit_density = minimum_hit_density
                out_purity.attrs.sensor_size = sensor_size
                out_purity.attrs.use_duts = select_duts
                out_purity.attrs.cut_distance = cut_distance
                out_purity.attrs.local_x_range = local_x_range
                out_purity.attrs.local_y_range = local_y_range
                out_purity.attrs.purity_average = total_pure_hit_hist.sum() / total_hit_hist.sum() * 100
                out_purity[:] = purity.T
                out_purity_mask[:] = purity.mask.T
                out_pure_hits[:] = total_pure_hit_hist.T
                out_total_total[:] = total_hit_hist.T

    if output_pdf is not None:
        output_pdf.close()

    return purities, pure_hits, total_hits, purities_sensor_mean


@save_arguments
def histogram_track_angle(telescope_configuration, input_tracks_file, select_duts, output_track_angle_file=None, n_bins=100, plot=True, chunk_size=1000000):
    '''Calculates and histograms the track angle of the fitted tracks for selected DUTs.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_tracks_file : string
        Filename of the input tracks file.
    select_duts : list
        Selecting DUTs that will be processed.
    output_track_angle_file: string
        Filename of the output track angle file with track angle histogram and fitted means and sigmas of track angles for selected DUTs.
    n_bins : uint, string
        Number of bins for the histogram.
        If "auto", automatic binning is used.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_track_angle_file : string
        Filename of the output track angle file.
    '''
    telescope = Telescope(telescope_configuration)
    logging.info('=== Histogramming track angles for %d DUTs ===' % len(select_duts))

    if output_track_angle_file is None:
        output_track_angle_file = os.path.splitext(input_tracks_file)[0] + '_track_angles.h5'

    # calculating DUT indices list with z-order
    intersections_z_axis = []
    for dut in telescope:
        intersections_z_axis.append(geometry_utils.get_line_intersections_with_dut(
            line_origins=np.array([[0.0, 0.0, 0.0]]),
            line_directions=np.array([[0.0, 0.0, 1.0]]),
            translation_x=dut.translation_x,
            translation_y=dut.translation_y,
            translation_z=dut.translation_z,
            rotation_alpha=dut.rotation_alpha,
            rotation_beta=dut.rotation_beta,
            rotation_gamma=dut.rotation_gamma)[0][2])
    z_sorted_dut_indices = np.argsort(intersections_z_axis)
    # calculate DUT indices with z-order from selection
    selected_z_sorted_dut_indices = z_sorted_dut_indices[np.sort(np.where((z_sorted_dut_indices - np.array(select_duts)[:, np.newaxis]) == 0)[1])]

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_track_angle_file, mode="w") as out_file_h5:
            # insert DUT with smallest z at the beginning of the nodes list for calculating telescope tracks
            selected_dut_indices_mod = list(selected_z_sorted_dut_indices)
            selected_dut_indices_mod.insert(0, selected_z_sorted_dut_indices[0])
            for index, actual_dut_index in enumerate(selected_dut_indices_mod):
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)
                actual_dut = telescope[actual_dut_index]
                logging.info('== Calculating track angles for %s ==', ('telescope' if index == 0 else actual_dut.name))

                initialize = True
                # only store track slopes of selected DUTs
                for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    track_slopes_local = np.column_stack([
                        tracks_chunk['slope_x'],
                        tracks_chunk['slope_y'],
                        tracks_chunk['slope_z']])

                    track_slopes_global = np.column_stack(actual_dut.local_to_global_position(
                        x=tracks_chunk['slope_x'],
                        y=tracks_chunk['slope_y'],
                        z=tracks_chunk['slope_z'],
                        # no translation for the slopes
                        translation_x=0.0,
                        translation_y=0.0,
                        translation_z=0.0,
                        rotation_alpha=actual_dut.rotation_alpha,
                        rotation_beta=actual_dut.rotation_beta,
                        rotation_gamma=actual_dut.rotation_gamma))

                    total_angles_global, alpha_angles_global, beta_angles_global = get_angles(
                        slopes=track_slopes_global,
                        xz_plane_normal=np.array([0.0, 1.0, 0.0]),
                        yz_plane_normal=np.array([1.0, 0.0, 0.0]),
                        dut_plane_normal=np.array([0.0, 0.0, 1.0]))
                    if index != 0:
                        total_angles_local, alpha_angles_local, beta_angles_local = get_angles(
                            slopes=track_slopes_local,
                            xz_plane_normal=np.array([0.0, 1.0, 0.0]),
                            yz_plane_normal=np.array([1.0, 0.0, 0.0]),
                            dut_plane_normal=np.array([0.0, 0.0, 1.0]))

                    if initialize:
                        global_total_mean = np.nanmean(total_angles_global)
                        global_total_std = np.nanstd(total_angles_global)
                        global_alpha_mean = np.nanmean(alpha_angles_global)
                        global_alpha_std = np.nanstd(alpha_angles_global)
                        global_beta_mean = np.nanmean(beta_angles_global)
                        global_beta_std = np.nanstd(beta_angles_global)
                        total_angle_global_hist, total_angle_global_hist_edges = np.histogram(total_angles_global, bins=n_bins, range=(global_total_mean - 5 * global_total_std, global_total_mean + 5 * global_total_std))
                        alpha_angle_global_hist, alpha_angle_global_hist_edges = np.histogram(alpha_angles_global, bins=n_bins, range=(global_alpha_mean - 5 * global_alpha_std, global_alpha_mean + 5 * global_alpha_std))
                        beta_angle_global_hist, beta_angle_global_hist_edges = np.histogram(beta_angles_global, bins=n_bins, range=(global_beta_mean - 5 * global_beta_std, global_beta_mean + 5 * global_beta_std))
                        if index != 0:
                            local_total_mean = np.nanmean(total_angles_local)
                            local_total_std = np.nanstd(total_angles_local)
                            local_alpha_mean = np.nanmean(alpha_angles_local)
                            local_alpha_std = np.nanstd(alpha_angles_local)
                            local_beta_mean = np.nanmean(beta_angles_local)
                            local_beta_std = np.nanstd(beta_angles_local)
                            total_angle_local_hist, total_angle_local_hist_edges = np.histogram(total_angles_local, bins=n_bins, range=(local_total_mean - 5 * local_total_std, local_total_mean + 5 * local_total_std))
                            alpha_angle_local_hist, alpha_angle_local_hist_edges = np.histogram(alpha_angles_local, bins=n_bins, range=(local_alpha_mean - 5 * local_alpha_std, local_alpha_mean + 5 * local_alpha_std))
                            beta_angle_local_hist, beta_angle_local_hist_edges = np.histogram(beta_angles_local, bins=n_bins, range=(local_beta_mean - 5 * local_beta_std, local_beta_mean + 5 * local_beta_std))
                        initialize = False
                    else:
                        total_angle_global_hist += np.histogram(total_angles_global, bins=total_angle_global_hist_edges)[0]
                        alpha_angle_global_hist += np.histogram(alpha_angles_global, bins=alpha_angle_global_hist_edges)[0]
                        beta_angle_global_hist += np.histogram(beta_angles_global, bins=beta_angle_global_hist_edges)[0]
                        if index != 0:
                            total_angle_local_hist += np.histogram(total_angles_local, bins=total_angle_local_hist_edges)[0]
                            alpha_angle_local_hist += np.histogram(alpha_angles_local, bins=alpha_angle_local_hist_edges)[0]
                            beta_angle_local_hist += np.histogram(beta_angles_local, bins=beta_angle_local_hist_edges)[0]

                # write results
                track_angle_total_global = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='Global_total_track_angle_hist%s' % ('' if index == 0 else ('_DUT%d' % actual_dut_index)),
                    title='Global total track angle distribution%s' % ('' if index == 0 else (' for %s' % actual_dut.name)),
                    atom=tb.Atom.from_dtype(total_angle_global_hist.dtype),
                    shape=total_angle_global_hist.shape)
                track_angle_total_edges_global = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='Global_total_track_angle_edges%s' % ('' if index == 0 else ('_DUT%d' % actual_dut_index)),
                    title='Global total track angle hist edges%s' % ('' if index == 0 else (' for %s' % actual_dut.name)),
                    atom=tb.Atom.from_dtype(total_angle_global_hist_edges.dtype),
                    shape=total_angle_global_hist_edges.shape)
                track_angle_alpha_global = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='Global_alpha_track_angle_hist%s' % ('' if index == 0 else ('_DUT%d' % actual_dut_index)),
                    title='Global alpha track angle distribution%s' % ('' if index == 0 else (' for %s' % actual_dut.name)),
                    atom=tb.Atom.from_dtype(alpha_angle_global_hist.dtype),
                    shape=alpha_angle_global_hist.shape)
                track_angle_alpha_edges_global = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='Global_alpha_track_angle_edges%s' % ('' if index == 0 else ('_DUT%d' % actual_dut_index)),
                    title='Global alpha track angle hist edges%s' % ('' if index == 0 else (' for %s' % actual_dut.name)),
                    atom=tb.Atom.from_dtype(alpha_angle_global_hist_edges.dtype),
                    shape=alpha_angle_global_hist_edges.shape)
                track_angle_beta_global = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='Global_beta_track_angle_hist%s' % ('' if index == 0 else ('_DUT%d' % actual_dut_index)),
                    title='Global beta track angle distribution%s' % ('' if index == 0 else (' for %s' % actual_dut.name)),
                    atom=tb.Atom.from_dtype(beta_angle_global_hist.dtype),
                    shape=beta_angle_global_hist.shape)
                track_angle_beta_edges_global = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='Global_beta_track_angle_edges%s' % ('' if index == 0 else ('_DUT%d' % actual_dut_index)),
                    title='Global beta track angle hist edges%s' % ('' if index == 0 else (' for %s' % actual_dut.name)),
                    atom=tb.Atom.from_dtype(beta_angle_global_hist_edges.dtype),
                    shape=beta_angle_global_hist_edges.shape)
                if index != 0:
                    track_angle_total_local = out_file_h5.create_carray(
                        where=out_file_h5.root,
                        name='Local_total_track_angle_hist_DUT%d' % actual_dut_index,
                        title='Local total track angle distribution for %s' % actual_dut.name,
                        atom=tb.Atom.from_dtype(total_angle_local_hist.dtype),
                        shape=total_angle_local_hist.shape)
                    track_angle_total_edges_local = out_file_h5.create_carray(
                        where=out_file_h5.root,
                        name='Local_total_track_angle_edges_DUT%d' % actual_dut_index,
                        title='Local total track angle hist edges for %s' % actual_dut.name,
                        atom=tb.Atom.from_dtype(total_angle_local_hist_edges.dtype),
                        shape=total_angle_local_hist_edges.shape)
                    track_angle_alpha_local = out_file_h5.create_carray(
                        where=out_file_h5.root,
                        name='Local_alpha_track_angle_hist_DUT%d' % actual_dut_index,
                        title='Local alpha track angle distribution for %s' % actual_dut.name,
                        atom=tb.Atom.from_dtype(alpha_angle_local_hist.dtype),
                        shape=alpha_angle_local_hist.shape)
                    track_angle_alpha_edges_local = out_file_h5.create_carray(
                        where=out_file_h5.root,
                        name='Local_alpha_track_angle_edges_DUT%d' % actual_dut_index,
                        title='Local alpha track angle hist edges for %s' % actual_dut.name,
                        atom=tb.Atom.from_dtype(alpha_angle_local_hist_edges.dtype),
                        shape=alpha_angle_local_hist_edges.shape)
                    track_angle_beta_local = out_file_h5.create_carray(
                        where=out_file_h5.root,
                        name='Local_beta_track_angle_hist_DUT%d' % actual_dut_index,
                        title='Local beta track angle distribution for %s' % actual_dut.name,
                        atom=tb.Atom.from_dtype(beta_angle_local_hist.dtype),
                        shape=beta_angle_local_hist.shape)
                    track_angle_beta_edges_local = out_file_h5.create_carray(
                        where=out_file_h5.root,
                        name='Local_beta_track_angle_edges_DUT%d' % actual_dut_index,
                        title='Local beta track angle hist edges for %s' % actual_dut.name,
                        atom=tb.Atom.from_dtype(beta_angle_local_hist_edges.dtype),
                        shape=beta_angle_local_hist_edges.shape)

                loop_over_data = [(total_angle_global_hist, total_angle_global_hist_edges, track_angle_total_global, track_angle_total_edges_global),
                                  (alpha_angle_global_hist, alpha_angle_global_hist_edges, track_angle_alpha_global, track_angle_alpha_edges_global),
                                  (beta_angle_global_hist, beta_angle_global_hist_edges, track_angle_beta_global, track_angle_beta_edges_global)]
                if index != 0:
                    loop_over_data.extend([
                        (total_angle_local_hist, total_angle_local_hist_edges, track_angle_total_local, track_angle_total_edges_local),
                        (alpha_angle_local_hist, alpha_angle_local_hist_edges, track_angle_alpha_local, track_angle_alpha_edges_local),
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
                        try:
                            fit, _ = curve_fit(analysis_utils.gauss, bin_center, hilb, p0=[np.amax(hist), mean, rms])
                        except RuntimeError:
                            fit = [np.nan, np.nan, np.nan]

                    # store the data
                    hist_carray.attrs.amplitude = fit[0]
                    hist_carray.attrs.mean = fit[1]
                    hist_carray.attrs.sigma = fit[2]
                    hist_carray[:] = hist
                    edges_carray[:] = edges

    if plot:
        plot_utils.plot_track_angle(input_track_angle_file=output_track_angle_file, select_duts=select_duts, output_pdf_file=None, dut_names=telescope.dut_names)

    return output_track_angle_file


def get_angles(slopes, xz_plane_normal, yz_plane_normal, dut_plane_normal):
    # normalize track slopes to 1
    slopes_mag = np.sqrt(np.einsum('ij,ij->i', slopes, slopes))
    slopes /= slopes_mag[:, np.newaxis]
    track_slopes_onto_xz_plane = slopes - np.matmul(xz_plane_normal, slopes.T).reshape(-1, 1) * xz_plane_normal
    track_slopes_onto_xz_plane /= np.sqrt(np.einsum('ij,ij->i', track_slopes_onto_xz_plane, track_slopes_onto_xz_plane)).reshape(-1, 1)
    track_slopes_onto_yz_plane = slopes - np.matmul(yz_plane_normal, slopes.T).reshape(-1, 1) * yz_plane_normal
    track_slopes_onto_yz_plane /= np.sqrt(np.einsum('ij,ij->i', track_slopes_onto_yz_plane, track_slopes_onto_yz_plane)).reshape(-1, 1)
    normal_onto_xz_plane = dut_plane_normal - np.inner(xz_plane_normal, dut_plane_normal) * xz_plane_normal
    normal_onto_yz_plane = dut_plane_normal - np.inner(yz_plane_normal, dut_plane_normal) * yz_plane_normal
    total_angles = np.arccos(np.inner(dut_plane_normal, slopes))
    alpha_angles = np.arctan2(np.matmul(yz_plane_normal, np.cross(track_slopes_onto_yz_plane, normal_onto_yz_plane).T), np.matmul(normal_onto_yz_plane, track_slopes_onto_yz_plane.T))
    beta_angles = np.arctan2(np.matmul(xz_plane_normal, np.cross(track_slopes_onto_xz_plane, normal_onto_xz_plane).T), np.matmul(normal_onto_xz_plane, track_slopes_onto_xz_plane.T))
    return total_angles, alpha_angles, beta_angles


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
            for index, actual_dut in enumerate(select_duts):
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut)
                logging.info('Calculating residual correlation for DUT%d', actual_dut)

                stop = [plot_n_pixels[index][0] * actual_dut.column_size, plot_n_pixels[index][1] * actual_dut.row_size]
                x_edges = np.linspace(start=0, stop=stop[0], num=plot_n_bins[index][0] + 1, endpoint=True)
                y_edges = np.linspace(start=0, stop=stop[1], num=plot_n_bins[index][1] + 1, endpoint=True)

                ref_x_residuals_earray = out_file_h5.create_earray(
                    out_file_h5.root,
                    name='Column_residuals_reference_DUT%d' % actual_dut,
                    atom=tb.Float32Atom(),
                    shape=(0,),
                    title='Reference column residuals',
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                x_residuals_earray = out_file_h5.create_earray(
                    where=out_file_h5.root,
                    name='Column_residuals_DUT%d' % actual_dut,
                    atom=tb.Float32Atom(),
                    shape=(0, plot_n_bins[index][0]),
                    title='Column residuals',
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                x_residuals_earray.attrs.edges = x_edges
                ref_y_residuals_earray = out_file_h5.create_earray(
                    where=out_file_h5.root,
                    name='Row_residuals_reference_DUT%d' % actual_dut,
                    atom=tb.Float32Atom(),
                    shape=(0,),
                    title='Reference row residuals',
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                y_residuals_earray = out_file_h5.create_earray(
                    where=out_file_h5.root,
                    name='Row_residuals_DUT%d' % actual_dut,
                    atom=tb.Float32Atom(),
                    shape=(0, plot_n_bins[index][1]),
                    title='Row residuals',
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                y_residuals_earray.attrs.edges = y_edges
                if use_fit_limits:
                    fit_limit_x_local, fit_limit_y_local = fit_limits[actual_dut][0], fit_limits[actual_dut][1]
                else:
                    fit_limit_x_local = None
                    fit_limit_y_local = None

                correlate_n_tracks = min(correlate_n_tracks, node.nrows)
                pbar = tqdm(total=1.0, ncols=80)

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
                    ref_intersection_x, ref_intersection_y, ref_intersection_z = ref_chunk['offset_x'], ref_chunk['offset_y'], ref_chunk['offset_z']
#                     ref_offsets = np.column_stack([ref_chunk['offset_x'], ref_chunk['offset_y'], ref_chunk['offset_z']])
#                     ref_slopes = np.column_stack([ref_chunk['slope_x'], ref_chunk['slope_y'], ref_chunk['slope_z']])
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
#                         distance_local = np.sqrt(np.einsum('ij,ij->i', intersection_local - hit_local, intersection_local - hit_local))

                    for tracks_chunk, curr_correlate_index in analysis_utils.data_aligned_at_events(node, start_index=correlate_start_index, chunk_size=chunk_size):
                        selection = np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut]), ~np.isnan(tracks_chunk['track_chi2']))
                        tracks_chunk = tracks_chunk[selection]  # Take only tracks where actual dut has a hit, otherwise residual wrong

                        # Coordinates in global coordinate system (x, y, z)
                        hit_x_local, hit_y_local, hit_z_local = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
#                         hit_local = np.column_stack([hit_x_local, hit_y_local])
                        intersection_x_global, intersection_y_global, intersection_z_global = tracks_chunk['offset_x'], tracks_chunk['offset_y'], tracks_chunk['offset_z']
#                         offsets = np.column_stack([tracks_chunk['offset_x'], tracks_chunk['offset_y'], tracks_chunk['offset_z']])
#                         slopes = np.column_stack([tracks_chunk['slope_x'], tracks_chunk['slope_y'], tracks_chunk['slope_z']])
                        event_numbers = tracks_chunk['event_number']

                        if use_prealignment:
                            hit_x, hit_y, hit_z = geometry_utils.apply_alignment(hit_x_local, hit_y_local, hit_z_local,
                                                                                 dut_index=actual_dut,
                                                                                 prealignment=prealignment,
                                                                                 inverse=False)

                            intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x_global, intersection_y_global, intersection_z_global,
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
#                             intersection_x_global, intersection_y_global, intersection_z_global = dut_intersection[:, 0], dut_intersection[:, 1], dut_intersection[:, 2]

                            intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersection_x_global, intersection_y_global, intersection_z_global,
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
#                         distance_local = np.sqrt(np.einsum('ij,ij->i', intersection_local - hit_local, intersection_local - hit_local))

                        iterate_n_ref_tracks = min(ref_intersection_x_local_limit_xy.shape[0], correlate_n_tracks - correlate_n_tracks_total)
                        for ref_index in range(iterate_n_ref_tracks):
                            # Histogram residuals in different ways
                            x_res = [None] * plot_n_bins[index][0]
                            y_res = [None] * plot_n_bins[index][1]
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
                            x_res = analysis_utils.binned_statistic(np.absolute(diff_x[select_valid]), difference_x_local_limit_xy[select_valid], func=func, nbins=plot_n_bins[index][0], range=(0, stop[0]))
                            x_residuals_arr = np.full((np.max([arr.shape[0] for arr in x_res]), len(x_res)), fill_value=np.nan)
                            for index, arr in enumerate(x_res):
                                #                                 assert np.array_equal(x_res[index], x_residuals_alternative[index])
                                x_residuals_arr[:arr.shape[0], index] = arr
                            x_residuals_earray.append(x_residuals_arr)
                            x_residuals_earray.flush()
                            ref_x_residuals_arr = np.full(x_residuals_arr.shape[0], fill_value=ref_difference_x_local_limit_xy[ref_index])
                            ref_x_residuals_earray.append(ref_x_residuals_arr)
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
                            y_res = analysis_utils.binned_statistic(np.absolute(diff_y[select_valid]), difference_y_local_limit_xy[select_valid], func=func, nbins=plot_n_bins[index][1], range=(0, stop[1]))
                            y_residuals_arr = np.full((np.max([arr.shape[0] for arr in y_res]), len(y_res)), fill_value=np.nan)
                            for index, arr in enumerate(y_res):
                                y_residuals_arr[:arr.shape[0], index] = arr
                            y_residuals_earray.append(y_residuals_arr)
                            y_residuals_earray.flush()
                            ref_y_residuals_arr = np.full(y_residuals_arr.shape[0], fill_value=ref_difference_y_local_limit_xy[ref_index])
                            ref_y_residuals_earray.append(ref_y_residuals_arr)
                            ref_y_residuals_earray.flush()
                            pbar.update(min(1.0, (((ref_index + 1) / iterate_n_ref_tracks * (curr_correlate_index - correlate_index_last) / node.nrows) * iterate_n_ref_tracks / correlate_n_tracks) + ((correlate_index_last / node.nrows) * iterate_n_ref_tracks / correlate_n_tracks)) - pbar.n)
                        correlate_index_last = curr_correlate_index
                    correlate_n_tracks_total += iterate_n_ref_tracks

                    if correlate_n_tracks_total >= correlate_n_tracks:
                        break
                    correlate_start_index = curr_ref_index
                    ref_index_last = curr_ref_index

                pbar.close()

    if plot:
        plot_utils.plot_residual_correlation(input_residual_correlation_file=output_residual_correlation_file, select_duts=select_duts, pixel_size=pixel_size, output_pdf_file=None, dut_names=dut_names, chunk_size=chunk_size)
