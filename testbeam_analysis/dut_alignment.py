''' All DUT alignment functions in space and time are listed here plus additional alignment check functions'''
from __future__ import division

import logging
import re
import os
import progressbar
import warnings
from collections import Iterable

import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import tables as tb
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar, leastsq, basinhopping, OptimizeWarning, minimize
from matplotlib.backends.backend_pdf import PdfPages

from testbeam_analysis.tools import analysis_utils
from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.tools import data_selection

# Imports for track based alignment
from testbeam_analysis.track_analysis import fit_tracks
from testbeam_analysis.result_analysis import calculate_residuals

warnings.simplefilter("ignore", OptimizeWarning)  # Fit errors are handled internally, turn of warnings


def correlate_cluster(input_cluster_files, output_correlation_file, n_pixels, pixel_size=None, dut_names=None, plot=True, chunk_size=4999999):
    '''"Calculates the correlation histograms from the cluster arrays.
    The 2D correlation array of pairs of two different devices are created on event basis.
    All permutations are considered (all clusters of the first device are correlated with all clusters of the second device).

    Parameters
    ----------
    input_cluster_files : iterable
        Iterable of filenames of the cluster files.
    output_correlation_file : string
        Filename of the output correlation file with the correlation histograms.
    n_pixels : iterable of tuples
        One tuple per DUT describing the total number of pixels (column/row),
        e.g. for two FE-I4 DUTs [(80, 336), (80, 336)].
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension (column/row),
        e.g. for two FE-I4 DUTs [(250, 50), (250, 50)].
        If None, assuming same pixel size for all DUTs.
    dut_names : iterable of strings
        Names of the DUTs. If None, the DUT index will be used.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Correlating the index of %d DUTs ===', len(input_cluster_files))

    with tb.open_file(output_correlation_file, mode="w") as out_file_h5:
        n_duts = len(input_cluster_files)

        # Result arrays to be filled
        column_correlations = []
        row_correlations = []
        for dut_index in range(1, n_duts):
            shape_column = (n_pixels[dut_index][0], n_pixels[0][0])
            shape_row = (n_pixels[dut_index][1], n_pixels[0][1])
            column_correlations.append(np.zeros(shape_column, dtype=np.int32))
            row_correlations.append(np.zeros(shape_row, dtype=np.int32))

        start_indices = [None] * n_duts  # Store the loop indices for speed up

        with tb.open_file(input_cluster_files[0], mode='r') as in_file_h5:  # Open DUT0 cluster file
            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Cluster.shape[0], term_width=80)
            progress_bar.start()

            pool = Pool()  # Provide worker pool
            for cluster_dut_0, start_indices[0] in analysis_utils.data_aligned_at_events(in_file_h5.root.Cluster, start_index=start_indices[0], chunk_size=chunk_size):  # Loop over the cluster of DUT0 in chunks
                actual_event_numbers = cluster_dut_0[:]['event_number']

                # Create correlation histograms to the reference device for all other devices
                # Do this in parallel to safe time

                dut_results = []
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    dut_results.append(pool.apply_async(_correlate_cluster, kwds={'cluster_dut_0': cluster_dut_0,
                                                                                  'cluster_file': cluster_file,
                                                                                  'start_index': start_indices[dut_index],
                                                                                  'start_event_number': actual_event_numbers[0],
                                                                                  'stop_event_number': actual_event_numbers[-1] + 1,
                                                                                  'column_correlation': column_correlations[dut_index - 1],
                                                                                  'row_correlation': row_correlations[dut_index - 1],
                                                                                  'chunk_size': chunk_size
                                                                                  }
                                                        ))
                # Collect results when available
                for dut_index, dut_result in enumerate(dut_results, start=1):
                    (start_indices[dut_index], column_correlations[dut_index - 1], row_correlations[dut_index - 1]) = dut_result.get()

                progress_bar.update(start_indices[0])

            pool.close()
            pool.join()

        # Store the correlation histograms
        for dut_index in range(n_duts - 1):
            out_col = out_file_h5.create_carray(out_file_h5.root, name='CorrelationColumn_%d_0' % (dut_index + 1), title='Column Correlation between DUT%d and DUT%d' % (dut_index + 1, 0), atom=tb.Atom.from_dtype(column_correlations[dut_index].dtype), shape=column_correlations[dut_index].shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            out_row = out_file_h5.create_carray(out_file_h5.root, name='CorrelationRow_%d_0' % (dut_index + 1), title='Row Correlation between DUT%d and DUT%d' % (dut_index + 1, 0), atom=tb.Atom.from_dtype(row_correlations[dut_index].dtype), shape=row_correlations[dut_index].shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            out_col.attrs.filenames = [str(input_cluster_files[0]), str(input_cluster_files[dut_index])]
            out_row.attrs.filenames = [str(input_cluster_files[0]), str(input_cluster_files[dut_index])]
            out_col[:] = column_correlations[dut_index]
            out_row[:] = row_correlations[dut_index]
        progress_bar.finish()

    if plot:
        plot_utils.plot_correlations(input_correlation_file=output_correlation_file, pixel_size=pixel_size, dut_names=dut_names)


def merge_cluster_data(input_cluster_files, output_merged_file, n_pixels, pixel_size, chunk_size=4999999):
    '''Takes the cluster from all cluster files and merges them into one big table aligned at a common event number.

    Empty entries are signaled with column = row = charge = nan. Position is translated from indices to um. The
    local coordinate system origin (0, 0) is defined in the sensor center, to decouple translation and rotation.
    Cluster position errors are calculated from cluster dimensions.

    Parameters
    ----------
    input_cluster_files : list of pytables files
        File name of the input cluster files with correlation data.
    output_merged_file : pytables file
        File name of the output tracklet file.
    n_pixels : iterable of tuples
        One tuple per DUT describing the total number of pixels (column/row),
        e.g. for two FE-I4 DUTs [(80, 336), (80, 336)].
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension (column/row),
        e.g. for two FE-I4 DUTs [(250, 50), (250, 50)].
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Merge cluster files from %d DUTs to merged hit file ===', len(input_cluster_files))

    # Create result array description, depends on the number of DUTs
    description = [('event_number', np.int64)]
    for index, _ in enumerate(input_cluster_files):
        description.append(('x_dut_%d' % index, np.float))
    for index, _ in enumerate(input_cluster_files):
        description.append(('y_dut_%d' % index, np.float))
    for index, _ in enumerate(input_cluster_files):
        description.append(('z_dut_%d' % index, np.float))
    for index, _ in enumerate(input_cluster_files):
        description.append(('charge_dut_%d' % index, np.float))
    for index, _ in enumerate(input_cluster_files):
        description.append(('n_hits_dut_%d' % index, np.int8))
    description.extend([('track_quality', np.uint32), ('n_tracks', np.int8)])
    for index, _ in enumerate(input_cluster_files):
        description.append(('xerr_dut_%d' % index, np.float))
    for index, _ in enumerate(input_cluster_files):
        description.append(('yerr_dut_%d' % index, np.float))
    for index, _ in enumerate(input_cluster_files):
        description.append(('zerr_dut_%d' % index, np.float))

    start_indices_merging_loop = [None] * len(input_cluster_files)  # Store the merging loop indices for speed up
    start_indices_data_loop = [None] * len(input_cluster_files)  # Additional store indices for the data loop
    actual_start_event_number = None  # Defines the first event number of the actual chunk for speed up. Cannot be deduced from DUT0, since this DUT could have missing event numbers.

    # Merge the cluster data from different DUTs into one table
    with tb.open_file(output_merged_file, mode='w') as out_file_h5:
        merged_cluster_table = out_file_h5.create_table(out_file_h5.root, name='MergedCluster', description=np.zeros((1,), dtype=description).dtype, title='Merged cluster on event number', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        with tb.open_file(input_cluster_files[0], mode='r') as in_file_h5:  # Open DUT0 cluster file
            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Cluster.shape[0], term_width=80)
            progress_bar.start()
            for actual_cluster_dut_0, start_indices_data_loop[0] in analysis_utils.data_aligned_at_events(in_file_h5.root.Cluster, start_index=start_indices_data_loop[0], start_event_number=actual_start_event_number, stop_event_number=None, chunk_size=chunk_size):  # Loop over the cluster of DUT0 in chunks
                actual_event_numbers = actual_cluster_dut_0[:]['event_number']

                # First loop: calculate the minimum event number indices needed to merge all cluster from all files to this event number index
                common_event_numbers = actual_event_numbers
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open DUT0 cluster file
                        for actual_cluster, start_indices_merging_loop[dut_index] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Cluster, start_index=start_indices_merging_loop[dut_index], start_event_number=actual_start_event_number, stop_event_number=actual_event_numbers[-1] + 1, chunk_size=chunk_size, fail_on_missing_events=False):  # Loop over the cluster in the actual cluster file in chunks
                            common_event_numbers = analysis_utils.get_max_events_in_both_arrays(common_event_numbers, actual_cluster[:]['event_number'])
                merged_cluster_array = np.zeros(shape=(common_event_numbers.shape[0],), dtype=description)  # resulting array to be filled
                for index, _ in enumerate(input_cluster_files):
                    # for no hit: column = row = charge = nan
                    merged_cluster_array['x_dut_%d' % (index)] = np.nan
                    merged_cluster_array['y_dut_%d' % (index)] = np.nan
                    merged_cluster_array['z_dut_%d' % (index)] = np.nan
                    merged_cluster_array['charge_dut_%d' % (index)] = np.nan
                    merged_cluster_array['xerr_dut_%d' % (index)] = np.nan
                    merged_cluster_array['yerr_dut_%d' % (index)] = np.nan
                    merged_cluster_array['zerr_dut_%d' % (index)] = np.nan

                # Set the event number
                merged_cluster_array['event_number'] = common_event_numbers[:]

                # Fill result array with DUT 0 data
                actual_cluster_dut_0 = analysis_utils.map_cluster(common_event_numbers, actual_cluster_dut_0)
                # Select real hits, values with nan are virtual hits
                selection = ~np.isnan(actual_cluster_dut_0['mean_column'])
                # Convert indices to positions, origin defined in the center of the sensor
                merged_cluster_array['x_dut_0'][selection] = pixel_size[0][0] * (actual_cluster_dut_0['mean_column'][selection] - 0.5 - (0.5 * n_pixels[0][0]))
                merged_cluster_array['y_dut_0'][selection] = pixel_size[0][1] * (actual_cluster_dut_0['mean_row'][selection] - 0.5 - (0.5 * n_pixels[0][1]))
                merged_cluster_array['z_dut_0'][selection] = 0.0
                xerr = np.zeros(selection.shape)
                yerr = np.zeros(selection.shape)
                zerr = np.zeros(selection.shape)
                xerr[selection] = actual_cluster_dut_0['err_column'][selection] * pixel_size[0][0]
                yerr[selection] = actual_cluster_dut_0['err_row'][selection] * pixel_size[0][1]
                merged_cluster_array['xerr_dut_0'][selection] = xerr[selection]
                merged_cluster_array['yerr_dut_0'][selection] = yerr[selection]
                merged_cluster_array['zerr_dut_0'][selection] = zerr[selection]
                merged_cluster_array['charge_dut_0'][selection] = actual_cluster_dut_0['charge'][selection]
                merged_cluster_array['n_hits_dut_0'][selection] = actual_cluster_dut_0['n_hits'][selection]

                # Fill result array with other DUT data
                # Second loop: get the cluster from all files and merge them to the common event number
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open other DUT cluster file
                        for actual_cluster_dut, start_indices_data_loop[dut_index] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Cluster, start_index=start_indices_data_loop[dut_index], start_event_number=common_event_numbers[0], stop_event_number=common_event_numbers[-1] + 1, chunk_size=chunk_size, fail_on_missing_events=False):  # Loop over the cluster in the actual cluster file in chunks
                            actual_cluster_dut = analysis_utils.map_cluster(common_event_numbers, actual_cluster_dut)
                            # Select real hits, values with nan are virtual hits
                            selection = ~np.isnan(actual_cluster_dut['mean_column'])
                            # Convert indices to positions, origin in the center of the sensor, remaining DUTs
                            merged_cluster_array['x_dut_%d' % (dut_index)][selection] = pixel_size[dut_index][0] * (actual_cluster_dut['mean_column'][selection] - 0.5 - (0.5 * n_pixels[dut_index][0]))
                            merged_cluster_array['y_dut_%d' % (dut_index)][selection] = pixel_size[dut_index][1] * (actual_cluster_dut['mean_row'][selection] - 0.5 - (0.5 * n_pixels[dut_index][1]))
                            merged_cluster_array['z_dut_%d' % (dut_index)][selection] = 0.0
                            xerr = np.zeros(selection.shape)
                            yerr = np.zeros(selection.shape)
                            zerr = np.zeros(selection.shape)
                            xerr[selection] = actual_cluster_dut['err_column'][selection] * pixel_size[dut_index][0]
                            yerr[selection] = actual_cluster_dut['err_row'][selection] * pixel_size[dut_index][1]
                            merged_cluster_array['xerr_dut_%d' % (dut_index)][selection] = xerr[selection]
                            merged_cluster_array['yerr_dut_%d' % (dut_index)][selection] = yerr[selection]
                            merged_cluster_array['zerr_dut_%d' % (dut_index)][selection] = zerr[selection]
                            merged_cluster_array['charge_dut_%d' % (dut_index)][selection] = actual_cluster_dut['charge'][selection]
                            merged_cluster_array['n_hits_dut_%d' % (dut_index)][selection] = actual_cluster_dut['n_hits'][selection]

                merged_cluster_table.append(merged_cluster_array)
                actual_start_event_number = common_event_numbers[-1] + 1  # Set the starting event number for the next chunked read
                progress_bar.update(start_indices_data_loop[0])
            progress_bar.finish()


def prealignment(input_correlation_file, output_alignment_file, z_positions, pixel_size, s_n=0.1, fit_background=False, reduce_background=False, dut_names=None, no_fit=False, non_interactive=True, iterations=3, plot=True, gui=False, queue=False):
    '''Deduce a pre-alignment from the correlations, by fitting the correlations with a straight line (gives offset, slope, but no tild angles).
       The user can define cuts on the fit error and straight line offset in an interactive way.

    Parameters
    ----------
    input_correlation_file : string
        Filename of the input correlation file.
    output_alignment_file : string
        Filename of the output alignment file.
    z_positions : iterable
        The z positions of the DUTs in um.
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension (column/row),
        e.g. for two FE-I4 DUTs [(250, 50), (250, 50)].
    s_n : float
        The signal to noise ratio for peak signal over background peak. This should be specified when the background is fitted with a gaussian function.
        Usually data with a lot if tracks per event have a gaussian background. A good S/N value can be estimated by investigating the correlation plot.
        The default value is usually fine.
    fit_background : bool
        Data with a lot if tracks per event have a gaussian background from the beam profile. Also try to fit this background to determine the correlation
        peak correctly. If you see a clear 2D gaussian in the correlation plot this shoud be activated. If you have 1-2 tracks per event and large pixels
        this option should be off, because otherwise overfitting is possible.
    reduce_background : bool
        Reduce background (uncorrelated events) by using SVD of the 2D correlation array.
    dut_names : iterable
        Names of the DUTs. If None, the DUT index will be used.
    no_fit : bool
        Use Hough transformation to calculate slope and offset.
    non_interactive : bool
        Deactivate user interaction and estimate fit range automatically.
    iterations : uint
        The number of iterations in non-interactive mode.
    plot : bool
        If True, create additional output plots.
    gui : bool
        If True, this function is excecuted from GUI and returns figures
    queue : bool, dict
        If gui is True and non_interactive is False, queue is a dict with a in and output queue to communicate with GUI thread
    '''
    logging.info('=== Pre-alignment ===')

    if no_fit:
        if not reduce_background:
            logging.warning("no_fit is True, setting reduce_background to True")
            reduce_background = True

    if reduce_background:
        if fit_background:
            logging.warning("reduce_background is True, setting fit_background to False")
            fit_background = False

    if plot is True and not gui:
        output_pdf = PdfPages(os.path.splitext(output_alignment_file)[0] + '_prealigned.pdf', keep_empty=False)
    else:
        output_pdf = None

    figs = [] if gui else None

    with tb.open_file(input_correlation_file, mode="r") as in_file_h5:
        n_duts = len(in_file_h5.list_nodes("/")) // 2 + 1  # no correlation for reference DUT0
        result = np.zeros(shape=(n_duts,), dtype=[('DUT', np.uint8), ('column_c0', np.float), ('column_c0_error', np.float), ('column_c1', np.float), ('column_c1_error', np.float), ('column_sigma', np.float), ('column_sigma_error', np.float), ('row_c0', np.float), ('row_c0_error', np.float), ('row_c1', np.float), ('row_c1_error', np.float), ('row_sigma', np.float), ('row_sigma_error', np.float), ('z', np.float)])
        # Set std. settings for reference DUT0
        result[0]['column_c0'], result[0]['column_c0_error'] = 0.0, 0.0
        result[0]['column_c1'], result[0]['column_c1_error'] = 1.0, 0.0
        result[0]['row_c0'], result[0]['row_c0_error'] = 0.0, 0.0
        result[0]['row_c1'], result[0]['row_c1_error'] = 1.0, 0.0

        fit_limits = np.full((n_duts, 2, 2), fill_value=np.nan, dtype=np.float)  # col left, right, row left, right

        result[0]['z'] = z_positions[0]
        for node in in_file_h5.root:
            table_prefix = 'column' if 'column' in node.name.lower() else 'row'
            indices = re.findall(r'\d+', node.name)
            dut_idx = int(indices[0])
            ref_idx = int(indices[1])
            result[dut_idx]['DUT'] = dut_idx
            dut_name = dut_names[dut_idx] if dut_names else ("DUT" + str(dut_idx))
            ref_name = dut_names[ref_idx] if dut_names else ("DUT" + str(ref_idx))
            logging.info('Aligning data from %s', node.name)

            if "column" in node.name.lower():
                pixel_size_dut, pixel_size_ref = pixel_size[dut_idx][0], pixel_size[ref_idx][0]
            else:
                pixel_size_dut, pixel_size_ref = pixel_size[dut_idx][1], pixel_size[ref_idx][1]

            data = node[:]

            n_pixel_dut, n_pixel_ref = data.shape[0], data.shape[1]

            # Initialize arrays with np.nan (invalid), adding 0.5 to change from index to position
            # matrix index 0 is cluster index 1 ranging from 0.5 to 1.4999, which becomes position 0.0 to 0.999 with center at 0.5, etc.
            x_ref = (np.linspace(0.0, n_pixel_ref, num=n_pixel_ref, endpoint=False, dtype=np.float) + 0.5)
            x_dut = (np.linspace(0.0, n_pixel_dut, num=n_pixel_dut, endpoint=False, dtype=np.float) + 0.5)
            coeff_fitted = [None] * n_pixel_dut
            mean_fitted = np.empty(shape=(n_pixel_dut,), dtype=np.float)  # Peak of the Gauss fit
            mean_fitted.fill(np.nan)
            mean_error_fitted = np.empty(shape=(n_pixel_dut,), dtype=np.float)  # Error of the fit of the peak
            mean_error_fitted.fill(np.nan)
            sigma_fitted = np.empty(shape=(n_pixel_dut,), dtype=np.float)  # Sigma of the Gauss fit
            sigma_fitted.fill(np.nan)
            chi2 = np.empty(shape=(n_pixel_dut,), dtype=np.float)  # Chi2 of the fit
            chi2.fill(np.nan)
            n_cluster = np.sum(data, axis=1)  # Number of hits per bin

            if reduce_background:
                uu, dd, vv = np.linalg.svd(data)  # sigular value decomposition
                background = np.matrix(uu[:, :1]) * np.diag(dd[:1]) * np.matrix(vv[:1, :])  # take first sigular value for background
                background = np.array(background, dtype=np.int32)  # make Numpy array
                data = (data - background).astype(np.int32)  # remove background
                data -= data.min()  # only positive values

            # calculate half hight
            median = np.median(data)
            median_max = np.median(np.max(data, axis=1))
            half_median_data = (data > ((median + median_max) / 2))
            # calculate maximum per column
            max_select = np.argmax(data, axis=1)
            hough_data = np.zeros_like(data)
            hough_data[np.arange(data.shape[0]), max_select] = 1
            # select maximums if larger than half hight
            hough_data = hough_data & half_median_data
            # transpose for correct angle
            hough_data = hough_data.T
            accumulator, theta, rho, theta_edges, rho_edges = analysis_utils.hough_transform(hough_data, theta_res=0.1, rho_res=1.0, return_edges=True)
            rho_idx, th_idx = np.unravel_index(accumulator.argmax(), accumulator.shape)
            rho_val, theta_val = rho[rho_idx], theta[th_idx]
            slope_idx, offset_idx = -np.cos(theta_val) / np.sin(theta_val), rho_val / np.sin(theta_val)
            slope = slope_idx * (pixel_size_ref / pixel_size_dut)
            offset = offset_idx * pixel_size_ref
            # offset in the center of the pixel matrix
            offset_center = offset + slope * pixel_size_dut * n_pixel_dut * 0.5 - pixel_size_ref * n_pixel_ref * 0.5
            offset_center += 0.5 * pixel_size_ref - slope * 0.5 * pixel_size_dut  # correct for half bin

            if no_fit:
                result[dut_idx][table_prefix + '_c0'], result[dut_idx][table_prefix + '_c0_error'] = offset_center, 0.0
                result[dut_idx][table_prefix + '_c1'], result[dut_idx][table_prefix + '_c1_error'] = slope, 0.0
                result[dut_idx][table_prefix + '_sigma'], result[dut_idx][table_prefix + '_sigma_error'] = 0.0, 0.0
                result[dut_idx]['z'] = z_positions[dut_idx]

                fit_limits[dut_idx][0 if table_prefix == "column" else 1] = [(x_dut.min() - 0.5 * n_pixel_dut) * pixel_size_dut, (x_dut.max() - 0.5 * n_pixel_dut) * pixel_size_dut]

                plot_utils.plot_hough(x=x_dut,
                                      data=hough_data,
                                      accumulator=accumulator,
                                      offset=offset_idx,
                                      slope=slope_idx,
                                      theta_edges=theta_edges,
                                      rho_edges=rho_edges,
                                      n_pixel_ref=n_pixel_ref,
                                      n_pixel_dut=n_pixel_dut,
                                      pixel_size_ref=pixel_size_ref,
                                      pixel_size_dut=pixel_size_dut,
                                      ref_name=ref_name,
                                      dut_name=dut_name,
                                      prefix=table_prefix,
                                      output_pdf=output_pdf,
                                      gui=gui,
                                      figs=figs)

            else:
                # fill the arrays from above with values
                _fit_data(x=x_ref, data=data, s_n=s_n, coeff_fitted=coeff_fitted, mean_fitted=mean_fitted, mean_error_fitted=mean_error_fitted, sigma_fitted=sigma_fitted, chi2=chi2, fit_background=fit_background, reduce_background=reduce_background)

                # Convert fit results to metric units for alignment fit
                # Origin is center of pixel matrix
                x_dut_scaled = (x_dut - 0.5 * n_pixel_dut) * pixel_size_dut
                mean_fitted_scaled = (mean_fitted - 0.5 * n_pixel_ref) * pixel_size_ref
                mean_error_fitted_scaled = mean_error_fitted * pixel_size_ref

                # Selected data arrays
                x_selected = x_dut.copy()
                x_dut_scaled_selected = x_dut_scaled.copy()
                mean_fitted_scaled_selected = mean_fitted_scaled.copy()
                mean_error_fitted_scaled_selected = mean_error_fitted_scaled.copy()
                sigma_fitted_selected = sigma_fitted.copy()
                chi2_selected = chi2.copy()
                n_cluster_selected = n_cluster.copy()

                # Show the straigt line correlation fit including fit errors and offsets from the fit
                # Let the user change the cuts (error limit, offset limit) and refit until result looks good
                refit = True
                selected_data = np.ones_like(x_dut, dtype=np.bool)
                actual_iteration = 0  # Refit counter for non interactive mode
                while refit:
                    if gui and not non_interactive:
                        # Put data in queue to be processed interactively on GUI thread
                        queue['in'].put([x_dut_scaled_selected, mean_fitted_scaled_selected,
                                         mean_error_fitted_scaled_selected, n_cluster_selected,
                                         ref_name, dut_name, table_prefix])
                        # Blocking statement to wait for processed data from GUI thread
                        selected_data, fit, refit = queue['out'].get()
                    else:
                        selected_data, fit, refit = plot_utils.plot_prealignments(x=x_dut_scaled_selected,
                                                                                  mean_fitted=mean_fitted_scaled_selected,
                                                                                  mean_error_fitted=mean_error_fitted_scaled_selected,
                                                                                  n_cluster=n_cluster_selected,
                                                                                  ref_name=ref_name,
                                                                                  dut_name=dut_name,
                                                                                  prefix=table_prefix,
                                                                                  non_interactive=non_interactive,
                                                                                  pre_fit=[offset_center, slope] if actual_iteration == 0 else None)
                    x_selected = x_selected[selected_data]
                    x_dut_scaled_selected = x_dut_scaled_selected[selected_data]
                    mean_fitted_scaled_selected = mean_fitted_scaled_selected[selected_data]
                    mean_error_fitted_scaled_selected = mean_error_fitted_scaled_selected[selected_data]
                    sigma_fitted_selected = sigma_fitted_selected[selected_data]
                    chi2_selected = chi2_selected[selected_data]
                    n_cluster_selected = n_cluster_selected[selected_data]
                    # Stop in non interactive mode if the number of refits (iterations) is reached
                    if non_interactive:
                        actual_iteration += 1
                        if actual_iteration >= iterations:
                            break

                # Linear fit, usually describes correlation very well, slope is close to 1.
                # With low energy beam and / or beam with diverse agular distribution, the correlation will not be perfectly straight
                # Use results from straight line fit as start values for this final fit
                re_fit, re_fit_pcov = curve_fit(analysis_utils.linear, x_dut_scaled_selected, mean_fitted_scaled_selected, sigma=mean_error_fitted_scaled_selected, absolute_sigma=True, p0=[fit[0], fit[1]])

                # Write fit results to array
                result[dut_idx][table_prefix + '_c0'], result[dut_idx][table_prefix + '_c0_error'] = re_fit[0], np.absolute(re_fit_pcov[0][0]) ** 0.5
                result[dut_idx][table_prefix + '_c1'], result[dut_idx][table_prefix + '_c1_error'] = re_fit[1], np.absolute(re_fit_pcov[1][1]) ** 0.5
                result[dut_idx]['z'] = z_positions[dut_idx]

                fit_limits[dut_idx][0 if table_prefix == "column" else 1] = [x_dut_scaled_selected.min(), x_dut_scaled_selected.max()]

                # Calculate mean sigma (is a residual when assuming straight tracks) and its error and store the actual data in result array
                # This error is needed for track finding and track quality determination
                mean_sigma = pixel_size_ref * np.mean(np.array(sigma_fitted_selected))
                mean_sigma_error = pixel_size_ref * np.std(np.array(sigma_fitted_selected)) / np.sqrt(np.array(sigma_fitted_selected).shape[0])

                result[dut_idx][table_prefix + '_sigma'], result[dut_idx][table_prefix + '_sigma_error'] = mean_sigma, mean_sigma_error

                # Calculate the index of the beam center based on valid indices
                plot_index = np.average(x_selected - 1, weights=np.sum(data, axis=1)[np.array(x_selected - 1, dtype=np.int32)])
                # Find nearest valid index to the calculated index
                idx = (np.abs(x_selected - 1 - plot_index)).argmin()
                plot_index = np.array(x_selected - 1, dtype=np.int32)[idx]

                x_fit = np.linspace(start=x_ref.min(), stop=x_ref.max(), num=500, endpoint=True)
                indices_lower = np.arange(plot_index)
                indices_higher = np.arange(plot_index, n_pixel_dut)
                alternating_indices = np.vstack((np.hstack([indices_higher, indices_lower[::-1]]), np.hstack([indices_lower[::-1], indices_higher]))).reshape((-1,), order='F')
                unique_indices = np.unique(alternating_indices, return_index=True)[1]
                alternating_indices = alternating_indices[np.sort(unique_indices)]
                for plot_index in alternating_indices:
                    plot_correlation_fit = False
                    if coeff_fitted[plot_index] is not None:
                        plot_correlation_fit = True
                        break
                if plot_correlation_fit:
                    if np.all(np.isnan(coeff_fitted[plot_index][3:6])):
                        y_fit = analysis_utils.gauss_offset(x_fit, *coeff_fitted[plot_index][[0, 1, 2, 6]])
                        fit_label = "Gauss-Offset"
                    else:
                        y_fit = analysis_utils.double_gauss_offset(x_fit, *coeff_fitted[plot_index])
                        fit_label = "Gauss-Gauss-Offset"

                    plot_utils.plot_correlation_fit(x=x_ref,
                                                    y=data[plot_index, :],
                                                    x_fit=x_fit,
                                                    y_fit=y_fit,
                                                    xlabel='%s %s' % ("Column" if "column" in node.name.lower() else "Row", ref_name),
                                                    fit_label=fit_label,
                                                    title="Correlation of %s: %s vs. %s at %s %d" % (table_prefix + "s", ref_name, dut_name, table_prefix, plot_index),
                                                    output_pdf=output_pdf,
                                                    gui=gui,
                                                    figs=figs)
                else:
                    logging.warning("Cannot plot correlation fit, no fit data available")

                # Plot selected data with fit
                fit_fn = np.poly1d(re_fit[::-1])
                selected_indices = np.searchsorted(x_dut_scaled, x_dut_scaled_selected)
                mask = np.zeros_like(x_dut_scaled, dtype=np.bool)
                mask[selected_indices] = True

                plot_utils.plot_prealignment_fit(x=x_dut_scaled,
                                                 mean_fitted=mean_fitted_scaled,
                                                 mask=mask,
                                                 fit_fn=fit_fn,
                                                 fit=re_fit,
                                                 fit_limit=fit_limits[dut_idx][0 if table_prefix == "column" else 1],
                                                 pcov=re_fit_pcov,
                                                 chi2=chi2,
                                                 mean_error_fitted=mean_error_fitted_scaled,
                                                 n_cluster=n_cluster,
                                                 n_pixel_ref=n_pixel_ref,
                                                 n_pixel_dut=n_pixel_dut,
                                                 pixel_size_ref=pixel_size_ref,
                                                 pixel_size_dut=pixel_size_dut,
                                                 ref_name=ref_name,
                                                 dut_name=dut_name,
                                                 prefix=table_prefix,
                                                 output_pdf=output_pdf,
                                                 gui=gui,
                                                 figs=figs)

        if gui and not non_interactive:
            queue['in'].put([None])  # Put random element in queue to signal GUI thread end of interactive prealignment

        logging.info('Store pre-alignment data in %s', output_alignment_file)
        with tb.open_file(output_alignment_file, mode="w") as out_file_h5:
            try:
                result_table = out_file_h5.create_table(out_file_h5.root, name='PreAlignment', description=result.dtype, title='Prealignment alignment from correlation', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                result_table.append(result)
                result_table.attrs.fit_limits = fit_limits
            except tb.exceptions.NodeError:
                logging.warning('Coarse alignment table exists already. Do not create new.')

    if output_pdf is not None:
        output_pdf.close()

    if gui:
        return figs


def _fit_data(x, data, s_n, coeff_fitted, mean_fitted, mean_error_fitted, sigma_fitted, chi2, fit_background, reduce_background):

    def calc_limits_from_fit(x, coeff):
        ''' Calculates the fit limits from the last successfull fit.'''
        limits = [
            [0.1 * coeff[0], x.min(), 0.5 * coeff[2], 0.01 * coeff[3], x.min(), 0.5 * coeff[5], 0.5 * coeff[6]],
            [10.0 * coeff[0], x.max(), 2.0 * coeff[2], 10.0 * coeff[3], x.max(), 2.0 * coeff[5], 2.0 * coeff[6]]
        ]
        # Fix too small sigma, sigma < 1 is unphysical
        if limits[1][2] < 1.:
            limits[1][2] = 10.
        return limits

    def signal_sanity_check(coeff, s_n, A_peak):
        ''' Sanity check if signal was deducted correctly from background.

            3 Conditions:
            1. The given signal to noise value has to be fullfilled: S/N > Amplitude Signal / ( Amplidude background + Offset)
            2. The signal + background has to be large enough: Amplidute 1 + Amplitude 2 + Offset > Data maximum / 2
            3. The Signal Sigma has to be smaller than the background sigma, otherwise beam would be larger than one pixel pitch
        '''
        if coeff[0] < (coeff[3] + coeff[6]) * s_n or coeff[0] + coeff[3] + coeff[6] < A_peak / 2.0 or coeff[2] > coeff[5] / 2.0:
            return False
        return True

    n_pixel_dut, n_pixel_ref = data.shape[0], data.shape[1]
    # Start values for fitting
    # Correlation peak
    mu_peak = x[np.argmax(data, axis=1)]
    A_peak = np.max(data, axis=1)  # signal / correlation peak
    # Background of uncorrelated data
    n_entries = np.sum(data, axis=1)
    A_background = np.mean(data, axis=1)  # noise / background halo
    mu_background = np.zeros_like(n_entries)
    mu_background[n_entries > 0] = np.average(data, axis=1, weights=x)[n_entries > 0] * np.sum(x) / n_entries[n_entries > 0]

    coeff = None
    fit_converged = False  # To signal that las fit was good, thus the results can be taken as start values for next fit

    # for logging
    no_correlation_indices = []
    few_correlation_indices = []
    # get index of the highest background value
    fit_start_index = np.argmax(A_background)
    indices_lower = np.arange(fit_start_index)[::-1]
    indices_higher = np.arange(fit_start_index, n_pixel_dut)
    stacked_indices = np.hstack([indices_lower, indices_higher])

    for index in stacked_indices:  # Loop over x dimension of correlation histogram
        if index == fit_start_index:
            if index > 0 and coeff_fitted[index - 1] is not None:
                coeff = coeff_fitted[index - 1]
                fit_converged = True
            else:
                fit_converged = False
        # TODO: start fitting from the beam center to get a higher chance to pick up the correlation peak

        # omit correlation fit with no entries / correlation (e.g. sensor edges, masked columns)
        if np.all(data[index, :] == 0):
            no_correlation_indices.append(index)
            continue

        # omit correlation fit if sum of correlation entries is < 1 % of total entries devided by number of indices
        # (e.g. columns not in the beam)
        n_cluster_curr_index = data[index, :].sum()
        if fit_converged and n_cluster_curr_index < data.sum() / n_pixel_dut * 0.01:
            few_correlation_indices.append(index)
            continue

        # Set start parameters and fit limits
        # Parameters: A_1, mu_1, sigma_1, A_2, mu_2, sigma_2, offset
        if fit_converged and not reduce_background:  # Set start values from last successfull fit, no large difference expected
            p0 = coeff  # Set start values from last successfull fit
            bounds = calc_limits_from_fit(x, coeff)  # Set boundaries from previous converged fit
        else:  # No (last) successfull fit, try to dedeuce reasonable start values
            p0 = [A_peak[index], mu_peak[index], 5.0, A_background[index], mu_background[index], analysis_utils.get_rms_from_histogram(data[index, :], x), 0.0]
            bounds = [[0.0, x.min(), 0.0, 0.0, x.min(), 0.0, 0.0], [2.0 * A_peak[index], x.max(), x.max() - x.min(), 2.0 * A_peak[index], x.max(), np.inf, A_peak[index]]]

        # Fit correlation
        if fit_background:  # Describe background with addidional gauss + offset
            try:
                coeff, var_matrix = curve_fit(analysis_utils.double_gauss_offset, x, data[index, :], p0=p0, bounds=bounds)
            except RuntimeError:  # curve_fit failed
                fit_converged = False
            else:
                fit_converged = True
                # do some result checks
                if not signal_sanity_check(coeff, s_n, A_peak[index]):
                    logging.debug('No correlation peak found. Try another fit...')
                    # Use parameters from last fit as start parameters for the refit
                    y_fit = analysis_utils.double_gauss_offset(x, *coeff)
                    try:
                        coeff, var_matrix = refit_advanced(x_data=x, y_data=data[index, :], y_fit=y_fit, p0=coeff)
                    except RuntimeError:  # curve_fit failed
                        fit_converged = False
                    else:
                        fit_converged = True
                        # Check result again:
                        if not signal_sanity_check(coeff, s_n, A_peak[index]):
                            logging.debug('No correlation peak found after refit!')
                            fit_converged = False

        else:  # Describe background with offset only.
            # Change start parameters and boundaries
            p0_gauss_offset = [p0_val for i, p0_val in enumerate(p0) if i in (0, 1, 2, 6)]
            bounds_gauss_offset = [0, np.inf]
            bounds_gauss_offset[0] = [bound_val for i, bound_val in enumerate(bounds[0]) if i in (0, 1, 2, 6)]
            bounds_gauss_offset[1] = [bound_val for i, bound_val in enumerate(bounds[1]) if i in (0, 1, 2, 6)]
            try:
                coeff_gauss_offset, var_matrix = curve_fit(analysis_utils.gauss_offset, x, data[index, :], p0=p0_gauss_offset, bounds=bounds_gauss_offset)
            except RuntimeError:  # curve_fit failed
                fit_converged = False
            else:
                # Correlation should have at least 2 entries to avoid random fluctuation peaks to be selected
                if coeff_gauss_offset[0] > 2:
                    fit_converged = True
                    # Change back coefficents
                    coeff = np.insert(coeff_gauss_offset, 3, [np.nan] * 3)  # Parameters: A_1, mu_1, sigma_1, A_2, mu_2, sigma_2, offset
                else:
                    fit_converged = False

        # Set fit results for given index if successful
        if fit_converged:
            coeff_fitted[index] = coeff
            mean_fitted[index] = coeff[1]
            mean_error_fitted[index] = np.sqrt(np.abs(np.diag(var_matrix)))[1]
            sigma_fitted[index] = np.abs(coeff[2])
            chi2[index] = analysis_utils.get_chi2(y_data=data[index, :], y_fit=analysis_utils.double_gauss_offset(x, *coeff))

    if no_correlation_indices:
        logging.info('No correlation entries for indices %s. Omit correlation fit.', str(no_correlation_indices)[1:-1])

    if few_correlation_indices:
        logging.info('Very few correlation entries for indices %s. Omit correlation fit.', str(few_correlation_indices)[1:-1])


def refit_advanced(x_data, y_data, y_fit, p0):
    ''' Substract the fit from the data, thus only the small signal peak should be left.
    Fit this peak, and refit everything with start values'''
    y_peak = y_data - y_fit  # Fit most likely only describes background, thus substract it
    peak_A = np.max(y_peak)  # Determine start value for amplitude
    peak_mu = np.argmax(y_peak)  # Determine start value for mu
    fwhm_1, fwhm_2 = analysis_utils.fwhm(x_data, y_peak)
    peak_sigma = (fwhm_2 - fwhm_1) / 2.35  # Determine start value for sigma

    # Fit a Gauss + Offset to the background substracted data
    coeff_peak, _ = curve_fit(analysis_utils.gauss_offset_slope, x_data, y_peak, p0=[peak_A, peak_mu, peak_sigma, 0.0, 0.0], bounds=([0.0, 0.0, 0.0, -10000.0, -10.0], [1.1 * peak_A, np.inf, np.inf, 10000.0, 10.0]))

    # Refit orignial double Gauss function with proper start values for the small signal peak
    coeff, var_matrix = curve_fit(analysis_utils.double_gauss_offset, x_data, y_data, p0=[coeff_peak[0], coeff_peak[1], coeff_peak[2], p0[3], p0[4], p0[5], p0[6]], bounds=[0.0, np.inf])

    return coeff, var_matrix


def apply_alignment(input_hit_file, input_alignment_file, output_hit_file, inverse=False,
                    force_prealignment=False, no_z=False, use_duts=None, chunk_size=1000000):
    ''' Takes a file with tables containing hit information (x, y, z) and applies the alignment to each DUT hit (positions and errors).
    The alignment data is used. If this is not available a fallback to the pre-alignment is done.
    One can also inverse the alignment or apply the alignment without changing the z position.

    Note:
    -----
    This function cannot be easily made faster with multiprocessing since the computation function (apply_alignment_to_chunk) does not
    contribute significantly to the runtime (< 20 %), but the copy overhead for not shared memory needed for multipgrocessing is higher.
    Also the hard drive IO can be limiting (30 Mb/s read, 20 Mb/s write to the same disk)

    Parameters
    ----------
    input_hit_file : string
        Filename of the input hits file (e.g. merged data file, tracklets file, etc.).
    input_alignment_file : string
        Filename of the input alignment file.
    output_hit_file : string
        Filename of the output hits file with hit data after alignment was applied.
    inverse : bool
        If True, apply the inverse alignment.
    force_prealignment : bool
        If True, use pre-alignment, even if alignment data is availale.
    no_z : bool
        If True, do not change the z alignment. Needed since the z position is special for x / y based plane measurements.
    use_duts : iterable
        Iterable of DUT indices to apply the alignment to. If None, use all DUTs.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('== Apply alignment to %s ==', input_hit_file)

    use_prealignment = True if force_prealignment else False

    try:
        with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
            if use_prealignment:
                logging.info('Use pre-alignment data')
                prealignment = in_file_h5.root.PreAlignment[:]
                n_duts = prealignment.shape[0]
            else:
                logging.info('Use alignment data')
                alignment = in_file_h5.root.Alignment[:]
                n_duts = alignment.shape[0]
    except TypeError:  # The input_alignment_file is an array
        alignment = input_alignment_file
        try:  # Check if array is prealignent array
            alignment['column_c0']
            logging.info('Use pre-alignment data')
            n_duts = prealignment.shape[0]
            use_prealignment = True
        except ValueError:
            logging.info('Use alignment data')
            n_duts = alignment.shape[0]
            use_prealignment = False

    def apply_alignment_to_chunk(hits_chunk, dut_index, use_prealignment, alignment, inverse, no_z):
        if use_prealignment:  # Apply transformation from pre-alignment information
            (hits_chunk['x_dut_%d' % dut_index],
             hits_chunk['y_dut_%d' % dut_index],
             hit_z,
             hits_chunk['xerr_dut_%d' % dut_index],
             hits_chunk['yerr_dut_%d' % dut_index],
             hits_chunk['zerr_dut_%d' % dut_index]) = geometry_utils.apply_alignment(
                hits_x=hits_chunk['x_dut_%d' % dut_index],
                hits_y=hits_chunk['y_dut_%d' % dut_index],
                hits_z=hits_chunk['z_dut_%d' % dut_index],
                hits_xerr=hits_chunk['xerr_dut_%d' % dut_index],
                hits_yerr=hits_chunk['yerr_dut_%d' % dut_index],
                hits_zerr=hits_chunk['zerr_dut_%d' % dut_index],
                dut_index=dut_index,
                prealignment=prealignment,
                inverse=inverse)
        else:  # Apply transformation from fine alignment information
            (hits_chunk['x_dut_%d' % dut_index],
             hits_chunk['y_dut_%d' % dut_index],
             hit_z,
             hits_chunk['xerr_dut_%d' % dut_index],
             hits_chunk['yerr_dut_%d' % dut_index],
             hits_chunk['zerr_dut_%d' % dut_index]) = geometry_utils.apply_alignment(
                hits_x=hits_chunk['x_dut_%d' % dut_index],
                hits_y=hits_chunk['y_dut_%d' % dut_index],
                hits_z=hits_chunk['z_dut_%d' % dut_index],
                hits_xerr=hits_chunk['xerr_dut_%d' % dut_index],
                hits_yerr=hits_chunk['yerr_dut_%d' % dut_index],
                hits_zerr=hits_chunk['zerr_dut_%d' % dut_index],
                dut_index=dut_index,
                alignment=alignment,
                inverse=inverse)
        if not no_z:
            hits_chunk['z_dut_%d' % dut_index] = hit_z

    # Looper over the hits of all DUTs of all hit tables in chunks and apply the alignment
    with tb.open_file(input_hit_file, mode='r') as in_file_h5:
        with tb.open_file(output_hit_file, mode='w') as out_file_h5:
            for node in in_file_h5.root:  # Loop over potential hit tables in data file
                hits = node
                new_node_name = hits.name

                if new_node_name == 'MergedCluster':  # Merged cluster with alignment are tracklets
                    new_node_name = 'Tracklets'

                hits_aligned_table = out_file_h5.create_table(out_file_h5.root, name=new_node_name, description=np.zeros((1,), dtype=hits.dtype).dtype, title=hits.title, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=hits.shape[0], term_width=80)
                progress_bar.start()

                for hits_chunk, index in analysis_utils.data_aligned_at_events(hits, chunk_size=chunk_size):  # Loop over the hits
                    for dut_index in range(0, n_duts):  # Loop over the DUTs in the hit table
                        if use_duts is not None and dut_index not in use_duts:  # omit DUT
                            continue

                        apply_alignment_to_chunk(hits_chunk=hits_chunk, dut_index=dut_index, use_prealignment=use_prealignment, alignment=prealignment if use_prealignment else alignment, inverse=inverse, no_z=no_z)

                    hits_aligned_table.append(hits_chunk)
                    progress_bar.update(index)
                progress_bar.finish()

    logging.debug('File with realigned hits %s', output_hit_file)


def alignment(input_track_candidates_file, input_alignment_file, n_pixels, pixel_size, align_duts=None, selection_fit_duts=None, selection_hit_duts=None, selection_track_quality=1, alignment_order=None, initial_rotation=None, initial_translation=None, max_iterations=3, use_n_tracks=200000, use_fit_limits=True, new_alignment=True, plot=False, chunk_size=100000):
    ''' This function does an alignment of the DUTs and sets translation and rotation values for all DUTs.
    The reference DUT defines the global coordinate system position at 0, 0, 0 and should be well in the beam and not heavily rotated.

    To solve the chicken-and-egg problem that a good dut alignment needs hits belonging to one track, but good track finding needs a good dut alignment this
    function work only on already prealigned hits belonging to one track. Thus this function can be called only after track finding.

    These steps are done
    1. Take the found tracks and revert the pre-alignment
    2. Take the track hits belonging to one track and fit tracks for all DUTs
    3. Calculate the residuals for each DUT
    4. Deduce rotations from the residuals and apply them to the hits
    5. Deduce the translation of each plane
    6. Store and apply the new alignment

    repeat step 3 - 6 until the total residual does not decrease (RMS_total = sqrt(RMS_x_1^2 + RMS_y_1^2 + RMS_x_2^2 + RMS_y_2^2 + ...))

    Parameters
    ----------
    input_track_candidates_file : string
        file name with the track candidates table
    input_alignment_file : pytables file
        File name of the input aligment data
    n_pixels : iterable of tuples
        One tuple per DUT describing the total number of pixels (column/row),
        e.g. for two FE-I4 DUTs [(80, 336), (80, 336)].
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension (column/row),
        e.g. for two FE-I4 DUTs [(250, 50), (250, 50)].
    align_duts : iterable or iterable of iterable
        The combination of duts that are algined at once. One should always align the high resolution planes first.
        E.g. for a telesope (first and last 3 planes) with 2 devices in the center (3, 4):
        align_duts=[[0, 1, 2, 5, 6, 7],  # align the telescope planes first
        [4],  # Align first DUT
        [3]],  # Align second DUT
    selection_fit_duts : iterable or iterable of iterable
        Defines for each align_duts combination wich devices to use in the track fit.
        E.g. To use only the telescope planes (first and last 3 planes) but not the 2 center devices
        selection_fit_duts=[0, 1, 2, 5, 6, 7]
    selection_hit_duts : iterable or iterable of iterable
        Defines for each align_duts combination wich devices must have a hit to use the track for fitting. The hit
        does not have to be used in the fit itself! This is useful for time reference planes.
        E.g.  To use telescope planes (first and last 3 planes) + time reference plane (3)
        selection_hit_duts = [0, 1, 2, 4, 5, 6, 7]
    selection_track_quality : uint or iterable or iterable of iterable
        Track quality for each hit DUT.
    initial_rotation : array
        Initial rotation array. If None, deduce the rotation from pre-alignment.
    initial_translation : array
        Initial translation array. If None, deduce the translation from pre-alignment.
    max_iterations : uint
        Maximum number of iterations of calc residuals, apply rotation refit loop until constant result is expected.
        Usually the procedure converges rather fast (< 5 iterations)
    use_n_tracks : uint
        Defines the amount of tracks to be used for the alignment. More tracks can potentially make the result
        more precise, but will also increase the calculation time.
    use_fit_limits : bool
        If True, use fit limits from pre-alignment for residual calculation.
    new_alignment : bool
        If True, discard existig alignment parameters from input alignment file and start all over.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Aligning DUTs ===')

    # Open the pre-alignment and create empty alignment info (at the beginning only the z position is set)
    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        prealignment = in_file_h5.root.PreAlignment[:]
        n_duts = prealignment.shape[0]
        alignment_parameters = _create_alignment_array(n_duts)
        alignment_parameters['translation_z'] = prealignment['z']

        if initial_rotation is None:
            for dut_index in range(n_duts):
                alignment_parameters['alpha'][dut_index] = np.pi if np.isclose(-1, prealignment[dut_index]["row_c1"], atol=0.1) else 0.0
                alignment_parameters['beta'][dut_index] = np.pi if np.isclose(-1, prealignment[dut_index]["column_c1"], atol=0.1) else 0.0
#                 alignment_parameters['gamma'][dut_index] = initial_rotation[dut_index][2]
        else:
            if isinstance(initial_rotation[0], Iterable):
                for dut_index in range(n_duts):
                    alignment_parameters['alpha'][dut_index] = initial_rotation[dut_index][0]
                    alignment_parameters['beta'][dut_index] = initial_rotation[dut_index][1]
                    alignment_parameters['gamma'][dut_index] = initial_rotation[dut_index][2]
            else:
                for dut_index in range(n_duts):
                    alignment_parameters['alpha'][dut_index] = initial_rotation[0]
                    alignment_parameters['beta'][dut_index] = initial_rotation[1]
                    alignment_parameters['gamma'][dut_index] = initial_rotation[2]

        if initial_translation is None:
            for dut_index in range(n_duts):
                alignment_parameters['translation_x'][dut_index] = prealignment[dut_index]["column_c0"]
                alignment_parameters['translation_y'][dut_index] = prealignment[dut_index]["row_c0"]
        else:
            if isinstance(initial_translation[0], Iterable):
                for dut_index in range(n_duts):
                    alignment_parameters['translation_x'][dut_index] = initial_translation[dut_index][0]
                    alignment_parameters['translation_y'][dut_index] = initial_translation[dut_index][1]
            else:
                for dut_index in range(n_duts):
                    alignment_parameters['translation_x'][dut_index] = initial_translation[0]
                    alignment_parameters['translation_y'][dut_index] = initial_translation[1]

    # Create list with combinations of DUTs to align
    if align_duts is None:  # If None: align all DUTs
        align_duts = range(n_duts)
    # Check for value errors
    if not isinstance(align_duts, Iterable):
        raise ValueError("align_duts is no iterable")
    elif not align_duts:  # empty iterable
        raise ValueError("align_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), align_duts)):
        align_duts = [align_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), align_duts)):
        raise ValueError("not all items in align_duts are iterable")
    # Finally check length of all iterables in iterable
    for dut in align_duts:
        if not dut:  # check the length of the items
            raise ValueError("item in align_duts has length 0")

    # Check if some DUTs will not be aligned
    all_align_duts = []
    for duts in align_duts:
        all_align_duts.extend(duts)
    no_align_duts = set(range(n_duts)) - set(all_align_duts)
    if no_align_duts:
        logging.warning('These DUTs will not be aligned: %s', ", ".join(str(align_dut) for align_dut in no_align_duts))

    # overwrite configuration for align DUTs
    # keep configuration for DUTs that will not be aligned
    if new_alignment:
        geometry_utils.store_alignment_parameters(
            alignment_file=input_alignment_file,
            alignment_parameters=alignment_parameters,
            select_duts=np.unique(np.hstack(np.array(align_duts))),
            mode='absolute')
    else:
        pass  # do nothing here, keep existing configuration

    # Create track, hit selection
    if selection_hit_duts is None:  # If None: use all DUTs
        selection_hit_duts = []
        # copy each item
        for duts in align_duts:
            selection_hit_duts.append(duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(selection_hit_duts, Iterable):
        raise ValueError("selection_hit_duts is no iterable")
    elif not selection_hit_duts:  # empty iterable
        raise ValueError("selection_hit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), selection_hit_duts)):
        selection_hit_duts = [selection_hit_duts[:] for _ in align_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), selection_hit_duts)):
        raise ValueError("not all items in selection_hit_duts are iterable")
    # Finally check length of all arrays
    if len(selection_hit_duts) != len(align_duts):  # empty iterable
        raise ValueError("selection_hit_duts has the wrong length")
    for hit_dut in selection_hit_duts:
        if len(hit_dut) < 2:  # check the length of the items
            raise ValueError("item in selection_hit_duts has length < 2")

    # Create track, hit selection
    if selection_fit_duts is None:  # If None: use all DUTs
        selection_fit_duts = []
        # copy each item
        for hit_duts in selection_hit_duts:
            selection_fit_duts.append(hit_duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(selection_fit_duts, Iterable):
        raise ValueError("selection_fit_duts is no iterable")
    elif not selection_fit_duts:  # empty iterable
        raise ValueError("selection_fit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), selection_fit_duts)):
        selection_fit_duts = [selection_fit_duts[:] for _ in align_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), selection_fit_duts)):
        raise ValueError("not all items in selection_fit_duts are iterable")
    # Finally check length of all arrays
    if len(selection_fit_duts) != len(align_duts):  # empty iterable
        raise ValueError("selection_fit_duts has the wrong length")
    for index, fit_dut in enumerate(selection_fit_duts):
        if len(fit_dut) < 2:  # check the length of the items
            raise ValueError("item in selection_fit_duts has length < 2")
        if set(fit_dut) - set(selection_hit_duts[index]):  # fit DUTs are required to have a hit
            raise ValueError("DUT in selection_fit_duts is not in selection_hit_duts")

    # Create track, hit selection
    if not isinstance(selection_track_quality, Iterable):  # all items the same, special case for selection_track_quality
        selection_track_quality = [[selection_track_quality] * len(hit_duts) for hit_duts in selection_hit_duts]  # every hit DUTs require a track quality value
    # Check iterable and length
    if not isinstance(selection_track_quality, Iterable):
        raise ValueError("selection_track_quality is no iterable")
    elif not selection_track_quality:  # empty iterable
        raise ValueError("selection_track_quality has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), selection_track_quality)):
        selection_track_quality = [selection_track_quality for _ in align_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), selection_track_quality)):
        raise ValueError("not all items in selection_track_quality are iterable")
    # Finally check length of all arrays
    if len(selection_track_quality) != len(align_duts):  # empty iterable
        raise ValueError("selection_track_quality has the wrong length")
    for index, track_quality in enumerate(selection_track_quality):
        if len(track_quality) != len(selection_hit_duts[index]):  # check the length of each items
            raise ValueError("item in selection_track_quality and selection_hit_duts does not have the same length")

    # Loop over all combinations of DUTs to align, simplest case: use all DUTs at once to align
    # Usual case: align high resolution devices first, then other devices
    for index, actual_align_duts in enumerate(align_duts):
        logging.info('Aligning DUTs: %s', ", ".join(str(dut) for dut in actual_align_duts))

        _duts_alignment(
            track_candidates_file=input_track_candidates_file,
            alignment_file=input_alignment_file,
            align_duts=actual_align_duts,
            selection_fit_duts=selection_fit_duts[index],
            selection_hit_duts=selection_hit_duts[index],
            selection_track_quality=selection_track_quality[index],
            alignment_order=alignment_order,
            n_pixels=n_pixels,
            pixel_size=pixel_size,
            use_n_tracks=use_n_tracks,
            max_iterations=max_iterations,
            use_fit_limits=use_fit_limits,
            plot=plot,
            chunk_size=chunk_size)

    logging.info('Alignment finished successfully!')


def _duts_alignment(track_candidates_file, alignment_file, align_duts, selection_fit_duts, selection_hit_duts, selection_track_quality, alignment_order, n_pixels, pixel_size, use_n_tracks, max_iterations, use_fit_limits=False, plot=True, chunk_size=100000):  # Called for each list of DUTs to align
    alignment_duts = "_".join(str(dut) for dut in align_duts)
    # Step 0: Reduce the number of tracks to increase the calculation time
    logging.info('= Alignment step 0: Reduce number of tracks to %d =', use_n_tracks)
    track_quality_mask = 0
    for index, dut in enumerate(selection_hit_duts):
        for quality in range(3):
            if quality <= selection_track_quality[index]:
                track_quality_mask |= ((1 << dut) << quality * 8)

    logging.info('Use track with hits in DUTs %s', str(selection_hit_duts)[1:-1])
    data_selection.select_hits(hit_file=track_candidates_file,
                               output_file=os.path.splitext(track_candidates_file)[0] + '_reduced_duts_%s.h5' % alignment_duts,
                               max_hits=use_n_tracks,
                               track_quality=track_quality_mask,
                               track_quality_mask=track_quality_mask,
                               chunk_size=chunk_size)
    track_candidates_reduced = os.path.splitext(track_candidates_file)[0] + '_reduced_duts_%s.h5' % alignment_duts

    # Step 1: Take the found tracks and revert the pre-alignment to start alignment from the beginning
    logging.info('= Alignment step 1: Revert pre-alignment =')
    apply_alignment(input_hit_file=track_candidates_reduced,
                    input_alignment_file=alignment_file,  # Revert prealignent
                    output_hit_file=os.path.splitext(track_candidates_reduced)[0] + '_not_aligned.h5',
                    inverse=True,
                    force_prealignment=True,
                    chunk_size=chunk_size)

    # Stage N: Repeat alignment with constrained residuals until total residual does not decrease anymore
    _calculate_translation_alignment(track_candidates_file=os.path.splitext(track_candidates_reduced)[0] + '_not_aligned.h5',
                                     alignment_file=alignment_file,
                                     use_duts=align_duts,  # Only use the actual DUTs to align
                                     selection_fit_duts=selection_fit_duts,
                                     selection_hit_duts=selection_hit_duts,
                                     selection_track_quality=selection_track_quality,
                                     alignment_order=alignment_order,
                                     n_pixels=n_pixels,
                                     pixel_size=pixel_size,
                                     max_iterations=max_iterations,
                                     plot_title_prefix='',
                                     use_fit_limits=use_fit_limits,
                                     output_pdf=None,
                                     chunk_size=chunk_size)

    # Plot final result
    if plot:
        logging.info('= Alignment step 7: Plot final result =')
        with PdfPages(os.path.join(os.path.dirname(os.path.realpath(track_candidates_file)), 'Alignment_duts_%s.pdf' % alignment_duts), keep_empty=False) as output_pdf:
            # Apply final alignment result
            apply_alignment(input_hit_file=os.path.splitext(track_candidates_reduced)[0] + '_not_aligned.h5',
                            input_alignment_file=alignment_file,
                            output_hit_file=os.path.splitext(track_candidates_file)[0] + '_final_tmp_duts_%s.h5' % alignment_duts,
                            chunk_size=chunk_size)
            fit_tracks(input_track_candidates_file=os.path.splitext(track_candidates_file)[0] + '_final_tmp_duts_%s.h5' % alignment_duts,
                       input_alignment_file=alignment_file,
                       output_tracks_file=os.path.splitext(track_candidates_file)[0] + '_tracks_final_tmp_duts_%s.h5' % alignment_duts,
                       fit_duts=align_duts,  # Only create residuals of selected DUTs
                       selection_fit_duts=selection_fit_duts,  # Only use selected duts
                       selection_hit_duts=selection_hit_duts,
                       exclude_dut_hit=True,  # For unconstrained residuals
                       selection_track_quality=selection_track_quality,
                       chunk_size=chunk_size)
            calculate_residuals(input_tracks_file=os.path.splitext(track_candidates_file)[0] + '_tracks_final_tmp_duts_%s.h5' % alignment_duts,
                                input_alignment_file=alignment_file,
                                output_residuals_file=os.path.splitext(track_candidates_file)[0] + '_residuals_final_tmp_duts_%s.h5' % alignment_duts,
                                n_pixels=n_pixels,
                                pixel_size=pixel_size,
                                use_fit_limits=use_fit_limits,
                                plot=plot,
                                chunk_size=chunk_size)

            # remove temporary files for plotting
            os.remove(os.path.splitext(track_candidates_file)[0] + '_final_tmp_duts_%s.h5' % alignment_duts)
            os.remove(os.path.splitext(track_candidates_file)[0] + '_tracks_final_tmp_duts_%s.h5' % alignment_duts)
            os.remove(os.path.splitext(track_candidates_file)[0] + '_tracks_final_tmp_duts_%s.pdf' % alignment_duts)
            os.remove(os.path.splitext(track_candidates_file)[0] + '_residuals_final_tmp_duts_%s.h5' % alignment_duts)

    # remove temporary files
    os.remove(os.path.splitext(track_candidates_reduced)[0] + '_not_aligned.h5')
    os.remove(os.path.splitext(track_candidates_file)[0] + '_reduced_duts_%s.h5' % alignment_duts)

def _calculate_translation_alignment(track_candidates_file, alignment_file, use_duts, selection_fit_duts, selection_hit_duts, selection_track_quality, alignment_order, n_pixels, pixel_size, max_iterations, plot_title_prefix='', use_fit_limits=False, output_pdf=None, chunk_size=100000):
    ''' Main function that fits tracks, calculates the residuals, deduces rotation and translation values from the residuals
    and applies the new alignment to the track hits. The alignment result is scored as a combined
    residual value of all planes that are being aligned in x and y weighted by the pixel pitch in x and y. '''
    with tb.open_file(alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        alignment_last_iteration = in_file_h5.root.Alignment[:]

    if alignment_order is None:
#         alignment_order = [["gamma"], ["translation_x", "translation_y", "translation_z"], ["alpha", "beta"], ["translation_x", "translation_y", "translation_z"]]
        alignment_order = [["gamma"], ["translation_x", "translation_y"]]

    total_residual = None

    max_iterations = range(max_iterations)

    for iteration in max_iterations:
        for alignment_index, alignment_parameters in enumerate(alignment_order):
            iteration_step = iteration * len(alignment_order) + alignment_index

            apply_alignment(input_hit_file=track_candidates_file,  # Always apply alignment to starting file
                            input_alignment_file=alignment_file,
                            output_hit_file=os.path.splitext(track_candidates_file)[0] + '_new_alignment_%d_tmp.h5' % (iteration_step),
                            inverse=False,
                            force_prealignment=False,
                            chunk_size=chunk_size)

            # Step 2: Fit tracks for all DUTs
            logging.info('= Alignment step 2 / iteration %d: Fit tracks for all DUTs =', (iteration_step))
            fit_tracks(input_track_candidates_file=os.path.splitext(track_candidates_file)[0] + '_new_alignment_%d_tmp.h5' % (iteration_step),
                       input_alignment_file=alignment_file,
                       output_tracks_file=os.path.splitext(track_candidates_file)[0] + '_tracks_aligned_%d_tmp.h5' % (iteration_step),
                       # TODO: really None?
                       fit_duts=None,  # fit tracks for all DUTs
                       selection_fit_duts=selection_fit_duts,   # Only use selected DUTs for track fit
                       selection_hit_duts=selection_hit_duts,  # Only use selected duts
                       exclude_dut_hit=False,  # For constrained residuals
                       selection_track_quality=selection_track_quality,
                       force_prealignment=False,
                       chunk_size=chunk_size)

            # Step 3: Calculate the residuals for each DUT
            logging.info('= Alignment step 3 / iteration %d: Calculate the residuals for each selected DUT =', (iteration_step))
            calculate_residuals(input_tracks_file=os.path.splitext(track_candidates_file)[0] + '_tracks_aligned_%d_tmp.h5' % (iteration_step),
                                input_alignment_file=alignment_file,
                                output_residuals_file=os.path.splitext(track_candidates_file)[0] + '_residuals_aligned_%d_tmp.h5' % (iteration_step),
                                n_pixels=n_pixels,
                                pixel_size=pixel_size,
                                force_prealignment=False,
                                use_duts=use_duts,
                                # smaller devices needs None, otherwise npixels_per_bin=5 and nbins_per_pixel=1 might improve the first step
                                npixels_per_bin=None,
                                nbins_per_pixel=None,
                                use_fit_limits=use_fit_limits,
                                # set to True for residual plot
                                plot=False,
                                chunk_size=chunk_size)

            # Step 4: Deduce rotations from the residuals
            logging.info('= Alignment step 4 / iteration %d: Deduce rotations and translations from the residuals =', (iteration_step))
            alignment_parameters_changed, new_total_residual = _analyze_residuals(residuals_file=os.path.splitext(track_candidates_file)[0] + '_residuals_aligned_%d_tmp.h5' % (iteration_step),
                                                                                  alignment_file=alignment_file,
                                                                                  use_duts=use_duts,  # fit all duts currently beeing investigated
                                                                                  pixel_size=pixel_size,
                                                                                  translation_only=False,
                                                                                  plot_title_prefix=plot_title_prefix,
                                                                                  relaxation_factor=1.0,  # FIXME: good code practice: nothing hardcoded
                                                                                  output_pdf=output_pdf)

            # Create actual alignment (old alignment + the actual relative change)
            new_alignment_parameters = geometry_utils.merge_alignment_parameters(
                    old_alignment=alignment_last_iteration,
                    new_alignment=alignment_parameters_changed,
                    select_duts=use_duts,
                    parameters=alignment_parameters,
                    mode='relative')

            # FIXME: This step does not work well
#             # Step 5: Try to find better rotation by minimizing the residual in x + y for different angles
#             logging.info('= Alignment step 5 / iteration %d: Optimize alignment by minimizing residuals =', iteration)
#             new_alignment_parameters, new_total_residual = _optimize_alignment(tracks_file=os.path.splitext(track_candidates_file)[0] + '_tracks_%d_tmp.h5' % iteration,
#                                                                                alignment_last_iteration=alignment_last_iteration,
#                                                                                new_alignment_parameters=new_alignment_parameters,
#                                                                                pixel_size=pixel_size)

            # Delete temporary files
            os.remove(os.path.splitext(track_candidates_file)[0] + '_new_alignment_%d_tmp.h5' % (iteration_step))
            os.remove(os.path.splitext(track_candidates_file)[0] + '_tracks_aligned_%d_tmp.h5' % (iteration_step))
            os.remove(os.path.splitext(track_candidates_file)[0] + '_tracks_aligned_%d_tmp.pdf' % (iteration_step))
            os.remove(os.path.splitext(track_candidates_file)[0] + '_residuals_aligned_%d_tmp.h5' % (iteration_step))
            logging.info('Total residual %1.4e', new_total_residual)

            logging.info('= Alignment step 6 / iteration %d: Set new rotation / translation information in alignment file =', (iteration_step))
            geometry_utils.store_alignment_parameters(alignment_file=alignment_file,
                                                      alignment_parameters=new_alignment_parameters, # alignment_parameters_changed,
                                                      mode='absolute',
                                                      select_duts=use_duts,
                                                      parameters=["alpha", "beta", "gamma", "translation_x", "translation_y", "translation_z"])

            alignment_last_iteration = new_alignment_parameters.copy()


# Helper functions for the alignment. Not to be used directly.
def _create_alignment_array(n_duts):
    # Result Translation / rotation table
    description = [('DUT', np.int32)]
    description.append(('translation_x', np.float))
    description.append(('translation_y', np.float))
    description.append(('translation_z', np.float))
    description.append(('alpha', np.float))
    description.append(('beta', np.float))
    description.append(('gamma', np.float))
    description.append(('correlation_x', np.float))
    description.append(('correlation_y', np.float))

    array = np.zeros((n_duts,), dtype=description)
    array[:]['DUT'] = np.array(range(n_duts))
    return array


def _analyze_residuals(residuals_file, alignment_file, use_duts, pixel_size, translation_only=False, relaxation_factor=1.0, plot_title_prefix='', output_pdf=None):
    ''' Take the residual plots and deduce rotation and translation angles from them '''
    with tb.open_file(alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        alignment = in_file_h5.root.Alignment[:]

    n_duts = alignment.shape[0]
    alignment_parameters = _create_alignment_array(n_duts)

    total_residual = 0  # Sum of all residuals to judge the overall alignment

    with tb.open_file(residuals_file) as in_file_h5:
        for dut_index in use_duts:
            mirror_y = np.sign(np.cos(alignment[dut_index]["alpha"]))
            mirror_x = np.sign(np.cos(alignment[dut_index]["beta"]))

            alignment_parameters[dut_index]['DUT'] = dut_index
            # Global residuals
            hist_node = in_file_h5.get_node('/ResidualsX_DUT%d' % dut_index)
            std_x = hist_node._v_attrs.fit_coeff[2]

            # Add resdidual to total residual normalized to pixel pitch in x
            total_residual = np.sqrt(np.square(total_residual) + np.square(std_x / pixel_size[dut_index][0]))

            if output_pdf is not None:
                plot_utils.plot_residuals(histogram=hist_node[:],
                                          edges=hist_node._v_attrs.xedges,
                                          fit=hist_node._v_attrs.fit_coeff,
                                          fit_errors=hist_node._v_attrs.fit_cov,
                                          title='Residuals for DUT%d' % dut_index,
                                          x_label='X residual [um]',
                                          output_pdf=output_pdf)

            hist_node = in_file_h5.get_node('/ResidualsY_DUT%d' % dut_index)
            std_y = hist_node._v_attrs.fit_coeff[2]

            # Add resdidual to total residual normalized to pixel pitch in y
            total_residual = np.sqrt(np.square(total_residual) + np.square(std_y / pixel_size[dut_index][1]))

            if translation_only:
                return alignment_parameters, total_residual

            if output_pdf is not None:
                plot_utils.plot_residuals(histogram=hist_node[:],
                                          edges=hist_node._v_attrs.xedges,
                                          fit=hist_node._v_attrs.fit_coeff,
                                          fit_errors=hist_node._v_attrs.fit_cov,
                                          title='Residuals for DUT%d' % dut_index,
                                          x_label='Y residual [um]',
                                          output_pdf=output_pdf)

            # use offset at origin of sensor (center of sensor) to calculate x and y correction
            # do not use mean/median of 1D residual since it depends on the beam spot position when the device is rotated
            # use local residuals
            mu_x = in_file_h5.get_node_attr('/RowResidualsCol_DUT%d' % dut_index, 'fit_coeff')[0]
            mu_y = in_file_h5.get_node_attr('/ColResidualsRow_DUT%d' % dut_index, 'fit_coeff')[0]
            # use slope to calculate alpha, beta and gamma
            m_xx = in_file_h5.get_node_attr('/ColResidualsCol_DUT%d' % dut_index, 'fit_coeff')[1]
            m_yy = in_file_h5.get_node_attr('/RowResidualsRow_DUT%d' % dut_index, 'fit_coeff')[1]
            m_xy = mirror_x * mirror_y * in_file_h5.get_node_attr('/ColResidualsRow_DUT%d' % dut_index, 'fit_coeff')[1]
            m_yx = mirror_x * mirror_y * in_file_h5.get_node_attr('/RowResidualsCol_DUT%d' % dut_index, 'fit_coeff')[1]

            alpha, beta, gamma = analysis_utils.get_rotation_from_residual_fit(m_xx=m_xx, m_xy=m_xy, m_yx=m_yx, m_yy=m_yy, mirror_x=mirror_x, mirror_y=mirror_y)

            alignment_parameters[dut_index]['correlation_x'] = std_x
            alignment_parameters[dut_index]['translation_x'] = -1 * mirror_x * mu_x  # local
            alignment_parameters[dut_index]['correlation_y'] = std_y
            alignment_parameters[dut_index]['translation_y'] = -1 * mirror_y * mu_y  # local
            alignment_parameters[dut_index]['alpha'] = alpha * relaxation_factor
            alignment_parameters[dut_index]['beta'] = beta * relaxation_factor
            alignment_parameters[dut_index]['gamma'] = gamma * relaxation_factor

    return alignment_parameters, total_residual


def _optimize_alignment(tracks_file, alignment_last_iteration, new_alignment_parameters, pixel_size):
    ''' Changes the angles of a virtual plane such that the projected track intersections onto this virtual plane
    are most close to the measured hits on the real DUT at this position. Then the angles of the virtual plane
    should correspond to the real DUT angles. The distance is not weighted quadratically (RMS) but linearly since
    this leads to better results (most likely heavily scattered tracks / beam angle spread at the edges are weighted less).'''
    # Create new absolute alignment
    alignment_result = new_alignment_parameters

    def _minimize_me(align, dut_position, hit_x_local, hit_y_local, hit_z_local, pixel_size, offsets, slopes):
        # Calculate intersections with a dut plane given by alpha, beta, gamma at the dut_position in the global coordinate system
        rotation_matrix = geometry_utils.rotation_matrix(alpha=align[0],
                                                         beta=align[1],
                                                         gamma=align[2])
        basis_global = rotation_matrix.T.dot(np.eye(3))
        dut_plane_normal = basis_global[2]
        actual_dut_position = dut_position.copy()
        actual_dut_position[2] = align[3] * 1e6  # Convert z position from m to um
        intersections = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
                                                                         line_directions=slopes,
                                                                         position_plane=actual_dut_position,
                                                                         normal_plane=dut_plane_normal)

        # Transform to the local coordinate system to compare with measured hits
        transformation_matrix = geometry_utils.global_to_local_transformation_matrix(x=actual_dut_position[0],
                                                                                     y=actual_dut_position[1],
                                                                                     z=actual_dut_position[2],
                                                                                     alpha=align[0],
                                                                                     beta=align[1],
                                                                                     gamma=align[2])

        intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_transformation_matrix(x=intersections[:, 0],
                                                                                                                      y=intersections[:, 1],
                                                                                                                      z=intersections[:, 2],
                                                                                                                      transformation_matrix=transformation_matrix)

        # Cross check if transformations are correct (z == 0 in the local coordinate system)
        if not np.allclose(hit_z_local[np.isfinite(hit_z_local)], 0) or not np.allclose(intersection_z_local, 0):
            logging.error('Hit z position = %s and z intersection %s',
                          str(hit_z_local[~np.isclose(hit_z_local, 0)][:3]),
                          str(intersection_z_local[~np.isclose(intersection_z_local, 0)][:3]))
            raise RuntimeError('The transformation to the local coordinate system did not give all z = 0. Wrong alignment used?')

        return np.sum(np.abs(hit_x_local - intersection_x_local) / pixel_size[0]) + np.sum(np.abs(hit_y_local - intersection_y_local)) / pixel_size[1]
#         return np.sqrt(np.square(np.std(hit_x_local - intersection_x_local) / pixel_size[0]) + np.square(np.std(hit_y_local - intersection_y_local)) / pixel_size[1])

    with tb.open_file(tracks_file, mode='r') as in_file_h5:
        residuals_before = []
        residuals_after = []
        for node in in_file_h5.root:
            actual_dut = int(re.findall(r'\d+', node.name)[-1])
            dut_position = np.array([alignment_last_iteration[actual_dut]['translation_x'], alignment_last_iteration[actual_dut]['translation_y'], alignment_last_iteration[actual_dut]['translation_z']])

            # Hits with the actual alignment
            hits = np.column_stack((node[:]['x_dut_%d' % actual_dut], node[:]['y_dut_%d' % actual_dut], node[:]['z_dut_%d' % actual_dut]))

            # Transform hits to the local coordinate system
            hit_x_local, hit_y_local, hit_z_local = geometry_utils.apply_alignment(hits_x=hits[:, 0],
                                                                                   hits_y=hits[:, 1],
                                                                                   hits_z=hits[:, 2],
                                                                                   dut_index=actual_dut,
                                                                                   alignment=alignment_last_iteration,
                                                                                   inverse=True)

            # Track infos
            offsets = np.column_stack((node[:]['offset_0'], node[:]['offset_1'], node[:]['offset_2']))
            slopes = np.column_stack((node[:]['slope_0'], node[:]['slope_1'], node[:]['slope_2']))

            # Rotation start values of minimizer
            alpha = alignment_result[actual_dut]['alpha']
            beta = alignment_result[actual_dut]['beta']
            gamma = alignment_result[actual_dut]['gamma']
            z_position = alignment_result[actual_dut]['translation_z']

            # Trick to have the same order of magnitue of variation for angles and position, otherwise scipy minimizers
            # do not converge if step size of parameters is very different
            z_position_in_m = z_position / 1e6

            residual = _minimize_me(np.array([alpha, beta, gamma, z_position_in_m]),
                                    dut_position,
                                    hit_x_local,
                                    hit_y_local,
                                    hit_z_local,
                                    pixel_size[actual_dut],
                                    offsets,
                                    slopes)
            residuals_before.append(residual)
            logging.info('Optimize angles / z of DUT%d with start parameters: %1.2e, %1.2e, %1.2e Rad and z = %d um with residual %1.2e' % (actual_dut,
                                                                                                                                            alpha,
                                                                                                                                            beta,
                                                                                                                                            gamma,
                                                                                                                                            z_position_in_m * 1e6,
                                                                                                                                            residual))

            # FIXME:
            # Has to be heavily restricted otherwise converges to unphysical solutions since the scoring with residuals is not really working well
            bounds = [(alpha - 0.01, alpha + 0.01), (beta - 0.01, beta + 0.01), (gamma - 0.001, gamma + 0.001), (z_position_in_m - 10e-6, z_position_in_m + 10e-6)]

            result = minimize(fun=_minimize_me,
                              x0=np.array([alpha, beta, gamma, z_position_in_m]),  # Start values from residual fit
                              args=(dut_position, hit_x_local, hit_y_local, hit_z_local, pixel_size[actual_dut], offsets, slopes),
                              bounds=bounds,
                              method='SLSQP')

            alpha, beta, gamma, z_position_in_m = result.x
            residual = _minimize_me(result.x,
                                    dut_position,
                                    hit_x_local,
                                    hit_y_local,
                                    hit_z_local,
                                    pixel_size[actual_dut],
                                    offsets,
                                    slopes)
            residuals_after.append(residual)

            logging.info('Found angles of DUT%d with best angles: %1.2e, %1.2e, %1.2e Rad and z = %d um with residual %1.2e' % (actual_dut,
                                                                                                                                alpha,
                                                                                                                                beta,
                                                                                                                                gamma,
                                                                                                                                z_position_in_m * 1e6,
                                                                                                                                residual))
            # Rotation start values of minimizer
            alignment_result[actual_dut]['alpha'] = alpha
            alignment_result[actual_dut]['beta'] = beta
            alignment_result[actual_dut]['gamma'] = gamma
            alignment_result[actual_dut]['translation_z'] = z_position_in_m * 1e6  # convert z position from m to um

    total_residuals_before = np.sqrt(np.sum(np.square(np.array(residuals_before))))
    total_residuals_after = np.sqrt(np.sum(np.square(np.array(residuals_after))))
    logging.info('Reduced the total residuals in the optimization steps from %1.2e to %1.2e', total_residuals_before, total_residuals_after)
    if total_residuals_before < total_residuals_after:
        raise RuntimeError('Alignment optimization did not converge!')

    return alignment_result, total_residuals_after  # Return alignment result and total residual


# Helper functions to be called from multiple processes
def _correlate_cluster(cluster_dut_0, cluster_file, start_index, start_event_number, stop_event_number, column_correlation, row_correlation, chunk_size):
    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open other DUT cluster file
        for actual_dut_cluster, start_index in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Cluster, start_index=start_index, start_event_number=start_event_number, stop_event_number=stop_event_number, chunk_size=chunk_size, fail_on_missing_events=False):  # Loop over the cluster in the actual cluster file in chunks

            analysis_utils.correlate_cluster_on_event_number(data_1=cluster_dut_0,
                                                             data_2=actual_dut_cluster,
                                                             column_corr_hist=column_correlation,
                                                             row_corr_hist=row_correlation)

    return start_index, column_correlation, row_correlation
