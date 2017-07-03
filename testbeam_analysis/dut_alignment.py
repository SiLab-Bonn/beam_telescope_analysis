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
from scipy.optimize import curve_fit, minimize_scalar, leastsq, basinhopping, OptimizeWarning
from matplotlib.backends.backend_pdf import PdfPages

from testbeam_analysis.tools import analysis_utils
from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.tools import data_selection

# Imports for track based alignment
from testbeam_analysis.track_analysis import find_tracks, fit_tracks
from testbeam_analysis.result_analysis import calculate_residuals, histogram_track_angle

warnings.simplefilter("ignore", OptimizeWarning)  # Fit errors are handled internally, turn of warnings


def correlate_cluster(input_cluster_files, output_correlation_file, n_pixels, ref_dut=0, pixel_size=None, dut_names=None, plot=True, chunk_size=4999999):
    '''"Calculates the correlation histograms from the cluster arrays.
    The 2D correlation array of pairs of two different devices are created on event basis.
    All permutations are considered (all clusters of the first device are correlated with all clusters of the second device).

    Parameters
    ----------
    input_cluster_files : iterable
        Iterable of filenames of the cluster files.
        The DUT index of each input file is equivalent to its position in the list.
    output_correlation_file : string
        Filename of the output correlation file with the correlation histograms.
    n_pixels : iterable of tuples
        One tuple per DUT describing the total number of pixels (column/row),
        e.g. for two FE-I4 DUTs [(80, 336), (80, 336)].
    ref_dut : uint
        Selecting the reference DUT. Default is DUT 0.
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

    ref_index = ref_dut

    n_duts = len(input_cluster_files)

    # remove reference DUT from list of DUTs
    select_duts = list(set(range(n_duts)) - set([ref_index]))

    with tb.open_file(output_correlation_file, mode="w") as out_file_h5:
        # Result arrays to be filled
        column_correlations = []
        row_correlations = []
        start_indices = []
        for dut_index in select_duts:
            shape_column = (n_pixels[dut_index][0], n_pixels[ref_index][0])
            shape_row = (n_pixels[dut_index][1], n_pixels[ref_index][1])
            column_correlations.append(np.zeros(shape_column, dtype=np.int32))
            row_correlations.append(np.zeros(shape_row, dtype=np.int32))
            start_indices.append(None)  # Store the loop indices for speed up

        with tb.open_file(input_cluster_files[ref_index], mode='r') as in_file_h5:  # Open DUT0 cluster file
            widgets = ['', progressbar.Percentage(), ' ',
                       progressbar.Bar(marker='*', left='|', right='|'),
                       ' ', progressbar.AdaptiveETA()]
            progress_bar = progressbar.ProgressBar(widgets=widgets,
                                                   maxval=in_file_h5.root.Cluster.shape[0],
                                                   term_width=80)
            progress_bar.start()

            pool = Pool()
            # Loop over the clusters of reference DUT
            for clusters_ref_dut, ref_read_index in analysis_utils.data_aligned_at_events(in_file_h5.root.Cluster, chunk_size=chunk_size):
                actual_event_numbers = clusters_ref_dut[:]['event_number']

                # Create correlation histograms to the reference device for all other devices
                # Do this in parallel to safe time

                dut_results = []
                # Loop over other DUTs
                for index, dut_index in enumerate(select_duts):
                    dut_results.append(pool.apply_async(_correlate_cluster, kwds={'clusters_ref_dut': clusters_ref_dut,
                                                                                  'cluster_file': input_cluster_files[dut_index],
                                                                                  'start_index': start_indices[index],
                                                                                  'start_event_number': actual_event_numbers[0],
                                                                                  'stop_event_number': actual_event_numbers[-1] + 1,
                                                                                  'column_correlation': column_correlations[index],
                                                                                  'row_correlation': row_correlations[index],
                                                                                  'chunk_size': chunk_size}))
                # Collect results when available
                for index, dut_result in enumerate(dut_results):
                    (start_indices[index], column_correlations[index], row_correlations[index]) = dut_result.get()

                progress_bar.update(ref_read_index)

            pool.close()
            pool.join()

        # Store the correlation histograms
        for index, dut_index in enumerate(select_duts):
            out_col = out_file_h5.create_carray(where=out_file_h5.root,
                                                name='Correlation_column_%d_%d' % (ref_index, dut_index),
                                                title='Column Correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                                                atom=tb.Atom.from_dtype(column_correlations[index].dtype),
                                                shape=column_correlations[index].shape,
                                                filters=tb.Filters(complib='blosc',
                                                                   complevel=5,
                                                                   fletcher32=False))
            out_row = out_file_h5.create_carray(where=out_file_h5.root,
                                                name='Correlation_row_%d_%d' % (ref_index, dut_index),
                                                title='Row Correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                                                atom=tb.Atom.from_dtype(row_correlations[index].dtype),
                                                shape=row_correlations[index].shape,
                                                filters=tb.Filters(complib='blosc',
                                                                   complevel=5,
                                                                   fletcher32=False))
            out_col_reduced = out_file_h5.create_carray(where=out_file_h5.root,
                                                        name='Reduced_background_correlation_column_%d_%d' % (ref_index, dut_index),
                                                        title='Column Correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                                                        atom=tb.Atom.from_dtype(column_correlations[index].dtype),
                                                        shape=column_correlations[index].shape,
                                                        filters=tb.Filters(complib='blosc',
                                                                           complevel=5,
                                                                           fletcher32=False))
            out_row_reduced = out_file_h5.create_carray(where=out_file_h5.root,
                                                        name='Reduced_background_correlation_row_%d_%d' % (ref_index, dut_index),
                                                        title='Row Correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                                                        atom=tb.Atom.from_dtype(row_correlations[index].dtype),
                                                        shape=row_correlations[index].shape,
                                                        filters=tb.Filters(complib='blosc',
                                                                           complevel=5,
                                                                           fletcher32=False))
            out_col.attrs.filenames = [str(input_cluster_files[ref_index]), str(input_cluster_files[dut_index])]
            out_row.attrs.filenames = [str(input_cluster_files[ref_index]), str(input_cluster_files[dut_index])]
            out_col_reduced.attrs.filenames = [str(input_cluster_files[ref_index]), str(input_cluster_files[dut_index])]
            out_row_reduced.attrs.filenames = [str(input_cluster_files[ref_index]), str(input_cluster_files[dut_index])]
            out_col[:] = column_correlations[index]
            out_row[:] = row_correlations[index]
            for correlation in [column_correlations[index], row_correlations[index]]:
                uu, dd, vv = np.linalg.svd(correlation)  # sigular value decomposition
                background = np.matrix(uu[:, :1]) * np.diag(dd[:1]) * np.matrix(vv[:1, :])  # take first sigular value for background
                background = np.array(background, dtype=np.int32)  # make Numpy array
                correlation -= background  # remove background
                correlation -= correlation.min()  # only positive values
            out_col_reduced[:] = column_correlations[index]
            out_row_reduced[:] = row_correlations[index]
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
        description.append(('n_hits_dut_%d' % index, np.uint32))
    for index, _ in enumerate(input_cluster_files):
        description.append(('n_cluster_dut_%d' % index, np.uint32))
    description.extend([('hit_flag', np.uint16), ('quality_flag', np.uint16), ('n_tracks', np.uint32)])
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
        merged_cluster_table = out_file_h5.create_table(out_file_h5.root, name='MergedCluster', description=np.dtype(description), title='Merged cluster on event number', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
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
                # TODO
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
                merged_cluster_array['n_cluster_dut_0'][selection] = actual_cluster_dut_0['n_cluster'][selection]

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
                            merged_cluster_array['n_cluster_dut_%d' % (dut_index)][selection] = actual_cluster_dut['n_cluster'][selection]

                merged_cluster_table.append(merged_cluster_array)
                merged_cluster_table.flush()
                actual_start_event_number = common_event_numbers[-1] + 1  # Set the starting event number for the next chunked read
                progress_bar.update(start_indices_data_loop[0])
            progress_bar.finish()


def prealignment(input_correlation_file, output_alignment_file, z_positions, pixel_size, s_n=0.1, fit_background=False, reduce_background=True, dut_names=None, no_fit=False, non_interactive=True, iterations=3, plot=True, gui=False, queue=False):
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
        max_dut_index = None
        ref_index = None
        for node in in_file_h5.root:
            if "correlation" in node.name:
                indices = re.findall(r'\d+', node.name)
                if len(indices) == 2:
                    max_dut_index = max(int(indices[0]), max_dut_index)
                    max_dut_index = max(int(indices[1]), max_dut_index)
                    if ref_index is None:
                        ref_index = int(indices[0])
                    elif ref_index != int(indices[0]):
                        raise ValueError('Referece DUT cannot be determined in file %s.' % input_correlation_file)
        if max_dut_index is None:
            raise ValueError('Can\'t find correlation data in file %s.' % input_correlation_file)

        n_duts = max_dut_index + 1
        prealignment = np.full(shape=(n_duts,),
                               dtype=[('DUT', np.uint8),
                                      ('column_c0', np.float),
                                      ('column_c0_error', np.float),
                                      ('column_c1', np.float),
                                      ('column_c1_error', np.float),
                                      ('column_sigma', np.float),
                                      ('column_sigma_error', np.float),
                                      ('row_c0', np.float),
                                      ('row_c0_error', np.float),
                                      ('row_c1', np.float),
                                      ('row_c1_error', np.float),
                                      ('row_sigma', np.float),
                                      ('row_sigma_error', np.float),
                                      ('z', np.float)],
                               fill_value=np.nan)
        prealignment['DUT'] = range(n_duts)
        prealignment['z'] = z_positions
        # Set standard settings for reference DUT
        prealignment[ref_index]['column_c0'], prealignment[ref_index]['column_c0_error'] = 0.0, 0.0
        prealignment[ref_index]['column_c1'], prealignment[ref_index]['column_c1_error'] = 1.0, 0.0
        prealignment[ref_index]['row_c0'], prealignment[ref_index]['row_c0_error'] = 0.0, 0.0
        prealignment[ref_index]['row_c1'], prealignment[ref_index]['row_c1_error'] = 1.0, 0.0
        prealignment[ref_index]['column_sigma'], prealignment[ref_index]['column_sigma_error'] = np.nan, np.nan
        prealignment[ref_index]['row_sigma'], prealignment[ref_index]['row_sigma_error'] = np.nan, np.nan

        fit_limits = np.full((n_duts, 2, 2), fill_value=np.nan, dtype=np.float)  # col left, right, row left, right

        # remove reference DUT from list of DUTs
        select_duts = list(set(range(n_duts)) - set([ref_index]))

        for dut_index in select_duts:
            for direction in ["column", "row"]:
                if reduce_background:
                    node = in_file_h5.get_node(in_file_h5.root, 'Reduced_background_correlation_%s_%d_%d' % (direction, ref_index, dut_index))
                else:
                    node = in_file_h5.get_node(in_file_h5.root, 'Correlation_%s_%d_%d' % (direction, ref_index, dut_index))
                dut_name = dut_names[dut_index] if dut_names else ("DUT" + str(dut_index))
                ref_name = dut_names[ref_index] if dut_names else ("DUT" + str(ref_index))
                logging.info('Pre-aligning data from %s', node.name)

                if direction == "column":
                    pixel_size_dut, pixel_size_ref = pixel_size[dut_index][0], pixel_size[ref_index][0]
                else:
                    pixel_size_dut, pixel_size_ref = pixel_size[dut_index][1], pixel_size[ref_index][1]

                # retrieve data
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
                    # TODO: that needs to be addressed when correlation plot is hogh, proabably not needed anymore
                    prealignment[dut_index][direction + '_c0'], prealignment[dut_index][direction + '_c0_error'] = offset_center, np.nan
                    prealignment[dut_index][direction + '_c1'], prealignment[dut_index][direction + '_c1_error'] = slope, np.nan
                    prealignment[dut_index][direction + '_sigma'], prealignment[dut_index][direction + '_sigma_error'] = np.nan, np.nan

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
                                          prefix=direction,
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
                                                                                      prefix=direction,
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
                    prealignment[dut_index][direction + '_c0'], prealignment[dut_index][direction + '_c0_error'] = re_fit[0], np.absolute(re_fit_pcov[0][0]) ** 0.5
                    prealignment[dut_index][direction + '_c1'], prealignment[dut_index][direction + '_c1_error'] = re_fit[1], np.absolute(re_fit_pcov[1][1]) ** 0.5

                    fit_limits[dut_index][0 if direction == "column" else 1] = [x_dut_scaled_selected.min() if x_dut_scaled.min() < x_dut_scaled_selected.min() else np.nan,
                                                                                x_dut_scaled_selected.max() if x_dut_scaled.max() > x_dut_scaled_selected.max() else np.nan]

                    # Calculate mean sigma (is a residual when assuming straight tracks) and its error and store the actual data in result array
                    # This error is needed for track finding and track quality determination
                    mean_sigma = pixel_size_ref * np.mean(np.array(sigma_fitted_selected))
                    mean_sigma_error = pixel_size_ref * np.std(np.array(sigma_fitted_selected)) / np.sqrt(np.array(sigma_fitted_selected).shape[0])

                    prealignment[dut_index][direction + '_sigma'], prealignment[dut_index][direction + '_sigma_error'] = mean_sigma, mean_sigma_error

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
                                                        title="Correlation of %s: %s vs. %s at %s %d" % (direction + "s", ref_name, dut_name, direction, plot_index),
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
                                                     fit_limit=fit_limits[dut_index][0 if direction == "column" else 1],
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
                                                     prefix=direction,
                                                     output_pdf=output_pdf,
                                                     gui=gui,
                                                     figs=figs)

        logging.info('Store pre-alignment in %s', output_alignment_file)
        with tb.open_file(output_alignment_file, mode="w") as out_file_h5:
            result_table = out_file_h5.create_table(where=out_file_h5.root,
                                                    name='PreAlignment',
                                                    description=prealignment.dtype,
                                                    title='Pre-alignment')
            result_table.append(prealignment)
            result_table.attrs.fit_limits = fit_limits

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
                    use_prealignment=False, no_z=False, use_duts=None, chunk_size=1000000):
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
    use_prealignment : bool
        If True, use pre-alignment; if False, use alignment.
    no_z : bool
        If True, do not change the z alignment. Needed since the z position is special for x / y based plane measurements.
    use_duts : iterable
        Iterable of DUT indices to apply the alignment to. If None, use all DUTs.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('== Apply alignment to %s ==', input_hit_file)

    try:
        with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
            if use_prealignment:
                logging.info('Use pre-alignment data')
                alignment = in_file_h5.root.PreAlignment[:]
                n_duts = alignment.shape[0]
            else:
                logging.info('Use alignment data')
                alignment = in_file_h5.root.Alignment[:]
                n_duts = alignment.shape[0]
    except TypeError:  # The input_alignment_file is an array
        alignment = input_alignment_file
        try:  # Check if array is prealignent array
            alignment['column_c0']
            logging.info('Use pre-alignment data')
            n_duts = alignment.shape[0]
            use_prealignment = True
        except ValueError:
            logging.info('Use alignment data')
            n_duts = alignment.shape[0]
            use_prealignment = False

    # Looper over the hits of all DUTs of all hit tables in chunks and apply the alignment
    with tb.open_file(input_hit_file, mode='r') as in_file_h5:
        with tb.open_file(output_hit_file, mode='w') as out_file_h5:
            for node in in_file_h5.root:  # Loop over potential hit tables in data file
                hits = node
                new_node_name = hits.name

                if new_node_name == 'MergedCluster':  # Merged cluster with alignment are tracklets
                    new_node_name = 'Tracklets'

                hits_aligned_table = out_file_h5.create_table(out_file_h5.root, name=new_node_name, description=hits.dtype, title=hits.title, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=hits.shape[0], term_width=80)
                progress_bar.start()

                for hits_chunk, index in analysis_utils.data_aligned_at_events(hits, chunk_size=chunk_size):  # Loop over the hits
                    for dut_index in range(0, n_duts):  # Loop over the DUTs in the hit table
                        if use_duts is not None and dut_index not in use_duts:  # omit DUT
                            continue

                        geometry_utils.apply_alignment_to_hits(hits=hits_chunk, dut_index=dut_index, use_prealignment=use_prealignment, alignment=alignment, inverse=inverse, no_z=no_z)

                    hits_aligned_table.append(hits_chunk)
                    progress_bar.update(index)
                progress_bar.finish()

    logging.debug('File with realigned hits %s', output_hit_file)

# TODO: selection_track_quality to selection_track_quality_sigma
def alignment(input_merged_file, input_alignment_file, n_pixels, pixel_size, dut_names=None, use_prealignment=True, align_duts=None, align_telescope=None, selection_fit_duts=None, selection_hit_duts=None, quality_sigma=5.0, alignment_order=None, initial_rotation=None, initial_translation=None, max_iterations=3, max_events=100000, use_fit_limits=True, new_alignment=True, plot=False, chunk_size=100000):
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
    input_merged_file : string
        Input file name of the merged cluster hit table from all DUTs.
    input_alignment_file : pytables file
        File name of the input aligment data.
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
                    [4],  # align first DUT
                    [3]]  # align second DUT
    align_telescope : iterable
        The telescope will be aligned to the given DUTs. The translation in x and y of these DUTs will not be changed.
        Usually the two outermost telescope DUTs are selected.
    selection_fit_duts : iterable or iterable of iterable
        Defines for each align_duts combination wich devices to use in the track fit.
        E.g. To use only the telescope planes (first and last 3 planes) but not the 2 center devices
        selection_fit_duts=[0, 1, 2, 5, 6, 7]
    selection_hit_duts : iterable or iterable of iterable
        Defines for each align_duts combination wich devices must have a hit to use the track for fitting. The hit
        does not have to be used in the fit itself! This is useful for time reference planes.
        E.g.  To use telescope planes (first and last 3 planes) + time reference plane (3)
        selection_hit_duts = [0, 1, 2, 4, 5, 6, 7]
    quality_sigma : float
        Track quality for each hit DUT.
    initial_rotation : array
        Initial rotation array. If None, deduce the rotation from pre-alignment.
    initial_translation : array
        Initial translation array. If None, deduce the translation from pre-alignment.
    max_iterations : uint
        Maximum number of iterations of calc residuals, apply rotation refit loop until constant result is expected.
        Usually the procedure converges rather fast (< 5 iterations)
    max_events: uint
        Radomly select max_events for alignment. If None, use all events, which might slow down the alignment.
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

        beta = np.arctan(prealignment[align_telescope[-1]]["column_c0"] / prealignment[align_telescope[-1]]["z"])
        alpha = -np.arctan(prealignment[align_telescope[-1]]["row_c0"] / prealignment[align_telescope[-1]]["z"])
        global_rotation = geometry_utils.rotation_matrix(alpha=alpha,
                                                         beta=beta,
                                                         gamma=0.0)

        for dut_index in range(n_duts):
            if (isinstance(initial_translation, Iterable) and initial_translation[dut_index] is None) or initial_translation is None:
                if dut_index in align_telescope:
                    # Telescope is aligned to these DUTs, so align them to the z-axis
                    alignment_parameters['translation_x'][dut_index] = 0.0
                    alignment_parameters['translation_y'][dut_index] = 0.0
                else:
                    # Correct x and y, so that the telescope is aligned to the z-axis
                    aligned_offset = np.dot(global_rotation, np.array([prealignment[dut_index]["column_c0"], prealignment[dut_index]["row_c0"], prealignment[dut_index]["z"]]))
                    alignment_parameters['translation_x'][dut_index] = aligned_offset[0]
                    alignment_parameters['translation_y'][dut_index] = aligned_offset[1]
            elif isinstance(initial_translation, Iterable) and isinstance(initial_translation[dut_index], Iterable) and len(initial_translation[dut_index]) in [2, 3]:
                alignment_parameters['translation_x'][dut_index] = initial_translation[dut_index][0]
                alignment_parameters['translation_y'][dut_index] = initial_translation[dut_index][1]
                if len(initial_translation[dut_index]) == 3 and initial_translation[dut_index][2] is not None:
                    alignment_parameters['translation_z'][dut_index] = initial_translation[dut_index][2]
            else:
                raise ValueError('initial_translation format not supported')

            if (isinstance(initial_rotation, Iterable) and initial_rotation[dut_index] is None) or initial_rotation is None:
                alignment_parameters['alpha'][dut_index] = np.pi if np.isclose(-1, prealignment[dut_index]["row_c1"], atol=0.1) else 0.0
                alignment_parameters['beta'][dut_index] = np.pi if np.isclose(-1, prealignment[dut_index]["column_c1"], atol=0.1) else 0.0
#                 alignment_parameters['gamma'][dut_index] = initial_rotation[dut_index][2]
            elif isinstance(initial_rotation, Iterable) and isinstance(initial_rotation[dut_index], Iterable) and len(initial_rotation[dut_index]) == 3:
                    alignment_parameters['alpha'][dut_index] = initial_rotation[dut_index][0]
                    alignment_parameters['beta'][dut_index] = initial_rotation[dut_index][1]
                    alignment_parameters['gamma'][dut_index] = initial_rotation[dut_index][2]
            else:
                raise ValueError('initial_rotation format not supported')

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
    no_align_duts = set(range(n_duts)) - set(np.unique(np.hstack(np.array(align_duts))).tolist())
    if no_align_duts:
        logging.warning('These DUTs will not be aligned: %s', ", ".join(str(align_dut) for align_dut in no_align_duts))

    # overwrite configuration for align DUTs
    # keep configuration for DUTs that will not be aligned
    if new_alignment:
        geometry_utils.save_alignment_parameters(alignment_file=input_alignment_file,
                                                 alignment_parameters=alignment_parameters,
                                                 select_duts=np.unique(np.hstack(np.array(align_duts))).tolist(),
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

#     # Create track, hit selection
#     if not isinstance(selection_track_quality, Iterable):  # all items the same, special case for selection_track_quality
#         selection_track_quality = [[selection_track_quality] * len(hit_duts) for hit_duts in selection_hit_duts]  # every hit DUTs require a track quality value
#     # Check iterable and length
#     if not isinstance(selection_track_quality, Iterable):
#         raise ValueError("selection_track_quality is no iterable")
#     elif not selection_track_quality:  # empty iterable
#         raise ValueError("selection_track_quality has no items")
#     # Check if only non-iterable in iterable
#     if all(map(lambda val: not isinstance(val, Iterable), selection_track_quality)):
#         selection_track_quality = [selection_track_quality for _ in align_duts]
#     # Check if only iterable in iterable
#     if not all(map(lambda val: isinstance(val, Iterable), selection_track_quality)):
#         raise ValueError("not all items in selection_track_quality are iterable")
#     # Finally check length of all arrays
#     if len(selection_track_quality) != len(align_duts):  # empty iterable
#         raise ValueError("selection_track_quality has the wrong length")
#     for index, track_quality in enumerate(selection_track_quality):
#         if len(track_quality) != len(selection_hit_duts[index]):  # check the length of each items
#             raise ValueError("item in selection_track_quality and selection_hit_duts does not have the same length")

    if not isinstance(quality_sigma, Iterable):
        quality_sigma = [quality_sigma] * len(align_duts)
    # Finally check length of all arrays
    if len(quality_sigma) != len(align_duts):  # empty iterable
        raise ValueError("quality_sigma has the wrong length")

    if not isinstance(max_iterations, Iterable):
        max_iterations = [max_iterations] * len(align_duts)
    # Finally check length of all arrays
    if len(max_iterations) != len(align_duts):  # empty iterable
        raise ValueError("max_iterations has the wrong length")

    if not isinstance(max_events, Iterable):
        max_events = [max_events] * len(align_duts)
    # Finally check length of all arrays
    if len(max_events) != len(align_duts):  # empty iterable
        raise ValueError("max_events has the wrong length")

    # Loop over all combinations of DUTs to align, simplest case: use all DUTs at once to align
    # Usual case: align high resolution devices first, then other devices
    for index, actual_align_duts in enumerate(align_duts):
        logging.info('Aligning DUTs: %s', ", ".join(str(dut) for dut in actual_align_duts))

        _duts_alignment(
            merged_file=input_merged_file,
            alignment_file=input_alignment_file,
            align_duts=actual_align_duts,
            align_telescope=align_telescope,
            selection_fit_duts=selection_fit_duts[index],
            selection_hit_duts=selection_hit_duts[index],
            quality_sigma=quality_sigma[index],
            alignment_order=alignment_order,
            dut_names=dut_names,
            n_pixels=n_pixels,
            pixel_size=pixel_size,
            max_events=max_events[index],
            max_iterations=max_iterations[index],
            use_prealignment=use_prealignment,
            use_fit_limits=use_fit_limits,
            plot=plot,
            chunk_size=chunk_size)


def _duts_alignment(merged_file, alignment_file, align_duts, align_telescope, selection_fit_duts, selection_hit_duts, quality_sigma, alignment_order, dut_names, n_pixels, pixel_size, max_events, max_iterations, use_prealignment, use_fit_limits=False, plot=True, chunk_size=100000):  # Called for each list of DUTs to align
    alignment_duts = "_".join(str(dut) for dut in align_duts)
    alignment_duts_str = ", ".join(str(dut) for dut in align_duts)

    if alignment_order is None:
        alignment_order = [["alpha", "beta", "gamma", "translation_x", "translation_y", "translation_z"]]

    duts_selection = [list(set([dut]) | set(selection_hit_duts)) for dut in align_duts]
    print "************* duts_selection ****************", duts_selection

#     output_track_candidates_file = os.path.splitext(merged_file)[0] + '_track_candidates_aligned_duts_%s_tmp.h5' % (alignment_duts)
#     logging.info('= Alignment step 0: Finding tracks for DUTs %s =', alignment_duts_str)
#     find_tracks(input_tracklets_file=merged_file,
#                 input_alignment_file=alignment_file,
#                 output_track_candidates_file=output_track_candidates_file,
#                 use_prealignment=True,
#                 max_events=max_events)


    for iteration in range(max_iterations):
        for alignment_index, selected_alignment_parameters in enumerate(alignment_order):
            iteration_step = iteration * len(alignment_order) + alignment_index

#             if iteration == 0 or set(align_duts) & set(selection_fit_duts):
#                 # recalculate tracks if DUT is fit DUT
#                 if iteration != 0:
#                     # remove temporary files before continuing
#                     os.remove(output_tracks_file)
#                     os.remove(output_selected_tracks_file)
            output_tracks_file = os.path.splitext(merged_file)[0] + '_tracks_aligned_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
            output_selected_tracks_file = os.path.splitext(merged_file)[0] + '_tracks_aligned_selected_tracks_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)

            # find tracks in the beginning and each time for telescope/fit DUTs
            # find tracks only once for non-fit/non-telescope DUTs
            if iteration == 0 or (set(align_duts) & set(selection_fit_duts)):
                if iteration != 0:
                    os.remove(output_track_candidates_file)
                output_track_candidates_file = os.path.splitext(merged_file)[0] + '_track_candidates_aligned_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
                # use pre-alignment for fit/telescope DUT and first iteration step to find proper track candidates
                if iteration != 0 or not (set(align_duts) & set(selection_fit_duts)):
                    use_prealignment = False
                else:
                    use_prealignment = True
                print "************* use pre-alignment for find_tracks()", use_prealignment
                logging.info('= Alignment step 0: Finding tracks for DUTs %s =', alignment_duts_str)
                find_tracks(input_merged_file=merged_file,
                            input_alignment_file=alignment_file,
                            output_track_candidates_file=output_track_candidates_file,
                            use_prealignment=use_prealignment,
                            correct_beam_alignment=True,
                            max_events=max_events)

            # Step 2: Fit tracks for all DUTs
            logging.info('= Alignment step 1 / iteration %d: Fitting tracks for DUTs %s =', iteration_step, alignment_duts_str)
            fit_tracks(input_track_candidates_file=output_track_candidates_file,
                       input_alignment_file=alignment_file,
                       output_tracks_file=output_tracks_file,
#                        max_events=max_events,
                       select_duts=align_duts,
                       dut_names=dut_names,
                       n_pixels=n_pixels,
                       pixel_size=pixel_size,
                       selection_fit_duts=selection_fit_duts,  # Only use selected DUTs for track fit
                       selection_hit_duts=selection_hit_duts,  # Only use selected duts
                       quality_sigma=quality_sigma,
                       exclude_dut_hit=False,  # For constrained residuals
                       use_prealignment=False,
                       plot=plot,
                       chunk_size=chunk_size)

            logging.info('= Alignment step 2 / iteration %d: Selecting tracks for DUTs %s =', iteration_step, alignment_duts_str)
            data_selection.select_tracks(input_tracks_file=output_tracks_file,
                                         output_tracks_file=output_selected_tracks_file,
                                         select_duts=align_duts,
                                         duts_hit_selection=duts_selection,
                                         duts_quality_selection=duts_selection,
                                         chunk_size=chunk_size,
                                         condition=['n_hits_dut_%d < 3' % dut for dut in align_duts])

            if set(align_duts) & set(selection_fit_duts):
                track_angles_file = os.path.splitext(merged_file)[0] + '_tracks_angles_aligned_selected_tracks_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
                histogram_track_angle(input_tracks_file=output_selected_tracks_file,
                                      output_track_angle_file=track_angles_file,
                                      input_alignment_file=alignment_file,
                                      select_duts=align_duts,
                                      dut_names=dut_names,
                                      n_bins=200,  # TODO: fix n_bins
                                      plot=plot)
                with tb.open_file(track_angles_file, mode="r") as in_file_h5:
                    beam_alpha = -in_file_h5.root.Alpha_Track_Angle_Hist.attrs.mean
                    beam_beta = -in_file_h5.root.Beta_Track_Angle_Hist.attrs.mean
                beam_alignment = _create_beam_alignment_array()
                beam_alignment[0]['beam_alpha'] = beam_alpha
                beam_alignment[0]['beam_beta'] = beam_beta
                print "beam alignment from track angles", beam_alignment
                geometry_utils.save_beam_alignment_parameters(alignment_file=alignment_file,
                                                              beam_alignment=beam_alignment)
                os.remove(track_angles_file)

            if plot:
                output_residuals_file = os.path.splitext(merged_file)[0] + '_residuals_aligned_selected_tracks_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
                calculate_residuals(input_tracks_file=output_selected_tracks_file,
                                    input_alignment_file=alignment_file,
                                    output_residuals_file=output_residuals_file,
                                    dut_names=dut_names,
                                    n_pixels=n_pixels,
                                    pixel_size=pixel_size,
                                    use_prealignment=False,
                                    select_duts=align_duts,
                                    npixels_per_bin=None,
                                    nbins_per_pixel=None,
                                    use_fit_limits=use_fit_limits,
                                    plot=True,
                                    chunk_size=chunk_size)
                os.remove(output_residuals_file)

            logging.info('= Alignment step 3 / iteration %d: Calculating alignment for DUTs %s =', iteration_step, alignment_duts_str)
            new_alignment = calculate_transformation(input_tracks_file=output_selected_tracks_file,
                                                     input_alignment_file=alignment_file,
                                                     use_prealignment=False,
                                                     select_duts=align_duts,
                                                     align_telescope=align_telescope,
                                                     use_fit_limits=use_fit_limits,
                                                     chunk_size=chunk_size)

            # Delete temporary files
            os.remove(output_tracks_file)
            os.remove(output_selected_tracks_file)
#             os.remove(os.path.splitext(merged_file)[0] + '_new_alignment_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step))
#             os.remove(os.path.splitext(merged_file)[0] + '_tracks_aligned_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step))
#             os.remove(os.path.splitext(merged_file)[0] + '_tracks_aligned_duts_%s_tmp_%d.pdf' % (alignment_duts, iteration_step)
#             os.remove(os.path.splitext(merged_file)[0] + '_residuals_aligned_reduced_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step))

            geometry_utils.save_alignment_parameters(alignment_file=alignment_file,
                                                     alignment_parameters=new_alignment,
                                                     mode='absolute',
                                                     select_duts=align_duts,
                                                     parameters=selected_alignment_parameters)

#     # remove temporary files
    os.remove(output_track_candidates_file)
#     os.remove(output_tracks_file)
#     os.remove(output_selected_tracks_file)

    if plot or set(align_duts) & set(selection_fit_duts):
        final_track_candidates_file = os.path.splitext(merged_file)[0] + '_track_candidates_final_tmp_duts_%s.h5' % alignment_duts
        find_tracks(input_merged_file=merged_file,
                    input_alignment_file=alignment_file,
                    output_track_candidates_file=final_track_candidates_file,
                    use_prealignment=False,
                    correct_beam_alignment=True,
                    max_events=max_events)

        fit_tracks(input_track_candidates_file=final_track_candidates_file,
                   input_alignment_file=alignment_file,
                   output_tracks_file=os.path.splitext(merged_file)[0] + '_tracks_final_tmp_duts_%s.h5' % alignment_duts,
#                    max_events=max_events,
                   select_duts=align_duts,  # Only create residuals of selected DUTs
                   dut_names=dut_names,
                   n_pixels=n_pixels,
                   pixel_size=pixel_size,
                   selection_fit_duts=selection_fit_duts,  # Only use selected duts
                   selection_hit_duts=selection_hit_duts,
                   quality_sigma=quality_sigma,
                   exclude_dut_hit=False,  # For unconstrained residuals
                   use_prealignment=False,
                   plot=plot,
                   chunk_size=chunk_size)

        data_selection.select_tracks(input_tracks_file=os.path.splitext(merged_file)[0] + '_tracks_final_tmp_duts_%s.h5' % alignment_duts,
                                     output_tracks_file=os.path.splitext(merged_file)[0] + '_tracks_final_selected_tracks_tmp_duts_%s.h5' % alignment_duts,
                                     select_duts=align_duts,
                                     duts_hit_selection=duts_selection,
                                     duts_quality_selection=duts_selection,
                                     chunk_size=chunk_size)

        if set(align_duts) & set(selection_fit_duts):
            track_angles_file = os.path.splitext(merged_file)[0] + '_tracks_angles_final_reduced_tmp_duts_%s.h5' % alignment_duts
            histogram_track_angle(input_tracks_file=os.path.splitext(merged_file)[0] + '_tracks_final_selected_tracks_tmp_duts_%s.h5' % alignment_duts,
                                  output_track_angle_file=track_angles_file,
                                  input_alignment_file=alignment_file,
                                  select_duts=align_duts,
                                  dut_names=dut_names,
                                  n_bins=200,  # TODO: fix n_bins
                                  plot=True)
            with tb.open_file(track_angles_file, mode="r") as in_file_h5:
                beam_alpha = -in_file_h5.root.Alpha_Track_Angle_Hist.attrs.mean
                beam_beta = -in_file_h5.root.Beta_Track_Angle_Hist.attrs.mean
            beam_alignment = _create_beam_alignment_array()
            beam_alignment[0]['beam_alpha'] = beam_alpha
            beam_alignment[0]['beam_beta'] = beam_beta
            print "beam alignment from track angles", beam_alignment
            geometry_utils.save_beam_alignment_parameters(alignment_file=alignment_file,
                                                          beam_alignment=beam_alignment)

        # Plotting final results
        if plot:
            calculate_residuals(input_tracks_file=os.path.splitext(merged_file)[0] + '_tracks_final_selected_tracks_tmp_duts_%s.h5' % alignment_duts,
                                input_alignment_file=alignment_file,
                                output_residuals_file=os.path.splitext(merged_file)[0] + '_residuals_final_duts_%s.h5' % alignment_duts,
                                select_duts=align_duts,
                                dut_names=dut_names,
                                n_pixels=n_pixels,
                                pixel_size=pixel_size,
                                use_prealignment=False,
                                use_fit_limits=use_fit_limits,
                                plot=plot,
                                chunk_size=chunk_size)

            # remove temporary files for plotting
#             os.remove(os.path.splitext(merged_file)[0] + '_final_tmp_duts_%s.h5' % alignment_duts)
        os.remove(final_track_candidates_file)
        os.remove(os.path.splitext(merged_file)[0] + '_tracks_final_tmp_duts_%s.h5' % alignment_duts)
#         os.remove(os.path.splitext(merged_file)[0] + '_tracks_final_tmp_duts_%s.pdf' % alignment_duts)
        os.remove(os.path.splitext(merged_file)[0] + '_tracks_final_selected_tracks_tmp_duts_%s.h5' % alignment_duts)
        os.remove(os.path.splitext(merged_file)[0] + '_residuals_final_duts_%s.h5' % alignment_duts)

    # remove temporary files
#     os.remove(os.path.splitext(track_candidates_reduced)[0] + '_not_aligned.h5')
    # keep file for testing
#     os.remove(track_candidates_reduced)


def calculate_transformation(input_tracks_file, input_alignment_file, select_duts, align_telescope=None, use_prealignment=False, use_fit_limits=True, chunk_size=1000000):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    input_alignment_file : string
        Filename of the input aligment file.
    select_duts : iterable
        Selecting DUTs that will be processed.
    align_telescope : iterable
        The translation is not changed for given DUTs.
    use_prealignment : bool
        If True, use pre-alignment; if False, use alignment.
    use_fit_limits : bool
        If True, use fit limits from pre-alignment for selecting fit range for the alignment.
    chunk_size : int
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

    euler_angles = [np.zeros(3, dtype=np.double)] * n_duts
    translation = [np.zeros(3, dtype=np.double)] * n_duts
    total_n_tracks = [0] * n_duts

    new_alignment = alignment.copy()

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        for actual_dut in select_duts:
            node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT_%d' % actual_dut)
            if align_telescope and actual_dut in align_telescope:
                no_translation = True
            else:
                no_translation = False
            logging.debug('Calculate transformation for DUT%d', actual_dut)

            if use_fit_limits:
                fit_limit_x_local, fit_limit_y_local = fit_limits[actual_dut][0], fit_limits[actual_dut][1]
            else:
                fit_limit_x_local = None
                fit_limit_y_local = None

            actual_chunk = -1
            for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                actual_chunk += 1
                # select good hits and tracks
                selection = np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut]), ~np.isnan(tracks_chunk['track_chi2']))
                tracks_chunk = tracks_chunk[selection]  # Take only tracks where actual dut has a hit, otherwise residual wrong

                # Coordinates in global coordinate system (x, y, z)
                hit_x_local, hit_y_local, hit_z_local = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
                slopes = np.column_stack([tracks_chunk['slope_0'], tracks_chunk['slope_1'], tracks_chunk['slope_2']])
                offsets = np.column_stack([tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']])


                if not np.allclose(hit_z_local, 0.0):
                    raise RuntimeError("Transformation into local coordinate system gives z != 0")

                limit_xy_local_sel = np.ones_like(hit_x_local, dtype=np.bool)
                if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local[0]):
                    limit_xy_local_sel &= hit_x_local >= fit_limit_x_local[0]
                if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local[1]):
                    limit_xy_local_sel &= hit_x_local <= fit_limit_x_local[1]
                if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local[0]):
                    limit_xy_local_sel &= hit_y_local >= fit_limit_y_local[0]
                if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local[1]):
                    limit_xy_local_sel &= hit_y_local <= fit_limit_y_local[1]

                hit_x_local = hit_x_local[limit_xy_local_sel]
                hit_y_local = hit_y_local[limit_xy_local_sel]
                hit_z_local = hit_z_local[limit_xy_local_sel]
                hit_local = np.column_stack([hit_x_local, hit_y_local, hit_z_local])
                slopes = slopes[limit_xy_local_sel]
                offsets = offsets[limit_xy_local_sel]
                n_tracks = np.count_nonzero(limit_xy_local_sel)

                x_dut_start = alignment[actual_dut]['translation_x']
                y_dut_start = alignment[actual_dut]['translation_y']
                z_dut_start = alignment[actual_dut]['translation_z']
                translation_dut_start = np.array([x_dut_start, y_dut_start, z_dut_start])
                alpha_dut_start = alignment[actual_dut]['alpha']
                beta_dut_start = alignment[actual_dut]['beta']
                gamma_dut_start = alignment[actual_dut]['gamma']
                print "n_tracks for chunk %d:" % actual_chunk, n_tracks
                delta_t = 0.8  # TODO: optimize
                iterations = 100
                lin_alpha = 1.0

                print "alpha, beta, gamma at start", alpha_dut_start, beta_dut_start, gamma_dut_start
                print "translation at start", x_dut_start, y_dut_start, z_dut_start

                initialize_angles = True
                for i in range(iterations):
                    print "****************** iteration step %d DUT%d chunk %d *********************" % (i, actual_dut, actual_chunk)
                    if initialize_angles:
                        alpha, beta, gamma = alpha_dut_start, beta_dut_start, gamma_dut_start
                        rotation_dut = geometry_utils.rotation_matrix(alpha=alpha,
                                                                      beta=beta,
                                                                      gamma=gamma)
                        rotation_dut_old = np.identity(3, dtype=np.double)
                        translation_dut = np.array([x_dut_start, y_dut_start, z_dut_start])
                        translation_dut_old = np.array([0.0, 0.0, 0.0])
                        initialize_angles = False

#                     if i == 0:
#                         lin_alpha_opt = 0.000001
#                         lin_alpha_opt = start_alpha * np.exp(i * (np.log(stop_alpha) - np.log(start_alpha)) / iterations)
#                         delta_t = start_delta_t * np.exp(i * (np.log(stop_delta_t) - np.log(start_delta_t)) / iterations)
#                         if i==0:
#                             delta_t = lin_alpha_opt**-1 / 100.0
#                         else:
#                         delta_t = lin_alpha_opt**-1
#                         delta_t = 1./lin_alpha_opt

                    tot_matr = None
                    tot_b = None
                    identity = np.identity(3, dtype=np.double)
                    n_identity = -np.identity(3, dtype=np.double)

                    # vectorized calculation of matrix and b vector
                    outer_prod_slopes = np.einsum('ij,ik->ijk', slopes, slopes)
                    l_matr = identity - outer_prod_slopes
                    R_y = np.matmul(rotation_dut, hit_local.T).T
                    skew_R_y = geometry_utils.skew(R_y)
                    tot_matr = np.concatenate([np.matmul(l_matr, skew_R_y), np.matmul(l_matr, n_identity)], axis=2)
                    tot_matr = tot_matr.reshape(-1, tot_matr.shape[2])
                    tot_b = np.matmul(l_matr, np.expand_dims(offsets, axis=2)) - np.matmul(l_matr, np.expand_dims(R_y + translation_dut, axis=2))
                    tot_b = tot_b.reshape(-1)

                    # iterative calculation of matrix and b vector
#                     for count in range(len(hit_x_local)):
#                         if count >= max_n_tracks:
#                             count = count - 1
#                             break
#                         slope = slopes[count]
#
#                         l_matr = identity - np.outer(slope, slope)
#                         p = offsets[count]
#                         y = hit_local[count]
#                         R_y = np.dot(rotation_dut, y)
#
#                         b = np.dot(l_matr, p) - np.dot(l_matr, (R_y + translation_dut))
#                         if tot_b is None:
#                             tot_b = b
#                         else:
#                             tot_b = np.hstack([tot_b, b])
#
#                         skew_R_y = geometry_utils.skew(R_y)
#
#                         matr = np.dot(l_matr, np.hstack([skew_R_y, n_identity]))
#                         if tot_matr is None:
#                             tot_matr = matr
#                         else:
#                             tot_matr = np.vstack([tot_matr, matr])

                    # SVD
                    print "calculating SVD..."
                    u, s, v = np.linalg.svd(tot_matr, full_matrices=False)
#                     u, s, v = svds(tot_matr, k=5)
                    print "...finished calculating SVD"

                    diag = np.diag(s ** -1)
                    tot_matr_inv = np.dot(v.T, np.dot(diag, u.T))
                    omega_b_dot = np.dot(tot_matr_inv, -lin_alpha * tot_b)

                    translation_dut_old = translation_dut
                    rotation_dut_old = rotation_dut
#                     alpha_old, beta_old, gamma_old = alpha, beta, gamma
#                     print "alpha, beta, gamma BEFORE calc", alpha, beta, gamma
                    rotation_dut = np.dot((np.identity(3, dtype=np.double) + delta_t * geometry_utils.skew(omega_b_dot[:3])), rotation_dut)

                    # apply UP (polar) decomposition to normalize/orthogonalize rotation matrix (det = 1)
                    u_rot, _, v_rot = np.linalg.svd(rotation_dut, full_matrices=False)
                    rotation_dut = np.dot(u_rot, v_rot)  # orthogonal matrix U

                    if not no_translation:
                        translation_dut = translation_dut + delta_t * omega_b_dot[3:]

#                     try:
#                         # TODO: do not use minimizer, use algebra
#                         alpha, beta, gamma = geometry_utils.euler_angles_minimizer(R=rotation_dut, alpha_start=alpha, beta_start=beta, gamma_start=gamma, limit=0.1)
#                         print "alpha, beta, gamma AFTER minimizer", alpha, beta, gamma
# #                         alpha, beta, gamma = geometry_utils.euler_angles(R=rotation_dut)
# #                         print "alpha, beta, gamma AFTER calc", alpha, beta, gamma
#                     except ValueError:
#                         update = False
#                         alpha, beta, gamma = alpha_old, beta_old, gamma_old
#                         rotation_dut = rotation_dut_old
#                         translation_dut = translation_dut_old
#                     else:
#                         update = True
#                         rotation_dut = geometry_utils.rotation_matrix(alpha=alpha,
#                                                                       beta=beta,
#                                                                       gamma=gamma)
#
#                     if update:
                    allclose_trans = np.allclose(translation_dut, translation_dut_old, rtol=0.0, atol=1e-2)
                    allclose_rot = np.allclose(rotation_dut, rotation_dut_old, rtol=1e-5, atol=0.0)
                    print "allclose rot trans", allclose_rot, allclose_trans
                    if allclose_rot and allclose_trans:
                        print "*** ALL CLOSE, BREAK ***"
                        break

                # TODO: do not use minimizer, use algebra
                alpha, beta, gamma = geometry_utils.euler_angles_minimizer(R=rotation_dut, alpha_start=alpha, beta_start=beta, gamma_start=gamma, limit=0.1)
                euler_angles_dut = np.array([alpha, beta, gamma], dtype=np.double)
                print "alpha, beta, gamma AFTER minimizer", alpha, beta, gamma
                fitted_rotation_dut = geometry_utils.rotation_matrix(alpha=alpha, beta=beta, gamma=gamma)
                if not np.allclose(rotation_dut, fitted_rotation_dut):
                    raise RuntimeError("Fit of alpha/beta/gamma returned no proper values.")
#                         alpha, beta, gamma = geometry_utils.euler_angles(R=rotation_dut)
#                         print "alpha, beta, gamma AFTER calc", alpha, beta, gamma

                print "euler_angles before average", euler_angles_dut, euler_angles[actual_dut]
                print "translation before average", translation_dut, translation[actual_dut]
                print "tracks in chunk", n_tracks
                euler_angles[actual_dut] = np.average([euler_angles_dut, euler_angles[actual_dut]], weights=[n_tracks, total_n_tracks[actual_dut]], axis=0)
                translation[actual_dut] = np.average([translation_dut, translation[actual_dut]], weights=[n_tracks, total_n_tracks[actual_dut]], axis=0)
                total_n_tracks[actual_dut] += n_tracks
                print "euler_angles average", euler_angles[actual_dut]
                print "translation average", translation[actual_dut]
                print "after total n tracks", total_n_tracks[actual_dut]

            new_alignment[actual_dut]['translation_x'] = translation[actual_dut][0]
            new_alignment[actual_dut]['translation_y'] = translation[actual_dut][1]
            new_alignment[actual_dut]['translation_z'] = translation[actual_dut][2]
            new_alignment[actual_dut]['alpha'] = euler_angles[actual_dut][0]
            new_alignment[actual_dut]['beta'] = euler_angles[actual_dut][1]
            new_alignment[actual_dut]['gamma'] = euler_angles[actual_dut][2]

    return new_alignment


# Helper functions for the alignment. Not to be used directly.
def _create_alignment_array(n_duts):
    # Result Translation / rotation table
    description = [('DUT', np.int32)]
    description.append(('translation_x', np.double))
    description.append(('translation_y', np.double))
    description.append(('translation_z', np.double))
    description.append(('alpha', np.double))
    description.append(('beta', np.double))
    description.append(('gamma', np.double))

    array = np.zeros((n_duts,), dtype=description)
    array[:]['DUT'] = np.array(range(n_duts))
    return array


def _create_beam_alignment_array():
    description = [('beam_alpha', np.double), ('beam_beta', np.double)]
    array = np.zeros((1,), dtype=description)
    return array


# Helper functions to be called from multiple processes
def _correlate_cluster(clusters_ref_dut, cluster_file, start_index, start_event_number, stop_event_number, column_correlation, row_correlation, chunk_size):
    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open other DUT cluster file
        for cluster_dut, start_index in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Cluster, start_index=start_index, start_event_number=start_event_number, stop_event_number=stop_event_number, chunk_size=chunk_size, fail_on_missing_events=False):  # Loop over the cluster in the actual cluster file in chunks

            analysis_utils.correlate_cluster_on_event_number(data_1=clusters_ref_dut,
                                                             data_2=cluster_dut,
                                                             column_corr_hist=column_correlation,
                                                             row_corr_hist=row_correlation)

    return start_index, column_correlation, row_correlation
