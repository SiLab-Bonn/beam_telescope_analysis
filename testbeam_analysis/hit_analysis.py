''' All functions acting on the hits of one DUT are listed here'''
from __future__ import division

import logging
import os.path
import re
from multiprocessing import Pool

import tables as tb
import numpy as np
import numba
from scipy.ndimage import median_filter

import progressbar

from pixel_clusterizer.clusterizer import HitClusterizer

from testbeam_analysis.telescope.telescope import Telescope
from testbeam_analysis.tools import smc
from testbeam_analysis.tools import analysis_utils, plot_utils
from testbeam_analysis.tools.plot_utils import plot_masked_pixels, plot_cluster_hists


def check(telescope_configuration, input_hits_files, output_check_file=None, check_duts=None, event_ranges=None, plot=True, chunk_size=1000000):
    '''"Checking hit files. Wrapper for check_file(). For detailed description of the parameters see check_file().

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_hits_files : list
        Filenames of the input hit files.
    output_check_file : list
        Filenames of the output check files.
    check_duts : list
        List of selected DUTs.
    event_ranges : list
        The event ranges.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Check hit files of %d DUTs ===', n_duts)

    if not output_check_file:
        output_check_file = [None] * n_duts
    for i, dut in enumerate(telescope):
        if not check_duts or (check_duts and i in check_duts):
            output_check_file[i] = check_file(input_hits_file=input_hits_files[i],
                                              n_pixel=dut.n_pixel,
                                              output_check_file=output_check_file[i],
                                              event_range=event_ranges[i] if (event_ranges and len(event_ranges) == n_duts and event_ranges[i]) else 1,
                                              plot=plot,
                                              chunk_size=chunk_size)
    return output_check_file


def check_file(input_hits_file, n_pixel, output_check_file=None, event_range=1, plot=True, chunk_size=1000000):
    '''Checks the hit table to have proper data.

    The checks include:
      - hit definitions:
          - position has to start at 1 (not 0)
          - position should not exceed number of pixels (n_pixel)
      - event building
          - event number has to be strictly monotone
          - hit position correlations of consecutive events are
            created. Should be zero for distinctly
            built events.

    Parameters
    ----------
    input_hits_file : string
        File name of the hit table.
    output_check_file : string
        Filename of the output file with the correlation histograms.
    n_pixel : tuple
        Tuple of the total number of pixels (column/row).
    event_range : integer
        The range of events to correlate.
        E.g.: event_range = 2 correlates to predecessing event hits.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''

    logging.info('== Check hit file %s ==', input_hits_file)

    if output_check_file is None:
        output_check_file = input_hits_file[:-3] + '_check.h5'

    with tb.open_file(output_check_file, mode="w") as out_file_h5:
        with tb.open_file(input_hits_file, 'r') as input_file_h5:
            shape_column = (n_pixel[0], n_pixel[0])
            shape_row = (n_pixel[1], n_pixel[1])
            col_corr = np.zeros(shape_column, dtype=np.int32)
            row_corr = np.zeros(shape_row, dtype=np.int32)
            last_event = None
            out_dE = out_file_h5.create_earray(out_file_h5.root, name='EventDelta',
                                               title='Change of event number per non empty event',
                                               shape=(0, ),
                                               atom=tb.Atom.from_dtype(np.dtype(np.uint64)),
                                               filters=tb.Filters(complib='blosc',
                                                                  complevel=5,
                                                                  fletcher32=False))
            out_E = out_file_h5.create_earray(out_file_h5.root, name='EventNumber',
                                              title='Event number of non empty event',
                                              shape=(0, ),
                                              atom=tb.Atom.from_dtype(np.dtype(np.uint64)),
                                              filters=tb.Filters(complib='blosc',
                                                                 complevel=5,
                                                                 fletcher32=False))

            for hits, _ in analysis_utils.data_aligned_at_events(
                    input_file_h5.root.Hits,
                    chunk_size=chunk_size):
                if not np.all(np.diff(hits['event_number']) >= 0):
                    raise RuntimeError('The event number does not always increase. \
                    The hits cannot be used like this!')
                if np.any(hits['column'] < 1) or np.any(hits['row'] < 1):
                    raise RuntimeError('The column/row definition does not \
                    start at 1!')
                if (np.any(hits['column'] > n_pixel[0]) or np.any(hits['row'] > n_pixel[1])):
                    raise RuntimeError('The column/row definition exceed the nuber \
                    of pixels (%s/%s)!', n_pixel[0], n_pixel[1])

                analysis_utils.correlate_hits_on_event_range(hits=hits,
                                                             column_corr_hist=col_corr,
                                                             row_corr_hist=row_corr,
                                                             event_range=event_range)

                event_numbers = np.unique(hits['event_number'])
                event_delta = np.diff(event_numbers)

                if last_event:
                    event_delta = np.concatenate((np.array([event_numbers[0] - last_event]),
                                                  event_delta))
                last_event = event_numbers[-1]

                out_dE.append(event_delta)
                out_E.append(event_numbers)

            out_col = out_file_h5.create_carray(out_file_h5.root, name='CorrelationColumns',
                                                title='Column Correlation with event range=%s' % event_range,
                                                atom=tb.Atom.from_dtype(col_corr.dtype),
                                                shape=col_corr.shape,
                                                filters=tb.Filters(complib='blosc',
                                                                   complevel=5,
                                                                   fletcher32=False))
            out_row = out_file_h5.create_carray(out_file_h5.root, name='CorrelationRows',
                                                title='Row Correlation with event range=%s' % event_range,
                                                atom=tb.Atom.from_dtype(row_corr.dtype),
                                                shape=row_corr.shape,
                                                filters=tb.Filters(complib='blosc',
                                                                   complevel=5,
                                                                   fletcher32=False))
            out_col[:] = col_corr
            out_row[:] = row_corr

    if plot:
        plot_utils.plot_checks(input_corr_file=output_check_file)


def mask(telescope_configuration, input_hits_files, output_mask_files=None, mask_duts=None, pixel_mask_names=None, thresholds=None, filter_sizes=None, plot=True, chunk_size=1000000):
    '''"Masking noisy pixels. Wrapper for generate_pixel_mask(). For detailed description of the parameters see generate_pixel_mask().

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_hits_files : list
        Filenames of the input hit files.
    output_mask_files : list
        Filenames of the output mask files.
    mask_duts : list
        List of selected DUTs.
    pixel_mask_names : list
        Pixel mask type for each DUT. Possible mask types:
          - DisabledPixelMask
          - NoisyPixelMask
    thresholds : list
        The thresholds for pixel masking.
    filter_sizes : list
        Median filter sizes.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Cluster hits of %d DUTs ===', n_duts)

    if not output_mask_files:
        output_mask_files = [None] * n_duts
    for i, dut in enumerate(telescope):
        if not mask_duts or (mask_duts and i in mask_duts):
            output_mask_files[i] = generate_pixel_mask(input_hits_file=input_hits_files[i],
                                                       n_pixel=dut.n_pixel,
                                                       pixel_mask_name=pixel_mask_names[i] if (pixel_mask_names and len(pixel_mask_names) == n_duts and pixel_mask_names[i]) else "NoisyPixelMask",
                                                       output_mask_file=output_mask_files[i],
                                                       pixel_size=dut.pixel_size,
                                                       threshold=thresholds[i] if (thresholds and len(thresholds) == n_duts and thresholds[i]) else 10.0,
                                                       filter_size=filter_sizes[i] if (filter_sizes and len(filter_sizes) == n_duts and filter_sizes[i]) else 3,
                                                       dut_name=dut.name,
                                                       plot=plot,
                                                       chunk_size=chunk_size)
    return output_mask_files


def generate_pixel_mask(input_hits_file, n_pixel, pixel_mask_name="NoisyPixelMask", output_mask_file=None, pixel_size=None, threshold=10.0, filter_size=3, dut_name=None, plot=True, chunk_size=1000000):
    '''Generating pixel mask from the hit table.

    Parameters
    ----------
    input_hits_file : string
        Filename of the input hit file.
    n_pixel : tuple
        Tuple of the total number of pixels (column/row).
    pixel_mask_name : string
        Name of the node containing the mask inside the output file. Possible mask types:
          - DisabledPixelMask: Any noisy pixels is masked and not taken into account when building clusters.
          - NoisyPixelMask: Clusters solely containing noisy pixels are not built.
    output_mask_file : string
        File name of the output mask file.
    pixel_size : tuple
        Tuple of the pixel size (column/row). If None, assuming square pixels.
    threshold : float
        The threshold for pixel masking. The threshold is given in units of
        sigma of the pixel noise (background subtracted). The lower the value
        the more pixels are masked.
    filter_size : scalar or tuple
        Adjust the median filter size by giving the number of columns and rows.
        The higher the value the more the background is smoothed and more
        pixels are masked.
    dut_name : string
        Name of the DUT. If None, file name of the hit table will be printed.
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('== Generating %s for %s ==', ' '.join(item.lower() for item in re.findall('[A-Z][^A-Z]*', pixel_mask_name)), input_hits_file)

    if pixel_mask_name not in ["DisabledPixelMask", "NoisyPixelMask"]:
        raise ValueError("'%s' is not a valid pixel mask name." % pixel_mask_name)

    if output_mask_file is None:
        output_mask_file = os.path.splitext(input_hits_file)[0] + '_mask.h5'

    # Create occupancy histogram
    def work(hit_chunk):
        col, row = hit_chunk['column'], hit_chunk['row']
        return analysis_utils.hist_2d_index(col - 1, row - 1, shape=n_pixel)

    smc.SMC(table_file_in=input_hits_file,
            file_out=output_mask_file,
            func=work,
            node_desc={'name': 'HistOcc'},
            chunk_size=chunk_size)

    # Create mask from occupancy histogram
    with tb.open_file(output_mask_file, 'r+') as out_file_h5:
        occupancy = out_file_h5.root.HistOcc[:]
        # Run median filter across data, assuming 0 filling past the edges to get expected occupancy
        blurred = median_filter(occupancy.astype(np.int32), size=filter_size, mode='constant', cval=0.0)
        # Spot noisy pixels maxima by substracting expected occupancy
        difference = np.ma.masked_array(occupancy - blurred)
        std = np.ma.std(difference)
        abs_occ_threshold = threshold * std
        occupancy = np.ma.masked_where(difference > abs_occ_threshold, occupancy)
        logging.info('Masked %d pixels at threshold %.1f in %s', np.ma.count_masked(occupancy), threshold, input_hits_file)
        # Generate tuple col / row array of hot pixels, do not use getmask()
        pixel_mask = np.ma.getmaskarray(occupancy)

        # Create masked pixels array
        masked_pixel_table = out_file_h5.create_carray(out_file_h5.root, name=pixel_mask_name, title='Pixel Mask', atom=tb.Atom.from_dtype(pixel_mask.dtype), shape=pixel_mask.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        masked_pixel_table[:] = pixel_mask

    if plot:
        plot_masked_pixels(input_mask_file=output_mask_file, pixel_size=pixel_size, dut_name=dut_name)

    return output_mask_file


def cluster(telescope_configuration, input_hits_files, input_mask_files=None, output_cluster_files=None, cluster_duts=None, min_hit_charges=None, max_hit_charges=None, column_cluster_distances=None, row_cluster_distances=None, frame_cluster_distances=None, plot=True, chunk_size=1000000):
    '''Clustering hits. Wrapper for cluster_hits(). For detailed description of the parameters see cluster_hits().

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_hits_files : list
        Filenames of the input hit files.
    input_mask_files : list
        Filenames of the input mask files.
    output_cluster_files : list
        Filenames of the output cluster files.
    cluster_duts : list
        List of selected DUTs.
    min_hit_charges : list
        Minimum hit charges.
    max_hit_charges : list
        Maximum hit charges.
    column_cluster_distances : list
        Maximum column distances.
    row_cluster_distances : list
        Maximum row distances.
    frame_cluster_distances : list
        Maximum frame distances.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Cluster hits of %d DUTs ===', n_duts)

    if not output_cluster_files:
        output_cluster_files = [None] * n_duts
    for i, dut in enumerate(telescope):
        if not cluster_duts or (cluster_duts and i in cluster_duts):
            output_cluster_files[i] = cluster_hits(input_hits_file=input_hits_files[i],
                                                   output_cluster_file=output_cluster_files[i],
                                                   input_mask_file=input_mask_files[i] if (input_mask_files and len(input_mask_files) == n_duts and input_mask_files[i]) else None,
                                                   min_hit_charge=min_hit_charges[i] if (min_hit_charges and len(min_hit_charges) == n_duts and min_hit_charges[i]) else 0,
                                                   max_hit_charge=max_hit_charges[i] if (max_hit_charges and len(max_hit_charges) == n_duts and max_hit_charges[i]) else None,
                                                   column_cluster_distance=column_cluster_distances[i] if (column_cluster_distances and len(column_cluster_distances) == n_duts and column_cluster_distances[i]) else 1,
                                                   row_cluster_distance=row_cluster_distances[i] if (row_cluster_distances and len(row_cluster_distances) == n_duts and row_cluster_distances[i]) else 1,
                                                   frame_cluster_distance=frame_cluster_distances[i] if (frame_cluster_distances and len(frame_cluster_distances) == n_duts and frame_cluster_distances[i]) else 0,
                                                   dut_name=dut.name,
                                                   plot=plot,
                                                   chunk_size=chunk_size)
    return output_cluster_files


def cluster_hits(input_hits_file, output_cluster_file=None, input_mask_file=None, min_hit_charge=0, max_hit_charge=None, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=0, dut_name=None, plot=True, chunk_size=1000000):
    '''Clusters the hits in the data file containing the hit table.

    Parameters
    ----------
    input_hits_file : string
        Filename of the input hits file.
    output_cluster_file : string
        Filename of the output cluster file. If None, the filename will be derived from the input hits file.
    input_mask_file : string
        Filename of the input mask file.
    min_hit_charge : uint
        Minimum hit charge. Minimum possible hit charge must be given in order to correcly calculate the cluster coordinates.
    max_hit_charge : uint
        Maximum hit charge. Hits wit charge above the limit will be ignored.
    column_cluster_distance : uint
        Maximum column distance between hist so that they are assigned to the same cluster. Value of 0 effectively disables the clusterizer in column direction.
    row_cluster_distance : uint
        Maximum row distance between hist so that they are assigned to the same cluster. Value of 0 effectively disables the clusterizer in row direction.
    frame_cluster_distance : uint
        Sometimes an event has additional timing information (e.g. bunch crossing ID, frame ID). Value of 0 effectively disables the clusterization in time.
    dut_name : string
        Name of the DUT. If None, filename of the output cluster file will be used.
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('== Clustering hits in %s ==', input_hits_file)

    if output_cluster_file is None:
        output_cluster_file = os.path.splitext(input_hits_file)[0] + '_clustered.h5'

    # Getting noisy and disabled pixel mask
    if input_mask_file is not None:
        with tb.open_file(input_mask_file, 'r') as input_mask_file_h5:
            try:
                disabled_pixels = np.dstack(np.nonzero(input_mask_file_h5.root.DisabledPixelMask[:]))[0] + 1
            except tb.NoSuchNodeError:
                disabled_pixels = None
            try:
                noisy_pixels = np.dstack(np.nonzero(input_mask_file_h5.root.NoisyPixelMask[:]))[0] + 1
            except tb.NoSuchNodeError:
                noisy_pixels = None

    # Prepare clusterizer

    # Define end of cluster function to
    # calculate the size in col/row for each cluster
    @numba.njit(locals={'diff_col': numba.int32, 'diff_row': numba.int32, 'cluster_shape': numba.int64})
    def end_of_cluster_function(hits, clusters, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels, disabled_pixels, seed_hit_index):
        hit_arr = np.zeros((15, 15), dtype=np.bool_)
        center_col = hits[cluster_hit_indices[0]].column
        center_row = hits[cluster_hit_indices[0]].row
        hit_arr[7, 7] = 1
        min_col = hits[cluster_hit_indices[0]].column
        max_col = hits[cluster_hit_indices[0]].column
        min_row = hits[cluster_hit_indices[0]].row
        max_row = hits[cluster_hit_indices[0]].row
        for i in cluster_hit_indices[1:]:
            if i < 0:  # Not used indeces = -1
                break
            diff_col = hits[i].column - center_col
            diff_row = hits[i].row - center_row
            if np.abs(diff_col) < 8 and np.abs(diff_row) < 8:
                hit_arr[7 + hits[i].column - center_col, 7 + hits[i].row - center_row] = 1
            if hits[i].column < min_col:
                min_col = hits[i].column
            if hits[i].column > max_col:
                max_col = hits[i].column
            if hits[i].row < min_row:
                min_row = hits[i].row
            if hits[i].row > max_row:
                max_row = hits[i].row

        if max_col - min_col < 8 and max_row - min_row < 8:
            # make 8x8 array
            col_base = 7 + min_col - center_col
            row_base = 7 + min_row - center_row
            cluster_arr = hit_arr[col_base:col_base + 8, row_base:row_base + 8]
            # finally calculate cluster shape
            # uint64 is desired, but Numexpr and other tools are limited to the dtype int64
            if cluster_arr[7, 7] == 1:
                cluster_shape = -1
            else:
                cluster_shape = analysis_utils.calculate_cluster_shape(cluster_arr)
        else:
            # cluster is exceeding 8x8 array
            cluster_shape = -1

        clusters[cluster_index].cluster_shape = cluster_shape
        clusters[cluster_index].err_column = max_col - min_col + 1
        clusters[cluster_index].err_row = max_row - min_row + 1

    # Adding number of clusters
    def end_of_event_function(hits, clusters, start_event_hit_index, stop_event_hit_index, start_event_cluster_index, stop_event_cluster_index):
        for i in range(start_event_cluster_index, stop_event_cluster_index):
            clusters[i]['n_cluster'] = hits["n_cluster"][start_event_hit_index]

    hit_dtype = np.dtype([('event_number', '<i8'),
                          ('frame', '<u1'),
                          ('column', '<u2'),
                          ('row', '<u2'),
                          ('charge', '<u2'),  # TODO: change that
                          ('cluster_ID', '<i4'),  # TODO: change that
                          ('is_seed', '<u1'),
                          ('cluster_size', '<u4'),
                          ('n_cluster', '<u4')])

    cluster_dtype = np.dtype([('event_number', '<i8'),
                              ('ID', '<u4'),
                              ('n_hits', '<u4'),
                              ('charge', '<f4'),
                              ('seed_column', '<u2'),
                              ('seed_row', '<u2'),
                              ('mean_column', '<f4'),
                              ('mean_row', '<f4')])

    clz = HitClusterizer(column_cluster_distance=column_cluster_distance,
                         row_cluster_distance=row_cluster_distance,
                         frame_cluster_distance=frame_cluster_distance,
                         min_hit_charge=min_hit_charge,
                         max_hit_charge=max_hit_charge,
                         hit_dtype=hit_dtype,
                         cluster_dtype=cluster_dtype)
    clz.add_cluster_field(description=('err_column', '<f4'))  # Add an additional field to hold the cluster size in x
    clz.add_cluster_field(description=('err_row', '<f4'))  # Add an additional field to hold the cluster size in y
    clz.add_cluster_field(description=('n_cluster', '<u4'))  # Adding additional field for number of clusters per event
    clz.add_cluster_field(description=('cluster_shape', '<i8'))  # Adding additional field for the cluster shape
    clz.set_end_of_cluster_function(end_of_cluster_function)  # Set the new function to the clusterizer
    clz.set_end_of_event_function(end_of_event_function)

    # Run clusterizer on hit table in parallel on all cores
    def cluster_func(hits, clz, noisy_pixels, disabled_pixels):
        _, cl = clz.cluster_hits(hits,
                                 noisy_pixels=noisy_pixels,
                                 disabled_pixels=disabled_pixels)
        return cl

    smc.SMC(table_file_in=input_hits_file,
            file_out=output_cluster_file,
            func=cluster_func,
            func_kwargs={'clz': clz,
                         'noisy_pixels': noisy_pixels,
                         'disabled_pixels': disabled_pixels},
            node_desc={'name': 'Clusters'},
            align_at='event_number',
            chunk_size=chunk_size)

    # Calculate cluster size histogram
    def hist_func(cluster):
        n_hits = cluster['n_hits']
        hist = analysis_utils.hist_1d_index(n_hits, shape=(np.max(n_hits) + 1,))
        return hist

    smc.SMC(table_file_in=output_cluster_file,
            file_out=output_cluster_file[:-3] + '_hist.h5',
            func=hist_func,
            node_desc={'name': 'HistClusterSize'},
            chunk_size=chunk_size)

    # Load infos from cluster size for error determination and plotting
    with tb.open_file(output_cluster_file[:-3] + '_hist.h5', 'r') as input_file_h5:
        hight = input_file_h5.root.HistClusterSize[:]

    # Calculate position error from cluster size
    def get_eff_pitch(hist, cluster_size):
        ''' Effective pitch to describe the cluster
            size propability distribution

        hist : array like
            Histogram with cluster size distribution
        cluster_size : Cluster size to calculate the pitch for
        '''

        return np.sqrt(hight[int(cluster_size)].astype(np.float) / hight.sum())

    def pos_error_func(clusters):
        # Check if end_of_cluster function was called
        # Under unknown and rare circumstances this might not be the case
        if not np.any(clusters['err_column']):
            raise RuntimeError('Clustering failed, please report bug at:'
                               'https://github.com/SiLab-Bonn/testbeam_analysis/issues')
        # Set errors for small clusters, where charge sharing enhances
        # resolution
        for css in [(1, 1), (1, 2), (2, 1), (2, 2)]:
            sel = np.logical_and(clusters['err_column'] == css[0],
                                 clusters['err_row'] == css[1])
            clusters['err_column'][sel] = get_eff_pitch(hist=hight,
                                                        cluster_size=css[0]) / np.sqrt(12)
            clusters['err_row'][sel] = get_eff_pitch(hist=hight,
                                                     cluster_size=css[1]) / np.sqrt(12)
        # Set errors for big clusters, where delta electrons reduce resolution
        sel = np.logical_or(clusters['err_column'] > 2, clusters['err_row'] > 2)
        clusters['err_column'][sel] = clusters['err_column'][sel] / np.sqrt(12)
        clusters['err_row'][sel] = clusters['err_row'][sel] / np.sqrt(12)

        return clusters

    smc.SMC(table_file_in=output_cluster_file,
            file_out=output_cluster_file,
            func=pos_error_func,
            chunk_size=chunk_size)

    # Copy masks to result cluster file

    # Copy nodes to result file
    if input_mask_file is not None:
        with tb.open_file(input_mask_file, 'r') as input_mask_file_h5:
            with tb.open_file(output_cluster_file, 'r+') as output_file_h5:
                try:
                    input_mask_file_h5.root.DisabledPixelMask._f_copy(newparent=output_file_h5.root)
                except tb.NoSuchNodeError:
                    pass
                try:
                    input_mask_file_h5.root.NoisyPixelMask._f_copy(newparent=output_file_h5.root)
                except tb.NoSuchNodeError:
                    pass

    if plot:
        plot_cluster_hists(input_cluster_file=output_cluster_file,
                           dut_name=dut_name,
                           chunk_size=chunk_size,
                           gui=False)

    return output_cluster_file


def correlate(telescope_configuration, input_files, output_correlation_file, resolution=(100.0, 100.0), ref_index=0, plot=True, chunk_size=100000):
    '''"Calculates the correlation histograms from the hit/cluster indices.
    The 2D correlation array of pairs of two different devices are created on event basis.
    All permutations are considered (all hits/clusters of the first device are correlated with all hits/clusters of the second device).

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_files : list
        Filenames of the input hit/cluster files.
    output_correlation_file : string
        Filename of the output correlation file with the correlation histograms.
    resolution : tuple
        Resolution of the correlation histogram in x and y direction (in um).
    ref_index : uint
        DUT index of the reference plane. Default is DUT 0. If None, generate correlation histograms of all DUT pairs.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Correlating the index of %d DUTs ===', n_duts)

    if ref_index is None:
        ref_duts = range(n_duts)
    else:
        ref_duts = [ref_index]

    with tb.open_file(output_correlation_file, mode="w") as out_file_h5:
        for ref_index in ref_duts:
            logging.info('== Correlating with DUT%d ==' % (ref_index,))
            # Result arrays to be filled
            x_correlations = []
            y_correlations = []
            start_indices = []
            dut_x_size = []
            dut_y_size = []
            ref_x_size = []
            ref_y_size = []

            # remove reference DUT from list of DUTs
            select_duts = list(set(range(n_duts)) - set([ref_index]))
            for dut_index in select_duts:
                ref_x_size.append((np.ceil(telescope[ref_index].x_size(global_position=True) / resolution[0]) * resolution[0]).astype(np.int32))
                ref_y_size.append((np.ceil(telescope[ref_index].y_size(global_position=True) / resolution[1]) * resolution[1]).astype(np.int32))
                dut_x_size.append((np.ceil(telescope[dut_index].x_size(global_position=True) / resolution[0]) * resolution[0]).astype(np.int32))
                dut_y_size.append((np.ceil(telescope[dut_index].y_size(global_position=True) / resolution[1]) * resolution[1]).astype(np.int32))
                x_correlations.append(np.zeros((np.array([dut_x_size[-1], ref_x_size[-1]]) / resolution[0]).astype(np.int32), dtype=np.int32))
                y_correlations.append(np.zeros((np.array([dut_y_size[-1], ref_y_size[-1]]) / resolution[1]).astype(np.int32), dtype=np.int32))
                start_indices.append(None)  # Store the loop indices for speed up

            with tb.open_file(input_files[ref_index], mode='r') as in_file_h5:  # Open DUT0 hit/cluster file
                try:
                    ref_node = in_file_h5.root.Clusters
                    ref_use_clusters = True
                except tb.NoSuchNodeError:
                    ref_node = in_file_h5.root.Hits
                    ref_use_clusters = False
                widgets = ['', progressbar.Percentage(), ' ',
                           progressbar.Bar(marker='*', left='|', right='|'),
                           ' ', progressbar.AdaptiveETA()]
                progress_bar = progressbar.ProgressBar(widgets=widgets,
                                                       maxval=ref_node.shape[0],
                                                       term_width=80)
                progress_bar.start()

                pool = Pool()
                # Loop over the hits/clusters of reference DUT
                for ref_indices, ref_read_index in analysis_utils.data_aligned_at_events(ref_node, chunk_size=chunk_size):
                    actual_event_numbers = ref_indices[:]['event_number']

                    # Create correlation histograms to the reference device for all other devices
                    # Do this in parallel to safe time

                    dut_results = []
                    # Loop over other DUTs
                    for index, dut_index in enumerate(select_duts):
                        dut_results.append(pool.apply_async(_correlate_position, kwds={
                            'ref': telescope[ref_index],
                            'dut': telescope[dut_index],
                            'ref_use_clusters': ref_use_clusters,
                            'ref_indices': ref_indices,
                            'dut_input_file': input_files[dut_index],
                            'start_index': start_indices[index],
                            'start_event_number': actual_event_numbers[0],
                            'stop_event_number': actual_event_numbers[-1] + 1,
                            'resolution': resolution,
                            'ref_x_size': ref_x_size[index],
                            'ref_y_size': ref_y_size[index],
                            'dut_x_size': dut_x_size[index],
                            'dut_y_size': dut_y_size[index],
                            'x_correlation': x_correlations[index],
                            'y_correlation': y_correlations[index],
                            'chunk_size': chunk_size}))
                    # Collect results when available
                    for index, dut_result in enumerate(dut_results):
                        (start_indices[index], x_correlations[index], y_correlations[index]) = dut_result.get()

                    progress_bar.update(ref_read_index)

                pool.close()
                pool.join()

            # Store the correlation histograms
            for index, dut_index in enumerate(select_duts):
                out_x = out_file_h5.create_carray(where=out_file_h5.root,
                                                  name='Correlation_x_%d_%d' % (ref_index, dut_index),
                                                  title='X correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                                                  atom=tb.Atom.from_dtype(x_correlations[index].dtype),
                                                  shape=x_correlations[index].shape,
                                                  filters=tb.Filters(complib='blosc',
                                                                     complevel=5,
                                                                     fletcher32=False))
                out_x.attrs.resolution = resolution[0]
                out_x.attrs.ref_size = ref_x_size[index]
                out_x.attrs.dut_size = dut_x_size[index]
                out_y = out_file_h5.create_carray(where=out_file_h5.root,
                                                  name='Correlation_y_%d_%d' % (ref_index, dut_index),
                                                  title='Y correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                                                  atom=tb.Atom.from_dtype(y_correlations[index].dtype),
                                                  shape=y_correlations[index].shape,
                                                  filters=tb.Filters(complib='blosc',
                                                                     complevel=5,
                                                                     fletcher32=False))
                out_y.attrs.resolution = resolution[1]
                out_y.attrs.ref_size = ref_y_size[index]
                out_y.attrs.dut_size = dut_y_size[index]
                out_x_reduced = out_file_h5.create_carray(where=out_file_h5.root,
                                                          name='Correlation_x_%d_%d_reduced_background' % (ref_index, dut_index),
                                                          title='X correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                                                          atom=tb.Atom.from_dtype(x_correlations[index].dtype),
                                                          shape=x_correlations[index].shape,
                                                          filters=tb.Filters(complib='blosc',
                                                                             complevel=5,
                                                                             fletcher32=False))
                out_x_reduced.attrs.resolution = resolution[0]
                out_x_reduced.attrs.ref_size = ref_x_size[index]
                out_x_reduced.attrs.dut_size = dut_x_size[index]
                out_y_reduced = out_file_h5.create_carray(where=out_file_h5.root,
                                                          name='Correlation_y_%d_%d_reduced_background' % (ref_index, dut_index),
                                                          title='Y correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                                                          atom=tb.Atom.from_dtype(y_correlations[index].dtype),
                                                          shape=y_correlations[index].shape,
                                                          filters=tb.Filters(complib='blosc',
                                                                             complevel=5,
                                                                             fletcher32=False))
                out_y_reduced.attrs.resolution = resolution[1]
                out_y_reduced.attrs.ref_size = ref_y_size[index]
                out_y_reduced.attrs.dut_size = dut_y_size[index]
                # correlation matrix
                out_x[:] = x_correlations[index]
                out_y[:] = y_correlations[index]
                for correlation in [x_correlations[index], y_correlations[index]]:
                    uu, dd, vv = np.linalg.svd(correlation)  # sigular value decomposition
                    background = np.matrix(uu[:, :1]) * np.diag(dd[:1]) * np.matrix(vv[:1, :])  # take first sigular value for background
                    background = np.array(background, dtype=np.int32)  # make Numpy array
                    correlation -= background  # remove background
                    correlation -= correlation.min()  # only positive values
                out_x_reduced[:] = x_correlations[index]
                out_y_reduced[:] = y_correlations[index]
            progress_bar.finish()

    if plot:
        plot_utils.plot_correlations(input_correlation_file=output_correlation_file, resolution=resolution, dut_names=telescope.dut_names)


# Helper functions to be called from multiple processes
def _correlate_position(ref, dut, ref_use_clusters, ref_indices, dut_input_file, start_index, start_event_number, stop_event_number, resolution, ref_x_size, ref_y_size, dut_x_size, dut_y_size, x_correlation, y_correlation, chunk_size):
    with tb.open_file(dut_input_file, mode='r') as in_file_h5:  # Open other DUT hit/cluster file
        try:
            dut_node = in_file_h5.root.Clusters
            dut_use_clusters = True
        except tb.NoSuchNodeError:
            dut_node = in_file_h5.root.Hits
            dut_use_clusters = False
        for dut_indices, start_index in analysis_utils.data_aligned_at_events(dut_node, start_index=start_index, start_event_number=start_event_number, stop_event_number=stop_event_number, chunk_size=chunk_size, fail_on_missing_events=False):  # Loop over the cluster in the actual cluster file in chunks
            ref_event_number = ref_indices['event_number']
            dut_event_number = dut_indices['event_number']
            if ref_use_clusters:
                ref_x_pos, ref_y_pos, _ = ref.index_to_global_position(column=ref_indices['mean_column'], row=ref_indices['mean_row'])
            else:
                ref_x_pos, ref_y_pos, _ = ref.index_to_global_position(column=ref_indices['column'], row=ref_indices['row'])
            if dut_use_clusters:
                dut_x_pos, dut_y_pos, _ = dut.index_to_global_position(column=dut_indices['mean_column'], row=dut_indices['mean_row'])
            else:
                dut_x_pos, dut_y_pos, _ = dut.index_to_global_position(column=dut_indices['column'], row=dut_indices['row'])
            ref_x_index = ((ref_x_pos - ref.translation_x + ref_x_size / 2.0) / resolution[0]).astype(np.uint32)
            ref_y_index = ((ref_y_pos - ref.translation_y + ref_y_size / 2.0) / resolution[1]).astype(np.uint32)
            dut_x_index = ((dut_x_pos - dut.translation_x + dut_x_size / 2.0) / resolution[0]).astype(np.uint32)
            dut_y_index = ((dut_y_pos - dut.translation_y + dut_y_size / 2.0) / resolution[1]).astype(np.uint32)

            analysis_utils.correlate_position_on_event_number(
                ref_event_number=ref_event_number,
                dut_event_number=dut_event_number,
                ref_x_index=ref_x_index,
                ref_y_index=ref_y_index,
                dut_x_index=dut_x_index,
                dut_y_index=dut_y_index,
                x_corr_hist=x_correlation,
                y_corr_hist=y_correlation)

    return start_index, x_correlation, y_correlation


def merge_cluster_data(telescope_configuration, input_cluster_files, output_merged_file, chunk_size=1000000):
    '''Takes the cluster from all cluster files and merges them into one big table aligned at a common event number.

    Empty entries are signaled with column = row = charge = nan. Position is translated from indices to local position (um).
    The local coordinate system origin (0, 0, 0) is defined to be in the sensor center, decoupling translation and rotation.
    Cluster position errors are calculated from cluster dimensions.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_cluster_files : list of pytables files
        File name of the input cluster files with correlation data.
    output_merged_file : pytables file
        File name of the output tracklet file.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Merge cluster files from %d DUTs ===', n_duts)

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
        description.append(('cluster_shape_dut_%d' % index, np.int64))
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
        merged_cluster_table = out_file_h5.create_table(out_file_h5.root, name='MergedClusters', description=np.dtype(description), title='Merged cluster on event number', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
        with tb.open_file(input_cluster_files[0], mode='r') as in_file_h5:  # Open DUT0 cluster file
            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Clusters.shape[0], term_width=80)
            progress_bar.start()
            for actual_cluster_dut_0, start_indices_data_loop[0] in analysis_utils.data_aligned_at_events(in_file_h5.root.Clusters, start_index=start_indices_data_loop[0], start_event_number=actual_start_event_number, stop_event_number=None, chunk_size=chunk_size):  # Loop over the cluster of DUT0 in chunks
                actual_event_numbers = actual_cluster_dut_0[:]['event_number']

                # First loop: calculate the minimum event number indices needed to merge all cluster from all files to this event number index
                common_event_numbers = actual_event_numbers
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open DUT0 cluster file
                        for actual_cluster, start_indices_merging_loop[dut_index] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Clusters, start_index=start_indices_merging_loop[dut_index], start_event_number=actual_start_event_number, stop_event_number=actual_event_numbers[-1] + 1, chunk_size=chunk_size, fail_on_missing_events=False):  # Loop over the cluster in the actual cluster file in chunks
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
                merged_cluster_array['x_dut_0'][selection], merged_cluster_array['y_dut_0'][selection], merged_cluster_array['z_dut_0'][selection] = telescope[0].index_to_local_position(
                    column=actual_cluster_dut_0['mean_column'][selection],
                    row=actual_cluster_dut_0['mean_row'][selection])
                # TODO: Calculate error
                xerr = np.zeros(selection.shape)
                yerr = np.zeros(selection.shape)
                zerr = np.zeros(selection.shape)
                xerr[selection] = actual_cluster_dut_0['err_column'][selection] * telescope[0].column_size
                yerr[selection] = actual_cluster_dut_0['err_row'][selection] * telescope[0].row_size
                merged_cluster_array['xerr_dut_0'][selection] = xerr[selection]
                merged_cluster_array['yerr_dut_0'][selection] = yerr[selection]
                merged_cluster_array['zerr_dut_0'][selection] = zerr[selection]
                merged_cluster_array['charge_dut_0'][selection] = actual_cluster_dut_0['charge'][selection]
                merged_cluster_array['n_hits_dut_0'][selection] = actual_cluster_dut_0['n_hits'][selection]
                merged_cluster_array['cluster_shape_dut_0'][selection] = actual_cluster_dut_0['cluster_shape'][selection]
                merged_cluster_array['n_cluster_dut_0'][selection] = actual_cluster_dut_0['n_cluster'][selection]

                # Fill result array with other DUT data
                # Second loop: get the cluster from all files and merge them to the common event number
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open other DUT cluster file
                        for actual_cluster_dut, start_indices_data_loop[dut_index] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Clusters, start_index=start_indices_data_loop[dut_index], start_event_number=common_event_numbers[0], stop_event_number=common_event_numbers[-1] + 1, chunk_size=chunk_size, fail_on_missing_events=False):  # Loop over the cluster in the actual cluster file in chunks
                            actual_cluster_dut = analysis_utils.map_cluster(common_event_numbers, actual_cluster_dut)
                            # Select real hits, values with nan are virtual hits
                            selection = ~np.isnan(actual_cluster_dut['mean_column'])
                            # Convert indices to positions, origin in the center of the sensor, remaining DUTs
                            merged_cluster_array['x_dut_%d' % (dut_index)][selection], merged_cluster_array['y_dut_%d' % (dut_index)][selection], merged_cluster_array['z_dut_%d' % (dut_index)][selection] = telescope[dut_index].index_to_local_position(
                                column=actual_cluster_dut['mean_column'][selection],
                                row=actual_cluster_dut['mean_row'][selection])
                            xerr = np.zeros(selection.shape)
                            yerr = np.zeros(selection.shape)
                            zerr = np.zeros(selection.shape)
                            xerr[selection] = actual_cluster_dut['err_column'][selection] * telescope[dut_index].column_size
                            yerr[selection] = actual_cluster_dut['err_row'][selection] * telescope[dut_index].row_size
                            merged_cluster_array['xerr_dut_%d' % (dut_index)][selection] = xerr[selection]
                            merged_cluster_array['yerr_dut_%d' % (dut_index)][selection] = yerr[selection]
                            merged_cluster_array['zerr_dut_%d' % (dut_index)][selection] = zerr[selection]
                            merged_cluster_array['charge_dut_%d' % (dut_index)][selection] = actual_cluster_dut['charge'][selection]
                            merged_cluster_array['n_hits_dut_%d' % (dut_index)][selection] = actual_cluster_dut['n_hits'][selection]
                            merged_cluster_array['cluster_shape_dut_%d' % (dut_index)][selection] = actual_cluster_dut['cluster_shape'][selection]
                            merged_cluster_array['n_cluster_dut_%d' % (dut_index)][selection] = actual_cluster_dut['n_cluster'][selection]

                merged_cluster_table.append(merged_cluster_array)
                merged_cluster_table.flush()
                actual_start_event_number = common_event_numbers[-1] + 1  # Set the starting event number for the next chunked read
                progress_bar.update(start_indices_data_loop[0])
            progress_bar.finish()


if __name__ == '__main__':
    pass
