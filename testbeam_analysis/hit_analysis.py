''' All functions acting on the hits of one DUT are listed here'''
from __future__ import division

import logging
import os.path
import re

import tables as tb
import numpy as np
from scipy.ndimage import median_filter
from pixel_clusterizer.clusterizer import HitClusterizer

from testbeam_analysis.tools import smc
from testbeam_analysis.tools import analysis_utils, plot_utils
from testbeam_analysis.tools.plot_utils import plot_masked_pixels, plot_cluster_size


def check_file(input_hits_file, n_pixel, output_check_file=None,
               event_range=1, plot=True, chunk_size=1000000):
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

    logging.info('=== Check data of hit file %s ===', input_hits_file)

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
                if (np.any(hits['column'] > n_pixel[0])
                        or np.any(hits['row'] > n_pixel[1])):
                    raise RuntimeError('The column/row definition exceed the nuber \
                    of pixels (%s/%s)!', n_pixel[0], n_pixel[1])

                analysis_utils.correlate_hits_on_event_range(hits,
                                                             col_corr,
                                                             row_corr,
                                                             event_range)

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


def generate_pixel_mask(input_hits_file, n_pixel, pixel_mask_name="NoisyPixelMask", output_mask_file=None, pixel_size=None, threshold=10.0, filter_size=3, dut_name=None, plot=True, chunk_size=1000000):
    '''Generating pixel mask from the hit table.

    Parameters
    ----------
    input_hits_file : string
        File name of the hit table.
    n_pixel : tuple
        Tuple of the total number of pixels (column/row).
    pixel_mask_name : string
        Name of the node containing the mask inside the output file.
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
    logging.info('=== Generating %s for %s ===', ' '.join(item.lower() for item in re.findall('[A-Z][^A-Z]*', pixel_mask_name)), input_hits_file)

    if output_mask_file is None:
        output_mask_file = os.path.splitext(input_hits_file)[0] + '_' + '_'.join(item.lower() for item in re.findall('[A-Z][^A-Z]*', pixel_mask_name)) + '.h5'

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


def cluster_hits(input_hits_file, output_cluster_file=None, input_disabled_pixel_mask_file=None, input_noisy_pixel_mask_file=None, min_hit_charge=0, max_hit_charge=None, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=1, dut_name=None, plot=True, chunk_size=1000000):
    '''Clusters the hits in the data file containing the hit table.

    Parameters
    ----------
    input_hits_file : string
        Filename of the input hits file.
    output_cluster_file : string
        Filename of the output cluster file. If None, the filename will be derived from the input hits file.
    input_disabled_pixel_mask_file : string
        Filename of the input disabled mask file.
    input_noisy_pixel_mask_file : string
        Filename of the input disabled mask file.
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
    logging.info('=== Clustering hits in %s ===', input_hits_file)

    if output_cluster_file is None:
        output_cluster_file = os.path.splitext(input_hits_file)[0] + '_clustered.h5'

    # Get noisy and disabled pixel, they are excluded for clusters
    if input_disabled_pixel_mask_file is not None:
        with tb.open_file(input_disabled_pixel_mask_file, 'r') as input_mask_file_h5:
            disabled_pixels = np.dstack(np.nonzero(input_mask_file_h5.root.DisabledPixelMask[:]))[0] + 1
    else:
        disabled_pixels = None
    if input_noisy_pixel_mask_file is not None:
        with tb.open_file(input_noisy_pixel_mask_file, 'r') as input_mask_file_h5:
            noisy_pixels = np.dstack(np.nonzero(input_mask_file_h5.root.NoisyPixelMask[:]))[0] + 1
    else:
        noisy_pixels = None

    # Prepare clusterizer

    # Define end of cluster function to
    # calculate the size in col/row for each cluster
    def end_of_cluster_function(hits, clusters, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels, disabled_pixels, seed_hit_index):
        min_col = hits[cluster_hit_indices[0]].column
        max_col = hits[cluster_hit_indices[0]].column
        min_row = hits[cluster_hit_indices[0]].row
        max_row = hits[cluster_hit_indices[0]].row
        for i in cluster_hit_indices[1:]:
            if i < 0:  # Not used indeces = -1
                break
            if hits[i].column < min_col:
                min_col = hits[i].column
            if hits[i].column > max_col:
                max_col = hits[i].column
            if hits[i].row < min_row:
                min_row = hits[i].row
            if hits[i].row > max_row:
                max_row = hits[i].row
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
            node_desc={'name': 'Cluster'},
            align_at='event_number',
            chunk_size=chunk_size)

    # Calculate cluster size histogram
    def hist_func(cluster):
        n_hits = cluster['n_hits']
        hist = analysis_utils.hist_1d_index(n_hits,
                                            shape=(np.max(n_hits) + 1,))
        return hist

    smc.SMC(table_file_in=output_cluster_file,
            file_out=output_cluster_file[:-3] + '_hist.h5',
            func=hist_func,
            node_desc={'name': 'HistClusterSize'},
            chunk_size=chunk_size)

    # Load infos from cluster size for error determination and plotting
    with tb.open_file(output_cluster_file[:-3] + '_hist.h5', 'r') as input_file_h5:
        hight = input_file_h5.root.HistClusterSize[:]
        n_clusters = hight.sum()
        n_hits = (hight * np.arange(0, hight.shape[0])).sum()
        max_cluster_size = hight.shape[0] - 1

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
    with tb.open_file(output_cluster_file, 'r+') as output_file_h5:
        # Copy nodes to result file
        if input_disabled_pixel_mask_file is not None:
            with tb.open_file(input_disabled_pixel_mask_file, 'r') as input_mask_file_h5:
                input_mask_file_h5.root.DisabledPixelMask._f_copy(newparent=output_file_h5.root)
        if input_noisy_pixel_mask_file is not None:
            with tb.open_file(input_noisy_pixel_mask_file, 'r') as input_mask_file_h5:
                input_mask_file_h5.root.NoisyPixelMask._f_copy(newparent=output_file_h5.root)

    if plot:
        plot_cluster_size(output_cluster_file, dut_name=os.path.split(output_cluster_file)[1],
                          output_pdf_file=os.path.splitext(output_cluster_file)[0] + '_cluster_size.pdf',
                          chunk_size=chunk_size, gui=False)

    return output_cluster_file


if __name__ == '__main__':
    pass
