''' All functions acting on the hits of one DUT are listed here'''
from __future__ import division

import logging
import os.path
import math
from multiprocessing import Pool

import tables as tb
import numpy as np
import numba
from scipy.ndimage import median_filter

import progressbar

from pixel_clusterizer.clusterizer import HitClusterizer

from beam_telescope_analysis.telescope.telescope import Telescope
from beam_telescope_analysis.tools import smc
from beam_telescope_analysis.tools import analysis_utils, plot_utils
from beam_telescope_analysis.tools.plot_utils import plot_masked_pixels, plot_cluster_hists


def convert(telescope_configuration, input_hit_files, output_hit_files=None, select_duts=None, index_to_local=True, chunk_size=1000000):
    '''"Converting hit files. Wrapper for convert_coordinates(). For detailed description of the parameters see convert_coordinates().

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_hit_files : list
        Filenames of the input hit files.
    output_hit_files : list
        Filenames of the output hit files.
    select_duts : list
        List of selected DUTs.
    index_to_local : bool
        If True, convert index to local coordinates.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_files : list
        Filenames of the output hit files.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    if select_duts is None:
        select_duts = range(len(telescope))
    if not isinstance(select_duts, (list, tuple)):
        raise ValueError('Parameter "select_duts" is not a list.')
    if max(select_duts) > n_duts or min(select_duts) < 0:
        raise ValueError('Parameter "select_duts" contains ivalid values.')
    logging.info('=== Converting coordinates of %d DUTs ===', len(select_duts))

    selected_telescope_duts = [dut for i, dut in enumerate(telescope) if i in select_duts]
    if len(select_duts) != len(input_hit_files):
        raise ValueError('Parameter "input_hit_files" has wrong length.')
    if output_hit_files is not None and len(select_duts) != len(output_hit_files):
        raise ValueError('Parameter "output_hit_files" has wrong length.')

    output_files = []
    for i, dut in enumerate(selected_telescope_duts):
        output_files.append(convert_coordinates(
            dut=dut,
            input_hit_file=input_hit_files[i],
            output_hit_file=None if output_hit_files is None else output_hit_files[i],
            index_to_local=index_to_local,
            chunk_size=chunk_size))

    return output_files


def convert_coordinates(dut, input_hit_file, output_hit_file=None, index_to_local=True, chunk_size=1000000):
    '''Convert index to local coordinates and vice versa.

    Parameters
    ----------
    dut : object
        DUT object.
    input_hit_file : string
        Filename of the input hits file.
    output_hit_file : string
        Filename of the output hits file.
    index_to_local : bool
        If True, convert index to local coordinates.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_file : string
        Filename of the output hits file with local coordinates.
    '''

    if output_hit_file is None:
        output_file = os.path.splitext(input_hit_file)[0] + ('_local_coordinates.h5' if index_to_local else '_index.h5')

    with tb.open_file(input_hit_file, mode='r') as in_file_h5:
        with tb.open_file(output_file, mode='w') as out_file_h5:
            node = in_file_h5.root.Hits
            logging.info('== Converting coordinates of %s ==' % dut.name)

            out_dtype = []
            if index_to_local:
                for name, dtype in node.dtype.descr:
                    if name == 'column':
                        out_dtype.append(('x', np.float64))
                    elif name == 'row':
                        out_dtype.append(('y', np.float64))
                        # out_dtype.append(('z', np.float64))
                    else:
                        out_dtype.append((name, dtype))
            else:
                for name, dtype in node.dtype.descr:
                    if name == 'x':
                        out_dtype.append(('column', np.uint16))
                    elif name == 'y':
                        out_dtype.append(('row', np.uint16))
                    elif name == 'z':
                        continue
                    else:
                        out_dtype.append((name, dtype))
            out_dtype = np.dtype(out_dtype)
            out_table = out_file_h5.create_table(
                where=out_file_h5.root,
                name=node.name,
                description=out_dtype,
                title=node.title,
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))

            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=node.shape[0], term_width=80)
            progress_bar.start()

            for data_chunk, index in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):  # Loop over the hits
                out_data = np.empty(data_chunk.shape[0], dtype=out_dtype)
                for name in node.dtype.names:
                    if index_to_local:
                        if name not in ['column', 'row']:
                            out_data[name] = data_chunk[name]
                    else:
                        if name not in ['x', 'y', 'z']:
                            out_data[name] = data_chunk[name]

                if index_to_local:
                    # out_data['x'], out_data['y'], out_data['z'] = dut.index_to_local_position(
                    out_data['x'], out_data['y'], _ = dut.index_to_local_position(
                        column=data_chunk['column'],
                        row=data_chunk['row'])
                else:
                    out_data['column'], out_data['row'] = dut.local_position_to_index(
                        x=data_chunk['x'],
                        y=data_chunk['y'],
                        # z=data_chunk['z'])
                        z=0.0)

                out_table.append(out_data)
                progress_bar.update(index)
            progress_bar.finish()

    return output_file


def check(telescope_configuration, input_hit_files, output_check_files=None, select_duts=None, event_ranges=1, resolutions=None, plot=True, chunk_size=1000000):
    '''"Checking hit files. Wrapper for check_file(). For detailed description of the parameters see check_file().

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_hit_files : list
        Filenames of the input hit files.
    output_check_files : list
        Filenames of the output check files.
    select_duts : list
        List of selected DUTs.
    event_ranges : list
        List of event ranges.
    resolutions : list
        List of resolutions.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_files : list
        Filenames of the output check files.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    if select_duts is None:
        select_duts = range(len(telescope))
    if not isinstance(select_duts, (list, tuple)):
        raise ValueError('Parameter "select_duts" is not a list.')
    if max(select_duts) > n_duts or min(select_duts) < 0:
        raise ValueError('Parameter "select_duts" contains ivalid values.')
    logging.info('=== Checking hit files of %d DUTs ===', len(select_duts))

    selected_telescope_duts = [dut for i, dut in enumerate(telescope) if i in select_duts]
    if len(select_duts) != len(input_hit_files):
        raise ValueError('Parameter "input_hit_files" has wrong length.')
    if output_check_files is not None and len(select_duts) != len(output_check_files):
        raise ValueError('Parameter "output_check_files" has wrong length.')
    if isinstance(event_ranges, (list, tuple)):
        if len(select_duts) != len(event_ranges):
            raise ValueError('Parameter "event_ranges" has wrong length.')
    else:
        event_ranges = [event_ranges] * len(select_duts)
    if isinstance(resolutions, (list, tuple)):
        if len(select_duts) != len(resolutions):
            raise ValueError('Parameter "resolutions" has wrong length.')
    else:
        resolutions = [resolutions] * len(select_duts)

    output_files = []
    for i, dut in enumerate(selected_telescope_duts):
        output_files.append(check_file(
            dut=dut,
            input_hit_file=input_hit_files[i],
            output_check_file=None if output_check_files is None else output_check_files[i],
            event_range=event_ranges[i],
            resolution=None if resolutions is None else resolutions[i],
            plot=plot,
            chunk_size=chunk_size))

    return output_files


def check_file(dut, input_hit_file, output_check_file=None, event_range=1, resolution=None, plot=True, chunk_size=1000000):
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
    dut : object
        DUT object.
    input_hit_file : string
        Filename of the hit table.
    output_check_file : string
        Filename of the output check file.
    event_range : uint
        The number of events to use for correlation,
        e.g., event_range = 1 correlates to predecessing event hits with the current event hits.
    resolution : list
        2-tuple of resolutions for the x and y axis.
        If None, the resolution is corresponding the pixel size.
    chunk_size : int
        Chunk size of the data when reading from file.

    Returns
    -------
    output_check_file : string
        Filename of the output check file.
    '''
    logging.info('== Checking hit file of %s ==', dut.name)

    if output_check_file is None:
        output_check_file = os.path.splitext(input_hit_file)[0] + '_check.h5'

    with tb.open_file(output_check_file, mode='w') as out_file_h5:
        with tb.open_file(input_hit_file, mode='r') as in_file_h5:
            # Check for whether hits or clusters are used
            try:
                node = in_file_h5.root.Clusters
                use_clusters = True
            except tb.NoSuchNodeError:
                node = in_file_h5.root.Hits
                use_clusters = False
            # Check for whether local coordinates or indices are used
            if 'mean_x' in node.dtype.names or 'x' in node.dtype.names:
                use_positions = True
            else:
                use_positions = False
            # Set resolution if not provided
            if resolution is None:
                resolution = (dut.column_size, dut.row_size)
            # DUT size
            x_extent = dut.x_extent(global_position=False)
            x_size = x_extent[1] - x_extent[0]
            y_extent = dut.y_extent(global_position=False)
            y_size = y_extent[1] - y_extent[0]
            # Hist size
            x_center = 0.0
            hist_x_size = math.ceil(x_size / resolution[0]) * resolution[0]
            hist_x_extent = [x_center - hist_x_size / 2.0, x_center + hist_x_size / 2.0]
            y_center = 0.0
            hist_y_size = math.ceil(y_size / resolution[1]) * resolution[1]
            hist_y_extent = [y_center - hist_y_size / 2.0, y_center + hist_y_size / 2.0]
            # Creating histograms for the correlation
            x_correlation = np.zeros([int(hist_x_size / resolution[0]), int(hist_x_size / resolution[0])], dtype=np.int32)
            y_correlation = np.zeros([int(hist_y_size / resolution[1]), int(hist_y_size / resolution[1])], dtype=np.int32)

            # shape_column = (dut.n_columns, dut.n_columns)
            # shape_row = (dut.n_rows, dut.n_rows)
            # x_correlation = np.zeros(shape_column, dtype=np.int32)
            # y_correlation = np.zeros(shape_row, dtype=np.int32)
            last_event = None
            out_dE = out_file_h5.create_earray(
                where=out_file_h5.root,
                name='EventDelta',
                title='Change of event number per non empty event',
                shape=(0, ),
                atom=tb.Atom.from_dtype(np.dtype(np.uint64)),
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))
            out_E = out_file_h5.create_earray(
                where=out_file_h5.root,
                name='EventNumber',
                title='Event number of non empty event',
                shape=(0, ),
                atom=tb.Atom.from_dtype(np.dtype(np.uint64)),
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))

            last_event_number = 0
            for data_chunk, _ in analysis_utils.data_aligned_at_events(table=node, chunk_size=chunk_size):
                if not np.all(np.diff(np.r_[last_event_number, data_chunk['event_number']]) >= 0):
                    raise RuntimeError('The event number does not always increase.')
                if use_positions is False:
                    if np.any(data_chunk['column'] < 1) or np.any(data_chunk['row'] < 1):
                        raise RuntimeError('The column/row index does not start at 1!')
                    if np.any(data_chunk['column'] > dut.n_columns) or np.any(data_chunk['row'] > dut.n_rows):
                        raise RuntimeError('The column/row index exceeds the number of pixels (max. %s/%s)!' % (dut.n_columns, dut.n_rows))
                if use_positions is True:
                    x_extent = dut.x_extent()
                    y_extent = dut.y_extent()
                    if np.any(data_chunk['x'] < x_extent[0]) or np.any(data_chunk['x'] > x_extent[1]):
                        raise RuntimeError('The local x coordinates are exceeding the limits (min./max. %s/%s)!' % (dut.x_extent[0], dut.x_extent[1]))
                    if np.any(data_chunk['y'] < y_extent[0]) or np.any(data_chunk['y'] > y_extent[1]):
                        raise RuntimeError('The local y coordinates are exceeding the limits (min./max. %s/%s)!' % (dut.y_extent[0], dut.y_extent[1]))

                if use_clusters:
                    if use_positions:
                        x_pos, y_pos = data_chunk['mean_x'], data_chunk['mean_y']
                    else:
                        x_pos, y_pos, _ = dut.index_to_local_position(column=data_chunk['mean_column'], row=data_chunk['mean_row'])
                else:
                    if use_positions:
                        x_pos, y_pos = data_chunk['x'], data_chunk['y']
                    else:
                        x_pos, y_pos, _ = dut.index_to_local_position(column=data_chunk['column'], row=data_chunk['row'])
                x_indices = ((x_pos - hist_x_extent[0]) / resolution[0]).astype(np.uint32)
                y_indices = ((y_pos - hist_y_extent[0]) / resolution[1]).astype(np.uint32)
                analysis_utils.correlate_hits_on_event_range(
                    event_numbers=data_chunk['event_number'],
                    x_indices=x_indices,
                    y_indices=y_indices,
                    x_corr_hist=x_correlation,
                    y_corr_hist=y_correlation,
                    event_range=event_range)
                event_numbers = np.unique(data_chunk['event_number'])
                event_delta = np.diff(event_numbers)

                if last_event:
                    event_delta = np.concatenate((np.array([event_numbers[0] - last_event]),
                                                  event_delta))
                last_event = event_numbers[-1]

                out_dE.append(event_delta)
                out_E.append(event_numbers)
                last_event_number = event_numbers[-1]

            out_col = out_file_h5.create_carray(
                where=out_file_h5.root,
                name='CorrelationColumns',
                title='Column Correlation with event range=%s' % event_range,
                atom=tb.Atom.from_dtype(x_correlation.dtype),
                shape=x_correlation.shape,
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))
            out_row = out_file_h5.create_carray(
                where=out_file_h5.root,
                name='CorrelationRows',
                title='Row Correlation with event range=%s' % event_range,
                atom=tb.Atom.from_dtype(y_correlation.dtype),
                shape=y_correlation.shape,
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))
            out_col[:] = x_correlation
            out_row[:] = y_correlation

    if plot:
        plot_utils.plot_checks(input_corr_file=output_check_file)

    return output_check_file


def mask(telescope_configuration, input_hit_files, output_mask_files=None, select_duts=None, pixel_mask_names="NoisyPixelMask", iterations=None, thresholds=10.0, filter_sizes=3, plot=True, chunk_size=1000000):
    '''"Masking noisy pixels.
    Wrapper for mask_pixels(). For detailed description of the parameters see mask_pixels().

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_hit_files : list
        Filenames of the input hit files.
    output_mask_files : list
        Filenames of the output mask files.
    select_duts : list
        List of selected DUTs.
    pixel_mask_names : list
        Pixel mask type for each DUT. Possible mask types:
          - DisabledPixelMask
          - NoisyPixelMask
    iterations : list
        The iterations for pixel masking.
    thresholds : list
        The thresholds for pixel masking.
    filter_sizes : list
        Median filter sizes.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_files : list
        Filenames of the output mask files.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    if select_duts is None:
        select_duts = range(len(telescope))
    if not isinstance(select_duts, (list, tuple)):
        raise ValueError('Parameter "select_duts" is not a list.')
    if max(select_duts) > n_duts or min(select_duts) < 0:
        raise ValueError('Parameter "select_duts" contains ivalid values.')
    logging.info('=== Masking pixels of %d DUTs ===', len(select_duts))

    selected_telescope_duts = [dut for i, dut in enumerate(telescope) if i in select_duts]
    if len(select_duts) != len(input_hit_files):
        raise ValueError('Parameter "input_hit_files" has wrong length.')
    if output_mask_files is not None and len(select_duts) != len(output_mask_files):
        raise ValueError('Parameter "output_mask_files" has wrong length.')
    if isinstance(pixel_mask_names, (list, tuple)):
        if len(select_duts) != len(pixel_mask_names):
            raise ValueError('Parameter "pixel_mask_names" has wrong length.')
    else:
        pixel_mask_names = [pixel_mask_names] * len(select_duts)
    if isinstance(iterations, (list, tuple)):
        if len(select_duts) != len(iterations):
            raise ValueError('Parameter "iterations" has wrong length.')
    else:
        iterations = [iterations] * len(iterations)
    if isinstance(thresholds, (list, tuple)):
        if len(select_duts) != len(thresholds):
            raise ValueError('Parameter "thresholds" has wrong length.')
    else:
        thresholds = [thresholds] * len(select_duts)
    if isinstance(filter_sizes, (list, tuple)):
        if len(select_duts) != len(filter_sizes):
            raise ValueError('Parameter "filter_sizes" has wrong length.')
    else:
        filter_sizes = [filter_sizes] * len(select_duts)

    output_files = []
    for i, dut in enumerate(selected_telescope_duts):
        output_files.append(mask_pixels(
            dut=dut,
            input_hit_file=input_hit_files[i],
            pixel_mask_name=pixel_mask_names[i],
            output_mask_file=None if output_mask_files is None else output_mask_files[i],
            iterations=iterations[i],
            threshold=thresholds[i],
            filter_size=filter_sizes[i],
            plot=plot,
            chunk_size=chunk_size))
    return output_files


def mask_pixels(dut, input_hit_file, pixel_mask_name="NoisyPixelMask", output_mask_file=None, iterations=None, threshold=10.0, filter_size=3, plot=True, chunk_size=1000000):
    '''Generating pixel mask from the hit table.
    The pixel masking is an iterative process to identify and suppress any noisy pixel (cluster).
    The iterative process stops when no more noisy pixels are found or the maximum number of iterations is reached.

    Parameters
    ----------
    dut : object
        DUT object.
    input_hit_file : string
        Filename of the input hit file.
    pixel_mask_name : string
        Name of the node containing the mask inside the output file. Possible mask types:
          - DisabledPixelMask: Any noisy pixels is masked and not taken into account when building clusters.
          - NoisyPixelMask: Clusters solely containing noisy pixels are not built.
    output_mask_file : string
        Filename of the output mask file.
    iterations : int
        Maximum number of itaration steps.
        If None or 0, the number of steps is not limited.
    threshold : float
        The threshold for pixel masking controls the sensitivity for detecting noisy pixels.
        The threshold is multiplied with the standard deviation of the pixel hit count (background subtracted).
        The lower the threshold, the more pixels are masked.
    filter_size : scalar or tuple
        Adjust the median filter size by giving the number of columns and rows.
        The higher the value, the more the background is smoothed and the more pixels are masked.
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.

    Returns
    -------
    output_mask_file : string
        Filename of the output mask file.
    '''
    logging.info('== Masking pixels of %s (%s) ==', dut.name, input_hit_file)

    if pixel_mask_name not in ["DisabledPixelMask", "NoisyPixelMask"]:
        raise ValueError("'%s' is not a valid pixel mask name." % pixel_mask_name)

    if output_mask_file is None:
        output_mask_file = os.path.splitext(input_hit_file)[0] + '_mask.h5'

    # Check indices are available
    with tb.open_file(input_hit_file, mode='r') as in_file_h5:
        node = in_file_h5.root.Hits
        if 'column' not in node.dtype.names or 'row' not in node.dtype.names:
            raise RuntimeError('Input hit file has no column/row indices')

    # Create occupancy histogram
    def work(hit_chunk):
        col, row = hit_chunk['column'], hit_chunk['row']
        return analysis_utils.hist_2d_index(col - 1, row - 1, shape=dut.n_pixel)

    smc.SMC(input_filename=input_hit_file,
            output_filename=output_mask_file,
            table='Hits',
            func=work,
            node_desc={'name': 'HistOcc'},
            chunk_size=chunk_size)

    # Create mask from occupancy histogram
    with tb.open_file(output_mask_file, mode='r+') as out_file_h5:
        occupancy = out_file_h5.root.HistOcc[:]
        pixel_mask = np.zeros_like(occupancy, dtype=np.bool)
        iterations = 100
        i = 0
        while(True):
            i += 1
            # Run median filter across data, assuming 0 filling past the edges to get expected occupancy
            blurred = median_filter(occupancy.astype(np.int32), size=filter_size, mode='constant', cval=0.0)
            # Spot noisy pixels by substracting expected occupancy from measured occupancy
            difference = occupancy - blurred
            # try to normalize to statistics, prevent masking pixels with low hit counts and uneven hit distribution
            # also prevent from masking pixels with 0 hit count
            std = np.std(difference[occupancy != 0] / np.sqrt(occupancy[occupancy != 0]))
            # reverse normalization and calculate threshold per pixel
            abs_occ_threshold = threshold * std * np.sqrt(occupancy[occupancy != 0])
            tmp_pixel_mask = np.zeros_like(pixel_mask)
            tmp_pixel_mask[occupancy != 0] = difference[occupancy != 0] > abs_occ_threshold
            n_new_masked_pixels = np.count_nonzero(tmp_pixel_mask & (tmp_pixel_mask ^ pixel_mask))
            logging.info('Iteration %d: Masked %d pixels', i, n_new_masked_pixels)
            pixel_mask |= tmp_pixel_mask
            occupancy[pixel_mask] = blurred[pixel_mask]
            if n_new_masked_pixels == 0 or (iterations and i >= iterations):
                break
        logging.info('Masked %d pixels in total at threshold %.1f', np.count_nonzero(pixel_mask), threshold)

        # Create masked pixels array
        masked_pixel_table = out_file_h5.create_carray(
            where=out_file_h5.root,
            name=pixel_mask_name,
            title='Pixel Mask',
            atom=tb.Atom.from_dtype(pixel_mask.dtype),
            shape=pixel_mask.shape,
            filters=tb.Filters(
                complib='blosc',
                complevel=5,
                fletcher32=False))
        masked_pixel_table[:] = pixel_mask

    if plot:
        plot_masked_pixels(input_mask_file=output_mask_file, pixel_size=(dut.column_size, dut.row_size), dut_name=dut.name)

    return output_mask_file


def cluster(telescope_configuration, input_hit_files, input_mask_files=None, output_cluster_files=None, select_duts=None, use_positions=None, min_hit_charges=0, max_hit_charges=None, column_cluster_distances=1, row_cluster_distances=1, frame_cluster_distances=0, plot=True, chunk_size=1000000):
    '''Clustering hits. Wrapper for cluster_hits(). For detailed description of the parameters see cluster_hits().

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_hit_files : list
        Filenames of the input hit files.
    input_mask_files : list
        Filenames of the input mask files.
    output_cluster_files : list
        Filenames of the output cluster files.
    select_duts : list
        Selecting DUTs that will be processed.
        If None, for all DUTs are selected.
    use_positions : list
        If True, cluster local positions instead of hit indices.
        Conversion to local position is done on the fly, if input hit file provides hit indices.
        If None, automatically decide from input hit file format.
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

    Returns
    -------
    output_files : list
        Filenames of the output cluster files.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    if select_duts is None:
        select_duts = range(len(telescope))
    if not isinstance(select_duts, (list, tuple)):
        raise ValueError('Parameter "select_duts" is not a list.')
    if max(select_duts) > n_duts or min(select_duts) < 0:
        raise ValueError('Parameter "select_duts" contains ivalid values.')
    logging.info('=== Clustering hits of %d DUTs ===', len(select_duts))

    selected_telescope_duts = [dut for i, dut in enumerate(telescope) if i in select_duts]
    if len(select_duts) != len(input_hit_files):
        raise ValueError('Parameter "input_hit_files" has wrong length.')
    if input_mask_files is not None and len(select_duts) != len(input_mask_files):
        raise ValueError('Parameter "input_mask_files" has wrong length.')
    if output_cluster_files is not None and len(select_duts) != len(output_cluster_files):
        raise ValueError('Parameter "output_cluster_files" has wrong length.')
    if isinstance(use_positions, (list, tuple)):
        if len(select_duts) != len(use_positions):
            raise ValueError('Parameter "use_positions" has wrong length.')
    else:
        use_positions = [use_positions] * len(select_duts)
    if isinstance(min_hit_charges, (list, tuple)):
        if len(select_duts) != len(min_hit_charges):
            raise ValueError('Parameter "min_hit_charges" has wrong length.')
    else:
        min_hit_charges = [min_hit_charges] * len(select_duts)
    if isinstance(max_hit_charges, (list, tuple)):
        if len(select_duts) != len(max_hit_charges):
            raise ValueError('Parameter "max_hit_charges" has wrong length.')
    else:
        max_hit_charges = [max_hit_charges] * len(select_duts)
    if isinstance(column_cluster_distances, (list, tuple)):
        if len(select_duts) != len(column_cluster_distances):
            raise ValueError('Parameter "column_cluster_distances" has wrong length.')
    else:
        column_cluster_distances = [column_cluster_distances] * len(select_duts)
    if isinstance(row_cluster_distances, (list, tuple)):
        if len(select_duts) != len(row_cluster_distances):
            raise ValueError('Parameter "row_cluster_distances" has wrong length.')
    else:
        row_cluster_distances = [row_cluster_distances] * len(select_duts)
    if isinstance(frame_cluster_distances, (list, tuple)):
        if len(select_duts) != len(frame_cluster_distances):
            raise ValueError('Parameter "frame_cluster_distances" has wrong length.')
    else:
        frame_cluster_distances = [frame_cluster_distances] * len(select_duts)

    output_files = []
    for i, dut in enumerate(selected_telescope_duts):
        output_files.append(cluster_hits(
            dut=dut,
            input_hit_file=input_hit_files[i],
            output_cluster_file=None if output_cluster_files is None else output_cluster_files[i],
            input_mask_file=None if input_mask_files is None else input_mask_files[i],
            use_positions=use_positions[i],
            min_hit_charge=min_hit_charges[i],
            max_hit_charge=max_hit_charges[i],
            column_cluster_distance=column_cluster_distances[i],
            row_cluster_distance=row_cluster_distances[i],
            frame_cluster_distance=frame_cluster_distances[i],
            plot=plot,
            chunk_size=chunk_size))
    return output_files


def cluster_hits(dut, input_hit_file, output_cluster_file=None, input_mask_file=None, use_positions=None, min_hit_charge=0, max_hit_charge=None, column_cluster_distance=1, row_cluster_distance=1, frame_cluster_distance=0, plot=True, chunk_size=1000000):
    '''Clusters the hits in the data file containing the hit table.

    Parameters
    ----------
    dut : object
        DUT object.
    input_hit_file : string
        Filename of the input hits file.
    output_cluster_file : string
        Filename of the output cluster file. If None, the filename will be derived from the input hits file.
    input_mask_file : string
        Filename of the input mask file.
    use_positions : bool
        If True, cluster local positions instead of hit indices.
        Conversion to local position is done on the fly, if input hit file provides hit indices.
        If None, automatically decide from input hit file format.
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
    plot : bool
        If True, create additional output plots.
    chunk_size : int
        Chunk size of the data when reading from file.

    Returns
    -------
    output_cluster_file : string
        Filename of the output cluster file.
    '''
    logging.info('== Clustering hits in %s ==', input_hit_file)

    if output_cluster_file is None:
        output_cluster_file = os.path.splitext(input_hit_file)[0] + '_clustered.h5'

    # Check for whether local coordinates or indices are used
    with tb.open_file(input_hit_file, mode='r') as input_file_h5:
        node = input_file_h5.root.Hits
        if use_positions is None:
            if 'x' in node.dtype.names:
                use_positions = True
            else:
                use_positions = False
            convert_to_positions = False
        else:
            # check if hit indices needs to be converted to local coordinates
            if use_positions is True:
                if 'x' in node.dtype.names:
                    convert_to_positions = False
                else:
                    convert_to_positions = True
            else:
                convert_to_positions = False
        # check whether hit indices can be copied to cluster hits array
        if use_positions:
            if 'column' in node.dtype.names:
                copy_hit_indices = True
            else:
                copy_hit_indices = False
        else:
            copy_hit_indices = False

    # Getting noisy and disabled pixel mask
    if input_mask_file is not None:
        with tb.open_file(input_mask_file, mode='r') as input_mask_file_h5:
            try:
                disabled_pixels = np.dstack(np.nonzero(input_mask_file_h5.root.DisabledPixelMask[:]))[0] + 1  # index starts at 1
            except tb.NoSuchNodeError:
                disabled_pixels = None
            try:
                noisy_pixels = np.dstack(np.nonzero(input_mask_file_h5.root.NoisyPixelMask[:]))[0] + 1  # index starts at 1
            except tb.NoSuchNodeError:
                noisy_pixels = None
    else:
        disabled_pixels = None
        noisy_pixels = None
    # Check for valid inputs
    if use_positions and (disabled_pixels is not None or noisy_pixels is not None):
        raise ValueError('Cannot use pixel masks when using local coordinates.')

    # Prepare clusterizer
    # Define end of cluster function to
    # calculate the size in col/row for each cluster
    @numba.njit(locals={'diff_col': numba.int32, 'diff_row': numba.int32, 'cluster_shape': numba.int64})
    def end_of_cluster_function_with_index(hits, clusters, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels, disabled_pixels, seed_hit_index):
        hit_arr = np.zeros((15, 15), dtype=np.bool_)
        center_col = hits[cluster_hit_indices[0]].column
        center_row = hits[cluster_hit_indices[0]].row
        hit_arr[7, 7] = 1
        min_col = hits[cluster_hit_indices[0]].column
        max_col = hits[cluster_hit_indices[0]].column
        min_row = hits[cluster_hit_indices[0]].row
        max_row = hits[cluster_hit_indices[0]].row
        min_frame = hits[cluster_hit_indices[0]].frame
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
            if hits[i].frame < min_frame:
                min_frame = hits[i].frame

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
        clusters[cluster_index].err_column = max_col - min_col
        clusters[cluster_index].err_row = max_row - min_row
        clusters[cluster_index].frame = min_frame

    @numba.njit()
    def end_of_cluster_function_with_position(hits, clusters, cluster_size, cluster_hit_indices, cluster_index, cluster_id, charge_correction, noisy_pixels, disabled_pixels, seed_hit_index):
        min_col = hits[cluster_hit_indices[0]].column
        max_col = hits[cluster_hit_indices[0]].column
        min_row = hits[cluster_hit_indices[0]].row
        max_row = hits[cluster_hit_indices[0]].row
        min_frame = hits[cluster_hit_indices[0]].frame
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
            if hits[i].frame < min_frame:
                min_frame = hits[i].frame

        # no cluster shape available
        clusters[cluster_index].cluster_shape = -1
        clusters[cluster_index].err_x = max_col - min_col
        clusters[cluster_index].err_y = max_row - min_row
        clusters[cluster_index].frame = min_frame

    # Adding number of clusters
    def end_of_event_function(hits, clusters, start_event_hit_index, stop_event_hit_index, start_event_cluster_index, stop_event_cluster_index):
        for i in range(start_event_cluster_index, stop_event_cluster_index):
            clusters[i]['n_cluster'] = hits["n_cluster"][start_event_hit_index]

    if use_positions:
        hit_dtype = np.dtype([
            ('event_number', '<i8'),
            ('x', '<f8'),
            ('y', '<f8'),
            ('charge', '<u2'),
            ('frame', '<u1'),
            ('cluster_ID', '<i2'),
            ('is_seed', '<u1'),
            ('cluster_size', '<u4'),
            ('n_cluster', '<u4')])
        hit_fields = {
            'x': 'column',
            'y': 'row'}
        cluster_fields = {
            'seed_x': 'seed_column',
            'seed_y': 'seed_row',
            'mean_x': 'mean_column',
            'mean_y': 'mean_row',
            'cluster_ID': 'ID'}
        cluster_dtype = np.dtype([
            ('event_number', '<i8'),
            ('cluster_ID', '<u2'),
            ('n_hits', '<u4'),
            ('charge', '<f4'),
            ('frame', '<u1'),
            ('seed_x', '<u2'),
            ('seed_y', '<u2'),
            ('mean_x', '<f8'),
            ('mean_y', '<f8')])
    else:
        hit_dtype = np.dtype([
            ('event_number', '<i8'),
            ('column', '<u2'),
            ('row', '<u2'),
            ('charge', '<u2'),
            ('frame', '<u1'),
            ('cluster_ID', '<i2'),
            ('is_seed', '<u1'),
            ('cluster_size', '<u4'),
            ('n_cluster', '<u4')])
        hit_fields = None
        cluster_fields = {
            'cluster_ID': 'ID'}
        cluster_dtype = np.dtype([
            ('event_number', '<i8'),
            ('cluster_ID', '<u2'),
            ('n_hits', '<u4'),
            ('charge', '<f4'),
            ('frame', '<u1'),
            ('seed_column', '<u2'),
            ('seed_row', '<u2'),
            ('mean_column', '<f8'),
            ('mean_row', '<f8')])

    clusterizer = HitClusterizer(
        hit_fields=hit_fields,
        cluster_fields=cluster_fields,
        column_cluster_distance=column_cluster_distance,
        row_cluster_distance=row_cluster_distance,
        frame_cluster_distance=frame_cluster_distance,
        min_hit_charge=min_hit_charge,
        max_hit_charge=max_hit_charge,
        hit_dtype=hit_dtype,
        cluster_dtype=cluster_dtype,
        pure_python=False)
    if use_positions:
        clusterizer.add_cluster_field(description=('err_x', '<f4'))  # Add an additional field to hold the cluster size in x
        clusterizer.add_cluster_field(description=('err_y', '<f4'))  # Add an additional field to hold the cluster size in y
    else:
        clusterizer.add_cluster_field(description=('err_column', '<f4'))  # Add an additional field to hold the cluster size in x
        clusterizer.add_cluster_field(description=('err_row', '<f4'))  # Add an additional field to hold the cluster size in y

    clusterizer.add_cluster_field(description=('n_cluster', '<u4'))  # Adding additional field for number of clusters per event
    clusterizer.add_cluster_field(description=('cluster_shape', '<i8'))  # Adding additional field for the cluster shape
    if use_positions:
        clusterizer.set_end_of_cluster_function(end_of_cluster_function_with_position)  # Set the new function to the clusterizer
    else:
        clusterizer.set_end_of_cluster_function(end_of_cluster_function_with_index)  # Set the new function to the clusterizer
    clusterizer.set_end_of_event_function(end_of_event_function)

    # Run clusterizer on hit table in parallel on all cores
    def cluster_func(hits, clusterizer, noisy_pixels, disabled_pixels, dut, convert_to_positions, copy_hit_indices):
        if convert_to_positions:
            dut_x_pos, dut_y_pos, _ = dut.index_to_local_position(column=hits['column'], row=hits['row'])
            new_hits_dtype = []
            for name, dtype in hits.dtype.descr:
                if name == 'column':
                    new_hits_dtype.append(('x', np.float64))
                elif name == 'row':
                    new_hits_dtype.append(('y', np.float64))
                else:
                    new_hits_dtype.append((name, dtype))
            new_hits = np.empty(shape=hits.shape, dtype=new_hits_dtype)
            for name in new_hits.dtype.names:
                if name == 'x':
                    new_hits['x'] = dut_x_pos
                elif name == 'y':
                    new_hits['y'] = dut_y_pos
                else:
                    new_hits[name] = hits[name]
            cluster_hits, clusters = clusterizer.cluster_hits(
                hits=new_hits,
                noisy_pixels=noisy_pixels,
                disabled_pixels=disabled_pixels)
        else:
            cluster_hits, clusters = clusterizer.cluster_hits(
                hits=hits,
                noisy_pixels=noisy_pixels,
                disabled_pixels=disabled_pixels)
        if copy_hit_indices:
            new_cluster_hits_dtype = cluster_hits.dtype.descr + [('column', np.uint16), ('row', np.uint16)]
            new_cluster_hits = np.empty(shape=hits.shape, dtype=new_cluster_hits_dtype)
            for name in new_cluster_hits.dtype.names:
                if name == 'column':
                    new_cluster_hits['column'] = hits['column']
                elif name == 'row':
                    new_cluster_hits['row'] = hits['row']
                else:
                    new_cluster_hits[name] = cluster_hits[name]
            cluster_hits = new_cluster_hits
        return cluster_hits, clusters  # return hits array with additional columns for cluster ID, and cluster array

    smc.SMC(
        input_filename=input_hit_file,
        output_filename=output_cluster_file,
        table="Hits",
        func=cluster_func,
        func_kwargs={
            'clusterizer': clusterizer,
            'noisy_pixels': noisy_pixels,
            'disabled_pixels': disabled_pixels,
            'dut': dut,
            'convert_to_positions': convert_to_positions,
            'copy_hit_indices': copy_hit_indices},
        node_desc=[{'name': 'ClusterHits'}, {'name': 'Clusters'}],
        align_at='event_number',
        chunk_size=chunk_size)

    # Calculate cluster size histogram
    def cluster_size_hist_func(clusters):
        n_hits = clusters['n_hits']
        hist = analysis_utils.hist_1d_index(n_hits, shape=(np.max(n_hits) + 1,))
        return hist

    smc.SMC(
        input_filename=output_cluster_file,
        output_filename=output_cluster_file,
        table="Clusters",
        mode='r+',  # file must already exist
        func=cluster_size_hist_func,
        node_desc={'name': 'HistClusterSize'},
        chunk_size=chunk_size)

    # Calculate cluster shape histogram
    def cluster_shape_hist_func(clusters):
        cluster_shape = clusters['cluster_shape']
        hist, _ = np.histogram(a=cluster_shape, bins=np.arange(2**16))
        return hist

    smc.SMC(
        input_filename=output_cluster_file,
        output_filename=output_cluster_file,
        table="Clusters",
        mode='r+',  # file must already exist
        func=cluster_shape_hist_func,
        node_desc={'name': 'HistClusterShape'},
        chunk_size=chunk_size)

    def pos_error_func_with_index(clusters, dut, d, pxi, pyi):
        # Set errors for big clusters, where delta electrons reduce resolution
        # Set error on all clusters
        clusters['err_column'] = (clusters['err_column'] + 1) / math.sqrt(12)
        clusters['err_row'] = (clusters['err_row'] + 1) / math.sqrt(12)
        # TODO: do this not in all cases
        # if cluster_size_hist[4] > cluster_size_hist[4]:
        # Set errors for small clusters, where charge sharing enhances resolution
        # for 1x1, 1x2, 2x1, 2x2 clusters use specific formula
        # set different pixel shapes
        # TODO: check factor of 2
        sel = (clusters['cluster_shape'] == 1)
        clusters['err_column'][sel] = pxi / np.sqrt(12) / dut.column_size
        clusters['err_row'][sel] = pxi / np.sqrt(12) / dut.row_size
        sel = (clusters['cluster_shape'] == 3)
        clusters['err_column'][sel] = 2 * d / np.sqrt(12) / dut.column_size
        clusters['err_row'][sel] = pyi / np.sqrt(12) / dut.row_size
        sel = (clusters['cluster_shape'] == 5)
        clusters['err_column'][sel] = pxi / np.sqrt(12) / dut.column_size
        clusters['err_row'][sel] = 2 * d / np.sqrt(12) / dut.row_size
        sel = (clusters['cluster_shape'] == 7) | (clusters['cluster_shape'] == 11) | (clusters['cluster_shape'] == 13) | (clusters['cluster_shape'] == 14)
        clusters['err_column'][sel] = 2 * d * 2 / np.sqrt(12) / dut.column_size  # use 2 / sqrt(12) to compensate for the shape
        clusters['err_row'][sel] = 2 * d * 2 / np.sqrt(12) / dut.row_size  # use 2 / sqrt(12) to compensate for the shape
        sel = (clusters['cluster_shape'] == 15)
        clusters['err_column'][sel] = 2 * d / np.sqrt(12) / dut.column_size
        clusters['err_row'][sel] = 2 * d / np.sqrt(12) / dut.row_size
        return clusters

    def pos_error_func_with_position(clusters, dut):
        # Set errors for all clusters
        clusters['err_x'] = (clusters['err_x'] + dut.column_size) / math.sqrt(12)
        clusters['err_y'] = (clusters['err_y'] + dut.row_size) / math.sqrt(12)
        return clusters

    if use_positions:
        smc.SMC(
            input_filename=output_cluster_file,
            output_filename=output_cluster_file,
            table="Clusters",
            mode='r+',  # file must already exist
            func=pos_error_func_with_position,
            func_kwargs={
                'dut': dut},
            chunk_size=chunk_size)
    else:
        # Load infos from cluster size for error determination and plotting
        with tb.open_file(output_cluster_file, mode='r') as input_file_h5:
            # cluster_size_hist = input_file_h5.root.HistClusterSize[:]
            cluster_shape_hist = input_file_h5.root.HistClusterShape[:]

        # Calculate paramters from cluster shape histogram
        total_n_small_clusters = np.sum(cluster_shape_hist[[1, 3, 5, 7, 11, 13, 14, 15]])
        ratio_pxi_pyi = cluster_shape_hist[5] / cluster_shape_hist[3]
        ration_cs_1_all = cluster_shape_hist[1] / total_n_small_clusters
        pyi = math.sqrt(dut.column_size * dut.row_size * ration_cs_1_all / ratio_pxi_pyi)
        pxi = ratio_pxi_pyi * pyi
        d = (dut.column_size - pxi + dut.row_size - pyi) / 4

        smc.SMC(
            input_filename=output_cluster_file,
            output_filename=output_cluster_file,
            table="Clusters",
            mode='r+',  # file must already exist
            func=pos_error_func_with_index,
            func_kwargs={
                'dut': dut,
                'd': d,
                'pxi': pxi,
                'pyi': pyi},
            chunk_size=chunk_size)

    # Copy masks to result cluster file

    # Copy nodes to result file
    if input_mask_file is not None:
        with tb.open_file(input_mask_file, mode='r') as input_mask_file_h5:
            with tb.open_file(output_cluster_file, mode='r+') as output_file_h5:
                try:
                    input_mask_file_h5.root.DisabledPixelMask._f_copy(newparent=output_file_h5.root)
                except tb.NoSuchNodeError:
                    pass
                try:
                    input_mask_file_h5.root.NoisyPixelMask._f_copy(newparent=output_file_h5.root)
                except tb.NoSuchNodeError:
                    pass

    if plot:
        plot_cluster_hists(
            input_cluster_file=output_cluster_file,
            dut_name=dut.name,
            chunk_size=chunk_size)

    return output_cluster_file


def correlate(telescope_configuration, input_files, output_correlation_file=None, resolution=(100.0, 100.0), select_duts=None, select_reference_duts=0, plot=True, chunk_size=100000):
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
        Filename of the output correlation file.
    resolution : tuple
        Resolution of the correlation histogram in x and y direction (in um).
    select_reference_duts : uint, list
        DUT indices of the reference plane. Default is DUT 0. If None, generate correlation histograms of all DUT pairs.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_correlation_file : string
        Filename of the output correlation file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    if select_duts is None:
        select_duts = range(len(telescope))
    if not isinstance(select_duts, (list, tuple)):
        raise ValueError('Parameter "select_duts" is not a list.')
    if max(select_duts) > n_duts or min(select_duts) < 0:
        raise ValueError('Parameter "select_duts" contains ivalid values.')
    logging.info('=== Correlating %d DUTs ===', len(select_duts))

    if len(select_duts) != len(input_files):
        raise ValueError('Parameter "input_files" has wrong length.')

    if output_correlation_file is None:
        output_correlation_file = os.path.join(os.path.dirname(input_files[0]), 'Correlated.h5')

    if select_reference_duts is None:
        ref_duts = select_duts
    elif isinstance(select_reference_duts, (list, tuple)):
        ref_duts = select_reference_duts
    else:
        ref_duts = [select_reference_duts]  # make list

    with tb.open_file(output_correlation_file, mode="w") as out_file_h5:
        for ref_index in ref_duts:
            ref_file_index = np.where(ref_index == np.array(select_duts))[0][0]
            ref_dut = telescope[ref_index]
            # Result arrays to be filled
            x_correlations = []
            y_correlations = []
            start_indices = []
            ref_hist_x_extent = []
            ref_hist_y_extent = []
            dut_hist_x_extent = []
            dut_hist_y_extent = []

            # remove reference DUT from list of DUTs
            select_duts_removed = list(set(select_duts) - set([ref_index]))
            logging.info('== Correlating reference %s with %d other DUTs ==' % (ref_dut.name, len(select_duts_removed)))
            for dut_index in select_duts_removed:
                dut_file_index = np.where(dut_index == np.array(select_duts))[0][0]
                # Reference size
                ref_x_extent = telescope[ref_index].x_extent(global_position=True)
                ref_x_size = ref_x_extent[1] - ref_x_extent[0]
                ref_y_extent = telescope[ref_index].y_extent(global_position=True)
                ref_y_size = ref_y_extent[1] - ref_y_extent[0]
                # DUT size
                dut_x_extent = telescope[dut_index].x_extent(global_position=True)
                dut_x_size = dut_x_extent[1] - dut_x_extent[0]
                dut_y_extent = telescope[dut_index].y_extent(global_position=True)
                dut_y_size = dut_y_extent[1] - dut_y_extent[0]
                # Reference hist size
                ref_x_center = telescope[ref_index].translation_x
                ref_hist_x_size = math.ceil(ref_x_size / resolution[0]) * resolution[0]
                ref_hist_x_extent.append([ref_x_center - ref_hist_x_size / 2.0, ref_x_center + ref_hist_x_size / 2.0])
                ref_y_center = telescope[ref_index].translation_y
                ref_hist_y_size = math.ceil(ref_y_size / resolution[1]) * resolution[1]
                ref_hist_y_extent.append([ref_y_center - ref_hist_y_size / 2.0, ref_y_center + ref_hist_y_size / 2.0])
                # DUT hist size
                dut_x_center = telescope[dut_index].translation_x
                dut_hist_x_size = math.ceil(dut_x_size / resolution[0]) * resolution[0]
                dut_hist_x_extent.append([dut_x_center - dut_hist_x_size / 2.0, dut_x_center + dut_hist_x_size / 2.0])
                dut_y_center = telescope[dut_index].translation_y
                dut_hist_y_size = math.ceil(dut_y_size / resolution[1]) * resolution[1]
                dut_hist_y_extent.append([dut_y_center - dut_hist_y_size / 2.0, dut_y_center + dut_hist_y_size / 2.0])
                # Creating histograms for the correlation
                x_correlations.append(np.zeros([int(dut_hist_x_size / resolution[0]), int(ref_hist_x_size / resolution[0])], dtype=np.int32))
                y_correlations.append(np.zeros([int(dut_hist_y_size / resolution[1]), int(ref_hist_y_size / resolution[1])], dtype=np.int32))
                # Store the loop indices for speed up
                start_indices.append(None)

            with tb.open_file(input_files[ref_file_index], mode='r') as in_file_h5:  # Open reference hit/cluster file
                # Check for whether hits or clusters are used
                try:
                    ref_node = in_file_h5.root.Clusters
                    ref_use_clusters = True
                except tb.NoSuchNodeError:
                    ref_node = in_file_h5.root.Hits
                    ref_use_clusters = False
                # Check for whether local coordinates or indices are used
                if 'mean_x' in ref_node.dtype.names or 'x' in ref_node.dtype.names:
                    ref_use_positions = True
                else:
                    ref_use_positions = False

                widgets = ['', progressbar.Percentage(), ' ',
                           progressbar.Bar(marker='*', left='|', right='|'),
                           ' ', progressbar.AdaptiveETA()]
                progress_bar = progressbar.ProgressBar(widgets=widgets,
                                                       maxval=ref_node.shape[0],
                                                       term_width=80)
                progress_bar.start()

                pool = Pool()
                # Loop over the hits/clusters of reference DUT
                for ref_data_chunk, ref_read_index in analysis_utils.data_aligned_at_events(ref_node, chunk_size=chunk_size):
                    actual_event_numbers = ref_data_chunk['event_number']

                    # Create correlation histograms to the reference device for all other devices
                    dut_results = []
                    # Loop over other DUTs
                    for index, dut_index in enumerate(select_duts_removed):
                        dut_file_index = np.where(dut_index == np.array(select_duts))[0][0]
                        dut_results.append(pool.apply_async(_correlate_position, kwds={
                            'ref': telescope[ref_index],
                            'dut': telescope[dut_index],
                            'ref_use_clusters': ref_use_clusters,
                            'ref_use_positions': ref_use_positions,
                            'ref_data': ref_data_chunk,
                            'dut_input_file': input_files[dut_file_index],
                            'start_index': start_indices[index],
                            'start_event_number': actual_event_numbers[0],
                            'stop_event_number': actual_event_numbers[-1] + 1,
                            'resolution': resolution,
                            'ref_hist_x_extent': ref_hist_x_extent[index],
                            'ref_hist_y_extent': ref_hist_y_extent[index],
                            'dut_hist_x_extent': dut_hist_x_extent[index],
                            'dut_hist_y_extent': dut_hist_y_extent[index],
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
            for index, dut_index in enumerate(select_duts_removed):
                # x
                out_x = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='Correlation_x_%d_%d' % (ref_index, dut_index),
                    title='X correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                    atom=tb.Atom.from_dtype(x_correlations[index].dtype),
                    shape=x_correlations[index].shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_x.attrs.resolution = resolution[0]
                out_x.attrs.ref_hist_extent = ref_hist_x_extent[index]
                out_x.attrs.dut_hist_extent = dut_hist_x_extent[index]
                # y
                out_y = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='Correlation_y_%d_%d' % (ref_index, dut_index),
                    title='Y correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                    atom=tb.Atom.from_dtype(y_correlations[index].dtype),
                    shape=y_correlations[index].shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_y.attrs.resolution = resolution[1]
                out_y.attrs.ref_hist_extent = ref_hist_y_extent[index]
                out_y.attrs.dut_hist_extent = dut_hist_y_extent[index]

                # histograms with reduced background
                # x
                out_x_reduced = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='Correlation_x_%d_%d_reduced_background' % (ref_index, dut_index),
                    title='X correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                    atom=tb.Atom.from_dtype(x_correlations[index].dtype),
                    shape=x_correlations[index].shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_x_reduced.attrs.resolution = resolution[0]
                out_x_reduced.attrs.ref_hist_extent = ref_hist_x_extent[index]
                out_x_reduced.attrs.dut_hist_extent = dut_hist_x_extent[index]
                # y
                out_y_reduced = out_file_h5.create_carray(
                    where=out_file_h5.root,
                    name='Correlation_y_%d_%d_reduced_background' % (ref_index, dut_index),
                    title='Y correlation between DUT%d and DUT%d' % (ref_index, dut_index),
                    atom=tb.Atom.from_dtype(y_correlations[index].dtype),
                    shape=y_correlations[index].shape,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                out_y_reduced.attrs.resolution = resolution[1]
                out_y_reduced.attrs.ref_hist_extent = ref_hist_y_extent[index]
                out_y_reduced.attrs.dut_hist_extent = dut_hist_y_extent[index]
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
        plot_utils.plot_correlations(input_correlation_file=output_correlation_file, dut_names=telescope.dut_names)

    return output_correlation_file


def _correlate_position(ref, dut, ref_use_clusters, ref_use_positions, ref_data, dut_input_file, start_index, start_event_number, stop_event_number, resolution, ref_hist_x_extent, ref_hist_y_extent, dut_hist_x_extent, dut_hist_y_extent, x_correlation, y_correlation, chunk_size):
    # Open other DUT hit/cluster file
    with tb.open_file(dut_input_file, mode='r') as in_file_h5:
        # Check for whether hits or clusters are used
        try:
            dut_node = in_file_h5.root.Clusters
            dut_use_clusters = True
        except tb.NoSuchNodeError:
            dut_node = in_file_h5.root.Hits
            dut_use_clusters = False
        # Check for whether local coordinates or indices are used
        if 'mean_x' in dut_node.dtype.names or 'x' in dut_node.dtype.names:
            dut_use_positions = True
        else:
            dut_use_positions = False

        for dut_data, start_index in analysis_utils.data_aligned_at_events(dut_node, start_index=start_index, start_event_number=start_event_number, stop_event_number=stop_event_number, chunk_size=chunk_size, fail_on_missing_events=False):  # Loop over the cluster in the actual cluster file in chunks
            ref_event_number = ref_data['event_number']
            dut_event_number = dut_data['event_number']
            if ref_use_clusters:
                if ref_use_positions:
                    ref_x_pos, ref_y_pos, _ = ref.local_to_global_position(x=ref_data['mean_x'], y=ref_data['mean_y'], z=np.zeros_like(ref_data['mean_x']))
                else:
                    ref_x_pos, ref_y_pos, _ = ref.index_to_global_position(column=ref_data['mean_column'], row=ref_data['mean_row'])
            else:
                if ref_use_positions:
                    ref_x_pos, ref_y_pos, _ = ref.local_to_global_position(x=ref_data['x'], y=ref_data['y'], z=np.zeros_like(ref_data['x']))
                else:
                    ref_x_pos, ref_y_pos, _ = ref.index_to_global_position(column=ref_data['column'], row=ref_data['row'])
            if dut_use_clusters:
                if dut_use_positions:
                    dut_x_pos, dut_y_pos, _ = dut.local_to_global_position(x=dut_data['mean_x'], y=dut_data['mean_y'], z=np.zeros_like(dut_data['mean_x']))
                else:
                    dut_x_pos, dut_y_pos, _ = dut.index_to_global_position(column=dut_data['mean_column'], row=dut_data['mean_row'])
            else:
                if dut_use_positions:
                    dut_x_pos, dut_y_pos, _ = dut.local_to_global_position(x=dut_data['x'], y=dut_data['y'], z=np.zeros_like(dut_data['x']))
                else:
                    dut_x_pos, dut_y_pos, _ = dut.index_to_global_position(column=dut_data['column'], row=dut_data['row'])
            ref_x_indices = ((ref_x_pos - ref_hist_x_extent[0]) / resolution[0]).astype(np.uint32)
            ref_y_indices = ((ref_y_pos - ref_hist_y_extent[0]) / resolution[1]).astype(np.uint32)
            dut_x_indices = ((dut_x_pos - dut_hist_x_extent[0]) / resolution[0]).astype(np.uint32)
            dut_y_indices = ((dut_y_pos - dut_hist_y_extent[0]) / resolution[1]).astype(np.uint32)
            analysis_utils.correlate_position_on_event_number(
                ref_event_numbers=ref_event_number,
                dut_event_numbers=dut_event_number,
                ref_x_indices=ref_x_indices,
                ref_y_indices=ref_y_indices,
                dut_x_indices=dut_x_indices,
                dut_y_indices=dut_y_indices,
                x_corr_hist=x_correlation,
                y_corr_hist=y_correlation)

    return start_index, x_correlation, y_correlation


def merge_cluster_data(telescope_configuration, input_cluster_files, output_merged_file=None, chunk_size=1000000):
    '''Takes the cluster from all cluster files and merges them into one big table aligned at a common event number.

    Empty entries are signaled with column = row = charge = nan. Position is translated from indices to local position (um).
    The local coordinate system origin (0, 0, 0) is defined to be in the sensor center, decoupling translation and rotation.
    Cluster position errors are calculated from cluster dimensions.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_cluster_files : list
        Filename of the input cluster files with correlation data.
    output_merged_file : string
        Filename of the output tracklets file.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_merged_file : string
        Filename of the output tracklets file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Merge cluster files from %d DUTs ===', n_duts)

    if output_merged_file is None:
        output_merged_file = os.path.join(os.path.dirname(input_cluster_files[0]), 'Merged.h5')

    if len(input_cluster_files) != n_duts:
        raise ValueError('Parameter "input_cluster_files" has wrong length.')

    # Check for whether local coordinates or indices are used
    use_positions = []
    for input_file in input_cluster_files:
        with tb.open_file(input_file, mode='r') as input_file_h5:
            node = input_file_h5.root.Clusters
            if 'mean_x' in node.dtype.names:
                use_positions.append(True)
            else:
                use_positions.append(False)

    # Create result array description, depends on the number of DUTs
    description = [('event_number', np.int64)]
    for dimension in ['x', 'y', 'z']:
        for index_dut in range(n_duts):
            description.append(('%s_dut_%d' % (dimension, index_dut), np.float64))
    for index, _ in enumerate(input_cluster_files):
        description.append(('charge_dut_%d' % index, np.float32))
    for index, _ in enumerate(input_cluster_files):
        description.append(('frame_dut_%d' % index, np.uint8))
    for index, _ in enumerate(input_cluster_files):
        description.append(('n_hits_dut_%d' % index, np.uint32))
    for index, _ in enumerate(input_cluster_files):
        description.append(('cluster_ID_dut_%d' % index, np.int16))
    for index, _ in enumerate(input_cluster_files):
        description.append(('cluster_shape_dut_%d' % index, np.int64))
    for index, _ in enumerate(input_cluster_files):
        description.append(('n_cluster_dut_%d' % index, np.uint32))
    description.append(('hit_flag', np.uint32))
    for dimension in ['x', 'y', 'z']:
        for index_dut in range(n_duts):
            description.append(('%s_err_dut_%d' % (dimension, index_dut), np.float32))

    start_indices_merging_loop = [None] * len(input_cluster_files)  # Store the merging loop indices for speed up
    start_indices_data_loop = [None] * len(input_cluster_files)  # Additional store indices for the data loop
    actual_start_event_number = None  # Defines the first event number of the actual chunk for speed up. Cannot be deduced from DUT0, since this DUT could have missing event numbers.

    # Merge the cluster data from different DUTs into one table
    with tb.open_file(output_merged_file, mode='w') as out_file_h5:
        merged_cluster_table = out_file_h5.create_table(
            where=out_file_h5.root,
            name='MergedClusters',
            description=np.dtype(description),
            title='Merged cluster on event number',
            filters=tb.Filters(
                complib='blosc',
                complevel=5,
                fletcher32=False))
        with tb.open_file(input_cluster_files[0], mode='r') as in_file_h5:  # Open DUT0 cluster file
            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=in_file_h5.root.Clusters.shape[0], term_width=80)
            progress_bar.start()
            for actual_cluster_dut_0, start_indices_data_loop[0] in analysis_utils.data_aligned_at_events(in_file_h5.root.Clusters, start_index=start_indices_data_loop[0], start_event_number=actual_start_event_number, stop_event_number=None, chunk_size=chunk_size):  # Loop over the cluster of DUT0 in chunks
                actual_event_numbers = actual_cluster_dut_0['event_number']

                # First loop: calculate the minimum event number indices needed to merge all cluster from all files to this event number index
                common_event_numbers = actual_event_numbers
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open DUT0 cluster file
                        for actual_cluster, start_indices_merging_loop[dut_index] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Clusters, start_index=start_indices_merging_loop[dut_index], start_event_number=actual_start_event_number, stop_event_number=actual_event_numbers[-1] + 1, chunk_size=chunk_size, fail_on_missing_events=False):  # Loop over the cluster in the actual cluster file in chunks
                            common_event_numbers = analysis_utils.get_max_events_in_both_arrays(common_event_numbers, actual_cluster['event_number'])
                # array with merged DUT data
                merged_cluster_array = np.zeros(shape=(common_event_numbers.shape[0],), dtype=description)
                # set default values
                for index in range(n_duts):
                    # for no hit: column = row = charge = nan
                    merged_cluster_array['x_dut_%d' % (index)] = np.nan
                    merged_cluster_array['y_dut_%d' % (index)] = np.nan
                    merged_cluster_array['z_dut_%d' % (index)] = np.nan
                    merged_cluster_array['charge_dut_%d' % (index)] = np.nan
                    merged_cluster_array['x_err_dut_%d' % (index)] = np.nan
                    merged_cluster_array['y_err_dut_%d' % (index)] = np.nan
                    merged_cluster_array['z_err_dut_%d' % (index)] = np.nan

                mapped_clusters_dut_0 = np.empty(common_event_numbers.shape[0], dtype=actual_cluster_dut_0.dtype)
                if use_positions[0]:
                    mapped_clusters_dut_0['mean_x'] = np.nan
                else:
                    mapped_clusters_dut_0['mean_column'] = np.nan
                # Fill result array with DUT 0 data
                analysis_utils.map_cluster(
                    event_numbers=common_event_numbers,
                    clusters=actual_cluster_dut_0,
                    mapped_clusters=mapped_clusters_dut_0)
                # Set the event number
                merged_cluster_array['event_number'] = common_event_numbers[:]
                # Convert to local coordinates, origin is in the center of the sensor
                if use_positions[0]:
                    # Select real hits, values with nan are virtual hits
                    selection = ~np.isnan(mapped_clusters_dut_0['mean_x'])
                    merged_cluster_array['x_dut_0'][selection] = mapped_clusters_dut_0['mean_x'][selection]
                    merged_cluster_array['y_dut_0'][selection] = mapped_clusters_dut_0['mean_y'][selection]
                    merged_cluster_array['z_dut_0'][selection] = 0.0
                    merged_cluster_array['x_err_dut_0'][selection] = mapped_clusters_dut_0['err_x'][selection]
                    merged_cluster_array['y_err_dut_0'][selection] = mapped_clusters_dut_0['err_y'][selection]
                else:
                    # Select real hits, values with nan are virtual hits
                    selection = ~np.isnan(mapped_clusters_dut_0['mean_column'])
                    merged_cluster_array['x_dut_0'][selection], merged_cluster_array['y_dut_0'][selection], merged_cluster_array['z_dut_0'][selection] = telescope[0].index_to_local_position(
                        column=mapped_clusters_dut_0['mean_column'][selection],
                        row=mapped_clusters_dut_0['mean_row'][selection])
                    merged_cluster_array['x_err_dut_0'][selection] = mapped_clusters_dut_0['err_column'][selection] * telescope[0].column_size
                    merged_cluster_array['y_err_dut_0'][selection] = mapped_clusters_dut_0['err_row'][selection] * telescope[0].row_size
                merged_cluster_array['z_err_dut_0'][selection] = 0.0
                merged_cluster_array['charge_dut_0'][selection] = mapped_clusters_dut_0['charge'][selection]
                merged_cluster_array['frame_dut_0'][selection] = mapped_clusters_dut_0['frame'][selection]
                merged_cluster_array['n_hits_dut_0'][selection] = mapped_clusters_dut_0['n_hits'][selection]
                merged_cluster_array['cluster_ID_dut_0'][selection] = mapped_clusters_dut_0['cluster_ID'][selection]
                merged_cluster_array['cluster_shape_dut_0'][selection] = mapped_clusters_dut_0['cluster_shape'][selection]
                merged_cluster_array['n_cluster_dut_0'][selection] = mapped_clusters_dut_0['n_cluster'][selection]

                # Fill result array with other DUT data
                # Second loop: get the cluster from all files and merge them to the common event number
                for dut_index, cluster_file in enumerate(input_cluster_files[1:], start=1):  # Loop over the other cluster files
                    with tb.open_file(cluster_file, mode='r') as actual_in_file_h5:  # Open other DUT cluster file
                        for actual_cluster_dut, start_indices_data_loop[dut_index] in analysis_utils.data_aligned_at_events(actual_in_file_h5.root.Clusters, start_index=start_indices_data_loop[dut_index], start_event_number=common_event_numbers[0], stop_event_number=common_event_numbers[-1] + 1, chunk_size=chunk_size, fail_on_missing_events=False):  # Loop over the cluster in the actual cluster file in chunks
                            mapped_clusters_dut = np.empty(common_event_numbers.shape[0], dtype=actual_cluster_dut.dtype)
                            if use_positions[dut_index]:
                                mapped_clusters_dut['mean_x'] = np.nan
                            else:
                                mapped_clusters_dut['mean_column'] = np.nan
                            analysis_utils.map_cluster(
                                event_numbers=common_event_numbers,
                                clusters=actual_cluster_dut,
                                mapped_clusters=mapped_clusters_dut)
                            # Convert to local coordinates, origin is in the center of the sensor
                            if use_positions[dut_index]:
                                # Select real hits, values with nan are virtual hits
                                selection = ~np.isnan(mapped_clusters_dut['mean_x'])
                                merged_cluster_array['x_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['mean_x'][selection]
                                merged_cluster_array['y_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['mean_y'][selection]
                                merged_cluster_array['z_dut_%d' % (dut_index)][selection] = 0.0
                                merged_cluster_array['x_err_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['err_x'][selection]
                                merged_cluster_array['y_err_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['err_y'][selection]
                            else:
                                # Select real hits, values with nan are virtual hits
                                selection = ~np.isnan(mapped_clusters_dut['mean_column'])
                                merged_cluster_array['x_dut_%d' % (dut_index)][selection], merged_cluster_array['y_dut_%d' % (dut_index)][selection], merged_cluster_array['z_dut_%d' % (dut_index)][selection] = telescope[dut_index].index_to_local_position(
                                    column=mapped_clusters_dut['mean_column'][selection],
                                    row=mapped_clusters_dut['mean_row'][selection])
                                merged_cluster_array['x_err_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['err_column'][selection] * telescope[dut_index].column_size
                                merged_cluster_array['y_err_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['err_row'][selection] * telescope[dut_index].row_size
                            merged_cluster_array['z_err_dut_%d' % (dut_index)][selection] = 0.0
                            merged_cluster_array['charge_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['charge'][selection]
                            merged_cluster_array['frame_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['frame'][selection]
                            merged_cluster_array['n_hits_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['n_hits'][selection]
                            merged_cluster_array['cluster_ID_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['cluster_ID'][selection]
                            merged_cluster_array['cluster_shape_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['cluster_shape'][selection]
                            merged_cluster_array['n_cluster_dut_%d' % (dut_index)][selection] = mapped_clusters_dut['n_cluster'][selection]
                # calculate hit flags
                for dut_index in range(n_duts):
                    merged_cluster_array['hit_flag'] += np.isfinite(merged_cluster_array['x_dut_%d' % dut_index]).astype(merged_cluster_array['hit_flag'].dtype) << dut_index
                # append to table
                merged_cluster_table.append(merged_cluster_array)
                merged_cluster_table.flush()
                actual_start_event_number = common_event_numbers[-1] + 1  # Set the starting event number for the next chunked read
                progress_bar.update(start_indices_data_loop[0])
            progress_bar.finish()

    return output_merged_file


if __name__ == '__main__':
    pass
