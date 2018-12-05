"""This class provides often needed analysis functions, for analysis that is done with python.
"""
from __future__ import division

import logging
import os
import errno

import numpy as np
import numexpr as ne
import numba
from numba import njit
from scipy.interpolate import splrep, sproot
from scipy import stats
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.sparse import csr_matrix
from scipy.spatial import Voronoi
from scipy.special import erf
from scipy.ndimage import distance_transform_edt

import requests
import progressbar

from testbeam_analysis import analysis_functions

# A public secret representing public, read only owncloud folder
SCIBO_PUBLIC_FOLDER = 'NzfAx2zAQll5YXB'


@njit
def merge_on_event_number(data_1, data_2):
    """
    Merges the data_2 with data_1 on an event basis with all permutations
    That means: merge all hits of every event in data_2 on all hits of the same event in data_1.

    Does the same than the merge of the pandas package:
        df = data_1.merge(data_2, how='left', on='event_number')
        df.dropna(inplace=True)
    But results in 4 x faster code.

    Parameter
    --------

    data_1, data_2: np.recarray with event_number column

    Returns
    -------

    Tuple np.recarray, np.recarray
        Is the data_1, data_2 array extended by the permutations.

    """
    result_array_size = 0
    event_index_data_2 = 0

    # Loop to determine the needed result array size
    for index_data_1 in range(data_1.shape[0]):

        while event_index_data_2 < data_2.shape[0] and data_2[event_index_data_2]['event_number'] < data_1[index_data_1]['event_number']:
            event_index_data_2 += 1

        for index_data_2 in range(event_index_data_2, data_2.shape[0]):
            if data_1[index_data_1]['event_number'] == data_2[index_data_2]['event_number']:
                result_array_size += 1
            else:
                break

    # Create result array with correct size
    result_1 = np.zeros(shape=(result_array_size,), dtype=data_1.dtype)
    result_2 = np.zeros(shape=(result_array_size,), dtype=data_2.dtype)

    result_index_1 = 0
    result_index_2 = 0
    event_index_data_2 = 0

    for index_data_1 in range(data_1.shape[0]):

        while event_index_data_2 < data_2.shape[0] and data_2[event_index_data_2]['event_number'] < data_1[index_data_1]['event_number']:  # Catch up with outer loop
            event_index_data_2 += 1

        for index_data_2 in range(event_index_data_2, data_2.shape[0]):
            if data_1[index_data_1]['event_number'] == data_2[index_data_2]['event_number']:
                result_1[result_index_1] = data_1[index_data_1]
                result_2[result_index_2] = data_2[index_data_2]
                result_index_1 += 1
                result_index_2 += 1
            else:
                break

    return result_1, result_2


@njit
def correlate_position_on_event_number(ref_event_numbers, dut_event_numbers, ref_x_indices, ref_y_indices, dut_x_indices, dut_y_indices, x_corr_hist, y_corr_hist):
    """Correlating the hit/cluster positions on event basis including all permutations.
    The hit/cluster positions are used to fill the X and Y correlation histograms.

    Does the same than the merge of the pandas package:
        df = data_1.merge(data_2, how='left', on='event_number')
        df.dropna(inplace=True)
        correlation_column = np.hist2d(df[column_mean_dut_0], df[column_mean_dut_x])
        correlation_row = np.hist2d(df[row_mean_dut_0], df[row_mean_dut_x])
    The following code is > 10x faster than the above code.

    Parameters
    ----------
    ref_event_numbers: array
        Event number array of the reference DUT.
    dut_event_numbers: array
        Event number array of the second DUT.
    ref_x_indices: array
        X position indices of the refernce DUT.
    ref_y_indices: array
        Y position indices of the refernce DUT.
    dut_x_indices: array
        X position indices of the second DUT.
    dut_y_indices: array
        Y position indices of the second DUT.
    x_corr_hist: array
        X correlation array (2D).
    y_corr_hist: array
        Y correlation array (2D).
    """
    dut_index = 0

    # Loop to determine the needed result array size.astype(np.uint32)
    for ref_index in range(ref_event_numbers.shape[0]):

        while dut_index < dut_event_numbers.shape[0] and dut_event_numbers[dut_index] < ref_event_numbers[ref_index]:  # Catch up with outer loop
            dut_index += 1

        for curr_dut_index in range(dut_index, dut_event_numbers.shape[0]):
            if ref_event_numbers[ref_index] == dut_event_numbers[curr_dut_index]:
                x_index_ref = ref_x_indices[ref_index]
                y_index_ref = ref_y_indices[ref_index]
                x_index_dut = dut_x_indices[curr_dut_index]
                y_index_dut = dut_y_indices[curr_dut_index]

                # Add correlation to histogram
                x_corr_hist[x_index_dut, x_index_ref] += 1
                y_corr_hist[y_index_dut, y_index_ref] += 1
            else:
                break


@njit(locals={'curr_event_number': numba.int64, 'last_event_number': numba.int64, 'curr_index': numba.int64, 'corr_index': numba.int64})
def correlate_hits_on_event_range(event_numbers, x_indices, y_indices, x_corr_hist, y_corr_hist, event_range):
    """Correlating the hit indices of different events in a certain range.
    For unambiguous event building no correlation should be seen.


    Parameters
    ----------
    event_numbers: array
        Event number array.
    x_indices: array
        X position indices.
    y_indices: array
        Y position indices.
    x_corr_hist: array
        X correlation array (2D).
    y_corr_hist: array
        Y correlation array (2D).
    event_range : uint
        The number of events to use for correlation,
        e.g., event_range = 1 correlates to predecessing event hits with the current event hits.
    """
    last_event_number = -1
    # Loop over hits, outer loop
    for curr_index in range(event_numbers.shape[0]):
        curr_event_number = event_numbers[curr_index]
        # calculate new start index for inner loop if new event occurs
        if curr_event_number != last_event_number:
            corr_start_event_number = curr_event_number - event_range
            corr_start_index = np.searchsorted(event_numbers, corr_start_event_number)
        # set correlation index
        corr_index = corr_start_index
        # Iterate until current event number
        while event_numbers[corr_index] < curr_event_number:
            x_corr_hist[x_indices[corr_index], x_indices[curr_index]] += 1
            y_corr_hist[y_indices[corr_index], y_indices[curr_index]] += 1
            corr_index += 1
        last_event_number = curr_event_number


def in1d_events(ar1, ar2):
    """
    Does the same than np.in1d but uses the fact that ar1 and ar2 are sorted and the c++ library. Is therefore much much faster.

    """
    ar1 = np.ascontiguousarray(ar1)  # change memory alignement for c++ library
    ar2 = np.ascontiguousarray(ar2)  # change memory alignement for c++ library
    tmp = np.empty_like(ar1, dtype=np.uint8)  # temporary result array filled by c++ library, bool type is not supported with cython/numpy
    return analysis_functions.get_in1d_sorted(ar1, ar2, tmp)


def hist_quantiles(hist, prob=(0.05, 0.95), return_indices=False, copy=True):
    '''Calculate quantiles from histograms, cuts off hist below and above given quantile. This function will not cut off more than the given values.

    Parameters
    ----------
    hist : array_like, iterable
        Input histogram with dimension at most 1.
    prob : float, list, tuple
        List of quantiles to compute. Upper and lower limit. From 0 to 1. Default is 0.05 and 0.95.
    return_indices : bool, optional
        If true, return the indices of the hist.
    copy : bool, optional
        Whether to copy the input data (True), or to use a reference instead. Default is True.

    Returns
    -------
    masked_hist : masked_array
       Hist with masked elements.
    masked_hist : masked_array, tuple
        Hist with masked elements and indices.
    '''
    # make np array
    hist_t = np.array(hist)
    # calculate cumulative distribution
    cdf = np.cumsum(hist_t)
    # copy, convert and normalize
    if cdf[-1] == 0:
        normcdf = cdf.astype('float')
    else:
        normcdf = cdf.astype('float') / cdf[-1]
    # calculate unique values from cumulative distribution and their indices
    unormcdf, indices = np.unique(normcdf, return_index=True)
    # calculate limits
    try:
        hp = np.where(unormcdf > prob[1])[0][0]
        lp = np.where(unormcdf >= prob[0])[0][0]
    except IndexError:
        hp_index = hist_t.shape[0]
        lp_index = 0
    else:
        hp_index = indices[hp]
        lp_index = indices[lp]
    # copy and create ma
    masked_hist = np.ma.array(hist, copy=copy, mask=True)
    masked_hist.mask[lp_index:hp_index + 1] = False
    if return_indices:
        return masked_hist, (lp_index, hp_index)
    else:
        return masked_hist


def get_max_events_in_both_arrays(events_one, events_two):
    """
    Calculates the maximum count of events that exist in both arrays.

    """
    events_one = np.ascontiguousarray(events_one)  # change memory alignement for c++ library
    events_two = np.ascontiguousarray(events_two)  # change memory alignement for c++ library
    event_result = np.empty(shape=(events_one.shape[0] + events_two.shape[0],), dtype=events_one.dtype)
    count = analysis_functions.get_max_events_in_both_arrays(events_one, events_two, event_result)
    return event_result[:count]


@njit()
def map_cluster(event_numbers, clusters, mapped_clusters):
    '''
    Maps the cluster hits on events. Not existing cluster in events have all values set to 0 and column/row/charge set to nan.
    Too many cluster per event for the event number are omitted and lost!

    Parameters
    ----------
    event_numbers : numpy array
        One dimensional event number array with increasing event numbers.
    clusters : np.recarray
        Recarray with cluster info. The event number is increasing.
    mapped_clusters : np.recarray
        Recarray of the same length as event_numbers and same dtype as clusters with values initialized to NaN/0.

    Example
    -------
    event_numbers = [ 0  1  1  2  3  3 ]
    clusters.event_number = [ 0  1  2  2  3  4 ]

    gives mapped_clusters.event_number = [ 0  1  0  2  3  0 ]
    '''
    i = 0
    j = 0
    while i < event_numbers.shape[0]:
        # Find first Hit with a fitting event number
        while j < clusters.shape[0] and clusters['event_number'][j] < event_numbers[i]:  # Catch up to actual event number events[i]
            j += 1

        if j < clusters.shape[0]:
            if clusters['event_number'][j] == event_numbers[i]:
                mapped_clusters[i] = clusters[j]
                j += 1
        else:
            return
        i += 1


def get_events_in_both_arrays(events_one, events_two):
    """
    Calculates the events that exist in both arrays.

    """
    events_one = np.ascontiguousarray(events_one)  # change memory alignement for c++ library
    events_two = np.ascontiguousarray(events_two)  # change memory alignement for c++ library
    event_result = np.empty_like(events_one)
    count = analysis_functions.get_events_in_both_arrays(events_one, events_two, event_result)
    return event_result[:count]


def hist_1d_index(x, shape):
    """
    Fast 1d histogram of 1D indices with C++ inner loop optimization.
    Is more than 2 orders faster than np.histogram().
    The indices are given in coordinates and have to fit into a histogram of the dimensions shape.

    Parameters
    ----------
    x : array like
    shape : tuple
        tuple with x dimensions: (x,)

    Returns
    -------
    np.ndarray with given shape

    """
    if len(shape) != 1:
        raise NotImplementedError('The shape has to describe a 1-d histogram')

    # change memory alignment for c++ library
    x = np.ascontiguousarray(x.astype(np.int32))
    result = np.zeros(shape=shape, dtype=np.uint32)
    analysis_functions.hist_1d(x, shape[0], result)
    return result


def hist_2d_index(x, y, shape):
    """
    Fast 2d histogram of 2D indices with C++ inner loop optimization.
    Is more than 2 orders faster than np.histogram2d().
    The indices are given in x, y coordinates and have to fit into a histogram of the dimensions shape.
    Parameters
    ----------
    x : array like
    y : array like
    shape : tuple
        tuple with x,y dimensions: (x, y)

    Returns
    -------
    np.ndarray with given shape

    """
    if len(shape) != 2:
        raise NotImplementedError('The shape has to describe a 2-d histogram')

    if x.shape != y.shape:
        raise ValueError('The dimensions in x / y have to match')

    # change memory alignment for c++ library
    x = np.ascontiguousarray(x.astype(np.int32))
    y = np.ascontiguousarray(y.astype(np.int32))
    result = np.zeros(shape=shape, dtype=np.uint32).ravel()  # ravel hist in c-style, 3D --> 1D
    analysis_functions.hist_2d(x, y, shape[0], shape[1], result)
    return np.reshape(result, shape)  # rebuilt 3D hist from 1D hist


def hist_3d_index(x, y, z, shape):
    """
    Fast 3d histogram of 3D indices with C++ inner loop optimization.
    Is more than 2 orders faster than np.histogramdd().
    The indices are given in x, y, z coordinates and have to fit into a histogram of the dimensions shape.
    Parameters
    ----------
    x : array like
    y : array like
    z : array like
    shape : tuple
        tuple with x,y,z dimensions: (x, y, z)

    Returns
    -------
    np.ndarray with given shape

    """
    if len(shape) != 3:
        raise NotImplementedError('The shape has to describe a 3-d histogram')

    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError('The dimensions in x / y / z have to match')

    # change memory alignment for c++ library
    x = np.ascontiguousarray(x.astype(np.int32))
    y = np.ascontiguousarray(y.astype(np.int32))
    z = np.ascontiguousarray(z.astype(np.int32))
    result = np.zeros(shape=shape, dtype=np.uint16).ravel()  # ravel hist in c-style, 3D --> 1D
    analysis_functions.hist_3d(x, y, z, shape[0], shape[1], shape[2], result)
    return np.reshape(result, shape)  # rebuilt 3D hist from 1D hist


def get_data_in_event_range(array, event_start=None, event_stop=None, assume_sorted=True):
    '''Selects the data (rows of a table) that occurred in the given event range [event_start, event_stop[

    Parameters
    ----------
    array : numpy.array
    event_start : int, None
    event_stop : int, None
    assume_sorted : bool
        Set to true if the hits are sorted by the event_number. Increases speed.

    Returns
    -------
    numpy.array
        hit array with the hits in the event range.
    '''
    event_number = array['event_number']
    if not np.any(event_number):  # No events in selection
        return np.array([])
    if assume_sorted:
        data_event_start = event_number[0]
        data_event_stop = event_number[-1]
        if (event_start is not None and event_stop is not None) and (data_event_stop < event_start or data_event_start > event_stop or event_start == event_stop):  # special case, no intersection at all
            return array[0:0]

        # get min/max indices with values that are also in the other array
        if event_start is None:
            min_index_data = 0
        else:
            if event_number[0] > event_start:
                min_index_data = 0
            else:
                min_index_data = np.argmin(event_number < event_start)

        if event_stop is None:
            max_index_data = event_number.shape[0]
        else:
            if event_number[-1] < event_stop:
                max_index_data = event_number.shape[0]
            else:
                max_index_data = np.argmax(event_number >= event_stop)

        if min_index_data < 0:
            min_index_data = 0
        return array[min_index_data:max_index_data]
    else:
        return array[ne.evaluate('(event_number >= event_start) & (event_number < event_stop)')]


def data_aligned_at_events(table, start_event_number=None, stop_event_number=None, start_index=None, stop_index=None, chunk_size=1000000, try_speedup=False, first_event_aligned=True, fail_on_missing_events=True):
    '''Takes the table with a event_number column and returns chunks with the size up to chunk_size. The chunks are chosen in a way that the events are not splitted.
    Additional parameters can be set to increase the readout speed. Events between a certain range can be selected.
    Also the start and the stop indices limiting the table size can be specified to improve performance.
    The event_number column must be sorted.
    In case of try_speedup is True, it is important to create an index of event_number column with pytables before using this function. Otherwise the queries are slowed down.

    Parameters
    ----------
    table : pytables.table
        The data.
    start_event_number : int
        The retruned data contains events with event number >= start_event_number. If None, no limit is set.
    stop_event_number : int
        The retruned data contains events with event number < stop_event_number. If None, no limit is set.
    start_index : int
        Start index of data. If None, no limit is set.
    stop_index : int
        Stop index of data. If None, no limit is set.
    chunk_size : int
        Maximum chunk size per read.
    try_speedup : bool
        If True, try to reduce the index range to read by searching for the indices of start and stop event number. If these event numbers are usually
        not in the data this speedup can even slow down the function!

    The following parameters are not used when try_speedup is True:

    first_event_aligned : bool
        If True, assuming that the first event is aligned to the data chunk and will be added. If False, the lowest event number of the first chunk will not be read out.
    fail_on_missing_events : bool
        If True, an error is given when start_event_number or stop_event_number is not part of the data.

    Returns
    -------
    Iterator of tuples
        Data of the actual data chunk and start index for the next chunk.

    Example
    -------
    start_index = 0
    for scan_parameter in scan_parameter_range:
        start_event_number, stop_event_number = event_select_function(scan_parameter)
        for data, start_index in data_aligned_at_events(table, start_event_number=start_event_number, stop_event_number=stop_event_number, start_index=start_index):
            do_something(data)

    for data, index in data_aligned_at_events(table):
        do_something(data)
    '''
    # initialize variables
    start_index_known = False
    stop_index_known = False
    start_index = 0 if start_index is None else start_index
    stop_index = table.nrows if stop_index is None else stop_index
    if stop_index < start_index:
        raise ValueError('Invalid start/stop index')
    table_max_rows = table.nrows
    if stop_event_number is not None and start_event_number is not None and stop_event_number < start_event_number:
        raise ValueError('Invalid start/stop event number')

    # set start stop indices from the event numbers for fast read if possible; not possible if the given event number does not exist in the data stream
    if try_speedup and table.colindexed["event_number"]:
        if start_event_number is not None:
            start_condition = 'event_number==' + str(start_event_number)
            start_indices = table.get_where_list(start_condition, start=start_index, stop=stop_index)
            if start_indices.shape[0] != 0:  # set start index if possible
                start_index = start_indices[0]
                start_index_known = True

        if stop_event_number is not None:
            stop_condition = 'event_number==' + str(stop_event_number)
            stop_indices = table.get_where_list(stop_condition, start=start_index, stop=stop_index)
            if stop_indices.shape[0] != 0:  # set the stop index if possible, stop index is excluded
                stop_index = stop_indices[0]
                stop_index_known = True

    if start_index_known and stop_index_known and start_index + chunk_size >= stop_index:  # special case, one read is enough, data not bigger than one chunk and the indices are known
        yield table.read(start=start_index, stop=stop_index), stop_index
    else:  # read data in chunks, chunks do not divide events, abort if stop_event_number is reached

        # search for begin
        current_start_index = start_index
        if start_event_number is not None:
            while current_start_index < stop_index:
                current_stop_index = min(current_start_index + chunk_size, stop_index)
                array_chunk = table.read(start=current_start_index, stop=current_stop_index)  # stop index is exclusive, so add 1
                last_event_in_chunk = array_chunk["event_number"][-1]

                if last_event_in_chunk < start_event_number:
                    current_start_index = current_start_index + chunk_size  # not there yet, continue to next read (assuming sorted events)
                else:
                    first_event_in_chunk = array_chunk["event_number"][0]
#                     if stop_event_number is not None and first_event_in_chunk >= stop_event_number and start_index != 0 and start_index == current_start_index:
#                         raise ValueError('The stop event %d is missing. Change stop_event_number.' % stop_event_number)
                    if array_chunk.shape[0] == chunk_size and first_event_in_chunk == last_event_in_chunk:
                        raise ValueError('Chunk size too small. Increase chunk size to fit full event.')

                    if not first_event_aligned and first_event_in_chunk == start_event_number and start_index != 0 and start_index == current_start_index:  # first event in first chunk not aligned at index 0, so take next event
                        if fail_on_missing_events:
                            raise ValueError('The start event %d is missing. Change start_event_number.' % start_event_number)
                        chunk_start_index = np.searchsorted(array_chunk["event_number"], start_event_number + 1, side='left')
                    elif fail_on_missing_events and first_event_in_chunk > start_event_number and start_index == current_start_index:
                        raise ValueError('The start event %d is missing. Change start_event_number.' % start_event_number)
                    elif first_event_aligned and first_event_in_chunk == start_event_number and start_index == current_start_index:
                        chunk_start_index = 0
                    else:
                        chunk_start_index = np.searchsorted(array_chunk["event_number"], start_event_number, side='left')
                        if fail_on_missing_events and array_chunk["event_number"][chunk_start_index] != start_event_number and start_index == current_start_index:
                            raise ValueError('The start event %d is missing. Change start_event_number.' % start_event_number)
#                     if fail_on_missing_events and ((start_index == current_start_index and chunk_start_index == 0 and start_index != 0 and not first_event_aligned) or array_chunk["event_number"][chunk_start_index] != start_event_number):
#                         raise ValueError('The start event %d is missing. Change start_event_number.' % start_event_number)
                    current_start_index = current_start_index + chunk_start_index  # calculate index for next loop
                    break
        elif not first_event_aligned and start_index != 0:
            while current_start_index < stop_index:
                current_stop_index = min(current_start_index + chunk_size, stop_index)
                array_chunk = table.read(start=current_start_index, stop=current_stop_index)  # stop index is exclusive, so add 1
                first_event_in_chunk = array_chunk["event_number"][0]
                last_event_in_chunk = array_chunk["event_number"][-1]

                if array_chunk.shape[0] == chunk_size and first_event_in_chunk == last_event_in_chunk:
                    raise ValueError('Chunk size too small. Increase chunk size to fit full event.')

                chunk_start_index = np.searchsorted(array_chunk["event_number"], first_event_in_chunk + 1, side='left')
                current_start_index = current_start_index + chunk_start_index
                if not first_event_in_chunk == last_event_in_chunk:
                    break

        # data loop
        while current_start_index < stop_index:
            current_stop_index = min(current_start_index + chunk_size, stop_index)
            array_chunk = table.read(start=current_start_index, stop=current_stop_index)  # stop index is exclusive, so add 1
            first_event_in_chunk = array_chunk["event_number"][0]
            last_event_in_chunk = array_chunk["event_number"][-1]

            chunk_start_index = 0

            if stop_event_number is None:
                if current_stop_index == table_max_rows:
                    chunk_stop_index = array_chunk.shape[0]
                else:
                    chunk_stop_index = np.searchsorted(array_chunk["event_number"], last_event_in_chunk, side='left')
            else:
                if last_event_in_chunk >= stop_event_number:
                    chunk_stop_index = np.searchsorted(array_chunk["event_number"], stop_event_number, side='left')
                elif current_stop_index == table_max_rows:  # this will also add the last event of the table
                    chunk_stop_index = array_chunk.shape[0]
                else:
                    chunk_stop_index = np.searchsorted(array_chunk["event_number"], last_event_in_chunk, side='left')

            nrows = chunk_stop_index - chunk_start_index
            if nrows == 0:
                if array_chunk.shape[0] == chunk_size and first_event_in_chunk == last_event_in_chunk:
                    raise ValueError('Chunk size too small to fit event. Data corruption possible. Increase chunk size to read full event.')
                elif chunk_start_index == 0:  # not increasing current_start_index
                    return
                elif stop_event_number is not None and last_event_in_chunk >= stop_event_number:
                    return
            else:
                yield array_chunk[chunk_start_index:chunk_stop_index], current_start_index + nrows + chunk_start_index

            current_start_index = current_start_index + nrows + chunk_start_index  # events fully read, increase start index and continue reading


def find_closest(arr, values):
    '''Returns a list of indices with values closest to arr values.

    Parameters
    ----------
    arr : iterable
        Iterable of numbers. Arr must be sorted.
    values : iterable
        Iterable of numbers.

    Returns
    -------
    A list of indices with values closest to arr values.

    See also: http://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
    '''
    idx = arr.searchsorted(values)
    idx = np.clip(idx, 1, len(arr) - 1)
    left = arr[idx - 1]
    right = arr[idx]
    idx -= values - left < right - values
    return idx


def linear(x, c0, c1):
    return c0 + c1 * x


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2.0 / (2.0 * sigma ** 2.0))


def gauss2(x, *p):
    mu, sigma = p
    return (sigma * np.sqrt(2.0 * np.pi))**-1.0 * np.exp(-0.5 * ((x - mu) / sigma)**2.0)


def gauss_offset_slope(x, *p):
    A, mu, sigma, offset, slope = p
    return gauss(x, A, mu, sigma) + offset + x * slope


def gauss_offset(x, *p):
    A, mu, sigma, offset = p
    return gauss(x, A, mu, sigma) + offset


def double_gauss(x, *p):
    A_1, mu_1, sigma_1, A_2, mu_2, sigma_2 = p
    return gauss(x, A_1, mu_1, sigma_1) + gauss(x, A_2, mu_2, sigma_2)


def double_gauss_offset(x, *p):
    A_1, mu_1, sigma_1, A_2, mu_2, sigma_2, offset = p
    return gauss(x, A_1, mu_1, sigma_1) + gauss(x, A_2, mu_2, sigma_2) + offset


def gauss_box_non_vec(x, *p):
    ''''Convolution of gaussian and rectangle is a gaussian integral.

    Parameters
    ----------
    A, mu, sigma, a (width of the rectangle) : float

    See also:
    - http://stackoverflow.com/questions/24230233/fit-gaussian-integral-function-to-data
    - https://stackoverflow.com/questions/24386931/how-to-convolve-two-distirbutions-from-scipy-library
    '''
    A, mu, sigma, a = p
    return quad(lambda t: gauss(x - t, A, mu, sigma) / (np.sqrt(2.0 * np.pi) * sigma), -a / 2.0, a / 2.0)[0]


# Vetorize function to use with np.arrays
gauss_box = np.vectorize(gauss_box_non_vec, excluded=["*p"])


def gauss_box_erf(x, *p):
    ''' Identical to gauss_box().
    '''
    A, mu, sigma, width = p
    return 0.5 * A * erf((x - mu + width * 0.5) / (np.sqrt(2) * sigma)) + 0.5 * A * erf((mu + width * 0.5 - x) / (np.sqrt(2) * sigma))


def get_chi2(y_data, y_fit):
    return np.square(y_data - y_fit).sum()


def get_mean_from_histogram(counts, bin_positions):
    return np.dot(counts, np.array(bin_positions)) / np.sum(counts).astype('f4')


def get_rms_from_histogram(counts, bin_positions):
    return np.std(np.repeat(bin_positions, counts))


def get_median_from_histogram(counts, bin_positions):
    return np.median(np.repeat(bin_positions, counts))


def get_mean_efficiency(array_pass, array_total, interval=0.68):
    ''' Calculates the mean efficiency with statistical errors

        Parameters
        ----------
        array_pass, array_total : numpy array
        interval: float
            Confidence interval for error calculation

        Returns
        -------
        Tuple with: Mean efficiency and positive negative confidence interval limits
    '''

    def get_eff_pdf(eff, k, N):
        ''' Returns the propability density function for the efficiency
            estimator eff = k/N, where k are the passing events and N the
            total number of events.

            http://lss.fnal.gov/archive/test-tm/2000/fermilab-tm-2286-cd.pdf
            page 5

            This function gives plot 1 of paper, when multiplied by Gamma(N+1)
        '''

        # The paper has the function defined by gamma functions. These explode quickly
        # leading to numerical instabilities. The beta function does the same...
        # np.float(gamma(N + 2)) / np.float((gamma(k + 1) * gamma(N - k + 1))) * eff**k * (1. - eff)**(N - k)

        return stats.beta.pdf(eff, k + 1, N - k + 1)

    def get_eff_prop_int(eff, k, N):
        ''' C.d.f. of beta function = P.d.f. integral -infty..eff '''
        return stats.beta.cdf(eff, k + 1, N - k + 1)

    def interval_integ(a, b, k, N):
        ''' Return the integral of the efficiency pdf using limits [a, b]:
        '''
        return get_eff_prop_int(b, k, N) - get_eff_prop_int(a, k, N)

    def find_inter(k, N, interval):
        ''' Calculates Integral(pdf(k, N))_err-^err+ = interval with
        | err- - err+| != min.
        '''

        def minimizeMe(x):
            a, b = x[0], x[1]
            return b - a

        def get_start_values(k, N):
            # Discretize issue for numerical calculation
            eff = np.linspace(0.8 * float(k) / N, 1.2 * float(k) / N, 1000000)
            eff = eff[eff <= 1.]
            # Normalize by discretization
            p = get_eff_pdf(eff, k, N=N)
            max_i = np.argmin(np.abs(eff - float(k) / N))

            for y in np.linspace(p[max_i] * 0.9, 0, 1000):
                if max_i > 0:
                    idx_l = np.abs(y - p[:max_i]).argmin()
                else:
                    idx_l = 0

                if max_i < p.shape[0]:
                    idx_r = np.abs(y - p[max_i:]).argmin() + max_i
                else:
                    idx_r = p.shape[0] - 1

                if p[idx_l:idx_r].sum() * np.diff(eff)[0] > interval:
                    break
            return eff[idx_l], eff[idx_r]

        # Quick determination of start value to enhance convergence
        max_a = float(k) / N  # a < maximum eff
        min_b = float(k) / N  # b > maximum eff

        a0, b0 = get_start_values(k, N)

        cons = ({'type': 'eq', 'fun': lambda x: np.abs(interval_integ(x[0], x[1], k, N) - interval)})

        # Find b
        res = optimize.minimize(fun=minimizeMe, method='SLSQP', x0=(a0, b0),
                                bounds=((0., max_a), (min_b, 1.)),
                                constraints=cons)

        return res.x

    k = array_pass.sum()
    N = array_total.sum()
    eff = k.astype(np.float32) / N

    lim_e_m, lim_e_p = find_inter(k, N, interval)

    return eff, lim_e_m - eff, lim_e_p - eff


def fwhm(x, y):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset. The function
    uses a spline interpolation of order 3.

    See also http://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
    """
    half_max = np.max(y) / 2.0
    spl = splrep(x, y - half_max)
    roots = sproot(spl)

    if len(roots) != 2:  # multiple peaks or no peaks
        raise RuntimeError("Cannot determine FWHM")
    else:
        return roots[0], roots[1]


def peak_detect(x, y):
    try:
        fwhm_left_right = fwhm(x=x, y=y)
    except (RuntimeError, TypeError):
        raise RuntimeError("Cannot determine peak")
    fwhm_value = fwhm_left_right[-1] - fwhm_left_right[0]
    max_position = x[np.argmax(y)]
    center = (fwhm_left_right[0] + fwhm_left_right[-1]) / 2.0
    return max_position, center, fwhm_value, fwhm_left_right


def simple_peak_detect(x, y):
    y = np.array(y)
    half_maximum = np.max(y) * 0.5
    greater = (y > half_maximum)
    change_indices = np.where(greater[:-1] != greater[1:])[0]
    if not np.any(greater) or greater[0] is True or greater[-1] is True:
        raise RuntimeError("Cannot determine peak")
    x = np.array(x)
    # get center of indices for higher precision peak position and FWHM
    x_center = (x[1:] + x[:-1]) / 2.0
    fwhm_left_right = (x_center[change_indices[0]], x_center[change_indices[-1]])
    fwhm_value = fwhm_left_right[-1] - fwhm_left_right[0]
    max_position = x[np.argmax(y)]
    center = (fwhm_left_right[0] + fwhm_left_right[-1]) / 2.0
    return max_position, center, fwhm_value, fwhm_left_right


def fit_residuals(hist, edges):
    bin_center = (edges[1:] + edges[:-1]) / 2.0
    hist_mean = get_mean_from_histogram(hist, bin_center)
    hist_std = get_rms_from_histogram(hist, bin_center)
    if hist_std == 0:
        fit, cov = [np.amax(hist), hist_mean, hist_std], np.full((3, 3), np.nan)
    else:
        try:
            fit, cov = curve_fit(gauss, bin_center, hist, p0=[np.amax(hist), hist_mean, hist_std])
        except (RuntimeError, TypeError):
            fit, cov = [np.amax(hist), hist_mean, hist_std], np.full((3, 3), np.nan)
    return fit, cov


def fit_residuals_vs_position(hist, xedges, yedges, mean, count, limit=None):
    xcenter = (xedges[1:] + xedges[:-1]) / 2.0
    select = (count > 0)

    if limit is None:
        n_hits_threshold = np.percentile(count[select], 100.0 - 68.0)
        select &= (count > n_hits_threshold)
    else:
        limit_left = limit[0]
        limit_right = limit[1]
        if np.isfinite(limit_left):
            try:
                select_left = np.where(xcenter >= limit_left)[0][0]
            except IndexError:
                select_left = 0
        else:
            select_left = 0
        if np.isfinite(limit_right):
            try:
                select_right = np.where(xcenter <= limit_right)[0][-1] + 1
            except IndexError:
                select_right = select.shape[0]
        else:
            select_right = select.shape[0]
        select_range = np.zeros_like(select)
        select_range[select_left:select_right] = 1
        select &= select_range
#         n_hits_threshold = np.median(count[select]) * 0.1
#         select &= (count > n_hits_threshold)
    mean_fit = []
    for index in range(hist.shape[0]):
        if np.sum(hist[index, :]) == 0:
            mean_fit.append(np.nan)
        else:
            mean_fit.append(fit_residuals(hist[index, :].astype(np.int32), yedges)[0][1])
    select &= np.isfinite(mean_fit)
    mean_fit = np.ma.masked_invalid(mean_fit)
    if np.count_nonzero(select) > 1:
        y_rel_err = np.sum(count[select]) / count[select]
        fit, cov = curve_fit(linear, xcenter[select], mean_fit[select], p0=[0.0, 0.0], sigma=y_rel_err, absolute_sigma=False)
    else:
        fit, cov = [np.nan, np.nan], [[np.nan, np.nan], [np.nan, np.nan]]
    return fit, cov, select, mean_fit


def hough_transform(img, theta_res=1.0, rho_res=1.0, return_edges=False):
    thetas = np.linspace(-90.0, 0.0, np.ceil(90.0 / theta_res) + 1)
    thetas = np.concatenate((thetas, -thetas[len(thetas) - 2::-1]))
    thetas = np.deg2rad(thetas)
    width, height = img.shape
    diag_len = np.sqrt((width - 1)**2 + (height - 1)**2)
    q = np.ceil(diag_len / rho_res)
    nrhos = 2 * q + 1
    rhos = np.linspace(-q * rho_res, q * rho_res, nrhos)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    accumulator = np.zeros((rhos.size, thetas.size), dtype=np.int32)
    y_idxs, x_idxs = np.nonzero(img)

    @njit
    def loop(accumulator, x_idxs, y_idxs, thetas, rhos, sin_t, cos_t):

        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            for theta_idx in range(thetas.size):
                # rho_idx = np.around(x * cos_t[theta_idx] + y * sin_t[theta_idx]) + diag_len
                rhoVal = x * cos_t[theta_idx] + y * sin_t[theta_idx]
                rho_idx = (np.abs(rhos - rhoVal)).argmin()
                accumulator[rho_idx, theta_idx] += 1
    loop(accumulator, x_idxs, y_idxs, thetas, rhos, sin_t, cos_t)

    if return_edges:
        thetas_diff = thetas[1] - thetas[0]
        thetas_edges = (thetas[1:] + thetas[:-1]) / 2.0
        theta_edges = np.r_[thetas_edges[0] - thetas_diff, thetas_edges, thetas_edges[-1] + thetas_diff]
        rho_diff = rhos[1] - rhos[0]
        rho_edges = (rhos[1:] + rhos[:-1]) / 2.0
        rho_edges = np.r_[rho_edges[0] - rho_diff, rho_edges, rho_edges[-1] + rho_diff]
        return accumulator, thetas, rhos, theta_edges, rho_edges  # return histogram, bin centers, edges
    else:
        return accumulator, thetas, rhos  # return histogram and bin centers


def binned_statistic(x, values, func, nbins, range):
    '''The usage is approximately the same as the scipy one.

    See: https://stackoverflow.com/questions/26783719/efficiently-get-indices-of-histogram-bins-in-python
    '''
    N = len(values)
    r0, r1 = range

    digitized = (float(nbins) / (r1 - r0) * (x - r0)).astype(int)
    S = csr_matrix((values, [digitized, np.arange(N)]), shape=(nbins, N))

    return [func(group) for group in np.split(S.data, S.indptr[1:-1])]


@njit(numba.uint64(numba.uint32, numba.uint32))
def xy2d_morton(x, y):
    '''Tuple to number.

    See: https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits
    '''
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555

    y = (y | (y << 16)) & 0x0000FFFF0000FFFF
    y = (y | (y << 8)) & 0x00FF00FF00FF00FF
    y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F
    y = (y | (y << 2)) & 0x3333333333333333
    y = (y | (y << 1)) & 0x5555555555555555

    return x | (y << 1)


@njit(numba.uint64(numba.uint64,))
def morton_1(x):
    x = x & 0x5555555555555555
    x = (x | (x >> 1)) & 0x3333333333333333
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FF
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFF
    x = (x | (x >> 16)) & 0xFFFFFFFFFFFFFFFF  # TODO: 0x00000000FFFFFFFF
    return x


@njit((numba.uint64,))
def d2xy_morton(d):
    '''Number to tuple.

    See: https://stackoverflow.com/questions/30539347/2d-morton-code-encode-decode-64bits
    '''
    return morton_1(d), morton_1(d >> 1)


@njit(locals={'cluster_shape': numba.uint64})
def calculate_cluster_shape(cluster_array):
    '''Boolean 8x8 array to number.
    '''
    cluster_shape = 0
    indices_x, indices_y = np.nonzero(cluster_array)
    for index in np.arange(indices_x.size):
        cluster_shape += 2**xy2d_morton(indices_x[index], indices_y[index])
    return cluster_shape


@njit((numba.uint64,), locals={'val': numba.uint64})
def calculate_cluster_array(cluster_shape):
    '''Number to boolean 8x8 array.
    '''
    cluster_array = np.zeros((8, 8), dtype=np.bool_)
    for i in np.arange(63, -1, -1):
        val = 2**i
        if val <= cluster_shape:
            x, y = d2xy_morton(i)
            cluster_array[x, y] = 1
            cluster_shape -= val
    return cluster_array


def voronoi_finite_polygons_2d(points, dut_extent=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Useful links:
    - https://stackoverflow.com/questions/34968838/python-finite-boundary-voronoi-cells
    - https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram

    Parameters
    ----------
    points : ndarray of floats
        Coordinates of points to construct a convex hull from.
    dut_extent : list
        Boundary of the Voronoi diagram.
        If None, remove all Voronoi infinite regions.

    Returns
    -------
    points : ndarray of floats
        Coordinates of points to construct a convex hull from (including mirrored points).
    regions : list of list of ints
        Indices of vertices in each revised Voronoi regions.
    ridge_vetices : list of list of ints
        Indices of vertices in each revised Voronoi.
    vertices : ndarray of doubles
        Coordinates for revised Voronoi vertices.
    """
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError("Requires 2D voronoi data")
    vor = Voronoi(points, incremental=True)
    n_points = vor.points.shape[0]
    # select Voronoi regions with invalid ("infinite") vetices and vertices outside the boundary
    vertices_outside_sel = ((vor.vertices[:, 0] >= max(dut_extent[:2])) & ~np.isclose(vor.vertices[:, 0], max(dut_extent[:2])))
    vertices_outside_sel |= ((vor.vertices[:, 0] <= min(dut_extent[:2])) & ~np.isclose(vor.vertices[:, 0], min(dut_extent[:2])))
    vertices_outside_sel |= ((vor.vertices[:, 1] <= min(dut_extent[2:])) & ~np.isclose(vor.vertices[:, 1], min(dut_extent[2:])))
    vertices_outside_sel |= ((vor.vertices[:, 1] >= max(dut_extent[2:])) & ~np.isclose(vor.vertices[:, 1], max(dut_extent[2:])))
    vertices_indices_outside = np.where(vertices_outside_sel)
    length = len(sorted(vor.regions, key=len, reverse=True)[0])
    vor_regions = np.array([arr + [vor.vertices.shape[0]] * (length - len(arr)) for arr in vor.regions])
    regions_with_vertex_outside_sel = np.any(vor_regions == -1, axis=1)
    regions_with_vertex_outside_sel |= np.any(np.isin(vor_regions, vertices_indices_outside), axis=1)
    regions_indices_with_vertex_outside = np.where(regions_with_vertex_outside_sel)[0]
    points_indices_with_vertex_outside = np.in1d(vor.point_region, regions_indices_with_vertex_outside)
    points_with_vertex_outside = vor.points[points_indices_with_vertex_outside]
    # generate mirrored points at the boundary
    points_left = points_with_vertex_outside.copy()
    points_left[:, 0] -= (max(dut_extent[:2]) + min(dut_extent[:2])) / 2.0
    points_left[:, 0] *= -1
    points_left[:, 0] += (max(dut_extent[:2]) + min(dut_extent[:2])) / 2.0 - np.ptp(dut_extent[:2])
    points_right = points_with_vertex_outside.copy()
    points_right[:, 0] -= (max(dut_extent[:2]) + min(dut_extent[:2])) / 2.0
    points_right[:, 0] *= -1
    points_right[:, 0] += (max(dut_extent[:2]) + min(dut_extent[:2])) / 2.0 + np.ptp(dut_extent[:2])
    points_up = points_with_vertex_outside.copy()
    points_up[:, 1] -= (max(dut_extent[2:]) + min(dut_extent[2:])) / 2.0
    points_up[:, 1] *= -1
    points_up[:, 1] += (max(dut_extent[2:]) + min(dut_extent[2:])) / 2.0 + np.ptp(dut_extent[2:])
    points_down = points_with_vertex_outside.copy()
    points_down[:, 1] -= (max(dut_extent[2:]) + min(dut_extent[2:])) / 2.0
    points_down[:, 1] *= -1
    points_down[:, 1] += (max(dut_extent[2:]) + min(dut_extent[2:])) / 2.0 - np.ptp(dut_extent[2:])
    # adding the points and generate new Voronoi regions
    mirrored_points = np.concatenate((points_up, points_down, points_left, points_right))
    vor.add_points(mirrored_points)
    new_regions_indices = vor.point_region[:n_points]
    # select Voronoi regions with valid vetices and vertices inside the boundary
    vertices_inside_sel = ((vor.vertices[:, 0] <= max(dut_extent[:2])) | np.isclose(vor.vertices[:, 0], max(dut_extent[:2])))
    vertices_inside_sel &= ((vor.vertices[:, 0] >= min(dut_extent[:2])) | np.isclose(vor.vertices[:, 0], min(dut_extent[:2])))
    vertices_inside_sel &= ((vor.vertices[:, 1] >= min(dut_extent[2:])) | np.isclose(vor.vertices[:, 1], min(dut_extent[2:])))
    vertices_inside_sel &= ((vor.vertices[:, 1] <= max(dut_extent[2:])) | np.isclose(vor.vertices[:, 1], max(dut_extent[2:])))
    vertices_indices_inside = np.where(vertices_inside_sel)
    vor_ridge_vertices = np.array(vor.ridge_vertices)
    ridge_vertices_with_vertex_inside_sel = np.all(vor_ridge_vertices != -1.0, axis=1)
    ridge_vertices_with_vertex_inside_sel &= np.all(np.isin(vor_ridge_vertices, vertices_indices_inside), axis=1)
    ridge_vertices_indices_with_vertex_inside = np.where(ridge_vertices_with_vertex_inside_sel)[0]
    # TODO: remove not used vertices and update indices lists
    return vor.points, np.array(vor.regions)[new_regions_indices].tolist(), vor_ridge_vertices[ridge_vertices_indices_with_vertex_inside].tolist(), vor.vertices


def polygon_area(x, y):
    ''' Calculating the polygon area using Shoelace formula/Gausssche Trapezformel.

    Parameters
    ----------
    x : list
        X coodinates of the polygon.
    y : list
        Y coodinates of the polygon.

    Returns
    -------
    Area of the polygon.

    Note: Points must be provided in clockwise/counter clockwise order.

    See: https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    '''
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_area_multi(x, y):
    ''' Calculating the are of multiple polygons.
    '''
    return 0.5 * np.abs(np.einsum('ik,ik->i', x, np.roll(y, 1, axis=1)) - np.einsum('ik,ik->i', y, np.roll(x, 1, axis=1)))


def in1d_index(ar1, ar2, fill_invalid=None, assume_sorted=False):
    ''' Return indices of ar1 that overlap with ar2 and the indices of ar2 that occur in ar1.

    alternative implementation (only works if both ar1 and ar2 are unique!):
    def overlap(ar1, ar2):
        bool_ar1 = np.in1d(ar1, ar2)
        ind_ar1 = np.arange(len(ar1))
        ind_ar1 = ind_ar1[bool_ar1]
        ind_ar2 = np.array([np.argwhere(ar2 == ar1[x]) for x in ind_ar1]).flatten()
        return ind_ar1, ind_ar2

    Parameters
    ----------
    ar1 : array_like
        Input array.
    ar2 : array_like
        The values against which to test each value of ar1.
    fill_invalid : int
        If a value is given, in1d_index has the same lenght than ar1
        and invalid positions are filled with the given value.
    assume_sorted : bool
        If True, assume sorted ar2.

    Returns
    -------
    in1d_valid : ndarray
        The indices of ar1 overlapping with ar2.
    in1d_index : ndarray
        The indices of ar2 that occur in ar1.
    '''
    if assume_sorted:
        ar2_index = np.searchsorted(ar2, ar1)
    else:
        ar2_sorter = np.argsort(ar2)
        ar2_sorted = ar2[ar2_sorter]
        ar2_sorted_index = np.searchsorted(ar2_sorted, ar1)
        # Remove invalid indices
        ar2_sorted_index[ar2_sorted_index >= ar2.shape[0]] = ar2.shape[0] - 1
        # Go back into the original index
        ar2_index = ar2_sorter[ar2_sorted_index]
    if fill_invalid is None:
        valid = ar2.take(ar2_index, mode='clip') == ar1
        return np.where(valid)[0], ar2_index[valid]
    else:
        invalid = ar2.take(ar2_index, mode='clip') != ar1
        ar2_index[invalid] = fill_invalid
        return np.where(~invalid)[0], ar2_index


@njit
def unique_loop(mask, N, A, p, count):
    for j in range(N):
        mask[:] = True
        mask[A[0, j]] = False
        c = 1
        for i in range(1, p):
            if mask[A[i, j]]:
                c += 1
            mask[A[i, j]] = False
        count[j] = c
    return count


def unique_in_array(arr):
    ''' Return number of unique values along axis 0.

    See:
    - https://stackoverflow.com/questions/46893369/count-unique-elements-along-an-axis-of-a-numpy-array
    - https://stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array
    -
    '''
    p, m, n = arr.shape
    arr.shape = (-1, m * n)
    maxn = arr.max() + 1
    N = arr.shape[1]
    mask = np.empty(maxn, dtype=bool)
    count = np.empty(N, dtype=int)
    arr_out = unique_loop(mask, N, arr, p, count).reshape(m, n)
    arr.shape = (-1, m, n)
    return arr_out


def count_unique_neighbors(arr, structure=None):
    ''' Return number of unique neighbors (vertical, horizontal and diagonal) in a given 2D array.

    Parameters
    ----------
    data : ndarray
        2D array.
    structure : ndarray
        A 2D structuring element that defines the neighbors. The stucture
        must be symmetric.
        If None, the structuring element is [[1,1,1], [1,1,1], [1,1,1]].

    Returns
    -------
    Array with number of unique neighbor elements.

    See:
    - https://stackoverflow.com/questions/48248773/numpy-counting-unique-neighbours-in-2d-array
    - https://stackoverflow.com/questions/25169997/how-to-count-adjacent-elements-in-a-3d-numpy-array-efficiently
    - https://stackoverflow.com/questions/41550979/fill-holes-with-majority-of-surrounding-values-python
    '''
    if structure is None:
        structure = np.ones((3, 3), dtype=np.bool)
    else:
        structure = np.array(structure, dtype=np.bool)
    if len(structure.shape) != 2:
        raise ValueError('Structure must be a 2D array')
    if structure.shape[0] != structure.shape[1]:
        raise ValueError('Structure must be symmetrical')
    if structure.shape[0] % 2 == 0:
        raise ValueError('Structure shape must be odd')
    selected_indices = np.column_stack(np.where(structure))
    extent = int(structure.shape[0] / 2)
    selected_indices -= extent
    a = np.pad(arr, (extent, extent), mode='reflect')
    selected_arrays = []
    for selected_index in selected_indices:
        selected_arrays.append(a[extent + selected_index[0]:a.shape[0] - extent + selected_index[0], extent + selected_index[1]:a.shape[1] - extent + selected_index[1]])
    return unique_in_array(np.array(selected_arrays))


def fill(arr, invalid=None):
    '''
    Replace the value of invalid data cells by the value of the nearest valid data cell.

    See: https://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays

    Parameters
    ----------
    arr : ndarray
        Array of any dimension.
    invalid : ndarray
        Boolean array of the same shape as data where True indicates data cells which are to be replaced.
        If None (default), generate the array by assuming invalid values to be nan.

    Returns
    -------
    Array with filled data cells.
    '''
    if invalid is None:
        invalid = np.isnan(arr)
    else:
        invalid = np.array(invalid, dtype=np.bool)

    fill_indices = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return arr[tuple(fill_indices)]


def get_data(path, output=None, fail_on_overwrite=False):
    ''' Downloads data (eg. for examples, fixtures).

        Uses data in a public scibo folder. If you want
        write access contact the maintainer.

        Parameters
        ----------
        path : string
            File path with name. Location on online folder.
        output : string, None
            File path with name. Location where to store data.
            If None the path variable path is used.
        fail_on_overwrite : Bool
            If files exist already the download is skipped.
            If fail_on_overwrite this raises a RuntimeError.
    '''
    def download_scibo(public_secret, path, filename):
        folder = os.path.dirname(path)
        name = os.path.basename(path)

        url = "https://uni-bonn.sciebo.de/index.php/s/"
        url += public_secret + '/download?path=%2F'
        url += folder + '&files='
        url += name

        logging.info('Downloading %s' % name)

        r = requests.get(url, stream=True)
        file_size = int(r.headers['Content-Length'])
        logging.info('Downloading %s', name)
        with open(filename, 'wb') as f:
            num_bars = file_size / (32 * 1024)
            bar = progressbar.ProgressBar(maxval=num_bars).start()
            for i, chunk in enumerate(r.iter_content(32 * 1024)):
                f.write(chunk)
                bar.update(i)

    if not output:
        output = os.path.basename(path)
        output_path = os.path.dirname(os.path.realpath(path))
    else:
        output_path = os.path.dirname(os.path.realpath(output))

    if not os.path.isfile(os.path.join(output_path, output)):
        # Create output folder
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        download_scibo(public_secret=SCIBO_PUBLIC_FOLDER,
                       path=path,
                       filename=os.path.join(output_path, output))
    elif fail_on_overwrite:
        raise RuntimeError('The files %s exists already', output)

    return os.path.join(output_path, output)
