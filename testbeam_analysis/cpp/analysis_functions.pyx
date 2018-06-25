# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
from numpy cimport ndarray
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t
from testbeam_analysis.cpp.data_struct cimport numpy_hit_info, numpy_cluster_info

import numpy as np
cimport numpy as cnp

cnp.import_array()  # if array is used it has to be imported, otherwise possible runtime error


cdef extern from "AnalysisFunctions.h":
    cdef cppclass ClusterInfo:
        ClusterInfo()
    cdef cppclass HitInfo:
        HitInfo()
    unsigned int getNclusterInEvents(int64_t * & rEventNumber, const unsigned int & rSize, int64_t * & rResultEventNumber, unsigned int * & rResultCount)
    unsigned int getEventsInBothArrays(int64_t * & rEventArrayOne, const unsigned int & rSizeArrayOne, int64_t * & rEventArrayTwo, const unsigned int & rSizeArrayTwo, int64_t * & rEventArrayIntersection)
    unsigned int getMaxEventsInBothArrays(int64_t * & rEventArrayOne, const unsigned int & rSizeArrayOne, int64_t * & rEventArrayTwo, const unsigned int & rSizeArrayTwo, int64_t * & rEventArrayIntersection, const unsigned int & rSizeArrayResult) except +
    void in1d_sorted(int64_t * & rEventArrayOne, const unsigned int & rSizeArrayOne, int64_t * & rEventArrayTwo, const unsigned int & rSizeArrayTwo, uint8_t * & rSelection)
    void histogram_1d(int * & x, const unsigned int & rSize, const unsigned int & rNbinsX, uint32_t * & rResult) except +
    void histogram_2d(int * & x, int * & y, const unsigned int & rSize, const unsigned int & rNbinsX, const unsigned int & rNbinsY, uint32_t * & rResult) except +
    void histogram_3d(int * & x, int * & y, int * & z, const unsigned int & rSize, const unsigned int & rNbinsX, const unsigned int & rNbinsY, const unsigned int & rNbinsZ, uint16_t * & rResult) except +
    unsigned int fixEventAlignment(const int64_t * & rEventArray, const double * & rRefCol, double * & rCol, const double * & rRefRow, double * & rRow, const uint16_t * & rRefCharge, uint16_t * & rCharge, uint8_t * & rCorrelated, const unsigned int & nHits, const double & rError, const unsigned int & nBadEvents, const unsigned int & correltationSearchRange, const unsigned int & nGoodEvents, const unsigned int & goodEventsSearchRange) except +


def get_n_cluster_in_events(cnp.ndarray[cnp.int64_t, ndim=1] event_numbers, cnp.ndarray[cnp.int64_t, ndim=1] result_event_numbers, cnp.ndarray[cnp.uint32_t, ndim=1] result_cluster_count):
    return getNclusterInEvents(< int64_t*& > event_numbers.data, < const unsigned int&> event_numbers.shape[0], < int64_t*& > result_event_numbers.data, < unsigned int*& > result_cluster_count.data)


def get_events_in_both_arrays(cnp.ndarray[cnp.int64_t, ndim=1] array_one, cnp.ndarray[cnp.int64_t, ndim=1] array_two, cnp.ndarray[cnp.int64_t, ndim=1] array_result):
    return getEventsInBothArrays( < int64_t*& > array_one.data, < const unsigned int&> array_one.shape[0], < int64_t*& > array_two.data, < const unsigned int&> array_two.shape[0], < int64_t*& > array_result.data)


def get_max_events_in_both_arrays(cnp.ndarray[cnp.int64_t, ndim=1] array_one, cnp.ndarray[cnp.int64_t, ndim=1] array_two, cnp.ndarray[cnp.int64_t, ndim=1] array_result):
    return getMaxEventsInBothArrays(< int64_t*& > array_one.data, < const unsigned int&> array_one.shape[0], < int64_t*& > array_two.data, < const unsigned int&> array_two.shape[0], < int64_t*& > array_result.data, < const unsigned int&> array_result.shape[0])


def get_in1d_sorted(cnp.ndarray[cnp.int64_t, ndim=1] array_one, cnp.ndarray[cnp.int64_t, ndim=1] array_two, cnp.ndarray[cnp.uint8_t, ndim=1] array_result):
    in1d_sorted( < int64_t*& > array_one.data, < const unsigned int&> array_one.shape[0], < int64_t*& > array_two.data, < const unsigned int&> array_two.shape[0], < uint8_t*& > array_result.data)
    return (array_result == 1)


def hist_1d(cnp.ndarray[cnp.int32_t, ndim=1] x, const unsigned int & n_x, cnp.ndarray[cnp.uint32_t, ndim=1] array_result):
    histogram_1d( < int*& > x.data, < const unsigned int&> x.shape[0], < const unsigned int&> n_x, < uint32_t*& > array_result.data)


def hist_2d(cnp.ndarray[cnp.int32_t, ndim=1] x, cnp.ndarray[cnp.int32_t, ndim=1] y, const unsigned int & n_x, const unsigned int & n_y, cnp.ndarray[cnp.uint32_t, ndim=1] array_result):
    histogram_2d(< int*& > x.data, < int*& > y.data, < const unsigned int&> x.shape[0], < const unsigned int&> n_x, < const unsigned int&> n_y, < uint32_t*& > array_result.data)


def hist_3d(cnp.ndarray[cnp.int32_t, ndim=1] x, cnp.ndarray[cnp.int32_t, ndim=1] y, cnp.ndarray[cnp.int32_t, ndim=1] z, const unsigned int & n_x, const unsigned int & n_y, const unsigned int & n_z, cnp.ndarray[cnp.uint16_t, ndim=1] array_result, throw_exception=True):
    histogram_3d( < int*& > x.data, < int*& > y.data, < int*& > z.data, < const unsigned int&> x.shape[0], < const unsigned int&> n_x, < const unsigned int&> n_y, < const unsigned int&> n_z, < uint16_t*& > array_result.data)


def fix_event_alignment(cnp.ndarray[cnp.int64_t, ndim=1] event_array, cnp.ndarray[cnp.float_t, ndim=1] ref_column, cnp.ndarray[cnp.float_t, ndim=1] column, cnp.ndarray[cnp.float_t, ndim=1] ref_row, cnp.ndarray[cnp.float_t, ndim=1] row, cnp.ndarray[cnp.uint16_t, ndim=1] ref_charge, cnp.ndarray[cnp.uint16_t, ndim=1] charge, cnp.ndarray[cnp.uint8_t, ndim=1] correlated, const double & error, const unsigned int & n_bad_events, const unsigned int & correlation_search_range, const unsigned int & n_good_events, const unsigned int & good_events_search_range):
    return fixEventAlignment( < const int64_t*& > event_array.data, < const double*& > ref_column.data, < double*& > column.data, < const double*& > ref_row.data, < double*& > row.data, < const uint16_t*& > ref_charge.data, < uint16_t*& > charge.data, < uint8_t*& > correlated.data, < const unsigned int&> event_array.shape[0], < const double&> error, < const unsigned int&> n_bad_events, < const unsigned int&> correlation_search_range, < const unsigned int&> n_good_events, < const unsigned int&> good_events_search_range)
