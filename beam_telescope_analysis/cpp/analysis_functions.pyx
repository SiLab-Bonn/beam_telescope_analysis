# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=2

from numpy cimport ndarray
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int64_t
from beam_telescope_analysis.cpp.data_struct cimport numpy_hit_info, numpy_cluster_info

import numpy as np
cimport numpy as cnp

cnp.import_array()  # if array is used it has to be imported, otherwise possible runtime error


cdef extern from "AnalysisFunctions.h":
    cdef cppclass ClusterInfo:
        ClusterInfo()
    cdef cppclass HitInfo:
        HitInfo()
    unsigned int getEventsInBothArrays(int64_t * & rEventArrayOne, const unsigned int & rSizeArrayOne, int64_t * & rEventArrayTwo, const unsigned int & rSizeArrayTwo, int64_t * & rEventArrayIntersection)
    unsigned int getMaxEventsInBothArrays(int64_t * & rEventArrayOne, const unsigned int & rSizeArrayOne, int64_t * & rEventArrayTwo, const unsigned int & rSizeArrayTwo, int64_t * & rEventArrayIntersection, const unsigned int & rSizeArrayResult) except +
    void in1d_sorted(int64_t * & rEventArrayOne, const unsigned int & rSizeArrayOne, int64_t * & rEventArrayTwo, const unsigned int & rSizeArrayTwo, uint8_t * & rSelection)
    void histogram_1d(int * & x, const unsigned int & rSize, const unsigned int & rNbinsX, uint32_t * & rResult) except +
    void histogram_2d(int * & x, int * & y, const unsigned int & rSize, const unsigned int & rNbinsX, const unsigned int & rNbinsY, uint32_t * & rResult) except +
    void histogram_3d(int * & x, int * & y, int * & z, const unsigned int & rSize, const unsigned int & rNbinsX, const unsigned int & rNbinsY, const unsigned int & rNbinsZ, uint16_t * & rResult) except +


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
