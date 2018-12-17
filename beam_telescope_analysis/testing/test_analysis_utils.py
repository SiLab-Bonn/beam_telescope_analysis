''' Script to check the correctness of the analysis utils that are written in C++.
'''
import os

import unittest

import tables as tb
import numpy as np

from beam_telescope_analysis.cpp import data_struct
from beam_telescope_analysis.tools import analysis_utils, test_tools

testing_path = os.path.dirname(os.path.abspath(__file__))


class TestAnalysisUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):  # remove created files
        pass

    def test_analysis_utils_get_events_in_both_arrays(self):  # check compiled get_events_in_both_arrays function
        event_numbers = np.array([[0, 0, 2, 2, 2, 4, 5, 5, 6, 7, 7, 7, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        event_numbers_2 = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 7], dtype=np.int64)
        result = analysis_utils.get_events_in_both_arrays(event_numbers[0], event_numbers_2)
        self.assertListEqual([2, 4, 7], result.tolist())

    def test_analysis_utils_get_max_events_in_both_arrays(self):  # check compiled get_max_events_in_both_arrays function
        # Test 1
        event_numbers = np.array([[0, 0, 1, 1, 2], [0, 0, 0, 0, 0]], dtype=np.int64)
        event_numbers_2 = np.array([0, 3, 3, 4], dtype=np.int64)
        result = analysis_utils.get_max_events_in_both_arrays(event_numbers[0], event_numbers_2)
        self.assertListEqual([0, 0, 1, 1, 2, 3, 3, 4], result.tolist())
        # Test 2
        event_numbers = np.array([1, 1, 2, 4, 5, 6, 7], dtype=np.int64)
        event_numbers_2 = np.array([0, 3, 3, 4], dtype=np.int64)
        result = analysis_utils.get_max_events_in_both_arrays(event_numbers, event_numbers_2)
        self.assertListEqual([0, 1, 1, 2, 3, 3, 4, 5, 6, 7], result.tolist())
        # Test 3
        event_numbers = np.array([1, 1, 2, 4, 5, 6, 7], dtype=np.int64)
        event_numbers_2 = np.array([6, 7, 9, 10], dtype=np.int64)
        result = analysis_utils.get_max_events_in_both_arrays(event_numbers, event_numbers_2)
        self.assertListEqual([1, 1, 2, 4, 5, 6, 7, 9, 10], result.tolist())
        # Test 4
        event_numbers = np.array([1, 1, 2, 4, 5, 6, 7, 10, 10], dtype=np.int64)
        event_numbers_2 = np.array([1, 6, 7, 9, 10], dtype=np.int64)
        result = analysis_utils.get_max_events_in_both_arrays(event_numbers, event_numbers_2)
        self.assertListEqual([1, 1, 2, 4, 5, 6, 7, 9, 10, 10], result.tolist())
        # Test 5
        event_numbers = np.array([1, 1, 2, 4, 5, 6, 7, 10, 10], dtype=np.int64)
        event_numbers_2 = np.array([1, 1, 1, 6, 7, 9, 10], dtype=np.int64)
        result = analysis_utils.get_max_events_in_both_arrays(event_numbers, event_numbers_2)
        self.assertListEqual([1, 1, 1, 2, 4, 5, 6, 7, 9, 10, 10], result.tolist())

    def test_map_cluster(self):  # check the compiled function against result
        clusters = np.zeros((20, ), dtype=tb.dtype_from_descr(data_struct.ClusterInfoTable))
        result = np.zeros((20, ), dtype=tb.dtype_from_descr(data_struct.ClusterInfoTable))
        result["mean_column"] = np.nan
        result["mean_row"] = np.nan
        result["charge"] = np.nan
        result[1]["event_number"], result[3]["event_number"], result[7]["event_number"], result[8]["event_number"], result[9]["event_number"] = 1, 2, 4, 4, 19
        result[0]["mean_column"], result[1]["mean_column"], result[3]["mean_column"], result[7]["mean_column"], result[8]["mean_column"], result[9]["mean_column"] = 1, 2, 3, 5, 6, 20
        result[0]["mean_row"], result[1]["mean_row"], result[3]["mean_row"], result[7]["mean_row"], result[8]["mean_row"], result[9]["mean_row"] = 0, 0, 0, 0, 0, 0
        result[0]["charge"], result[1]["charge"], result[3]["charge"], result[7]["charge"], result[8]["charge"], result[9]["charge"] = 0, 0, 0, 0, 0, 0

        for index, cluster in enumerate(clusters):
            cluster['mean_column'] = index + 1
            cluster["event_number"] = index
        clusters[3]["event_number"] = 2
        clusters[5]["event_number"] = 4

        common_event_number = np.array([0, 1, 1, 2, 3, 3, 3, 4, 4], dtype=np.int64)

        data_equal = test_tools.nan_equal(first_array=analysis_utils.map_cluster(common_event_number, clusters),
                                          second_array=result[:common_event_number.shape[0]])
        self.assertTrue(data_equal)

    def test_analysis_utils_in1d_events(self):  # check compiled get_in1d_sorted function
        event_numbers = np.array([[0, 0, 2, 2, 2, 4, 5, 5, 6, 7, 7, 7, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        event_numbers_2 = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 7], dtype=np.int64)
        result = event_numbers[0][analysis_utils.in1d_events(event_numbers[0], event_numbers_2)]
        self.assertListEqual([2, 2, 2, 4, 7, 7, 7], result.tolist())

    def test_1d_index_histograming(self):  # check compiled hist_2D_index function
        x = np.random.randint(0, 100, 100)
        shape = (100, )
        array_fast = analysis_utils.hist_1d_index(x, shape=shape)
        array = np.histogram(x, bins=shape[0], range=(0, shape[0]))[0]
        shape = (5, )  # shape that is too small for the indices to trigger exception
        exception_ok = False
        try:
            array_fast = analysis_utils.hist_1d_index(x, shape=shape)
        except IndexError:
            exception_ok = True
        except:  # other exception that should not occur
            pass
        self.assertTrue(exception_ok & np.all(array == array_fast))

    def test_2d_index_histograming(self):  # check compiled hist_2D_index function
        x, y = np.random.randint(0, 100, 100), np.random.randint(0, 100, 100)
        shape = (100, 100)
        array_fast = analysis_utils.hist_2d_index(x, y, shape=shape)
        array = np.histogram2d(x, y, bins=shape, range=[[0, shape[0]], [0, shape[1]]])[0]
        shape = (5, 200)  # shape that is too small for the indices to trigger exception
        exception_ok = False
        try:
            array_fast = analysis_utils.hist_2d_index(x, y, shape=shape)
        except IndexError:
            exception_ok = True
        except:  # other exception that should not occur
            pass
        self.assertTrue(exception_ok & np.all(array == array_fast))

    def test_3d_index_histograming(self):  # check compiled hist_3D_index function
        with tb.open_file(analysis_utils.get_data('fixtures/analysis_utils/hist_data.h5',
                                                  output=os.path.join(testing_path, 'fixtures/analysis_utils/hist_data.h5')),
                          mode="r") as in_file_h5:
            xyz = in_file_h5.root.HistDataXYZ[:]
            x, y, z = xyz[0], xyz[1], xyz[2]
            shape = (100, 100, 100)
            array_fast = analysis_utils.hist_3d_index(x, y, z, shape=shape)
            array = np.histogramdd(np.column_stack((x, y, z)), bins=shape, range=[[0, shape[0] - 1], [0, shape[1] - 1], [0, shape[2] - 1]])[0]
            shape = (50, 200, 200)  # shape that is too small for the indices to trigger exception
            exception_ok = False
            try:
                array_fast = analysis_utils.hist_3d_index(x, y, z, shape=shape)
            except IndexError:
                exception_ok = True
            except:  # other exception that should not occur
                pass
            self.assertTrue(exception_ok & np.all(array == array_fast))

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAnalysisUtils)
    unittest.TextTestRunner(verbosity=2).run(suite)
