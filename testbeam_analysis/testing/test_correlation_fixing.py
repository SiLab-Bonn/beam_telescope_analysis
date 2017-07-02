import unittest

import numpy as np

from testbeam_analysis.tools import analysis_utils
from testbeam_analysis import analysis_functions


def get_random_data(n_hits, hits_per_event=2, seed=0):
    np.random.seed(seed)
    event_numbers = np.arange(n_hits, dtype=np.int64).repeat(hits_per_event)[:n_hits]
    ref_column, ref_row, ref_charge = np.random.uniform(high=80, size=n_hits), np.random.uniform(high=336, size=n_hits), np.random.uniform(high=256, size=n_hits).astype(np.uint16)
    column, row, charge = ref_column.copy(), ref_row.copy(), ref_charge.copy()
    corr = np.ascontiguousarray(np.ones(shape=event_numbers.shape, dtype=np.uint8))  # array to signal correlation to be ables to omit not correlated events in the analysis

    event_numbers = np.ascontiguousarray(event_numbers)
    ref_column = np.ascontiguousarray(ref_column)
    column = np.ascontiguousarray(column)
    ref_row = np.ascontiguousarray(ref_row)
    row = np.ascontiguousarray(row)
    ref_charge = np.ascontiguousarray(ref_charge)
    charge = np.ascontiguousarray(charge)

    return event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr


class TestCorrelationFixing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):  # remove created files
        pass

    def test_fix_event_alignment(self):  # check with multiple jumps data
        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, _ = get_random_data(50)
        column, row, charge = np.zeros_like(column), np.zeros_like(row), np.zeros_like(charge)
        # Create not correlated events
        column[10:] = ref_column[0:-10]
        row[10:] = ref_row[0:-10]
        charge[10:] = ref_charge[0:-10]
        column[20:] = ref_column[20:]
        row[20:] = ref_row[20:]
        charge[20:] = ref_charge[20:]
        row[10] = 3.14159
        column[10] = 3.14159
        charge[10] = 3

        corr, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=100)

        # Check fixes counter
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[0:10] == 1))
        self.assertTrue(np.all(corr[20:] == 1))

        # The data is correlated here
        self.assertTrue(np.all(ref_column[1:10] == column[1:10]))
        self.assertTrue(np.all(ref_row[1:10] == row[1:10]))
        self.assertTrue(np.all(ref_charge[1:10] == charge[1:10]))
        self.assertTrue(np.all(ref_column[20:] == column[20:]))
        self.assertTrue(np.all(ref_row[20:] == row[20:]))
        self.assertTrue(np.all(ref_charge[20:] == charge[20:]))

        # Shifted data has to leave zeroes
        self.assertEqual(column[0], 3.14159)
        self.assertEqual(row[0], 3.14159)
        self.assertEqual(charge[0], 3)
        self.assertTrue(np.all(row[10:20] == 0))
        self.assertTrue(np.all(column[10:20] == 0))
        self.assertTrue(np.all(charge[10:20] == 0))

    def test_missing_data(self):  # check behavior with missing data, but correlation
        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, _ = get_random_data(50)

        # Create no hits (virtual hits) in DUT 1
        column[5:15] = 0
        row[5:15] = 0

        corr, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=100)

        # Check that no fixes where done
        self.assertEqual(n_fixes, 0)

        # Correlation flag check
        self.assertTrue(np.all(corr[:6] == 1))
        self.assertTrue(np.all(corr[6:14] == 0))
        self.assertTrue(np.all(corr[14:] == 1))

        # Data is the same where there are hits and correlation flag is set
        self.assertTrue(np.all(ref_column[np.logical_and(corr == 1, column != 0)] == column[np.logical_and(corr == 1, column != 0)]))
        self.assertTrue(np.all(row[np.logical_and(corr == 1, column != 0)] == ref_row[np.logical_and(corr == 1, column != 0)]))
        self.assertTrue(np.all(charge[np.logical_and(corr == 1, column != 0)] == ref_charge[np.logical_and(corr == 1, column != 0)]))

    def test_correlation_flag(self):  # check behavior of the correlation flag
        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr = get_random_data(500)
        column[5:16] = 0
        row[5:16] = 0
        charge[5:16] = 0
        column[16:20] = ref_column[6:10]
        row[16:20] = ref_row[6:10]
        charge[16:20] = ref_charge[6:10]
        corr[16:18] = 0  # create not correlated event

        # Check with correlation hole
        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=100)

        self.assertEqual(n_fixes, 0)  # no fixes are expected
        # Correlation flag check
        self.assertTrue(np.all(corr[0:6] == 1))
        self.assertTrue(np.all(corr[6:19] == 0))
        self.assertTrue(np.all(corr[20:] == 1))

        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr = get_random_data(50)
        column[5:16] = 0
        row[5:16] = 0
        charge[5:16] = 0
        column[16:20] = ref_column[6:10]
        row[16:20] = ref_row[6:10]
        charge[16:20] = ref_charge[6:10]
        corr[16:18] = 0  # create not correlated event

        # check with event copying
        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr, error=0.1, n_bad_events=3, n_good_events=1, correlation_search_range=100, good_events_search_range=4)

        self.assertEqual(n_fixes, 1)  # 1 fixe are expected

        # Correlation flag check
        self.assertTrue(np.all(corr[0:6] == 1))
        self.assertTrue(np.all(corr[6:8] == 0))
        self.assertTrue(np.all(corr[8:10] == 1))
        self.assertTrue(np.all(corr[10:20] == 0))
        self.assertTrue(np.all(corr[20:] == 1))

        # Data check
        self.assertTrue(np.all(ref_row[:5] == row[:5]))
        self.assertTrue(np.all(ref_column[:5] == column[:5]))
        self.assertTrue(np.all(ref_charge[:5] == charge[:5]))
        self.assertTrue(np.all(ref_row[6:10] == row[6:10]))
        self.assertTrue(np.all(ref_column[6:10] == column[6:10]))
        self.assertTrue(np.all(ref_charge[6:10] == charge[6:10]))
        self.assertTrue(np.all(ref_row[20:] == row[20:]))
        self.assertTrue(np.all(ref_column[20:] == column[20:]))
        self.assertTrue(np.all(ref_charge[20:] == charge[20:]))

    def test_no_correction(self):  # check behavior if no correction is needed
        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr = get_random_data(5000)
        # Check with correlation hole
        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr, error=0.1, n_bad_events=3, n_good_events=3, correlation_search_range=100, good_events_search_range=100)

        self.assertEqual(n_fixes, 0)  # no fixes are expected
        self.assertTrue(np.all(corr == 1))  # Correlation flag check
        self.assertTrue(np.all(ref_column == column))  # Similarity check
        self.assertTrue(np.all(ref_row == row))  # Similarity check
        self.assertTrue(np.all(ref_charge == charge))  # Similarity check

    def test_virtual_hit_copying(self):  # check behavior for virtual hits
        # Test with virtual hits in dut
        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr = get_random_data(20)
        event_numbers[:12] = np.array([0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5])
        column[4:] = column[1:-3]
        row[4:] = row[1:-3]
        charge[4:] = charge[1:-3]
        column[1:4] = 0
        row[1:4] = 0
        charge[1:4] = 0
        column[4:-1] = column[5:]
        row[4:-1] = row[5:]
        charge[4:-1] = charge[5:]
        column[9:] = column[8:-1]
        row[9:] = row[8:-1]
        charge[9:] = charge[8:-1]
        column[8] = 0
        row[8] = 0
        charge[8] = 0
        column[13:] = ref_column[11:-2]
        row[13:] = ref_row[11:-2]
        charge[13:] = ref_charge[11:-2]
        column[12] = 0
        row[12] = 0
        charge[12] = 0

        # Check with correlation hole
        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr, error=0.1, n_bad_events=2, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[:18] == 1))
        self.assertTrue(np.all(corr[18:] == 0))

        # Similarity check
        self.assertEqual(ref_column[0], column[0])
        self.assertEqual(ref_row[0], row[0])
        self.assertEqual(ref_charge[0], charge[0])
        self.assertTrue(np.all(ref_column[4:9] == column[4:9]))
        self.assertTrue(np.all(ref_row[4:9] == row[4:9]))
        self.assertTrue(np.all(ref_charge[4:9] == charge[4:9]))
        self.assertTrue(np.all(ref_column[11:18] == column[11:18]))
        self.assertTrue(np.all(ref_row[11:18] == row[11:18]))
        self.assertTrue(np.all(ref_charge[11:18] == charge[11:18]))

        # Virtual hits check
        self.assertEqual(column[3], 0)
        self.assertEqual(row[3], 0)
        self.assertEqual(charge[3], 0)
        self.assertTrue(np.all(column[9:11] == 0))
        self.assertTrue(np.all(row[9:11] == 0))
        self.assertTrue(np.all(charge[9:11] == 0))
        self.assertTrue(np.all(column[18:] == 0))
        self.assertTrue(np.all(row[18:] == 0))
        self.assertTrue(np.all(charge[18:] == 0))

        # Test with virtual hits in reference, positive event number offset (ref to dut)
        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr = get_random_data(20)
        event_numbers[:12] = np.array([0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5])
        column[4:] = column[1:-3]
        row[4:] = row[1:-3]
        charge[4:] = charge[1:-3]
        column[1:4] = 0
        row[1:4] = 0
        charge[1:4] = 0
        ref_column[4] = 0
        ref_row[4] = 0
        ref_charge[4] = 0
        column[4:-1] = column[5:]
        row[4:-1] = row[5:]
        charge[4:-1] = charge[5:]
        column[9:] = column[8:-1]
        row[9:] = row[8:-1]
        charge[9:] = charge[8:-1]
        column[8] = 0
        row[8] = 0
        charge[8] = 0
        column[13:] = ref_column[11:-2]
        row[13:] = ref_row[11:-2]
        charge[13:] = ref_charge[11:-2]
        column[12] = 0
        row[12] = 0
        charge[12] = 0
        ref_column[14:16] = 0
        ref_row[14:16] = 0
        ref_charge[14:16] = 0

        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr, error=5, n_bad_events=2, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # Correlation flag check
        self.assertTrue(np.all(corr[:18] == 1))
        self.assertTrue(np.all(corr[18:] == 0))

        # Similarity check
        self.assertEqual(ref_column[0], column[0])
        self.assertEqual(ref_row[0], row[0])
        self.assertEqual(ref_charge[0], charge[0])
        self.assertTrue(np.all(ref_column[5:9] == column[5:9]))
        self.assertTrue(np.all(ref_row[5:9] == row[5:9]))
        self.assertTrue(np.all(ref_charge[5:9] == charge[5:9]))
        self.assertTrue(np.all(ref_column[11:14] == column[11:14]))
        self.assertTrue(np.all(ref_row[11:14] == row[11:14]))
        self.assertTrue(np.all(ref_charge[11:14] == charge[11:14]))
        self.assertTrue(np.all(ref_column[16:18] == column[16:18]))
        self.assertTrue(np.all(ref_row[16:18] == row[16:18]))
        self.assertTrue(np.all(ref_charge[16:18] == charge[16:18]))

        # Virtual hits check
        self.assertEqual(column[3], 0)
        self.assertEqual(row[3], 0)
        self.assertEqual(charge[3], 0)
        self.assertEqual(ref_column[4], 0)
        self.assertEqual(ref_row[4], 0)
        self.assertEqual(ref_charge[4], 0)
        self.assertTrue(np.all(column[9:11] == 0))
        self.assertTrue(np.all(row[9:11] == 0))
        self.assertTrue(np.all(charge[9:11] == 0))
        self.assertTrue(np.all(ref_column[14:16] == 0))
        self.assertTrue(np.all(ref_row[14:16] == 0))
        self.assertTrue(np.all(ref_charge[14:16] == 0))
        self.assertTrue(np.all(column[18:] == 0))
        self.assertTrue(np.all(row[18:] == 0))
        self.assertTrue(np.all(charge[18:] == 0))

        # Test with virtual hits, forward fixing, real data, negative event number offset (ref to dut)
        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr = get_random_data(20)
        event_numbers[:20] = np.array([0, 1, 2, 3, 4, 5, 7, 8, 8, 9, 10, 11, 11, 11, 12, 12, 12, 12, 13, 14])

        ref_column[0], column[0], ref_row[0], row[0] = 51.5, 0.000, 325.5, 0.0
        ref_column[1], column[1], ref_row[1], row[1] = 71.5, 51.57, 144.5, 324.72
        ref_column[2], column[2], ref_row[2], row[2] = 13.5, 71.26, 136.8, 143.54
        ref_column[3], column[3], ref_row[3], row[3] = 40.5, 14.03, 244.5, 136.59
        ref_column[4], column[4], ref_row[4], row[4] = 0.00, 41.55, 0.000, 241.10
        ref_column[5], column[5], ref_row[5], row[5] = 0.00, 1.995, 0.000, 144.11
        ref_column[6], column[6], ref_row[6], row[6] = 30.5, 0.000, 280.5, 0.0
        ref_column[7], column[7], ref_row[7], row[7] = 25.5, 30.18, 112.5, 278.87
        ref_column[8], column[8], ref_row[8], row[8] = 76.5, 0.000, 205.25, 0.0
        ref_column[9], column[9], ref_row[9], row[9] = 47.5, 0.000, 327.5, 0.0
        ref_column[10], column[10], ref_row[10], row[10] = 0.0, 47.47, 0.0, 330.97
        ref_column[11], column[11], ref_row[11], row[11] = 23.5, 0.0, 310.5, 0.0
        ref_column[12], column[12], ref_row[12], row[12] = 37.5, 0.0, 200.5, 0.0
        ref_column[13], column[13], ref_row[13], row[13] = 57.5, 0.0, 192.5, 0.0
        ref_column[14], column[14], ref_row[14], row[14] = 53.5, 25.89, 128.5, 333.5
        ref_column[15], column[15], ref_row[15], row[15] = 0.00, 13.38, 0.0, 188.23
        ref_column[16], column[16], ref_row[16], row[16] = 0.0, 38.59, 0.0, 203.75
        ref_column[17], column[17], ref_row[17], row[17] = 0.0, 58.31, 0.0, 193.67
        ref_column[18], column[18], ref_row[18], row[18] = 20.5, 53.54, 247.5, 124.68
        ref_column[19], column[19], ref_row[19], row[19] = 72.5, 20.95, 219.5, 246.44

        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr, error=8, n_bad_events=2, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix is expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[1:4] == 1))

        # Similarity check
        self.assertTrue(np.all(np.abs(ref_column[0:4] - column[0:4]) < 8))
        self.assertTrue(np.all(np.abs(ref_row[0:4] - row[0:4]) < 8))
        self.assertTrue(np.all(np.abs(ref_column[6] - column[6]) < 8))
        self.assertTrue(np.all(np.abs(ref_row[6] - row[6]) < 8))
        self.assertTrue(np.all(np.abs(ref_column[9] - column[9]) < 8))
        self.assertTrue(np.all(np.abs(ref_row[9] - row[9]) < 8))
        self.assertTrue(np.all(np.abs(ref_column[18] - column[18]) < 8))
        self.assertTrue(np.all(np.abs(ref_row[18] - row[18]) < 8))
#         self.assertTrue(np.all(np.abs(ref_column[7:11] - column[7:11]) < 8))
#         self.assertTrue(np.all(np.abs(ref_row[7:11] - row[7:11]) < 8))

        # Virtual hits check
        self.assertEqual(ref_column[4], 0)
        self.assertEqual(ref_row[4], 0)

        # 2. Test with virtual hits, forward fixing, real data, negative event number offset (ref to dut)
        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr = get_random_data(11)
        event_numbers[:11] = np.array([0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9])

        ref_column[0], column[0], ref_row[0], row[0] = 53.5, 40.55479431152344, 281.5, 105.0828857421875
        ref_column[1], column[1], ref_row[1], row[1] = 40.5, 30.663848876953125, 102.5, 266.4919738769531
        ref_column[2], column[2], ref_row[2], row[2] = 30.5, 58.33549880981445, 269.5, 171.42666625976562
        ref_column[3], column[3], ref_row[3], row[3] = 58.5, 50.44178771972656, 172.5, 64.61354064941406
        ref_column[4], column[4], ref_row[4], row[4] = 50.5, 64.23998260498047, 64.375, 189.2677764892578
        ref_column[5], column[5], ref_row[5], row[5] = 63.5, 72.08361053466797, 188.5, 116.95103454589844
        ref_column[6], column[6], ref_row[6], row[6] = 0.00, 0.0, 0.0, 0.0
        ref_column[7], column[7], ref_row[7], row[7] = 72.5, 73.06128692626953, 113.5, 87.29881286621094
        ref_column[8], column[8], ref_row[8], row[8] = 72.8, 0.0, 87.5, 0.0
        ref_column[9], column[9], ref_row[9], row[9] = 0.0, 8.888647079467773, 0.0, 332.7086486816406
        ref_column[10], column[10], ref_row[10], row[10] = 8.5, 4.0904927253723145, 336.5, 307.4698181152344

        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr, error=5, n_bad_events=2, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix is expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertEqual(corr[0], 0)
        self.assertTrue(np.all(corr[1:11] == 1))

        # Similarity check
        self.assertEqual(column[0], 0)
        self.assertEqual(row[0], 0)
        self.assertTrue(np.all(np.abs(ref_column[1:5] - column[1:5]) < 5))
        self.assertTrue(np.all(np.abs(ref_row[1:5] - row[1:5]) < 5))
        self.assertTrue(np.all(np.abs(ref_column[7:11] - column[7:11]) < 5))
        self.assertTrue(np.all(np.abs(ref_row[7:11] - row[7:11]) < 5))

        # Virtual hits check
        self.assertEqual(column[0], 0)
        self.assertEqual(row[0], 0)
        self.assertEqual(ref_column[6], 0)
        self.assertEqual(ref_row[6], 0)

    def test_missing_events(self):  # test behavior if events are missing
        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr = get_random_data(20, hits_per_event=1)
        # Event offset = 3 and two consecutive events missing
        column[:3] = 0
        row[:3] = 0
        charge[:3] = 0
        column[3:] = ref_column[:-3]
        row[3:] = ref_row[:-3]
        charge[3:] = ref_charge[:-3]
        event_numbers = np.delete(event_numbers, [9, 10], axis=0)
        ref_column = np.delete(ref_column, [9, 10], axis=0)
        column = np.delete(column, [9, 10], axis=0)
        ref_row = np.delete(ref_row, [9, 10], axis=0)
        row = np.delete(row, [9, 10], axis=0)
        ref_charge = np.delete(ref_charge, [9, 10], axis=0)
        charge = np.delete(charge, [9, 10], axis=0)
        corr = np.delete(corr, [9, 10], axis=0)

        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr, error=0.1, n_bad_events=3, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[:-3] == 1))
        self.assertTrue(np.all(corr[-3:] == 0))

        # Similarity check
        self.assertTrue(np.all(ref_column[column != 0] == column[column != 0]))
        self.assertTrue(np.all(ref_row[column != 0] == row[column != 0]))
        self.assertTrue(np.all(ref_charge[column != 0] == charge[column != 0]))
        self.assertTrue(np.all(column[6:8] == 0))
        self.assertTrue(np.all(row[6:8] == 0))
        self.assertTrue(np.all(charge[6:8] == 0))

        # Event offset = 1, no events missing, but hits of one event missing
        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr = get_random_data(20, hits_per_event=3)
        column[:3] = 0
        row[:3] = 0
        charge[:3] = 0
        column[3:] = ref_column[:-3]
        row[3:] = ref_row[:-3]
        charge[3:] = ref_charge[:-3]
        event_numbers = np.delete(event_numbers, [9, 10], axis=0)
        ref_column = np.delete(ref_column, [9, 10], axis=0)
        column = np.delete(column, [9, 10], axis=0)
        ref_row = np.delete(ref_row, [9, 10], axis=0)
        row = np.delete(row, [9, 10], axis=0)
        ref_charge = np.delete(ref_charge, [9, 10], axis=0)
        charge = np.delete(charge, [9, 10], axis=0)
        corr = np.delete(corr, [9, 10], axis=0)

        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr, error=0.1, n_bad_events=3, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[:9] == 1))
        self.assertEqual(corr[9], 0)
        self.assertTrue(np.all(corr[10:14] == 1))
        self.assertTrue(np.all(corr[16:] == 0))

        # Similarity check
        self.assertTrue(np.all(ref_column[0:6] == column[0:6]))
        self.assertTrue(np.all(ref_row[0:6] == row[0:6]))
        self.assertTrue(np.all(ref_charge[0:6] == charge[0:6]))
        self.assertTrue(np.all(ref_column[10:15] == column[10:15]))
        self.assertTrue(np.all(ref_row[10:15] == row[10:15]))
        self.assertTrue(np.all(ref_charge[10:15] == charge[10:15]))
        self.assertTrue(np.all(column[15:] == 0))
        self.assertTrue(np.all(row[15:] == 0))
        self.assertTrue(np.all(charge[15:] == 0))

        # Event offset = 1, 1 hit events, missing hits
        event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr = get_random_data(20, hits_per_event=1)

        ref_column[5] = 0
        ref_row[5] = 0
        ref_charge[5] = 0

        ref_column[11:13] = 0
        ref_row[11:13] = 0
        ref_charge[11:13] = 0

        column[:3] = 0
        row[:3] = 0
        charge[:3] = 0
        column[3:] = ref_column[:-3]
        row[3:] = ref_row[:-3]
        charge[3:] = ref_charge[:-3]
        corr = np.ones_like(event_numbers, dtype=np.uint8)

        n_fixes = analysis_functions.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, corr, error=0.1, n_bad_events=3, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[:17] == 1))
        self.assertTrue(np.all(corr[17:] == 0))

        # Similarity check
        self.assertTrue(np.all(ref_column[corr == 1] == column[corr == 1]))
        self.assertTrue(np.all(ref_row[corr == 1] == row[corr == 1]))
        self.assertTrue(np.all(ref_charge[corr == 1] == charge[corr == 1]))

    def test_tough_test_case(self):  # test crazy uncorrelated data
        #         raise SkipTest
        event_numbers = np.array([0, 0, 2, 2, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 10], dtype=np.int64)
        ref_column = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.float)
        ref_row = ref_column
        column = np.array([11, 11, 0, 2, 5, 5, 5, 3, 0, 4, 9, 10, 12, 12, 13, 14, 15, 17], dtype=np.float)
        row = column
        ref_charge = np.zeros_like(ref_row, dtype=np.uint16)
        charge = np.zeros_like(ref_charge)

        corr, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, error=0.1, n_bad_events=2, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[:16] == 1))
        self.assertTrue(np.all(corr[17:] == 0))

        # Similarity check
        self.assertTrue(np.all(column == np.array([2, 0, 3, 4, 0, 0, 0, 9, 10, 0, 13, 14, 15, 0, 17, 0, 0, 0])))
        self.assertTrue(np.all(column == row))

        # Small but important change of test case, event 4 is copied to 2 and their are too many hits in 4 -> correlation has to be 0
        event_numbers = np.array([0, 0, 2, 2, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8, 9, 10], dtype=np.int64)
        ref_column = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.float)
        ref_row = ref_column
        column = np.array([11, 11, 0, 2, 5, 5, 5, 3, 3, 4, 9, 10, 12, 12, 13, 14, 15, 17], dtype=np.float)
        row = column
        ref_charge = np.zeros_like(ref_row, dtype=np.uint16)
        charge = np.zeros_like(ref_charge)

        corr, n_fixes = analysis_utils.fix_event_alignment(event_numbers, ref_column, column, ref_row, row, ref_charge, charge, error=0.1, n_bad_events=3, n_good_events=2, correlation_search_range=100, good_events_search_range=100)

        # one fix are expected
        self.assertEqual(n_fixes, 1)

        # Correlation flag check
        self.assertTrue(np.all(corr[0:2] == 1))
        self.assertTrue(np.all(corr[2:4] == 0))
        self.assertTrue(np.all(corr[4:16] == 1))
        self.assertTrue(np.all(corr[17:] == 0))

        # Similarity check
        self.assertTrue(np.all(column == np.array([2, 0, 3, 3, 0, 0, 0, 9, 10, 0, 13, 14, 15, 0, 17, 0, 0, 0])))
        self.assertTrue(np.all(column == row))

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCorrelationFixing)
    unittest.TextTestRunner(verbosity=2).run(suite)
