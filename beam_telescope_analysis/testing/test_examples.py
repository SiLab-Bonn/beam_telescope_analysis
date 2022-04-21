''' Script to check that the examples run.

    The example data is reduced at the beginning to safe time.
'''
import os
import unittest
import mock
import logging
from shutil import copyfile
import tables as tb
import inspect

import beam_telescope_analysis
from beam_telescope_analysis.tools import analysis_utils, test_tools
from beam_telescope_analysis.examples import eutelescope, eutelescope_kalman

testing_path = os.path.dirname(os.path.abspath(__file__))


class TestExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        # TODO: remove files again
        pass

    def test_eutelescope_example(self):
        # Get the absolute path of example data
        tests_data_folder = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')
        print(tests_data_folder)
        # The location of the data files, one file per DUT
        hit_files = [analysis_utils.get_data(
            path='examples/TestBeamData_Mimosa26_DUT%d.h5' % i,
            output=os.path.join(tests_data_folder, 'TestBeamData_Mimosa26_DUT%d.h5' % i)) for i in range(6)]

        eutelescope.run_analysis(hit_files=hit_files)

    def test_eutelescope_kalman_example(self):
        # Get the absolute path of example data
        tests_data_folder = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')
        print(tests_data_folder)
        # The location of the data files, one file per DUT
        hit_files = [analysis_utils.get_data(
            path='examples/TestBeamData_Mimosa26_DUT%d.h5' % i,
            output=os.path.join(tests_data_folder, 'TestBeamData_Mimosa26_DUT%d.h5' % i)) for i in range(6)]

        eutelescope_kalman.run_analysis(hit_files=hit_files)



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExamples)
    unittest.TextTestRunner(verbosity=2).run(suite)
