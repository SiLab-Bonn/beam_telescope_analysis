''' Script to check that the examples run.

    The example data is reduced at the beginning to safe time.
'''
import os
import unittest
import logging
import shutil
import inspect

import beam_telescope_analysis
from beam_telescope_analysis.tools import analysis_utils, test_tools, data_selection
from beam_telescope_analysis.examples import eutelescope, eutelescope_kalman, fei4_telescope


class TestExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tests_data_folder = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tests_data_folder)

    def test_eutelescope_example(self):
        # The location of the data files, one file per DUT
        hit_files = [analysis_utils.get_data(
            path='examples/TestBeamData_Mimosa26_DUT%d.h5' % i,
            output=os.path.join(self.tests_data_folder, 'TestBeamData_Mimosa26_DUT%d.h5' % i)) for i in range(6)]

        # Reduce, remove old files and rename reduced files
        for file_name in hit_files:
            data_selection.reduce_events(file_name, max_events=70000)
            os.remove(file_name)
            os.rename(os.path.splitext(file_name)[0] + '_reduced.h5',
                      file_name)

        eutelescope.run_analysis(hit_files=hit_files)

    def test_eutelescope_kalman_example(self):
        # The location of the data files, one file per DUT
        hit_files = [analysis_utils.get_data(
            path='examples/TestBeamData_Mimosa26_DUT%d.h5' % i,
            output=os.path.join(self.tests_data_folder, 'TestBeamData_Mimosa26_DUT%d.h5' % i)) for i in range(6)]

        # Reduce, remove old files and rename reduced files
        for file_name in hit_files:
            data_selection.reduce_events(file_name, max_events=70000)
            os.remove(file_name)
            os.rename(os.path.splitext(file_name)[0] + '_reduced.h5',
                      file_name)

        eutelescope_kalman.run_analysis(hit_files=hit_files)

    def test_fei4_telescope_example(self):
        # The location of the data files, one file per DUT
        hit_files = [analysis_utils.get_data(
            path='examples/TestBeamData_FEI4_DUT%d.h5' % i,
            output=os.path.join(self.tests_data_folder, 'TestBeamData_FEI4_DUT%d.h5' % i)) for i in [0, 1, 4, 5]]

        fei4_telescope.run_analysis(hit_files=hit_files)


if __name__ == '__main__':
    unittest.main()
