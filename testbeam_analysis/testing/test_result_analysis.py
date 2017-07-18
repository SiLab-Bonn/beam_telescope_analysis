''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os
import shutil
import unittest

from testbeam_analysis import result_analysis
from testbeam_analysis.tools import analysis_utils, test_tools

testing_path = os.path.dirname(os.path.abspath(__file__))


class TestResultAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # virtual X server for plots under headless LINUX travis testing is needed
        if os.getenv('TRAVIS', False) and os.getenv('TRAVIS_OS_NAME', False) == 'linux':
            from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        cls.output_folder = 'tmp_test_res_output'
        test_tools.create_folder(cls.output_folder)
        cls.pixel_size = [250, 50] * 4  # in um
        cls.n_pixels = [80, 336] * 4
        cls.z_positions = [0., 19500, 108800, 128300]  # in um

    @classmethod
    def tearDownClass(cls):  # remove created files
        shutil.rmtree(cls.output_folder)

    @unittest.SkipTest
    def test_residuals_calculation(self):
        residuals = result_analysis.calculate_residuals(input_tracks_file=analysis_utils.get_data('fixtures/result_analysis/Tracks_result.h5'),
                                                        input_alignment_file=analysis_utils.get_data('fixtures/result_analysis/Alignment_result.h5'),
                                                        output_residuals_file=os.path.join(self.output_folder, 'Residuals.h5'),
                                                        n_pixels=self.n_pixels,
                                                        pixel_size=self.pixel_size)
        # Only test row residuals, columns are too large (250 um) for meaningfull gaussian residuals distribution
        self.assertAlmostEqual(residuals[1], 22.9135, msg='DUT 0 row residuals do not match', places=3)
        self.assertAlmostEqual(residuals[3], 18.7317, msg='DUT 1 row residuals do not match', places=3)
        self.assertAlmostEqual(residuals[5], 22.8645, msg='DUT 2 row residuals do not match', places=3)
        self.assertAlmostEqual(residuals[7], 27.2816, msg='DUT 3 row residuals do not match', places=3)

    @unittest.SkipTest
    def test_efficiency_calculation(self):
        efficiencies = result_analysis.calculate_efficiency(input_tracks_file=analysis_utils.get_data('fixtures/result_analysis/Tracks_result.h5'),
                                                            input_alignment_file=analysis_utils.get_data('fixtures/result_analysis/Alignment_result.h5'),
                                                            output_efficiency_file=os.path.join(self.output_folder, 'Efficiency.h5'),
                                                            bin_size=[(250, 50)]*4,
                                                            sensor_size=[(250 * 80., 336 * 50.)]*4,
                                                            pixel_size=[(250, 50)]*4,
                                                            n_pixels=[(80, 336)]*4,
                                                            minimum_track_density=2,
                                                            use_duts=None,
                                                            cut_distance=500,
                                                            max_distance=500,
                                                            #col_range=[(1250, 17500)]*4,
                                                            #row_range=[(1000, 16000)]*4,
                                                            force_prealignment=True)

        self.assertAlmostEqual(efficiencies[0], 100.000, msg='DUT 0 efficiencies do not match', places=3)
        self.assertAlmostEqual(efficiencies[1], 98.7013, msg='DUT 1 efficiencies do not match', places=3)
        self.assertAlmostEqual(efficiencies[2], 97.4684, msg='DUT 2 efficiencies do not match', places=3)
        self.assertAlmostEqual(efficiencies[3], 100.000, msg='DUT 3 efficiencies do not match', places=3)

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestResultAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
