''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os
import shutil
import unittest

from beam_telescope_analysis import result_analysis
from beam_telescope_analysis.tools import analysis_utils, test_tools, data_selection

testing_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(testing_path, 'fixtures')
aligned_configuration = os.path.join(data_folder, "telescope_aligned.yaml")


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

    @classmethod
    def tearDownClass(cls):  # remove created files
        shutil.rmtree(cls.output_folder)

    def test_residuals_calculation(self):
        # Select good tracks first
        data_selection.select_tracks(
            telescope_configuration=aligned_configuration,
            input_tracks_file=os.path.join(data_folder, 'Tracks_result.h5'),
            output_tracks_file=os.path.join(self.output_folder, 'Tracks_selected.h5'),
            select_duts=[0, 1, 2, 3, 4, 5],
            select_hit_duts=[0, 1, 2, 3, 4, 5],
            select_no_hit_duts=None,
            select_quality_duts=[[1, 2, 3, 4, 5], [0, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 1, 2, 4, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 4]],
            query='(track_chi2 < 15.0)')

        residuals = result_analysis.calculate_residuals(
            telescope_configuration=aligned_configuration,
            input_tracks_file=os.path.join(self.output_folder, 'Tracks_selected.h5'),
            output_residuals_file=os.path.join(self.output_folder, 'Residuals.h5'),
            select_duts=range(6),
            use_limits=True,
            nbins_per_pixel=20)

        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'Residuals_result.h5'), os.path.join(self.output_folder, 'Residuals.h5'))
        self.assertTrue(data_equal, msg=error_msg)

    def test_efficiency_calculation(self):
        # Test 1: Calculate efficiency
        result_analysis.calculate_efficiency(
            telescope_configuration=aligned_configuration,
            input_tracks_file=os.path.join(self.output_folder, 'Tracks_selected.h5'),
            output_efficiency_file=os.path.join(self.output_folder, 'Efficiency.h5'),
            select_duts=[3],
            resolutions=(18.4, 18.4),
            extend_areas=(2000, 2000),
            cut_distances=(25.0, 25.0))

        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'Efficiency_result.h5'), os.path.join(self.output_folder, 'Efficiency.h5'), ignore_nodes="/arguments/calculate_efficiency")
        self.assertTrue(data_equal, msg=error_msg)

        # Test 2: Calculate efficiency and define several regions for which efficiency is calculated
        result_analysis.calculate_efficiency(
            telescope_configuration=aligned_configuration,
            input_tracks_file=os.path.join(self.output_folder, 'Tracks_selected.h5'),
            output_efficiency_file=os.path.join(self.output_folder, 'Efficiency_regions.h5'),
            select_duts=[3],
            resolutions=(18.4, 18.4),
            extend_areas=(2000, 2000),
            efficiency_regions=[[[[-4000, 0], [-4000, 4000]],
                                [[2000, 7000], [-4000, 4000]]]],
            cut_distances=(25.0, 25.0),
            plot=False)

        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'Efficiency_regions_result.h5'), os.path.join(self.output_folder, 'Efficiency_regions.h5'))
        self.assertTrue(data_equal, msg=error_msg)

        # Test 3: Calculate efficiency using small chunks
        result_analysis.calculate_efficiency(
            telescope_configuration=aligned_configuration,
            input_tracks_file=os.path.join(self.output_folder, 'Tracks_selected.h5'),
            output_efficiency_file=os.path.join(self.output_folder, 'Efficiency_2.h5'),
            select_duts=[3],
            resolutions=(18.4, 18.4),
            extend_areas=(2000, 2000),
            cut_distances=(25.0, 25.0),
            chunk_size=4999,
            plot=False)

        # Exlcude these node since calculation of edges for these histogram is based on data and therefore (slightly) varies with chunk size.
        ignore_nodes = ["/arguments/calculate_efficiency",
                        "/DUT3/count_1d_alpha_angle_hist",
                        "/DUT3/count_1d_alpha_angle_hist_edges",
                        "/DUT3/count_1d_beta_angle_hist",
                        "/DUT3/count_1d_beta_angle_hist_edges",
                        "/DUT3/count_1d_total_angle_hist",
                        "/DUT3/count_1d_total_angle_hist_edges"
                        ]

        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'Efficiency_result.h5'), os.path.join(self.output_folder, 'Efficiency_2.h5'), ignore_nodes=ignore_nodes, exact=False)
        self.assertTrue(data_equal, msg=error_msg)


if __name__ == '__main__':
    unittest.main()
