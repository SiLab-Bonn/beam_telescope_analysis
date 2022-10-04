''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os
import shutil
import unittest

import numpy as np

from beam_telescope_analysis import dut_alignment, hit_analysis
from beam_telescope_analysis.tools import test_tools
from beam_telescope_analysis.tools import geometry_utils
from beam_telescope_analysis.tools import analysis_utils

testing_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(testing_path, 'fixtures')
initial_configuration = os.path.join(data_folder, "telescope.yaml")
prealigned_configuration = os.path.join(data_folder, "telescope_prealigned.yaml")


class TestAlignmentAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # virtual X server for plots under headless LINUX travis testing is needed
        if os.getenv('TRAVIS', False) and os.getenv('TRAVIS_OS_NAME', False) == 'linux':
            from xvfbwrapper import Xvfb
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        # Define test data input files, download if needed
        cls.data_files = [os.path.join(data_folder, 'TestBeamData_Mimosa26_DUT%i_clustered.h5' % i)
                          for i in range(6)]
        # Define and create tests output folder, is deleted at the end of tests
        cls.output_folder = 'tmp_alignment_test_output'
        test_tools.create_folder(cls.output_folder)

    @classmethod
    def tearDownClass(cls):  # remove created files
        try:
            for i in range(3):
                os.remove(os.path.join(data_folder, 'Merged_small_residuals_aligned_selected_tracks_0_1_2_3_4_5_tmp_%i.pdf' % i))
                os.remove(os.path.join(data_folder, 'Merged_small_tracks_aligned_duts_0_1_2_3_4_5_tmp_%i.pdf' % i))
                os.remove(os.path.join(data_folder, 'Merged_small_tracks_aligned_duts_0_1_2_3_4_5_tmp_%i_chi2.pdf' % i))
                os.remove(os.path.join(data_folder, 'Merged_small_tracks_angles_aligned_selected_tracks_duts_0_1_2_3_4_5_tmp_%i.pdf' % i))
            os.remove(os.path.join(data_folder, 'Correlation_result_prealigned.pdf'))
        except FileNotFoundError:
            pass
        shutil.rmtree(cls.output_folder)

    def test_cluster_correlation(self):  # Check the cluster correlation function
        hit_analysis.correlate(
            telescope_configuration=initial_configuration,
            input_files=self.data_files,
            output_correlation_file=os.path.join(self.output_folder, 'Correlation.h5'),
            resolution=(100.0, 100.0))

        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'Correlation_result.h5'), os.path.join(self.output_folder, 'Correlation.h5'), exact=True, ignore_nodes='/arguments/correlate')
        self.assertTrue(data_equal, msg=error_msg)

        # Retest with tiny chunk size to force chunked correlation
        hit_analysis.correlate(
            telescope_configuration=initial_configuration,
            input_files=self.data_files,
            output_correlation_file=os.path.join(self.output_folder, 'Correlation_2.h5'),
            resolution=(100.0, 100.0),
            chunk_size=293)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'Correlation_result.h5'), os.path.join(self.output_folder, 'Correlation_2.h5'), exact=True, ignore_nodes='/arguments/correlate')
        self.assertTrue(data_equal, msg=error_msg)

    def test_prealignment(self):  # Check the hit alignment function
        prealigned_configuration = dut_alignment.prealign(
            telescope_configuration=initial_configuration,
            output_telescope_configuration=os.path.join(self.output_folder, 'telescope_prealigned.yaml'),
            input_correlation_file=os.path.join(data_folder, 'Correlation_result.h5'),
            reduce_background=True)

        data_equal = test_tools.compare_yaml_files(os.path.join(data_folder, 'telescope_prealigned.yaml'), os.path.join(self.output_folder, 'telescope_prealigned.yaml'))
        self.assertTrue(data_equal)

    def test_cluster_merging(self):
        hit_analysis.merge_cluster_data(
            telescope_configuration=initial_configuration,
            input_cluster_files=self.data_files,
            output_merged_file=os.path.join(self.output_folder, 'Merged.h5'))

        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'Merged_result.h5'), os.path.join(self.output_folder, 'Merged.h5'), ignore_nodes="/arguments/merge_cluster_data")
        self.assertTrue(data_equal, msg=error_msg)

        # Retest with tiny chunk size to force chunked merging
        hit_analysis.merge_cluster_data(
            telescope_configuration=initial_configuration,
            input_cluster_files=self.data_files,
            output_merged_file=os.path.join(self.output_folder, 'Merged_2.h5'),
            chunk_size=293)

        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'Merged_result.h5'), os.path.join(self.output_folder, 'Merged_2.h5'), ignore_nodes="/arguments/merge_cluster_data")
        self.assertTrue(data_equal, msg=error_msg)

    def test_alignment(self):
        dut_alignment.align(
            telescope_configuration=prealigned_configuration,
            output_telescope_configuration=os.path.join(self.output_folder, 'telescope_aligned.yaml'),
            input_merged_file=os.path.join(data_folder, 'Merged_small.h5'),
            select_duts=[[0, 1, 2, 3, 4, 5]],
            select_telescope_duts=[0, 1, 2, 3, 4, 5],
            select_fit_duts=[[0, 1, 2, 3, 4, 5]],
            select_hit_duts=[[0, 1, 2, 3, 4, 5]],
            max_iterations=[3],
            track_chi2=15.0,
            quality_distances=[(18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2)],
            isolation_distances=(1000.0, 1000.0),
            use_limits=True,
            plot=True)

        # TODO: Alignment is slighlty different on CI runner. Convert yaml to .h5 file and test values with tolerance.
        # data_equal = test_tools.compare_yaml_files(os.path.join(data_folder, 'telescope_aligned.yaml'), os.path.join(self.output_folder, 'telescope_aligned.yaml'))
        # self.assertTrue(data_equal)

    def test_kalman_alignment(self):
        # Step 1
        aligned_configuration_kf = dut_alignment.align_kalman(
            telescope_configuration=prealigned_configuration,
            input_merged_file=os.path.join(data_folder, 'Merged_small.h5'),
            output_telescope_configuration=os.path.join(self.output_folder, 'telescope_aligned_kalman.yaml'),
            output_alignment_file=os.path.join(self.output_folder, 'Merged_KFA_alignment_1.h5'),
            select_duts=[[0, 1, 2, 3, 4, 5]],  # align the telescope planes first
            select_telescope_duts=[0, 5],  # telescope planes
            select_fit_duts=[[0, 1, 2, 3, 4, 5]],
            select_hit_duts=[[0, 1, 2, 3, 4, 5]],
            alignment_parameters=[[["translation_x", "translation_y", "rotation_gamma"]] * 6],
            alignment_parameters_errors=[200.0, 200.0, 2000.0, 50e-3, 50e-3, 50e-3],
            track_chi2=[5],
            beam_energy=5000.0,
            particle_mass=0.511,
            annealing_factor=10000,
            annealing_tracks=5000,
            max_tracks=10000,
            plot=True)
        # Step 2
        aligned_configuration = dut_alignment.align_kalman(
            telescope_configuration=aligned_configuration_kf,
            input_merged_file=os.path.join(data_folder, 'Merged_small.h5'),
            output_telescope_configuration=os.path.join(self.output_folder, 'telescope_aligned_kalman_2.yaml'),
            output_alignment_file=os.path.join(self.output_folder, 'Merged_KFA_alignment_2.h5'),
            select_duts=[[0, 1, 2, 3, 4, 5]],  # align the telescope planes first
            select_telescope_duts=[0, 5],  # telescope planes
            select_fit_duts=[[0, 1, 2, 3, 4, 5]],
            select_hit_duts=[[0, 1, 2, 3, 4, 5]],
            alignment_parameters=[[["translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma"]] * 6],
            alignment_parameters_errors=[200.0 / 10.0, 200.0 / 10.0, 2000.0, 50e-3, 50e-3, 50e-3 / 100.0],
            track_chi2=[5],
            beam_energy=5000.0,
            particle_mass=0.511,
            annealing_factor=10000,
            annealing_tracks=5000,
            max_tracks=10000,
            plot=True)

        # TODO: Alignment is slighlty different on CI runner. Convert yaml to .h5 file and test values with tolerance.
        # data_equal = test_tools.compare_yaml_files(os.path.join(data_folder, 'telescope_aligned_kalman.yaml'), os.path.join(self.output_folder, 'telescope_aligned_kalman.yaml'))
        # self.assertTrue(data_equal)

        # data_equal = test_tools.compare_yaml_files(os.path.join(data_folder, 'telescope_aligned_kalman_2.yaml'), os.path.join(self.output_folder, 'telescope_aligned_kalman_2.yaml'))
        # self.assertTrue(data_equal)

        # data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'Merged_KFA_alignment_1_result.h5'), os.path.join(self.output_folder, 'Merged_KFA_alignment_1.h5'))
        # self.assertTrue(data_equal, msg=error_msg)

        # data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'Merged_KFA_alignment_2_result.h5'), os.path.join(self.output_folder, 'Merged_KFA_alignment_2.h5'))
        # self.assertTrue(data_equal, msg=error_msg)


if __name__ == '__main__':
    unittest.main()
