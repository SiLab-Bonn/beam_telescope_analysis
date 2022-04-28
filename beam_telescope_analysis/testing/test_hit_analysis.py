''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os
import shutil
import unittest

from beam_telescope_analysis import hit_analysis
from beam_telescope_analysis.tools import analysis_utils, test_tools

testing_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(testing_path, 'fixtures')
initial_configuration = os.path.join(data_folder, "telescope.yaml")


class TestHitAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # virtual X server for plots under headless LINUX travis testing is needed
        if os.getenv('TRAVIS', False) and os.getenv('TRAVIS_OS_NAME', False) == 'linux':
            from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()

        cls.big_noisy_data_file = os.path.join(data_folder, 'TestBeamData_Mimosa26_DUT0.h5')
        cls.output_folder = 'tmp_hit_test_output'
        test_tools.create_folder(cls.output_folder)

    @classmethod
    def tearDownClass(cls):  # remove created files
        shutil.rmtree(cls.output_folder)

    def test_generate_pixel_mask(self):
        # Generate noisy pixel mask for all DUTs
        thresholds = [2]
        # last plane has noisy cluster, use larger median filter to mask cluster
        pixel_mask_names = ["NoisyPixelMask"]
        mask_files = hit_analysis.mask(
            telescope_configuration=initial_configuration,
            input_hit_files=[self.big_noisy_data_file],
            output_mask_files=[os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_noisy_pixel_mask.h5')],
            select_duts=[0],
            pixel_mask_names=pixel_mask_names,
            thresholds=thresholds)

        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'TestBeamData_Mimosa26_DUT0_noisy_pixel_mask.h5'), mask_files[0], ignore_nodes=['/arguments/mask', '/arguments/mask_pixels'])
        self.assertTrue(data_equal, msg=error_msg)

    def test_noisy_pixel_masking(self):
        # Test 1:
        # Generate noisy pixel mask for DUT
        thresholds = [2]
        # last plane has noisy cluster, use larger median filter to mask cluster
        pixel_mask_names = ["NoisyPixelMask"]
        mask_files = hit_analysis.mask(
            telescope_configuration=initial_configuration,
            input_hit_files=[self.big_noisy_data_file],
            output_mask_files=[os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_noisy_pixel_mask.h5')],
            select_duts=[0],
            pixel_mask_names=pixel_mask_names,
            thresholds=thresholds)

        # Cluster hits from all DUTs
        use_positions = [False]
        min_hit_charges = [1]
        max_hit_charges = [1]
        column_cluster_distances = [3]
        row_cluster_distances = [3]
        frame_cluster_distances = [0]
        output_cluster_files = hit_analysis.cluster(
            telescope_configuration=initial_configuration,
            select_duts=[0],
            input_hit_files=[self.big_noisy_data_file],
            output_cluster_files=[os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_noisy_pixels_clustered.h5')],
            input_mask_files=[mask_files[0]],
            use_positions=use_positions,
            min_hit_charges=min_hit_charges,
            max_hit_charges=max_hit_charges,
            column_cluster_distances=column_cluster_distances,
            row_cluster_distances=row_cluster_distances,
            frame_cluster_distances=frame_cluster_distances)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'TestBeamData_Mimosa26_DUT0_noisy_pixels_clustered_result.h5'), output_cluster_files[0], ignore_nodes='/arguments/cluster', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

        # Test 2: smaller chunks
        # Generate noisy pixel mask for DUT
        thresholds = [2]
        # last plane has noisy cluster, use larger median filter to mask cluster
        pixel_mask_names = ["NoisyPixelMask"]
        mask_files = hit_analysis.mask(
            telescope_configuration=initial_configuration,
            input_hit_files=[self.big_noisy_data_file],
            output_mask_files=[os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_noisy_pixel_mask.h5')],
            select_duts=[0],
            pixel_mask_names=pixel_mask_names,
            thresholds=thresholds)

        # Cluster hits from all DUTs
        use_positions = [False]
        min_hit_charges = [1]
        max_hit_charges = [1]
        column_cluster_distances = [3]
        row_cluster_distances = [3]
        frame_cluster_distances = [0]
        output_cluster_files = hit_analysis.cluster(
            telescope_configuration=initial_configuration,
            select_duts=[0],
            input_hit_files=[self.big_noisy_data_file],
            output_cluster_files=[os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_noisy_pixels_clustered.h5')],
            input_mask_files=[mask_files[0]],
            use_positions=use_positions,
            min_hit_charges=min_hit_charges,
            max_hit_charges=max_hit_charges,
            column_cluster_distances=column_cluster_distances,
            row_cluster_distances=row_cluster_distances,
            frame_cluster_distances=frame_cluster_distances,
            chunk_size=4999)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'TestBeamData_Mimosa26_DUT0_noisy_pixels_clustered_result.h5'), output_cluster_files[0], ignore_nodes='/arguments/cluster', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

    def test_noisy_pixel_remover(self):
        # Test 1:
        # Generate noisy pixel mask for DUT
        thresholds = [2]
        # last plane has noisy cluster, use larger median filter to mask cluster
        pixel_mask_names = ["DisabledPixelMask"]
        mask_files = hit_analysis.mask(
            telescope_configuration=initial_configuration,
            input_hit_files=[self.big_noisy_data_file],
            output_mask_files=[os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_noisy_pixel_mask.h5')],
            select_duts=[0],
            pixel_mask_names=pixel_mask_names,
            thresholds=thresholds)

        # Cluster hits from all DUTs
        use_positions = [False]
        min_hit_charges = [1]
        max_hit_charges = [1]
        column_cluster_distances = [3]
        row_cluster_distances = [3]
        frame_cluster_distances = [0]
        output_cluster_files = hit_analysis.cluster(
            telescope_configuration=initial_configuration,
            select_duts=[0],
            input_hit_files=[self.big_noisy_data_file],
            output_cluster_files=[os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_disabled_pixels_clustered.h5')],
            input_mask_files=[mask_files[0]],
            use_positions=use_positions,
            min_hit_charges=min_hit_charges,
            max_hit_charges=max_hit_charges,
            column_cluster_distances=column_cluster_distances,
            row_cluster_distances=row_cluster_distances,
            frame_cluster_distances=frame_cluster_distances)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'TestBeamData_Mimosa26_DUT0_disabled_pixels_clustered_result.h5'), output_cluster_files[0], ignore_nodes='/arguments/cluster', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

        # Test 2: smaller chunks
        # Generate noisy pixel mask for DUT
        thresholds = [2]
        # last plane has noisy cluster, use larger median filter to mask cluster
        pixel_mask_names = ["DisabledPixelMask"]
        mask_files = hit_analysis.mask(
            telescope_configuration=initial_configuration,
            input_hit_files=[self.big_noisy_data_file],
            output_mask_files=[os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_noisy_pixel_mask.h5')],
            select_duts=[0],
            pixel_mask_names=pixel_mask_names,
            thresholds=thresholds)

        # Cluster hits from all DUTs
        use_positions = [False]
        min_hit_charges = [1]
        max_hit_charges = [1]
        column_cluster_distances = [3]
        row_cluster_distances = [3]
        frame_cluster_distances = [0]
        output_cluster_files = hit_analysis.cluster(
            telescope_configuration=initial_configuration,
            select_duts=[0],
            input_hit_files=[self.big_noisy_data_file],
            output_cluster_files=[os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_disabled_pixels_clustered.h5')],
            input_mask_files=[mask_files[0]],
            use_positions=use_positions,
            min_hit_charges=min_hit_charges,
            max_hit_charges=max_hit_charges,
            column_cluster_distances=column_cluster_distances,
            row_cluster_distances=row_cluster_distances,
            frame_cluster_distances=frame_cluster_distances,
            chunk_size=4999)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'TestBeamData_Mimosa26_DUT0_disabled_pixels_clustered_result.h5'), output_cluster_files[0], ignore_nodes='/arguments/cluster', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

    def test_hit_clustering(self):
        # Test 1:
        # Cluster hits from all DUTs
        use_positions = [False]
        min_hit_charges = [1]
        max_hit_charges = [1]
        column_cluster_distances = [3]
        row_cluster_distances = [3]
        frame_cluster_distances = [0]
        output_cluster_files = hit_analysis.cluster(
            telescope_configuration=initial_configuration,
            select_duts=[0],
            input_hit_files=[self.big_noisy_data_file],
            output_cluster_files=[os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_clustered.h5')],
            use_positions=use_positions,
            min_hit_charges=min_hit_charges,
            max_hit_charges=max_hit_charges,
            column_cluster_distances=column_cluster_distances,
            row_cluster_distances=row_cluster_distances,
            frame_cluster_distances=frame_cluster_distances)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'TestBeamData_Mimosa26_DUT0_clustered_result.h5'), output_cluster_files[0], ignore_nodes='/arguments/cluster', exact=False)
        self.assertTrue(data_equal, msg=error_msg)
        # Test 2: smaller chunks
        # Cluster hits from all DUTs
        use_positions = [False]
        min_hit_charges = [1]
        max_hit_charges = [1]
        column_cluster_distances = [3]
        row_cluster_distances = [3]
        frame_cluster_distances = [0]
        output_cluster_files = hit_analysis.cluster(
            telescope_configuration=initial_configuration,
            select_duts=[0],
            input_hit_files=[self.big_noisy_data_file],
            output_cluster_files=[os.path.join(self.output_folder, 'TestBeamData_Mimosa26_DUT0_clustered.h5')],
            use_positions=use_positions,
            min_hit_charges=min_hit_charges,
            max_hit_charges=max_hit_charges,
            column_cluster_distances=column_cluster_distances,
            row_cluster_distances=row_cluster_distances,
            frame_cluster_distances=frame_cluster_distances,
            chunk_size=4999)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, 'TestBeamData_Mimosa26_DUT0_clustered_result.h5'), output_cluster_files[0], ignore_nodes='/arguments/cluster', exact=False)
        self.assertTrue(data_equal, msg=error_msg)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHitAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
