''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os
import shutil
import unittest

from beam_telescope_analysis import track_analysis
from beam_telescope_analysis.tools import analysis_utils, test_tools

testing_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.abspath(os.path.join(testing_path, 'fixtures'))
aligned_configuration = os.path.join(data_folder, "telescope_aligned.yaml")


class TestTrackAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # virtual X server for plots under headless LINUX travis testing is needed
        if os.getenv('TRAVIS', False) and os.getenv('TRAVIS_OS_NAME', False) == 'linux':
            from xvfbwrapper import Xvfb
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        cls.output_folder = 'tmp_track_test_output'
        test_tools.create_folder(cls.output_folder)

    @classmethod
    def tearDownClass(cls):  # Remove created files
        shutil.rmtree(cls.output_folder)

    def test_track_finding(self):
        # Test 1:
        track_analysis.find_tracks(
            telescope_configuration=aligned_configuration,
            input_merged_file=os.path.join(data_folder, "Merged_small.h5"),
            output_track_candidates_file=os.path.join(self.output_folder, 'TrackCandidates.h5'),
            align_to_beam=True)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, "TrackCandidates_result.h5"), os.path.join(self.output_folder, 'TrackCandidates.h5'), ignore_nodes='/arguments/find_tracks')
        self.assertTrue(data_equal, msg=error_msg)

        # Test 2: chunked
        track_analysis.find_tracks(
            telescope_configuration=aligned_configuration,
            input_merged_file=os.path.join(data_folder, "Merged_small.h5"),
            output_track_candidates_file=os.path.join(self.output_folder, 'TrackCandidates_2.h5'),
            align_to_beam=True,
            chunk_size=293)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, "TrackCandidates_result.h5"), os.path.join(self.output_folder, 'TrackCandidates_2.h5'), ignore_nodes='/arguments/find_tracks')
        self.assertTrue(data_equal, msg=error_msg)

    def test_track_fitting(self):
        # Test 1: Fit DUTs and always exclude one DUT (normal mode for unbiased residuals and efficiency determination)
        track_analysis.fit_tracks(
            telescope_configuration=aligned_configuration,
            input_track_candidates_file=os.path.join(self.output_folder, 'TrackCandidates.h5'),
            output_tracks_file=os.path.join(self.output_folder, 'Tracks.h5'),
            quality_distances=[(18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2)],
            isolation_distances=(1000.0, 1000.0))
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, "Tracks_result.h5"), os.path.join(self.output_folder, 'Tracks.h5'), ignore_nodes='/arguments/fit_tracks', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

        # Test 2: As test 1 but chunked data analysis, should result in the same tracks
        track_analysis.fit_tracks(
            telescope_configuration=aligned_configuration,
            input_track_candidates_file=os.path.join(self.output_folder, 'TrackCandidates.h5'),
            output_tracks_file=os.path.join(self.output_folder, 'Tracks_2.h5'),
            quality_distances=[(18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2)],
            isolation_distances=(1000.0, 1000.0),
            chunk_size=4999)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, "Tracks_result.h5"), os.path.join(self.output_folder, 'Tracks_2.h5'), ignore_nodes='/arguments/fit_tracks', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

        # Test 3: Fit all DUTs at once (special mode for constrained residuals)
        track_analysis.fit_tracks(
            telescope_configuration=aligned_configuration,
            input_track_candidates_file=os.path.join(self.output_folder, 'TrackCandidates.h5'),
            output_tracks_file=os.path.join(self.output_folder, 'Tracks_biased.h5'),
            quality_distances=[(18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2)],
            isolation_distances=(1000.0, 1000.0),
            exclude_dut_hit=False)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, "Tracks_biased_result.h5"), os.path.join(self.output_folder, 'Tracks_biased.h5'), ignore_nodes='/arguments/fit_tracks', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

        # Test 4: Fit DUTs consecutevly, but use always the same DUTs. Should result in the same data as above
        track_analysis.fit_tracks(
            telescope_configuration=aligned_configuration,
            input_track_candidates_file=os.path.join(self.output_folder, 'TrackCandidates.h5'),
            output_tracks_file=os.path.join(self.output_folder, 'Tracks_All_Iter.h5'),
            select_hit_duts=range(6),
            quality_distances=[(18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2)],
            isolation_distances=(1000.0, 1000.0),
            exclude_dut_hit=False)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(self.output_folder, "Tracks_biased.h5"), os.path.join(self.output_folder, 'Tracks_All_Iter.h5'), ignore_nodes='/arguments/fit_tracks', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

        # Test 5: Fit DUTs consecutevly, but use always the same DUTs defined for each DUT separately. Should result in the same data as above
        track_analysis.fit_tracks(
            telescope_configuration=aligned_configuration,
            input_track_candidates_file=os.path.join(self.output_folder, 'TrackCandidates.h5'),
            output_tracks_file=os.path.join(self.output_folder, 'Tracks_All_Iter_2.h5'),
            select_hit_duts=[range(6)] * 6,
            quality_distances=[(18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2)],
            isolation_distances=(1000.0, 1000.0),
            exclude_dut_hit=False)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(self.output_folder, 'Tracks_biased.h5'), os.path.join(self.output_folder, 'Tracks_All_Iter_2.h5'), ignore_nodes='/arguments/fit_tracks', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

        # Test 6 :Fit DUTs using Kalman Filter
        track_analysis.fit_tracks(
            telescope_configuration=aligned_configuration,
            input_track_candidates_file=os.path.join(self.output_folder, 'TrackCandidates.h5'),
            output_tracks_file=os.path.join(self.output_folder, 'Tracks_kalman.h5'),
            quality_distances=[(18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2)],
            isolation_distances=(1000.0, 1000.0),
            method='kalman',
            beam_energy=5000.0,
            particle_mass=0.511)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, "Tracks_kalman_result.h5"), os.path.join(self.output_folder, 'Tracks_kalman.h5'), ignore_nodes='/arguments/fit_tracks', exact=False)
        self.assertTrue(data_equal, msg=error_msg)

        # Test 7: Fit DUTs consecutevly, but specify min. number of track hits
        track_analysis.fit_tracks(
            telescope_configuration=aligned_configuration,
            input_track_candidates_file=os.path.join(self.output_folder, 'TrackCandidates.h5'),
            output_tracks_file=os.path.join(self.output_folder, 'Tracks_min_track_hits.h5'),
            min_track_hits=4)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(data_folder, "Tracks_min_track_hits_result.h5"), os.path.join(self.output_folder, 'Tracks_min_track_hits.h5'), ignore_nodes='/arguments/fit_tracks', exact=False)
        self.assertTrue(data_equal, msg=error_msg)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrackAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
