''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os
import shutil
import unittest

from testbeam_analysis import track_analysis
from testbeam_analysis.tools import analysis_utils, test_tools

testing_path = os.path.dirname(os.path.abspath(__file__))


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
        cls.pixel_size = (250, 50)  # in um

    @classmethod
    def tearDownClass(cls):  # Remove created files
        shutil.rmtree(cls.output_folder)

    def test_track_finding(self):
        # Test 1:
        track_analysis.find_tracks(input_tracklets_file=analysis_utils.get_data('fixtures/track_analysis/Tracklets_small.h5',
                                                                                output=os.path.join(testing_path,
                                                                                                    'fixtures/track_analysis/Tracklets_small.h5')),
                                   input_alignment_file=analysis_utils.get_data('fixtures/track_analysis/Alignment_result.h5',
                                                                                output=os.path.join(testing_path,
                                                                                                    'fixtures/track_analysis/Alignment_result.h5')),
                                   output_track_candidates_file=os.path.join(self.output_folder, 'TrackCandidates.h5'))
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/track_analysis/TrackCandidates_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/track_analysis/TrackCandidates_result.h5')), os.path.join(self.output_folder, 'TrackCandidates.h5'))
        self.assertTrue(data_equal, msg=error_msg)
        # Test 2: chunked
        track_analysis.find_tracks(input_tracklets_file=analysis_utils.get_data('fixtures/track_analysis/Tracklets_small.h5',
                                                                                output=os.path.join(testing_path,
                                                                                                    'fixtures/track_analysis/Tracklets_small.h5')),
                                   input_alignment_file=analysis_utils.get_data('fixtures/track_analysis/Alignment_result.h5',
                                                                                output=os.path.join(testing_path,
                                                                                                    'fixtures/track_analysis/Alignment_result.h5')),
                                   output_track_candidates_file=os.path.join(self.output_folder, 'TrackCandidates_2.h5'),
                                   chunk_size=293)
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/track_analysis/TrackCandidates_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/track_analysis/TrackCandidates_result.h5')),
                                                            os.path.join(self.output_folder, 'TrackCandidates_2.h5'))
        self.assertTrue(data_equal, msg=error_msg)

    def test_track_fitting(self):
        # Test 1: Fit DUTs and always exclude one DUT (normal mode for unbiased residuals and efficiency determination)
        track_analysis.fit_tracks(input_track_candidates_file=analysis_utils.get_data('fixtures/track_analysis/TrackCandidates_result.h5',
                                                                                      output=os.path.join(testing_path,
                                                                                                          'fixtures/track_analysis/TrackCandidates_result.h5')),
                                  input_alignment_file=analysis_utils.get_data('fixtures/track_analysis/Alignment_result.h5',
                                                                               output=os.path.join(testing_path,
                                                                                                   'fixtures/track_analysis/Alignment_result.h5')),
                                  output_tracks_file=os.path.join(self.output_folder, 'Tracks.h5'),
                                  selection_track_quality=1,
                                  force_prealignment=True)
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/track_analysis/Tracks_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/track_analysis/Tracks_result.h5')),
                                                            os.path.join(self.output_folder, 'Tracks.h5'), exact=False)
        self.assertTrue(data_equal, msg=error_msg)

        # Test 2: As test 1 but chunked data analysis, should result in the same tracks
        track_analysis.fit_tracks(input_track_candidates_file=analysis_utils.get_data('fixtures/track_analysis/TrackCandidates_result.h5',
                                                                                      output=os.path.join(testing_path,
                                                                                                          'fixtures/track_analysis/TrackCandidates_result.h5')),
                                  input_alignment_file=analysis_utils.get_data('fixtures/track_analysis/Alignment_result.h5',
                                                                               output=os.path.join(testing_path,
                                                                                                   'fixtures/track_analysis/Alignment_result.h5')),
                                  output_tracks_file=os.path.join(self.output_folder, 'Tracks_2.h5'),
                                  selection_track_quality=1,
                                  force_prealignment=True,
                                  chunk_size=4999)
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/track_analysis/Tracks_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/track_analysis/Tracks_result.h5')),
                                                            os.path.join(self.output_folder, 'Tracks_2.h5'), exact=False)
        self.assertTrue(data_equal, msg=error_msg)

        # Test 3: Fit all DUTs at once (special mode for constrained residuals)
        track_analysis.fit_tracks(input_track_candidates_file=analysis_utils.get_data('fixtures/track_analysis/TrackCandidates_result.h5',
                                                                                      output=os.path.join(testing_path,
                                                                                                          'fixtures/track_analysis/TrackCandidates_result.h5')),
                                  input_alignment_file=analysis_utils.get_data('fixtures/track_analysis/Alignment_result.h5',
                                                                               output=os.path.join(testing_path,
                                                                                                   'fixtures/track_analysis/Alignment_result.h5')),
                                  output_tracks_file=os.path.join(self.output_folder, 'Tracks_All.h5'),
                                  exclude_dut_hit=False,
                                  selection_track_quality=1,
                                  force_prealignment=True)
        # Fit DUTs consecutevly, but use always the same DUTs. Should result in the same data as above
        track_analysis.fit_tracks(input_track_candidates_file=analysis_utils.get_data('fixtures/track_analysis/TrackCandidates_result.h5',
                                                                                      output=os.path.join(testing_path, 'fixtures/track_analysis/TrackCandidates_result.h5')),
                                  input_alignment_file=analysis_utils.get_data('fixtures/track_analysis/Alignment_result.h5',
                                                                               output=os.path.join(testing_path,
                                                                                                   'fixtures/track_analysis/Alignment_result.h5')),
                                  output_tracks_file=os.path.join(self.output_folder, 'Tracks_All_Iter.h5'),
                                  selection_hit_duts=range(4),
                                  exclude_dut_hit=False,
                                  selection_track_quality=1,
                                  force_prealignment=True)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(self.output_folder, 'Tracks_All.h5'), os.path.join(self.output_folder, 'Tracks_All_Iter.h5'), exact=False)
        self.assertTrue(data_equal, msg=error_msg)
        # Fit DUTs consecutevly, but use always the same DUTs defined for each DUT separately. Should result in the same data as above
        track_analysis.fit_tracks(input_track_candidates_file=analysis_utils.get_data('fixtures/track_analysis/TrackCandidates_result.h5',
                                                                                      output=os.path.join(testing_path,
                                                                                                          'fixtures/track_analysis/TrackCandidates_result.h5')),
                                  input_alignment_file=analysis_utils.get_data('fixtures/track_analysis/Alignment_result.h5',
                                                                               output=os.path.join(testing_path,
                                                                                                   'fixtures/track_analysis/Alignment_result.h5')),
                                  output_tracks_file=os.path.join(self.output_folder, 'Tracks_All_Iter_2.h5'),
                                  selection_hit_duts=[range(4), range(4), range(4), range(4)],
                                  exclude_dut_hit=False,
                                  selection_track_quality=1,
                                  force_prealignment=True)
        data_equal, error_msg = test_tools.compare_h5_files(os.path.join(self.output_folder, 'Tracks_All.h5'), os.path.join(self.output_folder, 'Tracks_All_Iter_2.h5'), exact=False)
        self.assertTrue(data_equal, msg=error_msg)

        # Fit tracks and eliminate merged tracks
        track_analysis.fit_tracks(input_track_candidates_file=analysis_utils.get_data('fixtures/track_analysis/TrackCandidates_result.h5',
                                                                                      output=os.path.join(testing_path,
                                                                                                          'fixtures/track_analysis/TrackCandidates_result.h5')),
                                  input_alignment_file=analysis_utils.get_data('fixtures/track_analysis/Alignment_result.h5',
                                                                               output=os.path.join(testing_path,
                                                                                                   'fixtures/track_analysis/Alignment_result.h5')),
                                  output_tracks_file=os.path.join(self.output_folder, 'Tracks_merged.h5'),
                                  selection_track_quality=1,
                                  min_track_distance=True,  # Activate track merge cut,
                                  force_prealignment=True)
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/track_analysis/Tracks_merged_result.h5',
                                                                                    output=os.path.join(testing_path, 'fixtures/track_analysis/Tracks_merged_result.h5')), os.path.join(self.output_folder, 'Tracks_merged.h5'), exact=False)
        self.assertTrue(data_equal, msg=error_msg)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrackAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
