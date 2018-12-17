''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os
import shutil
import unittest

import numpy as np

from beam_telescope_analysis import dut_alignment
from beam_telescope_analysis.tools import test_tools
from beam_telescope_analysis.tools import geometry_utils
from beam_telescope_analysis.tools import analysis_utils

testing_path = os.path.dirname(os.path.abspath(__file__))


class TestAlignmentAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # virtual X server for plots under headless LINUX travis testing is needed
        if os.getenv('TRAVIS', False) and os.getenv('TRAVIS_OS_NAME', False) == 'linux':
            from xvfbwrapper import Xvfb
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()
        # Define test data input files, download if needed
        cls.data_files = [analysis_utils.get_data('fixtures/dut_alignment/Cluster_DUT%d_cluster.h5' % i,
                                                  output=os.path.join(testing_path,
                                                                      'fixtures/dut_alignment/Cluster_DUT%d_cluster.h5' % i))
                          for i in range(4)]
        # Define and create tests output folder, is deleted at the end of tests
        cls.output_folder = os.path.join(os.path.dirname(os.path.realpath(cls.data_files[0])), 'output')
        test_tools.create_folder(cls.output_folder)
        cls.n_pixels = [(80, 336)] * 4
        cls.pixel_size = [(250, 50)] * 4  # in um
        cls.z_positions = [0., 19500, 108800, 128300]

    @classmethod
    def tearDownClass(cls):  # remove created files
        shutil.rmtree(cls.output_folder)

    def test_cluster_correlation(self):  # Check the cluster correlation function
        dut_alignment.correlate_cluster(input_cluster_files=self.data_files,
                                        output_correlation_file=os.path.join(self.output_folder, 'Correlation.h5'),
                                        n_pixels=self.n_pixels,
                                        pixel_size=self.pixel_size
                                        )
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/dut_alignment/Correlation_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/dut_alignment/Correlation_result.h5')),
                                                            os.path.join(self.output_folder, 'Correlation.h5'), exact=True)
        self.assertTrue(data_equal, msg=error_msg)

        # Retest with tiny chunk size to force chunked correlation
        dut_alignment.correlate_cluster(input_cluster_files=self.data_files,
                                        output_correlation_file=os.path.join(self.output_folder, 'Correlation_2.h5'),
                                        n_pixels=self.n_pixels,
                                        pixel_size=self.pixel_size,
                                        chunk_size=293
                                        )
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/dut_alignment/Correlation_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/dut_alignment/Correlation_result.h5')),
                                                            os.path.join(self.output_folder, 'Correlation_2.h5'), exact=True)
        self.assertTrue(data_equal, msg=error_msg)

    # FIXME: fails under Linux, needs check why
    @unittest.SkipTest
    def test_prealignment(self):  # Check the hit alignment function
        dut_alignment.prealignment(input_correlation_file=analysis_utils.get_data('fixtures/dut_alignment/Correlation_result.h5',
                                                                                  output=os.path.join(testing_path,
                                                                                                      'fixtures/dut_alignment/Correlation_result.h5')),
                                   output_alignment_file=os.path.join(self.output_folder, 'Alignment.h5'),
                                   z_positions=self.z_positions,
                                   pixel_size=self.pixel_size,
                                   non_interactive=True,
                                   fit_background=False,
                                   iterations=3)  # Due to too little test data the alignment result is only rather stable for more iterations

        # FIXME: residuals should be checked not prealingment data
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/dut_alignment/Prealignment_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/dut_alignment/Prealignment_result.h5')),
                                                            os.path.join(self.output_folder, 'Alignment.h5'),
                                                            exact=False,
                                                            rtol=0.05,  # 5 % error allowed
                                                            atol=5)  # 5 um absolute tolerance allowed
        self.assertTrue(data_equal, msg=error_msg)

        dut_alignment.prealignment(input_correlation_file=analysis_utils.get_data('fixtures/dut_alignment/Correlation_difficult.h5',
                                                                                  output=os.path.join(testing_path,
                                                                                                      'fixtures/dut_alignment/Correlation_difficult.h5')),
                                   output_alignment_file=os.path.join(self.output_folder, 'Alignment_difficult.h5'),
                                   z_positions=self.z_positions,
                                   pixel_size=self.pixel_size,
                                   non_interactive=True,
                                   fit_background=True,
                                   iterations=2)  # Due to too little test data the alignment result is only rather stable for more iterations
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data(r'fixtures/dut_alignment/Alignment_difficult_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/dut_alignment/Alignment_difficult_result.h5')),
                                                            os.path.join(self.output_folder, 'Alignment_difficult.h5'),
                                                            exact=False,
                                                            rtol=0.05,  # 5 % error allowed
                                                            atol=5)  # 5 um absolute tolerance allowed
        self.assertTrue(data_equal, msg=error_msg)

    def test_cluster_merging(self):
        cluster_files = [analysis_utils.get_data('fixtures/dut_alignment/Cluster_DUT%d_cluster.h5' % i,
                                                 output=os.path.join(testing_path,
                                                                     'fixtures/dut_alignment/Cluster_DUT%d_cluster.h5' % i))
                                                 for i in range(4)]
        dut_alignment.merge_cluster_data(cluster_files,
                                         output_merged_file=os.path.join(self.output_folder, 'Merged.h5'),
                                         n_pixels=self.n_pixels,
                                         pixel_size=self.pixel_size)
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/dut_alignment/Merged_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/dut_alignment/Merged_result.h5')),
                                                                                    os.path.join(self.output_folder, 'Merged.h5'))
        self.assertTrue(data_equal, msg=error_msg)

        # Retest with tiny chunk size to force chunked merging
        dut_alignment.merge_cluster_data(cluster_files,
                                         output_merged_file=os.path.join(self.output_folder, 'Merged_2.h5'),
                                         pixel_size=self.pixel_size,
                                         n_pixels=self.n_pixels,
                                         chunk_size=293)

        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/dut_alignment/Merged_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/dut_alignment/Merged_result.h5')),
                                                                                    os.path.join(self.output_folder, 'Merged_2.h5'))
        self.assertTrue(data_equal, msg=error_msg)

    def test_apply_alignment(self):
        dut_alignment.apply_alignment(input_hit_file=analysis_utils.get_data('fixtures/dut_alignment/Merged_result.h5',
                                                                             output=os.path.join(testing_path,
                                                                                                 'fixtures/dut_alignment/Merged_result.h5')),
                                      input_alignment_file=analysis_utils.get_data('fixtures/dut_alignment/Prealignment_result.h5',
                                                                                   output=os.path.join(testing_path,
                                                                                                       'fixtures/dut_alignment/Prealignment_result.h5')),
                                      output_hit_file=os.path.join(self.output_folder, 'Tracklets.h5'),
                                      use_prealignment=True)
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/dut_alignment/Tracklets_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/dut_alignment/Tracklets_result.h5')),
                                                                                    os.path.join(self.output_folder, 'Tracklets.h5'))
        self.assertTrue(data_equal, msg=error_msg)

        # Retest with tiny chunk size to force chunked alignment apply
        dut_alignment.apply_alignment(input_hit_file=analysis_utils.get_data('fixtures/dut_alignment/Merged_result.h5',
                                                                             output=os.path.join(testing_path,
                                                                                                 'fixtures/dut_alignment/Merged_result.h5')),
                                      input_alignment_file=analysis_utils.get_data('fixtures/dut_alignment/Prealignment_result.h5',
                                                                                   output=os.path.join(testing_path,
                                                                                                       'fixtures/dut_alignment/Prealignment_result.h5')),
                                      output_hit_file=os.path.join(self.output_folder, 'Tracklets_2.h5'),
                                      use_prealignment=True,
                                      chunk_size=293)
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/dut_alignment/Tracklets_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/dut_alignment/Tracklets_result.h5')),
                                                                                    os.path.join(self.output_folder, 'Tracklets_2.h5'))
        self.assertTrue(data_equal, msg=error_msg)

    @unittest.SkipTest
    def test_alignment(self):
        dut_alignment.alignment(input_track_candidates_file=analysis_utils.get_data('fixtures/dut_alignment/TrackCandidates_prealigned.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/dut_alignment/TrackCandidates_prealigned.h5')),
                                input_alignment_file=analysis_utils.get_data('fixtures/dut_alignment/Alignment.h5',
                                                                             output=os.path.join(testing_path,
                                                                                                 'fixtures/dut_alignment/Alignment.h5')),
                                n_pixels=[(1152, 576)] * 6,
                                pixel_size=[(18.4, 18.4)] * 6)

        # FIXME: test should check residuals not alignment resulds
        # FIXME: translation error can be in the order of um, angle error not
        data_equal, error_msg = test_tools.compare_h5_files(analysis_utils.get_data('fixtures/dut_alignment/Alignment.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/dut_alignment/Alignment.h5')),
                                                            analysis_utils.get_data('fixtures/dut_alignment/Alignment_result.h5',
                                                                                    output=os.path.join(testing_path,
                                                                                                        'fixtures/dut_alignment/Alignment_result.h5')),
                                                            exact=False,
                                                            rtol=0.01,  # 1 % error allowed
                                                            atol=5)  # 0.0001 absolute tolerance allowed
        self.assertTrue(data_equal, msg=error_msg)

    # FIXME: fails under Linux
    @unittest.SkipTest
    def test_rotation_reconstruction(self):  # Create fake data with known angles and reconstruct the angles from the residuals and check for similarity. Does only work for the abolute annge not with sign.

        def create_track_intersections(alpha, beta, gamma, x_global, y_global):  # Create fake data

            # Construct a rotated plane

            # Define plane by plane normal from two orthogonal direction vectors in the x-y plane in the local coordinate system
            plane_normal_local = geometry_utils.get_plane_normal(direction_vector_1=np.array([1., 0., 0.]),
                                                                 direction_vector_2=np.array([0., 1., 0.]))

            # Rotate plane normal to global coordinate system
            rotation_matrix = geometry_utils.rotation_matrix(alpha=alpha,  # Rotation matrix for local to global rotation
                                                             beta=beta,
                                                             gamma=gamma)

            plane_normal_global = geometry_utils.apply_rotation_matrix(x=plane_normal_local[0],  # Transfer to global coordinate system
                                                                       y=plane_normal_local[1],
                                                                       z=plane_normal_local[2],
                                                                       rotation_matrix=rotation_matrix)
            plane_normal_global = np.array([plane_normal_global[0][0], plane_normal_global[1][0], plane_normal_global[2][0]])

            # Calculate intersections with the plane in the global coordinate system
            line_origins = np.vstack((x_global, y_global, np.zeros_like(x_global))).T
            line_directions = np.repeat(np.array([0., 0., 1.]), x_global.shape[0]).reshape((3, x_global.shape[0])).T

            intersections_global = geometry_utils.get_line_intersections_with_plane(line_origins=line_origins,
                                                                                    line_directions=line_directions,
                                                                                    position_plane=np.array([0., 0., 0.]),
                                                                                    normal_plane=plane_normal_global)

            # Calculate track intersections in local coordinate system
            intersections_local = geometry_utils.apply_rotation_matrix(x=intersections_global[:, 0],
                                                                       y=intersections_global[:, 1],
                                                                       z=intersections_global[:, 2],
                                                                       rotation_matrix=rotation_matrix.T)

            return intersections_local

        def get_rotation_reconstruction(alpha, beta, gamma):  # Returns reconstructed angles from given input angles
            # Set limits and dimensions
            x_min, x_max = -100, 100
            y_min, y_max = -100, 100
            d_xy = 1  # Step width of the points
            n_x = int((x_max - x_min) / d_xy)
            n_y = int((y_max - y_min) / d_xy)

            # Create data points in global reference system
            X, Y = np.meshgrid(np.arange(x_min, x_max, d_xy), np.arange(y_min, y_max, d_xy))
            x_global, y_global = np.vstack([X.ravel(), Y.ravel()])  # z_global is unknown as is determined by plane intersections

            x_local, y_local, _ = create_track_intersections(alpha, beta, gamma, x_global, y_global)
            x_residuals = x_global - x_local
            y_residuals = y_global - y_local

            # Fit residual hists
            x_residual_x_fit_popt = analysis_utils.fit_residuals(x_local, x_residuals, n_bins=n_x, min_pos=x_min, max_pos=x_max)[0]
            y_residual_y_fit_popt = analysis_utils.fit_residuals(y_local, y_residuals, n_bins=n_y, min_pos=y_min, max_pos=y_max)[0]
            x_residual_y_fit_popt = analysis_utils.fit_residuals(x_local, y_residuals, n_bins=n_x, min_pos=x_min, max_pos=y_max)[0]
            y_residual_x_fit_popt = analysis_utils.fit_residuals(y_local, x_residuals, n_bins=n_y, min_pos=y_min, max_pos=x_max)[0]

            m_xx, m_yx, m_xy, m_yy = x_residual_x_fit_popt[1], x_residual_y_fit_popt[1], y_residual_x_fit_popt[1], y_residual_y_fit_popt[1]

            return analysis_utils.get_rotation_from_residual_fit(m_xx, m_xy, m_yx, m_yy)

        # Test 1: angles corrections
        atol = 0.05
        rtol = 0.05
        for alpha in np.arange(-0.2, 0.2, 0.04):
            for beta in np.arange(-0.2, 0.2, 0.04):
                for gamma in np.arange(-0.2, 0.2, 0.04):
                    alpha_reco, beta_reco, gamma_reco = get_rotation_reconstruction(alpha, beta, gamma)
                    self.assertTrue(np.allclose(np.abs(alpha_reco), np.abs(alpha), atol=atol, rtol=rtol))
                    self.assertTrue(np.allclose(np.abs(beta_reco), np.abs(beta), atol=atol, rtol=rtol))
                    self.assertTrue(np.allclose(np.abs(gamma_reco), np.abs(gamma), atol=atol, rtol=rtol))

        # Test 2: small angles corrections but devices inverted in beam around x axis (alpha = pi, y coordinates are flipped)
        atol = 0.05
        rtol = 0.05
        for alpha in np.arange(-0.05, 0.05, 0.01):
            for beta in np.arange(-0.05, 0.05, 0.01):
                for gamma in np.arange(-0.05, 0.05, 0.01):
                    alpha_reco, beta_reco, gamma_reco = get_rotation_reconstruction(alpha + np.pi, beta, gamma)
                    self.assertTrue(np.allclose(np.abs(alpha_reco), np.abs(alpha + np.pi), atol=atol, rtol=rtol))
                    self.assertTrue(np.allclose(np.abs(beta_reco), np.abs(beta), atol=atol, rtol=rtol))
                    self.assertTrue(np.allclose(np.abs(gamma_reco), np.abs(gamma), atol=atol, rtol=rtol))

        # Test 3: small angles corrections but devices inverted in beam around y axis (beta = pi, x coordinates are flipped)
        atol = 0.05
        rtol = 0.05
        for alpha in np.arange(-0.05, 0.05, 0.01):
            for beta in np.arange(-0.05, 0.05, 0.01):
                for gamma in np.arange(-0.05, 0.05, 0.01):
                    alpha_reco, beta_reco, gamma_reco = get_rotation_reconstruction(alpha, beta + np.pi, gamma)
                    self.assertTrue(np.allclose(np.abs(alpha_reco), np.abs(alpha), atol=atol, rtol=rtol))
                    self.assertTrue(np.allclose(np.abs(beta_reco), np.abs(beta + np.pi), atol=atol, rtol=rtol))
                    self.assertTrue(np.allclose(np.abs(gamma_reco), np.abs(gamma), atol=atol, rtol=rtol))


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAlignmentAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
