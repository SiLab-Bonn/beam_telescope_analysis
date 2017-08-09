''' Script to check the correctness of the analysis. The analysis is done on raw data and all results are compared to a recorded analysis.
'''
import os

import unittest

import numpy as np

from testbeam_analysis import track_analysis
from testbeam_analysis.tools import test_tools


class TestTrackAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # virtual X server for plots under headless LINUX travis testing is needed
        if os.getenv('TRAVIS', False) and os.getenv('TRAVIS_OS_NAME', False) == 'linux':
            from xvfbwrapper import Xvfb  # virtual X server for plots under headless LINUX travis testing is needed
            cls.vdisplay = Xvfb()
            cls.vdisplay.start()

    @classmethod
    def tearDownClass(cls):  # Remove created files
        pass
        # os.remove(os.path.join(cls.output_folder, 'Tracks_merged.pdf'))

    def test_kalman(self):
        # pixel size of sensor
        pixel_size = np.array([(18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (18.5, 18.5), (250., 50.)])
        pixel_resolution = pixel_size / np.sqrt(12)
        material_budget = np.array([100., 100., 100., 100., 100., 100., 250.]) / np.array([125390., 125390., 125390., 125390., 125390., 125390., 93700.])
        prealignment = {'z': [0., 29900., 60300., 82100., 118700., 160700., 197800.]}
        kwargs = {'track_hits': np.array([[[-1229.22372954, 2828.19616302, 0., pixel_resolution[0][0], pixel_resolution[0][1], 0.],
                                           [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # [-1254.51224282, 2827.4291421, 29900.],
                                           [-1285.6117892, 2822.34536687, 60300., pixel_resolution[2][0], pixel_resolution[2][1], 0.],
                                           [-1311.31083616, 2823.56121414, 82100., pixel_resolution[3][0], pixel_resolution[3][1], 0.],
                                           [-1335.8529645, 2828.43359043, 118700., pixel_resolution[4][0], pixel_resolution[4][1], 0.],
                                           [-1357.81872222, 2840.86947964, 160700., pixel_resolution[5][0], pixel_resolution[5][1], 0.],
                                           [-1396.35698339, 2843.76799577, 197800., pixel_resolution[6][0], pixel_resolution[6][1], 0.]]]),
                  'dut_fit_selection': 61,
                  'z_positions': [[0., 29900, 60300, 82100, 118700, 160700, 197800]],
                  'alignment': [prealignment],
                  'use_prealignment': True,
                  'pixel_size': pixel_size,
                  'n_pixels': ((576, 1152), (576, 1152), (576, 1152), (576, 1152), (576, 1152), (576, 1152), (80, 336)),
                  'beam_energy': 2500.,
                  'material_budget': material_budget,
                  'add_scattering_plane': False}

        # expected result array: (state estimates, chi, x error, y errors)
        result = [[[-1.23001328e+03, 2.82727830e+03, 0.00000000e+00, -9.15879726e-04, -2.09033989e-05, 9.99999580e-01],
                   [-1.25862573e+03, 2.82521051e+03, 2.99000000e+04, -8.74821828e-04, 2.73502270e-05, 9.99999617e-01],
                   [-1.28712065e+03, 2.82381689e+03, 6.03000000e+04, -8.12311008e-04, 1.00543006e-04, 9.99999665e-01],
                   [-1.30702210e+03, 2.82479582e+03, 8.21000000e+04, -7.11712114e-04, 1.56181416e-04, 9.99999735e-01],
                   [-1.33425505e+03, 2.82962871e+03, 1.18700000e+05, -6.79354657e-04, 1.80316637e-04, 9.99999753e-01],
                   [-1.36061740e+03, 2.83880395e+03, 1.60700000e+05, -7.31034376e-04, 1.42175127e-04, 9.99999723e-01],
                   [-1.38773879e+03, 2.84407865e+03, 1.97800000e+05, -7.31034376e-04, 1.42175127e-04, 9.99999723e-01]],
                  [41.906100070254126],
                  [3.67202783, 4.76060863, 3.73543952, 3.67302492, 3.99327848, 4.88787552, 10.82919706],
                  [3.67202783, 4.76060863, 3.73543952, 3.67302492, 3.99327848, 4.88787552, 10.82919706]]

        for i in range(4):  # test each return (state estimates, chi, x error, y errors) seperatly
            test = test_tools._call_function_with_args(function=track_analysis._fit_tracks_kalman_loop,
                                                       **kwargs)[0][i]
            data_equal = np.allclose(test, result[i])
            self.assertTrue(data_equal)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTrackAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
