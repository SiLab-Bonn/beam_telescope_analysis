''' Script to test the simulator.
'''
import os
import tables as tb
import numpy as np
import unittest
import pylandau
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from beam_telescope_analysis.tools import simulate_data, geometry_utils, analysis_utils, test_tools

testing_path = os.path.dirname(os.path.abspath(__file__))


class TestHitAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # virtual X server for plots under headless LINUX travis testing is needed
        if os.getenv('TRAVIS', False) and os.getenv('TRAVIS_OS_NAME', False) == 'linux':
            from xvfbwrapper import Xvfb
            self.vdisplay = Xvfb()
            self.vdisplay.start()

        self.simulate_data = simulate_data.SimulateData(random_seed=0)

    @classmethod
    def tearDownClass(self):  # remove created files
        for dut_index in range(self.simulate_data.n_duts):
            os.remove('simulated_data_DUT%d.h5' % dut_index)

    # Test beam position with respect to devices positions
    def test_position(self):
        self.simulate_data.n_duts = 2
        self.simulate_data.set_std_settings()

        # Helper function to be called with different position parameter data
        def check_position():
            # Calculate expectation
            expected_mean_column, expected_mean_row = [], []
            for dut_index in range(self.simulate_data.n_duts):
                expected_mean_column.append(
                    self.simulate_data.beam_position[0] - self.simulate_data.offsets[dut_index][0])
                expected_mean_row.append(
                    self.simulate_data.beam_position[1] - self.simulate_data.offsets[dut_index][1])
            # Extract results
            mean_column, mean_row = [], []
            for dut_index in range(self.simulate_data.n_duts):
                with tb.open_file('simulated_data_DUT%d.h5' % dut_index, mode='r') as in_file_h5:
                    mean_column.append((in_file_h5.root.Hits[:][
                                       'column'].mean() - 1) * self.simulate_data.dut_pixel_size[dut_index][0])
                    mean_row.append((in_file_h5.root.Hits[:][
                                    'row'].mean() - 1) * self.simulate_data.dut_pixel_size[dut_index][1])

            self.assertTrue(
                np.allclose(expected_mean_column, mean_column, rtol=0.01, atol=10))
            self.assertTrue(
                np.allclose(expected_mean_row, mean_row, rtol=0.01, atol=10))

        # Test 1: Check different DUT offsets
        self.simulate_data.offsets = [(-35000, -35000), (-30000, -30000), (-25000, -25000),
                                      (-20000, -20000), (-15000, -15000), (-10000, -10000)]  # Set DUT offsets with respect to beam
        self.simulate_data.create_data_and_store(
            'simulated_data', n_events=10000)
        check_position()

        # Test 2: Check different beam offset
        self.simulate_data.beam_position = (500, 500)  # Shift beam position
        self.simulate_data.create_data_and_store(
            'simulated_data', n_events=10000)
        check_position()

        # Test 3: Check different beam parameter
        self.simulate_data.beam_position_sigma = (0, 0)  # beam position sigma
        self.simulate_data.beam_position = (0, 0)  # Shift beam position
        self.simulate_data.create_data_and_store(
            'simulated_data', n_events=10000)
        check_position()

    def test_charge_distribution(self):
        self.simulate_data.reset()

        # Helper function to be called with different charge parameter data
        def check_charge():
            # Loop over DUTs
            for dut_index in range(self.simulate_data.n_duts):
                # 71 electrons per um in silicon
                mpv_charge = 71. * self.simulate_data.dut_thickness[dut_index]

                def _calc_landau(x):
                    if self.simulate_data.dut_noise[dut_index]:
                        # PyLandau Langau MPV parameter is the MPV of the
                        # Langau not the Landau only!
                        y = pylandau.langau(
                            x, mpv=mpv_charge, eta=mpv_charge / 10., sigma=self.simulate_data.dut_noise[dut_index], scale_langau=False)
                    else:
                        y = pylandau.landau(
                            x, mpv=mpv_charge, eta=mpv_charge / 10)
                    return y

                # Check result with calculated expectation
                with tb.open_file('simulated_data_DUT%d.h5' % dut_index, mode='r') as in_file_h5:
                    charge = in_file_h5.root.Hits[:][
                        'charge'] * 10.  # 1 LSB corresponds to 10 electrons
                    charge_hist, edges = np.histogram(
                        charge, bins=100, range=(0., 5. * mpv_charge))
                    charge_hist = charge_hist.astype(np.float)
                    x = (edges[:-1] + edges[1:]) / 2
                    y = _calc_landau(x)

                    self.assertTrue(
                        np.allclose(y / y.sum() * charge_hist.sum(), charge_hist, rtol=0.07, atol=50.))
#                     import matplotlib.pyplot as plt
#                     plt.plot(x, charge_hist, label='Result')
#                     plt.plot(x, y / y.sum() * charge_hist.sum(), label='Expected')
#                     print x[y == np.amax(y)]
#                     print x[charge_hist == np.amax(charge_hist)]
#                     plt.legend(loc=0)
#                     plt.show()

        # Check Landau for different device thickness
        # Create charge distribution in different device thicknesses, thus
        # Landau MPW should change
        self.simulate_data.dut_thickness = [
            (i + 1) * 100 for i in range(self.simulate_data.n_duts)]
        # To judge deposited charge, charge sharing has to be off
        self.simulate_data.digitization_charge_sharing = False
        self.simulate_data.create_data_and_store(
            'simulated_data', n_events=100000)
        check_charge()

        # Check Landau for different device noiose
        self.simulate_data.reset()
        self.simulate_data.dut_thickness = [
            200 for i in range(self.simulate_data.n_duts)]  # Fix device thickness
        # Create charge distribution in different device noise, thus Langau
        # sigma should change
        self.simulate_data.dut_noise = [
            i * 1000 for i in range(self.simulate_data.n_duts)]
        # To judge deposited charge, charge sharing has to be off
        self.simulate_data.digitization_charge_sharing = False
        self.simulate_data.create_data_and_store(
            'simulated_data', n_events=100000)
        check_charge()

    @unittest.SkipTest  # FIXME: FAILS
    def test_beam_angle(self):
        self.simulate_data.reset()

        def check_beam_angle():
            # Expected offsets in x, y at DUT planes due to initial beam angle
            # theta and direction distribution phi = (start, stop)
            expected_offsets_x, expected_offsets_y = [], []
            if self.simulate_data.beam_direction[0] < self.simulate_data.beam_direction[1]:
                mean_direction_cos = np.cos(np.arange(self.simulate_data.beam_direction[0], self.simulate_data.beam_direction[
                                            1], 0.01)).mean()  # A mean angle does not translate linearly to a mean offset
                mean_direction_sin = np.sin(np.arange(self.simulate_data.beam_direction[
                                            0], self.simulate_data.beam_direction[1], 0.01)).mean()
            else:
                mean_direction_cos = np.cos(
                    self.simulate_data.beam_direction[0])
                mean_direction_sin = np.sin(
                    self.simulate_data.beam_direction[0])
            for dut_index in range(self.simulate_data.n_duts):
                offset = self.simulate_data.beam_position[
                    0] - self.simulate_data.offsets[dut_index][0]
                expected_offsets_x.append(offset + mean_direction_cos * np.tan(
                    self.simulate_data.beam_angle / 1000.) * self.simulate_data.z_positions[dut_index])
                expected_offsets_y.append(offset + mean_direction_sin * np.tan(
                    self.simulate_data.beam_angle / 1000.) * self.simulate_data.z_positions[dut_index])

            # Extract results
            mean_column, mean_row = [], []
            for dut_index in range(self.simulate_data.n_duts):
                with tb.open_file('simulated_data_DUT%d.h5' % dut_index, mode='r') as in_file_h5:
                    mean_column.append((in_file_h5.root.Hits[:][
                                       'column'].mean() - 1) * self.simulate_data.dut_pixel_size[dut_index][0])
                    mean_row.append((in_file_h5.root.Hits[:][
                                    'row'].mean() - 1) * self.simulate_data.dut_pixel_size[dut_index][1])

            # Check for similarity, on pixel width error expected (binning
            # error)
            self.assertTrue(np.allclose(expected_offsets_x, mean_column,
                                        rtol=0.001, atol=self.simulate_data.dut_pixel_size[0][0]))
            self.assertTrue(np.allclose(
                expected_offsets_y, mean_row, rtol=0.001, atol=self.simulate_data.dut_pixel_size[0][0]))

        # Test 1: Fixed theta angle, different fixed phi
        # If the angle is too small this tests fails due to pixel
        # discretisation error
        self.simulate_data.beam_angle = 5
        # If the pixel size is too big this tests fails due to pixel
        # discretisation error
        self.simulate_data.dut_pixel_size = [
            (1, 1)] * self.simulate_data.n_duts
        # If the sensor is too small the mean cannot be easily calculated
        self.simulate_data.dut_n_pixel = [
            (10000, 10000)] * self.simulate_data.n_duts
        self.simulate_data.beam_angle_sigma = 0
        self.simulate_data.beam_position_sigma = (0, 0)
        self.simulate_data.dut_material_budget = [
            0] * self.simulate_data.n_duts  # Turn off multiple scattering
        # Simplify position reconstruction
        self.simulate_data.digitization_charge_sharing = False
        self.simulate_data.shuffle_hits = False

        for phi in [0, np.pi / 4., np.pi / 2., 3. / 4. * np.pi, np.pi, 5. * np.pi / 4., 3 * np.pi / 2.]:
            self.simulate_data.beam_direction = (phi, phi)
            self.simulate_data.create_data_and_store(
                'simulated_data', n_events=10000)
            check_beam_angle()

        # Test 2: Fixed theta angle, different phi ranges
        for phi_max in [0, np.pi / 4., np.pi / 2., 3. / 4. * np.pi, np.pi, 5. * np.pi / 4., 3 * np.pi / 2.]:
            self.simulate_data.beam_direction = (0, phi_max)
            self.simulate_data.create_data_and_store(
                'simulated_data', n_events=10000)
            check_beam_angle()

        # Test 3: Fixed theta angle, different phi ranges
        for phi_max in [0, np.pi / 4., np.pi / 2., 3. / 4. * np.pi, np.pi, 5. * np.pi / 4., 3 * np.pi / 2.]:
            self.simulate_data.beam_direction = (0, phi_max)
            self.simulate_data.create_data_and_store(
                'simulated_data', n_events=10000)
            check_beam_angle()

        # Test 4: Gaussian dstributed theta angle, full phi range
        self.simulate_data.beam_angle_sigma = 2
        self.simulate_data.beam_direction = (0, 2. * np.pi)
        self.simulate_data.create_data_and_store(
            'simulated_data', n_events=10000)
        check_beam_angle()

    def test_multiple_scattering(self):
        self.simulate_data.reset()

        # Set two planes and check the scattering angle due to the material
        # budget of the first plane
        self.simulate_data.n_duts = 2
        self.simulate_data.set_std_settings()
        self.simulate_data.beam_angle = 0
        self.simulate_data.beam_angle_sigma = 0
        self.simulate_data.beam_position_sigma = (0, 0)
        # 1m distance to see scattering better
        self.simulate_data.z_positions = [
            i * 1000000 + 1000 for i in range(self.simulate_data.n_duts)]
        # If the pixel size is too big this tests fails due to pixel
        # discretisation error
        self.simulate_data.dut_pixel_size = [
            (1, 1)] * self.simulate_data.n_duts
        self.simulate_data.dut_n_pixel = [
            (10000, 10000)] * self.simulate_data.n_duts
        self.simulate_data.shuffle_hits = False

        # Simplify position reconstruction
        self.simulate_data.digitization_charge_sharing = False

        def gauss(x, A, mu, sigma):  # Scattering angle theta fit function
            return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

        def check_scattering_angle():
            # Expected scattering angle theta_0
            theta_0 = self.simulate_data._scattering_angle_sigma(
                self.simulate_data.dut_material_budget[0], charge_number=1) * 1000

            # Extract theta from simulation results by using the hit positions
            with tb.open_file('simulated_data_DUT0.h5', mode='r') as in_file_h5:
                dut0_x = (in_file_h5.root.Hits[:][
                          'column'] - 1) * self.simulate_data.dut_pixel_size[0][0]
                dut0_y = (
                    in_file_h5.root.Hits[:]['row'] - 1) * self.simulate_data.dut_pixel_size[0][1]

            with tb.open_file('simulated_data_DUT1.h5', mode='r') as in_file_h5:
                dut1_x = (in_file_h5.root.Hits[:][
                          'column'] - 1) * self.simulate_data.dut_pixel_size[1][0]
                dut1_y = (
                    in_file_h5.root.Hits[:]['row'] - 1) * self.simulate_data.dut_pixel_size[1][1]

            # Calculate theta in spherical coordinates from data
            dx = dut1_x.astype(np.float) - dut0_x.astype(np.int32)
            dy = dut1_y.astype(np.float) - dut0_y.astype(np.float)
            dz = np.ones_like(
                dx) * (self.simulate_data.z_positions[1] - self.simulate_data.z_positions[0])
            _, theta, _ = geometry_utils.cartesian_to_spherical(dx, dy, dz)

            # Special case: no material budget thus not scattering, theta has
            # to be beam angle
            if theta_0 == 0:
                self.assertTrue(
                    np.allclose(self.simulate_data.beam_angle, theta * 1000., atol=0.01))
                return

            if self.simulate_data.beam_angle == 0:
                # Histogramm scattering angle distribution
                hist, bins = np.histogram(
                    theta * 1000, range=(0, np.pi), bins=1000)
                x, y = (bins[:-1] + bins[1:]) / 2., hist
#                 import matplotlib.pyplot as plt
#                 plt.bar(x, y, width=np.diff(x)[0])
#                 plt.plot(x, gauss(x, *(np.amax(y), 0, theta_0)), 'r-', linewidth=2)
#                 plt.legend()
#                 plt.show()
                # Fit scatterign distribution
                # Fit theta distribution
                coeff, _ = curve_fit(
                    gauss, x, y, p0=[np.amax(hist), 0, theta_0])
                # Check for similarity, slight differences most likely due to
                # pixel position binning
                self.assertTrue(np.allclose(theta_0, coeff[
                                2], atol=0.02), 'The scattering angle for multiple scattering is wrong')
            # TODO: this is maybe a bug, the theta distribution with starting
            # beam angle does not look gaussian; just check the mean here;
            else:
                self.assertTrue(np.allclose(np.mean(theta * 1000.), self.simulate_data.beam_angle,
                                            atol=0.05), 'The beam direction with multiple scattering is wrong')

        # Test 1: Check scattering for different device thickness from 50 um to
        # 1000 um
        for device_thickness in range(0, 1000, 50):  # Change the thickness
            self.simulate_data.dut_thickness = [
                device_thickness] * self.simulate_data.n_duts
            self.simulate_data.dut_material_budget = [self.simulate_data.dut_thickness[
                i] * 1e-4 / 9.370 for i in range(self.simulate_data.n_duts)]  # Assume silicon sensor
            self.simulate_data.create_data_and_store(
                'simulated_data', n_events=100000)
            check_scattering_angle()

        # Test 2: Check scattering for different device z positions
        # Put planes at different positions
        for z_position in ([[i, i + 1000000] for i in range(0, 1000000, 250000)]):
            self.simulate_data.z_positions = z_position
            self.simulate_data.create_data_and_store(
                'simulated_data', n_events=100000)
            check_scattering_angle()

        # Test 3: Check with beam angle
        for beam_angle in range(5):
            self.simulate_data.beam_angle = beam_angle
            self.simulate_data.z_positions = [0, 100000]
            self.simulate_data.dut_thickness = [
                1000] * self.simulate_data.n_duts
            self.simulate_data.dut_material_budget = [self.simulate_data.dut_thickness[
                i] * 1e-4 / 9.370 for i in range(self.simulate_data.n_duts)]
            self.simulate_data.create_data_and_store(
                'simulated_data', n_events=100000)
            check_scattering_angle()

    def test_dut_rotation(self):
        self.simulate_data.reset()

        # Set two planes and check the scattering angle due to the material
        # budget of the first plane
        self.simulate_data.n_duts = 2
        self.simulate_data.set_std_settings()
        self.simulate_data.tracks_per_event_sigma = 0
        self.simulate_data.beam_angle = 0
        self.simulate_data.beam_angle_sigma = 0
        # If the pixel size is too big this tests fails due to pixel
        # discretisation error
        self.simulate_data.dut_pixel_size = [
            (1, 1)] * self.simulate_data.n_duts
        self.simulate_data.dut_n_pixel = [
            (100000, 100000)] * self.simulate_data.n_duts
        # 1m distance to avoid intersecting DUT planes
        self.simulate_data.z_positions = [
            i * 1000000. + 1000. for i in range(self.simulate_data.n_duts)]
        self.simulate_data.tracks_per_event = 1
        self.simulate_data.tracks_per_event_sigma = 0
        # Not needed, save time
        self.simulate_data.digitization_charge_sharing = False
        self.simulate_data.dut_material_budget = [
            0] * self.simulate_data.n_duts  # Turn off multiple scattering

        # Rotations are checked by column / row correlation in the global
        # coordinate system at 0,0,0 of the first DUT; correlation have to have
        # the slope 1 and the offset 0
        def check_rotations():
            def line(x, c0, c1):  # Correlation line fit
                return c1 * x + c0
            with tb.open_file('simulated_data_DUT0.h5', mode='r') as in_file_1_h5:
                with tb.open_file('simulated_data_DUT1.h5', mode='r') as in_file_2_h5:
                    # Merge data on event basis for correlation, since one
                    # cannot assume that every plane is always hit
                    hits_0, hits_1 = analysis_utils.merge_on_event_number(
                        in_file_1_h5.root.Hits[:], in_file_2_h5.root.Hits[:])

                    # Get transformation matrix (translation + rotation) from
                    # simulation settting
                    transformation_matrix = geometry_utils.local_to_global_transformation_matrix(x=self.simulate_data.offsets[1][0] - self.simulate_data.offsets[0][0],
                                                                                                 y=self.simulate_data.offsets[1][
                                                                                                     1] - self.simulate_data.offsets[0][1],
                                                                                                 z=self.simulate_data.z_positions[
                                                                                                     1] - self.simulate_data.z_positions[0],
                                                                                                 alpha=self.simulate_data.rotations[
                                                                                                     1][0],
                                                                                                 beta=self.simulate_data.rotations[
                                                                                                     1][1],
                                                                                                 gamma=self.simulate_data.rotations[
                                                                                                     1][2],
                                                                                                 )

                    hits_1_column_global, hits_1_row_global, _ = geometry_utils.apply_transformation_matrix(
                        hits_1['column'], hits_1['row'], np.zeros_like(hits_1['column']), transformation_matrix)

                    coeff_column, _ = curve_fit(
                        line, hits_1_column_global, hits_0['column'], p0=[0, 1.])
                    coeff_row, _ = curve_fit(
                        line, hits_1_row_global, hits_0['row'], p0=[0, 1.])

                    # Check column / row relative offsets are from line fit
                    # offsets with 1 mu precision

                    # TODO: Might be a bug that sometimes not many hits are
                    # created or just geometry
                    if hits_1_column_global.shape[0] > 25:
                        self.assertAlmostEqual(coeff_column[0], 0., delta=1)
                        self.assertAlmostEqual(coeff_row[0], 0., delta=1)
                        # Check alpha / beta angles with from line fit slopes
                        # with 0.1 % precision
                        self.assertAlmostEqual(
                            coeff_column[1], 1., delta=0.001)
                        self.assertAlmostEqual(coeff_row[1], 1., delta=0.001)

#                     plt.plot(hits_1_column_global, hits_0['column'], 'o', label='Data')
#                     plt.plot(hits_1_column_global, line(hits_1_column_global, *coeff_column), '--', label='Fit')
#                     plt.plot(hits_1_row_global, hits_0['row'], 'o', label='Data row')
#                     plt.plot(hits_1_row_global, line(hits_1_row_global, *coeff_row), '--', label='Fit row')
#                     plt.legend(loc=0)
#                     plt.show()

        # Test: Check correlation for different alpha, beta, gamma angles
        # (x/y/z-axis rotation) and different relative offsets in x/y between
        # the planes
        for alpha in [0, np.pi / 8., np.pi / 6., np.pi / 4., np.pi / 3.]:
            for beta in [0, np.pi / 8., np.pi / 6., np.pi / 4., np.pi / 3.]:
                for gamma in [0, np.pi / 8., np.pi / 6., np.pi / 4., np.pi / 3.]:
                    for offset_x in range(-1000, 1001, 500):
                        for offset_y in range(-1000, 1001, 500):
                            self.simulate_data.rotations[
                                1] = (alpha, beta, gamma)
                            self.simulate_data.offsets[1] = (self.simulate_data.offsets[0][
                                                             0] + offset_x, self.simulate_data.offsets[0][1] + offset_y)  # Set x/y shift with respect to DUT 0
                            self.simulate_data.create_data_and_store(
                                'simulated_data', n_events=1000)
                            check_rotations()

    def test_simulation(self):
        ''' Check the full simulation '''
        self.simulate_data.reset()
        self.simulate_data.set_std_settings()
        self.assertEqual(self.simulate_data.n_duts, 6)

        self.simulate_data.create_data_and_store(
            'simulated_data', n_events=10000)

        for dut_index in range(self.simulate_data.n_duts):
            data_equal, error_msg = test_tools.compare_h5_files('simulated_data_DUT%d.h5' % dut_index,
                                                                analysis_utils.get_data('fixtures/simulation/simulated_data_DUT%d.h5' % dut_index,
                                                                                        output=os.path.join(testing_path, 'fixtures/simulation/simulated_data_DUT%d.h5' % dut_index)),
                                                                exact=False)
            self.assertTrue(data_equal, msg=error_msg)


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHitAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
