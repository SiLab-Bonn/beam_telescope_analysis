''' This module provides a class that is able to simulate data via Monte Carlo
with a decend approximation of the reality. A random seed can be set.

The deposited charge follows a Landau * Gaus function. Special processes like
delta electrons are not simulated.

Multiple scattering is approximated with the assumption of gaussian distributed
scattering angles theta. Scattering is only calculated at the planes and not in
between (scattering in air).

Charge sharing between pixels is approximated using Einsteins diffusion
equation solved at the center z position within the sensor. Track angles within
the sensor do not influence the charge sharing.
'''
from __future__ import division

import logging
import math

import numpy as np
import tables as tb
from numba import njit

from tqdm import tqdm

import pylandau

from beam_telescope_analysis.tools import geometry_utils


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


# Jitted function for fast calculations
@njit()
def _calc_sigma_diffusion(distance, temperature, bias):
    ''' Calculates the sigma of the diffusion according to Einsteins equation.
    Parameters
    ----------
    length : number
        the drift distance
    temperature : number
        Temperature of the sensor
    bias : number
        bias voltage of the sensor

    Returns
    -------
    number

    '''

    boltzman_constant = 8.6173324e-5
    return distance * np.sqrt(2 * temperature / bias * boltzman_constant)


@njit()
def _bivariante_normal_cdf(a1, a2, b1, b2, mu1, mu2, sigma):
    '''Calculates the integral of the bivariante normal distribution.

    Integral between x = [a1, a2], y = [b1, b2].
    The normal distribution has two mu: mu1, mu2 but only one common sigma.

    Parameters
    ----------
    a1, a2: number
        Integration limits in x

    b1, b2: number
        Integration limits in y

    mu1, mu2: number, array like
        Position in x, y where the integral is evaluated

    sigma: number
        distribution parameter

    Returns
    -------
    number, array like
    '''

    return 1 / 4. * ((math.erf((a2 - mu1) / np.sqrt(2 * sigma ** 2)) -
                      math.erf((a1 - mu1) / np.sqrt(2 * sigma ** 2))) *
                     (math.erf((b2 - mu2) / np.sqrt(2 * sigma ** 2)) -
                      math.erf((b1 - mu2) / np.sqrt(2 * sigma ** 2))))


@njit()
def _calc_charge_fraction(position, position_z, pixel_index_x, pixel_index_y, pixel_size_x, pixel_size_y, temperature, bias, digitization_sigma_cc):
    ''' Calculates the fraction of charge [0, 1] within one rectangular pixel.

        Diffusion is considered. The calculation is done within the local pixel
        coordinate system, with the origin [x_pitch / 2, y_pitch / 2, 0]

        Parameters
        ----------
        position_i : array
            Position in x/y within the seed pixel where the charge is created
        pixel_index_x, pixel_index_y : number
            Pixel index relative to seed (= 0/0) to get the charge fraction for
        temperature: number
            Temperature of the sensor
        bias: number
            Bias voltage of the sensor
        digitization_sigma_cc: number
            The sigma is higher due to repulsion, so correct sigma with
            factor > 1, very simple approximation
            for further info see NIMA 606 (2009) 508-516
        pixel_size_x, pixel_size_y : number
            Pixel dimensions in x/y in um

        Returns
        -------
        number
    '''
    sigma = _calc_sigma_diffusion(
        distance=position_z,
        temperature=temperature,
        bias=bias) * digitization_sigma_cc

    if (sigma == 0):  # Tread not defined calculation input
        return 1.
    return _bivariante_normal_cdf(pixel_size_x * (pixel_index_x - 1. / 2.), pixel_size_x * (pixel_index_x + 1. / 2.), pixel_size_y * (pixel_index_y - 1. / 2.), pixel_size_y * (pixel_index_y + 1. / 2.), position[0], position[1], sigma)


@njit()
def _create_cs_hits(relative_position, column, row, charge, max_column, max_row, thickness, pixel_size_x, pixel_size_y, temperature, bias, digitization_sigma_cc, result_hits, index):
    ''' Create additional hits due to charge sharing.

    Run time optimized loops using an abort condition utilizing that the charge
    sharing always decreases for increased distance to seed pixel and the fact
    that the total charge fraction sum is 1
    '''

    # Charge fraction summed up for all pixels used; should be 1. if all
    # pixels are considered
    total_fraction = 0.
    min_fraction = 1e-2  # Distribute total charge with maximum 1% loss.
    n_hits = 0  # Total Number of hits created

    # FIXME: Charges are distributed along a track and not in the center z
    position_z = thickness / 2
    # Calc charge in pixels in + column direction
    for actual_column in range(int(column), max_column):
        # Omit row loop if charge fraction is already too low for seed row (=0)
        if (total_fraction >= 1. - min_fraction or
            _calc_charge_fraction(
                position=relative_position,
                position_z=position_z,
                pixel_index_x=actual_column - column,
                pixel_index_y=0,
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                temperature=temperature,
                bias=bias,
                digitization_sigma_cc=digitization_sigma_cc) < min_fraction):
            break

        # Calc charge in pixels in + row direction
        for actual_row in range(int(row), max_row):
            fraction = _calc_charge_fraction(
                position=relative_position,
                position_z=position_z,
                pixel_index_x=actual_column - column,
                pixel_index_y=actual_row - row,
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                temperature=temperature,
                bias=bias,
                digitization_sigma_cc=digitization_sigma_cc)
            total_fraction += fraction
            # Abort loop if fraction is too small, next pixel have even smaller
            # fraction
            if fraction < min_fraction:
                break
            # ADD HIT
            if index < result_hits.shape[0]:
                result_hits[index][0], result_hits[index][1], result_hits[
                    index][2] = actual_column, actual_row, fraction * charge
            else:
                raise RuntimeError(
                    'Provided result hist does not fit charge sharing hits')
            index += 1
            n_hits += 1
            if total_fraction >= 1. - min_fraction:
                break

        # Calc charge in pixels in - row direction
        for actual_row in range(int(row - 1), 0, -1):
            fraction = _calc_charge_fraction(
                position=relative_position,
                position_z=position_z,
                pixel_index_x=actual_column - column,
                pixel_index_y=actual_row - row,
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                temperature=temperature,
                bias=bias,
                digitization_sigma_cc=digitization_sigma_cc)
            total_fraction += fraction
            # Abort loop if fraction is too small, next pixel have even smaller
            # fraction
            if fraction < min_fraction:
                break
            # ADD HIT
            if index < result_hits.shape[0]:
                result_hits[index][0], result_hits[index][1], result_hits[index][2] = actual_column, actual_row, fraction * charge
            else:
                raise RuntimeError(
                    'Provided result hist does not fit charge sharing hits')
            index += 1
            n_hits += 1
            if total_fraction >= 1. - min_fraction:
                break

    # Calc charge in pixels in + column direction
    for actual_column in range(int(column - 1), 0, -1):
        # Omit row loop if charge fraction is already too low for seed row (=0)
        if (total_fraction >= 1. - min_fraction or
            _calc_charge_fraction(
                position=relative_position,
                position_z=position_z,
                pixel_index_x=actual_column - column,
                pixel_index_y=0,
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                temperature=temperature,
                bias=bias,
                digitization_sigma_cc=digitization_sigma_cc) < min_fraction):
            break

        # Calc charge in pixels in + row direction
        for actual_row in range(int(row), max_row):
            fraction = _calc_charge_fraction(
                position=relative_position,
                position_z=position_z,
                pixel_index_x=actual_column - column,
                pixel_index_y=actual_row - row,
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                temperature=temperature,
                bias=bias,
                digitization_sigma_cc=digitization_sigma_cc)
            total_fraction += fraction
            # Abort loop if fraction is too small, next pixel have even smaller
            # fraction
            if fraction < min_fraction:
                break
            # ADD HIT
            if index < result_hits.shape[0]:
                result_hits[index][0], result_hits[index][1], result_hits[
                    index][2] = actual_column, actual_row, fraction * charge
            else:
                raise RuntimeError(
                    'Provided result hist does not fit charge sharing hits')
            index += 1
            n_hits += 1
            if total_fraction >= 1. - min_fraction:
                break

        # Calc charge in pixels in - row direction
        for actual_row in range(int(row - 1), 0, -1):
            fraction = _calc_charge_fraction(
                position=relative_position,
                position_z=position_z,
                pixel_index_x=actual_column - column,
                pixel_index_y=actual_row - row,
                pixel_size_x=pixel_size_x,
                pixel_size_y=pixel_size_y,
                temperature=temperature,
                bias=bias,
                digitization_sigma_cc=digitization_sigma_cc)
            total_fraction += fraction
            # Abort loop if fraction is too small, next pixel have even smaller
            # fraction
            if fraction < min_fraction:
                break
            # ADD HIT
            if index < result_hits.shape[0]:
                result_hits[index][0], result_hits[index][1], result_hits[
                    index][2] = actual_column, actual_row, fraction * charge
            else:
                raise RuntimeError(
                    'Provided result hist does not fit charge sharing hits')
            index += 1
            n_hits += 1
            if total_fraction >= 1. - min_fraction:
                break

    return index, n_hits


@njit()
def _add_charge_sharing_hits(rel_pos, hits_digits, max_column, max_row, thickness, pixel_size_x, pixel_size_y, temperature, bias):
    ''' Add additional hits to seed hits due to charge sharing.
    '''

    n_hits_per_seed_hit = np.zeros(hits_digits.shape[0], dtype=np.int16)
    # Result array to be filled; up to 10 hits per seed hit is possible
    result_hits = np.zeros(
        shape=(5 * hits_digits.shape[0], 3), dtype=np.float64)
    index = 0
    for d_index in range(hits_digits.shape[0]):
        actual_hit_digit = hits_digits[d_index]
        index, n_hits = _create_cs_hits(
            relative_position=rel_pos[d_index],
            column=actual_hit_digit[0],
            row=actual_hit_digit[1],
            charge=actual_hit_digit[2],
            max_column=max_column,
            max_row=max_row,
            thickness=thickness,
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
            temperature=temperature,
            bias=bias,
            digitization_sigma_cc=1.,
            result_hits=result_hits,
            index=index)

        n_hits_per_seed_hit[d_index] = n_hits

    return result_hits[:index], n_hits_per_seed_hit


@njit()
def shuffle_event_hits(event_number, n_tracks_per_event, hits, seed):
    ''' Takes the hits of all DUTs and shuffles them for each event
    '''

    index = 0
    # Hack to allow np.shuffle on a multidimesnional array,
    # http://numba.pydata.org/numba-doc/dev/reference/numpysupported.html#simple-random-data
    indices = np.arange(hits.shape[0])

    np.random.seed(seed)

    while index < hits.shape[0]:  # Loop over actual DUT hits
        if n_tracks_per_event[index] == 1:  # One cannot shuffle one hit
            index += 1
            continue

        # Happens inplace
        np.random.shuffle(indices[index:index + n_tracks_per_event[index]])

        # Actual event is shuffled, increase index until new event
        while index < hits.shape[0] - 1:
            if event_number[index] != event_number[index + 1]:
                break
            index += 1

        index += 1

    # copy instruction, inplace not possible due to numba limitations
    return hits[indices]


class SimulateData(object):

    def __init__(self, random_seed=0):
        self.random_seed = random_seed
        self.reset()

    def set_random_seed(self, value):
        self.random_seed = value
        # Set the random number seed to be able to rerun with same results
        np.random.seed(self.random_seed)

    @property
    def n_duts(self):
        return self._n_duts

    @n_duts.setter
    def n_duts(self, value):
        if value > self._n_duts:
            logging.warning(
                'Number of DUTs increased, reset to standard settings!')
            self.set_std_settings()
        self._n_duts = value

    def set_std_settings(self):
        # Setup settings
        # in um; std: every 10 cm
        self.z_positions = [i * 10000 for i in range(self._n_duts)]
        self.offsets = [(-2500, -2500)] * self._n_duts  # in x, y in mu
        # in rotation around x, y, z axis in Rad
        self.rotations = [(0, 0, 0)] * self._n_duts
        # Temperature in Kelvin, needed for charge sharing calculation
        self.temperature = 300

        # Beam settings
        # Average beam position in x, y at z = 0 in mu
        self.beam_position = (0, 0)
        self.beam_position_sigma = (2000, 2000)  # in x, y at z = 0 in mu
        # Average beam angle from the beam axis in theta at z = 0 in mRad
        self.beam_angle = 0
        # Deviation from the average beam angle in theta at z = 0 in mRad
        self.beam_angle_sigma = 1
        # The range of directions of the beam (phi in spherical coordinates) at
        # z = 0 in Rad
        self.beam_direction = (0, 2. * np.pi)
        self.beam_momentum = 3200  # Beam momentum in MeV
        self.tracks_per_event = 1  # Average number of tracks per event
        # Deviation from the average number of tracks, makes no track per event
        # possible!
        self.tracks_per_event_sigma = 1

        # Device settings
        # Sensor bias voltage for each device in volt
        self.dut_bias = [50] * self._n_duts
        # Sensor thickness for each device in um
        self.dut_thickness = [100] * self._n_duts
        # Detection threshold for each device in electrons, influences
        # efficiency!
        self.dut_threshold = [0] * self._n_duts
        # Noise for each device in electrons
        self.dut_noise = [50] * self._n_duts
        # Pixel size for each device in x / y in um
        self.dut_pixel_size = [(50, 50)] * self._n_duts
        # Number of pixel for each device in x / y
        self.dut_n_pixel = [(1000, 1000)] * self._n_duts
        # Efficiency for each device from 0. to 1. for hits above threshold
        self.dut_efficiencies = [1.] * self._n_duts
        # The effective material budget (sensor + passive compoonents) given in
        # total material distance / total radiation length
        # (https://cdsweb.cern.ch/record/1279627/files/PH-EP-Tech-Note-2010-013.pdf);
        # 0 means no multiple scattering; std. setting is the sensor thickness
        # made of silicon as material budget
        self.dut_material_budget = [
            self.dut_thickness[i] * 1e-4 / 9.370 for i in range(self._n_duts)]

        # Digitization settings
        self.digitization_charge_sharing = True
        # Shuffle hit per event to challange track finding
        self.digitization_shuffle_hits = True
        # Correction factor for charge cloud sigma(z) to take into account also
        # repulsion; for further info see NIMA 606 (2009) 508-516
        self.digitization_sigma_cc = 1.35
        # Translate hit position on DUT plane to channel indices (column / row)
        self.digitization_pixel_discretization = True

        # Internals
        self._hit_dtype = np.dtype([('event_number', np.int64), ('frame', np.uint32), (
            'column', np.uint16), ('row', np.uint16), ('charge', np.uint16)])

    def reset(self):
        ''' Reset to init configuration '''
        self._n_duts = 6  # Std. settinng for the number of DUTs
        self.set_random_seed(self.random_seed)
        self.set_std_settings()

    def create_data_and_store(self, base_file_name, n_events, chunk_size=100000):
        logging.info('Simulate %d events with %d DUTs', n_events, self._n_duts)

        # Special case: all events can be created at once
        if chunk_size > n_events:
            chunk_size = n_events

        # Create output h5 files with emtpy hit ta
        output_files = []
        hit_tables = []
        for dut_index in range(self._n_duts):
            output_files.append(
                tb.open_file(base_file_name + '_DUT%d.h5' % dut_index, mode='w'))
            hit_tables.append(output_files[dut_index].create_table(
                where=output_files[dut_index].root,
                name='Hits',
                description=self._hit_dtype,
                title='Simulated hits for test beam analysis',
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False)))

        if n_events * self.tracks_per_event > 100000:
            show_progress = True
            progress_bar = tqdm(total=len(range(0, n_events, chunk_size)), ncols=80)
        else:
            show_progress = False
        # Fill output files in chunks
        for chunk_index, _ in enumerate(range(0, n_events, chunk_size)):
            actual_events, actual_digitized_hits = self._create_data(
                start_event_number=chunk_index * chunk_size, n_events=chunk_size)
            for dut_index in range(self._n_duts):
                actual_dut_events, actual_dut_hits = actual_events[
                    dut_index], actual_digitized_hits[dut_index]
                actual_hits = np.zeros(
                    shape=actual_dut_events.shape[0], dtype=self._hit_dtype)
                actual_hits['event_number'] = actual_dut_events
                actual_hits['column'] = actual_dut_hits.T[0]
                actual_hits['row'] = actual_dut_hits.T[1]
                actual_hits['charge'] = actual_dut_hits.T[
                    2] / 10.  # One charge LSB corresponds to 10 electrons
                hit_tables[dut_index].append(actual_hits)
            if show_progress:
                progress_bar.update(1)
        if show_progress:
            progress_bar.close()

        for output_file in output_files:
            output_file.close()

    def _create_tracks(self, n_tracks):
        '''Creates tracks with gaussian distributed angles at gaussian distributed positions at z=0.

        Parameters
        ----------
        n_tracks: number
            Number of tracks created

        Returns
        -------
        Four np.arrays with position x,y and angles phi, theta
        '''

        logging.debug('Create %d tracks at x/y = (%d/%d +- %d/%d) um and theta = (%d +- %d) mRad', n_tracks, self.beam_position[0], self.beam_position[1], self.beam_position_sigma[0], self.beam_position_sigma[1], self.beam_angle, self.beam_angle_sigma)

        if self.beam_angle / 1000. > np.pi or self.beam_angle / 1000. < 0:
            raise ValueError('beam_angle has to be between [0..pi] Rad')

        if self.beam_position_sigma[0] != 0:
            track_positions_x = np.random.normal(self.beam_position[0], self.beam_position_sigma[0], n_tracks)

        else:
            track_positions_x = np.repeat(self.beam_position[0], repeats=n_tracks)  # Constant x = mean_x

        if self.beam_position_sigma[1] != 0:
            track_positions_y = np.random.normal(self.beam_position[1], self.beam_position_sigma[0], n_tracks)

        else:
            track_positions_y = np.repeat(self.beam_position[1], repeats=n_tracks)  # Constant y = mean_y

        if self.beam_angle_sigma != 0:
            track_angles_theta = np.abs(np.random.normal(self.beam_angle / 1000., self.beam_angle_sigma / 1000., size=n_tracks))  # Gaussian distributed theta
        else:  # Allow sigma = 0
            track_angles_theta = np.repeat(self.beam_angle / 1000., repeats=n_tracks)  # Constant theta = 0

        # Cut down to theta = 0 .. Pi
        iterations = 0
        while(np.any(track_angles_theta > np.pi) or np.any(track_angles_theta < 0)):
            track_angles_theta[track_angles_theta > np.pi] = np.random.normal(
                self.beam_angle,
                self.beam_angle_sigma,
                size=track_angles_theta[track_angles_theta > np.pi].shape[0])
            track_angles_theta[track_angles_theta < 0] = np.random.normal(
                self.beam_angle,
                self.beam_angle_sigma,
                size=track_angles_theta[track_angles_theta < 0].shape[0])
            iterations += 1
            if iterations > 1000:
                raise RuntimeError(
                    'Cannot create theta between [0, 2 Pi[, decrease track angle sigma!')

        if (self.beam_direction[0] != self.beam_direction[1]):
            track_angles_phi = np.random.uniform(self.beam_direction[0], self.beam_direction[1], size=n_tracks)  # Flat distributed in phi
        else:
            # Constant phi = self.beam_direction
            track_angles_phi = np.repeat(self.beam_direction[0], repeats=n_tracks)

        return track_positions_x, track_positions_y, track_angles_phi, track_angles_theta

    def _create_hits_from_tracks(self, track_positions_x, track_positions_y, track_angles_phi, track_angles_theta):
        '''Creates exact intersection points (x, y, z) for each track taking into accout individual DUT z_positions and rotations.
        The DUT dimension are approximated by a infinite expanding plane. Tracks not intersecting the planes in one point have the
        intersection set to NaN.

        The tracks are defined in spherical coordinates with the position at z = 0 (track_positions_x, track_positions_y) and
        an angle (track_angles_phi, track_angles_theta).

        Returns
        -------
        Iterable of np.arrays
            For each DUT: a np.array with intersections for each track
        '''
        logging.debug('Intersect tracks with DUTs to create hits')

        intersections = []
        track_positions = np.column_stack((track_positions_x, track_positions_y, np.zeros_like(
            track_positions_x)))  # Track position at z = 0

        # Multiple scattering changes the angle at each plane, thus these
        # temporary array have to be filled
        actual_track_angles_phi = track_angles_phi
        actual_track_angles_theta = track_angles_theta

        # Loop over DUTs
        for dut_index, z_position in enumerate(self.z_positions):

            # Deduce geometry in global coordinates of actual DUT from DUT
            # position and rotation
            dut_position = np.array([self.offsets[dut_index][0], self.offsets[dut_index][
                                    1], z_position])  # Actual DUT position in global coordinates
            # Calculate plane x/y direction vectors in global coordinate system, taking into account DUT rotations around x/y axis
            # z-axis rotations do not influence the intersection with a plane
            # not expanding in z
            rotation_matrix = geometry_utils.rotation_matrix(
                *self.rotations[dut_index])
            basis_global = rotation_matrix.T.dot(
                np.eye(3))  # TODO: why transposed?
            # Normal vector of the actual DUT plane in the global coordinate
            # system, needed for line intersection
            # geometry_utils.get_plane_normal(direction_plane_x_global, direction_plane_y_global)
            normal_plane = basis_global[2]

            # Track does not scatter before first plane, thus just extrapolate
            # from x, y, z = (track_positions_x, track_positions_y, 0) to first
            # plane
            if dut_index == 0:
                track_directions = np.column_stack((geometry_utils.spherical_to_cartesian(
                    phi=actual_track_angles_phi,
                    theta=actual_track_angles_theta,
                    r=1.)))  # r does not define a direction in spherical coordinates, any r > 0 can be used

                actual_intersections = geometry_utils.get_line_intersections_with_plane(
                    line_origins=track_positions,
                    line_directions=track_directions,
                    position_plane=dut_position,
                    normal_plane=normal_plane)

            # Extrapolate from last plane position with last track angle to
            # this plane
            else:
                track_directions = np.column_stack((geometry_utils.spherical_to_cartesian(
                    phi=actual_track_angles_phi,
                    theta=actual_track_angles_theta,
                    r=1.)))  # r does not define a direction in spherical coordinates, any r > 0 can be used

                actual_intersections = geometry_utils.get_line_intersections_with_plane(
                    line_origins=intersections[-1],
                    line_directions=track_directions,
                    position_plane=dut_position,
                    normal_plane=normal_plane)

            # Scatter at actual plane, omit virtual planes (material_budget =
            # 0) and last plane
            if self.dut_material_budget[dut_index] != 0 and dut_index != len(self.z_positions) - 1:
                # Calculated the change of the direction vector due to multiple scattering, TODO: needs cross check
                # Vector addition in spherical coordinates needs transformation
                # into cartesian space:
                # http://math.stackexchange.com/questions/790057/how-to-sum-2-vectors-in-spherical-coordinate-system
                # r does not define a direction in spherical coordinates, any r
                # > 0 can be used
                x, y, z = geometry_utils.spherical_to_cartesian(
                    actual_track_angles_phi, actual_track_angles_theta, r=1.)

                # Calculate scattering in spherical coordinates
                scattering_phi = np.random.uniform(0, 2. * np.pi, size=actual_track_angles_phi.shape[0])
                # Scattering distribution theta_0, calculated from DUT material
                # budget
                theta_0 = self._scattering_angle_sigma(material_budget=self.dut_material_budget[dut_index])
                # Change theta angles due to scattering, abs because theta is
                # defined [0..np.pi]
                scattering_theta = np.abs(np.random.normal(0, theta_0, actual_track_angles_theta.shape[0]))

                # Add scattering to direction vector in cartesian coordinates
                # r does not define a direction in spherical coordinates, any r
                # > 0 can be used
                dx, dy, _ = geometry_utils.spherical_to_cartesian(scattering_phi, scattering_theta, r=1.)
                x += dx
                y += dy
                # r does not define a direction in spherical coordinates and is omitted
                actual_track_angles_phi, actual_track_angles_theta, _ = geometry_utils.cartesian_to_spherical(x, y, z)

            # Add intersections of actual DUT to result
            intersections.append(actual_intersections)
#         print intersections[0].shape
#         import matplotlib.pyplot as plt
#         plt.hist(intersections[0][:, 0], bins=100, alpha=0.2)
#         plt.hist(intersections[0][:, 1], bins=100, alpha=0.2)
#         plt.hist(intersections[0][:, 2], bins=100, alpha=0.2)
#         plt.hist(intersections[1][:, 0], bins=100, alpha=0.2)
#         plt.hist(intersections[1][:, 1], bins=100, alpha=0.2)
#         plt.hist(intersections[1][:, 2], bins=100, alpha=0.2)
#         plt.show()
#         import matplotlib.pyplot as plt
#         for j in range(10):
#             plt.clf()
#             x = [intersections[i][j][0] for i in range(self._n_duts)]
#             y = [intersections[i][j][1] for i in range(self._n_duts)]
#             z = [intersections[i][j][2] for i in range(self._n_duts)]
#             plt.plot(z, x, '.-', label='x')
#             plt.plot(z, y, '.-', label='y')
#             plt.legend()
#             plt.show()
        return intersections

    def _digitize_hits(self, event_number, hits):
        ''' Takes the Monte Carlo hits and transfers them to the local DUT coordinate system and discretizes the position and creates additional hit belonging to a cluster.'''
        logging.debug('Digitize hits')
        digitized_hits = []
        # The event number index can be different for each DUT due to noisy
        # pixel and charge sharing hits
        event_numbers = []

        for dut_index, dut_hits in enumerate(hits):  # Loop over DUTs
            # Since actual_event_number is changed depending on the DUT this is
            # needed
            actual_event_number = event_number.copy()
            # Transform hits from global coordinate system into local
            # coordinate system of actual DUT
            transformation_matrix = geometry_utils.global_to_local_transformation_matrix(
                x=self.offsets[dut_index][0],  # Get the transformation matrix
                y=self.offsets[dut_index][1],
                z=self.z_positions[dut_index],
                alpha=self.rotations[dut_index][0],
                beta=self.rotations[dut_index][1],
                gamma=self.rotations[dut_index][2])
            dut_hits[:, 0], dut_hits[:, 1], dut_hits[:, 2] = geometry_utils.apply_transformation_matrix(
                x=dut_hits[:, 0],
                y=dut_hits[:, 1],
                z=dut_hits[:, 2],
                transformation_matrix=transformation_matrix)

            # Output hit digits, with x/y information and charge
            # Create new array with additional charge column
            dut_hits_digits = np.zeros(shape=(dut_hits.shape[0], 3))
            dut_hits_digits[:, 2] = self._get_charge_deposited(
                dut_index, n_entries=dut_hits.shape[0])  # Fill charge column

            # Calculate discretized pixel hit position in x,y = column/row
            # (digit)
            if self.digitization_pixel_discretization:
                dut_hits_digits[:, :2] = dut_hits[
                    :, :2] / np.array(self.dut_pixel_size[dut_index])  # Position in pixel numbers
                # Pixel discretization, column/row index start from 1
                dut_hits_digits[:, :2] = np.around(
                    dut_hits_digits[:, :2] - 0.5) + 1

                # Create cluster from seed hits dut to charge sharing
                if self.digitization_charge_sharing:
                    # Calculate the relative position within the pixel, origin
                    # is in the center
                    relative_position = dut_hits[
                        :, :2] - (dut_hits_digits[:, :2] - 0.5) * self.dut_pixel_size[dut_index]
                    dut_hits_digits, n_hits_per_event = _add_charge_sharing_hits(
                        relative_position,  # This function takes 75 % of the time
                        hits_digits=dut_hits_digits,
                        max_column=self.dut_n_pixel[dut_index][0],
                        max_row=self.dut_n_pixel[dut_index][1],
                        thickness=self.dut_thickness[dut_index],
                        pixel_size_x=self.dut_pixel_size[dut_index][0],
                        pixel_size_y=self.dut_pixel_size[dut_index][1],
                        temperature=self.temperature,
                        bias=self.dut_bias[dut_index])
                    actual_event_number = np.repeat(actual_event_number, n_hits_per_event)

                # Delete hits outside of the DUT
                selection_x = np.logical_and(dut_hits_digits.T[0] > 0, dut_hits_digits.T[0] <= self.dut_n_pixel[dut_index][0])  # Hits that are inside the x dimension of the DUT
                selection_y = np.logical_and(dut_hits_digits.T[1] > 0, dut_hits_digits.T[1] <= self.dut_n_pixel[dut_index][1])  # Hits that are inside the y dimension of the DUT
                selection = np.logical_and(selection_x, selection_y)
                # reduce hits to valid hits
                dut_hits_digits = dut_hits_digits[selection]
                # Reducce event number to event number with valid hits
                actual_event_number = actual_event_number[selection]
            else:  # No position digitization
                dut_hits_digits[:, :2] = dut_hits[:, :2]
                # Hits can only have a positive position
                selection = np.logical_and(
                    dut_hits_digits.T[0] > 0, dut_hits_digits.T[1] > 0)
                # reduce hits to valid hits
                dut_hits_digits = dut_hits_digits[selection]
                # Reducce event number to event number with valid hits
                actual_event_number = actual_event_number[selection]

            # Mask hits due to inefficiency
            selection = np.ones_like(actual_event_number, dtype=np.bool)
            if self.dut_efficiencies[dut_index] < 1.:
                hit_indices = np.arange(
                    actual_event_number.shape[0])  # Indices of hits
                np.random.shuffle(hit_indices)  # shuffle hit indeces
                n_inefficient_hit = int(
                    hit_indices.shape[0] * (1. - self.dut_efficiencies[dut_index]))
                selection[hit_indices[:n_inefficient_hit]] = False

            dut_hits_digits = dut_hits_digits[selection]
            actual_event_number = actual_event_number[selection]

            # Add noise to charge
            if self.dut_noise[dut_index] != 0:
                dut_hits_digits[:, 2] += np.random.normal(0, self.dut_noise[dut_index], dut_hits_digits[:, 2].shape[0])

            # Delete hits below threshold
            if self.dut_threshold[dut_index] != 0:
                actual_event_number = actual_event_number[dut_hits_digits[:, 2] >= self.dut_threshold[dut_index]]
                dut_hits_digits = dut_hits_digits[dut_hits_digits[:, 2] >= self.dut_threshold[dut_index]]

            # Append results
            digitized_hits.append(dut_hits_digits)
            event_numbers.append(actual_event_number)

        return (event_numbers, digitized_hits)

    def _create_data(self, start_event_number=0, n_events=10000):
        # Calculate the number of tracks per event
        if self.tracks_per_event_sigma > 0:
            n_tracks_per_event = np.random.normal(self.tracks_per_event, self.tracks_per_event_sigma, n_events).astype(np.int32)
        else:
            n_tracks_per_event = np.ones(n_events, dtype=np.int32) * self.tracks_per_event
        # One cannot have less than 0 tracks per event, this will be triggered
        # events without a track
        n_tracks_per_event[n_tracks_per_event < 0] = 0

        # Create event number
        events = np.arange(n_events)
        # Create an event number of events with tracks
        event_number = np.repeat(events, n_tracks_per_event).astype(np.int64)
        event_number += start_event_number

        # Reduce to n_tracks_per_event > 0
        n_tracks = n_tracks_per_event.sum()
        # Create per event n track info, needed for hit shuffling
        n_tracks_per_event = np.repeat(n_tracks_per_event, n_tracks_per_event)

        # Create tracks
        track_positions_x, track_positions_y, track_angles_phi, track_angles_theta = self._create_tracks(n_tracks)

        # Create MC hits
        hits = self._create_hits_from_tracks(track_positions_x, track_positions_y, track_angles_phi, track_angles_theta)

        # Suffle event hits to simulate unordered hit data per trigger
        if self.digitization_shuffle_hits:
            for index, actual_dut_hits in enumerate(hits):
                # + Index is a trick to shuffle different for each device
                hits[index] = shuffle_event_hits(event_number, n_tracks_per_event, actual_dut_hits, self.random_seed + index)

        # Create detector response: digitized hits
        hits_digitized = self._digitize_hits(event_number, hits)

        return hits_digitized

    def _get_charge_deposited(self, dut_index, n_entries, eta=1.):
        ''' Calculates the charge distribution wich is approximated by a Landau and returns n_entries random samples from this
        distribution. The device thickness defines the MPV.

        '''

        x = np.linspace(0., 100., 10000)
        # eta is different depending on the device thickness; this is neglected
        # here
        y = pylandau.landau(x, mpv=10., eta=eta)
        p = y / np.sum(y)  # Propability by normalization to integral

        mpv = 71 * self.dut_thickness[dut_index]
        charge = x * mpv / 10.
#         from matplotlib import pyplot as plt
#         plt.plot(charge, p)
#         plt.show()

        return np.random.choice(charge, n_entries, p=p)

    def _scattering_angle_sigma(self, material_budget, charge_number=1):
        '''Calculates the scattering angle sigma for multiple scattering simulation. A Gaussian distribution
        is assumed with the sigma calculated here.

        Parameters
        ----------
        material_budget : number
            total distance / total radiation length
        charge number: int
            charge number of scattering particles, usually 1
        '''

        if material_budget == 0:
            return 0
        return 13.6 / self.beam_momentum * charge_number * np.sqrt(material_budget) * (1 + 0.038 * np.log(material_budget))


if __name__ == '__main__':
    simulate_data = SimulateData(0)
    simulate_data.dut_material_budget = [0] * simulate_data.n_duts
    simulate_data.rotations[1] = (-np.pi / 6., 0., 0.)
    simulate_data.beam_angle_sigma = 100
    simulate_data.beam_position_sigma = (0, 0)
    simulate_data.create_data_and_store('simulated_data', n_events=1000000)


# TEST: Plot charge sharing
#     import matplotlib.pyplot as plt
#     from matplotlib import cm
#     from mpl_toolkits.mplot3d.axes3d import Axes3D
#     from itertools import product, combinations
#     import mpl_toolkits.mplot3d.art3d as art3d
#     from matplotlib.patches import Circle, PathPatch, Rectangle
#     from matplotlib.widgets import Slider, Button, RadioButtons
#
#     simulate_data.digitization_sigma_cc = 1.
#     simulate_data.dut_pixel_size = [(50, 250)] * simulate_data.n_duts
#     simulate_data.dut_bias = [60] * simulate_data.n_duts
#
#     x_min, x_max, y_min, y_max, dx, dy = -150, 150, -150, 150, 4, 4
#
#     fig = plt.figure(figsize=(14, 6))
#     ax = fig.gca(projection='3d')
# l = ax.plot_wireframe(x_grid, y_grid, simulate_data._calc_charge_fraction(dut_index=0, position=(x_grid, y_grid), z=200, pixel_index=(0, 0)), label='cdf', color='blue', alpha=0.3)
#     x, y, z = [], [], []
#
#     for ix in np.arange(x_min, x_max, dx):
#         for iy in np.arange(y_min, y_max, dy):
#             x.append(ix)
#             y.append(iy)
#             z.append(_calc_charge_fraction(position_x=ix, position_y=iy, position_z=200, pixel_index_x=0, pixel_index_y=0, pixel_size_x=50, pixel_size_y=50, temperature=300, bias=60, digitization_sigma_cc=1.))
#     ax.plot(x, y, z, label='cdf', color='blue', alpha=0.3)
#
#     x, y, z = [], [], []
#     for ix in np.arange(x_min, x_max, dx):
#         for iy in np.arange(y_min, y_max, dy):
#             x.append(ix)
#             y.append(iy)
#             z.append(_calc_charge_fraction(position_x=ix, position_y=iy, position_z=200, pixel_index_x=1, pixel_index_y=0, pixel_size_x=50, pixel_size_y=50, temperature=300, bias=60, digitization_sigma_cc=1.))
#     ax.plot(x, y, z, label='cdf', color='red', alpha=0.3)
#     plt.show()
