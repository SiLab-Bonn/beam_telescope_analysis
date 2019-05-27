from __future__ import division

import numpy as np
from numpy import linalg
from numba import njit

from beam_telescope_analysis.tools import geometry_utils


@njit
def _filter_predict(transition_matrix, transition_covariance,
                    transition_offset, current_filtered_state,
                    current_filtered_state_covariance):
    """Calculates the predicted state and its covariance matrix. Prediction
    is done on whole track chunk with size chunk_size.

    Parameters
    ----------
    transition_matrix : [chunk_size, n_dim_state, n_dim_state] array
        state transition matrix from time t to t+1.
    transition_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix for state transition from time t to t+1.
    transition_offset : [chunk_size, n_dim_state] array
        offset for state transition from time t to t+1.
    current_filtered_state: [chunk_size, n_dim_state] array
        filtered state at time t.
    current_filtered_state_covariance: [chunk_size, n_dim_state, n_dim_state] array
        covariance of filtered state at time t.

    Returns
    -------
    predicted_state : [chunk_size, n_dim_state] array
        predicted state at time t+1.
    predicted_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of predicted state at time t+1.
    """
    predicted_state = _vec_mul(transition_matrix, current_filtered_state) + transition_offset

    predicted_state_covariance = _mat_mul(transition_matrix,
                                          _mat_mul(current_filtered_state_covariance,
                                                   _mat_trans(transition_matrix))) + transition_covariance

    return predicted_state, predicted_state_covariance


def _filter_correct(observation_matrix, observation_covariance,
                    observation_offset, predicted_state,
                    predicted_state_covariance, observation):
    r"""Filters a predicted state with the Kalman Filter. Filtering
    is done on whole track chunk with size chunk_size.

    Parameters
    ----------
    observation_matrix : [chunk_size, n_dim_obs, n_dim_obs] array
        observation matrix for time t.
    observation_covariance : [chunk_size, n_dim_obs, n_dim_obs] array
        covariance matrix for observation at time t.
    observation_offset : [chunk_size, n_dim_obs] array
        offset for observation at time t.
    predicted_state : [chunk_size, n_dim_state] array
        predicted state at time t.
    predicted_state_covariance : [n_dim_state, n_dim_state] array
        covariance matrix of predicted state at time t.
    observation : [chunk_size, n_dim_obs] array
        observation at time t.  If observation is a masked array and any of
        its values are masked, the observation will be not included in filtering.

    Returns
    -------
    kalman_gain : [chunk_size, n_dim_state, n_dim_obs] array
        Kalman gain matrix for time t.
    filtered_state : [chunk_size, n_dim_state] array
        filtered state at time t.
    filtered_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of filtered state at time t.
    """
    predicted_observation = _vec_mul(observation_matrix, predicted_state) + observation_offset

    predicted_observation_covariance = _mat_mul(observation_matrix,
                                                _mat_mul(predicted_state_covariance, _mat_trans(observation_matrix))) + observation_covariance

    kalman_gain = _mat_mul(predicted_state_covariance,
                           _mat_mul(_mat_trans(observation_matrix),
                                    _mat_inverse(predicted_observation_covariance)))

    filtered_state = predicted_state + _vec_mul(kalman_gain, observation - predicted_observation)

    filtered_state_covariance = predicted_state_covariance - _mat_mul(kalman_gain,
                                                                      _mat_mul(observation_matrix,
                                                                               predicted_state_covariance))
    # update filtered state where no observation is available
    no_observation_indices = np.isnan(observation[:, 0])
    kalman_gain[no_observation_indices, :, :] = 0.0
    filtered_state[no_observation_indices, :] = predicted_state[no_observation_indices, :]
    filtered_state_covariance[no_observation_indices, :, :] = predicted_state_covariance[no_observation_indices, :, :]

    # Calculate chi2
    filtered_residuals = observation - _vec_mul(observation_matrix, filtered_state)
    filtered_residuals_covariance = observation_covariance - _mat_mul(observation_matrix, _mat_mul(filtered_state_covariance, _mat_trans(observation_matrix)))[:, :3, :3]
    chi2 = _vec_vec_mul(filtered_residuals, _vec_mul(_mat_inverse(filtered_residuals_covariance), filtered_residuals))

    return kalman_gain, filtered_state, filtered_state_covariance, chi2


def _filter(dut_planes, z_sorted_dut_indices, thetas, observations, select_fit_duts,
            transition_matrices, observation_matrices, transition_covariances,
            observation_covariances, transition_offsets, observation_offsets,
            initial_state, initial_state_covariance):
    """Apply the Kalman Filter. First a prediction of the state is done, then a filtering is
    done which includes the observations.

    Parameters
    ----------
    dut_planes : list
        List of DUT parameters (material_budget, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma).
    z_sorted_dut_indices : list
        List of DUT indices in the order reflecting their z position.
    thetas : list
        List of scattering angle root mean squares (RMS).
    observations : [chunk_size, n_timesteps, n_dim_obs] array
        observations (measurements) from times [0...n_timesteps-1]. If any of observations is masked,
        then observations[:, t] will be treated as a missing observation
        and will not be included in the filtering step.
    select_fit_duts : iterable
        List of DUTs which should be included in Kalman Filter. DUTs which are not in list
        were treated as missing measurements and will not be included in the Filtering step.
    transition_matrices : [chunk_size, n_timesteps-1, n_dim_state, n_dim_state] array-like
        matrices to transport states from t to t+1.
    observation_matrices : [chunk_size, n_timesteps, n_dim_obs, n_dim_state] array-like
        observation matrices.
    transition_covariances : [chunk_size, n_timesteps-1, n_dim_state,n_dim_state]  array-like
        covariance matrices of transition matrices.
    observation_covariances : [chunk_size, n_timesteps, n_dim_obs, n_dim_obs] array-like
        covariance matrices of observation matrices.
    transition_offsets : [chunk_size, n_timesteps-1, n_dim_state] array-like
        offsets of transition matrices.
    observation_offsets : [chunk_size, n_timesteps, n_dim_obs] array-like
        offsets of observations.
    initial_state : [chunk_size, n_dim_state] array-like
        initial value of state.
    initial_state_covariance : [chunk_size, n_dim_state, n_dim_state] array-like
        initial value for observation covariance matrices.

    Returns
    -------
    predicted_states : [chunk_size, n_timesteps, n_dim_state] array
        predicted states of times [0...t].
    predicted_state_covariances : [chunk_size, n_timesteps, n_dim_state, n_dim_state] array
        covariance matrices of predicted states of times [0...t].
    kalman_gains : [chunk_size, n_timesteps, n_dim_state] array
        Kalman gain matrices of times [0...t].
    filtered_states : [chunk_size, n_timesteps, n_dim_state] array
        filtered states of times [0...t].
    filtered_state_covariances : [chunk_size, n_timesteps, n_dim_state] array
        covariance matrices of filtered states of times [0...t].
    transition_matrices_update : [chunk_size, n_timesteps-1, n_dim_state, n_dim_state] array-like
        updated transition matrices in case of rotated planes.
    """
    chunk_size, n_timesteps, n_dim_obs = observations.shape
    n_dim_state = transition_covariances.shape[2]

    predicted_states = np.zeros((chunk_size, n_timesteps, n_dim_state))
    predicted_state_covariances = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))
    kalman_gains = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_obs))
    filtered_states = np.zeros((chunk_size, n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))
    chi2 = np.zeros((chunk_size, n_timesteps))
    # array where new transition matrices are stored, needed to pass it to kalman smoother
    transition_matrices_update = np.zeros_like(transition_covariances)

    for i, dut_index in enumerate(z_sorted_dut_indices):
        dut = dut_planes[dut_index]
        if i == 0:  # first DUT
            predicted_states[:, dut_index] = initial_state
            predicted_state_covariances[:, dut_index] = initial_state_covariance
        else:
            # slopes (directional vectors) of the filtered estimates
            slopes_filtered_state = np.column_stack((
                filtered_states[:, z_sorted_dut_indices[i - 1], 3],
                filtered_states[:, z_sorted_dut_indices[i - 1], 4],
                filtered_states[:, z_sorted_dut_indices[i - 1], 5]))

            # offsets (support vectors) of the filtered states
            offsets_filtered_state = np.column_stack((
                filtered_states[:, z_sorted_dut_indices[i - 1], 0],
                filtered_states[:, z_sorted_dut_indices[i - 1], 1],
                filtered_states[:, z_sorted_dut_indices[i - 1], 2]))

            # offsets of filtered state with actual plane (plane on which the filtered estimate should be predicted)
            offsets_filtered_state_actual_plane = geometry_utils.get_line_intersections_with_dut(
                line_origins=offsets_filtered_state,
                line_directions=slopes_filtered_state,
                translation_x=dut.translation_x,
                translation_y=dut.translation_y,
                translation_z=dut.translation_z,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)

            z_diff = offsets_filtered_state_actual_plane[:, 2] - offsets_filtered_state[:, 2]
            if np.any(z_diff < 0.0):
                raise ValueError("Z differences give values smaller zero.")

            # update transition matrix according to the DUT rotation
            transition_matrices[np.nonzero(slopes_filtered_state[:, 0])[0], z_sorted_dut_indices[i - 1], 0, 3] = (offsets_filtered_state_actual_plane[np.nonzero(slopes_filtered_state[:, 0])[0], 0] - offsets_filtered_state[np.nonzero(slopes_filtered_state[:, 0])[0], 0]) / slopes_filtered_state[np.nonzero(slopes_filtered_state[:, 0])[0], 0]
            transition_matrices[np.nonzero(slopes_filtered_state[:, 0] == 0)[0], z_sorted_dut_indices[i - 1], 0, 3] = z_diff[np.nonzero(slopes_filtered_state[:, 0] == 0)[0]]
            transition_matrices[np.nonzero(slopes_filtered_state[:, 1])[0], z_sorted_dut_indices[i - 1], 1, 4] = (offsets_filtered_state_actual_plane[np.nonzero(slopes_filtered_state[:, 1])[0], 1] - offsets_filtered_state[np.nonzero(slopes_filtered_state[:, 1])[0], 1]) / slopes_filtered_state[np.nonzero(slopes_filtered_state[:, 1])[0], 1]
            transition_matrices[np.nonzero(slopes_filtered_state[:, 1] == 0)[0], z_sorted_dut_indices[i - 1], 1, 4] = z_diff[np.nonzero(slopes_filtered_state[:, 1] == 0)[0]]
            transition_matrices[np.nonzero(slopes_filtered_state[:, 2])[0], z_sorted_dut_indices[i - 1], 2, 5] = (offsets_filtered_state_actual_plane[np.nonzero(slopes_filtered_state[:, 2])[0], 2] - offsets_filtered_state[np.nonzero(slopes_filtered_state[:, 2])[0], 2]) / slopes_filtered_state[np.nonzero(slopes_filtered_state[:, 2])[0], 2]
            transition_matrices[np.nonzero(slopes_filtered_state[:, 2] == 0)[0], z_sorted_dut_indices[i - 1], 2, 5] = z_diff[np.nonzero(slopes_filtered_state[:, 2] == 0)[0]]

            # update transition covariance matrix according to the DUT rotation
            transition_covariances[np.nonzero(slopes_filtered_state[:, 0])[0], z_sorted_dut_indices[i - 1], 0, 0] = np.square((offsets_filtered_state_actual_plane[np.nonzero(slopes_filtered_state[:, 0])[0], 0] - offsets_filtered_state[np.nonzero(slopes_filtered_state[:, 0])[0], 0]) / slopes_filtered_state[np.nonzero(slopes_filtered_state[:, 0])[0], 0]) * np.square(thetas[z_sorted_dut_indices[i - 1]])
            transition_covariances[np.nonzero(slopes_filtered_state[:, 0] == 0)[0], z_sorted_dut_indices[i - 1], 0, 0] = np.square(z_diff[np.nonzero(slopes_filtered_state[:, 0] == 0)[0]]) * np.square(thetas[z_sorted_dut_indices[i - 1]])
            transition_covariances[np.nonzero(slopes_filtered_state[:, 1])[0], z_sorted_dut_indices[i - 1], 3, 0] = ((offsets_filtered_state_actual_plane[np.nonzero(slopes_filtered_state[:, 1])[0], 1] - offsets_filtered_state[np.nonzero(slopes_filtered_state[:, 1])[0], 1]) / slopes_filtered_state[np.nonzero(slopes_filtered_state[:, 1])[0], 1]) * np.square(thetas[z_sorted_dut_indices[i - 1]])
            transition_covariances[np.nonzero(slopes_filtered_state[:, 1] == 0)[0], z_sorted_dut_indices[i - 1], 3, 0] = z_diff[np.nonzero(slopes_filtered_state[:, 1] == 0)[0]] * np.square(thetas[z_sorted_dut_indices[i - 1]])
            transition_covariances[np.nonzero(slopes_filtered_state[:, 2])[0], z_sorted_dut_indices[i - 1], 1, 1] = np.square((offsets_filtered_state_actual_plane[np.nonzero(slopes_filtered_state[:, 2])[0], 2] - offsets_filtered_state[np.nonzero(slopes_filtered_state[:, 2])[0], 2]) / slopes_filtered_state[np.nonzero(slopes_filtered_state[:, 2])[0], 2]) * np.square(thetas[z_sorted_dut_indices[i - 1]])
            transition_covariances[np.nonzero(slopes_filtered_state[:, 2] == 0)[0], z_sorted_dut_indices[i - 1], 1, 1] = np.square(z_diff[np.nonzero(slopes_filtered_state[:, 2] == 0)[0]]) * np.square(thetas[z_sorted_dut_indices[i - 1]])
            transition_covariances[np.nonzero(slopes_filtered_state[:, 0])[0], z_sorted_dut_indices[i - 1], 4, 1] = ((offsets_filtered_state_actual_plane[np.nonzero(slopes_filtered_state[:, 0])[0], 0] - offsets_filtered_state[np.nonzero(slopes_filtered_state[:, 0])[0], 0]) / slopes_filtered_state[np.nonzero(slopes_filtered_state[:, 0])[0], 0]) * np.square(thetas[z_sorted_dut_indices[i - 1]])
            transition_covariances[np.nonzero(slopes_filtered_state[:, 0] == 0)[0], z_sorted_dut_indices[i - 1], 4, 1] = z_diff[np.nonzero(slopes_filtered_state[:, 0] == 0)[0]] * np.square(thetas[z_sorted_dut_indices[i - 1]])
            transition_covariances[np.nonzero(slopes_filtered_state[:, 1])[0], z_sorted_dut_indices[i - 1], 0, 3] = ((offsets_filtered_state_actual_plane[np.nonzero(slopes_filtered_state[:, 1])[0], 1] - offsets_filtered_state[np.nonzero(slopes_filtered_state[:, 1])[0], 1]) / slopes_filtered_state[np.nonzero(slopes_filtered_state[:, 1])[0], 1]) * np.square(thetas[z_sorted_dut_indices[i - 1]])
            transition_covariances[np.nonzero(slopes_filtered_state[:, 1] == 0)[0], z_sorted_dut_indices[i - 1], 0, 3] = z_diff[np.nonzero(slopes_filtered_state[:, 1] == 0)[0]] * np.square(thetas[z_sorted_dut_indices[i - 1]])
            transition_covariances[np.nonzero(slopes_filtered_state[:, 2])[0], z_sorted_dut_indices[i - 1], 1, 4] = ((offsets_filtered_state_actual_plane[np.nonzero(slopes_filtered_state[:, 2])[0], 2] - offsets_filtered_state[np.nonzero(slopes_filtered_state[:, 2])[0], 2]) / slopes_filtered_state[np.nonzero(slopes_filtered_state[:, 2])[0], 2]) * np.square(thetas[z_sorted_dut_indices[i - 1]])
            transition_covariances[np.nonzero(slopes_filtered_state[:, 2] == 0)[0], z_sorted_dut_indices[i - 1], 1, 4] = z_diff[np.nonzero(slopes_filtered_state[:, 2] == 0)[0]] * np.square(thetas[z_sorted_dut_indices[i - 1]])

            # store updated transition matrices
            transition_matrices_update[:, z_sorted_dut_indices[i - 1]] = transition_matrices[:, z_sorted_dut_indices[i - 1]]

            # calculate prediction from filter
            predicted_states[:, dut_index], predicted_state_covariances[:, dut_index] = _filter_predict(
                transition_matrix=transition_matrices[:, z_sorted_dut_indices[i - 1]],  # next plane in -z direction
                transition_covariance=transition_covariances[:, z_sorted_dut_indices[i - 1]],  # next plane in -z direction
                transition_offset=transition_offsets[:, z_sorted_dut_indices[i - 1]],  # next plane in -z direction
                current_filtered_state=filtered_states[:, z_sorted_dut_indices[i - 1]],  # next plane in -z direction
                current_filtered_state_covariance=filtered_state_covariances[:, z_sorted_dut_indices[i - 1]])  # next plane in -z direction

            # TODO:
            # Check for offsets_filtered_state_actual_plane == predicted_states

        if dut_index in select_fit_duts:
            # DUT is a fit dut:
            # set filter to prediction where no hit is available,
            # otherwise calculate filtered state.
            kalman_gains[:, dut_index], filtered_states[:, dut_index], filtered_state_covariances[:, dut_index], chi2[:, dut_index] = _filter_correct(
                observation_matrix=observation_matrices[:, dut_index],
                observation_covariance=observation_covariances[:, dut_index],
                observation_offset=observation_offsets[:, dut_index],
                predicted_state=predicted_states[:, dut_index],
                predicted_state_covariance=predicted_state_covariances[:, dut_index],
                observation=observations[:, dut_index])
        else:
            # DUT is not a fit dut:
            # set filter to prediction.
            kalman_gains[:, dut_index] = np.zeros((chunk_size, n_dim_state, n_dim_obs), dtype=np.float64)
            filtered_states[:, dut_index] = predicted_states[:, dut_index]
            filtered_state_covariances[:, dut_index] = predicted_state_covariances[:, dut_index]

        # Set the offset to the track intersection with the tilted plane
        intersections = geometry_utils.get_line_intersections_with_dut(
            line_origins=filtered_states[:, dut_index, 0:3],
            line_directions=filtered_states[:, dut_index, 3:6],
            translation_x=dut.translation_x,
            translation_y=dut.translation_y,
            translation_z=dut.translation_z,
            rotation_alpha=dut.rotation_alpha,
            rotation_beta=dut.rotation_beta,
            rotation_gamma=dut.rotation_gamma)
        # set x/y/z
        filtered_states[:, dut_index, 0:3] = intersections

    return predicted_states, predicted_state_covariances, kalman_gains, filtered_states, filtered_state_covariances, transition_matrices_update, chi2


@njit
def _smooth_update(observation, observation_matrix, observation_covariance,
                   transition_matrix, filtered_state,
                   filtered_state_covariance, predicted_state,
                   predicted_state_covariance, next_smoothed_state,
                   next_smoothed_state_covariance):
    """Smooth a filtered state with a Kalman Smoother. Smoothing
    is done on whole track chunk with size chunk_size.

    Parameters
    ----------
    transition_matrix : [chunk_size, n_dim_state, n_dim_state] array
        transition matrix to transport state from time t to t+1.
    filtered_state : [chunk_size, n_dim_state] array
        filtered state at time t.
    filtered_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of filtered state at time t.
    predicted_state : [chunk_size, n_dim_state] array
        predicted state at time t+1.
    predicted_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of filtered state at time t+1.
    next_smoothed_state : [chunk_size, n_dim_state] array
        smoothed state at time t+1.
    next_smoothed_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of smoothed state at time t+1.

    Returns
    -------
    smoothed_state : [chunk_size, n_dim_state] array
        smoothed state at time t.
    smoothed_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of smoothed state at time t.
    kalman_smoothing_gain : [chunk_size, n_dim_state, n_dim_state] array
        smoothed Kalman gain matrix at time t.
    """
    kalman_smoothing_gain = _mat_mul(filtered_state_covariance,
                                     _mat_mul(_mat_trans(transition_matrix),
                                              _mat_inverse(predicted_state_covariance)))

    smoothed_state = filtered_state + _vec_mul(kalman_smoothing_gain,
                                               next_smoothed_state - predicted_state)

    smoothed_state_covariance = filtered_state_covariance + _mat_mul(kalman_smoothing_gain,
                                                                     _mat_mul((next_smoothed_state_covariance - predicted_state_covariance),
                                                                              _mat_trans(kalman_smoothing_gain)))

    # Calculate chi2
    smoothed_residuals = observation - _vec_mul(observation_matrix, smoothed_state)
    smoothed_residuals_covariance = observation_covariance - _mat_mul(observation_matrix, _mat_mul(smoothed_state_covariance, _mat_trans(observation_matrix)))[:, :3, :3]
    chi2 = _vec_vec_mul(smoothed_residuals, _vec_mul(_mat_inverse(smoothed_residuals_covariance), smoothed_residuals))

    return smoothed_state, smoothed_state_covariance, kalman_smoothing_gain, chi2


def _smooth(dut_planes, z_sorted_dut_indices,
            observations, observation_matrices, observation_covariances,
            transition_matrices, filtered_states,
            filtered_state_covariances, predicted_states,
            predicted_state_covariances):
    """Apply the Kalman Smoother to filtered states. Estimate the smoothed states.
    Smoothing is done on whole track chunk with size chunk_size.

    Parameters
    ----------
    dut_planes : list
        List of DUT parameters (material_budget, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma).
    z_sorted_dut_indices : list
        List of DUT indices in the order reflecting their z position.
    transition_matrices : [chunk_size, n_timesteps-1, n_dim_state, n_dim_state] array-like
        matrices to transport states from t to t+1 of times [0...t-1].
    filtered_states : [chunk_size, n_timesteps, n_dim_state] array
        filtered states of times [0...t].
    filtered_state_covariances : [chunk_size, n_timesteps, n_dim_state] array
        covariance matrices of filtered states of times [0...t].
    predicted_states : [chunk_size, n_timesteps, n_dim_state] array
        predicted states of times [0...t].
    predicted_state_covariances : [chunk_size, n_timesteps, n_dim_state, n_dim_state] array
        covariance matrices of predicted states of times [0...t].

    Returns
    -------
    smoothed_states : [chunk_size, n_timesteps, n_dim_state]
        smoothed states for times [0...n_timesteps-1].
    smoothed_state_covariances : [chunk_size, n_timesteps, n_dim_state, n_dim_state] array
        covariance matrices of smoothed states for times [0...n_timesteps-1].
    kalman_smoothing_gains : [chunk_size, n_timesteps-1, n_dim_state] array
        smoothed kalman gain matrices fot times [0...n_timesteps-2].
    """
    chunk_size, n_timesteps, n_dim_state = filtered_states.shape
    smoothed_states = np.zeros((chunk_size, n_timesteps, n_dim_state))
    smoothed_state_covariances = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))
    kalman_smoothing_gains = np.zeros((chunk_size, n_timesteps - 1, n_dim_state, n_dim_state))
    chi2 = np.zeros((chunk_size, n_timesteps))

    smoothed_states[:, -1] = filtered_states[:, -1]
    smoothed_state_covariances[:, -1] = filtered_state_covariances[:, -1]

    # reverse order for smoother
    for i, dut_index in enumerate(z_sorted_dut_indices[:-1][::-1]):
        dut = dut_planes[dut_index]
        smoothed_states[:, dut_index], smoothed_state_covariances[:, dut_index], kalman_smoothing_gains[:, dut_index], chi2[:, dut_index] = _smooth_update(
            observations[:, dut_index],
            observation_matrices[:, dut_index],
            observation_covariances[:, dut_index],
            transition_matrices[:, dut_index],
            filtered_states[:, dut_index],
            filtered_state_covariances[:, dut_index],
            predicted_states[:, z_sorted_dut_indices[::-1][i]],  # next plane in +z direction
            predicted_state_covariances[:, z_sorted_dut_indices[::-1][i]],  # next plane +z direction
            smoothed_states[:, z_sorted_dut_indices[::-1][i]],  # next plane +z direction
            smoothed_state_covariances[:, z_sorted_dut_indices[::-1][i]])  # next plane +z direction

        # Set the offset to the track intersection with the tilted plane
        intersections_smooth = geometry_utils.get_line_intersections_with_dut(
            line_origins=smoothed_states[:, dut_index, 0:3],
            line_directions=smoothed_states[:, dut_index, 3:6],
            translation_x=dut.translation_x,
            translation_y=dut.translation_y,
            translation_z=dut.translation_z,
            rotation_alpha=dut.rotation_alpha,
            rotation_beta=dut.rotation_beta,
            rotation_gamma=dut.rotation_gamma)

        smoothed_states[:, dut_index, 0:3] = intersections_smooth

    return smoothed_states, smoothed_state_covariances, kalman_smoothing_gains, chi2


@njit
def _vec_vec_mul(X, Y):
    '''Helper function to multiply 3D vector with 3D vector. Multiplication is done on last two axes.
    '''
    result = np.zeros((X.shape[0]))
    for l in range(X.shape[0]):
        # iterate through rows of X
        for i in range(X.shape[1]):
            result[l] += X[l][i] * Y[l][i]
    return result


@njit
def _mat_mul(X, Y):
    '''Helper function to multiply two 3D matrices. Multiplication is done on last two axes.
    '''
    result = np.zeros((X.shape[0], X.shape[1], Y.shape[2]))
    for l in range(X.shape[0]):
        # iterate through rows of X
        for i in range(X.shape[1]):
            # iterate through columns of Y
            for j in range(Y.shape[2]):
                # iterate through rows of Y
                for k in range(Y.shape[1]):
                    result[l][i][j] += X[l][i][k] * Y[l][k][j]
    return result


@njit
def _vec_mul(X, Y):
    '''Helper function to multiply 3D matrix with 3D vector. Multiplication is done on last two axes.
    '''
    result = np.zeros((X.shape[0], X.shape[1]))
    for l in range(X.shape[0]):
        # iterate through rows of X
        for i in range(X.shape[1]):
            # iterate through columns of Y
            for k in range(X.shape[2]):
                result[l][i] += X[l][i][k] * Y[l][k]
    return result


@njit
def _mat_trans(X):
    '''Helper function to calculate transpose of 3D matrix. Transposition is done on last two axes.
    '''
    result = np.zeros((X.shape[0], X.shape[2], X.shape[1]))
    for l in range(X.shape[0]):
        for i in range(X.shape[2]):
            for j in range(X.shape[1]):
                result[l][i][j] = X[l][j][i]

    return result


@njit
def _mat_inverse(X):
    '''Helper function to calculate inverese of 3D matrix. Inversion is done on last two axes.
    '''
    inv = np.zeros((X.shape))
    for i in range(X.shape[0]):
            inv[i] = linalg.pinv(X[i])
    return inv


class KalmanFilter(object):
    def smooth(self, dut_planes, z_sorted_dut_indices, thetas, observations, select_fit_duts,
               transition_offsets, observation_matrices, observation_offsets, observation_covariances,
               initial_state, initial_state_covariance):
        """Apply the Kalman Smoother to the observations. In the first step a filtering is done,
        afterwards a smoothing is done. Calculation is done on whole track chunk with size chunk_size.

        Parameters
        ----------
        dut_planes : list
            List of DUT parameters (material_budget, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma).
        z_sorted_dut_indices : list
            List of DUT indices in the order reflecting their z position.
        thetas : list
            List of scattering angle root mean squares (RMS).
        observations : [chunk_size, n_timesteps, n_dim_obs] array
            observations (measurements) from times [0...n_timesteps-1]. If any of observations is masked,
            then observations[:, t] will be treated as a missing observation
            and will not be included in the filtering step.
        select_fit_duts : iterable
            List of DUTs which should be included in Kalman Filter. DUTs which are not in list
            were treated as missing measurements and will not be included in the Filtering step.
        transition_offsets : [chunk_size, n_timesteps-1, n_dim_state] array-like
            offsets of transition matrices.
        observation_matrices : [chunk_size, n_timesteps, n_dim_obs, n_dim_state] array-like
            observation matrices.
        observation_offsets : [chunk_size, n_timesteps, n_dim_obs] array-like
            offsets of observations.
        observation_covariances : [chunk_size, n_timesteps, n_dim_obs, n_dim_obs] array-like
            covariance matrices of observation matrices.
        initial_state : [chunk_size, n_dim_state] array-like
            initial value of state.
        initial_state_covariance : [chunk_size, n_dim_state, n_dim_state] array-like
            initial value for observation covariance matrices.

        Returns
        -------
        smoothed_states : [chunk_size, n_timesteps, n_dim_state]
            smoothed states for times [0...n_timesteps-1].
        smoothed_state_covariances : [chunk_size, n_timesteps, n_dim_state, n_dim_state] array
            covariance matrices of smoothed states for times [0...n_timesteps-1].
        """

        # express transition matrices
        # transition matrices are filled already here.
        # If alignment is used, transition matrices are updated (in Kalman Filter) before each prediction step in order to take
        # rotations of planes into account.

        n_duts = len(dut_planes)
        chunk_size = observations.shape[0]
        n_dim_state = transition_offsets.shape[2]

        # express transition matrix
        transition_matrices = np.full((chunk_size, n_duts, n_dim_state, n_dim_state), fill_value=np.nan, dtype=np.float64)

        # express transition covariance matrix
        transition_covariances = np.full((chunk_size, n_duts, n_dim_state, n_dim_state), fill_value=np.nan, dtype=np.float64)

        transition_matrices[:, :, :, 0] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        transition_matrices[:, :, :, 1] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        transition_matrices[:, :, :, 2] = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        transition_matrices[:, :, :, 3] = np.array([[np.nan] * n_duts,
                                                    [0] * n_duts,
                                                    [0] * n_duts,
                                                    [1] * n_duts,
                                                    [0] * n_duts,
                                                    [0] * n_duts]).T
        transition_matrices[:, :, :, 4] = np.array([[0] * n_duts,
                                                    [np.nan] * n_duts,
                                                    [0] * n_duts,
                                                    [0] * n_duts,
                                                    [1] * n_duts,
                                                    [0] * n_duts]).T
        transition_matrices[:, :, :, 5] = np.array([[0] * n_duts,
                                                    [0] * n_duts,
                                                    [np.nan] * n_duts,
                                                    [0] * n_duts,
                                                    [0] * n_duts,
                                                    [1] * n_duts]).T

        # express transition covariance matrices, according to http://web-docs.gsi.de/~ikisel/reco/Methods/CovarianceMatrices-NIMA329-1993.pdf
        transition_covariances[:, :, :, 0] = np.array([[np.nan] * n_duts,
                                                       [0] * n_duts,
                                                       [0] * n_duts,
                                                       [np.nan] * n_duts,
                                                       [0] * n_duts,
                                                       [0] * n_duts]).T
        transition_covariances[:, :, :, 1] = np.array([[0] * n_duts,
                                                       [np.nan] * n_duts,
                                                       [0] * n_duts,
                                                       [0] * n_duts,
                                                       [np.nan] * n_duts,
                                                       [0] * n_duts]).T
        transition_covariances[:, :, :, 2] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        transition_covariances[:, :, :, 3] = np.array([[np.nan] * n_duts,
                                                       [0] * n_duts,
                                                       [0] * n_duts,
                                                       np.square(thetas),
                                                       [0] * n_duts,
                                                       [0] * n_duts]).T
        transition_covariances[:, :, :, 4] = np.array([[0] * n_duts,
                                                       [np.nan] * n_duts,
                                                       [0] * n_duts,
                                                       [0] * n_duts,
                                                       np.square(thetas),
                                                       [0] * n_duts]).T
        transition_covariances[:, :, :, 5] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        predicted_states, predicted_state_covariances, _, filtered_states, filtered_state_covariances, transition_matrices, chi2s_filter = _filter(
            dut_planes=dut_planes,
            z_sorted_dut_indices=z_sorted_dut_indices,
            thetas=thetas,
            select_fit_duts=select_fit_duts,
            observations=observations,
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            transition_covariances=transition_covariances,
            observation_covariances=observation_covariances,
            transition_offsets=transition_offsets,
            observation_offsets=observation_offsets,
            initial_state=initial_state,
            initial_state_covariance=initial_state_covariance)

        smoothed_states, smoothed_state_covariances, smoothed_kalman_gains, chi2s_smooth = _smooth(
            dut_planes=dut_planes,
            z_sorted_dut_indices=z_sorted_dut_indices,
            observations=observations,
            observation_matrices=observation_matrices,
            observation_covariances=observation_covariances,
            transition_matrices=transition_matrices,
            filtered_states=filtered_states,
            filtered_state_covariances=filtered_state_covariances,
            predicted_states=predicted_states,
            predicted_state_covariances=predicted_state_covariances)

        return smoothed_states, smoothed_state_covariances, chi2s_smooth
