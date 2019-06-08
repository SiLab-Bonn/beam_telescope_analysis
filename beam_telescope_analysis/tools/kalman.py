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
    # Set filtered state to predicted state where no observation is available.
    valid_hit_selection = ~np.isnan(observation[:, 0])
    kalman_gain[~valid_hit_selection, :, :] = 0.0  # Zero kalman gain
    filtered_state[~valid_hit_selection, :] = predicted_state[~valid_hit_selection, :]
    filtered_state_covariance[~valid_hit_selection, :, :] = predicted_state_covariance[~valid_hit_selection, :, :]

    # Calculate chi2 (only if observation available)
    filtered_residuals = observation[valid_hit_selection] - _vec_mul(observation_matrix[valid_hit_selection], filtered_state[valid_hit_selection])
    filtered_residuals_covariance = observation_covariance[valid_hit_selection] - _mat_mul(observation_matrix[valid_hit_selection], _mat_mul(filtered_state_covariance[valid_hit_selection], _mat_trans(observation_matrix[valid_hit_selection])))
    check_covariance_matrix(filtered_residuals_covariance)  # Sanity check for covariance matrix
    chi2 = _vec_vec_mul(filtered_residuals, _vec_mul(_mat_inverse(filtered_residuals_covariance), filtered_residuals))

    return kalman_gain, filtered_state, filtered_state_covariance, chi2


def _filter(dut_planes, z_sorted_dut_indices, thetas, select_fit_duts, observations,
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
    select_fit_duts : iterable
        List of DUTs which should be included in Kalman Filter. DUTs which are not in list
        were treated as missing measurements and will not be included in the Filtering step.
    observations : [chunk_size, n_timesteps, n_dim_obs] array
        observations (measurements) from times [0...n_timesteps-1]. If any of observations is masked,
        then observations[:, t] will be treated as a missing observation
        and will not be included in the filtering step.
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
    chi2 = np.full((chunk_size, n_timesteps), fill_value=np.nan)
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

            check_covariance_matrix(transition_covariances[:, z_sorted_dut_indices[i - 1]])  # Sanity check for covariance matrix

            # calculate prediction from filter
            predicted_states[:, dut_index], predicted_state_covariances[:, dut_index] = _filter_predict(
                transition_matrix=transition_matrices[:, z_sorted_dut_indices[i - 1]],  # next plane in -z direction
                transition_covariance=transition_covariances[:, z_sorted_dut_indices[i - 1]],  # next plane in -z direction
                transition_offset=transition_offsets[:, z_sorted_dut_indices[i - 1]],  # next plane in -z direction
                current_filtered_state=filtered_states[:, z_sorted_dut_indices[i - 1]],  # next plane in -z direction
                current_filtered_state_covariance=filtered_state_covariances[:, z_sorted_dut_indices[i - 1]])  # next plane in -z direction

            check_covariance_matrix(predicted_state_covariances[:, dut_index])  # Sanity check for covariance matrix

            # TODO:
            # Check for offsets_filtered_state_actual_plane == predicted_states

        valid_hit_selection = ~np.isnan(observations[:, dut_index, 0])
        if dut_index in select_fit_duts:
            # DUT is a fit dut:
            # set filter to prediction where no hit is available,
            # otherwise calculate filtered state.
            kalman_gains[:, dut_index], filtered_states[:, dut_index], filtered_state_covariances[:, dut_index], chi2[valid_hit_selection, dut_index] = _filter_correct(
                observation_matrix=observation_matrices[:, dut_index],
                observation_covariance=observation_covariances[:, dut_index],
                observation_offset=observation_offsets[:, dut_index],
                predicted_state=predicted_states[:, dut_index],
                predicted_state_covariance=predicted_state_covariances[:, dut_index],
                observation=observations[:, dut_index])

            chi2[~valid_hit_selection, dut_index] = np.nan  # No hit, thus no chi2
            check_covariance_matrix(filtered_state_covariances[:, dut_index])  # Sanity check for covariance matrix
        else:
            # DUT is not a fit dut:
            # set filter to prediction.
            kalman_gains[:, dut_index] = np.zeros((chunk_size, n_dim_state, n_dim_obs), dtype=np.float64)
            filtered_states[:, dut_index] = predicted_states[:, dut_index]
            filtered_state_covariances[:, dut_index] = predicted_state_covariances[:, dut_index]

            check_covariance_matrix(filtered_state_covariances[:, dut_index])  # Sanity check for covariance matrix

            # Calculate chi2 (only if observation available).
            filtered_residuals = observations[valid_hit_selection, dut_index] - _vec_mul(observation_matrices[valid_hit_selection, dut_index], filtered_states[valid_hit_selection, dut_index])
            # Note: need to add here covariance matrices, since in this case (filter equals to prediction) need to use the formula for predicted residual covariance
            filtered_residuals_covariance = observation_covariances[valid_hit_selection, dut_index] + _mat_mul(observation_matrices[valid_hit_selection, dut_index], _mat_mul(filtered_state_covariances[valid_hit_selection, dut_index], _mat_trans(observation_matrices[valid_hit_selection, dut_index])))
            check_covariance_matrix(filtered_residuals_covariance)  # Sanity check for covariance matrix
            chi2[valid_hit_selection, dut_index] = _vec_vec_mul(filtered_residuals, _vec_mul(_mat_inverse(filtered_residuals_covariance), filtered_residuals))
            chi2[~valid_hit_selection, dut_index] = np.nan  # No hit, thus no chi2

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

    # Final check for valid chi2
    if np.any(chi2[~np.isnan(chi2)] < 0.0):
        raise RuntimeError('Some chi-square values are negative (during filter step)!')

    return predicted_states, predicted_state_covariances, kalman_gains, filtered_states, filtered_state_covariances, transition_matrices_update, chi2


# @njit
def _smooth_update(observation, observation_matrix, observation_covariance,
                   transition_matrix, predicted_state_covariance, filtered_state,
                   filtered_state_covariance, next_predicted_state,
                   next_predicted_state_covariance, next_smoothed_state,
                   next_smoothed_state_covariance, dut_used_in_fit):
    """Smooth a filtered state with a Kalman Smoother. Smoothing
    is done on whole track chunk with size chunk_size.

    Parameters
    ----------
    observation : [chunk_size, n_dim_obs] array
        observations (measurements) from times [0...n_timesteps-1]. If any of observations is masked,
        then observations[:, t] will be treated as a missing observation
        and will not be included in the filtering step.
    observation_matrix : [chunk_size, n_dim_obs, n_dim_state] array-like
        observation matrices.
    observation_covariance : [chunk_size, n_dim_obs, n_dim_obs] array-like
            covariance matrices of observation.
    transition_matrix : [chunk_size, n_dim_state, n_dim_state] array
        transition matrix to transport state from time t to t+1.
    predicted_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of predicted state at time t.
    filtered_state : [chunk_size, n_dim_state] array
        filtered state at time t.
    filtered_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of filtered state at time t.
    next_predicted_state : [chunk_size, n_dim_state] array
        predicted state at time t+1.
    next_predicted_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of predicted state at time t+1.
    next_smoothed_state : [chunk_size, n_dim_state] array
        smoothed state at time t+1.
    next_smoothed_state_covariance : [chunk_size, n_dim_state, n_dim_state] array
        covariance matrix of smoothed state at time t+1.
    dut_used_in_fit : bool
        True if actual plane is used in fit (filtering step was done). If not used in fit, False.

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
                                              _mat_inverse(next_predicted_state_covariance)))

    smoothed_state = filtered_state + _vec_mul(kalman_smoothing_gain,
                                               next_smoothed_state - next_predicted_state)

    smoothed_state_covariance = filtered_state_covariance + _mat_mul(kalman_smoothing_gain,
                                                                     _mat_mul((next_smoothed_state_covariance - next_predicted_state_covariance),
                                                                              _mat_trans(kalman_smoothing_gain)))

    # Calculate chi2 (only if observation available)
    valid_hit_selection = ~np.isnan(observation[:, 0])
    smoothed_residuals = observation[valid_hit_selection] - _vec_mul(observation_matrix[valid_hit_selection], smoothed_state[valid_hit_selection])
    if dut_used_in_fit:
        smoothed_residuals_covariance = observation_covariance[valid_hit_selection] - _mat_mul(observation_matrix[valid_hit_selection], _mat_mul(smoothed_state_covariance[valid_hit_selection], _mat_trans(observation_matrix[valid_hit_selection])))
    else:
        residual_covariance_matrix = 2 * predicted_state_covariance - smoothed_state_covariance
        smoothed_residuals_covariance = observation_covariance[valid_hit_selection] + _mat_mul(observation_matrix[valid_hit_selection], _mat_mul(residual_covariance_matrix, _mat_trans(observation_matrix[valid_hit_selection])))
    check_covariance_matrix(smoothed_residuals_covariance)  # Sanity check for covariance matrix
    chi2 = _vec_vec_mul(smoothed_residuals, _vec_mul(_mat_inverse(smoothed_residuals_covariance), smoothed_residuals))

    return smoothed_state, smoothed_state_covariance, kalman_smoothing_gain, chi2


def _smooth(dut_planes, z_sorted_dut_indices, select_fit_duts,
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
    select_fit_duts : iterable
        List of DUTs which should be included in Kalman Filter. DUTs which are not in list
        were treated as missing measurements and will not be included in the Filtering step.
    observations : [chunk_size, n_timesteps, n_dim_obs] array
        observations (measurements) from times [0...n_timesteps-1]. If any of observations is masked,
        then observations[:, t] will be treated as a missing observation
        and will not be included in the filtering step.
    observation_matrices : [chunk_size, n_timesteps, n_dim_obs, n_dim_state] array-like
        observation matrices.
    observation_covariances : [chunk_size, n_timesteps, n_dim_obs, n_dim_obs] array-like
            covariance matrices of observation.
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

    last_dut_index = n_timesteps - 1

    # Smoother not done for last plane (At last plane smoothed states are the same as filtered states)
    smoothed_states[:, last_dut_index] = filtered_states[:, last_dut_index]
    smoothed_state_covariances[:, last_dut_index] = filtered_state_covariances[:, last_dut_index]

    valid_hit_selection = ~np.isnan(observations[:, last_dut_index, 0])
    # Calculate chi2 (only if observation available).
    smoothed_residuals = observations[valid_hit_selection, last_dut_index] - _vec_mul(observation_matrices[valid_hit_selection, last_dut_index], smoothed_states[valid_hit_selection, last_dut_index])
    if not np.all(np.linalg.eigvalsh(smoothed_state_covariances) >= 0.0):
        non_psd_selection = np.any(np.linalg.eigvalsh(smoothed_state_covariances) < 0.0, axis=1)  # get not positive semidifinite matrices
        smoothed_state_covariances[non_psd_selection] = _make_matrix_psd(smoothed_state_covariances[non_psd_selection])
    if last_dut_index in select_fit_duts:
        smoothed_residuals_covariance = observation_covariances[valid_hit_selection, last_dut_index] - _mat_mul(observation_matrices[valid_hit_selection, last_dut_index], _mat_mul(smoothed_state_covariances[valid_hit_selection, last_dut_index], _mat_trans(observation_matrices[valid_hit_selection, last_dut_index])))
    else:  # smoothed state is same as prediction thus use residual covariance for prediction (for last plane smoothed state is filtered state, if no used in fit filtered state is predicted state)
        smoothed_residuals_covariance = observation_covariances[valid_hit_selection, last_dut_index] + _mat_mul(observation_matrices[valid_hit_selection, last_dut_index], _mat_mul(smoothed_state_covariances[valid_hit_selection, last_dut_index], _mat_trans(observation_matrices[valid_hit_selection, last_dut_index])))
    check_covariance_matrix(smoothed_residuals_covariance)  # Sanity check for covariance matrix
    chi2[valid_hit_selection, last_dut_index] = _vec_vec_mul(smoothed_residuals, _vec_mul(_mat_inverse(smoothed_residuals_covariance), smoothed_residuals))
    chi2[~valid_hit_selection, last_dut_index] = np.nan  # No hit, thus set chi2 to nan

    # reverse order for smoother
    for i, dut_index in enumerate(z_sorted_dut_indices[:-1][::-1]):
        valid_hit_selection = ~np.isnan(observations[:, dut_index, 0])
        dut = dut_planes[dut_index]
        smoothed_states[:, dut_index], smoothed_state_covariances[:, dut_index], kalman_smoothing_gains[:, dut_index], chi2[valid_hit_selection, dut_index] = _smooth_update(
            observation=observations[:, dut_index],
            observation_matrix=observation_matrices[:, dut_index],
            observation_covariance=observation_covariances[:, dut_index],
            transition_matrix=transition_matrices[:, dut_index],
            predicted_state_covariance=predicted_state_covariances[:, dut_index],
            filtered_state=filtered_states[:, dut_index],
            filtered_state_covariance=filtered_state_covariances[:, dut_index],
            next_predicted_state=predicted_states[:, z_sorted_dut_indices[::-1][i]],  # next plane in +z direction
            next_predicted_state_covariance=predicted_state_covariances[:, z_sorted_dut_indices[::-1][i]],  # next plane +z direction
            next_smoothed_state=smoothed_states[:, z_sorted_dut_indices[::-1][i]],  # next plane +z direction
            next_smoothed_state_covariance=smoothed_state_covariances[:, z_sorted_dut_indices[::-1][i]],  # next plane +z direction
            dut_used_in_fit=True if dut_index in select_fit_duts else False,
        )

        chi2[~valid_hit_selection, dut_index] = np.nan  # No hit, thus no chi2
        check_covariance_matrix(smoothed_state_covariances[:, dut_index])  # Sanity check for covariance matrix

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

    # Final check for valid chi2
    if np.any(chi2[~np.isnan(chi2)] < 0.0):
        raise RuntimeError('Some chi-square values are negative (during smoothing step)!')

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
def _mat_inverse(X, atol=1e-5, rtol=1e-8):
    '''Helper function to calculate inverese of 3D matrix. Inversion is done on last two axes.
    '''
    inv = np.zeros((X.shape), dtype=np.float64)
    for i in range(X.shape[0]):
        inv[i] = linalg.inv(X[i])
        # Check if inverse was succesfull
        X_c = np.dot(X[i], np.dot(inv[i], X[i]))
        inv_c = np.dot(inv[i], np.dot(X[i], inv[i]))
        tol_X = atol + rtol * np.absolute(X_c)
        tol_inv = atol + rtol * np.absolute(inv_c)
        if np.any(np.absolute(X[i] - X_c) > tol_X) or np.any(np.absolute(inv[i] - inv_c) > tol_inv):
            raise RuntimeError('Matrix inversion failed!')
    return inv


def check_covariance_matrix(cov):
    ''' This function checks if the input covariance matrix is positive semi-definite (psd).
    In case it is not, it will try to make the matrix psd with the condition that the psd-correced matrix does not
    differ to much from the original one (works only if the matrix has very small negative eigenvalues, e.g. due to numerical precision, ...)
    '''
    # Check for postive semi-definite covariance matrix. In case they are not psd, make them psd.
    if not np.all(np.linalg.eigvalsh(cov) >= 0.0):
        non_psd_selection = np.any(np.linalg.eigvalsh(cov) < 0.0, axis=1)
        cov[non_psd_selection] = _make_matrix_psd(cov[non_psd_selection])


@njit
def _make_matrix_psd(A, atol=1e-5, rtol=1e-8):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    A3 = np.zeros((A.shape))
    for i in range(A.shape[0]):
        B = (A[i] + A[i].T) / 2
        _, s, V = np.linalg.svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3[i] = (A2 + A2.T) / 2

        if np.all(np.linalg.eigvalsh(A3[i]) >= 0.0):
            tol_A3 = atol + rtol * np.absolute(A3[i])
            if np.any(np.absolute(A[i] - A3[i]) > tol_A3):  # Check if corrected (psd) matrix did not change too much.
                raise RuntimeError('Output matrix differs too much from input matrix during nearest PSD')
            continue
        else:
            spacing = np.spacing(np.linalg.norm(A[i]))
            I_d = np.eye(A[i].shape[0])
            k = 1
            while not np.all(np.linalg.eigvalsh(A3[i]) >= 0.0):
                mineig = np.min(np.real(np.linalg.eigvalsh(A3[i])))
                A3[i] += I_d * (-mineig * k**2 + spacing)
                k += 1
            tol_A3 = atol + rtol * np.absolute(A3[i])
            if np.any(np.absolute(A[i] - A3[i]) > tol_A3):  # Check if corrected (psd) matrix did not change too much.
                raise RuntimeError('Output matrix differs too much from input matrix during nearest PSD')
    return A3


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

        smoothed_states, smoothed_state_covariances, _, chi2s_smooth = _smooth(
            dut_planes=dut_planes,
            z_sorted_dut_indices=z_sorted_dut_indices,
            select_fit_duts=select_fit_duts,
            observations=observations,
            observation_matrices=observation_matrices,
            observation_covariances=observation_covariances,
            transition_matrices=transition_matrices,
            filtered_states=filtered_states,
            filtered_state_covariances=filtered_state_covariances,
            predicted_states=predicted_states,
            predicted_state_covariances=predicted_state_covariances)

        return smoothed_states, smoothed_state_covariances, chi2s_smooth
