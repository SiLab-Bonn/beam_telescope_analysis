from __future__ import division

import numpy as np
from numpy import linalg
from numba import njit

from beam_telescope_analysis.tools import geometry_utils


@njit(cache=True)
def _filter_predict(reference_state, track_jacobian, local_scatter_gain_matrix, transition_covariance, current_filtered_state, current_filtered_state_covariance):
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

    # Extrapolate current filtered state (plane k -> plane k+1)
    predicted_state = _vec_mul(track_jacobian, current_filtered_state)

    # Extrapolate current filtered covariance (plane k -> plane k+1). Neglect air gap between detectors
    predicted_state_covariance = _mat_mul(track_jacobian,
                                          _mat_mul(current_filtered_state_covariance,
                                                   _mat_trans(track_jacobian)))

    # Add process noise to covariance matrix
    general_scatter_gain_matrix = _mat_mul(track_jacobian, local_scatter_gain_matrix)
    predicted_state_covariance += _mat_mul(general_scatter_gain_matrix,
                                           _mat_mul(transition_covariance,
                                                    _mat_trans(general_scatter_gain_matrix)))

    return predicted_state, predicted_state_covariance


def _filter_correct(reference_state, observation_matrix, observation_covariance, predicted_state, predicted_state_covariance, observation):
    r"""Filters a predicted state with the Kalman Filter. Filtering
    is done on whole track chunk with size chunk_size.

    Parameters
    ----------
    observation_matrix : [chunk_size, n_dim_obs, n_dim_obs] array
        observation matrix for time t.
    observation_covariance : [chunk_size, n_dim_obs, n_dim_obs] array
        covariance matrix for observation at time t.
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

    predicted_observation = _vec_mul(observation_matrix, predicted_state + reference_state)

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
    filtered_residuals = observation[valid_hit_selection] - _vec_mul(observation_matrix[valid_hit_selection], filtered_state[valid_hit_selection] + reference_state[valid_hit_selection])
    filtered_residuals_covariance = observation_covariance[valid_hit_selection] - _mat_mul(observation_matrix[valid_hit_selection], _mat_mul(filtered_state_covariance[valid_hit_selection], _mat_trans(observation_matrix[valid_hit_selection])))
    check_covariance_matrix(filtered_residuals_covariance)  # Sanity check for covariance matrix
    chi2 = _vec_vec_mul(filtered_residuals, _vec_mul(_mat_inverse(filtered_residuals_covariance), filtered_residuals))

    return kalman_gain, filtered_state, filtered_state_covariance, chi2


def _filter_f(dut_planes, reference_states, z_sorted_dut_indices, select_fit_duts, observations, observation_matrices, transition_covariances, observation_covariances, initial_state, initial_state_covariance):
    """Apply the Kalman Filter. First a prediction of the state is done, then a filtering is
    done which includes the observations.

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
    transition_covariances : [chunk_size, n_timesteps-1, n_dim_state,n_dim_state]  array-like
        covariance matrices of transition matrices.
    observation_covariances : [chunk_size, n_timesteps, n_dim_obs, n_dim_obs] array-like
        covariance matrices of observation matrices.
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
    """
    chunk_size, n_timesteps, n_dim_obs = observations.shape
    n_dim_state = initial_state_covariance.shape[2]

    predicted_states = np.zeros((chunk_size, n_timesteps, n_dim_state))
    predicted_state_covariances = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))
    kalman_gains = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_obs))
    filtered_states = np.zeros((chunk_size, n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))
    chi2 = np.full((chunk_size, n_timesteps), fill_value=np.nan)
    Js = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))

    for i, dut_index in enumerate(z_sorted_dut_indices):
        # Get actual reference state
        reference_state = reference_states[:, dut_index, :]

        if i == 0:  # first DUT: Set predicted state to initial state
            predicted_states[:, dut_index] = initial_state
            predicted_state_covariances[:, dut_index] = initial_state_covariance
        else:
            dut = dut_planes[dut_index - 1]  # we use filter from plane before [dut_index - 1] and extrapolate/predict onto actual plane [dut_index]
            next_dut = dut_planes[dut_index]  # we want to get prediction onto actual plane [dut_index]
            check_covariance_matrix(transition_covariances[:, dut_index - 1])  # Sanity check for covariance matrix

            # Local to global transformation
            rotation_matrix = geometry_utils.rotation_matrix(
                                alpha=dut.rotation_alpha,
                                beta=dut.rotation_beta,
                                gamma=dut.rotation_gamma)
            rotation_matrix_next = geometry_utils.rotation_matrix(
                                alpha=next_dut.rotation_alpha,
                                beta=next_dut.rotation_beta,
                                gamma=next_dut.rotation_gamma)

            dut_position = np.array([dut.translation_x, dut.translation_y, dut.translation_z])
            next_dut_position = np.array([next_dut.translation_x, next_dut.translation_y, next_dut.translation_z])

            # Transition matrix: 0: not defined/needed, 1: 0->1, 2: 1->2 (k: k-1 --> k)
            Js[:, dut_index, :, :] = _calculate_track_jacobian_new(
                reference_state=reference_states[:, dut_index - 1, :],  # use reference state from before
                dut_position=np.tile(dut_position, reps=(reference_state.shape[0], 1)),
                next_dut_position=np.tile(next_dut_position, reps=(reference_state.shape[0], 1)),  # extrapolates to this position
                rotation_matrix=np.tile(rotation_matrix, reps=(reference_state.shape[0], 1, 1)),
                rotation_matrix_next=np.tile(rotation_matrix_next, reps=(reference_state.shape[0], 1, 1)))

            # According to Wolin et al. paper
            Gl_det = _calculate_scatter_gain_matrix(reference_state=reference_states[:, dut_index - 1, :])  # use reference state from before

            # Calculate prediction from filter
            predicted_states[:, dut_index], predicted_state_covariances[:, dut_index] = _filter_predict(
                reference_state=reference_states[:, dut_index - 1, :],  # use reference state from before
                track_jacobian=Js[:, dut_index, :, :],  # transition matrix from dut_index - 1 -> dut_index
                local_scatter_gain_matrix=Gl_det,
                transition_covariance=transition_covariances[:, dut_index - 1],  # next plane in -z direction
                current_filtered_state=filtered_states[:, dut_index - 1],  # next plane in -z direction
                current_filtered_state_covariance=filtered_state_covariances[:, dut_index - 1])  # next plane in -z direction

            # check_covariance_matrix(predicted_state_covariances[:, dut_index])  # Sanity check for covariance matrix

        valid_hit_selection = ~np.isnan(observations[:, dut_index, 0])
        if dut_index in select_fit_duts:
            # DUT is a fit dut:
            # set filter to prediction where no hit is available,
            # otherwise calculate filtered state.
            kalman_gains[:, dut_index], filtered_states[:, dut_index], filtered_state_covariances[:, dut_index], chi2[valid_hit_selection, dut_index] = _filter_correct(
                reference_state=reference_state,  # use reference state from actual plane for filtering
                observation_matrix=observation_matrices[:, dut_index],
                observation_covariance=observation_covariances[:, dut_index],
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
            filtered_residuals = observations[valid_hit_selection, dut_index] - _vec_mul(observation_matrices[valid_hit_selection, dut_index], filtered_states[valid_hit_selection, dut_index] + reference_state[valid_hit_selection])
            # Note: need to add here covariance matrices, since in this case (filter equals to prediction) need to use the formula for predicted residual covariance
            filtered_residuals_covariance = observation_covariances[valid_hit_selection, dut_index] + _mat_mul(observation_matrices[valid_hit_selection, dut_index], _mat_mul(filtered_state_covariances[valid_hit_selection, dut_index], _mat_trans(observation_matrices[valid_hit_selection, dut_index])))
            check_covariance_matrix(filtered_residuals_covariance)  # Sanity check for covariance matrix
            chi2[valid_hit_selection, dut_index] = _vec_vec_mul(filtered_residuals, _vec_mul(_mat_inverse(filtered_residuals_covariance), filtered_residuals))
            chi2[~valid_hit_selection, dut_index] = np.nan  # No hit, thus no chi2

    # Final check for valid chi2
    if np.any(chi2[~np.isnan(chi2)] < 0.0):
        raise RuntimeError('Some chi-square values are negative (during filter step)!')

    return predicted_states, predicted_state_covariances, kalman_gains, filtered_states, filtered_state_covariances, chi2, Js


def _filter_b(dut_planes, reference_states, z_sorted_dut_indices, select_fit_duts, observations, observation_matrices, transition_covariances, observation_covariances, initial_state, initial_state_covariance):
    """Apply the Kalman Filter. First a prediction of the state is done, then a filtering is
    done which includes the observations.

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
    transition_covariances : [chunk_size, n_timesteps-1, n_dim_state,n_dim_state]  array-like
        covariance matrices of transition matrices.
    observation_covariances : [chunk_size, n_timesteps, n_dim_obs, n_dim_obs] array-like
        covariance matrices of observation matrices.
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
    """
    chunk_size, n_timesteps, n_dim_obs = observations.shape
    n_dim_state = initial_state_covariance.shape[2]

    predicted_states = np.zeros((chunk_size, n_timesteps, n_dim_state))
    predicted_state_covariances = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))
    kalman_gains = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_obs))
    filtered_states = np.zeros((chunk_size, n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))
    chi2 = np.full((chunk_size, n_timesteps), fill_value=np.nan)
    Js = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))

    for i, dut_index in enumerate(z_sorted_dut_indices[::-1]):
        # Get actual reference state
        reference_state = reference_states[:, dut_index, :]

        if i == 0:  # first DUT / last DUT
            predicted_states[:, dut_index] = initial_state
            predicted_state_covariances[:, dut_index] = initial_state_covariance
        else:
            dut = dut_planes[dut_index + 1]  # we use filter from plane before [dut_index - 1] and extrapolate/predict onto actual plane [dut_index - 1]
            next_dut = dut_planes[dut_index]  # we want to get prediction onto actual plane [dut_index]
            check_covariance_matrix(transition_covariances[:, dut_index + 1])  # Sanity check for covariance matrix

            # Local to global transformation
            rotation_matrix = geometry_utils.rotation_matrix(
                                alpha=dut.rotation_alpha,
                                beta=dut.rotation_beta,
                                gamma=dut.rotation_gamma)
            rotation_matrix_next = geometry_utils.rotation_matrix(
                                alpha=next_dut.rotation_alpha,
                                beta=next_dut.rotation_beta,
                                gamma=next_dut.rotation_gamma)

            dut_position = np.array([dut.translation_x, dut.translation_y, dut.translation_z])
            next_dut_position = np.array([next_dut.translation_x, next_dut.translation_y, next_dut.translation_z])

            # Transition matrix: 7: not defined/needed, 6: 7->6, 5: 6->5 (k: k + 1 --> k)
            Js[:, dut_index, :, :] = _calculate_track_jacobian_new(
                reference_state=reference_states[:, dut_index + 1, :],  # use reference state from before (backward)
                dut_position=np.tile(dut_position, reps=(reference_state.shape[0], 1)),
                next_dut_position=np.tile(next_dut_position, reps=(reference_state.shape[0], 1)),  # extrapolates to this position
                rotation_matrix=np.tile(rotation_matrix, reps=(reference_state.shape[0], 1, 1)),
                rotation_matrix_next=np.tile(rotation_matrix_next, reps=(reference_state.shape[0], 1, 1)))

            # TODO: what is this exactly (according to Wolin et al. paper)
            Gl_det = _calculate_scatter_gain_matrix(reference_state=reference_states[:, dut_index + 1, :])  # use reference state from before

            # Calculate prediction from filter
            predicted_states[:, dut_index], predicted_state_covariances[:, dut_index] = _filter_predict(
                reference_state=reference_states[:, dut_index + 1, :],  # use reference state from before
                track_jacobian=Js[:, dut_index, :, :],  # transition matrix from dut_index - 1 -> dut_index
                local_scatter_gain_matrix=Gl_det,
                transition_covariance=transition_covariances[:, dut_index + 1],  # next plane in +z direction
                current_filtered_state=filtered_states[:, dut_index + 1],  # next plane in +z direction
                current_filtered_state_covariance=filtered_state_covariances[:, dut_index + 1])  # next plane in -z direction

            # check_covariance_matrix(predicted_state_covariances[:, dut_index])  # Sanity check for covariance matrix

        valid_hit_selection = ~np.isnan(observations[:, dut_index, 0])
        if dut_index in select_fit_duts:
            # DUT is a fit dut:
            # set filter to prediction where no hit is available,
            # otherwise calculate filtered state.
            kalman_gains[:, dut_index], filtered_states[:, dut_index], filtered_state_covariances[:, dut_index], chi2[valid_hit_selection, dut_index] = _filter_correct(
                reference_state=reference_state,  # use reference state from actual plane for filtering
                observation_matrix=observation_matrices[:, dut_index],
                observation_covariance=observation_covariances[:, dut_index],
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
            filtered_residuals = observations[valid_hit_selection, dut_index] - _vec_mul(observation_matrices[valid_hit_selection, dut_index], filtered_states[valid_hit_selection, dut_index] + reference_state[valid_hit_selection])
            # Note: need to add here covariance matrices, since in this case (filter equals to prediction) need to use the formula for predicted residual covariance
            filtered_residuals_covariance = observation_covariances[valid_hit_selection, dut_index] + _mat_mul(observation_matrices[valid_hit_selection, dut_index], _mat_mul(filtered_state_covariances[valid_hit_selection, dut_index], _mat_trans(observation_matrices[valid_hit_selection, dut_index])))
            check_covariance_matrix(filtered_residuals_covariance)  # Sanity check for covariance matrix
            chi2[valid_hit_selection, dut_index] = _vec_vec_mul(filtered_residuals, _vec_mul(_mat_inverse(filtered_residuals_covariance), filtered_residuals))
            chi2[~valid_hit_selection, dut_index] = np.nan  # No hit, thus no chi2

    # Final check for valid chi2
    if np.any(chi2[~np.isnan(chi2)] < 0.0):
        raise RuntimeError('Some chi-square values are negative (during filter step)!')

    return predicted_states, predicted_state_covariances, kalman_gains, filtered_states, filtered_state_covariances, chi2, Js


@njit(cache=True)
def _smooth_update(observation, reference_state, observation_matrix, observation_covariance, transition_matrix, predicted_state_covariance, filtered_state, filtered_state_covariance, next_predicted_state, next_predicted_state_covariance, next_smoothed_state, next_smoothed_state_covariance, dut_used_in_fit):
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

    smoothed_state = filtered_state + _vec_mul(kalman_smoothing_gain, next_smoothed_state - next_predicted_state)

    smoothed_state_covariance = filtered_state_covariance + _mat_mul(kalman_smoothing_gain,
                                                                     _mat_mul((next_smoothed_state_covariance - next_predicted_state_covariance),
                                                                              _mat_trans(kalman_smoothing_gain)))

    # Calculate chi2 (only if observation available)
    valid_hit_selection = ~np.isnan(observation[:, 0])
    smoothed_residuals = observation[valid_hit_selection] - _vec_mul(observation_matrix[valid_hit_selection], smoothed_state[valid_hit_selection] + reference_states[valid_hit_selection])
    if dut_used_in_fit:
        smoothed_residuals_covariance = observation_covariance[valid_hit_selection] - _mat_mul(observation_matrix[valid_hit_selection], _mat_mul(smoothed_state_covariance[valid_hit_selection], _mat_trans(observation_matrix[valid_hit_selection])))
    else:
        residual_covariance_matrix = 2 * predicted_state_covariance - smoothed_state_covariance
        smoothed_residuals_covariance = observation_covariance[valid_hit_selection] + _mat_mul(observation_matrix[valid_hit_selection], _mat_mul(residual_covariance_matrix[valid_hit_selection], _mat_trans(observation_matrix[valid_hit_selection])))
    check_covariance_matrix(smoothed_residuals_covariance)  # Sanity check for covariance matrix
    chi2 = _vec_vec_mul(smoothed_residuals, _vec_mul(_mat_inverse(smoothed_residuals_covariance), smoothed_residuals))

    return smoothed_state, smoothed_state_covariance, kalman_smoothing_gain, chi2


def _smooth(dut_planes, reference_states, z_sorted_dut_indices, select_fit_duts, observations, observation_matrices, observation_covariances, filtered_states, filtered_state_covariances, predicted_states, predicted_state_covariances):
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
    kalman_smoothing_gains = np.zeros((chunk_size, n_timesteps, n_dim_state, n_dim_state))
    chi2 = np.zeros((chunk_size, n_timesteps))

    last_dut_index = z_sorted_dut_indices[-1]

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
            reference_state=reference_states[:, dut_index],
            observation_matrix=observation_matrices[:, dut_index],
            observation_covariance=observation_covariances[:, dut_index],
            predicted_state_covariance=predicted_state_covariances[:, dut_index],
            filtered_state=filtered_states[:, dut_index],
            filtered_state_covariance=filtered_state_covariances[:, dut_index],
            next_predicted_state=predicted_states[:, z_sorted_dut_indices[::-1][i]],  # next plane in +z direction
            next_predicted_state_covariance=predicted_state_covariances[:, z_sorted_dut_indices[::-1][i]],  # next plane +z direction
            next_smoothed_state=smoothed_states[:, z_sorted_dut_indices[::-1][i]],  # next plane +z direction
            next_smoothed_state_covariance=smoothed_state_covariances[:, z_sorted_dut_indices[::-1][i]],  # next plane +z direction
            dut_used_in_fit=True if dut_index in select_fit_duts else False)

        chi2[~valid_hit_selection, dut_index] = np.nan  # No hit, thus no chi2
        # check_covariance_matrix(smoothed_state_covariances[:, dut_index])  # Sanity check for covariance matrix

    # Final check for valid chi2
    if np.any(chi2[~np.isnan(chi2)] < 0.0):
        raise RuntimeError('Some chi-square values are negative (during smoothing step)!')

    return smoothed_states, smoothed_state_covariances, kalman_smoothing_gains, chi2


@njit(cache=True)
def _vec_vec_mul(X, Y):
    '''Helper function to multiply 3D vector with 3D vector. Multiplication is done on last two axes.
    '''
    result = np.zeros((X.shape[0]))
    for l in range(X.shape[0]):
        # iterate through rows of X
        for i in range(X.shape[1]):
            result[l] += X[l][i] * Y[l][i]
    return result


@njit(cache=True)
def _mat_mul(X, Y):
    '''Helper function to multiply two 3D matrices. Multiplication is done on last two axes.
    '''
    result = np.zeros((X.shape[0], X.shape[1], Y.shape[2]))
    if not X.shape[2] == Y.shape[1]:
        raise RuntimeError('Matrix muliplication failed due to incorrect shape!')
    for l in range(X.shape[0]):
        # iterate through rows of X
        for i in range(X.shape[1]):
            # iterate through columns of Y
            for j in range(Y.shape[2]):
                # iterate through rows of Y
                for k in range(Y.shape[1]):
                    result[l][i][j] += X[l][i][k] * Y[l][k][j]
    return result


@njit(cache=True)
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


@njit(cache=True)
def _mat_trans(X):
    '''Helper function to calculate transpose of 3D matrix. Transposition is done on last two axes.
    '''
    result = np.zeros((X.shape[0], X.shape[2], X.shape[1]))
    for l in range(X.shape[0]):
        for i in range(X.shape[2]):
            for j in range(X.shape[1]):
                result[l][i][j] = X[l][j][i]

    return result


@njit(cache=True)
def _mat_inverse(X, atol=1e-4, rtol=1e-6):
    '''Helper function to calculate inverese of 3D matrix. Inversion is done on last two axes.
    '''
    X = np.ascontiguousarray(X)  # make array contiguous (avoid NumbaPerformance warning)
    inv = np.zeros((X.shape), dtype=np.float64)
    for i in range(X.shape[0]):
        inv[i] = linalg.inv(X[i])
        # Check if inverse was succesfull
        X_c = np.dot(X[i], np.dot(inv[i], X[i]))
        inv_c = np.dot(inv[i], np.dot(X[i], inv[i]))
        tol_X = atol + rtol * np.absolute(X_c)
        tol_inv = atol + rtol * np.absolute(inv_c)
        if np.any(np.absolute(X[i] - X_c) > tol_X) or np.any(np.absolute(inv[i] - inv_c) > tol_inv):
            # print(X[i], i)
            # raise RuntimeError('Matrix inversion failed!')
            print('Matrix inversion failed!')
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


@njit(cache=True)
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


def _calculate_scatter_gain_matrix(reference_state):
    """ Reference: Wolin and Ho (NIM A329 (1993) 493-500)
    """
    p3 = reference_state[:, 2]
    p4 = reference_state[:, 3]

    n_trk = np.zeros(shape=(reference_state.shape[0], 3), dtype=np.float64)
    n_trk[:, 0] = p3
    n_trk[:, 1] = p4
    n_trk[:, 2] = 1.0
    n_trk_mag = np.sqrt(_vec_vec_mul(n_trk, n_trk))
    n_trk[:, 0] /= n_trk_mag  # normalize to 1
    n_trk[:, 1] /= n_trk_mag  # normalize to 1
    n_trk[:, 2] /= n_trk_mag  # normalize to 1

    u_hat = np.zeros(shape=(reference_state.shape[0], 3), dtype=np.float64)
    u_hat[:, 0] = 1.0
    v_trk = np.cross(n_trk, u_hat)
    v_trk_mag = np.sqrt(_vec_vec_mul(v_trk, v_trk))
    v_trk[:, 0] /= v_trk_mag  # normalize to 1
    v_trk[:, 1] /= v_trk_mag  # normalize to 1
    v_trk[:, 2] /= v_trk_mag  # normalize to 1

    u_trk = np.cross(v_trk, n_trk)
    u_trk_mag = np.sqrt(_vec_vec_mul(u_trk, u_trk))
    u_trk[:, 0] /= u_trk_mag  # normalize to 1
    u_trk[:, 1] /= u_trk_mag  # normalize to 1
    u_trk[:, 2] /= u_trk_mag  # normalize to 1

    # Direction cosines
    a1 = u_trk[:, 0]
    a2 = v_trk[:, 0]
    a3 = n_trk[:, 0]
    b1 = u_trk[:, 1]
    b2 = v_trk[:, 1]
    b3 = n_trk[:, 1]
    g1 = u_trk[:, 2]
    g2 = v_trk[:, 2]
    g3 = n_trk[:, 2]

    # Scatter Gain Matrix
    G = np.zeros(shape=(reference_state.shape[0], 4, 2), dtype=np.float64)
    # dU'/dtheta1, 0, 0
    G[:, 2, 0] = (a1 * g3 - a3 * g1) / (g3 * g3)  # Eq (10)
    # dU'/dtheta2, 0, 1
    G[:, 2, 1] = (a2 * g3 - a3 * g2) / (g3 * g3)  # Eq (11)
    # dV'/dtheta1, 1, 0
    G[:, 3, 0] = (b1 * g3 - b3 * g1) / (g3 * g3)  # Eq (12)
    # dV'/dtheta2, 1, 1
    G[:, 3, 1] = (b2 * g3 - b3 * g2) / (g3 * g3)  # Eq (13)
    # Scattering angles affect the track
    # do not affect impact point
    # dU/dtheta1, 2, 0
    G[:, 0, 0] = 0.0
    # dU/dtheta2, 2, 1
    G[:, 0, 1] = 0.0
    # dV/dtheta1, 3, 0
    G[:, 1, 0] = 0.0
    # dV/dtheta2, 3, 1
    G[:, 1, 1] = 0.0

    return G


def _calculate_track_jacobian_new(reference_state, dut_position, next_dut_position, rotation_matrix, rotation_matrix_next):
    ''' Reference: V. Karimaki "Straight Line Fit for Pixel and Strip Detectors with Arbitrary Plane Orientations", CMS Note. (http://cds.cern.ch/record/687146/files/note99_041.pdf)
        Calculates change of local coordinates (u, v, u', w') from one dut to next dut (wrt. to reference state). Thus, this gives the
        transition of local coordinates from one dut to next dut (i.e. transition matrix for local track state).
        Assumes that rotation is given from local into global coordinates.
    '''

    # Coordinate transformation into global system of next dut
    Rot = _mat_mul(rotation_matrix_next, _mat_trans(rotation_matrix))
    Origin = _vec_mul(rotation_matrix, next_dut_position - dut_position)

    x_point = np.zeros(shape=(reference_state.shape[0], 3), dtype=np.float64)
    x_point[:, 0] = reference_state[:, 0]
    x_point[:, 1] = reference_state[:, 1]
    x_point[:, 2] = 0.0
    direc = np.zeros(shape=(reference_state.shape[0], 3), dtype=np.float64)
    direc[:, 0] = reference_state[:, 2]
    direc[:, 1] = reference_state[:, 3]
    direc[:, 2] = 1.0

    u = np.zeros(shape=(reference_state.shape[0], 3), dtype=np.float64)
    u[:, 0] = 1.0
    u[:, 1] = 0.0
    u[:, 2] = 0.0

    v = np.zeros(shape=(reference_state.shape[0], 3), dtype=np.float64)
    v[:, 0] = 0.0
    v[:, 1] = 1.0
    v[:, 2] = 0.0

    w = np.zeros(shape=(reference_state.shape[0], 3), dtype=np.float64)
    w[:, 0] = 0.0
    w[:, 1] = 0.0
    w[:, 2] = 1.0

    bru = _vec_vec_mul(direc, _vec_mul(Rot, u))
    brv = _vec_vec_mul(direc, _vec_mul(Rot, v))
    brw = _vec_vec_mul(direc, _vec_mul(Rot, w))

    t = _vec_vec_mul(Origin - x_point, _vec_mul(Rot, w)) / brw

    Up = bru / brw
    Vp = brv / brw

    J = np.zeros(shape=(reference_state.shape[0], 4, 4), dtype=np.float64)

    # dU'/du'
    J[:, 2, 2] = (Rot[:, 0, 0] * brw - bru * Rot[:, 0, 2]) / (brw * brw)
    # dU'/dv'
    J[:, 2, 3] = (Rot[:, 1, 0] * brw - bru * Rot[:, 1, 2]) / (brw * brw)
    # dU'/du
    J[:, 2, 0] = 0.0
    # dU'/dv
    J[:, 2, 1] = 0.0

    # dV'/du'
    J[:, 3, 2] = (Rot[:, 0, 1] * brw - brv * Rot[:, 0, 2]) / (brw * brw)
    # dV'/dv'
    J[:, 3, 3] = (Rot[:, 1, 1] * brw - brv * Rot[:, 1, 2]) / (brw * brw)
    # dV'/du
    J[:, 3, 0] = 0.0
    # dV'/dV
    J[:, 3, 1] = 0.0

    # dU/du
    J[:, 0, 0] = Rot[:, 0, 0] - Rot[:, 0, 2] * Up  # Eq (15)
    # dU/dv
    J[:, 0, 1] = Rot[:, 1, 0] - Rot[:, 1, 2] * Up  # Eq (16)
    # dU/du'
    J[:, 0, 2] = t * J[:, 0, 0]  # Eq (17)
    # dU/dv'
    J[:, 0, 3] = t * J[:, 0, 1]  # Eq (18)

    # dV/du
    J[:, 1, 0] = Rot[:, 0, 1] - Rot[:, 0, 2] * Vp  # Eq (15)
    # dV/dv
    J[:, 1, 1] = Rot[:, 1, 1] - Rot[:, 1, 2] * Vp  # Eq (16)
    # dV/du'
    J[:, 1, 2] = t * J[:, 1, 0]  # Eq (17)
    # dV/dv'
    J[:, 1, 3] = t * J[:, 1, 1]  # Eq (18)

    return J


class KalmanFilter(object):
    def smooth(self, dut_planes, reference_states, z_sorted_dut_indices, momentum, beta, observations, select_fit_duts,
               transition_offsets, transition_covariances, observation_matrices, observation_offsets, observation_covariances,
               initial_state, initial_state_covariance):
        """Apply the Kalman Smoother to the observations. In the first step a filtering is done,
        afterwards a smoothing is done. Calculation is done on whole track chunk with size chunk_size.

        Parameters
        ----------
        dut_planes : list
            List of DUT parameters (material_budget, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma).
        z_sorted_dut_indices : list
            List of DUT indices in the order reflecting their z position.
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

        n_duts = len(dut_planes)
        chunk_size = observations.shape[0]
        n_dim_state = transition_offsets.shape[2]

        # Run backward Kalman Filter in order to update the reference state trajectory with the predicted state trajectory of the filter result.
        # TODO: why not in forward direction?
        predicted_states_b, predicted_state_covariances_b, _, filtered_states_b, filtered_state_covariances_b, chi2s_filter_b, Js_b = _filter_b(
            dut_planes=dut_planes,
            reference_states=reference_states,
            z_sorted_dut_indices=z_sorted_dut_indices,
            select_fit_duts=select_fit_duts,
            observations=observations,
            observation_matrices=observation_matrices,
            transition_covariances=transition_covariances,
            observation_covariances=observation_covariances,
            initial_state=initial_state,
            initial_state_covariance=initial_state_covariance)

        # Update reference state trajectory
        reference_states += filtered_states_b

        # Need to re-calculate transition covariances since reference state trajectory was updated.
        for i, dut in enumerate(dut_planes):
            # Reference: Wolin paper
            L_ref = np.sqrt(1.0 + reference_states[:, i, 2]**2 + reference_states[:, i, 3]**2)  # Path length of reference track
            material_budget_ref = dut.material_budget * L_ref  # Material budget of reference track
            # Variance of projected multiple scattering angle
            theta = (13.6 / momentum / beta) * np.sqrt(material_budget_ref) * (1. + 0.038 * np.log(material_budget_ref))
            transition_covariances[:, i, 0, 0] = np.square(theta)  # projection on one axis of a plane that is perpendicular to the particle direction before scattering
            transition_covariances[:, i, 1, 1] = np.square(theta)

        for iter in range(1):
            # Run forward Kalman Filter
            predicted_states_f, predicted_state_covariances_f, _, filtered_states_f, filtered_state_covariances_f, chi2s_filter_f, Js_f = _filter_f(
                dut_planes=dut_planes,
                reference_states=reference_states,
                z_sorted_dut_indices=z_sorted_dut_indices,
                select_fit_duts=select_fit_duts,
                observations=observations,
                observation_matrices=observation_matrices,
                transition_covariances=transition_covariances,
                observation_covariances=observation_covariances,
                initial_state=initial_state,
                initial_state_covariance=initial_state_covariance)

            # Run backward Kalman Filter
            predicted_states_b, predicted_state_covariances_b, _, filtered_states_b, filtered_state_covariances_b, chi2s_filter_b, Js_b = _filter_b(
                dut_planes=dut_planes,
                reference_states=reference_states,
                z_sorted_dut_indices=z_sorted_dut_indices,
                select_fit_duts=select_fit_duts,
                observations=observations,
                observation_matrices=observation_matrices,
                transition_covariances=transition_covariances,
                observation_covariances=observation_covariances,
                initial_state=initial_state,
                initial_state_covariance=initial_state_covariance)

            smoothed_states = np.zeros_like(predicted_states_f)
            smoothed_state_covariances = np.zeros_like(filtered_state_covariances_f)
            smoothed_chi2 = np.full((chunk_size, n_duts), fill_value=np.nan)

            # Smoothing: Weigthed mean of forward and backward  Kalman Filter
            for i, dut_index in enumerate(z_sorted_dut_indices):
                if i == 0:  # First plane: Use backward filter result
                    smoothed_states[:, dut_index] = filtered_states_b[:, dut_index]
                    smoothed_state_covariances[:, dut_index] = filtered_state_covariances_b[:, dut_index]
                elif i == len(z_sorted_dut_indices) - 1:  # Last plane: Use forward filter result
                    smoothed_states[:, dut_index] = filtered_states_f[:, dut_index]
                    smoothed_state_covariances[:, dut_index] = filtered_state_covariances_f[:, dut_index]
                else:  # Calculate weigted mean of backward and forward filter result
                    xb = filtered_states_b[:, dut_index]
                    Cb = filtered_state_covariances_b[:, dut_index]
                    xf = filtered_states_f[:, dut_index]
                    Cf = filtered_state_covariances_f[:, dut_index]
                    Cf_inv = _mat_inverse(Cf)
                    Cb_inv = _mat_inverse(Cb)
                    smoothed_state_covariances[:, dut_index] = _mat_inverse(Cf_inv + Cb_inv)
                    smoothed_states[:, dut_index] = _vec_mul(smoothed_state_covariances[:, dut_index], (_vec_mul(Cf_inv, xf) + _vec_mul(Cb_inv, xb)))

                valid_hit_selection = ~np.isnan(observations[:, dut_index, 0])
                if dut_index in select_fit_duts:
                    # Calculate chi2 using smoothed result  (only if observation available)
                    smoothed_residuals = observations[valid_hit_selection, dut_index] - _vec_mul(observation_matrices[valid_hit_selection, dut_index], smoothed_states[valid_hit_selection, dut_index] + reference_states[valid_hit_selection, dut_index])
                    smoothed_residuals_covariance = observation_covariances[valid_hit_selection, dut_index] - _mat_mul(observation_matrices[valid_hit_selection, dut_index], _mat_mul(smoothed_state_covariances[valid_hit_selection, dut_index], _mat_trans(observation_matrices[valid_hit_selection, dut_index])))
                    check_covariance_matrix(smoothed_residuals_covariance)  # Sanity check for covariance matrix
                    smoothed_chi2[valid_hit_selection, dut_index] = _vec_vec_mul(smoothed_residuals, _vec_mul(_mat_inverse(smoothed_residuals_covariance), smoothed_residuals))
                    smoothed_chi2[~valid_hit_selection, dut_index] = np.nan  # No hit, thus no chi2
                else:
                    # Note: need to add here covariance matrices, since in this case (filter equals to prediction) need to use the formula for predicted residual covariance
                    smoothed_residuals = observations[valid_hit_selection, dut_index] - _vec_mul(observation_matrices[valid_hit_selection, dut_index], smoothed_states[valid_hit_selection, dut_index] + reference_states[valid_hit_selection, dut_index])
                    smoothed_residuals_covariance = observation_covariances[valid_hit_selection, dut_index] + _mat_mul(observation_matrices[valid_hit_selection, dut_index], _mat_mul(smoothed_state_covariances[valid_hit_selection, dut_index], _mat_trans(observation_matrices[valid_hit_selection, dut_index])))
                    check_covariance_matrix(smoothed_residuals_covariance)  # Sanity check for covariance matrix
                    smoothed_chi2[valid_hit_selection, dut_index] = _vec_vec_mul(smoothed_residuals, _vec_mul(_mat_inverse(smoothed_residuals_covariance), smoothed_residuals))
                    smoothed_chi2[~valid_hit_selection, dut_index] = np.nan  # No hit, thus no chi2

                # Linearization around reference state vector is used. Thus, add the reference state to the final result.
                smoothed_states[:, dut_index] += reference_states[:, dut_index]
                filtered_states_f[:, dut_index] += reference_states[:, dut_index]
                filtered_states_b[:, dut_index] += reference_states[:, dut_index]
                predicted_states_f[:, dut_index] += reference_states[:, dut_index]
                predicted_states_b[:, dut_index] += reference_states[:, dut_index]

                # In case more than one iteration is used: Update the reference state trajectory using the smoother result.
                reference_states += smoothed_states

        return smoothed_states, smoothed_state_covariances, smoothed_chi2, filtered_states_f, filtered_states_b, predicted_states_f, predicted_states_b
