''' Track finding and fitting functions are listed here.'''
from __future__ import division

import logging
import os.path
from multiprocessing import Pool, cpu_count
from math import sqrt
from collections import Iterable

import progressbar
import tables as tb
import numpy as np
from numba import njit

from testbeam_analysis.telescope.telescope import Telescope
from testbeam_analysis.tools import plot_utils
# from testbeam_analysis.tools import smc
from testbeam_analysis.tools import analysis_utils
from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.tools import kalman


def find_tracks(telescope_configuration, input_merged_file, output_track_candidates_file=None, max_events=None, align_to_beam=True, chunk_size=1000000):
    '''Sorting DUT hits and tries to find hits in subsequent DUTs matching the hits in the first DUT.
    The output is the track candidates array which has the hits in a different order compared to the tracklets array (merged array).

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_merged_file : string
        Filename of the input merged cluster file containing the hit information from all DUTs.
    output_track_candidates_file : string
        Filename of the output track candidates file.
    max_events : uint
        Maximum number of randomly chosen events. If None, all events are taken.
    align_to_beam : bool
        If True, the telescope alignment is used to align the DUTs so that the beam axis is parallel to the z axis.
        This improves the performance of track finding algorithm.
        If False, the beam axis is not corrected and large track angles and high track densities
        have an impact on the performance of the track finding algorithm.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_track_candidates_file : string
        Filename of the output track candidates file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Finding tracks in %d DUTs ===' % n_duts)

    if output_track_candidates_file is None:
        output_track_candidates_file = os.path.join(os.path.dirname(input_merged_file), 'Track_Candidates.h5')

# TODO: implement max_events into SMC
#     def work(tracklets_data_chunk):
#         ''' Track finding per cpu core '''
#         # Prepare hit data for track finding, create temporary arrays for x, y, z position and charge data
#         # This is needed to call a numba jitted function, since the number of DUTs is not fixed and thus the data format
#         x_local = np.copy(tracklets_data_chunk['x_dut_0'])
#         y_local = np.copy(tracklets_data_chunk['y_dut_0'])
#         z_local = np.copy(tracklets_data_chunk['z_dut_0'])
#         x_err_local = np.copy(tracklets_data_chunk['x_err_dut_0'])
#         y_err_local = np.copy(tracklets_data_chunk['y_err_dut_0'])
#         z_err_local = np.copy(tracklets_data_chunk['z_err_dut_0'])
#         geometry_utils.apply_alignment_to_hits(hits=tracklets_data_chunk, dut_index=0, use_prealignment=use_prealignment, alignment=alignment, beam_alignment=beam_alignment, inverse=False, no_z=False)
#         x = tracklets_data_chunk['x_dut_0']
#         y = tracklets_data_chunk['y_dut_0']
#         z = tracklets_data_chunk['z_dut_0']
#         x_err = tracklets_data_chunk['x_err_dut_0']
#         y_err = tracklets_data_chunk['y_err_dut_0']
#         z_err = tracklets_data_chunk['z_err_dut_0']
#         charge = tracklets_data_chunk['charge_dut_0']
#         n_hits = tracklets_data_chunk['n_hits_dut_0']
#         cluster_shape = tracklets_data_chunk['cluster_shape_dut_0']
#         n_cluster = tracklets_data_chunk['n_cluster_dut_0']
#         for dut_index in range(1, n_duts):
#             x_local = np.column_stack((x_local, tracklets_data_chunk['x_dut_%d' % (dut_index)]))
#             y_local = np.column_stack((y_local, tracklets_data_chunk['y_dut_%d' % (dut_index)]))
#             z_local = np.column_stack((z_local, tracklets_data_chunk['z_dut_%d' % (dut_index)]))
#             x_err_local = np.column_stack((x_err_local, tracklets_data_chunk['x_err_dut_%d' % (dut_index)]))
#             y_err_local = np.column_stack((y_err_local, tracklets_data_chunk['y_err_dut_%d' % (dut_index)]))
#             z_err_local = np.column_stack((z_err_local, tracklets_data_chunk['z_err_dut_%d' % (dut_index)]))
#             geometry_utils.apply_alignment_to_hits(hits=tracklets_data_chunk, dut_index=dut_index, use_prealignment=use_prealignment, alignment=alignment, beam_alignment=beam_alignment, inverse=False, no_z=False)
#             x = np.column_stack((x, tracklets_data_chunk['x_dut_%d' % (dut_index)]))
#             y = np.column_stack((y, tracklets_data_chunk['y_dut_%d' % (dut_index)]))
#             z = np.column_stack((z, tracklets_data_chunk['z_dut_%d' % (dut_index)]))
#             x_err = np.column_stack((x_err, tracklets_data_chunk['x_err_dut_%d' % (dut_index)]))
#             y_err = np.column_stack((y_err, tracklets_data_chunk['y_err_dut_%d' % (dut_index)]))
#             z_err = np.column_stack((z_err, tracklets_data_chunk['z_err_dut_%d' % (dut_index)]))
#             charge = np.column_stack((charge, tracklets_data_chunk['charge_dut_%d' % (dut_index)]))
#             n_hits = np.column_stack((n_hits, tracklets_data_chunk['n_hits_dut_%d' % (dut_index)]))
#             cluster_shape = np.column_stack((cluster_shape, tracklets_data_chunk['cluster_shape_dut_%d' % (dut_index)]))
#             n_cluster = np.column_stack((n_cluster, tracklets_data_chunk['n_cluster_dut_%d' % (dut_index)]))

#         event_number = tracklets_data_chunk['event_number']
#         hit_flag = np.zeros_like(tracklets_data_chunk['hit_flag'])
#         quality_flag = np.zeros_like(tracklets_data_chunk['quality_flag'])
#         n_tracks = tracklets_data_chunk['n_tracks']

#         # Perform the track finding with jitted loop
#         _find_tracks_loop(event_number=event_number,
#                           x_local=x_local,
#                           y_local=y_local,
#                           z_local=z_local,
#                           x_err_local=x_err_local,
#                           y_err_local=y_err_local,
#                           z_err_local=z_err_local,
#                           x=x,
#                           y=y,
#                           z=z,
#                           x_err=x_err,
#                           y_err=y_err,
#                           z_err=z_err,
#                           charge=charge,
#                           n_hits=n_hits,
#                           cluster_shape=cluster_shape,
#                           n_cluster=n_cluster,
#                           hit_flag=hit_flag,
#                           n_tracks=n_tracks)

# # TODO: also use local coordinates in find_tracks_loop to avoid transformation to local coordinate system
# #         for dut_index in range(0, n_duts):
# #             geometry_utils.apply_alignment_to_hits(hits=combined, dut_index=dut_index, use_prealignment=use_prealignment, alignment=alignment, inverse=True, no_z=False)

#         # Merge result data from arrays into one recarray
#         combined = np.column_stack((event_number, x_local, y_local, z_local, charge, n_hits, cluster_shape, n_cluster, hit_flag, quality_flag, n_tracks, x_err_local, y_err_local, z_err_local))
#         return np.core.records.fromarrays(combined.transpose(), dtype=tracklets_data_chunk.dtype)

#     smc.SMC(input_filename=input_tracklets_file,
#             output_filename=output_track_candidates_file,
#             func=work,
#             node_desc={'name':'TrackCandidates',
#                         'title':'Track candidates'},
#             # Apply track finding on tracklets or track candidates
#             table=['Tracklets', 'TrackCandidates'],
#             align_at='event_number',
#             chunk_size=chunk_size)

    # calculating DUT indices list with z-order
    intersections_z_axis = []
    for dut in telescope:
        intersections_z_axis.append(geometry_utils.get_line_intersections_with_dut(
            line_origins=np.array([[0.0, 0.0, 0.0]]),
            line_directions=np.array([[0.0, 0.0, 1.0]]),
            translation_x=dut.translation_x,
            translation_y=dut.translation_y,
            translation_z=dut.translation_z,
            rotation_alpha=dut.rotation_alpha,
            rotation_beta=dut.rotation_beta,
            rotation_gamma=dut.rotation_gamma)[0][2])
    z_sorted_dut_indices = np.argsort(intersections_z_axis)

    with tb.open_file(input_merged_file, mode='r') as in_file_h5:
        tracklets_node = in_file_h5.root.MergedClusters

        with tb.open_file(output_track_candidates_file, mode='w') as out_file_h5:
            track_candidates = out_file_h5.create_table(
                where=out_file_h5.root, name='TrackCandidates',
                description=tracklets_node.dtype,
                title='Track candidates',
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))

            total_n_tracks = tracklets_node.shape[0]
            total_n_tracks_stored = 0
            total_n_events_stored = 0

            widgets = ['', progressbar.Percentage(), ' ',
                       progressbar.Bar(marker='*', left='|', right='|'),
                       ' ', progressbar.AdaptiveETA()]
            progress_bar = progressbar.ProgressBar(widgets=widgets,
                                                   maxval=total_n_tracks,
                                                   term_width=80)
            progress_bar.start()

            for tracklets_data_chunk, index_chunk in analysis_utils.data_aligned_at_events(tracklets_node, chunk_size=chunk_size):
                n_tracks_chunk = tracklets_data_chunk.shape[0]

                unique_events = np.unique(tracklets_data_chunk["event_number"])
                n_events_chunk = unique_events.shape[0]

                if max_events:
                    if total_n_tracks == index_chunk:  # last chunk, adding all remaining events
                        select_n_events = max_events - total_n_events_stored
                    elif total_n_events_stored == 0:  # first chunk
                        select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks)))
                    else:
                        # calculate correction of number of selected events
                        correction = (total_n_tracks - index_chunk) / total_n_tracks * 1 / (((total_n_tracks - last_index_chunk) / total_n_tracks) / ((max_events - total_n_events_stored_last) / max_events)) \
                            + (index_chunk) / total_n_tracks * 1 / (((last_index_chunk) / total_n_tracks) / ((total_n_events_stored_last) / max_events))
                        select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks) * correction))
                    # do not store more events than in current chunk
                    select_n_events = min(n_events_chunk, select_n_events)
                    # do not store more events than given by max_events
                    select_n_events = min(select_n_events, max_events - total_n_events_stored)
                    np.random.seed(seed=0)
                    selected_events = np.random.choice(unique_events, size=select_n_events, replace=False)
                    store_n_events = selected_events.shape[0]
                    total_n_events_stored += store_n_events
                    selected_tracks = np.in1d(tracklets_data_chunk["event_number"], selected_events)
                    store_n_tracks = np.count_nonzero(selected_tracks)
                    total_n_tracks_stored += store_n_tracks
                    tracklets_data_chunk = tracklets_data_chunk[selected_tracks]
                indices = np.column_stack([np.arange(tracklets_data_chunk.shape[0], dtype=np.int64) for _ in range(n_duts)])
                event_numbers = tracklets_data_chunk['event_number']
                x_global = []
                y_global = []
                z_global = []
                for dut_index in range(n_duts):
                    x_global_dut, y_global_dut, z_global_dut = telescope[dut_index].local_to_global_position(
                        x=tracklets_data_chunk['x_dut_%d' % dut_index],
                        y=tracklets_data_chunk['y_dut_%d' % dut_index],
                        z=tracklets_data_chunk['z_dut_%d' % dut_index])
                    if align_to_beam:
                        x_global_dut, y_global_dut, z_global_dut = telescope[dut_index].local_to_global_position(
                            x=x_global_dut,
                            y=y_global_dut,
                            z=z_global_dut,
                            translation_x=telescope.translation_x,
                            translation_y=telescope.translation_y,
                            translation_z=telescope.translation_z,
                            rotation_alpha=telescope.rotation_alpha,
                            rotation_beta=telescope.rotation_beta,
                            rotation_gamma=telescope.rotation_gamma)
                    x_global.append(x_global_dut)
                    y_global.append(y_global_dut)
                    z_global.append(z_global_dut)
                x_global = np.column_stack(x_global)
                y_global = np.column_stack(y_global)
                z_global = np.column_stack(z_global)
                hit_flag = np.zeros_like(tracklets_data_chunk['hit_flag'])
                # Perform the track finding with jitted loop
                _find_tracks_loop(
                    event_numbers=event_numbers,
                    indices=indices,
                    z_sorted_dut_indices=z_sorted_dut_indices,
                    x=x_global,
                    y=y_global,
                    z=z_global)
                # copy the columns to the result array
                for dut_index in range(n_duts):
                    for column_name in tracklets_data_chunk.dtype.names:
                        if 'dut_%d' % dut_index in column_name:
                            tracklets_data_chunk[column_name] = tracklets_data_chunk[column_name][indices[:, dut_index]]
                # calculate new hit flags
                for dut_index in range(n_duts):
                    hit_flag += np.isfinite(tracklets_data_chunk['x_dut_%d' % dut_index]).astype(hit_flag.dtype) << dut_index
                tracklets_data_chunk['hit_flag'] = hit_flag
                # append data to table
                track_candidates.append(tracklets_data_chunk)
                track_candidates.flush()
                total_n_events_stored_last = total_n_events_stored
                total_n_tracks_last = total_n_tracks
                last_index_chunk = index_chunk
                progress_bar.update(index_chunk)

            progress_bar.finish()

    return output_track_candidates_file


@njit
def _find_tracks_loop(event_numbers, indices, z_sorted_dut_indices, x, y, z):
    ''' This function provides an algorithm to generates the track candidates from the tracklets array.
    Each hit is put to the best fitting track. Tracks are assumed to have
    no big angle, otherwise this approach does not work.
    '''
    actual_event_number = event_numbers[0]

    # Numba uses c scopes, thus define all used variables here
    track_index = 0
    actual_event_start_index = 0
    # for track_index, curr_event_number in enumerate(event_numbers):  # Loop over all possible tracks
    while track_index < event_numbers.shape[0]:
        curr_event_number = event_numbers[track_index]
        # Set variables for new event
        if curr_event_number != actual_event_number:  # Detect new event
            actual_event_number = curr_event_number
            actual_event_start_index = track_index  # Smallest index of current event

        reference_hit_set = False  # The first real hit (column, row != nan) is the reference hit of the actual track
        for dut_index in z_sorted_dut_indices:  # loop over all DUTs in the actual track
            if not reference_hit_set and not np.isnan(x[track_index][dut_index]):  # Search for first DUT that registered a hit
                # actual_reference_x, actual_reference_y = x[track_index][dut_index], y[track_index][dut_index]
                reference_hit_set = True
            elif reference_hit_set:  # First hit found, now find best (closest) DUT hit
                _find_tracks(
                    event_numbers=event_numbers,
                    indices=indices,
                    z_sorted_dut_indices=z_sorted_dut_indices,
                    event_start_index=actual_event_start_index,
                    index=track_index,
                    dut_index=dut_index,
                    x=x,
                    y=y,
                    z=z)
        # goto next possible track
        track_index += 1


@njit
def _find_tracks(event_numbers, indices, z_sorted_dut_indices, event_start_index, index, dut_index, x, y, z):
    # The hit distance of the actual assigned hit; -1 means not assigned
    reference_dut_index = _get_first_dut_index(
        x=x,
        index=index,
        z_sorted_dut_indices=z_sorted_dut_indices)
    actual_reference_x, actual_reference_y = x[index][reference_dut_index], y[index][reference_dut_index]
    best_index = index
    if np.isnan(x[index][dut_index]):
        best_hit_distance = -1  # Value for no hit
    else:
        # Calculate the hit distance of the actual assigned DUT hit towards the actual reference hit
        best_hit_distance = sqrt((x[index][dut_index] - actual_reference_x)**2 + (y[index][dut_index] - actual_reference_y)**2)
    # The shortest hit distance to the actual hit; -1 means not assigned
    # for hit_index in range(actual_event_start_index, event_numbers.shape[0]):  # Loop over all not sorted hits of actual DUT
    hit_index = event_start_index
    while hit_index < event_numbers.shape[0]:
        if event_numbers[hit_index] != event_numbers[event_start_index]:  # Abort condition
            break
        if index == hit_index:  # Check if hit swapping is needed
            hit_index += 1
            continue
        actual_hit_x, actual_hit_y = x[hit_index][dut_index], y[hit_index][dut_index]
        if np.isnan(actual_hit_x):  # x = nan is no hit
            hit_index += 1
            continue
        # Calculate the hit distance of the actual DUT hit towards the actual reference hit
        actual_x_distance, actual_y_distance = actual_hit_x - actual_reference_x, actual_hit_y - actual_reference_y
        actual_hit_distance = sqrt(actual_x_distance**2 + actual_y_distance**2)
        if best_hit_distance >= 0 and best_hit_distance < actual_hit_distance:  # Check if actual assigned hit is better
            hit_index += 1
            continue
        # TODO: do not take all hits, check for valid hits (i.e., inside scatter cone)
        # Get reference DUT index of other track
        first_dut_hit_index = _get_first_dut_index(
            x=x,
            index=hit_index,
            z_sorted_dut_indices=z_sorted_dut_indices)
        reference_x_other, reference_y_other = x[hit_index][first_dut_hit_index], y[hit_index][first_dut_hit_index]
        # Calculate hit distance to reference hit of other track
        x_distance_other, y_distance_other = actual_hit_x - reference_x_other, actual_hit_y - reference_y_other
        hit_distance_other = sqrt(x_distance_other**2 + y_distance_other**2)
        if actual_hit_distance > hit_distance_other and first_dut_hit_index != dut_index:  # Only take hit if it fits better to actual track; otherwise leave it with other track
            hit_index += 1
            continue
        # setting best hit
        best_index = hit_index
        best_hit_distance = actual_hit_distance
        hit_index += 1
    # swapping hits
    tmp_x, tmp_y, tmp_z = x[index][dut_index], y[index][dut_index], z[index][dut_index]
    tmp_index = indices[index][dut_index]

    x[index][dut_index], y[index][dut_index], z[index][dut_index] = x[best_index][dut_index], y[best_index][dut_index], z[best_index][dut_index]
    indices[index][dut_index] = indices[best_index][dut_index]

    x[best_index][dut_index], y[best_index][dut_index], z[best_index][dut_index] = tmp_x, tmp_y, tmp_z
    indices[best_index][dut_index] = tmp_index
    # recursively call _find_tracks in case of swapping
    # hits with other finished tracks
    first_dut_hit_index = _get_first_dut_index(
        x=x,
        index=best_index,
        z_sorted_dut_indices=z_sorted_dut_indices)
    if index > best_index and first_dut_hit_index != dut_index:
        _find_tracks(
            event_numbers=event_numbers,
            indices=indices,
            z_sorted_dut_indices=z_sorted_dut_indices,
            event_start_index=event_start_index,
            index=best_index,
            dut_index=dut_index,
            x=x,
            y=y,
            z=z)


@njit
def _get_first_dut_index(x, index, z_sorted_dut_indices):
    ''' Returns the first DUT that has a hit for the track at index '''
    for dut_index in z_sorted_dut_indices:  # Loop over duts, to get first DUT hit of track
        if not np.isnan(x[index][dut_index]):
            return dut_index
    return -1


def fit_tracks(telescope_configuration, input_track_candidates_file, output_tracks_file=None, max_events=None, select_duts=None, select_hit_duts=None, select_fit_duts=None, exclude_dut_hit=True, method='fit', beam_energy=None, particle_mass=None, scattering_planes=None, quality_distances=(250.0, 250.0), reject_quality_distances=(500.0, 500.0), use_limits=True, keep_data=False, full_track_info=False, plot=True, chunk_size=1000000):
    '''Calculate tracks and set tracks quality flag for selected DUTs.
    Two methods are available to generate tracks: a linear fit (method="fit") and a Kalman Filter (method="kalman").

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_track_candidates_file : string
        Filename of the input track candidate file.
    output_tracks_file : string
        Filename of the output tracks file.
    max_events : uint
        Maximum number of randomly chosen events. If None, all events are taken.
    select_duts : list
        Specify the fit DUTs for which tracks will be fitted and a track array will be generated.
        If None, for all DUTs are selected.
    select_hit_duts : list or list of lists
        Specifying DUTs that are required to have a hit. Tracks are not fitted if one of the DUTs has no hit.
        The actual fit DUT is excluded.
        If None, require all DUTs to have a hit.
    select_fit_duts : list or list of lists
        Specifying DUTs that are used for the fit.
        If None, the DUTs are taken from select_hit_duts.
        The select_fit_duts is a subset of select_hit_duts.
    exclude_dut_hit : bool or list
        Decide whether or not to use hits in the actual fit DUT for track fitting (for unconstrained residuals).
        If False, use all DUTs as specified in select_fit_duts and use them for track fitting if hits are available (potentially constrained residuals).
        If True, do not use hits form the actual fit DUT for track fitting, even if specified in select_fit_duts (unconstrained residuals).
    method : string
        Available methods are 'kalman', which uses a Kalman Filter for track calculation, and 'fit', which uses a simple
        straight line fit for track calculation.
    beam_energy : float
        Energy of the beam in MeV, e.g., 2500.0 MeV for ELSA beam. Only used for the Kalman Filter.
    particle_mass : float
        Mass of the particle in MeV, e.g., 0.511 MeV for electrons. Only used for the Kalman Filter.
    scattering_planes : list or dict
        Specifies additional scattering planes in case of DUTs which are not used or additional material in the way of the tracks.
        The list must contain dictionaries containing the following keys:
            material_budget: material budget of the scattering plane
            translation_x/translation_y/translation_z: x/y/z position of the plane (in um)
            rotation_alpha/rotation_beta/rotation_gamma: alpha/beta/gamma angle of scattering plane (in radians)
        The material budget is defined as the thickness devided by the radiation length.
        If scattering_planes is None, no scattering plane will be added.
    quality_distances : 2-tuple or list of 2-tuples
        X and y distance (in um) for each DUT to calculate the quality flag. The selected track and corresponding hit
        must have a smaller distance to have the quality flag to be set to 1.
        If None, use infinite distance.
    reject_quality_distances : 2-tuple or list of 2-tuples
        X and y distance (in um) for each DUT to calculate the quality flag. Any other occurence of tracks or hits from the same event
        within this distance will reject the quality flag.
        If None, use infinite distance.
    use_limits : bool
        If True, use column and row limits from pre-alignment for selecting the data.
    keep_data : bool
        Keep all track candidates in data and add track info only to fitted tracks. Necessary for purity calculations.
    full_track_info : bool
        If True, the track vector and position of all DUTs is appended to track table in order to get the full track information.
        If False, only the track vector and position of the actual fit DUT is appended to track table.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_tracks_file : string
        Filename of the output tracks file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Fitting tracks of %d DUTs ===' % n_duts)

    method = method.lower()
    if method not in ["fit", "kalman"]:
        raise ValueError('Unknown method "%s"!' % method)

    if method == "kalman" and not beam_energy:
        raise ValueError('Beam energy not given (in MeV).')

    if method == "kalman" and not particle_mass:
        raise ValueError('Particle mass not given (in MeV).')

    if output_tracks_file is None:
        output_tracks_file = os.path.join(os.path.dirname(input_track_candidates_file), 'Tracks_%s.h5' % method.title())

    if select_duts is None:
        select_duts = range(n_duts)  # standard setting: fit tracks for all DUTs
    elif not isinstance(select_duts, Iterable):
        select_duts = [select_duts]
    # Check for duplicates
    if len(select_duts) != len(set(select_duts)):
        raise ValueError("found douplicate in select_duts")
    # Check if any iterable in iterable
    if any(map(lambda val: isinstance(val, Iterable), select_duts)):
        raise ValueError("item in select_duts is iterable")

    # Create track, hit selection
    if select_fit_duts is None:  # If None: use all DUTs
        select_fit_duts = range(n_duts)
#         # copy each item
#         for hit_duts in select_hit_duts:
#             select_fit_duts.append(hit_duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(select_fit_duts, Iterable):
        raise ValueError("select_fit_duts is no iterable")
    elif not select_fit_duts:  # empty iterable
        raise ValueError("select_fit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_fit_duts)):
        select_fit_duts = [select_fit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_fit_duts)):
        raise ValueError("not all items in select_fit_duts are iterable")
    # Finally check length of all arrays
    if len(select_fit_duts) != len(select_duts):  # empty iterable
        raise ValueError("select_fit_duts has the wrong length")
    for index, fit_dut in enumerate(select_fit_duts):
        if len(fit_dut) < 2:  # check the length of the items
            raise ValueError("item in select_fit_duts has length < 2")
#         if set(fit_dut) - set(select_hit_duts[index]):  # fit DUTs are required to have a hit
#             raise ValueError("DUT in select_fit_duts is not in select_hit_duts")

    # Create track, hit selection
    if select_hit_duts is None:  # If None, require no hit
        #         select_hit_duts = range(n_duts)
        select_hit_duts = []
    # Check iterable and length
    if not isinstance(select_hit_duts, Iterable):
        raise ValueError("select_hit_duts is no iterable")
#     elif not select_hit_duts:  # empty iterable
#         raise ValueError("select_hit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_hit_duts)):
        select_hit_duts = [select_hit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_hit_duts)):
        raise ValueError("not all items in select_hit_duts are iterable")
    # Finally check length of all arrays
    if len(select_hit_duts) != len(select_duts):  # empty iterable
        raise ValueError("select_hit_duts has the wrong length")
#     for hit_dut in select_hit_duts:
#         if len(hit_dut) < 2:  # check the length of the items
#             raise ValueError("item in select_hit_duts has length < 2")

    # Create quality distance
    if isinstance(quality_distances, tuple) or quality_distances is None:
        quality_distances = [quality_distances] * n_duts
    # Check iterable and length
    if not isinstance(quality_distances, Iterable):
        raise ValueError("quality_distances is no iterable")
    elif not quality_distances:  # empty iterable
        raise ValueError("quality_distances has no items")
    # Finally check length of all arrays
    if len(quality_distances) != n_duts:  # empty iterable
        raise ValueError("quality_distances has the wrong length")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, quality_distances)):
        raise ValueError("not all items in quality_distances are iterable")
    # Finally check length of all arrays
    for distance in quality_distances:
        if distance is not None and len(distance) != 2:  # check the length of the items
            raise ValueError("item in quality_distances has length != 2")

    # Create reject quality distance
    if isinstance(reject_quality_distances, tuple) or reject_quality_distances is None:
        reject_quality_distances = [reject_quality_distances] * n_duts
    # Check iterable and length
    if not isinstance(reject_quality_distances, Iterable):
        raise ValueError("reject_quality_distances is no iterable")
    elif not reject_quality_distances:  # empty iterable
        raise ValueError("reject_quality_distances has no items")
    # Finally check length of all arrays
    if len(reject_quality_distances) != n_duts:  # empty iterable
        raise ValueError("reject_quality_distances has the wrong length")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, reject_quality_distances)):
        raise ValueError("not all items in reject_quality_distances are iterable")
    # Finally check length of all arrays
    for distance in reject_quality_distances:
        if distance is not None and len(distance) != 2:  # check the length of the items
            raise ValueError("item in reject_quality_distances has length != 2")

    # Check iterable and length
    if not isinstance(exclude_dut_hit, Iterable):
        exclude_dut_hit = [exclude_dut_hit] * len(select_duts)
    elif not exclude_dut_hit:  # empty iterable
        raise ValueError("exclude_dut_hit has no items")
    # Finally check length of all array
    if len(exclude_dut_hit) != len(select_duts):  # empty iterable
        raise ValueError("exclude_dut_hit has the wrong length")
    # Check if only bools in iterable
    if not all(map(lambda val: isinstance(val, (bool,)), exclude_dut_hit)):
        raise ValueError("not all items in exclude_dut_hit are boolean")

    fitted_duts = []
    pool = Pool()
    with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
        with tb.open_file(output_tracks_file, mode='w') as out_file_h5:
            for fit_dut_index, actual_fit_dut in enumerate(select_duts):  # Loop over the DUTs where tracks shall be fitted for
                if actual_fit_dut in fitted_duts:
                    continue
                # test whether other DUTs have identical tracks
                # if yes, save some CPU time and fit only once
                actual_fit_duts = []
                for curr_fit_dut_index, curr_fit_dut in enumerate(select_duts):
                    if ((exclude_dut_hit[curr_fit_dut_index] is False and exclude_dut_hit[fit_dut_index] is False) or (actual_fit_dut not in select_fit_duts[fit_dut_index])) and set(select_hit_duts[curr_fit_dut_index]) == set(select_hit_duts[fit_dut_index]) and set(select_fit_duts[curr_fit_dut_index]) == set(select_fit_duts[fit_dut_index]):
                        actual_fit_duts.append(curr_fit_dut)
                logging.info('= Fit tracks for %s =', ', '.join([telescope[curr_dut].name for curr_dut in actual_fit_duts]))
                # remove existing nodes
                for dut_index in actual_fit_duts:
                    try:  # Check if table already exists, then append data
                        out_file_h5.remove_node(out_file_h5.root, name='Tracks_DUT%d' % dut_index)
                        logging.info('Overwriting existing tracks for DUT%d', dut_index)
                    except tb.NodeError:  # Table does not exist, thus create new
                        pass

                total_n_tracks = in_file_h5.root.TrackCandidates.shape[0]
                total_n_tracks_stored = 0
                total_n_events_stored = 0

                # select hit DUTs based on input parameters
                dut_hit_selection = 0  # DUTs required to have hits
                hit_duts = list(set(select_hit_duts[fit_dut_index]) - set([actual_fit_duts])) if exclude_dut_hit[fit_dut_index] else select_hit_duts[fit_dut_index]
                for dut_index in hit_duts:
                    dut_hit_selection |= ((1 << dut_index))
                logging.info('Require hits in %d DUTs for track selection: %s', len(hit_duts), ', '.join([telescope[curr_dut].name for curr_dut in hit_duts]))
                # select fit DUTs based on input parameters
                dut_fit_selection = 0  # DUTs to be used for the fit
                fit_duts = list(set(select_fit_duts[fit_dut_index]) - set([actual_fit_duts])) if exclude_dut_hit[fit_dut_index] else select_fit_duts[fit_dut_index]
                for dut_index in fit_duts:
                    dut_fit_selection |= ((1 << dut_index))
                logging.info("Use %d DUTs for track fit: %s", len(fit_duts), ', '.join([telescope[curr_dut].name for curr_dut in fit_duts]))
                if len(fit_duts) < 2 and method == "fit":
                    raise ValueError('The number of required hit DUTs is smaller than 2. Cannot fit tracks for %s.', telescope[actual_fit_dut].name)
                widgets = ['', progressbar.Percentage(), ' ',
                           progressbar.Bar(marker='*', left='|', right='|'),
                           ' ', progressbar.AdaptiveETA()]
                progress_bar = progressbar.ProgressBar(widgets=widgets,
                                                       maxval=total_n_tracks,
                                                       term_width=80)
#                 progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=max_tracks if max_tracks is not None else in_file_h5.root.TrackCandidates.shape[0], term_width=80)
                progress_bar.start()

                for track_candidates_chunk, index_chunk in analysis_utils.data_aligned_at_events(in_file_h5.root.TrackCandidates, chunk_size=chunk_size):
                    #                     if max_tracks is not None and total_n_tracks >= max_tracks:
                    #                         break
                    # Select tracks based on the DUTs that are required to have a hit (dut_selection) with a certain quality (track_quality)
                    n_tracks_chunk = track_candidates_chunk.shape[0]
                    good_track_selection = (track_candidates_chunk['hit_flag'] & dut_hit_selection) == dut_hit_selection
                    # remove tracks that have only a single DUT with a hit
                    for index, bit in enumerate(np.binary_repr(dut_fit_selection)[::-1]):  # iterate from LSB to MSB
                        if bit == "0":
                            continue
                        dut_fit_selection_dut_removed = dut_fit_selection & ~(1 << index)
                        good_track_selection &= (track_candidates_chunk['hit_flag'] & dut_fit_selection_dut_removed) > 0
                    n_tracks_quality = np.count_nonzero(good_track_selection)

                    logging.info('Selected %d of %d (%.1f%%) track candidates for track fitting due to hit requirements',
                                 n_tracks_quality,
                                 n_tracks_chunk,
                                 100.0 * n_tracks_quality / n_tracks_chunk)

#                     if max_tracks is not None:
#                         cut_index = np.where(np.cumsum(good_track_selection) + total_n_tracks > max_tracks)[0]
#                         print "cut index", cut_index
#                         if len(cut_index) > 0:
#                             event_indices = np.where(track_candidates_chunk["event_number"][:-1] != track_candidates_chunk["event_number"][1:])[0] + 1
#                             event_cut_index = event_indices[event_indices >= cut_index[0]][0]
# #                             print track_candidates_chunk[event_cut_index-2:event_cut_index+2]["event_number"]
# #                             print track_candidates_chunk[event_cut_index-2:event_cut_index]["event_number"]
#                             good_track_selection = good_track_selection[:event_cut_index]
#                             track_candidates_chunk = track_candidates_chunk[:event_cut_index]
# #                             print "event_cut_index", event_cut_index, total_n_tracks, max_tracks

                    unique_events = np.unique(track_candidates_chunk["event_number"][good_track_selection])
                    n_events_chunk = unique_events.shape[0]
                    if n_events_chunk == 0:
                        continue

                    # print "n_events_chunk", n_events_chunk
                    # print "n_tracks_chunk", n_tracks_chunk
                    if max_events:
                        if total_n_tracks == index_chunk:  # last chunk, adding all remaining events
                            select_n_events = max_events - total_n_events_stored
                        elif total_n_events_stored == 0:  # first chunk
                            select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks)))
                        else:
                            # calculate correction of number of selected events
                            correction = (total_n_tracks - index_chunk) / total_n_tracks * 1 / (((total_n_tracks - last_index_chunk) / total_n_tracks) / ((max_events - total_n_events_stored_last) / max_events)) \
                                + (index_chunk) / total_n_tracks * 1 / (((last_index_chunk) / total_n_tracks) / ((total_n_events_stored_last) / max_events))
    #                         select_n_events = np.ceil(n_events_chunk * correction)
    #                         # calculate correction of number of selected events
    #                         correction = 1/(((total_n_tracks-last_index_chunk)/total_n_tracks_last)/((max_events-total_n_events_stored_last)/max_events))
                            select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks) * correction))
                            # print "correction", correction
                        # do not store more events than in current chunk
                        select_n_events = min(n_events_chunk, select_n_events)
                        # do not store more events than given by max_events
                        select_n_events = min(select_n_events, max_events - total_n_events_stored)
                        np.random.seed(seed=0)
                        selected_events = np.random.choice(unique_events, size=select_n_events, replace=False)
                        store_n_events = selected_events.shape[0]
                        total_n_events_stored += store_n_events
                        # print "store_n_events", store_n_events
                        good_track_selection &= np.in1d(track_candidates_chunk["event_number"], selected_events)
                        # TODO: total_n_tracks_stored not used...
                        store_n_tracks = np.count_nonzero(good_track_selection)
                        total_n_tracks_stored += store_n_tracks
#                         track_candidates_chunk = track_candidates_chunk[select_tracks]

                    # Prepare track hits array to be fitted
                    n_good_tracks = np.count_nonzero(good_track_selection)  # Index of tmp track hits array
                    if method == "fit":
                        track_hits = np.full((n_good_tracks, len(fit_duts), 3), fill_value=np.nan, dtype=np.float32)
                    elif method == "kalman":
                        track_hits = np.full((n_good_tracks, n_duts, 6), fill_value=np.nan, dtype=np.float32)

                    # print "hit flags", np.unique(track_candidates_chunk['hit_flag'][good_track_selection])  # , np.min(track_candidates_chunk['hit_flag'][good_track_selection])
                    # print "quality flags", np.unique(track_candidates_chunk['quality_flag'][good_track_selection])  # , np.min(track_candidates_chunk['quality_flag'][good_track_selection])
                    fit_array_index = 0
                    for dut_index, dut in enumerate(telescope):  # Fill index loop of new array
                        # Check if DUT is used for fit
                        if method == "fit" and dut_index in fit_duts:
                            # apply alignment for fitting the tracks
                            track_hits[:, fit_array_index, 0], track_hits[:, fit_array_index, 1], track_hits[:, fit_array_index, 2] = dut.local_to_global_position(
                                x=track_candidates_chunk['x_dut_%s' % dut_index][good_track_selection],
                                y=track_candidates_chunk['y_dut_%s' % dut_index][good_track_selection],
                                z=track_candidates_chunk['z_dut_%s' % dut_index][good_track_selection])
                            # increase index for tracks hits array
                            fit_array_index += 1
                        elif method == "kalman":
                            # TODO: taking telescope alignment into account for initial state
                            # apply alignment for fitting the tracks
                            track_hits[:, dut_index, 0], track_hits[:, dut_index, 1], track_hits[:, dut_index, 2] = dut.local_to_global_position(
                                x=track_candidates_chunk['x_dut_%s' % dut_index][good_track_selection],
                                y=track_candidates_chunk['y_dut_%s' % dut_index][good_track_selection],
                                z=track_candidates_chunk['z_dut_%s' % dut_index][good_track_selection])
                            track_hits[:, dut_index, 3], track_hits[:, dut_index, 4], track_hits[:, dut_index, 5] = np.abs(dut.local_to_global_position(
                                x=track_candidates_chunk['x_err_dut_%s' % dut_index][good_track_selection],
                                y=track_candidates_chunk['y_err_dut_%s' % dut_index][good_track_selection],
                                z=track_candidates_chunk['z_err_dut_%s' % dut_index][good_track_selection],
                                # no translation for the errors
                                translation_x=0.0,
                                translation_y=0.0,
                                translation_z=0.0))

                    # Split data and fit on all available cores
                    n_slices = cpu_count()
                    track_hits_slices = np.array_split(track_hits, n_slices)
                    if method == "fit":
                        results = [pool.apply_async(_fit_tracks_loop, kwds={
                            'track_hits': track_hits_slice}) for track_hits_slice in track_hits_slices]
                    elif method == "kalman":
                        results = [pool.apply_async(_fit_tracks_kalman_loop, kwds={
                            'track_hits': track_hits_slice,
                            'telescope': telescope,
                            'select_fit_duts': fit_duts,
                            'beam_energy': beam_energy,
                            'particle_mass': particle_mass,
                            'scattering_planes': scattering_planes}) for track_hits_slice in track_hits_slices]

                    # Store results
                    offsets = np.concatenate([result.get()[0] for result in results])  # Merge offsets from all cores in results
                    slopes = np.concatenate([result.get()[1] for result in results])  # Merge slopes from all cores in results
                    # Store the data
                    # Check if all DUTs were fitted at once
                    for index, dut_index in enumerate(actual_fit_duts):
                        store_track_data(
                            out_file_h5=out_file_h5,
                            track_candidates_chunk=track_candidates_chunk,
                            good_track_selection=good_track_selection,
                            telescope=telescope,
                            offsets=offsets,
                            slopes=slopes,
                            fit_dut=dut_index,
                            select_fit_duts=fit_duts,
                            quality_distances=quality_distances,
                            reject_quality_distances=reject_quality_distances,
                            use_limits=use_limits,
                            keep_data=keep_data,
                            method=method,
                            full_track_info=full_track_info)

                    # total_n_tracks += n_good_tracks
                    total_n_events_stored_last = total_n_events_stored
                    total_n_tracks_last = total_n_tracks
                    last_index_chunk = index_chunk
                    progress_bar.update(index_chunk)
                    # progress_bar.update(min(total_n_tracks, max_tracks) if max_tracks is not None else index_chunk)
                progress_bar.finish()
                # print "***************"
                # print "total_n_tracks_stored", total_n_tracks_stored
                # print "total_n_events_stored", total_n_events_stored
                fitted_duts.extend(actual_fit_duts)

    pool.close()
    pool.join()

    if plot:
        plot_utils.plot_track_chi2(input_tracks_file=output_tracks_file, output_pdf_file=None, dut_names=telescope.dut_names, chunk_size=chunk_size)

    return output_tracks_file


def store_track_data(out_file_h5, track_candidates_chunk, good_track_selection, telescope, offsets, slopes, fit_dut, select_fit_duts, quality_distances, reject_quality_distances, use_limits, keep_data, method, full_track_info):
    # xy_residuals_squared = np.empty((np.count_nonzero(good_track_selection), len(telescope)), dtype=np.float32)
    x_residuals_squared = np.empty((np.count_nonzero(good_track_selection), len(telescope)), dtype=np.float32)
    x_err_squared = np.empty((np.count_nonzero(good_track_selection), len(telescope)), dtype=np.float32)
    y_residuals_squared = np.empty((np.count_nonzero(good_track_selection), len(telescope)), dtype=np.float32)
    y_err_squared = np.empty((np.count_nonzero(good_track_selection), len(telescope)), dtype=np.float32)
    # reset quality flag
    quality_flag = np.zeros(np.count_nonzero(good_track_selection), dtype=track_candidates_chunk["hit_flag"].dtype)
    if full_track_info:
        track_estimates_chunk_full = np.full(shape=(np.count_nonzero(good_track_selection), len(telescope), 6), fill_value=np.nan, dtype=np.float32)
    else:
        track_estimates_chunk_full = None
    for dut_index, dut in enumerate(telescope):
        if use_limits:
            # local_limit_positions = dut.index_to_local_position(
            #     column=[dut.column_limit[0], dut.column_limit[1]],
            #     row=[dut.row_limit[0], dut.row_limit[1]])
            # limit_x_local = [local_limit_positions[0][0], local_limit_positions[0][1]]
            # limit_y_local = [local_limit_positions[1][0], local_limit_positions[1][1]]
            limit_x_local = dut.column_limit
            limit_y_local = dut.row_limit
        else:
            limit_x_local = None
            limit_y_local = None

        hit_x_local = track_candidates_chunk['x_dut_%s' % dut_index][good_track_selection]
        hit_y_local = track_candidates_chunk['y_dut_%s' % dut_index][good_track_selection]

        if method == "fit":
            # Set the offset to the track intersection with the tilted plane
            intersections_global = geometry_utils.get_line_intersections_with_dut(
                line_origins=offsets,
                line_directions=slopes,
                translation_x=dut.translation_x,
                translation_y=dut.translation_y,
                translation_z=dut.translation_z,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)
            slopes_x_local, slopes_y_local, slopes_z_local = dut.global_to_local_position(
                x=slopes[:, 0],
                y=slopes[:, 1],
                z=slopes[:, 2],
                translation_x=0.0,
                translation_y=0.0,
                translation_z=0.0,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)
        elif method == "kalman":
            # Set the offset to the track intersection with the tilted plane
            intersections_global = geometry_utils.get_line_intersections_with_dut(
                line_origins=offsets[:, dut_index],
                line_directions=slopes[:, dut_index],
                translation_x=dut.translation_x,
                translation_y=dut.translation_y,
                translation_z=dut.translation_z,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)
            slopes_x_local, slopes_y_local, slopes_z_local = dut.global_to_local_position(
                x=slopes[:, dut_index, 0],
                y=slopes[:, dut_index, 1],
                z=slopes[:, dut_index, 2],
                translation_x=0.0,
                translation_y=0.0,
                translation_z=0.0,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)

        # force the 3rd component (z) to be positive
        # and normalize to 1
        slopes_local = np.column_stack([slopes_x_local, slopes_y_local, slopes_z_local])
        slopes_local[slopes_local[:, 2] < 0.0] = -slopes_local[slopes_local[:, 2] < 0.0]
        slope_local_mag = np.sqrt(np.einsum('ij,ij->i', slopes_local, slopes_local))
        slopes_local /= slope_local_mag[:, np.newaxis]
        slopes_x_local, slopes_y_local, slopes_z_local = slopes_local[:, 0], slopes_local[:, 1], slopes_local[:, 2]

        intersection_x_local, intersection_y_local, intersection_z_local = dut.global_to_local_position(
            x=intersections_global[:, 0],
            y=intersections_global[:, 1],
            z=intersections_global[:, 2])

        x_residuals = hit_x_local - intersection_x_local
        y_residuals = hit_y_local - intersection_y_local

        # xy_residuals_squared[:, dut_index] = np.square(hit_x_local - intersection_x_local) + np.square(hit_y_local - intersection_y_local)
        x_residuals_squared[:, dut_index] = np.square(x_residuals)
        x_err_squared[:, dut_index] = np.square(track_candidates_chunk['x_err_dut_%d' % dut_index][good_track_selection])
        y_residuals_squared[:, dut_index] = np.square(y_residuals)
        y_err_squared[:, dut_index] = np.square(track_candidates_chunk['y_err_dut_%d' % dut_index][good_track_selection])

        # generate quality array
        dut_quality_flag_sel = np.ones_like(intersection_x_local, dtype=np.bool)
        select_valid_hit = np.isfinite(hit_x_local)
        select_finite_distance = np.isfinite(x_residuals)
        select_finite_distance &= np.isfinite(y_residuals)
        dut_quality_flag_sel[~select_valid_hit] = False
        dut_quality_flag_sel[~select_finite_distance] = False
        # select tracks within limits and set quality flag
        if limit_x_local is not None and np.isfinite(limit_x_local[0]):
            dut_quality_flag_sel[select_valid_hit] &= (hit_x_local[select_valid_hit] >= limit_x_local[0])
        if limit_x_local is not None and np.isfinite(limit_x_local[1]):
            dut_quality_flag_sel[select_valid_hit] &= (hit_x_local[select_valid_hit] <= limit_x_local[1])
        if limit_y_local is not None and np.isfinite(limit_y_local[0]):
            dut_quality_flag_sel[select_valid_hit] &= (hit_y_local[select_valid_hit] >= limit_y_local[0])
        if limit_y_local is not None and np.isfinite(limit_y_local[1]):
            dut_quality_flag_sel[select_valid_hit] &= (hit_y_local[select_valid_hit] <= limit_y_local[1])
        # distance for quality flag calculation
        if quality_distances[dut_index] is None:
            quality_distance_x = np.inf
            quality_distance_y = np.inf
        else:
            quality_distance_x = quality_distances[dut_index][0]
            quality_distance_y = quality_distances[dut_index][1]

        # select data where distance between the hit and track is smaller than the given value and set quality flag
        if quality_distance_x >= 2.5 * dut.pixel_size[0] and quality_distance_y >= 2.5 * dut.pixel_size[1]:  # use ellipse
            use_ellipse = True
            dut_quality_flag_sel[select_finite_distance] &= ((x_residuals_squared[select_finite_distance, dut_index] / quality_distance_x**2) + (y_residuals_squared[select_finite_distance, dut_index] / quality_distance_y**2)) <= 1
        else:  # use square
            use_ellipse = False
            dut_quality_flag_sel[select_finite_distance] &= (np.abs(x_residuals[select_finite_distance]) <= quality_distance_x)
            dut_quality_flag_sel[select_finite_distance] &= (np.abs(y_residuals[select_finite_distance]) <= quality_distance_y)
        quality_flag[dut_quality_flag_sel] |= np.uint32((1 << dut_index))

        # distance to find close-by hits and tracks
        if reject_quality_distances[dut_index] is None:
            reject_quality_distance_x = np.inf
            reject_quality_distance_y = np.inf
        else:
            reject_quality_distance_x = reject_quality_distances[dut_index][0]
            reject_quality_distance_y = reject_quality_distances[dut_index][1]
        # Select tracks that are too close when extrapolated to the actual DUT
        # All selected tracks will result in a quality_flag = 0 for the actual DUT
        dut_small_track_distance_flag_sel = np.zeros_like(dut_quality_flag_sel)
        _find_small_distance(
            event_number_array=track_candidates_chunk['event_number'][good_track_selection],
            position_array_x=intersection_x_local,
            position_array_y=intersection_y_local,
            max_distance_x=reject_quality_distance_x,
            max_distance_y=reject_quality_distance_y,
            small_distance_flag_array=dut_small_track_distance_flag_sel,
            use_ellipse=True)
        # logging.info("Unset track quality flag for %d of %d tracks for DUT%d due to close-by tracks", np.count_nonzero(dut_small_track_distance_flag_sel & dut_quality_flag_sel), np.count_nonzero(dut_quality_flag_sel), dut_index)
        # unset quality flag
        quality_flag[dut_small_track_distance_flag_sel] &= np.uint32(~(1 << dut_index))

        # Select hits that are too close in a DUT
        # All selected hits will result in a quality_flag = 0 for the actual DUT
        dut_small_hit_distance_flag_sel = np.zeros_like(good_track_selection)
        _find_small_distance(
            event_number_array=track_candidates_chunk['event_number'],
            position_array_x=track_candidates_chunk['x_dut_%s' % dut_index],
            position_array_y=track_candidates_chunk['y_dut_%s' % dut_index],
            max_distance_x=reject_quality_distance_x,
            max_distance_y=reject_quality_distance_y,
            small_distance_flag_array=dut_small_hit_distance_flag_sel,
            use_ellipse=use_ellipse)
        # logging.info("Unset track quality flag for %d of %d tracks for DUT%d due to close-by hits", np.count_nonzero(dut_small_hit_distance_flag_sel[good_track_selection] & dut_quality_flag_sel), np.count_nonzero(dut_quality_flag_sel), dut_index)
        # unset quality flag
        quality_flag[dut_small_hit_distance_flag_sel[good_track_selection]] &= np.uint32(~(1 << dut_index))

        if dut_index == fit_dut:
            # use offsets at the location of the fit DUT, local coordinates
            dut_offsets = np.column_stack([
                intersection_x_local,
                intersection_y_local,
                intersection_z_local])
            # use slopes at the location of the fit DUT, local coordinates
            dut_slopes = np.column_stack([
                slopes_x_local,
                slopes_y_local,
                slopes_z_local])

        if full_track_info:
            track_estimates_chunk_full[:, dut_index] = np.column_stack([
                intersection_x_local,
                intersection_y_local,
                intersection_z_local,
                slopes_x_local,
                slopes_y_local,
                slopes_z_local])

    # calculate the sum of the squared x/y residuals of the fit DUT planes in the local coordinate system, divided by n fit DUT hits per track (normalization)
    # track_chi2s = np.sum(np.ma.masked_invalid(xy_residuals_squared[:, select_fit_duts]), axis=1) / np.count_nonzero(~np.isnan(xy_residuals_squared[:, select_fit_duts]), axis=1)
    track_chi2s = (np.sum(np.ma.masked_invalid(x_residuals_squared[:, select_fit_duts] / x_err_squared[:, select_fit_duts]), axis=1) + np.sum(np.ma.masked_invalid(y_residuals_squared[:, select_fit_duts] / y_err_squared[:, select_fit_duts]), axis=1))
    # select tracks that have more than 2 data points
    select_nonzero = (track_chi2s != 0.0)
    # divide by d.o.f.
    track_chi2s[select_nonzero] /= (2 * (np.count_nonzero(~np.isnan(x_residuals_squared[:, select_fit_duts]), axis=1)[select_nonzero] - 2))

    tracks_array = create_results_array(
        n_duts=len(telescope),
        dut_offsets=dut_offsets,
        dut_slopes=dut_slopes,
        track_chi2s=track_chi2s,
        quality_flag=quality_flag,
        good_track_selection=good_track_selection,
        track_candidates_chunk=track_candidates_chunk,
        keep_data=keep_data,
        track_estimates_chunk_full=track_estimates_chunk_full)

    try:  # Check if table exists already, then append data
        tracklets_table = out_file_h5.get_node('/Tracks_DUT%d' % fit_dut)
    except tb.NoSuchNodeError:  # Table does not exist, thus create new
        tracklets_table = out_file_h5.create_table(
            where=out_file_h5.root,
            name='Tracks_DUT%d' % fit_dut,
            description=tracks_array.dtype,
            title='%s tracks for DUT%d' % (method.title(), fit_dut),
            filters=tb.Filters(
                complib='blosc',
                complevel=5,
                fletcher32=False))

    tracklets_table.append(tracks_array)
    tracklets_table.flush()


@njit
def _find_small_distance(event_number_array, position_array_x, position_array_y, max_distance_x, max_distance_y, small_distance_flag_array, use_ellipse):
    max_index = event_number_array.shape[0]
    index = 0
    while index < max_index:
        current_event_number = event_number_array[index]
        while (index < max_index) and (event_number_array[index] == current_event_number):  # Next event reached, break loop
            event_index = index + 1
            while (event_index < max_index) and (event_number_array[event_index] == current_event_number):  # Loop over other event hits
                if np.isfinite(position_array_x[index]) and np.isfinite(position_array_x[event_index]):
                    # check if distance is smaller than limit
                    if use_ellipse:  # use ellipse
                        if ((np.square(position_array_x[index] - position_array_x[event_index]) / max_distance_x**2) + (np.square(position_array_y[index] - position_array_y[event_index]) / max_distance_y**2)) <= 1:
                            small_distance_flag_array[index] = 1
                            small_distance_flag_array[event_index] = 1
                    else:  # use square
                        if (abs(position_array_x[index] - position_array_x[event_index]) <= max_distance_x) and (abs(position_array_y[index] - position_array_y[event_index]) <= max_distance_y):
                            small_distance_flag_array[index] = 1
                            small_distance_flag_array[event_index] = 1
                event_index += 1
            index += 1


def create_results_array(n_duts, dut_offsets, dut_slopes, track_chi2s, quality_flag, good_track_selection, track_candidates_chunk, keep_data, track_estimates_chunk_full):
    # Tracks description, additional columns
    tracks_descr = []
    for dimension in ['x', 'y', 'z']:
        tracks_descr.append(('offset_%s' % dimension, track_candidates_chunk["x_dut_0"].dtype))
    for dimension in ['x', 'y', 'z']:
        tracks_descr.append(('slope_%s' % dimension, track_candidates_chunk["x_dut_0"].dtype))
    if track_estimates_chunk_full is not None:
        for index_dut in range(n_duts):
            for index in ['offset', 'slope']:
                for dimension in ['x', 'y', 'z']:
                    tracks_descr.append(('%s_%s_dut_%d' % (index, dimension, index_dut), track_candidates_chunk["x_dut_0"].dtype))
    tracks_descr.extend([('track_chi2', np.float32), ('quality_flag', track_candidates_chunk["hit_flag"].dtype)])

    # Select only fitted tracks (keep_data is False) or keep all track candidates (keep_data is True)
    if not keep_data:
        track_candidates_chunk = track_candidates_chunk[good_track_selection]

    tracks_array = np.empty((track_candidates_chunk.shape[0],), dtype=track_candidates_chunk.dtype.descr + tracks_descr)

    tracks_array['hit_flag'] = track_candidates_chunk['hit_flag']
    tracks_array['event_number'] = track_candidates_chunk['event_number']
    for index_dut in range(n_duts):
        tracks_array['x_dut_%d' % index_dut] = track_candidates_chunk['x_dut_%d' % index_dut]
        tracks_array['y_dut_%d' % index_dut] = track_candidates_chunk['y_dut_%d' % index_dut]
        tracks_array['z_dut_%d' % index_dut] = track_candidates_chunk['z_dut_%d' % index_dut]
        tracks_array['x_err_dut_%d' % index_dut] = track_candidates_chunk['x_err_dut_%d' % index_dut]
        tracks_array['y_err_dut_%d' % index_dut] = track_candidates_chunk['y_err_dut_%d' % index_dut]
        tracks_array['z_err_dut_%d' % index_dut] = track_candidates_chunk['z_err_dut_%d' % index_dut]
        tracks_array['charge_dut_%d' % index_dut] = track_candidates_chunk['charge_dut_%d' % index_dut]
        tracks_array['frame_dut_%d' % index_dut] = track_candidates_chunk['frame_dut_%d' % index_dut]
        tracks_array['n_hits_dut_%d' % index_dut] = track_candidates_chunk['n_hits_dut_%d' % index_dut]
        tracks_array['cluster_ID_dut_%d' % index_dut] = track_candidates_chunk['cluster_ID_dut_%d' % index_dut]
        tracks_array['cluster_shape_dut_%d' % index_dut] = track_candidates_chunk['cluster_shape_dut_%d' % index_dut]
        tracks_array['n_cluster_dut_%d' % index_dut] = track_candidates_chunk['n_cluster_dut_%d' % index_dut]

    if keep_data:
        for index, dimension in enumerate(['x', 'y', 'z']):
            tracks_array['offset_%s' % dimension][good_track_selection] = dut_offsets[:, index]
            tracks_array['slope_%s' % dimension][good_track_selection] = dut_slopes[:, index]
            tracks_array['offset_%s' % dimension][~good_track_selection] = np.nan
            tracks_array['slope_%s' % dimension][~good_track_selection] = np.nan
        if track_estimates_chunk_full is not None:
            for index_dut in range(n_duts):
                tracks_array['offset_x_dut_%d' % index_dut][good_track_selection] = track_estimates_chunk_full[:, index_dut, 0]
                tracks_array['offset_y_dut_%d' % index_dut][good_track_selection] = track_estimates_chunk_full[:, index_dut, 1]
                tracks_array['offset_z_dut_%d' % index_dut][good_track_selection] = track_estimates_chunk_full[:, index_dut, 2]
                tracks_array['slope_x_dut_%d' % index_dut][good_track_selection] = track_estimates_chunk_full[:, index_dut, 3]
                tracks_array['slope_y_dut_%d' % index_dut][good_track_selection] = track_estimates_chunk_full[:, index_dut, 4]
                tracks_array['slope_z_dut_%d' % index_dut][good_track_selection] = track_estimates_chunk_full[:, index_dut, 5]
                tracks_array['offset_x_dut_%d' % index_dut][~good_track_selection] = np.nan
                tracks_array['offset_y_dut_%d' % index_dut][~good_track_selection] = np.nan
                tracks_array['offset_z_dut_%d' % index_dut][~good_track_selection] = np.nan
                tracks_array['slope_x_dut_%d' % index_dut][~good_track_selection] = np.nan
                tracks_array['slope_y_dut_%d' % index_dut][~good_track_selection] = np.nan
                tracks_array['slope_z_dut_%d' % index_dut][~good_track_selection] = np.nan
        tracks_array['track_chi2'][good_track_selection] = track_chi2s
        tracks_array['track_chi2'][~good_track_selection] = np.nan
        tracks_array['quality_flag'][good_track_selection] = quality_flag
        tracks_array['quality_flag'][~good_track_selection] = 0
    else:
        for index, dimension in enumerate(['x', 'y', 'z']):
            tracks_array['offset_%s' % dimension] = dut_offsets[:, index]
            tracks_array['slope_%s' % dimension] = dut_slopes[:, index]
        if track_estimates_chunk_full is not None:
            for index_dut in range(n_duts):
                tracks_array['offset_x_dut_%d' % index_dut] = track_estimates_chunk_full[:, index_dut, 0]
                tracks_array['offset_y_dut_%d' % index_dut] = track_estimates_chunk_full[:, index_dut, 1]
                tracks_array['offset_z_dut_%d' % index_dut] = track_estimates_chunk_full[:, index_dut, 2]
                tracks_array['slope_x_dut_%d' % index_dut] = track_estimates_chunk_full[:, index_dut, 3]
                tracks_array['slope_y_dut_%d' % index_dut] = track_estimates_chunk_full[:, index_dut, 4]
                tracks_array['slope_z_dut_%d' % index_dut] = track_estimates_chunk_full[:, index_dut, 5]
        tracks_array['track_chi2'] = track_chi2s
        tracks_array['quality_flag'] = quality_flag

    return tracks_array


def _fit_tracks_loop(track_hits):
    '''
    Loop over the selected tracks. In this function all matrices for the Kalman Filter are calculated track by track
    and the Kalman Filter is started. With dut_fit_selection only the duts which are selected are included in the Kalman Filter.
    Not included DUTs are masked.

    Parameters
    ----------
    track_hits : array
        Array which contains the x, y and z hit position of each DUT for all tracks.

    Returns
    -------
    offset : array
        Array, which contains the track offsets.
    slope : array
        Array, which contains the track slopes.
    chi2 : array
        Array, which contains the track Chi^2.
    '''
    slope = np.empty((track_hits.shape[0], 3), dtype=np.float32)
    offset = np.empty((track_hits.shape[0], 3), dtype=np.float32)

    # Loop over selected track candidate hits and fit
    for index, hits in enumerate(track_hits):
        try:
            offset[index], slope[index] = line_fit_3d(positions=hits)
        except np.linalg.linalg.LinAlgError:
            offset[index], slope[index] = np.nan, np.nan

    return offset, slope


def line_fit_3d(positions, n=None):
    ''' Do 3D line fit and calculate chi2 for each fit.
    '''
    # remove NaNs from data
    positions = positions[~np.isnan(positions).any(axis=1)]
    if n:
        positions = positions[:n]
    # subtract mean for each component (x,y,z) for SVD calculation
    datamean = positions.mean(axis=0)
    offset = datamean
    # calculating offset and slope
    # TODO: mean calculation and substraction can be raplced with svd(cov(points))
    slope = np.linalg.svd(positions - datamean, full_matrices=False)[2][0]  # http://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
    # normalize to 1
    slope_mag = np.sqrt(slope.dot(slope))
    # force the 3rd component (z) to be positive
    if slope[2] < 0:
        slope = -slope
    # intersections = offset + slope / slope[2] * (positions.T[2][:, np.newaxis] - offset[2])  # Fitted line and DUT plane intersections (here: points)
    # # calculate the sum of the squared x/y residuals
    # chi2 = np.sum(np.square(positions - intersections))
    return offset, slope / slope_mag  # , chi2


def _fit_tracks_kalman_loop(track_hits, telescope, select_fit_duts, beam_energy, particle_mass, scattering_planes):
    '''
    Loop over the selected tracks. In this function all matrices for the Kalman Filter are calculated track by track
    and the Kalman Filter is started. With dut_fit_selection only the duts which are selected are included in the Kalman Filter.
    Not included DUTs are masked.

    Parameters
    ----------
    track_hits : array
        Array which contains the x/y/z hit position and error for all DUTs and all tracks.
    telsescope : object
        Telescope object.
    select_fit_duts : list
        The select_fit_duts is a subset of all DUT indices. A DUT that is not included, will be omitted during the filtering step.
    beam_energy : float
        Energy of the beam in MeV, e.g., 2500.0 MeV for ELSA beam.
    particle_mass : float
        Mass of the particle in MeV, e.g., 0.511 MeV for electrons.
    scattering_planes : list or dict
        Specifies additional scattering planes in case of DUTs which are not used or additional material in the way of the tracks.
        The list must contain dictionaries containing the following keys:
            material_budget: material budget of the scattering plane
            translation_x/translation_y/translation_z: x/y/z position of the plane (in um)
            rotation_alpha/rotation_beta/rotation_gamma: alpha/beta/gamma angle of scattering plane (in radians)
        If scattering_planes is None, no scattering plane will be added.

    Returns
    -------
    smoothed_state_estimates : array_like
        Smoothed state vectors, which contains (smoothed x position, smoothed y position, slope_x, slope_y).
    chi2 : uint
        Chi2 of track.
    x_err : array_like
        Error of smoothed hit position in x direction. Calculated from smoothed
        state covariance matrix. Only approximation, since only diagonal element is taken.
    y_err : array_like
        Error of smoothed hit position in y direction. Calculated from smoothed
        state covariance matrix. Only approximation, since only diagonal element is taken.
    '''
    if scattering_planes is None:
        scattering_planes = []
    elif isinstance(scattering_planes, dict):
        scattering_planes = [scattering_planes]
    alignment = []
    material_budget = []
    all_dut_planes = [dut for dut in telescope]
    all_dut_planes.extend(scattering_planes)
    for dut in all_dut_planes:
        alignment.append([dut.translation_x, dut.translation_y, dut.translation_z, dut.rotation_alpha, dut.rotation_beta, dut.rotation_gamma])
        # TODO: take rotation into account for material budget
        material_budget.append(dut.material_budget)
    alignment = np.array(alignment)
    material_budget = np.array(material_budget)

    # calculating DUT indices list with z-order
    intersections_z_axis = []
    for dut in all_dut_planes:
        intersections_z_axis.append(geometry_utils.get_line_intersections_with_dut(
            line_origins=np.array([[0.0, 0.0, 0.0]]),
            line_directions=np.array([[0.0, 0.0, 1.0]]),
            translation_x=dut.translation_x,
            translation_y=dut.translation_y,
            translation_z=dut.translation_z,
            rotation_alpha=dut.rotation_alpha,
            rotation_beta=dut.rotation_beta,
            rotation_gamma=dut.rotation_gamma)[0][2])
    z_sorted_dut_indices = np.argsort(intersections_z_axis)
    # z_sorted_fit_dut_indices = []
    # for dut_index in z_sorted_dut_indices:
    #     if dut_index in select_fit_duts:
    #         z_sorted_fit_dut_indices.append(dut_index)

    # TODO: check if calculation of pixel size is necessary
    # try:
    #     first_dut = telescope[z_sorted_dut_indices[0]]
    #     first_dut_x_pixel_size, first_dut_y_pixel_size, _ = np.abs(first_dut.local_to_global_position(
    #         x=[first_dut.column_size],
    #         y=[first_dut.row_size],
    #         z=[0.0],
    #         translation_x=0.0,
    #         translation_y=0.0,
    #         translation_z=0.0))
    #     first_dut_pixel_size = [first_dut_x_pixel_size[0], first_dut_y_pixel_size[0]]
    # except AttributeError:  # First plane is scattering plane
    #     first_dut_pixel_size = [0.0, 0.0]

    if scattering_planes:
        track_hits = np.append(arr=track_hits, values=np.full((track_hits.shape[0], track_hits.shape[2] * len(scattering_planes)), fill_value=np.nan, dtype=np.float64), axis=1)

    chunk_size = track_hits.shape[0]
    n_duts = len(all_dut_planes)

    # Calculate multiple scattering
    momentum = np.sqrt(beam_energy**2 - particle_mass**2)
    beta = momentum / beam_energy  # almost 1

    if np.any(np.isclose(material_budget[z_sorted_dut_indices[:-1]], 0.0)):
        raise ValueError("Material budget is zero.")

    # rms angle of multiple scattering
    thetas = np.array(((13.6 / momentum / beta) * np.sqrt(material_budget) * (1. + 0.038 * np.log(material_budget))))

    # express transition and observation offset matrices
    # these are additional offsets, which are not used at the moment
    transition_offsets = np.full((chunk_size, n_duts, 6), fill_value=np.nan, dtype=np.float64)
    transition_offsets[:, z_sorted_dut_indices[:-1], :] = 0.0
    observation_offsets = np.zeros((chunk_size, n_duts, 3), dtype=np.float64)

    # express initial state. Contains (x_pos, y_pos, z_pos, slope_x, slope_y, slope_z).
    initial_state_mean = np.zeros((chunk_size, 6), dtype=np.float64)

    # express observation matrix, only observe (x,y,z)
    observation_matrices = np.zeros((chunk_size, n_duts, 3, 6), dtype=np.float64)
    observation_matrices[:, :, 0, 0] = 1.0
    observation_matrices[:, :, 1, 1] = 1.0
    observation_matrices[:, :, 2, 2] = 1.0
    # express observation covariance matrices
    observation_covariances = np.zeros((chunk_size, n_duts, 3, 3), dtype=np.float64)

    # express initial state covariance matrices
    initial_state_covariance = np.zeros((chunk_size, 6, 6), dtype=np.float64)
    # error on initial slope is roughly divergence of beam (5 mrad).
    initial_state_covariance[:, 3, 3] = np.square(5e-3)
    initial_state_covariance[:, 4, 4] = np.square(5e-3)
    initial_state_covariance[:, 5, 5] = np.square(5e-3)

    for index, actual_hits in enumerate(track_hits):  # Loop over selected track candidate hits and fit
        # Take cluster hit position error as measurement error for duts which have a hit.
        # For those who have no hit, need no error, since the should not be included in fit via fit selection
        duts_with_hits = np.array(range(n_duts), dtype=np.int)[~np.isnan(actual_hits[:, 0])]
        observation_covariances[index, duts_with_hits, 0, 0] = np.square(actual_hits[duts_with_hits, 3])
        observation_covariances[index, duts_with_hits, 1, 1] = np.square(actual_hits[duts_with_hits, 4])
        # FIXME: include meaningful error on z-position and investigate behavior of z_err
        # observation_covariances[index, duts_with_hits, 2, 2] = np.square(actual_hits[duts_with_hits, 5])

        if np.isnan(actual_hits[z_sorted_dut_indices[0], 0]):  # The first plane has no hit
            # Take planes from fit selction and fit a line to the hits,
            # then extrapolate the line to first plane in order to find initial state.
            # The position error is estimated with the pixel size.
            # TODO: Can't we handle this as any other scattering plane with error=0?
            # Edit: Any plane without hit is treated as scatter plane.
            try:
                # Fit all DUTs with hits
                offset, slope = line_fit_3d(positions=actual_hits[select_fit_duts, :3])
                # TODO: For lower energies and lighter particles use the first hit DUT as position for the first scatter plane
                # Fit the first 2 DUTs with hits
                # offset, slope = line_fit_3d(positions=actual_hits[z_sorted_fit_dut_indices, :3], n=2)
            except np.linalg.linalg.LinAlgError:
                offset, slope = np.nan, np.nan

            intersections = geometry_utils.get_line_intersections_with_dut(
                line_origins=np.array([offset]),
                line_directions=np.array([slope]),
                translation_x=dut.translation_x,
                translation_y=dut.translation_y,
                translation_z=dut.translation_z,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)

            # The beam angle goes along the z axis (0.0, 0.0, 1.0).
            initial_state_mean[index] = [intersections[0, 0], intersections[0, 1], intersections[0, 2], 0.0, 0.0, 1.0]
            initial_state_covariance[index, 0, 0] = 0.0
            initial_state_covariance[index, 1, 1] = 0.0
            # initial_state_covariance[index, 0, 0] = np.square(first_dut_pixel_size[0])
            # initial_state_covariance[index, 1, 1] = np.square(first_dut_pixel_size[1])
        else:  # The first plane has a hit
            # If first plane should be included in track building, take first dut hit as initial value and
            # its corresponding cluster position error as the error on the measurement.
            # The beam angle goes along the z axis (0.0, 0.0, 1.0).
            initial_state_mean[index] = [actual_hits[z_sorted_dut_indices[0], 0], actual_hits[z_sorted_dut_indices[0], 1], actual_hits[z_sorted_dut_indices[0], 2], 0.0, 0.0, 1.0]
            initial_state_covariance[index, 0, 0] = np.square(actual_hits[z_sorted_dut_indices[0], 3])  # x_err
            initial_state_covariance[index, 1, 1] = np.square(actual_hits[z_sorted_dut_indices[0], 4])  # y_err
            # FIXME: include meaningful error on z-position and investigate behavior of z_err
            # initial_state_covariance[index, 2, 2] = np.square(actual_hits[z_sorted_dut_indices[0], 5])

    # run kalman filter
    track_estimates_chunk, x_err, y_err = _kalman_fit_3d(
        dut_planes=all_dut_planes,
        z_sorted_dut_indices=z_sorted_dut_indices,
        hits=track_hits[:, :, 0:3],
        thetas=thetas,
        select_fit_duts=select_fit_duts,
        transition_offsets=transition_offsets,
        observation_matrices=observation_matrices,
        observation_covariances=observation_covariances,
        observation_offsets=observation_offsets,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance)

    # remove scatter planes from data
    x_err = x_err[:, :len(telescope)]
    y_err = y_err[:, :len(telescope)]

    offsets = track_estimates_chunk[:, :len(telescope), :3]

    slopes = track_estimates_chunk[:, :len(telescope), 3:]
    # force the 3rd component (z) to be positive
    # and normalize to 1
    slopes[slopes[:, :, 2] < 0.0] = -slopes[slopes[:, :, 2] < 0.0]
    slopes_mag = np.sqrt(np.einsum('ijk,ijk->ij', slopes, slopes))
    slopes /= slopes_mag[:, :, np.newaxis]

    return offsets, slopes, x_err, y_err


def _kalman_fit_3d(dut_planes, z_sorted_dut_indices, hits, thetas, select_fit_duts, transition_offsets, observation_matrices, observation_covariances, observation_offsets, initial_state_mean, initial_state_covariance):
    '''
    This function calls the Kalman Filter. It returns track by track the smoothed state vector which contains in the first two components
    the smoothed hit positions and in the last two components the respective slopes. Additionally the chi square of the track is calculated
    and returned.

    Parameters
    ----------
    dut_planes : list
        List of DUT parameters (material_budget, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma).
    z_sorted_dut_indices : list
        List of DUT indices in the order reflecting their z position.
    hits : array_like
        Array which contains the x, y and z hit position of each DUT for one track.
    thetas : list
        List of scattering angle root mean squares (RMS).
    select_fit_duts : list
        List of DUTs which should be included in Kalman Filter. DUTs which are not in list
        were treated as missing measurements and will not be included in the Filtering step.
    transition_offset : array_like
        Vector which array_like the offset of each transition.
    observation_matrix : array_like
        Matrix which converts the state vector to the actual measurement vector.
    observation_covariances : array_like
        Matrix which describes the covariance of the measurement.
    observation_offset : array_like
        Vector which describes the offset of each measurement.
    initial_state_mean : array_like
        Vector which describes the starting point of the state vector.
    initial_state_covariance : array_like
        Error on the starting pointin of the state vector.

    Returns
    -------
    smoothed_state_estimates : array_like
        Smoothed state vectors.
    chi2 : uint
        Chi2 of track.
    x_err : array_like
        Error of smoothed hit position in x direction. Calculated from smoothed
        state covariance matrix.
    y_err : array_like
        Error of smoothed hit position in y direction. Calculated from smoothed
        state covariance matrix.
    '''
    kf = kalman.KalmanFilter()
    smoothed_state_estimates, cov = kf.smooth(
        dut_planes=dut_planes,
        z_sorted_dut_indices=z_sorted_dut_indices,
        observations=hits[:, :, 0:3],
        thetas=thetas,
        select_fit_duts=select_fit_duts,
        transition_offsets=transition_offsets,
        observation_matrices=observation_matrices,
        observation_offsets=observation_offsets,
        observation_covariances=observation_covariances,
        initial_state=initial_state_mean,
        initial_state_covariance=initial_state_covariance)

    # calculate the sum of the squared x/y residuals, divided by n hits per track
    # chi2 = np.sum(np.square(np.ma.masked_invalid(hits[:, :, 0:2]) - smoothed_state_estimates[:, :, 0:2]), dtype=np.float64, axis=(1, 2)) / np.count_nonzero(~np.isnan(hits[:, :, 0]), axis=1)

    # rough estimate for error on x and y of smoothed estimate
    x_err = np.sqrt(np.diagonal(cov, axis1=3, axis2=2)[:, :, 0])
    y_err = np.sqrt(np.diagonal(cov, axis1=3, axis2=2)[:, :, 1])

    # Check for invalid values (NaN)
    if np.any(np.isnan(smoothed_state_estimates)):
        logging.warning('Smoothed state estimates contain invalid values (NaNs). Check input of Kalman Filter.')

    return smoothed_state_estimates, x_err, y_err
