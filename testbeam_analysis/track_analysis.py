''' Track finding and fitting functions are listed here.'''
from __future__ import division

import logging
from multiprocessing import Pool, cpu_count
from math import sqrt
import progressbar
import os
from collections import Iterable
import functools

import tables as tb
import numpy as np
from numba import njit
from matplotlib.backends.backend_pdf import PdfPages
from numpy import ma

from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import smc
from testbeam_analysis.tools import analysis_utils
from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.tools import kalman


# TODO: implement n_tracks/n_events
def find_tracks(input_merged_file, input_alignment_file, output_track_candidates_file, use_prealignment, max_events=None, correct_beam_alignment=True, chunk_size=1000000):
    '''Takes first DUT track hit and tries to find matching hits in subsequent DUTs.
    The output is the same array with resorted hits into tracks. A track quality is set to
    be able to cut on good (less scattered) tracks.
    This function is uses numba to increase the speed on the inner loop (_find_tracks_loop()).

    This function can also be called on TrackCandidates arrays. That is usefull if an additional alignment step
    was done and the track finding has to be repeated.

    Parameters
    ----------
    input_merged_file : string
        Input file name of the merged cluster hit table from all DUTs.
    input_alignment_file : string
        File containing the alignment information
    output_track_candidates_file : string
        Output file name for track candidate array
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Finding tracks ===')

    # Get alignment errors from file
    with tb.open_file(input_alignment_file, mode='r') as in_file_h5:
        if use_prealignment:
            logging.info('Use pre-alignment')
            alignment = in_file_h5.root.PreAlignment[:]
            beam_alignment = None
            n_duts = alignment.shape[0]
            # TODO: remove that
            column_sigma = np.zeros(shape=n_duts, dtype=np.double)
            row_sigma = np.zeros(shape=n_duts, dtype=np.double)
            column_sigma[0], row_sigma[0] = 0.0, 0.0  # DUT0 has no correlation error
            for index in range(1, n_duts):
                column_sigma[index] = alignment[index]['column_sigma']
                row_sigma[index] = alignment[index]['row_sigma']
            print "column_sigma", column_sigma
            print "row_sigma", row_sigma
        else:
            # TODO: use track slopes to correct telescope alignment
#             raise
            logging.info('Use alignment')
#             raise tb.exceptions.NoSuchNodeError  # FIXME: sigma is to small after alignment, track finding with tracks instead of correlation needed
            alignment = in_file_h5.root.Alignment[:]
            if correct_beam_alignment:
                beam_alignment = in_file_h5.root.BeamAlignment[:]
            else:
                beam_alignment = None
            n_duts = alignment.shape[0]
            # TODO: remove that
            column_sigma = np.zeros(shape=n_duts, dtype=np.double)
            row_sigma = np.zeros(shape=n_duts, dtype=np.double)
            print "column_sigma", column_sigma
            print "row_sigma", row_sigma

    def work(tracklets_data_chunk):
        ''' Track finding per cpu core '''
        # Prepare hit data for track finding, create temporary arrays for x, y, z position and charge data
        # This is needed to call a numba jitted function, since the number of DUTs is not fixed and thus the data format
        x_local = np.copy(tracklets_data_chunk['x_dut_0'])
        y_local = np.copy(tracklets_data_chunk['y_dut_0'])
        z_local = np.copy(tracklets_data_chunk['z_dut_0'])
        x_err_local = np.copy(tracklets_data_chunk['xerr_dut_0'])
        y_err_local = np.copy(tracklets_data_chunk['yerr_dut_0'])
        z_err_local = np.copy(tracklets_data_chunk['zerr_dut_0'])
        geometry_utils.apply_alignment_to_hits(hits=tracklets_data_chunk, dut_index=0, use_prealignment=use_prealignment, alignment=alignment, beam_alignment=beam_alignment, inverse=False, no_z=False)
        x = tracklets_data_chunk['x_dut_0']
        y = tracklets_data_chunk['y_dut_0']
        z = tracklets_data_chunk['z_dut_0']
        x_err = tracklets_data_chunk['xerr_dut_0']
        y_err = tracklets_data_chunk['yerr_dut_0']
        z_err = tracklets_data_chunk['zerr_dut_0']
        charge = tracklets_data_chunk['charge_dut_0']
        n_hits = tracklets_data_chunk['n_hits_dut_0']
        n_cluster = tracklets_data_chunk['n_cluster_dut_0']
        for dut_index in range(1, n_duts):
            x_local = np.column_stack((x_local, tracklets_data_chunk['x_dut_%d' % (dut_index)]))
            y_local = np.column_stack((y_local, tracklets_data_chunk['y_dut_%d' % (dut_index)]))
            z_local = np.column_stack((z_local, tracklets_data_chunk['z_dut_%d' % (dut_index)]))
            x_err_local = np.column_stack((x_err_local, tracklets_data_chunk['xerr_dut_%d' % (dut_index)]))
            y_err_local = np.column_stack((y_err_local, tracklets_data_chunk['yerr_dut_%d' % (dut_index)]))
            z_err_local = np.column_stack((z_err_local, tracklets_data_chunk['zerr_dut_%d' % (dut_index)]))
            geometry_utils.apply_alignment_to_hits(hits=tracklets_data_chunk, dut_index=dut_index, use_prealignment=use_prealignment, alignment=alignment, beam_alignment=beam_alignment, inverse=False, no_z=False)
            x = np.column_stack((x, tracklets_data_chunk['x_dut_%d' % (dut_index)]))
            y = np.column_stack((y, tracklets_data_chunk['y_dut_%d' % (dut_index)]))
            z = np.column_stack((z, tracklets_data_chunk['z_dut_%d' % (dut_index)]))
            x_err = np.column_stack((x_err, tracklets_data_chunk['xerr_dut_%d' % (dut_index)]))
            y_err = np.column_stack((y_err, tracklets_data_chunk['yerr_dut_%d' % (dut_index)]))
            z_err = np.column_stack((z_err, tracklets_data_chunk['zerr_dut_%d' % (dut_index)]))
            charge = np.column_stack((charge, tracklets_data_chunk['charge_dut_%d' % (dut_index)]))
            n_hits = np.column_stack((n_hits, tracklets_data_chunk['n_hits_dut_%d' % (dut_index)]))
            n_cluster = np.column_stack((n_cluster, tracklets_data_chunk['n_cluster_dut_%d' % (dut_index)]))

        event_number = tracklets_data_chunk['event_number']
        hit_flag = np.zeros_like(tracklets_data_chunk['hit_flag'])
        quality_flag = np.zeros_like(tracklets_data_chunk['quality_flag'])
        n_tracks = tracklets_data_chunk['n_tracks']

        # Perform the track finding with jitted loop
        _find_tracks_loop(event_number=event_number,
                          x_local=x_local,
                          y_local=y_local,
                          z_local=z_local,
                          x_err_local=x_err_local,
                          y_err_local=y_err_local,
                          z_err_local=z_err_local,
                          x=x,
                          y=y,
                          z=z,
                          x_err=x_err,
                          y_err=y_err,
                          z_err=z_err,
                          charge=charge,
                          n_hits=n_hits,
                          n_cluster=n_cluster,
                          hit_flag=hit_flag,
                          n_tracks=n_tracks,
                          column_sigma=column_sigma,
                          row_sigma=row_sigma)

# TODO: also use local coordinates in find_tracks_loop to avoid transformation to local coordinate system
#         for dut_index in range(0, n_duts):
#             geometry_utils.apply_alignment_to_hits(hits=combined, dut_index=dut_index, use_prealignment=use_prealignment, alignment=alignment, inverse=True, no_z=False)

        # Merge result data from arrays into one recarray
        combined = np.column_stack((event_number, x_local, y_local, z_local, charge, n_hits, n_cluster, hit_flag, quality_flag, n_tracks, x_err_local, y_err_local, z_err_local))
        return np.core.records.fromarrays(combined.transpose(), dtype=tracklets_data_chunk.dtype)

    smc.SMC(table_file_in=input_tracklets_file,
            file_out=output_track_candidates_file,
            func=work,
            node_desc={'name':'TrackCandidates',
                        'title':'Track candidates'},
            # Apply track finding on tracklets or track candidates
            table=['Tracklets', 'TrackCandidates'],
            align_at='event_number',
            chunk_size=chunk_size)


def fit_tracks(input_track_candidates_file, input_alignment_file, output_tracks_file, max_events=None, fit_duts=None, selection_hit_duts=None, selection_fit_duts=None, exclude_dut_hit=True, quality_sigma=5.0, selection_track_quality=None, pixel_size=None, n_pixels=None, beam_energy=None, material_budget=None, add_scattering_plane=False, max_tracks_per_event=None, use_prealignment=False, use_correlated=False, min_track_distance=None, keep_data=False, method='Fit', full_track_info=False, mode='w', chunk_size=1000000):
    '''Fits either a line through selected DUT hits for selected DUTs (method=Fit) or uses a Kalman Filter to build tracks (method=Kalman).
    The selection criterion for the track candidates to fit is the track quality and the maximum number of hits per event.
    The fit is done for specified DUTs only (fit_duts). This DUT is then not included in the fit (include_duts).
    Bad DUTs can be always ignored in the fit (ignore_duts).

    Parameters
    ----------
    input_track_candidates_file : string
        Filename of the input track candidate file.
    input_alignment_file : string
        Filename of the input alignment file.
    output_tracks_file : string
        Filename of the output tracks file.
    max_events : uint
        Radomly select max_events. If None, fit and store all events.
    fit_duts : iterable
        Specify DUTs for which tracks will be fitted. A track table will be generated for each fit DUT.
        If None, all existing DUTs are used.
    selection_hit_duts : iterable, or iterable of iterable
        The duts that are required to have a hit with the given track quality. Otherwise the track is omitted.
        If None: require all DUTs to have a hit, but if exclude_dut_hit = True do not use actual fit_dut.
        If iterable: use selection for all devices, e.g.: Require hit in DUT 0, and 3: selection_hit_duts = (0, 3).
        If iterable of iterable: define dut with hits for all devices seperately,
        e.g. for 3 devices: selection_hit_duts = ((1, 2), (0, 1, 2), (0, 1))
    selection_fit_duts : iterable, or iterable of iterable or None
        If None, selection_hit_duts are used for fitting.
        Cannot define DUTs that are not in selection_hit_duts,
        e.g. require hits in DUT0, DUT1, DUT3, DUT4 but do not use DUT3 in the fit:
        selection_hit_duts = (0, 1, 3, 4)
        selection_fit_duts = (0, 1, 4)
    exclude_dut_hit : bool
        Set to not require a hit in the actual fit DUT (e.g.: for unconstrained residuals).
        False: Just use all devices as specified in selection_hit_duts.
        True: Do not take the DUT hit for track selection / fitting, even if specified in selection_hit_duts.
    max_tracks_per_event : uint
        Select only events with number of tracks <= max_tracks_per_event. If None, take any event.
    use_prealignment : bool
        If True, use pre-alignment, even if alignment data is availale.
    selection_track_quality : iterable
        TODO: change docstring
        One number valid for all DUTs or an iterable with a number for each DUT.
        0: All tracks with hits in DUT and references are taken
        1: The track hits in DUT and reference are within 2-sigma of the correlation
        2: The track hits in DUT and reference are within 1-sigma of the correlation
        Track quality is saved for each DUT as boolean in binary representation. 8-bit integer for each 'quality stage', one digit per DUT.
        E.g. 0000 0101 assigns hits in DUT0 and DUT2 to the corresponding track quality.
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension (column/row),
        e.g. for two FE-I4 DUTs [(250, 50), (250, 50)]. Only needed for Kalman Filter.
    n_pixels : iterable of tuples
        One tuple per DUT describing the number of pixels in column, row direction
        e.g. for 2 DUTs: n_pixels = [(80, 336), (80, 336)]. Only needed for Kalman Filter.
    beam_energy : uint
        Energy of electron beam in MeV. Only needed for Kalman Filter.
    material_budget : iterable
        Material budget of all DUTs. The material budget is defined as the thickness (sensor + other scattering materials)
        devided by the radiation length (Silicon: 93700 um, M26(50 um Si + 50 um Kapton): 125390 um). Only needed for Kalman Filter.
    add_scattering_plane : dict
        Specifies additional scattering planes in case of additional DUTs which are not used.
        The dictionary must contain the following keys:
            z_scatter: z position of scattering plane in um
            material_budget_scatter: material budget of scattering plane
            alignment_scatter: list which contains alpha, beta and gamma angles of scattering plane.
                               If None, no rotation will be considered.
        In case of multiple scattering planes, each value of a key is a list, with items corresponding to each scattering plane.
        If add_scattering_plane is False, no scattering plane will be added.
    use_correlated : bool
        Use only events that are correlated. Can (at the moment) be applied only if function uses corrected Tracklets file.
    keep_data : bool
        Keep all track candidates in data and add track info only to fitted tracks. Needed for purity calculations!
    method : string
        Available methods are 'Kalman' which uses a Kalman Filter for track building or 'Fit' which uses a simple
        straight line fit for track building.
    full_track_info : bool
        TODO: implement also for straight line fit
        If True, predicted state vector of all DUTs is appended to track table in order to get full information on track.
        If False only state vector of DUT for which the track is fitted is appended to track table.
        This option is only possible, if the Kalman Filter method is choosen.
    min_track_distance : iterable
        A minimum distance all track intersection at the DUT have to be apart, otherwise these tracks are removed.
        This is needed to get a correct efficiency number, since assigning the same cluster to several tracks is error prone
        and will not be implemented.
        If None, no limit on the track distance is applied.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''

    logging.info('=== Fitting tracks (Method: %s) ===' % method)

    if method != "Fit" and method != "Kalman":
        raise ValueError('Method "%s" not recognized!' % method)
    if method == "Kalman" and not pixel_size:
        raise ValueError('Kalman filter requires pixel size for covariance matrix!')
    if method != "Kalman" and full_track_info is True:
        raise ValueError('Full track information option only possible for Kalman Filter method.')

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        if use_prealignment:
            logging.info('Use pre-alignment data')
            prealignment = in_file_h5.root.PreAlignment[:]
            n_duts = prealignment.shape[0]
            z_positions = prealignment['z']
        else:
            logging.info('Use alignment data')
            alignment = in_file_h5.root.Alignment[:]
            n_duts = alignment.shape[0]
            z_positions = alignment['translation_z']

    if fit_duts is None:
        fit_duts = range(n_duts)  # standard setting: fit tracks for all DUTs
    elif not isinstance(fit_duts, Iterable):
        fit_duts = [fit_duts]
    # Check for duplicates
    if len(fit_duts) != len(set(fit_duts)):
        raise ValueError("found douplicate in fit_duts")
    # Check if any iterable in iterable
    if any(map(lambda val: isinstance(val, Iterable), fit_duts)):
        raise ValueError("item in fit_duts is iterable")

    # Create track, hit selection
    if selection_fit_duts is None:  # If None: use all DUTs
        selection_fit_duts = range(n_duts)
#         # copy each item
#         for hit_duts in selection_hit_duts:
#             selection_fit_duts.append(hit_duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(selection_fit_duts, Iterable):
        raise ValueError("selection_fit_duts is no iterable")
    elif not selection_fit_duts:  # empty iterable
        raise ValueError("selection_fit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), selection_fit_duts)):
        selection_fit_duts = [selection_fit_duts[:] for _ in fit_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), selection_fit_duts)):
        raise ValueError("not all items in selection_fit_duts are iterable")
    # Finally check length of all arrays
    if len(selection_fit_duts) != len(fit_duts):  # empty iterable
        raise ValueError("selection_fit_duts has the wrong length")
    for index, fit_dut in enumerate(selection_fit_duts):
        if len(fit_dut) < 2:  # check the length of the items
            raise ValueError("item in selection_fit_duts has length < 2")
#         if set(fit_dut) - set(selection_hit_duts[index]):  # fit DUTs are required to have a hit
#             raise ValueError("DUT in selection_fit_duts is not in selection_hit_duts")

    # Create track, hit selection
    if selection_hit_duts is None:  # If None, require no hit
#         selection_hit_duts = range(n_duts)
        selection_hit_duts = []
    # Check iterable and length
    if not isinstance(selection_hit_duts, Iterable):
        raise ValueError("selection_hit_duts is no iterable")
#     elif not selection_hit_duts:  # empty iterable
#         raise ValueError("selection_hit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), selection_hit_duts)):
        selection_hit_duts = [selection_hit_duts[:] for _ in fit_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), selection_hit_duts)):
        raise ValueError("not all items in selection_hit_duts are iterable")
    # Finally check length of all arrays
    if len(selection_hit_duts) != len(fit_duts):  # empty iterable
        raise ValueError("selection_hit_duts has the wrong length")
#     for hit_dut in selection_hit_duts:
#         if len(hit_dut) < 2:  # check the length of the items
#             raise ValueError("item in selection_hit_duts has length < 2")


#     # Create track, hit selection
#     if selection_track_quality is None:  # If None, require no quality hit
# #         selection_hit_duts = range(n_duts)
#         selection_track_quality = []
#     # Check iterable and length
#     if not isinstance(selection_track_quality, Iterable):
#         raise ValueError("selection_track_quality is no iterable")
# #     elif not selection_track_quality:  # empty iterable
# #         raise ValueError("selection_track_quality has no items")
#     # Check if only non-iterable in iterable
#     if all(map(lambda val: not isinstance(val, Iterable), selection_track_quality)):
#         selection_track_quality = [selection_track_quality[:] for _ in fit_duts]
#     # Check if only iterable in iterable
#     if not all(map(lambda val: isinstance(val, Iterable), selection_track_quality)):
#         raise ValueError("not all items in selection_track_quality are iterable")
#     # Finally check length of all arrays
#     if len(selection_track_quality) != len(fit_duts):  # empty iterable
#         raise ValueError("selection_track_quality has the wrong length")


#     # Create track, hit selection
#     if not isinstance(selection_track_quality, Iterable):  # all items the same, special case for selection_track_quality
#         selection_track_quality = [[selection_track_quality] * len(hit_duts) for hit_duts in selection_hit_duts]  # every hit DUTs require a track quality value
#     # Check iterable and length
#     if not isinstance(selection_track_quality, Iterable):
#         raise ValueError("selection_track_quality is no iterable")
#     elif not selection_track_quality:  # empty iterable
#         raise ValueError("selection_track_quality has no items")
#     # Check if only non-iterable in iterable
#     if all(map(lambda val: not isinstance(val, Iterable), selection_track_quality)):
#         selection_track_quality = [selection_track_quality for _ in fit_duts]
#     # Check if only iterable in iterable
#     if not all(map(lambda val: isinstance(val, Iterable), selection_track_quality)):
#         raise ValueError("not all items in selection_track_quality are iterable")
#     # Finally check length of all arrays
#     if len(selection_track_quality) != len(fit_duts):  # empty iterable
#         raise ValueError("selection_track_quality has the wrong length")
#     for index, track_quality in enumerate(selection_track_quality):
#         if len(track_quality) != len(selection_hit_duts[index]):  # check the length of each items
#             raise ValueError("item in selection_track_quality and selection_hit_duts does not have the same length")

    # Create min hit distance
    if not isinstance(min_track_distance, Iterable):
        min_track_distance = [min_track_distance] * len(fit_duts)
    # Check iterable and length
    if not isinstance(min_track_distance, Iterable):
        raise ValueError("min_track_distance is no iterable")
    elif not min_track_distance:  # empty iterable
        raise ValueError("min_track_distance has no items")
    # Finally check length of all arrays
    if len(min_track_distance) != len(fit_duts):  # empty iterable
        raise ValueError("min_track_distance has the wrong length")

    # Special mode: use all DUTs in the fit and the selections are all the same --> the data does only have to be fitted once
    if not exclude_dut_hit and all(set(x) == set(selection_hit_duts[0]) for x in selection_hit_duts) and all(set(x) == set(selection_fit_duts[0]) for x in selection_fit_duts):# and all(list(x) == list(selection_track_quality[0]) for x in selection_track_quality):
        same_tracks_for_all_duts = True
        logging.info('All fit DUTs uses the same parameters, generate single output table')
    else:
        same_tracks_for_all_duts = False

    def create_results_array(slopes, offsets, chi2s, n_duts, good_track_selection, track_candidates_chunk, track_estimates_chunk_full=None):
        # Define description
        description = [('event_number', np.int64)]
        for index in range(n_duts):
            description.append(('x_dut_%d' % index, np.double))
        for index in range(n_duts):
            description.append(('y_dut_%d' % index, np.double))
        for index in range(n_duts):
            description.append(('z_dut_%d' % index, np.double))
        for index in range(n_duts):
            description.append(('charge_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('n_hits_dut_%d' % index, np.uint32))
        for index in range(n_duts):
            description.append(('n_cluster_dut_%d' % index, np.uint32))
        for dimension in range(3):
            description.append(('offset_%d' % dimension, np.double))
        for dimension in range(3):
            description.append(('slope_%d' % dimension, np.double))
        if full_track_info is True and method == "Kalman":
            for index in range(n_duts):
                description.append(('predicted_x_dut_%d' % index, np.double))
            for index in range(n_duts):
                description.append(('predicted_y_dut_%d' % index, np.double))
            for index in range(n_duts):
                description.append(('predicted_z_dut_%d' % index, np.double))
            for index in range(n_duts):
                description.append(('slope_x_dut_%d' % index, np.double))
            for index in range(n_duts):
                description.append(('slope_y_dut_%d' % index, np.double))
            for index in range(n_duts):
                description.append(('slope_z_dut_%d' % index, np.double))
        description.extend([('track_chi2', np.float), ('hit_flag', np.uint16), ('quality_flag', np.uint16), ('n_tracks', np.uint32)])
        for index in range(n_duts):
            description.append(('xerr_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('yerr_dut_%d' % index, np.float))
        for index in range(n_duts):
            description.append(('zerr_dut_%d' % index, np.float))

        # Select only fitted tracks (keep_data is False) or keep all track candidates (keep_data is True)
        if not keep_data:
            track_candidates_chunk = track_candidates_chunk[good_track_selection]

        tracks_array = np.empty((track_candidates_chunk.shape[0],), dtype=description)

        tracks_array['hit_flag'] = track_candidates_chunk['hit_flag']
        tracks_array['quality_flag'] = track_candidates_chunk['quality_flag']
        tracks_array['event_number'] = track_candidates_chunk['event_number']
        tracks_array['n_tracks'] = track_candidates_chunk['n_tracks']
        for index in range(n_duts):
            tracks_array['x_dut_%d' % index] = track_candidates_chunk['x_dut_%d' % index]
            tracks_array['y_dut_%d' % index] = track_candidates_chunk['y_dut_%d' % index]
            tracks_array['z_dut_%d' % index] = track_candidates_chunk['z_dut_%d' % index]
            tracks_array['xerr_dut_%d' % index] = track_candidates_chunk['xerr_dut_%d' % index]
            tracks_array['yerr_dut_%d' % index] = track_candidates_chunk['yerr_dut_%d' % index]
            tracks_array['zerr_dut_%d' % index] = track_candidates_chunk['zerr_dut_%d' % index]
            tracks_array['charge_dut_%d' % index] = track_candidates_chunk['charge_dut_%d' % index]
            tracks_array['n_hits_dut_%d' % index] = track_candidates_chunk['n_hits_dut_%d' % index]
            tracks_array['n_cluster_dut_%d' % index] = track_candidates_chunk['n_cluster_dut_%d' % index]

        if keep_data:
            for dimension in range(3):
                tracks_array['offset_%d' % dimension][good_track_selection] = offsets[:, dimension]
                tracks_array['slope_%d' % dimension][good_track_selection] = slopes[:, dimension]
                tracks_array['offset_%d' % dimension][~good_track_selection] = np.nan
                tracks_array['slope_%d' % dimension][~good_track_selection] = np.nan
            if full_track_info is True and method == "Kalman":
                for index in range(n_duts):
                    tracks_array['predicted_x_dut_%d' % index][good_track_selection] = track_estimates_chunk_full[:, index, 0]
                    tracks_array['predicted_y_dut_%d' % index][good_track_selection] = track_estimates_chunk_full[:, index, 1]
                    tracks_array['predicted_z_dut_%d' % index][good_track_selection] = track_estimates_chunk_full[:, index, 2]
                    tracks_array['slope_x_dut_%d' % index][good_track_selection] = track_estimates_chunk_full[:, index, 3]
                    tracks_array['slope_y_dut_%d' % index][good_track_selection] = track_estimates_chunk_full[:, index, 4]
                    tracks_array['slope_z_dut_%d' % index][good_track_selection] = track_estimates_chunk_full[:, index, 5]
                    tracks_array['predicted_x_dut_%d' % index][~good_track_selection] = np.nan
                    tracks_array['predicted_y_dut_%d' % index][~good_track_selection] = np.nan
                    tracks_array['predicted_z_dut_%d' % index][~good_track_selection] = np.nan
                    tracks_array['slope_x_dut_%d' % index][~good_track_selection] = np.nan
                    tracks_array['slope_y_dut_%d' % index][~good_track_selection] = np.nan
                    tracks_array['slope_z_dut_%d' % index][~good_track_selection] = np.nan
            tracks_array['track_chi2'][good_track_selection] = chi2s
            tracks_array['track_chi2'][~good_track_selection] = np.nan
        else:
            for dimension in range(3):
                tracks_array['offset_%d' % dimension] = offsets[:, dimension]
                tracks_array['slope_%d' % dimension] = slopes[:, dimension]
            if full_track_info is True and method == "Kalman":
                for index in range(n_duts):
                    tracks_array['predicted_x_dut_%d' % index] = track_estimates_chunk_full[:, index, 0]
                    tracks_array['predicted_y_dut_%d' % index] = track_estimates_chunk_full[:, index, 1]
                    tracks_array['predicted_z_dut_%d' % index] = track_estimates_chunk_full[:, index, 2]
                    tracks_array['slope_x_dut_%d' % index] = track_estimates_chunk_full[:, index, 3]
                    tracks_array['slope_y_dut_%d' % index] = track_estimates_chunk_full[:, index, 4]
                    tracks_array['slope_z_dut_%d' % index] = track_estimates_chunk_full[:, index, 5]
            tracks_array['track_chi2'] = chi2s

        return tracks_array

    def store_track_data(fit_dut, fit_dut_index, min_track_distance):
        # reset quality flag
        track_candidates_chunk["quality_flag"] = 0
        for dut_index in range(n_duts):
            if use_prealignment:  # Pre-alignment does not set any plane rotations thus plane normal = (0, 0, 1) and position = (0, 0, z)
                dut_position = np.array([0.0, 0.0, prealignment['z'][dut_index]])
                dut_plane_normal = np.array([0.0, 0.0, 1.0])
            else:  # Deduce plane orientation in 3D for track extrapolation; not needed if rotation info is not available (e.g. only prealigned data)
                dut_position = np.array([alignment[dut_index]['translation_x'], alignment[dut_index]['translation_y'], alignment[dut_index]['translation_z']])
                rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[dut_index]['alpha'],
                                                                 beta=alignment[dut_index]['beta'],
                                                                 gamma=alignment[dut_index]['gamma'])
                basis_global = rotation_matrix.T.dot(np.eye(3))
                dut_plane_normal = basis_global[2]
                if dut_plane_normal[2] < 0:
                    dut_plane_normal = -dut_plane_normal

            # TODO: use plane at z=0
    #         dut_position = np.array([0.0, 0.0, 0.0])
    #         dut_plane_normal = np.array([0.0, 0.0, 1.0])

            # Set the offset to the track intersection with the tilted plane
            intersections = geometry_utils.get_line_intersections_with_plane(line_origins=offsets,
                                                                             line_directions=slopes,
                                                                             position_plane=dut_position,
                                                                             normal_plane=dut_plane_normal)

            if use_prealignment:
                intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersections[:,0], intersections[:,1], intersections[:,2],
                                                                                                                  dut_index=dut_index,
                                                                                                                  prealignment=prealignment,
                                                                                                                  inverse=True)
            else:
                intersection_x_local, intersection_y_local, intersection_z_local = geometry_utils.apply_alignment(intersections[:,0], intersections[:,1], intersections[:,2],
                                                                                                                  dut_index=dut_index,
                                                                                                                  alignment=alignment,
                                                                                                                  inverse=True)
            if not np.allclose(intersection_z_local, 0.0):
                raise RuntimeError("Transformation into local coordinate system gives z != 0")
            dut_quality_flag_sel = ((np.abs(intersection_x_local - track_candidates_chunk['x_dut_%s' % dut_index][good_track_selection]) <= quality_sigma * track_candidates_chunk['xerr_dut_%s' % dut_index][good_track_selection])
                                    & (np.abs(intersection_y_local - track_candidates_chunk['y_dut_%s' % dut_index][good_track_selection]) <= quality_sigma * track_candidates_chunk['yerr_dut_%s' % dut_index][good_track_selection]))
            dut_quality_flag = track_candidates_chunk["quality_flag"][good_track_selection]
            dut_quality_flag[dut_quality_flag_sel] |= (1 << dut_index)
            track_candidates_chunk["quality_flag"][good_track_selection] = dut_quality_flag

            if dut_index == fit_dut:
                # use offsets at the location of the fit DUT
                dut_offsets = intersections
                # select only quality tracks

#             print "good tracks selection before quality cut", np.count_nonzero(good_track_selection)
#             good_track_selection &= (track_candidates_chunk["quality_flag"] & dut_quality_selection == dut_quality_selection)
#             print "good tracks selection after quality cut", np.count_nonzero(good_track_selection)

        tracks_array = create_results_array(slopes=slopes,
                                            offsets=dut_offsets,
                                            chi2s=chi2s,
                                            n_duts=n_duts,
                                            good_track_selection=good_track_selection,
                                            track_candidates_chunk=track_candidates_chunk,
                                            track_estimates_chunk_full=None)

        # Remove tracks that are too close when extrapolated to the actual DUT
        # All merged tracks have quality_flag = 0
        actual_min_track_distance = min_track_distance[fit_dut_index]
        if actual_min_track_distance is not None and actual_min_track_distance > 0:
            _find_merged_tracks(tracks_array=tracks_array,
                                min_tracks_distancce=actual_min_track_distance)
#             selection = tracks_array['n_tracks'] > 0
#             logging.info('Removed %d merged tracks (%1.1f%%)', np.count_nonzero(~selection), float(np.count_nonzero(~selection)) / selection.shape[0] * 100.0)
#             tracks_array = tracks_array[selection]

        try:  # Check if table exists already, than append data
            tracklets_table = out_file_h5.get_node('/Tracks_DUT_%d' % fit_dut)
        except tb.NoSuchNodeError:  # Table does not exist, thus create new
            tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Tracks_DUT_%d' % fit_dut, description=tracks_array.dtype, title='Tracks fitted for DUT_%d' % fit_dut, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

        tracklets_table.append(tracks_array)
        tracklets_table.flush()

        # Plot chi2 distribution
        # TODO: make this external function
#         plot_utils.plot_track_chi2(chi2s=chi2s, fit_dut=fit_dut, output_pdf=output_pdf)

    def store_track_data_kalman(fit_dut, min_track_distance):  # Set the offset to the track intersection with the tilted plane and store the data
        # reset quality flag
        track_candidates_chunk["quality_flag"] = 0
        if use_prealignment:  # Pre-alignment does not set any plane rotations thus plane normal = (0, 0, 1) and position = (0, 0, z)
            dut_position = np.array([0.0, 0.0, prealignment['z'][fit_dut]])
            dut_plane_normal = np.array([0.0, 0.0, 1.0])
        else:  # Deduce plane orientation in 3D for track extrapolation; not needed if rotation info is not available (e.g. only prealigned data)
            dut_position = np.array([alignment[fit_dut]['translation_x'], alignment[fit_dut]['translation_y'], alignment[fit_dut]['translation_z']])
            rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[fit_dut]['alpha'],
                                                             beta=alignment[fit_dut]['beta'],
                                                             gamma=alignment[fit_dut]['gamma'])
            basis_global = rotation_matrix.T.dot(np.eye(3))  # TODO: why transposed?
            dut_plane_normal = basis_global[2]

        # FIXME: calculate real slope in z direction
        slopes = np.column_stack((track_estimates_chunk[:, fit_dut, 2],
                                  track_estimates_chunk[:, fit_dut, 3],
                                  np.ones((track_estimates_chunk.shape[0],)).reshape(track_estimates_chunk.shape[0], 1)))

        # z position of each track estimate
        z_position = geometry_utils.get_line_intersections_with_plane(line_origins=np.column_stack((track_estimates_chunk[:, fit_dut, 0],
                                                                                                    track_estimates_chunk[:, fit_dut, 1],
                                                                                                    np.ones(track_estimates_chunk[:, fit_dut, 0].shape))),
                                                                      line_directions=np.column_stack((np.zeros((track_estimates_chunk[:, fit_dut, 0].shape)),
                                                                                                       np.zeros(track_estimates_chunk[:, fit_dut, 0].shape),
                                                                                                       np.ones(track_estimates_chunk[:, fit_dut, 0].shape))),
                                                                      position_plane=dut_position,
                                                                      normal_plane=dut_plane_normal)[:, -1]

        offsets = np.column_stack((track_estimates_chunk[:, fit_dut, 0],
                                   track_estimates_chunk[:, fit_dut, 1],
                                   z_position))
        # do not need to calculate intersection with plane, since track parameters are estimated at the respective plane in kalman filter.
        # This is different than for straight line fit, where intersection calculation is needed.
        actual_offsets = offsets

        if full_track_info is True and method == "Kalman":
            # array to store x,y,z position and respective slopes of other DUTs
            track_estimates_chunk_full = np.full(shape=(track_estimates_chunk.shape[0], n_duts, 6), fill_value=np.nan)
            for dut_index in range(n_duts):
                if dut_index == fit_dut:  # do not need to transform data of actual fit dut, this is already done,
                    continue
                if use_prealignment:  # Pre-alignment does not set any plane rotations thus plane normal = (0, 0, 1) and position = (0, 0, z)
                    dut_position = np.array([0., 0., prealignment['z'][dut_index]])
                    dut_plane_normal = np.array([0., 0., 1.])
                else:  # Deduce plane orientation in 3D for track extrapolation; not needed if rotation info is not available (e.g. only prealigned data)
                    dut_position = np.array([alignment[dut_index]['translation_x'], alignment[dut_index]['translation_y'], alignment[dut_index]['translation_z']])
                    rotation_matrix = geometry_utils.rotation_matrix(alpha=alignment[dut_index]['alpha'],
                                                                     beta=alignment[dut_index]['beta'],
                                                                     gamma=alignment[dut_index]['gamma'])
                    basis_global = rotation_matrix.T.dot(np.eye(3))  # TODO: why transposed?
                    dut_plane_normal = basis_global[2]

                # FIXME: calculate real slope in z direction
                slopes_full = np.column_stack((track_estimates_chunk[:, dut_index, 2],
                                               track_estimates_chunk[:, dut_index, 3],
                                               np.ones((track_estimates_chunk.shape[0],)).reshape(track_estimates_chunk.shape[0], 1)))

                # z position of each track estimate
                z_position = geometry_utils.get_line_intersections_with_plane(line_origins=np.column_stack((track_estimates_chunk[:, dut_index, 0],
                                                                                                            track_estimates_chunk[:, dut_index, 1],
                                                                                                            np.ones(track_estimates_chunk[:, dut_index, 0].shape))),
                                                                              line_directions=np.column_stack((np.zeros((track_estimates_chunk[:, dut_index, 0].shape)),
                                                                                                               np.zeros(track_estimates_chunk[:, dut_index, 0].shape),
                                                                                                               np.ones(track_estimates_chunk[:, dut_index, 0].shape))),
                                                                              position_plane=dut_position,
                                                                              normal_plane=dut_plane_normal)[:, -1]

                offsets_full = np.column_stack((track_estimates_chunk[:, dut_index, 0],
                                                track_estimates_chunk[:, dut_index, 1],
                                                z_position))

                # do not need to calculate intersection with plane, since track parameters are estimated at the respective plane in kalman filter.
                # This is different than for straight line fit, where intersection calculation is needed.
                actual_offsets_full = offsets_full

                track_estimates_chunk_full[:, dut_index] = np.column_stack((actual_offsets_full[:, 0],
                                                                           actual_offsets_full[:, 1],
                                                                           actual_offsets_full[:, 2],
                                                                           slopes_full[:, 0],
                                                                           slopes_full[:, 1],
                                                                           slopes_full[:, 2]))

            tracks_array = create_results_array(slopes, actual_offsets, chi2s, n_duts, good_track_selection, track_candidates_chunk, track_estimates_chunk_full)
        else:
            tracks_array = create_results_array(slopes, actual_offsets, chi2s, n_duts, good_track_selection, track_candidates_chunk)

        try:  # Check if table exists already, than append data
            tracklets_table = out_file_h5.get_node('/Kalman_Tracks_DUT_%d' % fit_dut)
        except tb.NoSuchNodeError:  # Table does not exist, thus create new
            tracklets_table = out_file_h5.create_table(out_file_h5.root, name='Kalman_Tracks_DUT_%d' % fit_dut, description=np.zeros((1,), dtype=tracks_array.dtype).dtype, title='Tracks fitted for DUT_%d_with_Kalman_Filter' % fit_dut, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

        # Remove tracks that are too close when extrapolated to the actual DUT
        # All merged tracks have quality_flag = 0
        actual_min_track_distance = min_track_distance[fit_dut]
        if actual_min_track_distance is not None and actual_min_track_distance > 0:
            _find_merged_tracks(tracks_array=tracks_array,
                                min_tracks_distancce=actual_min_track_distance)
#             selection = tracks_array['n_tracks'] > 0
#             logging.info('Removed %d merged tracks (%1.1f%%)', np.count_nonzero(~selection), float(np.count_nonzero(~selection)) / selection.shape[0] * 100.0)
#             tracks_array = tracks_array[selection]

        tracklets_table.append(tracks_array)
        tracklets_table.flush()

        # Plot chi2 distribution
#         plot_utils.plot_track_chi2(chi2s=chi2s, fit_dut=fit_dut, output_pdf=output_pdf)

    def select_data(dut_index):  # Select track by and DUT hits to use

        dut_hit_selection = 0  # DUTs required to have hits
        dut_fit_selection = 0  # DUTs to be used for the fit
        dut_quality_selection = 0  # DUTs required to have quality hits
        info_str_hit = []  # For info output
        info_str_fit = []  # For info output
        info_str_quality = []  # For info output

        for selected_hit_dut in selection_hit_duts[dut_index]:
            if exclude_dut_hit and selected_hit_dut == fit_duts[dut_index]:
                continue
            dut_hit_selection |= (1 << selected_hit_dut)
            info_str_hit.append('DUT%d' % selected_hit_dut)
        n_slection_duts = np.binary_repr(dut_hit_selection).count("1")
        logging.info('Require hits in %d DUTs for track selection: %s', n_slection_duts, ', '.join(info_str_hit))

        for selected_fit_dut in selection_fit_duts[dut_index]:
            if exclude_dut_hit and selected_fit_dut == fit_duts[dut_index]:
                continue
            dut_fit_selection |= ((1 << selected_fit_dut))
            info_str_fit.append('DUT%d' % selected_fit_dut)
        n_fit_duts = np.binary_repr(dut_fit_selection).count("1")
        logging.info("Use %d DUTs for track fit: %s", n_fit_duts, ', '.join(info_str_fit))

#         track_quality_mask = 0
#         quality_index = 0
#         info_quality = ['no hit'] * n_slection_duts
#         for index, dut in enumerate(selection_hit_duts[dut_index]):
#             if exclude_dut_hit and dut == fit_duts[dut_index]:
#                 continue
#             for quality in range(3):
#                 if quality <= selection_track_quality[dut_index][index]:
#                     track_quality_mask |= ((1 << dut) << quality * 8)
#                     if quality == 0:
#                         info_quality[quality_index] = 'only hit'
#                     else:
#                         info_quality[quality_index] = str(quality)
#             quality_index += 1
#         logging.info("Use track quality for track selection: %s", ', '.join(info_quality))
#
#         return dut_selection, dut_fit_selection, track_quality_mask

#         for selected_quality_dut in selection_track_quality[dut_index]:
#             dut_quality_selection |= ((1 << selected_quality_dut))
#             info_str_quality.append('DUT%d' % selected_fit_dut)
#         n_quality_duts = np.binary_repr(dut_quality_selection).count("1")
#         logging.info("Select quality tracks for %d DUTs: %s", n_quality_duts, ', '.join(info_str_quality))

        return dut_hit_selection, dut_fit_selection#, dut_quality_selection

    pool = Pool()
#     with PdfPages(os.path.splitext(output_tracks_file)[0] + '.pdf', keep_empty=False) as output_pdf:
    with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
#         try:  # If file exists already delete it first
#             os.remove(output_tracks_file)
#         except OSError:
#             pass
        with tb.open_file(output_tracks_file, mode=mode) as out_file_h5:
            for fit_dut_index, actual_fit_dut in enumerate(fit_duts):  # Loop over the DUTs where tracks shall be fitted for
                logging.info('Fit tracks for DUT%d', actual_fit_dut)

                try:  # Check if table exists already, than append data
                    tracklets_table = out_file_h5.remove_node(out_file_h5.root, name='Tracks_DUT_%d' % actual_fit_dut)
                    logging.info('Overwriting existing tracks for DUT%d', actual_fit_dut)
                except tb.NodeError:  # Table does not exist, thus create new
                    pass

                total_n_tracks = in_file_h5.root.TrackCandidates.shape[0]
                total_n_tracks_stored = 0
                total_n_events_stored = 0
#                     initialize = True


#                     dut_hit_selection, dut_fit_selection, track_quality_mask = select_data(fit_dut_index)
#                     dut_hit_selection, dut_fit_selection, dut_quality_selection = select_data(fit_dut_index)
                dut_hit_selection, dut_fit_selection = select_data(fit_dut_index)
                n_fit_duts = np.binary_repr(dut_fit_selection).count("1")
#                     if n_fit_duts < 2:
#                         logging.warning('Insufficient track hits to do the fit (< 2). Omit building tracks for DUT%d', actual_fit_dut)
#                         continue
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
                    # Select tracks based on the dut that are required to have a hit (dut_selection) with a certain quality (track_quality)
                    n_tracks_chunk = track_candidates_chunk.shape[0]

                    print "dut_hit_selection", dut_hit_selection
                    print "dut_fit_selection", dut_fit_selection
#                         print "dut_quality_selection", dut_quality_selection
                    good_track_selection = (track_candidates_chunk['hit_flag'] & dut_hit_selection) == dut_hit_selection
                    # remove tracks that have only single DUT with hits
                    for index, bit in enumerate(np.binary_repr(dut_fit_selection)[::-1]):
                        if bit == "0":
                            continue
                        dut_fit_selection_dut_removed = dut_fit_selection & ~(1 << index)
                        # also include tracks where no hit is required but have hit in fit DUT
                        good_track_selection &= (track_candidates_chunk['hit_flag'] & dut_fit_selection_dut_removed) > 0
                    n_tracks_quality = np.count_nonzero(good_track_selection)
                    removed_n_tracks_quality = n_tracks_chunk - n_tracks_quality
                    # remove merged clusters
                    good_track_selection &= (track_candidates_chunk['n_tracks'] > 0)  # n_tracks < 0 means merged cluster, omit these to allow valid efficiency calculation
                    n_tracks_not_merged = np.count_nonzero(good_track_selection)
                    removed_n_tracks_merged = n_tracks_quality - n_tracks_not_merged

#                     if max_tracks_per_event:  # Option to neglect events with too many hits
#                         good_track_selection &= (track_candidates_chunk['n_tracks'] <= max_tracks_per_event)
#                         n_tracks_max_tracks = np.count_nonzero(good_track_selection)
#                         removed_n_tracks_max_tracks = n_tracks_not_merged - n_tracks_max_tracks
#                         removed_n_tracks = removed_n_tracks_quality + removed_n_tracks_merged + removed_n_tracks_max_tracks
#                         logging.info('Removed %d of %d (%.1f%%) track candidates (quality: %d tracks, merged clusters: %d tracks, max tracks: %d tracks)',
#                                      removed_n_tracks,
#                                      n_tracks_chunk,
#                                      100.0 * removed_n_tracks / n_tracks_chunk,
#                                      removed_n_tracks_quality,
#                                      removed_n_tracks_merged,
#                                      removed_n_tracks_max_tracks)
#                     else:
                    removed_n_tracks = removed_n_tracks_quality + removed_n_tracks_merged
                    logging.info('Removed %d of %d (%.1f%%) track candidates (require hits: %d tracks, merged clusters: %d tracks)',
                                 removed_n_tracks,
                                 n_tracks_chunk,
                                 100.0 * removed_n_tracks / n_tracks_chunk,
                                 removed_n_tracks_quality,
                                 removed_n_tracks_merged)

#                     if use_correlated:  # Reduce track selection to correlated DUTs only
#                         good_track_selection &= (track_candidates_chunk['track_quality'] & (dut_selection << 24) == (dut_selection << 24))
#                         n_tracks_correlated = np.count_nonzero(good_track_selection)
#                         removed_n_tracks_correlated = n_tracks_chunk - n_tracks_correlated - removed_n_tracks
#                         logging.info('Removed %d of %d (%.1f%%) track candidates due to correlated cuts',
#                                      removed_n_tracks_correlated,
#                                      n_tracks_chunk,
#                                      100.0 * removed_n_tracks_correlated / n_tracks_chunk)
#                         good_track_candidates = track_candidates_chunk[good_track_selection]

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

                    print "n_events_chunk", n_events_chunk
                    print "n_tracks_chunk", n_tracks_chunk
                    if max_events:
                        if total_n_tracks == index_chunk:  # last chunk, adding all remaining events
                            select_n_events = max_events - total_n_events_stored
                        elif total_n_events_stored == 0:  # first chunk
                            select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks)))
                        else:
                            # calculate correction of number of selected events
                            correction = (total_n_tracks - index_chunk)/total_n_tracks * 1 / (((total_n_tracks-last_index_chunk)/total_n_tracks)/((max_events-total_n_events_stored_last)/max_events)) \
                                         + (index_chunk)/total_n_tracks * 1 / (((last_index_chunk)/total_n_tracks)/((total_n_events_stored_last)/max_events))
    #                         select_n_events = np.ceil(n_events_chunk * correction)
    #                         # calculate correction of number of selected events
    #                         correction = 1/(((total_n_tracks-last_index_chunk)/total_n_tracks_last)/((max_events-total_n_events_stored_last)/max_events))
                            select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks) * correction))
                            print "correction", correction
                        # do not store more events than in current chunk
                        select_n_events = min(n_events_chunk, select_n_events)
                        # do not store more events than given by max_events
                        select_n_events = min(select_n_events, max_events - total_n_events_stored)
                        np.random.seed(seed=0)
                        selected_events = np.random.choice(unique_events, size=select_n_events, replace=False)
                        store_n_events = selected_events.shape[0]
                        total_n_events_stored += store_n_events
                        print "store_n_events", store_n_events
                        good_track_selection &= np.in1d(track_candidates_chunk["event_number"], selected_events)
                        # TODO: total_n_tracks_stored not used...
                        store_n_tracks = np.count_nonzero(good_track_selection)
                        total_n_tracks_stored += store_n_tracks
#                         track_candidates_chunk = track_candidates_chunk[select_tracks]

                    # Prepare track hits array to be fitted
                    n_good_tracks = np.count_nonzero(good_track_selection)  # Index of tmp track hits array
                    if method == "Fit":
                        track_hits = np.full((n_good_tracks, n_fit_duts, 3), np.nan, dtype=np.double)
                    elif method == "Kalman":
                        track_hits = np.full((n_good_tracks, n_duts, 5), np.inf)

                    print "hit flags", np.unique(track_candidates_chunk['hit_flag'][good_track_selection])#, np.min(track_candidates_chunk['hit_flag'][good_track_selection])
                    print "quality flags", np.unique(track_candidates_chunk['quality_flag'][good_track_selection])#, np.min(track_candidates_chunk['quality_flag'][good_track_selection])
                    fit_index = 0
                    for dut_index in range(n_duts):  # Fill index loop of new array
                        if method == "Fit" and (dut_fit_selection & (1 << dut_index)) == (1 << dut_index):  # True if DUT is used in fit
                            # apply alignment for fitting the tracks
                            if use_prealignment:
                                track_hits[:, fit_index, 0], track_hits[:, fit_index, 1], track_hits[:, fit_index, 2] = geometry_utils.apply_alignment(track_candidates_chunk['x_dut_%s' % dut_index][good_track_selection],
                                                                                                                                                       track_candidates_chunk['y_dut_%s' % dut_index][good_track_selection],
                                                                                                                                                       track_candidates_chunk['z_dut_%s' % dut_index][good_track_selection],
                                                                                                                                                       dut_index=dut_index,
                                                                                                                                                       prealignment=prealignment,
                                                                                                                                                       inverse=False)
                            else:
                                track_hits[:, fit_index, 0], track_hits[:, fit_index, 1], track_hits[:, fit_index, 2] = geometry_utils.apply_alignment(track_candidates_chunk['x_dut_%s' % dut_index][good_track_selection],
                                                                                                                                                       track_candidates_chunk['y_dut_%s' % dut_index][good_track_selection],
                                                                                                                                                       track_candidates_chunk['z_dut_%s' % dut_index][good_track_selection],
                                                                                                                                                       dut_index=dut_index,
                                                                                                                                                       alignment=alignment,
                                                                                                                                                       inverse=False)
                            fit_index += 1
                        elif method == "Kalman":
                            xyz = np.column_stack(np.ma.array((good_track_candidates['x_dut_%s' % dut_index],
                                                               good_track_candidates['y_dut_%s' % dut_index],
                                                               good_track_candidates['z_dut_%s' % dut_index],
                                                               good_track_candidates['xerr_dut_%s' % dut_index],
                                                               good_track_candidates['yerr_dut_%s' % dut_index])))
                            track_hits[:, dut_index, :] = xyz

                    # Split data and fit on all available cores
                    n_slices = cpu_count()
                    slices = np.array_split(track_hits, n_slices)
                    if method == "Fit":
                        results = pool.map(_fit_tracks_loop, slices)
                    elif method == "Kalman":
                        if use_prealignment:
                            # if prealignment is used, planes are not rotated, thus do not have to correct
                            # rotation in kalman filter
                            alignment = None
                        else:
                            alignment = alignment
                        results = pool.map(functools.partial(
                            _function_wrapper_fit_tracks_kalman_loop, pixel_size,
                            n_pixels, dut_fit_selection, z_positions, alignment,
                            beam_energy, material_budget, add_scattering_plane), slices)
#                     del track_hits

                    # Store results
                    if method == "Fit":
                        offsets = np.concatenate([i[0] for i in results])  # Merge offsets from all cores in results
                        slopes = np.concatenate([i[1] for i in results])  # Merge slopes from all cores in results
                        chi2s = np.concatenate([i[2] for i in results])  # Merge chi2 from all cores in results
                        # Store the data
                        if not same_tracks_for_all_duts:  # Check if all DUTs were fitted at once
                            store_track_data(fit_dut=actual_fit_dut,
                                             fit_dut_index=fit_dut_index,
                                             min_track_distance=min_track_distance)
                        else:
                            for index, dut_index in enumerate(fit_duts):
                                store_track_data(fit_dut=dut_index,
                                                 fit_dut_index=index,
                                                 min_track_distance=min_track_distance)

                    elif method == "Kalman":
                        track_estimates_chunk = np.concatenate([i[0] for i in results])  # merge predicted x,y pos from all cores in results
                        chi2s = np.concatenate([i[1] for i in results])  # merge chi2 from all cores in results

                        # Store the data
                        if not same_tracks_for_all_duts:  # Check if all DUTs were fitted at once
                            store_track_data_kalman(actual_fit_dut, min_track_distance)
                        else:
                            for dut_index in fit_duts:
                                store_track_data_kalman(dut_index, min_track_distance)
#                     total_n_tracks += n_good_tracks

                    total_n_events_stored_last = total_n_events_stored
                    total_n_tracks_last = total_n_tracks
                    last_index_chunk = index_chunk
                    progress_bar.update(index_chunk)
#                     progress_bar.update(min(total_n_tracks, max_tracks) if max_tracks is not None else index_chunk)
                progress_bar.finish()
                print "***************"
                print "total_n_tracks_stored", total_n_tracks_stored
                print "total_n_events_stored", total_n_events_stored
                if same_tracks_for_all_duts:  # Stop fit Dut loop since all DUTs were fitted at once
                    break
    pool.close()
    pool.join()


# Helper functions that are not meant to be called directly during analysis
@njit
def _set_dut_hit_flag(dut_x, dut_y, curr_x, curr_y, hit_flag, track_index, dut_index, dut_column_sigma, dut_row_sigma):
    # Set hit flag of actual DUT from actual DUT hit
    if not np.isnan(curr_x):  # curr_x = nan is no hit
        hit_flag[track_index] |= (1 << dut_index)
    else:
        hit_flag[track_index] &= ~(1 << dut_index)


@njit
def _reset_dut_hit_flag(dut_x, dut_y, first_dut_x, first_dut_y, hit_flag, hit_index, dut_index, dut_column_sigma, dut_row_sigma):
    hit_flag[hit_index] &= ~(1 << dut_index)
    if not np.isnan(dut_x):  # x = nan is no hit
        hit_flag[hit_index] |= (1 << dut_index)


@njit
def _get_first_dut_index(x, index):
    ''' Returns the first DUT that has a hit for the track at index '''
    dut_index = 0
    for dut_index in range(x.shape[1]):  # Loop over duts, to get first DUT hit of track
        if not np.isnan(x[index][dut_index]):
            break
    return dut_index


@njit
def _swap_hits(x_local, y_local, z_local, x_err_local, y_err_local, z_err_local, x, y, z, charge, n_hits, x_err, y_err, z_err, track_index, dut_index, hit_index, swap_x_local, swap_y_local, swap_z_local, swap_x_err_local, swap_y_err_local, swap_z_err_local, swap_x, swap_y, swap_z, swap_charge, swap_n_hits, swap_x_err, swap_y_err, swap_z_err):
    #     print 'Swap hits', x[track_index][dut_index], x[hit_index][dut_index]
    tmp_x_local, tmp_y_local, tmp_z_local = x_local[track_index][dut_index], y_local[track_index][dut_index], z_local[track_index][dut_index]
    tmp_x_err_local, tmp_y_err_local, tmp_z_err_local = x_err_local[track_index][dut_index], y_err_local[track_index][dut_index], z_err_local[track_index][dut_index]
    tmp_charge, tmp_n_hits = charge[track_index][dut_index], n_hits[track_index][dut_index]
    tmp_x, tmp_y, tmp_z = x[track_index][dut_index], y[track_index][dut_index], z[track_index][dut_index]
    tmp_x_err, tmp_y_err, tmp_z_err = x_err[track_index][dut_index], y_err[track_index][dut_index], z_err[track_index][dut_index]

    x_local[track_index][dut_index], y_local[track_index][dut_index], z_local[track_index][dut_index] = swap_x_local, swap_y_local, swap_z_local
    x_err_local[track_index][dut_index], y_err_local[track_index][dut_index], z_err_local[track_index][dut_index] = swap_x_err_local, swap_y_err_local, swap_z_err_local
    charge[track_index][dut_index], n_hits[track_index][dut_index] = swap_charge, swap_n_hits
    x[track_index][dut_index], y[track_index][dut_index], z[track_index][dut_index] = swap_x, swap_y, swap_z
    x_err[track_index][dut_index], y_err[track_index][dut_index], z_err[track_index][dut_index] = swap_x_err, swap_y_err, swap_z_err

    x_local[hit_index][dut_index], y_local[hit_index][dut_index], z_local[hit_index][dut_index] = tmp_x_local, tmp_y_local, tmp_z_local
    x_err_local[hit_index][dut_index], y_err_local[hit_index][dut_index], z_err_local[hit_index][dut_index] = tmp_x_err_local, tmp_y_err_local, tmp_z_err_local
    charge[hit_index][dut_index], n_hits[hit_index][dut_index] = tmp_charge, tmp_n_hits
    x[hit_index][dut_index], y[hit_index][dut_index], z[hit_index][dut_index] = tmp_x, tmp_y, tmp_z
    x_err[hit_index][dut_index], y_err[hit_index][dut_index], z_err[hit_index][dut_index] = tmp_x_err, tmp_y_err, tmp_z_err


@njit
def _set_n_tracks(x, y, start_index, stop_index, n_tracks, n_actual_tracks, n_duts):
    if start_index < 0:
        start_index = 0

    # Called if no merged track is found
    for i in range(start_index, stop_index):  # Set number of tracks of previous event
        n_tracks[i] = n_actual_tracks


@njit
def _find_tracks_loop(event_number, x_local, y_local, z_local, x_err_local, y_err_local, z_err_local, x, y, z, x_err, y_err, z_err, charge, n_hits, n_cluster, hit_flag, n_tracks, column_sigma, row_sigma):
    ''' Complex loop to resort the tracklets array inplace to form track candidates. Each track candidate
    is given a quality identifier. Each hit is put to the best fitting track. Tracks are assumed to have
    no big angle, otherwise this approach does not work.
    Optimizations included to make it compile with numba. Can be called from
    several real threads if they work on different areas of the array'''
    n_duts = x.shape[1]
    actual_event_number = event_number[0]

    # Numba uses c scopes, thus define all used variables here
    n_actual_tracks = 0
    track_index, actual_hit_track_index = 0, 0  # Track index of table and first track index of actual event

    for track_index, curr_event_number in enumerate(event_number):  # Loop over all possible tracks
        # Set variables for new event
        if curr_event_number != actual_event_number:  # Detect new event
            actual_event_number = curr_event_number
            _set_n_tracks(x=x,
                          y=y,
                          start_index=track_index - n_actual_tracks,
                          stop_index=track_index,
                          n_tracks=n_tracks,
                          n_actual_tracks=n_actual_tracks,
                          n_duts=n_duts)
            n_actual_tracks = 0
            actual_hit_track_index = track_index

        n_actual_tracks += 1
        reference_hit_set = False  # The first real hit (column, row != nan) is the reference hit of the actual track
        n_track_hits = 0

        for dut_index in range(n_duts):  # loop over all DUTs in the actual track
            actual_column_sigma, actual_row_sigma = column_sigma[dut_index], row_sigma[dut_index]

            if not reference_hit_set and not np.isnan(x[track_index][dut_index]):  # Search for first DUT that registered a hit
                actual_x, actual_y = x[track_index][dut_index], y[track_index][dut_index]
                reference_hit_set = True
                hit_flag[track_index] |= (1 << dut_index)
                n_track_hits += 1
            elif reference_hit_set:  # First hit found, now find best (closest) DUT hit
                # Calculate the hit distance of the actual assigned DUT hit towards the actual reference hit
                actual_x_distance, actual_y_distance = abs(x[track_index][dut_index] - actual_x), abs(y[track_index][dut_index] - actual_y)
                actual_hit_distance = sqrt(actual_x_distance**2 + actual_y_distance**2)  # The hit distance of the actual assigned hit
                if np.isnan(x[track_index][dut_index]):
                    actual_hit_distance = -1  # Signal no hit
                shortest_hit_distance = -1  # The shortest hit distance to the actual hit; -1 means not assigned
                for hit_index in range(actual_hit_track_index, event_number.shape[0]):  # Loop over all not sorted hits of actual DUT
                    if event_number[hit_index] != actual_event_number:  # Abort condition
                        break
                    curr_x_local, curr_y_local, curr_z_local = x_local[hit_index][dut_index], y_local[hit_index][dut_index], z_local[hit_index][dut_index]
                    curr_x_err_local, curr_y_err_local, curr_z_err_local = x_err_local[hit_index][dut_index], y_err_local[hit_index][dut_index], z_err_local[hit_index][dut_index]
                    curr_x, curr_y, curr_z = x[hit_index][dut_index], y[hit_index][dut_index], z[hit_index][dut_index]
                    curr_x_err, curr_y_err, curr_z_err = x_err[hit_index][dut_index], y_err[hit_index][dut_index], z_err[hit_index][dut_index]
                    curr_charge, curr_n_hits, curr_n_cluster = charge[hit_index][dut_index], n_hits[hit_index][dut_index], n_cluster[hit_index][dut_index]
                    if not np.isnan(curr_x):  # x = nan is no hit
                        # Calculate the hit distance of the actual DUT hit towards the actual reference hit
                        x_distance, y_distance = abs(curr_x - actual_x), abs(curr_y - actual_y)
                        hit_distance = sqrt(x_distance**2 + y_distance**2)
                        if shortest_hit_distance < 0 or hit_distance < shortest_hit_distance:  # Check if the hit is closer to reference hit
                            if track_index != hit_index:  # Check if hit swapping is needed
                                if track_index > hit_index:  # Check if hit is already assigned to other track
                                    first_dut_index = _get_first_dut_index(x, hit_index)  # Get reference DUT index of other track
                                    first_dut_x, first_dut_y = x[hit_index][first_dut_index], y[hit_index][first_dut_index]
                                    # Calculate hit distance to reference hit of other track
                                    x_distance_tmp, y_distance_tmp = abs(curr_x - first_dut_x), abs(curr_y - first_dut_y)
                                    hit_distance_old = sqrt(x_distance_tmp**2 + y_distance_tmp**2)
                                    if actual_hit_distance >= 0 and actual_hit_distance < hit_distance:  # Check if actual assigned hit is better
                                        continue
                                    if hit_distance > hit_distance_old:  # Only take hit if it fits better to actual track; otherwise leave it with other track
                                        continue
                                # swapping hits
                                tmp_x_local, tmp_y_local, tmp_z_local = x_local[track_index][dut_index], y_local[track_index][dut_index], z_local[track_index][dut_index]
                                tmp_x_err_local, tmp_y_err_local, tmp_z_err_local = x_err_local[track_index][dut_index], y_err_local[track_index][dut_index], z_err_local[track_index][dut_index]
                                tmp_x, tmp_y, tmp_z = x[track_index][dut_index], y[track_index][dut_index], z[track_index][dut_index]
                                tmp_x_err, tmp_y_err, tmp_z_err = x_err[track_index][dut_index], y_err[track_index][dut_index], z_err[track_index][dut_index]
                                tmp_charge, tmp_n_hits, tmp_n_cluster = charge[track_index][dut_index], n_hits[track_index][dut_index], n_cluster[track_index][dut_index]

                                x_local[track_index][dut_index], y_local[track_index][dut_index], z_local[track_index][dut_index] = curr_x_local, curr_y_local, curr_z_local
                                x_err_local[track_index][dut_index], y_err_local[track_index][dut_index], z_err_local[track_index][dut_index] = curr_x_err_local, curr_y_err_local, curr_z_err_local
                                x[track_index][dut_index], y[track_index][dut_index], z[track_index][dut_index] = curr_x, curr_y, curr_z
                                x_err[track_index][dut_index], y_err[track_index][dut_index], z_err[track_index][dut_index] = curr_x_err, curr_y_err, curr_z_err
                                charge[track_index][dut_index], n_hits[track_index][dut_index], n_cluster[track_index][dut_index] = curr_charge, curr_n_hits, curr_n_cluster

                                x_local[hit_index][dut_index], y_local[hit_index][dut_index], z_local[hit_index][dut_index] = tmp_x_local, tmp_y_local, tmp_z_local
                                x_err_local[hit_index][dut_index], y_err_local[hit_index][dut_index], z_err_local[hit_index][dut_index] = tmp_x_err_local, tmp_y_err_local, tmp_z_err_local
                                x[hit_index][dut_index], y[hit_index][dut_index], z[hit_index][dut_index] = tmp_x, tmp_y, tmp_z
                                x_err[hit_index][dut_index], y_err[hit_index][dut_index], z_err[hit_index][dut_index] = tmp_x_err, tmp_y_err, tmp_z_err
                                charge[hit_index][dut_index], n_hits[hit_index][dut_index], n_cluster[hit_index][dut_index] = tmp_charge, tmp_n_hits, tmp_n_cluster
                                if track_index > hit_index:  # Check if hit is already assigned to other track
                                    dut_x, dut_y = x[hit_index][dut_index], y[hit_index][dut_index]
                                    first_dut_index = _get_first_dut_index(x, hit_index)  # Get reference DUT index of other track
                                    first_dut_x, first_dut_y = x[hit_index][first_dut_index], y[hit_index][first_dut_index]
                                    _reset_dut_hit_flag(dut_x=dut_x,
                                                             dut_y=dut_y,
                                                             first_dut_x=first_dut_x,
                                                             first_dut_y=first_dut_y,
                                                             hit_flag=hit_flag,
                                                             hit_index=hit_index,
                                                             dut_index=dut_index,
                                                             dut_column_sigma=actual_column_sigma,
                                                             dut_row_sigma=actual_row_sigma)
                            shortest_hit_distance = hit_distance
                            n_track_hits += 1
                curr_x, curr_y = x[track_index][dut_index], y[track_index][dut_index]
                _set_dut_hit_flag(dut_x=actual_x,
                                       dut_y=actual_y,
                                       curr_x=curr_x,
                                       curr_y=curr_y,
                                       hit_flag=hit_flag,
                                       track_index=track_index,
                                       dut_index=dut_index,
                                       dut_column_sigma=actual_column_sigma,
                                       dut_row_sigma=actual_row_sigma)

        # Set number of tracks of last event
        _set_n_tracks(x=x,
                      y=y,
                      start_index=track_index - n_actual_tracks + 1,
                      stop_index=track_index + 1,
                      n_tracks=n_tracks,
                      n_actual_tracks=n_actual_tracks,
                      n_duts=n_duts)


@njit
def _find_merged_tracks(tracks_array, min_track_distance):  # Check if several tracks are less than min_track_distance apart. Then exclude these tracks (set n_tracks = -1)
    i = 0
    for _ in range(0, tracks_array.shape[0]):
        track_index = i
        if track_index >= tracks_array.shape[0]:
            break
        actual_event = tracks_array[track_index]['event_number']
        for _ in range(track_index, tracks_array.shape[0]):  # Loop over event hits
            if tracks_array[i]['event_number'] != actual_event:  # Next event reached, break loop
                break
            if tracks_array[i]['n_tracks'] < 2:  # Only if the event has more than one track check the min_track_distance
                i += 1
                break
            offset_x, offset_y = tracks_array[i]['offset_0'], tracks_array[i]['offset_1']
            for j in range(i + 1, tracks_array.shape[0]):  # Loop over other event hits
                if tracks_array[j]['event_number'] != actual_event:  # Next event reached, break loop
                    break
                if sqrt((offset_x - tracks_array[j]['offset_0'])**2 + (offset_y - tracks_array[j]['offset_1'])**2) < min_track_distance:
                    tracks_array[i]['quality_flag'] = 0
                    tracks_array[j]['quality_flag'] = 0
            i += 1


def _fit_tracks_loop(track_hits):
    ''' Do 3d line fit and calculate chi2 for each fit. '''
    def line_fit_3d(hits):
        # subtract mean for each component (x,y,z) for SVD calculation
        datamean = hits.mean(axis=0)
        offset, slope = datamean, np.linalg.svd(hits - datamean, full_matrices=False)[2][0]  # http://stackoverflow.com/questions/2298390/fitting-a-line-in-3d
        slope_mag = np.sqrt(slope.dot(slope))
        # force the 3rd component (z) to be positive
        if slope[2] < 0:
            slope = -slope
        intersections = offset + slope / slope[2] * (hits.T[2][:, np.newaxis] - offset[2])  # Fitted line and DUT plane intersections (here: points)
        chi2 = np.sum(np.square(hits - intersections))  # Chi2 of the fit in um
        return datamean, slope / slope_mag, chi2

    slope = np.empty((track_hits.shape[0], 3), dtype=np.float)
    offset = np.empty((track_hits.shape[0], 3), dtype=np.float)
    chi2 = np.empty((track_hits.shape[0],), dtype=np.float)

    for index, hits in enumerate(track_hits):  # Loop over selected track candidate hits and fit
#         print "***"
#         print "hits with nn", hits
        hits = hits[~np.isnan(hits).any(axis=1)]
#         print "hits w/o nn", hits
#         if len(hits) <= 1:
#             print "***** only one"
#             offset[index], slope[index], chi2[index] = np.nan, np.nan, np.nan
#             continue
        try:
            offset[index], slope[index], chi2[index] = line_fit_3d(hits)
        except np.linalg.linalg.LinAlgError:
            offset[index], slope[index], chi2[index] = np.nan, np.nan, np.nan

    return offset, slope, chi2


def _function_wrapper_fit_tracks_kalman_loop(*args):  # Needed for multiprocessing call with arguments
    '''
    Function for multiprocessing call with arguments for speed up.
    '''
    pixel_size, n_pixels, dut_fit_selection, z_positions, alignment, beam_energy, material_budget, add_scattering_plane, track_hits = args

    return _fit_tracks_kalman_loop(track_hits, dut_fit_selection, pixel_size, n_pixels, z_positions, alignment, beam_energy, material_budget, add_scattering_plane)[0:2]


def _kalman_fit_3d(hits, alignment, dut_fit_selection, transition_matrix, transition_covariance, transition_offset, observation_matrix, observation_covariance, observation_offset, initial_state_mean, initial_state_covariance):
    '''
    This function calls the Kalman Filter. It returns track by track the smoothed state vector which contains in the first two components
    the smoothed hit positions and in the last two components the respective slopes. Additionally the chi square of the track is calculated
    and returned.

    Parameters
    ----------
    hits : array_like
        Array which contains the x, y and z hit position of each DUT for one track.
    alignment : array_like or None
        Aligment data, which contains rotations and translations for each DUT. Needed to take rotations of DUTs into account,
        in order to get correct transition matrices. If pre-alignment data is used, this is set to None since no rotations have to be
        taken into account.
    dut_fit_selection : iterable
        List of DUTs which should be included in Kalman Filter. DUTs which are not in list
        were treated as missing measurements and will not be included in the Filtering step.
    transition_matrix : array_like
        Transition matrix for each DUT except the last one. The transition matrix transports the state vector from one DUT to another DUT.
    transition_covariance : array_like
        Matrix which describes for each DUT except the last one the covariance of the transition matrix.
    transition_offset : array_like
        Vector which array_like the offset of each transition.
    observation_matrix : array_like
        Matrix which converts the state vector to the actual measurement vector.
    observation_covariance : array_like
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

    measurements = ma.array(np.append(hits[:, :, 0:2], (np.repeat([np.zeros((hits.shape[1], 2))], hits.shape[0], axis=0)), axis=-1))

    # mask duts which should not used in fit
    for dut_index in range(0, measurements.shape[1]):
        if dut_index not in dut_fit_selection:
            measurements[:, dut_index, ] = ma.masked

    # Check for invalid values (NaN)
    if np.any(np.isnan(measurements)):
        logging.warning('Not all measurements have valid values (Array contains NANs).')

    smoothed_state_estimates, cov = kf.smooth(alignment, transition_matrix, transition_offset, transition_covariance,
                                              observation_matrix, observation_offset, observation_covariance,
                                              initial_state_mean, initial_state_covariance, measurements)

    chi2 = np.sum(np.square(measurements[:, :, 0:2] - smoothed_state_estimates[:, :,  0:2]), dtype=np.uint32, axis=(1, 2))
    x_err = np.sqrt(np.diagonal(cov, axis1=3, axis2=2))[:, :, 0]
    y_err = np.sqrt(np.diagonal(cov, axis1=3, axis2=2))[:, :, 1]

    # Check for invalid values (NaN)
    if np.any(np.isnan(smoothed_state_estimates)):
        logging.warning('Not all smoothed state estimates have valid values (Array contains NANs)! Check input of  Kalman Filter.')

    return smoothed_state_estimates, chi2, x_err, y_err


def _fit_tracks_kalman_loop(track_hits, dut_fit_selection, pixel_size, n_pixels, z_positions, alignment, beam_energy, material_budget, add_scattering_plane):
    '''
    Loop over the selected tracks. In this function all matrices for the Kalman Filter are calculated track by track
    and the Kalman Filter is started. With dut_fit_selection only the duts which are selected are included in the Kalman Filter.
    Not included DUTs are masked.

    Parameters
    ----------
    track_hits : array_like
        Array which contains the x, y and z hit position of each DUT for all tracks.
    dut_fit_selection : uint
        8-bit- integer in binary representation. E.g. dut_fit_selection = 61 (0b111101) means that measurements of DUT 1 and
        DUT 6 are treated as missing measurements and will not be included in the Filtering step.
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension (column/row),
        e.g. for two FE-I4 DUTs [(250, 50), (250, 50)].
    n_pixels : iterable of tuples
        One tuple per DUT describing the number of pixels in column, row direction
        e.g. for 2 DUTs: n_pixels = [(80, 336), (80, 336)]. Only needed for Kalman Filter.
    z_positions : iterable
        The z positions of the DUTs in um. Here, needed for Kalman Filter.
    alignment : array_like or None
        Aligment data, which contains rotations and translations for each DUT. Needed to take rotations of DUTs into account,
        in order to get correct transition matrices. If pre-alignment data is used, this is set to None since no rotations have to be
        taken into account.
    beam_energy : uint
        Energy of electron beam in MeV.
    material_budget : iterable
        Material budget of all DUTs. The material budget is defined as the thickness (sensor + other scattering materials)
        devided by the radiation length (Silicon: 93700 um, M26(50 um Si + 50 um Kapton): 125390 um).
    add_scattering_plane : dict
        Specifies additional scattering planes in case of additional DUTs which are not used.
        The dictionary must contain the following keys:
            z_scatter: z position of scattering plane in um
            material_budget_scatter: material budget of scattering plane
            alignment_scatter: list which contains alpha, beta and gamma angles of scattering plane.
                               If None, no rotation will be considered.
        In case of multiple scattering planes, each value of a key is a list, with items corresponding to each scattering plane.
        If add_scattering_plane is False, no scattering plane will be added.
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
    z_positions = np.array(z_positions)
    n_pixels = np.array(n_pixels)
    n_duts = track_hits.shape[1]
    chunk_size = track_hits.shape[0]
    dut_selection = np.array(range(0, n_duts))

    # set multiple scattering environment
    material_budget = np.array(material_budget)

    if add_scattering_plane:
        # initialize scattering plane values
        z_scatter = add_scattering_plane['z_scatter']
        index_scatter = [np.where(np.sort(np.append(z_positions, z_scatter)) == z_scatter[i])[0][0] for i in range(len(z_scatter))]
        material_budget_scatter = add_scattering_plane['material_budget_scatter']
        alignment_scatter_total = []
        for i in range(len(add_scattering_plane['alignment_scatter'])):
            if add_scattering_plane['alignment_scatter'][i] is not None:
                alignment_scatter = [(index_scatter[i], 0., 0., z_scatter[i], add_scattering_plane['alignment_scatter'][i][0],
                                     add_scattering_plane['alignment_scatter'][i][1], add_scattering_plane['alignment_scatter'][i][2], 0., 0.)]
                alignment_scatter_total.append(alignment_scatter)
            else:
                alignment_scatter = [(index_scatter[i], 0., 0., z_scatter[i], 0., 0., 0., 0., 0.)]
                alignment_scatter_total.append(alignment_scatter)
        # append new values
        for i in range(len(z_scatter)):
            material_budget = np.insert(material_budget, index_scatter[i], material_budget_scatter[i])
            z_positions = np.insert(z_positions, index_scatter[i], z_scatter[i])
            alignment = np.insert(alignment, index_scatter[i], alignment_scatter_total[i])
            track_hits = np.insert(track_hits, index_scatter[i], np.full((track_hits.shape[0], track_hits.shape[2]), fill_value=np.nan), axis=1)
        # fix dut number in alignment array
        for index in range(alignment.shape[0]):
            alignment[index][0] = index
        # correct number of duts and update dut selection
        n_duts = n_duts + len(z_scatter)
        dut_selection = np.array(range(0, n_duts))
    # Calculate multiple scattering
    mass = 0.511  # mass in MeV (electrons)
    momentum = np.sqrt(beam_energy**2 - mass**2)
    beta = momentum / beam_energy  # almost 1

    # rms angle of multiple scattering
    theta = np.array(((13.6 / momentum / beta) * np.sqrt(material_budget) * (1. + 0.038 * np.log(material_budget))))

    # express transition covariance matrix
    transition_covariance = np.zeros((chunk_size, n_duts - 1, 4, 4))

    # express transition matrix
    transition_matrix = np.zeros((chunk_size, n_duts - 1, 4, 4))

    # express transition and observation offset matrices
    transition_offset = np.zeros((chunk_size, n_duts - 1, 4))
    observation_offset = np.zeros((chunk_size, n_duts, 4))

    # express initial state. Contains (x_pos, y_pos, slope_x, slope_y).
    initial_state_mean = np.zeros((chunk_size, 4))

    # express observatipon matrix
    observation_matrix = np.zeros((chunk_size, n_duts, 4, 4))

    # express observation covariance matrices
    observation_covariance = np.zeros((chunk_size, n_duts, 4, 4))
    # only observe x and y position
    observation_matrix[:, :, 0, 0] = 1.
    observation_matrix[:, :, 1, 1] = 1.

    # express initial state covariance matrices: x and y pos have initial error of pixel resolution and x and y slopes have large error
    initial_state_covariance = np.zeros((chunk_size, 4, 4))
    # error on initial slope is roughly divergence of beam (5 mrad). Error on initial x-y position depends on fit selection
    initial_state_covariance[:, 2, 2] = np.square(5e-3)
    initial_state_covariance[:, 3, 3] = np.square(5e-3)

    # create a list of duts which should be included in the fit
    dut_list = np.full(shape=(n_duts), fill_value=np.nan)
    for index in range(n_duts):
        dut_n = index
        if np.bitwise_and(1 << index, dut_fit_selection) == 2 ** index:
            dut_list[dut_n] = dut_n
    dut_fit_selection = dut_list[~np.isnan(dut_list)].astype(int)

    # This selection is needed for matrices for the kalman filter.
    # It selects all duts, except the last one.
    sel = dut_selection[:-1]
    z_diff = z_positions[sel + 1] - z_positions[sel]

    if add_scattering_plane:
        for i in range(len(index_scatter)):  # need to shift dut fit selection in case of additional scattering plane
                dut_fit_selection[np.where(dut_fit_selection > (index_scatter[i] - 1))[0][0]:] = dut_fit_selection[np.where(dut_fit_selection > (index_scatter[i] - 1))[0][0]:] + 1

    for index, actual_hits in enumerate(track_hits):  # Loop over selected track candidate hits and fit
        # cluster hit position error
        x_err = np.array(actual_hits[:, 3])
        y_err = np.array(actual_hits[:, 4])

        # Take cluster hit position error as measurement error for duts which have a hit.
        # For those who have no hit, need no error, since the should not be included in fit via fit selection
        observation_covariance[index, dut_selection[~np.isnan(x_err)], 0, 0] = np.square(x_err[dut_selection[~np.isnan(x_err)]])
        observation_covariance[index, dut_selection[~np.isnan(x_err)], 1, 1] = np.square(y_err[dut_selection[~np.isnan(x_err)]])

        if dut_selection[0] in dut_fit_selection:  # first dut is in fit selection
            # If first dut is used in track building, take first dut hit as initial value and
            # its corresponding cluster position error as the error on the measurement.
            initial_state_mean[index] = np.array([actual_hits[0, 0], actual_hits[0, 1], 0., 0.])
            initial_state_covariance[index, 0, 0] = np.square(x_err[0])
            initial_state_covariance[index, 1, 1] = np.square(y_err[0])
        else:  # first dut is not in fit selction
            # Take hit from first dut which is in fit selection. Cannot take hit from first dut,
            # since do not want to pass measurement to kalman filter (unbiased).
            # Due to the fact that this hit position can be very off through multiple scattering,
            # take whole sensor as error (this error must be very large).
            initial_state_mean[index] = np.array([actual_hits[dut_fit_selection[0], 0], actual_hits[dut_fit_selection[0], 1], 0., 0.])
            initial_state_covariance[index, 0, 0] = np.square(n_pixels * pixel_size)[dut_fit_selection[0], 0]
            initial_state_covariance[index, 1, 1] = np.square(n_pixels * pixel_size)[dut_fit_selection[0], 1]

        # express transition matrices
        # transition matrices are filled already here. In case of prealignment matrices will not be updated.
        # If alignment is used, transition matrices are updated (in Kalman Filter) before each prediction step in order to take
        # rotations of planes into account.
        transition_matrix[index, sel, :, 0] = np.array([1., 0., 0., 0.])
        transition_matrix[index, sel, :, 1] = np.array([0., 1., 0., 0.])
        transition_matrix[index, sel, :, 2] = np.array([(z_diff), np.zeros((len(sel),)),
                                                        np.ones((len(sel),)), np.zeros((len(sel),))]).T
        transition_matrix[index, sel, :, 3] = np.array([np.zeros((len(sel),)), (z_diff),
                                                        np.zeros((len(sel),)), np.ones((len(sel),))]).T

        # express transition covariance matrices, according to http://web-docs.gsi.de/~ikisel/reco/Methods/CovarianceMatrices-NIMA329-1993.pdf
        transition_covariance[index, sel, :, 0] = np.array([(z_diff)**2 * theta[sel]**2,
                                                            np.zeros((len(sel),)),
                                                            -(z_diff) * theta[sel]**2,
                                                            np.zeros((len(sel),))]).T
        transition_covariance[index, sel, :, 1] = np.array([np.zeros((len(sel),)),
                                                            (z_diff)**2 * theta[sel]**2,
                                                            np.zeros((len(sel),)),
                                                            -(z_diff) * theta[sel]**2]).T
        transition_covariance[index, sel, :, 2] = np.array([-(z_diff) * theta[sel]**2,
                                                            np.zeros((len(sel),)),
                                                            theta[sel]**2,
                                                            np.zeros((len(sel),))]).T
        transition_covariance[index, sel, :, 3] = np.array([np.zeros((len(sel),)),
                                                            -(z_diff) * theta[sel]**2,
                                                            np.zeros((len(sel),)),
                                                            theta[sel]**2]).T

    # run kalman filter
    track_estimate_chunks, chi2, x_err, y_err = _kalman_fit_3d(track_hits[:, :, 0:2], alignment, dut_fit_selection,
                                                               transition_matrix, transition_covariance,
                                                               transition_offset, observation_matrix,
                                                               observation_covariance, observation_offset,
                                                               initial_state_mean, initial_state_covariance)

    if add_scattering_plane:  # delete estimated state vector at scattering plane
        track_estimate_chunks = np.delete(track_estimate_chunks, index_scatter, axis=1)
        x_err = np.delete(x_err, index_scatter, axis=1)
        y_err = np.delete(y_err, index_scatter, axis=1)

    return track_estimate_chunks, chi2, x_err, y_err
