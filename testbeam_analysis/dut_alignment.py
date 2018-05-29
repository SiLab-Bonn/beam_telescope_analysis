''' All DUT alignment functions in space and time are listed here plus additional alignment check functions'''
from __future__ import division

import logging
import os
from collections import Iterable

import tables as tb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import progressbar

from testbeam_analysis.telescope.telescope import Telescope
from testbeam_analysis.tools import analysis_utils
from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.tools import data_selection
from testbeam_analysis.track_analysis import find_tracks, fit_tracks
from testbeam_analysis.result_analysis import calculate_residuals, histogram_track_angle


def apply_alignment(telescope_configuration, input_merged_file, output_merged_file=None, local_to_global=True, chunk_size=1000000):
    '''Convert local to global coordinates and vice versa.

    Note:
    -----
    This function cannot be easily made faster with multiprocessing since the computation function (apply_alignment_to_chunk) does not
    contribute significantly to the runtime (< 20 %), but the copy overhead for not shared memory needed for multipgrocessing is higher.
    Also the hard drive IO can be limiting (30 Mb/s read, 20 Mb/s write to the same disk)

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_merged_file : string
        Filename of the input merged file.
    output_merged_file : string
        Filename of the output merged file with the converted coordinates.
    local_to_global : bool
        If True, convert from local to global coordinates.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('== Apply alignment to %d DUTs ==', n_duts)

    if output_merged_file is None:
        output_merged_file = os.path.splitext(input_merged_file)[0] + ('_global_coordinates.h5' if local_to_global else '_local_coordinates.h5')

    # Looper over the hits of all DUTs of all hit tables in chunks and apply the alignment
    with tb.open_file(input_merged_file, mode='r') as in_file_h5:
        with tb.open_file(output_merged_file, mode='w') as out_file_h5:
            for node in in_file_h5.root:  # Loop over potential hit tables in data file
                hits = node
                new_node_name = hits.name

                if new_node_name == 'MergedClusters':  # Merged cluster with alignment are tracklets
                    new_node_name = 'Tracklets'

                hits_aligned_table = out_file_h5.create_table(out_file_h5.root, name=new_node_name, description=hits.dtype, title=hits.title, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))

                progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=hits.shape[0], term_width=80)
                progress_bar.start()

                for hits_chunk, index in analysis_utils.data_aligned_at_events(hits, chunk_size=chunk_size):  # Loop over the hits
                    for dut_index, dut in enumerate(telescope):  # Loop over the DUTs
                        if local_to_global:
                            conv = dut.local_to_global_position
                        else:
                            conv = dut.global_to_local_position

                        hits_chunk['x_dut_%d' % dut_index], hits_chunk['y_dut_%d' % dut_index], hits_chunk['z_dut_%d' % dut_index] = conv(
                            x=hits_chunk['x_dut_%d' % dut_index],
                            y=hits_chunk['y_dut_%d' % dut_index],
                            z=hits_chunk['z_dut_%d' % dut_index])

                    hits_aligned_table.append(hits_chunk)
                    progress_bar.update(index)
                progress_bar.finish()

    return output_merged_file


def prealignment(telescope_configuration, input_correlation_file, output_alignment_file, ref_index=0, reduce_background=True, plot=True, gui=False, queue=False):
    '''Deduce a pre-alignment from the correlations, by fitting the correlations with a straight line (gives offset, slope, but no tild angles).
       The user can define cuts on the fit error and straight line offset in an interactive way.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_correlation_file : string
        Filename of the input correlation file.
    output_alignment_file : string
        Filename of the output alignment file.
    ref_index : uint
        DUT index of the reference plane. Default is DUT 0.
    reduce_background : bool
        Reduce background (uncorrelated events) by applying SVD method on the 2D correlation array.
    plot : bool
        If True, create additional output plots.
    gui : bool
        If True, this function is excecuted from GUI and returns figures
    queue : bool, dict
        If gui is True and non_interactive is False, queue is a dict with a in and output queue to communicate with GUI thread
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Pre-alignment of %d DUTs ===' % n_duts)

    if plot is True and not gui:
        output_pdf = PdfPages(os.path.splitext(output_alignment_file)[0] + '_prealigned.pdf', keep_empty=False)
    else:
        output_pdf = None

    figs = [] if gui else None

    with tb.open_file(input_correlation_file, mode="r") as in_file_h5:
        # remove reference DUT from list of DUTs
        select_duts = list(set(range(n_duts)) - set([ref_index]))

        for dut_index in select_duts:
            for x_direction in [True, False]:
                if reduce_background:
                    node = in_file_h5.get_node(in_file_h5.root, 'Correlation_%s_%d_%d_reduced_background' % ('x' if x_direction else 'y', ref_index, dut_index))
                else:
                    node = in_file_h5.get_node(in_file_h5.root, 'Correlation_%s_%d_%d' % ('x' if x_direction else 'y', ref_index, dut_index))
                dut_name = telescope[dut_index].name
                ref_name = telescope[ref_index].name
                logging.info('Pre-aligning data from %s', node.name)
                bin_size = node.attrs.resolution
                ref_size = node.attrs.ref_size
                dut_size = node.attrs.dut_size

                # retrieve data
                data = node[:]

                # Initialize arrays with np.nan (invalid), adding 0.5 to change from index to position
                # matrix index 0 is cluster index 1 ranging from 0.5 to 1.4999, which becomes position 0.0 to 0.999 with center at 0.5, etc.
                dut_pos = np.arange(start=0.0, stop=dut_size, step=bin_size) + 0.5 * bin_size - 0.5 * dut_size
                mean_fitted = np.empty(shape=(len(dut_pos),), dtype=np.float)  # Peak of the Gauss fit
                mean_fitted.fill(np.nan)
                mean_error_fitted = np.empty(shape=(len(dut_pos),), dtype=np.float)  # Error of the fit of the peak
                mean_error_fitted.fill(np.nan)
                sigma_fitted = np.empty(shape=(len(dut_pos),), dtype=np.float)  # Sigma of the Gauss fit
                sigma_fitted.fill(np.nan)
                chi2 = np.empty(shape=(len(dut_pos),), dtype=np.float)  # Chi2 of the fit
                chi2.fill(np.nan)

                # calculate half hight
                median = np.median(data)
                median_max = np.median(np.max(data, axis=1))
                half_median_data = (data > ((median + median_max) / 2))
                # calculate maximum per column
                max_select = np.argmax(data, axis=1)
                hough_data = np.zeros_like(data)
                hough_data[np.arange(data.shape[0]), max_select] = 1
                # select maximums if larger than half hight
                hough_data = hough_data & half_median_data
                # transpose for correct angle
                hough_data = hough_data.T
                accumulator, theta, rho, theta_edges, rho_edges = analysis_utils.hough_transform(hough_data, theta_res=0.1, rho_res=1.0, return_edges=True)

                def largest_indices(ary, n):
                    ''' Returns the n largest indices from a numpy array.

                    https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
                    '''
                    flat = ary.flatten()
                    indices = np.argpartition(flat, -n)[-n:]
                    indices = indices[np.argsort(-flat[indices])]
                    return np.unravel_index(indices, ary.shape)

                # finding correlation
                count_nonzero = np.count_nonzero(accumulator)
                indices = np.vstack(largest_indices(accumulator, count_nonzero)).T
                for index in indices:
                    rho_idx, th_idx = index[0], index[1]
                    rho_val, theta_val = rho[rho_idx], theta[th_idx]
                    slope_idx, offset_idx = -np.cos(theta_val) / np.sin(theta_val), rho_val / np.sin(theta_val)
                    slope = slope_idx
                    offset = offset_idx * bin_size
                    if np.isclose(slope, 1.0, rtol=0.0, atol=0.1) or np.isclose(slope, -1.0, rtol=0.0, atol=0.1):
                        break
                else:
                    raise RuntimeError('Cannot find correlation between %s and %s' % (telescope[ref_index].name, telescope[dut_index].name))
                # offset in the center of the pixel matrix
                offset_center = offset - (0.5 * ref_size - 0.5 * bin_size) + slope * (0.5 * dut_size - 0.5 * bin_size)

                plot_utils.plot_hough(dut_pos=dut_pos,
                                      data=hough_data,
                                      accumulator=accumulator,
                                      offset=offset_center,
                                      slope=slope,
                                      theta_edges=theta_edges,
                                      rho_edges=rho_edges,
                                      ref_size=ref_size,
                                      dut_size=dut_size,
                                      ref_name=ref_name,
                                      dut_name=dut_name,
                                      x_direction=x_direction,
                                      reduce_background=reduce_background,
                                      output_pdf=output_pdf,
                                      gui=gui,
                                      figs=figs)

                if x_direction:
                    if slope < 0.0:
                        telescope[dut_index].rotation_beta = np.pi
                    telescope[dut_index].translation_x = offset_center
                else:
                    if slope < 0.0:
                        telescope[dut_index].rotation_alpha = np.pi
                    telescope[dut_index].translation_y = offset_center

    telescope.save_configuration(configuration_file=os.path.splitext(telescope_configuration)[0] + '_prealigned.yaml')

    if output_pdf is not None:
        output_pdf.close()

    if gui:
        return figs


# TODO: selection_track_quality to selection_track_quality_sigma
def alignment(input_merged_file, input_alignment_file, n_pixels, pixel_size, dut_names=None, align_duts=None, select_telescope_duts=None, select_fit_duts=None, select_hit_duts=None, quality_sigma=5.0, alignment_order=None, initial_rotation=None, initial_translation=None, max_iterations=3, max_events=100000, fit_method='Fit', min_track_distance=None, beam_energy=None, material_budget=None, use_fit_limits=True, new_alignment=True, plot=False, chunk_size=100000):
    ''' This function does an alignment of the DUTs and sets translation and rotation values for all DUTs.
    The reference DUT defines the global coordinate system position at 0, 0, 0 and should be well in the beam and not heavily rotated.

    To solve the chicken-and-egg problem that a good dut alignment needs hits belonging to one track, but good track finding needs a good dut alignment this
    function work only on already prealigned hits belonging to one track. Thus this function can be called only after track finding.

    These steps are done
    1. Take the found tracks and revert the pre-alignment
    2. Take the track hits belonging to one track and fit tracks for all DUTs
    3. Calculate the residuals for each DUT
    4. Deduce rotations from the residuals and apply them to the hits
    5. Deduce the translation of each plane
    6. Store and apply the new alignment

    repeat step 3 - 6 until the total residual does not decrease (RMS_total = sqrt(RMS_x_1^2 + RMS_y_1^2 + RMS_x_2^2 + RMS_y_2^2 + ...))

    Parameters
    ----------
    input_merged_file : string
        Input file name of the merged cluster hit table from all DUTs.
    input_alignment_file : pytables file
        File name of the input aligment data.
    n_pixels : iterable of tuples
        One tuple per DUT describing the total number of pixels (column/row),
        e.g. for two FE-I4 DUTs [(80, 336), (80, 336)].
    pixel_size : iterable of tuples
        One tuple per DUT describing the pixel dimension (column/row),
        e.g. for two FE-I4 DUTs [(250, 50), (250, 50)].
    align_duts : iterable or iterable of iterable
        The combination of duts that are algined at once. One should always align the high resolution planes first.
        E.g. for a telesope (first and last 3 planes) with 2 devices in the center (3, 4):
        align_duts=[[0, 1, 2, 5, 6, 7],  # align the telescope planes first
                    [4],  # align first DUT
                    [3]]  # align second DUT
    select_telescope_duts : iterable
        The telescope will be aligned to the given DUTs. The translation in x and y of these DUTs will not be changed.
        Usually the two outermost telescope DUTs are selected.
    select_fit_duts : iterable or iterable of iterable
        Defines for each align_duts combination wich devices to use in the track fit.
        E.g. To use only the telescope planes (first and last 3 planes) but not the 2 center devices
        select_fit_duts=[0, 1, 2, 5, 6, 7]
    select_hit_duts : iterable or iterable of iterable
        Defines for each align_duts combination wich devices must have a hit to use the track for fitting. The hit
        does not have to be used in the fit itself! This is useful for time reference planes.
        E.g.  To use telescope planes (first and last 3 planes) + time reference plane (3)
        select_hit_duts = [0, 1, 2, 4, 5, 6, 7]
    quality_sigma : float
        Track quality for each hit DUT.
    initial_rotation : array
        Initial rotation array. If None, deduce the rotation from pre-alignment.
    initial_translation : array
        Initial translation array. If None, deduce the translation from pre-alignment.
    max_iterations : uint
        Maximum number of iterations of calc residuals, apply rotation refit loop until constant result is expected.
        Usually the procedure converges rather fast (< 5 iterations)
    max_events: uint
        Radomly select max_events for alignment. If None, use all events, which might slow down the alignment.
    use_fit_limits : bool
        If True, use fit limits from pre-alignment for residual calculation.
    new_alignment : bool
        If True, discard existig alignment parameters from input alignment file and start all over.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Aligning DUTs ===')

    # Open the pre-alignment and create empty alignment info (at the beginning only the z position is set)
    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        prealignment = in_file_h5.root.PreAlignment[:]
        n_duts = prealignment.shape[0]
        alignment_parameters = _create_alignment_array(n_duts)
        alignment_parameters['translation_z'] = prealignment['z']

        beta = np.arctan(prealignment[select_telescope_duts[-1]]["column_c0"] / prealignment[select_telescope_duts[-1]]["z"])
        alpha = -np.arctan(prealignment[select_telescope_duts[-1]]["row_c0"] / prealignment[select_telescope_duts[-1]]["z"])
        global_rotation = geometry_utils.rotation_matrix(alpha=alpha,
                                                         beta=beta,
                                                         gamma=0.0)

        for dut_index in range(n_duts):
            if (isinstance(initial_translation, Iterable) and initial_translation[dut_index] is None) or initial_translation is None:
                if dut_index in select_telescope_duts:
                    # Telescope is aligned to these DUTs, so align them to the z-axis
                    alignment_parameters['translation_x'][dut_index] = 0.0
                    alignment_parameters['translation_y'][dut_index] = 0.0
                else:
                    # Correct x and y, so that the telescope is aligned to the z-axis
                    aligned_offset = np.dot(global_rotation, np.array([prealignment[dut_index]["column_c0"], prealignment[dut_index]["row_c0"], prealignment[dut_index]["z"]]))
                    alignment_parameters['translation_x'][dut_index] = aligned_offset[0]
                    alignment_parameters['translation_y'][dut_index] = aligned_offset[1]
            elif isinstance(initial_translation, Iterable) and isinstance(initial_translation[dut_index], Iterable) and len(initial_translation[dut_index]) in [2, 3]:
                alignment_parameters['translation_x'][dut_index] = initial_translation[dut_index][0]
                alignment_parameters['translation_y'][dut_index] = initial_translation[dut_index][1]
                if len(initial_translation[dut_index]) == 3 and initial_translation[dut_index][2] is not None:
                    alignment_parameters['translation_z'][dut_index] = initial_translation[dut_index][2]
            else:
                raise ValueError('initial_translation format not supported')

            if (isinstance(initial_rotation, Iterable) and initial_rotation[dut_index] is None) or initial_rotation is None:
                alignment_parameters['alpha'][dut_index] = np.pi if np.isclose(-1, prealignment[dut_index]["row_c1"], atol=0.1) else 0.0
                alignment_parameters['beta'][dut_index] = np.pi if np.isclose(-1, prealignment[dut_index]["column_c1"], atol=0.1) else 0.0
#                 alignment_parameters['gamma'][dut_index] = initial_rotation[dut_index][2]
            elif isinstance(initial_rotation, Iterable) and isinstance(initial_rotation[dut_index], Iterable) and len(initial_rotation[dut_index]) == 3:
                    alignment_parameters['alpha'][dut_index] = initial_rotation[dut_index][0]
                    alignment_parameters['beta'][dut_index] = initial_rotation[dut_index][1]
                    alignment_parameters['gamma'][dut_index] = initial_rotation[dut_index][2]
            else:
                raise ValueError('initial_rotation format not supported')

    # Create list with combinations of DUTs to align
    if align_duts is None:  # If None: align all DUTs
        align_duts = range(n_duts)
    # Check for value errors
    if not isinstance(align_duts, Iterable):
        raise ValueError("align_duts is no iterable")
    elif not align_duts:  # empty iterable
        raise ValueError("align_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), align_duts)):
        align_duts = [align_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), align_duts)):
        raise ValueError("not all items in align_duts are iterable")
    # Finally check length of all iterables in iterable
    for dut in align_duts:
        if not dut:  # check the length of the items
            raise ValueError("item in align_duts has length 0")

    # Check if some DUTs will not be aligned
    no_align_duts = set(range(n_duts)) - set(np.unique(np.hstack(np.array(align_duts))).tolist())
    if no_align_duts:
        logging.warning('These DUTs will not be aligned: %s', ", ".join(str(align_dut) for align_dut in no_align_duts))

    # overwrite configuration for align DUTs
    # keep configuration for DUTs that will not be aligned
    if new_alignment:
        geometry_utils.save_alignment_parameters(alignment_file=input_alignment_file,
                                                 alignment_parameters=alignment_parameters,
                                                 select_duts=np.unique(np.hstack(np.array(align_duts))).tolist(),
                                                 mode='absolute')
    else:
        pass  # do nothing here, keep existing configuration

    # Create track, hit selection
    if select_hit_duts is None:  # If None: use all DUTs
        select_hit_duts = []
        # copy each item
        for duts in align_duts:
            select_hit_duts.append(duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(select_hit_duts, Iterable):
        raise ValueError("select_hit_duts is no iterable")
    elif not select_hit_duts:  # empty iterable
        raise ValueError("select_hit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_hit_duts)):
        select_hit_duts = [select_hit_duts[:] for _ in align_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_hit_duts)):
        raise ValueError("not all items in select_hit_duts are iterable")
    # Finally check length of all arrays
    if len(select_hit_duts) != len(align_duts):  # empty iterable
        raise ValueError("select_hit_duts has the wrong length")
    for hit_dut in select_hit_duts:
        if len(hit_dut) < 2:  # check the length of the items
            raise ValueError("item in select_hit_duts has length < 2")

    # Create track, hit selection
    if select_fit_duts is None:  # If None: use all DUTs
        select_fit_duts = []
        # copy each item
        for hit_duts in select_hit_duts:
            select_fit_duts.append(hit_duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(select_fit_duts, Iterable):
        raise ValueError("select_fit_duts is no iterable")
    elif not select_fit_duts:  # empty iterable
        raise ValueError("select_fit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_fit_duts)):
        select_fit_duts = [select_fit_duts[:] for _ in align_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_fit_duts)):
        raise ValueError("not all items in select_fit_duts are iterable")
    # Finally check length of all arrays
    if len(select_fit_duts) != len(align_duts):  # empty iterable
        raise ValueError("select_fit_duts has the wrong length")
    for index, fit_dut in enumerate(select_fit_duts):
        if len(fit_dut) < 2:  # check the length of the items
            raise ValueError("item in select_fit_duts has length < 2")
        if set(fit_dut) - set(select_hit_duts[index]):  # fit DUTs are required to have a hit
            raise ValueError("DUT in select_fit_duts is not in select_hit_duts")

#     # Create track, hit selection
#     if not isinstance(selection_track_quality, Iterable):  # all items the same, special case for selection_track_quality
#         selection_track_quality = [[selection_track_quality] * len(hit_duts) for hit_duts in select_hit_duts]  # every hit DUTs require a track quality value
#     # Check iterable and length
#     if not isinstance(selection_track_quality, Iterable):
#         raise ValueError("selection_track_quality is no iterable")
#     elif not selection_track_quality:  # empty iterable
#         raise ValueError("selection_track_quality has no items")
#     # Check if only non-iterable in iterable
#     if all(map(lambda val: not isinstance(val, Iterable), selection_track_quality)):
#         selection_track_quality = [selection_track_quality for _ in align_duts]
#     # Check if only iterable in iterable
#     if not all(map(lambda val: isinstance(val, Iterable), selection_track_quality)):
#         raise ValueError("not all items in selection_track_quality are iterable")
#     # Finally check length of all arrays
#     if len(selection_track_quality) != len(align_duts):  # empty iterable
#         raise ValueError("selection_track_quality has the wrong length")
#     for index, track_quality in enumerate(selection_track_quality):
#         if len(track_quality) != len(select_hit_duts[index]):  # check the length of each items
#             raise ValueError("item in selection_track_quality and select_hit_duts does not have the same length")

    if not isinstance(quality_sigma, Iterable):
        quality_sigma = [quality_sigma] * len(align_duts)
    # Finally check length of all arrays
    if len(quality_sigma) != len(align_duts):  # empty iterable
        raise ValueError("quality_sigma has the wrong length")

    if not isinstance(max_iterations, Iterable):
        max_iterations = [max_iterations] * len(align_duts)
    # Finally check length of all arrays
    if len(max_iterations) != len(align_duts):  # empty iterable
        raise ValueError("max_iterations has the wrong length")

    if not isinstance(max_events, Iterable):
        max_events = [max_events] * len(align_duts)
    # Finally check length of all arrays
    if len(max_events) != len(align_duts):  # empty iterable
        raise ValueError("max_events has the wrong length")

    # Loop over all combinations of DUTs to align, simplest case: use all DUTs at once to align
    # Usual case: align high resolution devices first, then other devices
    for index, actual_align_duts in enumerate(align_duts):
        logging.info('Aligning DUTs: %s', ", ".join(str(dut) for dut in actual_align_duts))

        _duts_alignment(
            merged_file=input_merged_file,
            alignment_file=input_alignment_file,
            align_duts=actual_align_duts,
            select_telescope_duts=select_telescope_duts,
            select_fit_duts=select_fit_duts[index],
            select_hit_duts=select_hit_duts[index],
            quality_sigma=quality_sigma[index],
            alignment_order=alignment_order,
            dut_names=dut_names,
            n_pixels=n_pixels,
            pixel_size=pixel_size,
            max_events=max_events[index],
            max_iterations=max_iterations[index],
            fit_method=fit_method,
            min_track_distance=min_track_distance,
            beam_energy=beam_energy,
            material_budget=material_budget,
            use_fit_limits=use_fit_limits,
            plot=plot,
            chunk_size=chunk_size)


def _duts_alignment(merged_file, alignment_file, align_duts, select_telescope_duts, select_fit_duts, select_hit_duts, quality_sigma, alignment_order, dut_names, n_pixels, pixel_size, max_events, max_iterations, fit_method, min_track_distance, beam_energy, material_budget, use_fit_limits=False, plot=True, chunk_size=100000):  # Called for each list of DUTs to align
    alignment_duts = "_".join(str(dut) for dut in align_duts)
    alignment_duts_str = ", ".join(str(dut) for dut in align_duts)

    if alignment_order is None:
        alignment_order = [["alpha", "beta", "gamma", "translation_x", "translation_y", "translation_z"]]

    for iteration in range(max_iterations):
        for alignment_index, selected_alignment_parameters in enumerate(alignment_order):
            iteration_step = iteration * len(alignment_order) + alignment_index

#             if iteration == 0 or set(align_duts) & set(select_fit_duts):
#                 # recalculate tracks if DUT is fit DUT
#                 if iteration != 0:
#                     # remove temporary files before continuing
#                     os.remove(output_tracks_file)
#                     os.remove(output_selected_tracks_file)
            output_tracks_file = os.path.splitext(merged_file)[0] + '_tracks_aligned_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
            output_selected_tracks_file = os.path.splitext(merged_file)[0] + '_tracks_aligned_selected_tracks_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)

            # find tracks in the beginning and each time for telescope/fit DUTs
            # find tracks only once for non-fit/non-telescope DUTs
            if iteration == 0 or (set(align_duts) & set(select_fit_duts)):
                if iteration != 0:
                    os.remove(output_track_candidates_file)
                output_track_candidates_file = os.path.splitext(merged_file)[0] + '_track_candidates_aligned_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
                # use pre-alignment for fit/telescope DUT and first iteration step to find proper track candidates
                if iteration != 0 or not (set(align_duts) & set(select_fit_duts)):
                    use_prealignment = False
                    current_align_duts = align_duts
                else:
                    use_prealignment = True
                    current_align_duts = list(set(align_duts) - set(select_telescope_duts)) if use_prealignment else align_duts

                duts_selection = [list(set([dut]) | set(select_hit_duts)) for dut in current_align_duts]
                print "************* current_align_duts ****************", current_align_duts
                print "************* duts_selection ****************", duts_selection
                print "************* use pre-alignment for find_tracks()", use_prealignment
                logging.info('= Alignment step 0: Finding tracks for DUTs %s =', alignment_duts_str)
                find_tracks(input_merged_file=merged_file,
                            input_alignment_file=alignment_file,
                            output_track_candidates_file=output_track_candidates_file,
                            use_prealignment=use_prealignment,
                            correct_beam_alignment=True,
                            max_events=max_events)

            # Step 2: Fit tracks for all DUTs
            logging.info('= Alignment step 1 / iteration %d: Fitting tracks for DUTs %s =', iteration_step, alignment_duts_str)
            fit_tracks(input_track_candidates_file=output_track_candidates_file,
                       input_alignment_file=alignment_file,
                       output_tracks_file=output_tracks_file,
                       # max_events=max_events,
                       select_duts=current_align_duts,
                       dut_names=dut_names,
                       n_pixels=n_pixels,
                       pixel_size=pixel_size,
                       select_fit_duts=select_telescope_duts if use_prealignment else select_fit_duts,  # Only use selected DUTs for track fit
                       select_hit_duts=select_hit_duts,  # Only use selected duts
                       quality_sigma=quality_sigma,
                       exclude_dut_hit=False,  # For constrained residuals
                       use_prealignment=False,
                       plot=plot,
                       method=fit_method,
                       min_track_distance=min_track_distance,
                       beam_energy=beam_energy,
                       material_budget=material_budget,
                       chunk_size=chunk_size)

            logging.info('= Alignment step 2 / iteration %d: Selecting tracks for DUTs %s =', iteration_step, alignment_duts_str)
            data_selection.select_tracks(input_tracks_file=output_tracks_file,
                                         output_tracks_file=output_selected_tracks_file,
                                         select_duts=current_align_duts,
                                         duts_hit_selection=duts_selection,
                                         duts_quality_selection=duts_selection,
                                         # select good tracks und limit cluster shapes to 1x1, 1x2 and 2x1
                                         # occurrences of other cluster shapes do not matter much
                                         # TODO: sleect quality tracks for all telescope DUTs
                                         condition=['((cluster_shape_dut_{0} == 1) | (cluster_shape_dut_{0} == 3) | (cluster_shape_dut_{0} == 4))'.format(dut) for dut in current_align_duts],
                                         chunk_size=chunk_size)

            if set(align_duts) & set(select_fit_duts):
                track_angles_file = os.path.splitext(merged_file)[0] + '_tracks_angles_aligned_selected_tracks_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
                histogram_track_angle(input_tracks_file=output_selected_tracks_file,
                                      output_track_angle_file=track_angles_file,
                                      input_alignment_file=alignment_file,
                                      use_prealignment=False,
                                      select_duts=current_align_duts,
                                      dut_names=dut_names,
                                      n_bins="auto",
                                      plot=plot)
                with tb.open_file(track_angles_file, mode="r") as in_file_h5:
                    beam_alpha = -in_file_h5.root.Alpha_track_angle_hist.attrs.mean
                    beam_beta = -in_file_h5.root.Beta_track_angle_hist.attrs.mean
                beam_alignment = _create_beam_alignment_array()
                beam_alignment[0]['beam_alpha'] = beam_alpha
                beam_alignment[0]['beam_beta'] = beam_beta
                print "beam alignment from track angles", beam_alignment
                geometry_utils.save_beam_alignment_parameters(alignment_file=alignment_file,
                                                              beam_alignment=beam_alignment)
                os.remove(track_angles_file)

            if plot:
                output_residuals_file = os.path.splitext(merged_file)[0] + '_residuals_aligned_selected_tracks_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
                calculate_residuals(input_tracks_file=output_selected_tracks_file,
                                    input_alignment_file=alignment_file,
                                    output_residuals_file=output_residuals_file,
                                    dut_names=dut_names,
                                    n_pixels=n_pixels,
                                    pixel_size=pixel_size,
                                    use_prealignment=False,
                                    select_duts=current_align_duts,
                                    npixels_per_bin=None,
                                    nbins_per_pixel=None,
                                    use_fit_limits=use_fit_limits,
                                    plot=True,
                                    chunk_size=chunk_size)
                os.remove(output_residuals_file)

            logging.info('= Alignment step 3 / iteration %d: Calculating alignment for DUTs %s =', iteration_step, alignment_duts_str)
            new_alignment = calculate_transformation(input_tracks_file=output_selected_tracks_file,
                                                     input_alignment_file=alignment_file,
                                                     use_prealignment=False,
                                                     select_duts=current_align_duts,
                                                     use_fit_limits=use_fit_limits,
                                                     chunk_size=chunk_size)
            # Delete temporary files
            os.remove(output_tracks_file)
            os.remove(output_selected_tracks_file)
#             os.remove(os.path.splitext(merged_file)[0] + '_new_alignment_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step))
#             os.remove(os.path.splitext(merged_file)[0] + '_tracks_aligned_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step))
#             os.remove(os.path.splitext(merged_file)[0] + '_tracks_aligned_duts_%s_tmp_%d.pdf' % (alignment_duts, iteration_step)
#             os.remove(os.path.splitext(merged_file)[0] + '_residuals_aligned_reduced_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step))

            for align_dut in current_align_duts:
                print "save alignment parameters", list(set(selected_alignment_parameters) - set(["translation_x", "translation_y", "translation_z"])) if align_dut in select_telescope_duts else selected_alignment_parameters
                geometry_utils.save_alignment_parameters(alignment_file=alignment_file,
                                                         alignment_parameters=new_alignment,
                                                         mode='absolute',
                                                         select_duts=[align_dut],
                                                         parameters=list(set(selected_alignment_parameters) - set(["translation_x", "translation_y", "translation_z"])) if align_dut in select_telescope_duts else selected_alignment_parameters)

#     # remove temporary files
    os.remove(output_track_candidates_file)
#     os.remove(output_tracks_file)
#     os.remove(output_selected_tracks_file)

    if plot or set(align_duts) & set(select_fit_duts):
        final_track_candidates_file = os.path.splitext(merged_file)[0] + '_track_candidates_final_tmp_duts_%s.h5' % alignment_duts
        find_tracks(input_merged_file=merged_file,
                    input_alignment_file=alignment_file,
                    output_track_candidates_file=final_track_candidates_file,
                    use_prealignment=False,
                    correct_beam_alignment=True,
                    max_events=max_events)
        fit_tracks(input_track_candidates_file=final_track_candidates_file,
                   input_alignment_file=alignment_file,
                   output_tracks_file=os.path.splitext(merged_file)[0] + '_tracks_final_tmp_duts_%s.h5' % alignment_duts,
                   # max_events=max_events,
                   select_duts=align_duts,  # Only create residuals of selected DUTs
                   dut_names=dut_names,
                   n_pixels=n_pixels,
                   pixel_size=pixel_size,
                   select_fit_duts=select_fit_duts,  # Only use selected duts
                   select_hit_duts=select_hit_duts,
                   quality_sigma=quality_sigma,
                   exclude_dut_hit=True,  # For unconstrained residuals
                   use_prealignment=False,
                   plot=plot,
                   method=fit_method,
                   min_track_distance=min_track_distance,
                   beam_energy=beam_energy,
                   material_budget=material_budget,
                   chunk_size=chunk_size)

        data_selection.select_tracks(input_tracks_file=os.path.splitext(merged_file)[0] + '_tracks_final_tmp_duts_%s.h5' % alignment_duts,
                                     output_tracks_file=os.path.splitext(merged_file)[0] + '_tracks_final_selected_tracks_tmp_duts_%s.h5' % alignment_duts,
                                     select_duts=align_duts,
                                     duts_hit_selection=duts_selection,
                                     duts_quality_selection=duts_selection,
                                     condition=None,
                                     chunk_size=chunk_size)

        if set(align_duts) & set(select_fit_duts):
            track_angles_file = os.path.splitext(merged_file)[0] + '_tracks_angles_final_reduced_tmp_duts_%s.h5' % alignment_duts
            histogram_track_angle(input_tracks_file=os.path.splitext(merged_file)[0] + '_tracks_final_selected_tracks_tmp_duts_%s.h5' % alignment_duts,
                                  output_track_angle_file=track_angles_file,
                                  input_alignment_file=alignment_file,
                                  use_prealignment=False,
                                  select_duts=align_duts,
                                  dut_names=dut_names,
                                  n_bins="auto",
                                  plot=True)
            with tb.open_file(track_angles_file, mode="r") as in_file_h5:
                beam_alpha = -in_file_h5.root.Alpha_track_angle_hist.attrs.mean
                beam_beta = -in_file_h5.root.Beta_track_angle_hist.attrs.mean
            beam_alignment = _create_beam_alignment_array()
            beam_alignment[0]['beam_alpha'] = beam_alpha
            beam_alignment[0]['beam_beta'] = beam_beta
            print "beam alignment from track angles", beam_alignment
            geometry_utils.save_beam_alignment_parameters(alignment_file=alignment_file,
                                                          beam_alignment=beam_alignment)

        # Plotting final results
        if plot:
            calculate_residuals(input_tracks_file=os.path.splitext(merged_file)[0] + '_tracks_final_selected_tracks_tmp_duts_%s.h5' % alignment_duts,
                                input_alignment_file=alignment_file,
                                output_residuals_file=os.path.splitext(merged_file)[0] + '_residuals_final_duts_%s.h5' % alignment_duts,
                                select_duts=align_duts,
                                dut_names=dut_names,
                                n_pixels=n_pixels,
                                pixel_size=pixel_size,
                                use_prealignment=False,
                                use_fit_limits=use_fit_limits,
                                plot=plot,
                                chunk_size=chunk_size)

            # remove temporary files for plotting
#             os.remove(os.path.splitext(merged_file)[0] + '_final_tmp_duts_%s.h5' % alignment_duts)
        os.remove(final_track_candidates_file)
        os.remove(os.path.splitext(merged_file)[0] + '_tracks_final_tmp_duts_%s.h5' % alignment_duts)
#         os.remove(os.path.splitext(merged_file)[0] + '_tracks_final_tmp_duts_%s.pdf' % alignment_duts)
        os.remove(os.path.splitext(merged_file)[0] + '_tracks_final_selected_tracks_tmp_duts_%s.h5' % alignment_duts)
        os.remove(os.path.splitext(merged_file)[0] + '_residuals_final_duts_%s.h5' % alignment_duts)

    # remove temporary files
#     os.remove(os.path.splitext(track_candidates_reduced)[0] + '_not_aligned.h5')
    # keep file for testing
#     os.remove(track_candidates_reduced)


def calculate_transformation(input_tracks_file, input_alignment_file, select_duts, use_prealignment, use_fit_limits=True, chunk_size=1000000):
    '''Takes the tracks and calculates residuals for selected DUTs in col, row direction.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    input_alignment_file : string
        Filename of the input aligment file.
    select_duts : iterable
        Selecting DUTs that will be processed.
    use_prealignment : bool
        If True, use pre-alignment from correlation data; if False, use alignment.
    use_fit_limits : bool
        If True, use fit limits from pre-alignment for selecting fit range for the alignment.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    logging.info('=== Calculating residuals ===')

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        if use_prealignment:
            logging.info('Use pre-alignment data')
            prealignment = in_file_h5.root.PreAlignment[:]
            n_duts = prealignment.shape[0]
        else:
            logging.info('Use alignment data')
            alignment = in_file_h5.root.Alignment[:]
            n_duts = alignment.shape[0]
        if use_fit_limits:
            fit_limits = in_file_h5.root.PreAlignment.attrs.fit_limits

    euler_angles = [np.zeros(3, dtype=np.double)] * n_duts
    translation = [np.zeros(3, dtype=np.double)] * n_duts
    total_n_tracks = [0] * n_duts

    new_alignment = alignment.copy()

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        for actual_dut in select_duts:
            node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT_%d' % actual_dut)
            logging.debug('Calculate transformation for DUT%d', actual_dut)

            if use_fit_limits:
                fit_limit_x_local, fit_limit_y_local = fit_limits[actual_dut][0], fit_limits[actual_dut][1]
            else:
                fit_limit_x_local = None
                fit_limit_y_local = None

            actual_chunk = -1
            for tracks_chunk, _ in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                actual_chunk += 1
                # select good hits and tracks
                selection = np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut]), ~np.isnan(tracks_chunk['track_chi2']))
                tracks_chunk = tracks_chunk[selection]  # Take only tracks where actual dut has a hit, otherwise residual wrong

                # Coordinates in global coordinate system (x, y, z)
                hit_x_local, hit_y_local, hit_z_local = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
                slopes = np.column_stack([tracks_chunk['slope_0'], tracks_chunk['slope_1'], tracks_chunk['slope_2']])
                offsets = np.column_stack([tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']])

                if not np.allclose(hit_z_local, 0.0):
                    raise RuntimeError("Transformation into local coordinate system gives z != 0")

                limit_xy_local_sel = np.ones_like(hit_x_local, dtype=np.bool)
                if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local[0]):
                    limit_xy_local_sel &= hit_x_local >= fit_limit_x_local[0]
                if fit_limit_x_local is not None and np.isfinite(fit_limit_x_local[1]):
                    limit_xy_local_sel &= hit_x_local <= fit_limit_x_local[1]
                if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local[0]):
                    limit_xy_local_sel &= hit_y_local >= fit_limit_y_local[0]
                if fit_limit_y_local is not None and np.isfinite(fit_limit_y_local[1]):
                    limit_xy_local_sel &= hit_y_local <= fit_limit_y_local[1]

                hit_x_local = hit_x_local[limit_xy_local_sel]
                hit_y_local = hit_y_local[limit_xy_local_sel]
                hit_z_local = hit_z_local[limit_xy_local_sel]
                hit_local = np.column_stack([hit_x_local, hit_y_local, hit_z_local])
                slopes = slopes[limit_xy_local_sel]
                offsets = offsets[limit_xy_local_sel]
                n_tracks = np.count_nonzero(limit_xy_local_sel)

                x_dut_start = alignment[actual_dut]['translation_x']
                y_dut_start = alignment[actual_dut]['translation_y']
                z_dut_start = alignment[actual_dut]['translation_z']
                alpha_dut_start = alignment[actual_dut]['alpha']
                beta_dut_start = alignment[actual_dut]['beta']
                gamma_dut_start = alignment[actual_dut]['gamma']
                print "n_tracks for chunk %d:" % actual_chunk, n_tracks
                delta_t = 1.0  # TODO: optimize
                iterations = 100
                lin_alpha = 1.0

                print "alpha, beta, gamma at start", alpha_dut_start, beta_dut_start, gamma_dut_start
                print "translation at start", x_dut_start, y_dut_start, z_dut_start

                initialize_angles = True
                for i in range(iterations):
                    print "****************** iteration step %d DUT%d chunk %d *********************" % (i, actual_dut, actual_chunk)
                    if initialize_angles:
                        alpha, beta, gamma = alpha_dut_start, beta_dut_start, gamma_dut_start
                        rotation_dut = geometry_utils.rotation_matrix(alpha=alpha,
                                                                      beta=beta,
                                                                      gamma=gamma)
                        rotation_dut_old = np.identity(3, dtype=np.double)
                        translation_dut = np.array([x_dut_start, y_dut_start, z_dut_start])
                        translation_dut_old = np.array([0.0, 0.0, 0.0])
                        initialize_angles = False

#                     start_delta_t = 0.01
#                     stop_delta_t = 1.0
#                     delta_t = start_delta_t * np.exp(i * (np.log(stop_delta_t) - np.log(start_delta_t)) / iterations)
#                     print "delta_t", delta_t

                    tot_matr = None
                    tot_b = None
                    identity = np.identity(3, dtype=np.double)
                    n_identity = -np.identity(3, dtype=np.double)

                    # vectorized calculation of matrix and b vector
                    outer_prod_slopes = np.einsum('ij,ik->ijk', slopes, slopes)
                    l_matr = identity - outer_prod_slopes
                    R_y = np.matmul(rotation_dut, hit_local.T).T
                    skew_R_y = geometry_utils.skew(R_y)
                    tot_matr = np.concatenate([np.matmul(l_matr, skew_R_y), np.matmul(l_matr, n_identity)], axis=2)
                    tot_matr = tot_matr.reshape(-1, tot_matr.shape[2])
                    tot_b = np.matmul(l_matr, np.expand_dims(offsets, axis=2)) - np.matmul(l_matr, np.expand_dims(R_y + translation_dut, axis=2))
                    tot_b = tot_b.reshape(-1)

                    # iterative calculation of matrix and b vector
#                     for count in range(len(hit_x_local)):
#                         if count >= max_n_tracks:
#                             count = count - 1
#                             break
#                         slope = slopes[count]
#
#                         l_matr = identity - np.outer(slope, slope)
#                         p = offsets[count]
#                         y = hit_local[count]
#                         R_y = np.dot(rotation_dut, y)
#
#                         b = np.dot(l_matr, p) - np.dot(l_matr, (R_y + translation_dut))
#                         if tot_b is None:
#                             tot_b = b
#                         else:
#                             tot_b = np.hstack([tot_b, b])
#
#                         skew_R_y = geometry_utils.skew(R_y)
#
#                         matr = np.dot(l_matr, np.hstack([skew_R_y, n_identity]))
#                         if tot_matr is None:
#                             tot_matr = matr
#                         else:
#                             tot_matr = np.vstack([tot_matr, matr])

                    # SVD
                    print "calculating SVD..."
                    u, s, v = np.linalg.svd(tot_matr, full_matrices=False)
#                     u, s, v = svds(tot_matr, k=5)
                    print "...finished calculating SVD"

                    diag = np.diag(s ** -1)
                    tot_matr_inv = np.dot(v.T, np.dot(diag, u.T))
                    omega_b_dot = np.dot(tot_matr_inv, -lin_alpha * tot_b)

                    translation_dut_old = translation_dut
                    rotation_dut_old = rotation_dut
#                     alpha_old, beta_old, gamma_old = alpha, beta, gamma
#                     print "alpha, beta, gamma BEFORE calc", alpha, beta, gamma
                    rotation_dut = np.dot((np.identity(3, dtype=np.double) + delta_t * geometry_utils.skew(omega_b_dot[:3])), rotation_dut)

                    # apply UP (polar) decomposition to normalize/orthogonalize rotation matrix (det = 1)
                    u_rot, _, v_rot = np.linalg.svd(rotation_dut, full_matrices=False)
                    rotation_dut = np.dot(u_rot, v_rot)  # orthogonal matrix U

                    translation_dut = translation_dut + delta_t * omega_b_dot[3:]

#                     try:
#                         # TODO: do not use minimizer, use algebra
#                         alpha, beta, gamma = geometry_utils.euler_angles_minimizer(R=rotation_dut, alpha_start=alpha, beta_start=beta, gamma_start=gamma, limit=0.1)
#                         print "alpha, beta, gamma AFTER minimizer", alpha, beta, gamma
# #                         alpha, beta, gamma = geometry_utils.euler_angles(R=rotation_dut)
# #                         print "alpha, beta, gamma AFTER calc", alpha, beta, gamma
#                     except ValueError:
#                         update = False
#                         alpha, beta, gamma = alpha_old, beta_old, gamma_old
#                         rotation_dut = rotation_dut_old
#                         translation_dut = translation_dut_old
#                     else:
#                         update = True
#                         rotation_dut = geometry_utils.rotation_matrix(alpha=alpha,
#                                                                       beta=beta,
#                                                                       gamma=gamma)
#
#                     if update:
                    allclose_trans = np.allclose(translation_dut, translation_dut_old, rtol=0.0, atol=1e-2)
                    allclose_rot = np.allclose(rotation_dut, rotation_dut_old, rtol=1e-7, atol=0.0)
                    print "allclose rot trans", allclose_rot, allclose_trans
                    if allclose_rot and allclose_trans:
                        print "*** ALL CLOSE, BREAK ***"
                        break

                # TODO: do not use minimizer, use algebra
                alpha, beta, gamma = geometry_utils.euler_angles_minimizer(R=rotation_dut, alpha_start=alpha, beta_start=beta, gamma_start=gamma, limit=0.1)
                euler_angles_dut = np.array([alpha, beta, gamma], dtype=np.double)
                print "alpha, beta, gamma AFTER minimizer", alpha, beta, gamma
                fitted_rotation_dut = geometry_utils.rotation_matrix(alpha=alpha, beta=beta, gamma=gamma)
                if not np.allclose(rotation_dut, fitted_rotation_dut):
                    raise RuntimeError("Fit of alpha/beta/gamma returned no proper values.")
#                         alpha, beta, gamma = geometry_utils.euler_angles(R=rotation_dut)
#                         print "alpha, beta, gamma AFTER calc", alpha, beta, gamma

                print "euler_angles before average", euler_angles_dut, euler_angles[actual_dut]
                print "translation before average", translation_dut, translation[actual_dut]
                print "tracks in chunk", n_tracks
                euler_angles[actual_dut] = np.average([euler_angles_dut, euler_angles[actual_dut]], weights=[n_tracks, total_n_tracks[actual_dut]], axis=0)
                translation[actual_dut] = np.average([translation_dut, translation[actual_dut]], weights=[n_tracks, total_n_tracks[actual_dut]], axis=0)
                total_n_tracks[actual_dut] += n_tracks
                print "euler_angles average", euler_angles[actual_dut]
                print "translation average", translation[actual_dut]
                print "after total n tracks", total_n_tracks[actual_dut]

            new_alignment[actual_dut]['translation_x'] = translation[actual_dut][0]
            new_alignment[actual_dut]['translation_y'] = translation[actual_dut][1]
            new_alignment[actual_dut]['translation_z'] = translation[actual_dut][2]
            new_alignment[actual_dut]['alpha'] = euler_angles[actual_dut][0]
            new_alignment[actual_dut]['beta'] = euler_angles[actual_dut][1]
            new_alignment[actual_dut]['gamma'] = euler_angles[actual_dut][2]

    return new_alignment


# Helper functions for the alignment. Not to be used directly.
def _create_alignment_array(n_duts):
    # Result Translation / rotation table
    description = [('DUT', np.int32)]
    description.append(('translation_x', np.double))
    description.append(('translation_y', np.double))
    description.append(('translation_z', np.double))
    description.append(('alpha', np.double))
    description.append(('beta', np.double))
    description.append(('gamma', np.double))

    array = np.zeros((n_duts,), dtype=description)
    array[:]['DUT'] = np.array(range(n_duts))
    return array


def _create_beam_alignment_array():
    description = [('beam_alpha', np.double), ('beam_beta', np.double)]
    array = np.zeros((1,), dtype=description)
    return array
