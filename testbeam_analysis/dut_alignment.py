''' All DUT alignment functions in space and time are listed here plus additional alignment check functions'''
from __future__ import division

import logging
import sys
import os
from collections import Iterable
import math

import tables as tb
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import progressbar

from testbeam_analysis.telescope.telescope import Telescope
from testbeam_analysis.tools import analysis_utils
from testbeam_analysis.tools import plot_utils
from testbeam_analysis.tools import geometry_utils
from testbeam_analysis.tools import data_selection
from testbeam_analysis.track_analysis import find_tracks, fit_tracks, line_fit_3d
from testbeam_analysis.result_analysis import calculate_residuals, histogram_track_angle, get_angles


all_alignment_parameters = ["translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma"]


def apply_alignment(telescope_configuration, input_file, output_file=None, local_to_global=True, align_to_beam=False, chunk_size=1000000):
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
    input_file : string
        Filename of the input file (merged or tracks file).
    output_file : string
        Filename of the output file with the converted coordinates (merged or tracks file).
    local_to_global : bool
        If True, convert from local to global coordinates.
    align_to_beam : bool
        If True, use telescope aligment to align to the beam (beam along z axis).
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_file : string
        Filename of the output file with new coordinates.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Apply alignment to %d DUTs ===', n_duts)

    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ('_global_coordinates.h5' if local_to_global else '_local_coordinates.h5')

    def convert_data(dut, dut_index, node, conv, data):
        if isinstance(dut, Telescope):
            data['x_dut_%d' % dut_index], data['y_dut_%d' % dut_index], data['z_dut_%d' % dut_index] = conv(
                x=data['x_dut_%d' % dut_index],
                y=data['y_dut_%d' % dut_index],
                z=data['z_dut_%d' % dut_index],
                translation_x=dut.translation_x,
                translation_y=dut.translation_y,
                translation_z=dut.translation_z,
                rotation_alpha=dut.rotation_alpha,
                rotation_beta=dut.rotation_beta,
                rotation_gamma=dut.rotation_gamma)
        else:
            data['x_dut_%d' % dut_index], data['y_dut_%d' % dut_index], data['z_dut_%d' % dut_index] = conv(
                x=data['x_dut_%d' % dut_index],
                y=data['y_dut_%d' % dut_index],
                z=data['z_dut_%d' % dut_index])

        if "Tracks" in node.name:
            format_strings = ['offset_{dimension}_dut_{dut_index}']
            if "DUT%d" % dut_index in node.name:
                format_strings.extend(['offset_{dimension}'])
            for format_string in format_strings:
                if format_string.format(dimension='x', dut_index=dut_index) in node.dtype.names:
                    data[format_string.format(dimension='x', dut_index=dut_index)], data[format_string.format(dimension='y', dut_index=dut_index)], data[format_string.format(dimension='z', dut_index=dut_index)] = conv(
                        x=data[format_string.format(dimension='x', dut_index=dut_index)],
                        y=data[format_string.format(dimension='y', dut_index=dut_index)],
                        z=data[format_string.format(dimension='z', dut_index=dut_index)],
                        translation_x=dut.translation_x,
                        translation_y=dut.translation_y,
                        translation_z=dut.translation_z,
                        rotation_alpha=dut.rotation_alpha,
                        rotation_beta=dut.rotation_beta,
                        rotation_gamma=dut.rotation_gamma)

            format_strings = ['slope_{dimension}_dut_{dut_index}']
            if "DUT%d" % dut_index in node.name:
                format_strings.extend(['slope_{dimension}'])
            for format_string in format_strings:
                if format_string.format(dimension='x', dut_index=dut_index) in node.dtype.names:
                    data[format_string.format(dimension='x', dut_index=dut_index)], data[format_string.format(dimension='y', dut_index=dut_index)], data[format_string.format(dimension='z', dut_index=dut_index)] = conv(
                        x=data[format_string.format(dimension='x', dut_index=dut_index)],
                        y=data[format_string.format(dimension='y', dut_index=dut_index)],
                        z=data[format_string.format(dimension='z', dut_index=dut_index)],
                        # no translation for the slopes
                        translation_x=0.0,
                        translation_y=0.0,
                        translation_z=0.0,
                        rotation_alpha=dut.rotation_alpha,
                        rotation_beta=dut.rotation_beta,
                        rotation_gamma=dut.rotation_gamma)

            format_strings = ['{dimension}_err_dut_{dut_index}']
            for format_string in format_strings:
                if format_string.format(dimension='x', dut_index=dut_index) in node.dtype.names:
                    data[format_string.format(dimension='x', dut_index=dut_index)], data[format_string.format(dimension='y', dut_index=dut_index)], data[format_string.format(dimension='z', dut_index=dut_index)] = np.abs(conv(
                        x=data[format_string.format(dimension='x', dut_index=dut_index)],
                        y=data[format_string.format(dimension='y', dut_index=dut_index)],
                        z=data[format_string.format(dimension='z', dut_index=dut_index)],
                        # no translation for the errors
                        translation_x=0.0,
                        translation_y=0.0,
                        translation_z=0.0,
                        rotation_alpha=dut.rotation_alpha,
                        rotation_beta=dut.rotation_beta,
                        rotation_gamma=dut.rotation_gamma))

    # Looper over the hits of all DUTs of all hit tables in chunks and apply the alignment
    with tb.open_file(input_file, mode='r') as in_file_h5:
        with tb.open_file(output_file, mode='w') as out_file_h5:
            for node in in_file_h5.root:  # Loop over potential hit tables in data file
                logging.info('== Apply alignment to node %s ==', node.name)
                hits_aligned_table = out_file_h5.create_table(
                    where=out_file_h5.root,
                    name=node.name,
                    description=node.dtype,
                    title=node.title,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))

                progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=node.shape[0], term_width=80)
                progress_bar.start()

                for data_chunk, index in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):  # Loop over the hits
                    for dut_index, dut in enumerate(telescope):  # Loop over the DUTs
                        if local_to_global:
                            conv = dut.local_to_global_position
                        else:
                            conv = dut.global_to_local_position

                        if align_to_beam and not local_to_global:
                            convert_data(dut=telescope, dut_index=dut_index, node=node, conv=conv, data=data_chunk)
                        convert_data(dut=dut, dut_index=dut_index, node=node, conv=conv, data=data_chunk)
                        if align_to_beam and local_to_global:
                            convert_data(dut=telescope, dut_index=dut_index, node=node, conv=conv, data=data_chunk)
                    hits_aligned_table.append(data_chunk)
                    progress_bar.update(index)
                progress_bar.finish()

    return output_file


def prealign(telescope_configuration, input_correlation_file, output_telescope_configuration=None, select_reference_dut=0, reduce_background=True, use_location=False, plot=True, gui=False):
    '''Deduce a pre-alignment from the correlations, by fitting the correlations with a straight line (gives offset, slope, but no tild angles).
       The user can define cuts on the fit error and straight line offset in an interactive way.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_correlation_file : string
        Filename of the input correlation file.
    output_telescope_configuration : string
        Filename of the output telescope configuration file.
    select_reference_dut : uint
        DUT index of the reference plane. Default is DUT 0.
    reduce_background : bool
        If True, use correlation histograms with reduced background (by applying SVD method to the correlation matrix).
    plot : bool
        If True, create additional output plots.
    gui : bool
        If True, this function is excecuted from GUI and returns figures
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Pre-alignment of %d DUTs ===' % n_duts)

    if output_telescope_configuration is None:
        output_telescope_configuration = os.path.splitext(telescope_configuration)[0] + '_prealigned.yaml'

    if plot is True and not gui:
        output_pdf = PdfPages(os.path.splitext(input_correlation_file)[0] + '_prealigned.pdf', keep_empty=False)
    else:
        output_pdf = None

    figs = [] if gui else None

    with tb.open_file(input_correlation_file, mode="r") as in_file_h5:
        # remove reference DUT from list of DUTs
        select_duts = list(set(range(n_duts)) - set([select_reference_dut]))

        for dut_index in select_duts:
            x_global_pixel, y_global_pixel, z_global_pixel = [], [], []
            for column in range(1, telescope[dut_index].n_columns + 1):
                global_positions = telescope[dut_index].index_to_global_position(
                    column=[column] * telescope[dut_index].n_rows,
                    row=range(1, telescope[dut_index].n_rows + 1))
                x_global_pixel = np.hstack([x_global_pixel, global_positions[0]])
                y_global_pixel = np.hstack([y_global_pixel, global_positions[1]])
                z_global_pixel = np.hstack([z_global_pixel, global_positions[2]])
            for x_direction in [True, False]:
                if reduce_background:
                    node = in_file_h5.get_node(in_file_h5.root, 'Correlation_%s_%d_%d_reduced_background' % ('x' if x_direction else 'y', select_reference_dut, dut_index))
                else:
                    node = in_file_h5.get_node(in_file_h5.root, 'Correlation_%s_%d_%d' % ('x' if x_direction else 'y', select_reference_dut, dut_index))
                dut_name = telescope[dut_index].name
                ref_name = telescope[select_reference_dut].name
                logging.info('Pre-aligning data from %s', node.name)
                bin_size = node.attrs.resolution
                ref_hist_extent = node.attrs.ref_hist_extent
                ref_hist_size = (ref_hist_extent[1] - ref_hist_extent[0])
                dut_hist_extent = node.attrs.dut_hist_extent
                dut_hist_size = (dut_hist_extent[1] - dut_hist_extent[0])

                # retrieve data
                data = node[:]

                # Calculate the positions on the x axis
                dut_pos = np.linspace(start=dut_hist_extent[0] + bin_size / 2.0, stop=dut_hist_extent[1] - bin_size / 2.0, num=data.shape[0], endpoint=True)

                # calculate maximum per column
                max_select = np.argmax(data, axis=1)
                hough_data = np.zeros_like(data)
                hough_data[np.arange(data.shape[0]), max_select] = 1
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
                # check for non-zero values to improve speed
                count_nonzero = np.count_nonzero(accumulator)
                indices = np.vstack(largest_indices(accumulator, count_nonzero)).T
                for index in indices:
                    rho_idx, th_idx = index[0], index[1]
                    rho_val, theta_val = rho[rho_idx], theta[th_idx]
                    slope_idx, offset_idx = -np.cos(theta_val) / np.sin(theta_val), rho_val / np.sin(theta_val)
                    slope = slope_idx
                    offset = offset_idx * bin_size + ref_hist_extent[0] + 0.5 * bin_size
                    # check for proper slope
                    if np.isclose(slope, 1.0, rtol=0.0, atol=0.1) or np.isclose(slope, -1.0, rtol=0.0, atol=0.1):
                        break
                else:
                    raise RuntimeError('Cannot find %s correlation between %s and %s' % ("X" if x_direction else "Y", telescope[select_reference_dut].name, telescope[dut_index].name))
                # offset in the center of the pixel matrix
                offset_center = offset + slope * (0.5 * dut_hist_size - 0.5 * bin_size)
                offset_plot = offset - slope * dut_pos[0]
                _, _, _, x_list, y_list = find_ransac(
                    x=dut_pos[max_select != 0],
                    y=(max_select[max_select != 0] * bin_size - ref_hist_size / 2.0 + bin_size / 2.0),
                    threshold=bin_size)
                dut_pos_limit = [np.min(x_list), np.max(x_list)]

                plot_utils.plot_hough(
                    dut_pos=dut_pos,
                    data=hough_data,
                    accumulator=accumulator,
                    offset=offset_plot,
                    slope=slope,
                    dut_pos_limit=dut_pos_limit,
                    theta_edges=theta_edges,
                    rho_edges=rho_edges,
                    ref_hist_extent=ref_hist_extent,
                    dut_hist_extent=dut_hist_extent,
                    ref_name=ref_name,
                    dut_name=dut_name,
                    x_direction=x_direction,
                    reduce_background=reduce_background,
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs)

                if x_direction:
                    select = (x_global_pixel > dut_pos_limit[0]) & (x_global_pixel < dut_pos_limit[1])
                    if slope < 0.0:
                        if telescope[dut_index].rotation_beta <= 0.0:
                            rotation_beta = telescope[dut_index].rotation_beta + np.pi
                        else:
                            rotation_beta = telescope[dut_index].rotation_beta - np.pi
                    else:
                        rotation_beta = telescope[dut_index].rotation_beta
                    translation_x = offset_center
                else:
                    select &= (y_global_pixel > dut_pos_limit[0]) & (y_global_pixel < dut_pos_limit[1])
                    if slope < 0.0:
                        if telescope[dut_index].rotation_alpha <= 0.0:
                            rotation_alpha = telescope[dut_index].rotation_alpha + np.pi
                        else:
                            rotation_alpha = telescope[dut_index].rotation_alpha - np.pi
                    else:
                        rotation_alpha = telescope[dut_index].rotation_alpha
                    translation_y = offset_center
            # Calculate index of the limit before setting new alignment parameters
            # Use indices
            # indices = telescope[dut_index].global_position_to_index(
            #     x=x_global_pixel[select],
            #     y=y_global_pixel[select],
            #     z=z_global_pixel[select])
            # Use local coordinates
            local_coordinates = telescope[dut_index].global_to_local_position(
                x=x_global_pixel[select],
                y=y_global_pixel[select],
                z=z_global_pixel[select])
            # set new parameters
            # telescope[dut_index].column_limit = (min(indices[0]), max(indices[0]))
            # telescope[dut_index].row_limit = (min(indices[1]), max(indices[1]))
            telescope[dut_index].column_limit = (min(local_coordinates[0]), max(local_coordinates[0]))
            telescope[dut_index].row_limit = (min(local_coordinates[1]), max(local_coordinates[1]))
            telescope[dut_index].translation_x = translation_x
            telescope[dut_index].translation_y = translation_y
            telescope[dut_index].rotation_alpha = rotation_alpha
            telescope[dut_index].rotation_beta = rotation_beta

    telescope.save_configuration(configuration_file=output_telescope_configuration)

    if output_pdf is not None:
        output_pdf.close()

    if gui:
        return figs
    else:
        return output_telescope_configuration


def find_ransac(x, y, iterations=100, threshold=1.0, ratio=0.5):
    ''' RANSAC implementation

    Note
    ----
    Implementation from Alexey Abramov,
    https://salzis.wordpress.com/2014/06/10/robust-linear-model-estimation-using-ransac-python-implementation/

    Parameters
    ----------
    x : list
        X coordinates.
    y : list
        Y coordinates.
    iterations : int
        Maximum number of iterations.
    threshold : float
        Maximum distance of the data points for inlier selection.
    ration : float
        Break condition for inliers.

    Returns
    -------
    model_ratio : float
        Ration of inliers to outliers.
    model_m : float
        Slope.
    model_c : float
        Offset.
    model_x_list : array
        X coordianates of inliers.
    model_y_list :  array
        X coordianates of inliers.
    '''

    def find_line_model(points):
        """ find a line model for the given points
        :param points selected points for model fitting
        :return line model
        """

        # [WARNING] vertical and horizontal lines should be treated differently
        #           here we just add some noise to avoid division by zero

        # find a line model for these points
        m = (points[1, 1] - points[0, 1]) / (points[1, 0] - points[0, 0] + sys.float_info.epsilon)  # slope (gradient) of the line
        c = points[1, 1] - m * points[1, 0]  # y-intercept of the line

        return m, c

    def find_intercept_point(m, c, x0, y0):
        """ find an intercept point of the line model with
            a normal from point (x0,y0) to it
        :param m slope of the line model
        :param c y-intercept of the line model
        :param x0 point's x coordinate
        :param y0 point's y coordinate
        :return intercept point
        """

        # intersection point with the model
        x = (x0 + m * y0 - m * c) / (1 + m**2)
        y = (m * x0 + (m**2) * y0 - (m**2) * c) / (1 + m**2) + c

        return x, y

    data = np.column_stack((x, y))
    n_samples = x.shape[0]

    model_ratio = 0.0
    model_m = 0.0
    model_c = 0.0

    # perform RANSAC iterations
    for it in range(iterations):
        all_indices = np.arange(n_samples)
        np.random.shuffle(all_indices)

        indices_1 = all_indices[:2]  # pick up two random points
        indices_2 = all_indices[2:]

        maybe_points = data[indices_1, :]
        test_points = data[indices_2, :]

        # find a line model for these points
        m, c = find_line_model(maybe_points)

        x_list = []
        y_list = []
        num = 0

        # find orthogonal lines to the model for all testing points
        for ind in range(test_points.shape[0]):

            x0 = test_points[ind, 0]
            y0 = test_points[ind, 1]

            # find an intercept point of the model with a normal from point (x0,y0)
            x1, y1 = find_intercept_point(m, c, x0, y0)

            # distance from point to the model
            dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)

            # check whether it's an inlier or not
            if dist < threshold:
                x_list.append(x0)
                y_list.append(y0)
                num += 1

        # in case a new model is better - cache it
        if num / float(n_samples) > model_ratio:
            model_ratio = num / float(n_samples)
            model_m = m
            model_c = c
            model_x_list = np.array(x_list)
            model_y_list = np.array(y_list)

        # we are done in case we have enough inliers
        if num > n_samples * ratio:
            break

    return model_ratio, model_m, model_c, model_x_list, model_y_list


def align(telescope_configuration, input_merged_file, output_telescope_configuration=None, align_duts=None, alignment_parameters=None, select_telescope_duts=None, select_fit_duts=None, select_hit_duts=None, max_iterations=3, max_events=100000, fit_method='fit', beam_energy=None, particle_mass=None, scattering_planes=None, track_chi2=10.0, quality_distances=(250.0, 250.0), reject_quality_distances=(500.0, 500.0), use_limits=True, plot=True, chunk_size=100000):
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
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_merged_file : string
        Filename of the input merged file.
    output_telescope_configuration : string
        Filename of the output telescope configuration file.
    align_duts : iterable or iterable of iterable
        The combination of duts that are algined at once. One should always align the high resolution planes first.
        E.g. for a telesope (first and last 3 planes) with 2 devices in the center (3, 4):
        align_duts=[[0, 1, 2, 5, 6, 7],  # align the telescope planes first
                    [4],  # align first DUT
                    [3]]  # align second DUT
    alignment_parameters : list of lists of strings
        The list of alignment parameters for each align_dut. Valid parameters:
        - translation_x: horizontal axis
        - translation_y: vertical axis
        - translation_z: beam axis
        - rotation_alpha: rotation around x-axis
        - rotation_beta: rotation around y-axis
        - rotation_gamma: rotation around z-axis (beam axis)
        If None, all paramters will be selected.
    select_telescope_duts : iterable
        The given DUTs will be used to align the telescope along the z-axis.
        Usually the coordinates of these DUTs are well specified.
        At least 2 DUTs need to be specified. The z-position of the selected DUTs will not be changed by default.
    select_fit_duts : iterable or iterable of iterable
        Defines for each align_duts combination wich devices to use in the track fit.
        E.g. To use only the telescope planes (first and last 3 planes) but not the 2 center devices
        select_fit_duts=[0, 1, 2, 5, 6, 7]
    select_hit_duts : iterable or iterable of iterable
        Defines for each align_duts combination wich devices must have a hit to use the track for fitting. The hit
        does not have to be used in the fit itself! This is useful for time reference planes.
        E.g.  To use telescope planes (first and last 3 planes) + time reference plane (3)
        select_hit_duts = [0, 1, 2, 4, 5, 6, 7]
    max_iterations : uint
        Maximum number of iterations of calc residuals, apply rotation refit loop until constant result is expected.
        Usually the procedure converges rather fast (< 5 iterations).
    max_events: uint
        Radomly select max_events for alignment. If None, use all events, which might slow down the alignment.
    fit_method : string
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
    track_chi2 : float or list
        Setting the limit on the track chi^2. If None or 0.0, no cut will be applied.
        A smaller value reduces the number of tracks for the alignment.
        A large value increases the number of tracks but at the cost of alignment efficiency bacause of potentially bad tracks.
        A good start value is 5.0 to 10.0 for high energy beams and 15.0 to 50.0 for low energy beams.
    quality_distances : 2-tuple or list of 2-tuples
        X and y distance (in um) for each DUT to calculate the quality flag. The selected track and corresponding hit
        must have a smaller distance to have the quality flag to be set to 1.
        The purpose of quality_distances is to find good tracks for the alignment.
        A good start value is 1-2x the pixel pitch for large pixels and high-energy beams and 5-10x the pixel pitch for small pixels and low-energy beams.
        A too small value will remove good tracks, a too large value will allow bad tracks. A cut on the track chi^2 will have a similar effect.
        If None, use infinite distance.
    reject_quality_distances : 2-tuple or list of 2-tuples
        X and y distance (in um) for each DUT to calculate the quality flag. Any other occurence of tracks or hits from the same event
        within this distance will reject the quality flag.
        The purpose of reject_quality_distances is to remove tracks from alignment that could be potentially fake tracks (noisy detector / high beam density).
        If None, use infinite distance.
    use_limits : bool
        If True, use column and row limits from pre-alignment for selecting the data.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Alignment of %d DUTs ===' % len(set(np.unique(np.hstack(np.array(align_duts))).tolist())))

    if output_telescope_configuration is None:
        if 'prealigned' in telescope_configuration:
            output_telescope_configuration = telescope_configuration.replace('prealigned', 'aligned')
        else:
            output_telescope_configuration = os.path.splitext(telescope_configuration)[0] + '_aligned.yaml'
    elif output_telescope_configuration == telescope_configuration:
        raise ValueError('Output configuration file must be different from input configuration file.')
    if os.path.isfile(output_telescope_configuration):
        logging.info('Configuration file already exists.')
    else:
        telescope.save_configuration(configuration_file=output_telescope_configuration)

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
        logging.info('These DUTs will not be aligned: %s' % ", ".join(telescope[dut_index].name for dut_index in no_align_duts))

    # Create list
    if alignment_parameters is None:
        alignment_parameters = [[None] * len(duts) for duts in align_duts]
    # Check for value errors
    if not isinstance(alignment_parameters, Iterable):
        raise ValueError("alignment_parameters is no iterable")
    elif not alignment_parameters:  # empty iterable
        raise ValueError("alignment_parameters has no items")
    # Finally check length of all arrays
    if len(alignment_parameters) != len(align_duts):  # empty iterable
        raise ValueError("alignment_parameters has the wrong length")
    for index, alignment_parameter in enumerate(alignment_parameters):
        if alignment_parameter is None:
            alignment_parameters[index] = [None] * len(align_duts[index])
        if len(alignment_parameters[index]) != len(align_duts[index]):  # check the length of the items
            raise ValueError("item in alignment_parameter wrong length")

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

    if not isinstance(track_chi2, Iterable):
        track_chi2 = [track_chi2] * len(align_duts)
    # Finally check length of all arrays
    if len(track_chi2) != len(track_chi2):  # empty iterable
        raise ValueError("track_chi2 has the wrong length")

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
        logging.info('== Aligning %d DUTs: %s ==', len(actual_align_duts), ", ".join(telescope[dut_index].name for dut_index in actual_align_duts))

        _duts_alignment(
            input_telescope_configuration=telescope_configuration,  # pre-aligned configuration
            output_telescope_configuration=output_telescope_configuration,  # aligned configuration
            merged_file=input_merged_file,
            align_duts=actual_align_duts,
            alignment_parameters=alignment_parameters[index],
            select_telescope_duts=select_telescope_duts,
            select_fit_duts=select_fit_duts[index],
            select_hit_duts=select_hit_duts[index],
            max_iterations=max_iterations[index],
            max_events=max_events[index],
            fit_method=fit_method,
            beam_energy=beam_energy,
            particle_mass=particle_mass,
            scattering_planes=scattering_planes,
            track_chi2=track_chi2[index],
            quality_distances=quality_distances,
            reject_quality_distances=reject_quality_distances,
            use_limits=use_limits,
            plot=plot,
            chunk_size=chunk_size)

    return output_telescope_configuration


def _duts_alignment(input_telescope_configuration, output_telescope_configuration, merged_file, align_duts, alignment_parameters, select_telescope_duts, select_fit_duts, select_hit_duts, max_iterations, max_events, fit_method, beam_energy, particle_mass, scattering_planes, track_chi2, quality_distances, reject_quality_distances, use_limits, plot=True, chunk_size=100000):  # Called for each list of DUTs to align
    alignment_duts = "_".join(str(dut) for dut in align_duts)
    alignment_duts_str = ", ".join(str(dut) for dut in align_duts)
    output_prealigned_track_candidates_file = os.path.splitext(merged_file)[0] + '_track_candidates_prealigned.h5'
    prealigned_telescope = Telescope(configuration_file=input_telescope_configuration)
    aligned_telescope = Telescope(configuration_file=output_telescope_configuration)
    for dut in align_duts:
        aligned_telescope[dut] = prealigned_telescope[dut]
    aligned_telescope.save_configuration()
    find_tracks(
        telescope_configuration=input_telescope_configuration,
        input_merged_file=merged_file,
        output_track_candidates_file=output_prealigned_track_candidates_file,
        align_to_beam=True,
        max_events=max_events)
    output_track_candidates_file = None
    iteration_steps = range(max_iterations)
    for iteration_step in iteration_steps:
        # find tracks in the beginning and each iteration step when aligning fit DUTs
        if set(align_duts) & set(select_telescope_duts):
            if iteration_step == 0:
                align_telescope(
                    telescope_configuration=output_telescope_configuration,
                    select_telescope_duts=select_telescope_duts)
            # aligning the fit DUTs needs some adjustments to the parameters
            actual_align_duts = align_duts
            actual_fit_duts = select_fit_duts
            # reqire hits in each DUT that will be aligned
            actual_hit_duts = [list(set(select_hit_duts) | set([dut_index])) for dut_index in actual_align_duts]
            actual_quality_duts = actual_hit_duts
        # Regular case for non-fit DUTs
        else:
            # aligning non-fit DUTs needs some adjustments to the parameters
            actual_align_duts = align_duts
            actual_fit_duts = select_fit_duts
            # reqire hits in each DUT that will be aligned
            actual_hit_duts = [list(set(select_hit_duts) | set([dut_index])) for dut_index in actual_align_duts]
            actual_quality_duts = actual_hit_duts
        fit_quality_distances = np.zeros_like(quality_distances)
        for index, item in enumerate(quality_distances):
            if index in align_duts:
                fit_quality_distances[index, 0] = np.linspace(max(item) * max_iterations, item[0], max_iterations)[iteration_step]
                fit_quality_distances[index, 1] = np.linspace(max(item) * max_iterations, item[1], max_iterations)[iteration_step]
            else:
                fit_quality_distances[index, 0] = item[0]
                fit_quality_distances[index, 1] = item[1]
        fit_quality_distances = fit_quality_distances.tolist()
        logging.info('= Alignment step 1 - iteration %d: Finding tracks for DUTs %s =', iteration_step, alignment_duts_str)
        if iteration_step > 0:
            # remove temporary file
            if output_track_candidates_file is not None:
                os.remove(output_track_candidates_file)
            output_track_candidates_file = os.path.splitext(merged_file)[0] + '_track_candidates_aligned_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
            find_tracks(
                telescope_configuration=output_telescope_configuration,
                input_merged_file=merged_file,
                output_track_candidates_file=output_track_candidates_file,
                align_to_beam=True,
                max_events=max_events)

        # The quality flag of the actual align DUT depends on the alignment calculated
        # in the previous iteration, therefore this step has to be done every time
        logging.info('= Alignment step 2 - iteration %d: Fitting tracks for DUTs %s =', iteration_step, alignment_duts_str)
        output_tracks_file = os.path.splitext(merged_file)[0] + '_tracks_aligned_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
        fit_tracks(
            telescope_configuration=output_telescope_configuration,
            input_track_candidates_file=output_prealigned_track_candidates_file if iteration_step == 0 else output_track_candidates_file,
            output_tracks_file=output_tracks_file,
            select_duts=actual_align_duts,
            select_fit_duts=actual_fit_duts,
            select_hit_duts=actual_hit_duts,
            exclude_dut_hit=False,  # For biased residuals
            method=fit_method,
            beam_energy=beam_energy,
            particle_mass=particle_mass,
            scattering_planes=scattering_planes,
            quality_distances=fit_quality_distances,
            reject_quality_distances=reject_quality_distances,
            use_limits=use_limits,
            plot=plot,
            chunk_size=chunk_size)

        logging.info('= Alignment step 3a - iteration %d: Selecting tracks for DUTs %s =', iteration_step, alignment_duts_str)
        output_selected_tracks_file = os.path.splitext(merged_file)[0] + '_tracks_aligned_selected_tracks_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
        # TODO: adding detection if cluster shape is always -1 (not set)
        # Select good tracks und limit cluster shapes to 1x1, 1x2, 2x1, and 2x2.
        # Occurrences of other cluster shapes should not be included.
        # if np.all(cluster_shape == -1):
        #     select_condition = None
        # else:
        #     select_condition = ['((cluster_shape_dut_{0} == 1) | (cluster_shape_dut_{0} == 3) | (cluster_shape_dut_{0} == 5) | (cluster_shape_dut_{0} == 15))'.format(dut_index) for dut_index in actual_align_duts]
        data_selection.select_tracks(
            telescope_configuration=output_telescope_configuration,
            input_tracks_file=output_tracks_file,
            output_tracks_file=output_selected_tracks_file,
            select_duts=actual_align_duts,
            select_hit_duts=actual_hit_duts,
            select_quality_duts=actual_quality_duts,
            # Select good tracks und limit cluster size to 4
            condition=[(('(track_chi2 < %f) & ' % track_chi2) if track_chi2 else '') + '(n_hits_dut_{0} <= 4)'.format(dut_index) for dut_index in actual_align_duts],
            max_events=None,
            chunk_size=chunk_size)

        # if fit DUTs were aligned, update telescope alignment
        if set(align_duts) & set(select_fit_duts):
            logging.info('= Alignment step 3b - iteration %d: Aligning telescope =', iteration_step)
            output_track_angles_file = os.path.splitext(merged_file)[0] + '_tracks_angles_aligned_selected_tracks_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
            histogram_track_angle(
                telescope_configuration=output_telescope_configuration,
                input_tracks_file=output_selected_tracks_file,
                output_track_angle_file=output_track_angles_file,
                select_duts=actual_align_duts,
                n_bins="auto",
                plot=False)
            # Read and store beam angle to improve track finding
            if (set(align_duts) & set(select_fit_duts)):
                with tb.open_file(output_track_angles_file, mode="r") as in_file_h5:
                    if not np.isnan(in_file_h5.root.Global_alpha_track_angle_hist.attrs.mean) and not np.isnan(in_file_h5.root.Global_beta_track_angle_hist.attrs.mean):
                        aligned_telescope = Telescope(configuration_file=output_telescope_configuration)
                        aligned_telescope.rotation_alpha = in_file_h5.root.Global_alpha_track_angle_hist.attrs.mean
                        aligned_telescope.rotation_beta = in_file_h5.root.Global_beta_track_angle_hist.attrs.mean
                        aligned_telescope.save_configuration()
                    else:
                        logging.warning("Cannot read track angle histograms, track finding might be spoiled")
            os.remove(output_track_angles_file)

        if plot:
            logging.info('= Alignment step 3c - iteration %d: Calculating residuals =', iteration_step)
            output_residuals_file = os.path.splitext(merged_file)[0] + '_residuals_aligned_selected_tracks_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
            calculate_residuals(
                telescope_configuration=output_telescope_configuration,
                input_tracks_file=output_selected_tracks_file,
                output_residuals_file=output_residuals_file,
                select_duts=actual_align_duts,
                use_limits=use_limits,
                plot=True,
                chunk_size=chunk_size)
            os.remove(output_residuals_file)

        logging.info('= Alignment step 4 - iteration %d: Calculating transformation matrix for DUTs %s =', iteration_step, alignment_duts_str)
        calculate_transformation(
            telescope_configuration=output_telescope_configuration,
            input_tracks_file=output_selected_tracks_file,
            select_duts=actual_align_duts,
            select_alignment_parameters=[(["translation_x", "translation_y", "rotation_alpha", "rotation_beta", "rotation_gamma"] if (dut_index in select_telescope_duts and alignment_parameters[i] is None) else (all_alignment_parameters if alignment_parameters[i] is None else alignment_parameters[i])) for i, dut_index in enumerate(actual_align_duts)],
            use_limits=use_limits,
            max_iterations=100,
            chunk_size=chunk_size)

        # Delete temporary files
        os.remove(output_tracks_file)
        os.remove(output_selected_tracks_file)

    # Delete temporary files
    os.remove(output_prealigned_track_candidates_file)
    if output_track_candidates_file is not None:
        os.remove(output_track_candidates_file)

    # if plot or set(align_duts) & set(select_fit_duts):
    #     final_track_candidates_file = os.path.splitext(merged_file)[0] + '_track_candidates_final_tmp_duts_%s.h5' % alignment_duts
    #     find_tracks(
    #         telescope_configuration=output_telescope_configuration,
    #         input_merged_file=merged_file,
    #         output_track_candidates_file=final_track_candidates_file,
    #         align_to_beam=True,
    #         max_events=max_events)

    #     final_tracks_file = os.path.splitext(merged_file)[0] + '_tracks_final_tmp_duts_%s.h5' % alignment_duts
    #     fit_tracks(
    #         telescope_configuration=output_telescope_configuration,
    #         input_track_candidates_file=final_track_candidates_file,
    #         output_tracks_file=final_tracks_file,
    #         select_duts=align_duts,  # Only create residuals of selected DUTs
    #         select_fit_duts=select_fit_duts,  # Only use selected duts
    #         select_hit_duts=select_hit_duts,
    #         exclude_dut_hit=True,  # For unbiased residuals for the final plots
    #         method=fit_method,
    #         beam_energy=beam_energy,
    #         particle_mass=particle_mass,
    #         scattering_planes=scattering_planes,
    #         quality_distances=fit_quality_distances,
    #         reject_quality_distances=reject_quality_distances,
    #         plot=plot,
    #         chunk_size=chunk_size)

    #     final_selected_tracks_file = os.path.splitext(merged_file)[0] + '_selected_tracks_final_tmp_duts_%s.h5' % alignment_duts
    #     data_selection.select_tracks(
    #         input_tracks_file=final_tracks_file,
    #         output_tracks_file=final_selected_tracks_file,
    #         select_duts=align_duts,
    #         duts_hit_selection=duts_selection,
    #         duts_quality_selection=duts_selection,
    #         condition=None,
    #         chunk_size=chunk_size)

    #     if set(align_duts) & set(select_fit_duts):
    #         track_angles_file = os.path.splitext(merged_file)[0] + '_tracks_angles_final_reduced_tmp_duts_%s.h5' % alignment_duts
    #         histogram_track_angle(
    #             telescope_configuration=output_telescope_configuration,
    #             input_tracks_file=final_selected_tracks_file,
    #             output_track_angle_file=track_angles_file,
    #             select_duts=align_duts,
    #             n_bins="auto",
    #             plot=plot)

    #     # Plotting final results
    #     if plot:
    #         final_residuals_file = os.path.splitext(merged_file)[0] + '_residuals_final_duts_%s.h5' % alignment_duts
    #         calculate_residuals(
    #             telescope_configuration=output_telescope_configuration,
    #             input_tracks_file=final_selected_tracks_file,
    #             output_residuals_file=final_residuals_file,
    #             select_duts=align_duts,
    #             use_limits=use_limits,
    #             plot=plot,
    #             chunk_size=chunk_size)

    #     # Delete temporary files
    #     os.remove(final_track_candidates_file)
    #     os.remove(final_tracks_file)
    #     os.remove(final_selected_tracks_file)
    #     os.remove(final_residuals_file)


def align_telescope(telescope_configuration, select_telescope_duts, reference_dut=None):
    telescope = Telescope(telescope_configuration)
    logging.info('=== Alignment of the telescope ===')

    print "align telescope rotation"
    print telescope
    telescope_duts_positions = np.full((len(select_telescope_duts), 3), fill_value=np.nan, dtype=np.float64)
    for index, dut_index in enumerate(select_telescope_duts):
        telescope_duts_positions[index, 0] = telescope[dut_index].translation_x
        telescope_duts_positions[index, 1] = telescope[dut_index].translation_y
        telescope_duts_positions[index, 2] = telescope[dut_index].translation_z
    if reference_dut is not None:
        first_telescope_dut_index = reference_dut
    else:
        first_telescope_dut_index = select_telescope_duts[np.argmin(telescope_duts_positions[:, 2])]
    offset, slope = line_fit_3d(positions=telescope_duts_positions)
    first_telescope_dut = telescope[first_telescope_dut_index]
    first_dut_translation_x = first_telescope_dut.translation_x
    first_dut_translation_y = first_telescope_dut.translation_y

    first_telescope_dut_intersection = geometry_utils.get_line_intersections_with_dut(
        line_origins=offset[np.newaxis, :],
        line_directions=slope[np.newaxis, :],
        translation_x=first_telescope_dut.translation_x,
        translation_y=first_telescope_dut.translation_y,
        translation_z=first_telescope_dut.translation_z,
        rotation_alpha=first_telescope_dut.rotation_alpha,
        rotation_beta=first_telescope_dut.rotation_beta,
        rotation_gamma=first_telescope_dut.rotation_gamma)

    for actual_dut in telescope:
        dut_intersection = geometry_utils.get_line_intersections_with_dut(
            line_origins=offset[np.newaxis, :],
            line_directions=slope[np.newaxis, :],
            translation_x=actual_dut.translation_x,
            translation_y=actual_dut.translation_y,
            translation_z=actual_dut.translation_z,
            rotation_alpha=actual_dut.rotation_alpha,
            rotation_beta=actual_dut.rotation_beta,
            rotation_gamma=actual_dut.rotation_gamma)
        actual_dut.translation_x -= (dut_intersection[0, 0] - first_telescope_dut_intersection[0, 0] + first_dut_translation_x)
        actual_dut.translation_y -= (dut_intersection[0, 1] - first_telescope_dut_intersection[0, 1] + first_dut_translation_y)

    # set telescope alpha/beta rotation for a better beam alignment and track finding improvement
    # this is compensating the previously made changes to the DUT coordinates
    total_angles, alpha_angles, beta_angles = get_angles(
        slopes=slope[np.newaxis, :],
        xz_plane_normal=np.array([0.0, 1.0, 0.0]),
        yz_plane_normal=np.array([1.0, 0.0, 0.0]),
        dut_plane_normal=np.array([0.0, 0.0, 1.0]))
    telescope.rotation_alpha -= alpha_angles[0]
    telescope.rotation_beta -= beta_angles[0]
    print "\n", telescope, "\n"

    telescope.save_configuration()


def calculate_transformation(telescope_configuration, input_tracks_file, select_duts, select_alignment_parameters=None, use_limits=True, max_iterations=None, chunk_size=1000000):
    '''Takes the tracks and calculates and stores the transformation parameters.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_tracks_file : string
        Filename of the input tracks file.
    select_duts : list
        Selecting DUTs that will be processed.
    select_alignment_parameters : list
        Selecting the transformation parameters that will be stored to the telescope configuration file for each selected DUT.
        If None, all 6 transformation parameters will be calculated.
    use_limits : bool
        If True, use column and row limits from pre-alignment for selecting the data.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    logging.info('== Calculating transformation for %d DUTs ==' % len(select_duts))

    if select_alignment_parameters is None:
        select_alignment_parameters = [all_alignment_parameters] * len(select_duts)
    if len(select_duts) != len(select_alignment_parameters):
        raise ValueError("select_alignment_parameters has the wrong length")
    for index, actual_alignment_parameters in enumerate(select_alignment_parameters):
        if actual_alignment_parameters is None:
            select_alignment_parameters[index] = all_alignment_parameters
        else:
            non_valid_paramters = set(actual_alignment_parameters) - set(all_alignment_parameters)
            if non_valid_paramters:
                raise ValueError("found invalid values in select_alignment_parameters: %s" % ", ".join(non_valid_paramters))

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        for index, actual_dut_index in enumerate(select_duts):
            actual_dut = telescope[actual_dut_index]
            node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)
            logging.debug('== Calculate transformation for %s ==', actual_dut.name)

            if use_limits:
                limit_x_local = actual_dut.column_limit
                limit_y_local = actual_dut.row_limit
            else:
                limit_x_local = None
                limit_y_local = None

            rotation_average = None
            translation_average = None
            # euler_angles_average = None

            # calculate equal chunk size
            start_val = max(int(node.nrows / chunk_size), 2)
            while True:
                chunk_indices = np.linspace(0, node.nrows, start_val).astype(np.int)
                if np.all(np.diff(chunk_indices) <= chunk_size):
                    break
                start_val += 1
            chunk_index = 0
            n_tracks = 0
            total_n_tracks = 0
            while chunk_indices[chunk_index] < node.nrows:
                tracks_chunk = node.read(start=chunk_indices[chunk_index], stop=chunk_indices[chunk_index + 1])
                # select good hits and tracks
                selection = np.logical_and(~np.isnan(tracks_chunk['x_dut_%d' % actual_dut_index]), ~np.isnan(tracks_chunk['track_chi2']))
                tracks_chunk = tracks_chunk[selection]  # Take only tracks where actual dut has a hit, otherwise residual wrong

                # Coordinates in global coordinate system (x, y, z)
                hit_x_local, hit_y_local, hit_z_local = tracks_chunk['x_dut_%d' % actual_dut_index], tracks_chunk['y_dut_%d' % actual_dut_index], tracks_chunk['z_dut_%d' % actual_dut_index]

                offsets = np.column_stack(actual_dut.local_to_global_position(
                    x=tracks_chunk['offset_x'],
                    y=tracks_chunk['offset_y'],
                    z=tracks_chunk['offset_z']))

                slopes = np.column_stack(actual_dut.local_to_global_position(
                    x=tracks_chunk['slope_x'],
                    y=tracks_chunk['slope_y'],
                    z=tracks_chunk['slope_z'],
                    translation_x=0.0,
                    translation_y=0.0,
                    translation_z=0.0,
                    rotation_alpha=actual_dut.rotation_alpha,
                    rotation_beta=actual_dut.rotation_beta,
                    rotation_gamma=actual_dut.rotation_gamma))

                if not np.allclose(hit_z_local, 0.0):
                    raise RuntimeError("Transformation into local coordinate system gives z != 0")

                limit_xy_local_sel = np.ones_like(hit_x_local, dtype=np.bool)
                if limit_x_local is not None and np.isfinite(limit_x_local[0]):
                    limit_xy_local_sel &= hit_x_local >= limit_x_local[0]
                if limit_x_local is not None and np.isfinite(limit_x_local[1]):
                    limit_xy_local_sel &= hit_x_local <= limit_x_local[1]
                if limit_y_local is not None and np.isfinite(limit_y_local[0]):
                    limit_xy_local_sel &= hit_y_local >= limit_y_local[0]
                if limit_y_local is not None and np.isfinite(limit_y_local[1]):
                    limit_xy_local_sel &= hit_y_local <= limit_y_local[1]

                hit_x_local = hit_x_local[limit_xy_local_sel]
                hit_y_local = hit_y_local[limit_xy_local_sel]
                hit_z_local = hit_z_local[limit_xy_local_sel]
                hit_local = np.column_stack([hit_x_local, hit_y_local, hit_z_local])
                slopes = slopes[limit_xy_local_sel]
                offsets = offsets[limit_xy_local_sel]
                n_tracks = np.count_nonzero(limit_xy_local_sel)

                x_dut_start = actual_dut.translation_x
                y_dut_start = actual_dut.translation_y
                z_dut_start = actual_dut.translation_z
                alpha_dut_start = actual_dut.rotation_alpha
                beta_dut_start = actual_dut.rotation_beta
                gamma_dut_start = actual_dut.rotation_gamma
                print "n_tracks for chunk %d:" % chunk_index, n_tracks
                delta_t = 0.9  # TODO: optimize
                if max_iterations is None:
                    iterations = 100
                else:
                    iterations = max_iterations
                lin_alpha = 1.0

                print "alpha, beta, gamma at start", alpha_dut_start, beta_dut_start, gamma_dut_start
                print "translation at start", x_dut_start, y_dut_start, z_dut_start

                initialize_angles = True
                translation_old = None
                rotation_old = None
                for i in range(iterations):
                    print "****************** %s: iteration step %d, chunk %d *********************" % (actual_dut.name, i, chunk_index)
                    if initialize_angles:
                        alpha, beta, gamma = alpha_dut_start, beta_dut_start, gamma_dut_start
                        rotation = geometry_utils.rotation_matrix(
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma)
                        translation = np.array([x_dut_start, y_dut_start, z_dut_start], dtype=np.float64)
                        initialize_angles = False

                    # start_delta_t = 0.01
                    # stop_delta_t = 1.0
                    # delta_t = start_delta_t * np.exp(i * (np.log(stop_delta_t) - np.log(start_delta_t)) / iterations)

                    tot_matr = None
                    tot_b = None
                    identity = np.identity(3, dtype=np.float64)
                    n_identity = -np.identity(3, dtype=np.float64)

                    # vectorized calculation of matrix and b vector
                    outer_prod_slopes = np.einsum('ij,ik->ijk', slopes, slopes)
                    l_matr = identity - outer_prod_slopes
                    R_y = np.matmul(rotation, hit_local.T).T
                    skew_R_y = geometry_utils.skew(R_y)
                    tot_matr = np.concatenate([np.matmul(l_matr, skew_R_y), np.matmul(l_matr, n_identity)], axis=2)
                    tot_matr = tot_matr.reshape(-1, tot_matr.shape[2])
                    tot_b = np.matmul(l_matr, np.expand_dims(offsets, axis=2)) - np.matmul(l_matr, np.expand_dims(R_y + translation, axis=2))
                    tot_b = tot_b.reshape(-1)

                    # iterative calculation of matrix and b vector
                    # for count in range(len(hit_x_local)):
                    #     if count >= max_n_tracks:
                    #         count = count - 1
                    #         break
                    #     slope = slopes[count]

                    #     l_matr = identity - np.outer(slope, slope)
                    #     p = offsets[count]
                    #     y = hit_local[count]
                    #     R_y = np.dot(rotation_dut, y)

                    #     b = np.dot(l_matr, p) - np.dot(l_matr, (R_y + translation))
                    #     if tot_b is None:
                    #         tot_b = b
                    #     else:
                    #         tot_b = np.hstack([tot_b, b])

                    #     skew_R_y = geometry_utils.skew(R_y)

                    #     matr = np.dot(l_matr, np.hstack([skew_R_y, n_identity]))
                    #     if tot_matr is None:
                    #         tot_matr = matr
                    #     else:
                    #         tot_matr = np.vstack([tot_matr, matr])

                    # SVD
                    print "calculating SVD..."
                    u, s, v = np.linalg.svd(tot_matr, full_matrices=False)
                    print "...finished calculating SVD"

                    diag = np.diag(s ** -1)
                    tot_matr_inv = np.dot(v.T, np.dot(diag, u.T))
                    omega_b_dot = np.dot(tot_matr_inv, -lin_alpha * tot_b)
                    # Some alignment parameters can be fixed to initial values
                    # If parameter is not in list, set infinitesimal change to zero
                    # Note: An impact on the rotation parameters cannot be completely
                    #       avoided because of the orthogonalization of the rotation matrix
                    if 'translation_x' not in select_alignment_parameters[index]:
                        omega_b_dot[3] = 0.0
                    if 'translation_y' not in select_alignment_parameters[index]:
                        omega_b_dot[4] = 0.0
                    if 'translation_z' not in select_alignment_parameters[index]:
                        omega_b_dot[5] = 0.0
                    if 'rotation_alpha' not in select_alignment_parameters[index]:
                        omega_b_dot[0] = 0.0
                    if 'rotation_beta' not in select_alignment_parameters[index]:
                        omega_b_dot[1] = 0.0
                    if 'rotation_gamma' not in select_alignment_parameters[index]:
                        omega_b_dot[2] = 0.0

                    translation_old2 = translation_old
                    rotation_old2 = rotation_old
                    translation_old = translation
                    rotation_old = rotation

                    rotation = np.dot((np.identity(3, dtype=np.float64) + delta_t * geometry_utils.skew(omega_b_dot[:3])), rotation)
                    # apply UP (polar) decomposition to normalize/orthogonalize rotation matrix (det = 1)
                    u_rot, _, v_rot = np.linalg.svd(rotation, full_matrices=False)
                    rotation = np.dot(u_rot, v_rot)  # orthogonal matrix U
                    translation = translation + delta_t * omega_b_dot[3:]

                    if i >= 2:
                        print "translation"
                        print translation_old2
                        print translation_old
                        print translation
                        print "rotation"
                        print rotation_old2
                        print rotation_old
                        print rotation
                        allclose_trans = np.allclose(translation, translation_old, rtol=0.0, atol=1e-4)
                        allclose_rot = np.allclose(rotation, rotation_old, rtol=0.0, atol=1e-5)
                        allclose_trans2 = np.allclose(translation, translation_old2, rtol=0.0, atol=1e-3)
                        allclose_rot2 = np.allclose(rotation, rotation_old2, rtol=0.0, atol=1e-4)
                        print "allclose rot trans", allclose_rot, allclose_trans
                        print "allclose2 rot trans", allclose_rot2, allclose_trans2
                        # exit if paramters are more or less constant
                        if (allclose_rot and allclose_trans):
                            print "*** ALL CLOSE, BREAK ***"
                            break
                        # change to smaller step size for smaller transformation parameters
                        # and check for oscillating result (every second result is identical)
                        elif (allclose_rot2 or allclose_trans2 or allclose_rot or allclose_trans):
                            print "*** ALL CLOSE2, CONTINUE ***"
                            delta_t = 0.3

                if translation_average is None:
                    translation_average = translation
                else:
                    translation_average = np.average([translation, translation_average], weights=[n_tracks, total_n_tracks], axis=0)
                # alpha, beta, gamma = geometry_utils.euler_angles(R=rotation)
                # euler_angles = np.array([alpha, beta, gamma], dtype=np.float64)
                # if euler_angles_average is None:
                #     euler_angles_average = euler_angles
                # else:
                #     euler_angles_average = np.average([euler_angles, euler_angles_average], weights=[n_tracks, total_n_tracks], axis=0)
                if rotation_average is None:
                    rotation_average = rotation
                else:
                    rotation_average = np.average([rotation, rotation_average], weights=[n_tracks, total_n_tracks], axis=0)

                total_n_tracks += n_tracks
                chunk_index += 1

            # average rotation matrices from different chunks
            u_rot, _, v_rot = np.linalg.svd(rotation_average, full_matrices=False)
            rotation_average = np.dot(u_rot, v_rot)  # orthogonal matrix U
            alpha_average, beta_average, gamma_average = geometry_utils.euler_angles(R=rotation)

            print "%s: alignment parameters before:\n" % actual_dut.name, actual_dut
            actual_dut.translation_x = translation_average[0]
            actual_dut.translation_y = translation_average[1]
            actual_dut.translation_z = translation_average[2]
            actual_dut.rotation_alpha = alpha_average
            actual_dut.rotation_beta = beta_average
            actual_dut.rotation_gamma = gamma_average
            print "%s: alignment parameters after:\n" % actual_dut.name, actual_dut
    telescope.save_configuration()
