''' All DUT alignment functions in space and time are listed here plus additional alignment check functions'''
from __future__ import division

import logging
import sys
import os
from collections.abc import Iterable
import math

import tables as tb
import numpy as np
import scipy
from matplotlib.backends.backend_pdf import PdfPages

from tqdm import tqdm
from beam_telescope_analysis.telescope.telescope import Telescope
from beam_telescope_analysis.tools import analysis_utils
from beam_telescope_analysis.tools import plot_utils
from beam_telescope_analysis.tools import geometry_utils
from beam_telescope_analysis.tools import data_selection
from beam_telescope_analysis.track_analysis import find_tracks, fit_tracks, line_fit_3d, _fit_tracks_kalman_loop
from beam_telescope_analysis.result_analysis import calculate_residuals, histogram_track_angle, get_angles
from beam_telescope_analysis.tools.storage_utils import save_arguments


default_alignment_parameters = ["translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma"]
default_cluster_shapes = [1, 3, 5, 13, 14, 7, 11, 15]
kfa_alignment_descr = np.dtype([('translation_x', np.float64),
                                ('translation_y', np.float64),
                                ('translation_z', np.float64),
                                ('rotation_alpha', np.float64),
                                ('rotation_beta', np.float64),
                                ('rotation_gamma', np.float64),
                                ('translation_x_err', np.float64),
                                ('translation_y_err', np.float64),
                                ('translation_z_err', np.float64),
                                ('rotation_alpha_err', np.float64),
                                ('rotation_beta_err', np.float64),
                                ('rotation_gamma_err', np.float64),
                                ('translation_x_delta', np.float64),
                                ('translation_y_delta', np.float64),
                                ('translation_z_delta', np.float64),
                                ('rotation_alpha_delta', np.float64),
                                ('rotation_beta_delta', np.float64),
                                ('rotation_gamma_delta', np.float64),
                                ('annealing_factor', np.float64)])


@save_arguments
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
        If True, use telescope alignment to align to the beam (beam along z axis).
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

                pbar = tqdm(total=node.shape[0], ncols=80)

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
                    pbar.update(data_chunk.shape[0])
                pbar.close()

    return output_file


def prealign(telescope_configuration, input_correlation_file, output_telescope_configuration=None, select_duts=None, select_reference_dut=0, reduce_background=True, use_location=False, plot=True):
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
    select_duts : iterable
        List of duts for which the prealignment is done. If None, prealignment is done for all duts.
    select_reference_dut : uint
        DUT index of the reference plane. Default is DUT 0.
    reduce_background : bool
        If True, use correlation histograms with reduced background (by applying SVD method to the correlation matrix).
    plot : bool
        If True, create additional output plots.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Pre-alignment of %d DUTs ===' % n_duts)

    if output_telescope_configuration is None:
        output_telescope_configuration = os.path.splitext(telescope_configuration)[0] + '_prealigned.yaml'
    elif output_telescope_configuration == telescope_configuration:
        raise ValueError('Output telescope configuration file must be different from input telescope configuration file.')

    # remove reference DUT from list of all DUTs
    if select_duts is None:
        select_duts = list(set(range(n_duts)) - set([select_reference_dut]))
    else:
        select_duts = list(set(select_duts) - set([select_reference_dut]))

    if plot is True:
        output_pdf = PdfPages(os.path.splitext(input_correlation_file)[0] + '_prealigned.pdf', keep_empty=False)
    else:
        output_pdf = None

    with tb.open_file(input_correlation_file, mode="r") as in_file_h5:
        # loop over DUTs for pre-alignment
        for actual_dut_index in select_duts:
            actual_dut = telescope[actual_dut_index]
            logging.info("== Pre-aligning %s ==" % actual_dut.name)
            x_global_pixel, y_global_pixel, z_global_pixel = [], [], []
            for column in range(1, actual_dut.n_columns + 1):
                global_positions = actual_dut.index_to_global_position(
                    column=[column] * actual_dut.n_rows,
                    row=range(1, actual_dut.n_rows + 1))
                x_global_pixel = np.hstack([x_global_pixel, global_positions[0]])
                y_global_pixel = np.hstack([y_global_pixel, global_positions[1]])
                z_global_pixel = np.hstack([z_global_pixel, global_positions[2]])
            # calculate rotation matrix for later rotation corrections
            rotation_alpha = actual_dut.rotation_alpha
            rotation_beta = actual_dut.rotation_beta
            rotation_gamma = actual_dut.rotation_gamma
            R = geometry_utils.rotation_matrix(
                alpha=rotation_alpha,
                beta=rotation_beta,
                gamma=rotation_gamma)
            select = None
            # loop over x- and y-axis
            for x_direction in [True, False]:
                if reduce_background:
                    node = in_file_h5.get_node(in_file_h5.root, 'Correlation_%s_%d_%d_reduced_background' % ('x' if x_direction else 'y', select_reference_dut, actual_dut_index))
                else:
                    node = in_file_h5.get_node(in_file_h5.root, 'Correlation_%s_%d_%d' % ('x' if x_direction else 'y', select_reference_dut, actual_dut_index))
                dut_name = actual_dut.name
                ref_name = telescope[select_reference_dut].name
                pixel_size = actual_dut.column_size if x_direction else actual_dut.row_size
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
                    raise RuntimeError('Cannot find %s correlation between %s and %s' % ("X" if x_direction else "Y", telescope[select_reference_dut].name, actual_dut.name))
                # offset in the center of the pixel matrix
                offset_center = offset + slope * (0.5 * dut_hist_size - 0.5 * bin_size)
                # calculate offset for local frame
                offset_plot = offset - slope * dut_pos[0]
                # find loactions where the max. correlation is close to expected value
                x_list = find_inliers(
                    x=dut_pos[max_select != 0],
                    y=(max_select[max_select != 0] * bin_size - ref_hist_size / 2.0 + bin_size / 2.0),
                    m=slope,
                    c=offset_plot,
                    threshold=pixel_size * np.sqrt(12) * 2)
                # 1-dimensional clustering of calculated locations
                kernel = scipy.stats.gaussian_kde(x_list)
                densities = kernel(dut_pos)
                max_density = np.max(densities)
                # calculate indices where value is close to max. density
                indices = np.where(densities > max_density * 0.5)
                # get locations from indices
                x_list = dut_pos[indices]
                # calculate range where correlation exists
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
                    output_pdf=output_pdf)

                if select is None:
                    select = np.ones_like(x_global_pixel, dtype=bool)
                if x_direction:
                    select &= (x_global_pixel >= dut_pos_limit[0]) & (x_global_pixel <= dut_pos_limit[1])
                    if slope < 0.0:
                        R = np.linalg.multi_dot([geometry_utils.rotation_matrix_y(beta=np.pi), R])
                    translation_x = offset_center
                else:
                    select &= (y_global_pixel >= dut_pos_limit[0]) & (y_global_pixel <= dut_pos_limit[1])
                    if slope < 0.0:
                        R = np.linalg.multi_dot([geometry_utils.rotation_matrix_x(alpha=np.pi), R])
                    translation_y = offset_center
            # Setting new parameters
            # Only use new limits if they are narrower
            # Convert from global to local coordinates
            local_coordinates = actual_dut.global_to_local_position(
                x=x_global_pixel[select],
                y=y_global_pixel[select],
                z=z_global_pixel[select])
            if actual_dut.x_limit is None:
                actual_dut.x_limit = (min(local_coordinates[0]), max(local_coordinates[0]))
            else:
                actual_dut.x_limit = (max((min(local_coordinates[0]), actual_dut.x_limit[0])), min((max(local_coordinates[0]), actual_dut.x_limit[1])))
            if actual_dut.y_limit is None:
                actual_dut.y_limit = (min(local_coordinates[1]), max(local_coordinates[1]))
            else:
                actual_dut.y_limit = (max((min(local_coordinates[1]), actual_dut.y_limit[0])), min((max(local_coordinates[1]), actual_dut.y_limit[1])))
            # Setting geometry
            actual_dut.translation_x = translation_x
            actual_dut.translation_y = translation_y
            rotation_alpha, rotation_beta, rotation_gamma = geometry_utils.euler_angles(R=R)
            actual_dut.rotation_alpha = rotation_alpha
            actual_dut.rotation_beta = rotation_beta
            actual_dut.rotation_gamma = rotation_gamma

    telescope.save_configuration(configuration_file=output_telescope_configuration)

    if output_pdf is not None:
        output_pdf.close()

    return output_telescope_configuration


def find_inliers(x, y, m, c, threshold=1.0):
    ''' Find inliers.

    Parameters
    ----------
    x : list
        X coordinates.
    y : list
        Y coordinates.
    threshold : float
        Maximum distance of the data points for inlier selection.

    Returns
    -------
    x_list : array
        X coordianates of inliers.
    '''
    # calculate distance to reference hit
    dist = np.abs(m * x + c - y)
    sel = dist < threshold
    return x[sel]


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
    ratio : float
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
        Y coordianates of inliers.
    '''
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


def align(telescope_configuration, input_merged_file, output_telescope_configuration=None, select_duts=None, alignment_parameters=None, select_telescope_duts=None, select_extrapolation_duts=None, select_fit_duts=None, select_hit_duts=None, max_iterations=3, max_events=None, fit_method='fit', beam_energy=None, particle_mass=None, scattering_planes=None, track_chi2=10.0, cluster_shapes=None, quality_distances=(250.0, 250.0), isolation_distances=(500.0, 500.0), use_limits=True, plot=True, chunk_size=1000000):
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
    select_duts : iterable or iterable of iterable
        The combination of duts that are algined at once. One should always align the high resolution planes first.
        E.g. for a telesope (first and last 3 planes) with 2 devices in the center (3, 4):
        select_duts=[[0, 1, 2, 5, 6, 7],  # align the telescope planes first
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
    select_extrapolation_duts : list
        The given DUTs will be used for track extrapolation for improving track finding efficiency.
        In some rare cases, removing DUTs with a coarse resolution might improve track finding efficiency.
        If None, select all DUTs.
        If list is empty or has a single entry, disable extrapolation (at least 2 DUTs are required for extrapolation to work).
    select_fit_duts : iterable or iterable of iterable
        Defines for each select_duts combination wich devices to use in the track fit.
        E.g. To use only the telescope planes (first and last 3 planes) but not the 2 center devices
        select_fit_duts=[0, 1, 2, 5, 6, 7]
    select_hit_duts : iterable or iterable of iterable
        Defines for each select_duts combination wich devices must have a hit to use the track for fitting. The hit
        does not have to be used in the fit itself! This is useful for time reference planes.
        E.g.  To use telescope planes (first and last 3 planes) + time reference plane (3)
        select_hit_duts = [0, 1, 2, 4, 5, 6, 7]
    max_iterations : uint
        Maximum number of iterations of calc residuals, apply rotation refit loop until constant result is expected.
        Usually the procedure converges rather fast (< 5 iterations).
        Non-telescope DUTs usually require 2 itearations.
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
    cluster_shapes : iterable or iterable of iterables
        List of cluster shapes (unsigned integer) for each DUT. Only the selected cluster shapes will be used for the alignment.
        Cluster shapes have impact on precision of the alignment. Larger clusters and certain cluster shapes can have a significant uncertainty for the hit position.
        If None, use default cluster shapes [1, 3, 5, 13, 14, 7, 11, 15], i.e. 1x1, 2x1, 1x2, 3-pixel cluster, 4-pixel cluster. If empty list, all cluster sizes will be used.
        The cluster shape can be calculated with the help of beam_telescope_analysis.tools.analysis_utils.calculate_cluster_array/calculate_cluster_shape.
    quality_distances : 2-tuple or list of 2-tuples
        X and y distance (in um) for each DUT to calculate the quality flag. The selected track and corresponding hit
        must have a smaller distance to have the quality flag to be set to 1.
        The purpose of quality_distances is to find good tracks for the alignment.
        A good start value is 1-2x the pixel pitch for large pixels and high-energy beams and 5-10x the pixel pitch for small pixels and low-energy beams.
        A too small value will remove good tracks, a too large value will allow bad tracks to contribute to the alignment.
        If None, set distance to infinite.
    isolation_distances : 2-tuple or list of 2-tuples
        X and y distance (in um) for each DUT to calculate the isolated track/hit flag. Any other occurence of tracks or hits from the same event
        within this distance will prevent the flag from beeing set.
        The purpose of isolation_distances is to find good tracks for the alignment. Hits and tracks which are too close to each other should be removed.
        The value given by isolation_distances should be larger than the quality_distances value to be effective,
        A too small value will remove almost no tracks, a too large value will remove good tracks.
        If None, set distance to 0.
    isolation_distances : 2-tuple or list of 2-tuples
        X and y distance (in um) for each DUT to calculate the quality flag. Any other occurence of tracks or hits from the same event
        within this distance will reject the quality flag.
        The purpose of isolation_distances is to remove tracks from alignment that could be potentially fake tracks (noisy detector / high beam density).
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
    logging.info('=== Alignment of %d DUTs ===' % len(set(np.unique(np.hstack(np.array(select_duts))).tolist())))

    # Create list with combinations of DUTs to align
    if select_duts is None:  # If None: align all DUTs
        select_duts = list(range(n_duts))
    # Check for value errors
    if not isinstance(select_duts, Iterable):
        raise ValueError("Parameter select_duts is not an iterable.")
    elif not select_duts:  # empty iterable
        raise ValueError("Parameter select_duts has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_duts)):
        select_duts = [select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_duts)):
        raise ValueError("Not all items in parameter select_duts are iterable.")
    # Finally check length of all iterables in iterable
    for dut in select_duts:
        if not dut:  # check the length of the items
            raise ValueError("Item in parameter select_duts has length 0.")

    # Check if some DUTs will not be aligned
    non_select_duts = set(range(n_duts)) - set(np.unique(np.hstack(np.array(select_duts))).tolist())
    if non_select_duts:
        logging.info('These DUTs will not be aligned: %s' % ", ".join(telescope[dut_index].name for dut_index in non_select_duts))

    # Create list
    if alignment_parameters is None:
        alignment_parameters = [[None] * len(duts) for duts in select_duts]
    # Check for value errors
    if not isinstance(alignment_parameters, Iterable):
        raise ValueError("Parameter alignment_parameters is not an iterable.")
    elif not alignment_parameters:  # empty iterable
        raise ValueError("Parameter alignment_parameters has no items.")
    # Finally check length of all arrays
    if len(alignment_parameters) != len(select_duts):  # empty iterable
        raise ValueError("Parameter alignment_parameters has the wrong length.")
    for index, alignment_parameter in enumerate(alignment_parameters):
        if alignment_parameter is None:
            alignment_parameters[index] = [None] * len(select_duts[index])
        if len(alignment_parameters[index]) != len(select_duts[index]):  # check the length of the items
            raise ValueError("Item in parameter alignment_parameter has the wrong length.")

    # Create track, hit selection
    if select_hit_duts is None:  # If None: use all DUTs
        select_hit_duts = []
        # copy each item
        for duts in select_duts:
            select_hit_duts.append(duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(select_hit_duts, Iterable):
        raise ValueError("Parameter select_hit_duts is not an iterable.")
    elif not select_hit_duts:  # empty iterable
        raise ValueError("Parameter select_hit_duts has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_hit_duts)):
        select_hit_duts = [select_hit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_hit_duts)):
        raise ValueError("Not all items in parameter select_hit_duts are iterable.")
    # Finally check length of all arrays
    if len(select_hit_duts) != len(select_duts):  # empty iterable
        raise ValueError("Parameter select_hit_duts has the wrong length.")
    for hit_dut in select_hit_duts:
        if len(hit_dut) < 2:  # check the length of the items
            raise ValueError("Item in parameter select_hit_duts has length < 2.")

    # Create track, hit selection
    if select_fit_duts is None:  # If None: use all DUTs
        select_fit_duts = []
        # copy each item from select_hit_duts
        for hit_duts in select_hit_duts:
            select_fit_duts.append(hit_duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(select_fit_duts, Iterable):
        raise ValueError("Parameter select_fit_duts is not an iterable.")
    elif not select_fit_duts:  # empty iterable
        raise ValueError("Parameter select_fit_duts has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_fit_duts)):
        select_fit_duts = [select_fit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_fit_duts)):
        raise ValueError("Not all items in parameter select_fit_duts are iterable.")
    # Finally check length of all arrays
    if len(select_fit_duts) != len(select_duts):  # empty iterable
        raise ValueError("Parameter select_fit_duts has the wrong length.")
    for index, fit_dut in enumerate(select_fit_duts):
        if len(fit_dut) < 2:  # check the length of the items
            raise ValueError("Item in parameter select_fit_duts has length < 2.")
        if set(fit_dut) - set(select_hit_duts[index]):  # fit DUTs are required to have a hit
            raise ValueError("DUT in select_fit_duts is not in select_hit_duts.")

    # Create chi2 array
    if not isinstance(track_chi2, Iterable):
        track_chi2 = [track_chi2] * len(select_duts)
    # Finally check length
    if len(track_chi2) != len(select_duts):
        raise ValueError("Parameter track_chi2 has the wrong length.")
    # expand dimensions
    # Check iterable and length for each item
    for index, chi2 in enumerate(track_chi2):
        # Check if non-iterable
        if not isinstance(chi2, Iterable):
            track_chi2[index] = [chi2] * len(select_duts[index])
    # again check for consistency
    for index, chi2 in enumerate(track_chi2):
        # Check iterable and length
        if not isinstance(chi2, Iterable):
            raise ValueError("Item in parameter track_chi2 is not an iterable.")
        if len(chi2) != len(select_duts[index]):  # empty iterable
            raise ValueError("Item in parameter track_chi2 has the wrong length.")

    # Create cluster shape selection
    if cluster_shapes is None:  # If None: set default value for all DUTs
        cluster_shapes = [cluster_shapes] * len(select_duts)
    # Check iterable and length
    if not isinstance(cluster_shapes, Iterable):
        raise ValueError("Parameter cluster_shapes is not an iterable.")
    # elif not cluster_shapes:  # empty iterable
    #     raise ValueError("Parameter cluster_shapes has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable) and val is not None, cluster_shapes)):
        cluster_shapes = [cluster_shapes[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, cluster_shapes)):
        raise ValueError("Not all items in parameter cluster_shapes are iterable or None.")
    # Finally check length of all arrays
    if len(cluster_shapes) != len(select_duts):  # empty iterable
        raise ValueError("Parameter cluster_shapes has the wrong length.")
    # expand dimensions
    # Check iterable and length for each item
    for index, shapes in enumerate(cluster_shapes):
        # Check if only non-iterable in iterable
        if shapes is None:
            cluster_shapes[index] = [shapes] * len(select_duts[index])
        elif all(map(lambda val: not isinstance(val, Iterable) and val is not None, shapes)):
            cluster_shapes[index] = [shapes[:] for _ in select_duts[index]]
    # again check for consistency
    for index, shapes in enumerate(cluster_shapes):
        # Check iterable and length
        if not isinstance(shapes, Iterable):
            raise ValueError("Item in parameter cluster_shapes is not an iterable.")
        elif not shapes:  # empty iterable
            raise ValueError("Item in parameter cluster_shapes has no items.")
        # Check if only iterable in iterable
        if not all(map(lambda val: isinstance(val, Iterable) or val is None, shapes)):
            raise ValueError("Not all items of item in cluster_shapes are iterable or None.")
        if len(shapes) != len(select_duts[index]):  # empty iterable
            raise ValueError("Item in parameter cluster_shapes has the wrong length.")

    # Create quality distance
    if isinstance(quality_distances, tuple) or quality_distances is None:
        quality_distances = [quality_distances] * n_duts
    # Check iterable and length
    if not isinstance(quality_distances, Iterable):
        raise ValueError("Parameter quality_distances is not an iterable.")
    elif not quality_distances:  # empty iterable
        raise ValueError("Parameter quality_distances has no items.")
    # Finally check length of all arrays
    if len(quality_distances) != n_duts:  # empty iterable
        raise ValueError("Parameter quality_distances has the wrong length.")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, quality_distances)):
        raise ValueError("Not all items in parameter quality_distances are iterable or None.")
    # Finally check length of all arrays
    for distance in quality_distances:
        if distance is not None and len(distance) != 2:  # check the length of the items
            raise ValueError("Item in parameter quality_distances has length != 2.")

    # Create reject quality distance
    if isinstance(isolation_distances, tuple) or isolation_distances is None:
        isolation_distances = [isolation_distances] * n_duts
    # Check iterable and length
    if not isinstance(isolation_distances, Iterable):
        raise ValueError("Parameter isolation_distances is no iterable.")
    elif not isolation_distances:  # empty iterable
        raise ValueError("Parameter isolation_distances has no items.")
    # Finally check length of all arrays
    if len(isolation_distances) != n_duts:  # empty iterable
        raise ValueError("Parameter isolation_distances has the wrong length.")
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, isolation_distances)):
        raise ValueError("Not all items in Parameter isolation_distances are iterable or None.")
    # Finally check length of all arrays
    for distance in isolation_distances:
        if distance is not None and len(distance) != 2:  # check the length of the items
            raise ValueError("Item in parameter isolation_distances has length != 2.")

    if not isinstance(max_iterations, Iterable):
        max_iterations = [max_iterations] * len(select_duts)
    # Finally check length of all arrays
    if len(max_iterations) != len(select_duts):  # empty iterable
        raise ValueError("Parameter max_iterations has the wrong length.")

    if not isinstance(max_events, Iterable):
        max_events = [max_events] * len(select_duts)
    # Finally check length
    if len(max_events) != len(select_duts):
        raise ValueError("Parameter max_events has the wrong length.")

    if output_telescope_configuration is None:
        if 'prealigned' in telescope_configuration:
            output_telescope_configuration = telescope_configuration.replace('prealigned', 'aligned')
        else:
            output_telescope_configuration = os.path.splitext(telescope_configuration)[0] + '_aligned.yaml'
    elif output_telescope_configuration == telescope_configuration:
        raise ValueError('Output telescope configuration file must be different from input telescope configuration file.')
    if os.path.isfile(output_telescope_configuration):
        logging.info('Output telescope configuration file already exists. Keeping telescope configuration file.')
        aligned_telescope = Telescope(configuration_file=output_telescope_configuration)
        # For the case where not all DUTs are aligned,
        # only revert the alignment for the DUTs that will be aligned.
        for align_duts in select_duts:
            for dut in align_duts:
                aligned_telescope[dut] = telescope[dut]
        aligned_telescope.save_configuration()
    else:
        telescope.save_configuration(configuration_file=output_telescope_configuration)
    prealigned_track_candidates_file = os.path.splitext(input_merged_file)[0] + '_track_candidates_prealigned_tmp.h5'
    # clean up remaining files
    if os.path.isfile(prealigned_track_candidates_file):
        os.remove(prealigned_track_candidates_file)

    for index, align_duts in enumerate(select_duts):
        # Find pre-aligned tracks for the 1st step of the alignment.
        # This file can be used for different sets of alignment DUTs,
        # so keep the file and remove later.
        if not os.path.isfile(prealigned_track_candidates_file):
            logging.info('= Alignment step 1: Finding pre-aligned tracks =')
            find_tracks(
                telescope_configuration=telescope_configuration,
                input_merged_file=input_merged_file,
                output_track_candidates_file=prealigned_track_candidates_file,
                select_extrapolation_duts=select_extrapolation_duts,
                align_to_beam=True,
                max_events=None)

        logging.info('== Aligning %d DUTs: %s ==', len(align_duts), ", ".join(telescope[dut_index].name for dut_index in align_duts))
        _duts_alignment(
            output_telescope_configuration=output_telescope_configuration,  # aligned configuration
            merged_file=input_merged_file,
            prealigned_track_candidates_file=prealigned_track_candidates_file,
            align_duts=align_duts,
            alignment_parameters=alignment_parameters[index],
            select_telescope_duts=select_telescope_duts,
            select_extrapolation_duts=select_extrapolation_duts,
            select_fit_duts=select_fit_duts[index],
            select_hit_duts=select_hit_duts[index],
            max_iterations=max_iterations[index],
            max_events=max_events[index],
            fit_method=fit_method,
            beam_energy=beam_energy,
            particle_mass=particle_mass,
            scattering_planes=scattering_planes,
            track_chi2=track_chi2[index],
            cluster_shapes=cluster_shapes[index],
            quality_distances=quality_distances,
            isolation_distances=isolation_distances,
            use_limits=use_limits,
            plot=plot,
            chunk_size=chunk_size)

    if os.path.isfile(prealigned_track_candidates_file):
        os.remove(prealigned_track_candidates_file)

    return output_telescope_configuration


def align_kalman(telescope_configuration, input_merged_file, output_telescope_configuration=None, output_alignment_file=None, select_duts=None, alignment_parameters=None, alignment_parameters_errors=None, select_telescope_duts=None, select_extrapolation_duts=None, select_fit_duts=None, select_hit_duts=None, min_track_hits=None, max_events=None, beam_energy=None, particle_mass=None, scattering_planes=None, track_chi2=10.0, annealing_factor=10000, annealing_tracks=5000, max_tracks=10000, plot=True, chunk_size=1000):
    ''' This function does an alignment of the DUTs and sets translation and rotation values for all DUTs.
    The reference DUT defines the global coordinate system position at 0, 0, 0 and should be well in the beam and not heavily rotated.

    A Kalman Filter is used to iteratively update the alignment parameters. Alignment parameters as well as initial errors on the alignment (from pre-alignment)
    can be specified. Furhermore, the annealing (scaling of covariance matrix) can be changed with `annealing_factor, annealing_tracks`. A maximum number of tracks can be given after which
    the alignment stops. More information about the algorithm can be found in the referenced papers and/or PhD thesis of Y. Dieter. An example of the Kalman Filter alignment
    can be found in /examples/eutelescope_kalman.py

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_merged_file : string
        Filename of the input merged file.
    output_telescope_configuration : string
        Filename of the output telescope configuration file.
    output_alignment_file : string
        Filename of the output alignment file containing the alignment parameters (and many more things) for each iteration.
    select_duts : iterable or iterable of iterable
        The combination of duts that are algined at once. One should always align the high resolution planes first.
        E.g. for a telesope (first and last 3 planes) with 2 devices in the center (3, 4):
        select_duts=[[0, 1, 2, 5, 6, 7],  # align the telescope planes first
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
    alignment_parameters_errors : list
        Initial error for each alignment parameter.
    select_telescope_duts : iterable
        The given DUTs will be used to align the telescope along the z-axis.
        Usually the coordinates of these DUTs are well specified.
        At least 2 DUTs need to be specified. The z-position of the selected DUTs will not be changed by default.
    select_extrapolation_duts : list
        The given DUTs will be used for track extrapolation for improving track finding efficiency.
        In some rare cases, removing DUTs with a coarse resolution might improve track finding efficiency.
        If None, select all DUTs.
        If list is empty or has a single entry, disable extrapolation (at least 2 DUTs are required for extrapolation to work).
    select_fit_duts : iterable or iterable of iterable
        Defines for each select_duts combination wich devices to use in the track fit.
        E.g. To use only the telescope planes (first and last 3 planes) but not the 2 center devices
        select_fit_duts=[0, 1, 2, 5, 6, 7]
    select_hit_duts : iterable or iterable of iterable
        Defines for each select_duts combination wich devices must have a hit to use the track for fitting. The hit
        does not have to be used in the fit itself! This is useful for time reference planes.
        E.g.  To use telescope planes (first and last 3 planes) + time reference plane (3)
        select_hit_duts = [0, 1, 2, 4, 5, 6, 7]
    min_track_hits : uint or list
        Minimum number of track hits for each selected DUT from select_fit_duts. E.g. min_track_hits=5 and select_fit_duts=[0, 1, 2, 3, 4, 5] will fit any track
        which has at least 5 hits out of DUT0, DUT1, ..., DUT5.
    max_iterations : uint
        Maximum number of iterations of calc residuals, apply rotation refit loop until constant result is expected.
        Usually the procedure converges rather fast (< 5 iterations).
        Non-telescope DUTs usually require 2 itearations.
    max_events: uint
        Radomly select max_events for alignment. If None, use all events, which might slow down the alignment.
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
        Setting the limit on the track chi^2 (reduced!). If None or 0.0, no cut will be applied.
        A smaller value reduces the number of tracks for the alignment.
        A large value increases the number of tracks but at the cost of alignment efficiency bacause of potentially bad tracks.
        A good start value is 5.0 to 10.0 for high energy beams and 15.0 to 50.0 for low energy beams.
    annealing_factor : uint
        Annealing factor (starting value) used for geometric annealing scheme.
    annealing_tracks : uint
        Number of tracks after which annealing is turned off.
    max_tracks : uint
        Maximum number of tracks which are used for alignment. Alignment is stopped after specified amount of tracks
        is processed for every DUT.
    plot : bool
        If True, create additional output plots.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)
    logging.info('=== Alignment of %d DUTs ===' % len(set(np.unique(np.hstack(np.array(select_duts))).tolist())))

    # Create list with combinations of DUTs to align
    if select_duts is None:  # If None: align all DUTs
        select_duts = list(range(n_duts))
    # Check for value errors
    if not isinstance(select_duts, Iterable):
        raise ValueError("Parameter select_duts is not an iterable.")
    elif not select_duts:  # empty iterable
        raise ValueError("Parameter select_duts has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_duts)):
        select_duts = [select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_duts)):
        raise ValueError("Not all items in parameter select_duts are iterable.")
    # Finally check length of all iterables in iterable
    for dut in select_duts:
        if not dut:  # check the length of the items
            raise ValueError("Item in parameter select_duts has length 0.")

    # Check if some DUTs will not be aligned
    non_select_duts = set(range(n_duts)) - set(np.unique(np.hstack(np.array(select_duts))).tolist())
    if non_select_duts:
        logging.info('These DUTs will not be aligned: %s' % ", ".join(telescope[dut_index].name for dut_index in non_select_duts))

    # Create list
    if alignment_parameters is None:
        alignment_parameters = [[None] * len(duts) for duts in select_duts]
    # Check for value errors
    if not isinstance(alignment_parameters, Iterable):
        raise ValueError("Parameter alignment_parameters is not an iterable.")
    elif not alignment_parameters:  # empty iterable
        raise ValueError("Parameter alignment_parameters has no items.")
    # Finally check length of all arrays
    if len(alignment_parameters) != len(select_duts):  # empty iterable
        raise ValueError("Parameter alignment_parameters has the wrong length.")
    # for index, alignment_parameter in enumerate(alignment_parameters):
    #     if alignment_parameter is None:
    #         alignment_parameters[index] = [None] * len(select_duts[index])
    #     if len(alignment_parameters[index]) != len(select_duts[index]):  # check the length of the items
    #         raise ValueError("Item in parameter alignment_parameter has the wrong length.")

    # Create track, hit selection
    if select_hit_duts is None:  # If None: use all DUTs
        select_hit_duts = []
        # copy each item
        for duts in select_duts:
            select_hit_duts.append(duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(select_hit_duts, Iterable):
        raise ValueError("Parameter select_hit_duts is not an iterable.")
    elif not select_hit_duts:  # empty iterable
        raise ValueError("Parameter select_hit_duts has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_hit_duts)):
        select_hit_duts = [select_hit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_hit_duts)):
        raise ValueError("Not all items in parameter select_hit_duts are iterable.")
    # Finally check length of all arrays
    if len(select_hit_duts) != len(select_duts):  # empty iterable
        raise ValueError("Parameter select_hit_duts has the wrong length.")
    for hit_dut in select_hit_duts:
        if len(hit_dut) < 2:  # check the length of the items
            raise ValueError("Item in parameter select_hit_duts has length < 2.")

    # Create track, hit selection
    if select_fit_duts is None:  # If None: use all DUTs
        select_fit_duts = []
        # copy each item from select_hit_duts
        for hit_duts in select_hit_duts:
            select_fit_duts.append(hit_duts[:])  # require a hit for each fit DUT
    # Check iterable and length
    if not isinstance(select_fit_duts, Iterable):
        raise ValueError("Parameter select_fit_duts is not an iterable.")
    elif not select_fit_duts:  # empty iterable
        raise ValueError("Parameter select_fit_duts has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_fit_duts)):
        select_fit_duts = [select_fit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_fit_duts)):
        raise ValueError("Not all items in parameter select_fit_duts are iterable.")
    # Finally check length of all arrays
    if len(select_fit_duts) != len(select_duts):  # empty iterable
        raise ValueError("Parameter select_fit_duts has the wrong length.")
    for index, fit_dut in enumerate(select_fit_duts):
        if len(fit_dut) < 2:  # check the length of the items
            raise ValueError("Item in parameter select_fit_duts has length < 2.")
        if set(fit_dut) - set(select_hit_duts[index]):  # fit DUTs are required to have a hit
            raise ValueError("DUT in select_fit_duts is not in select_hit_duts.")

    if not isinstance(max_events, Iterable):
        max_events = [max_events] * len(select_duts)
    # Finally check length
    if len(max_events) != len(select_duts):
        raise ValueError("Parameter max_events has the wrong length.")

    if not isinstance(max_tracks, Iterable):
        max_tracks = [max_tracks] * len(select_duts)
    # Finally check length
    if len(max_tracks) != len(select_duts):
        raise ValueError("Parameter max_tracks has the wrong length.")

    # Check iterable and length
    if not isinstance(min_track_hits, Iterable):
        min_track_hits = [min_track_hits] * len(select_duts)
    # Finally check length of all arrays
    if len(min_track_hits) != len(select_duts):  # empty iterable
        raise ValueError("Parameter min_track_hits has the wrong length.")

    if output_telescope_configuration is None:
        if 'prealigned' in telescope_configuration:
            output_telescope_configuration = telescope_configuration.replace('prealigned', 'aligned_kalman')
        else:
            output_telescope_configuration = os.path.splitext(telescope_configuration)[0] + '_aligned_kalman.yaml'
    elif output_telescope_configuration == telescope_configuration:
        raise ValueError('Output telescope configuration file must be different from input telescope configuration file.')
    if os.path.isfile(output_telescope_configuration):
        logging.info('Output telescope configuration file already exists. Keeping telescope configuration file.')
        aligned_telescope = Telescope(configuration_file=output_telescope_configuration)
        # For the case where not all DUTs are aligned,
        # only revert the alignment for the DUTs that will be aligned.
        for align_duts in select_duts:
            for dut in align_duts:
                aligned_telescope[dut] = telescope[dut]
        aligned_telescope.save_configuration()
    else:
        telescope.save_configuration(configuration_file=output_telescope_configuration)

    if output_alignment_file is None:
        output_alignment_file = os.path.splitext(input_merged_file)[0] + '_KFA_alignment.h5'
    else:
        output_alignment_file = output_alignment_file

    for index, align_duts in enumerate(select_duts):
        # Find pre-aligned tracks for the 1st step of the alignment.
        # This file can be used for different sets of alignment DUTs,
        # so keep the file and remove later.
        prealigned_track_candidates_file = os.path.splitext(input_merged_file)[0] + '_track_candidates_prealigned_%i_tmp.h5' % index
        find_tracks(
            telescope_configuration=telescope_configuration,
            input_merged_file=input_merged_file,
            output_track_candidates_file=prealigned_track_candidates_file,
            select_extrapolation_duts=select_extrapolation_duts,
            align_to_beam=True,
            max_events=max_events[index])

        logging.info('== Aligning %d DUTs: %s ==', len(align_duts), ", ".join(telescope[dut_index].name for dut_index in align_duts))
        _duts_alignment_kalman(
            telescope_configuration=output_telescope_configuration,  # aligned configuration
            output_alignment_file=output_alignment_file,
            input_track_candidates_file=prealigned_track_candidates_file,
            select_duts=align_duts,
            alignment_parameters=alignment_parameters[index],
            select_telescope_duts=select_telescope_duts,
            select_fit_duts=select_fit_duts[index],
            select_hit_duts=select_hit_duts[index],
            min_track_hits=min_track_hits,
            beam_energy=beam_energy,
            particle_mass=particle_mass,
            scattering_planes=scattering_planes,
            track_chi2=track_chi2[index],
            annealing_factor=annealing_factor,
            annealing_tracks=annealing_tracks,
            max_tracks=max_tracks[index],
            alignment_parameters_errors=alignment_parameters_errors,
            plot=plot,
            chunk_size=chunk_size,
            iteration_index=index)

    return output_telescope_configuration


def _duts_alignment(output_telescope_configuration, merged_file, align_duts, prealigned_track_candidates_file, alignment_parameters, select_telescope_duts, select_extrapolation_duts, select_fit_duts, select_hit_duts, max_iterations, max_events, fit_method, beam_energy, particle_mass, scattering_planes, track_chi2, cluster_shapes, quality_distances, isolation_distances, use_limits, plot=True, chunk_size=100000):  # Called for each list of DUTs to align
    alignment_duts = "_".join(str(dut) for dut in align_duts)
    aligned_telescope = Telescope(configuration_file=output_telescope_configuration)

    output_track_candidates_file = None
    iteration_steps = range(max_iterations)
    for iteration_step in iteration_steps:
        # aligning telescope DUTs to the beam axis (z-axis)
        if set(align_duts) & set(select_telescope_duts):
            align_telescope(
                telescope_configuration=output_telescope_configuration,
                select_telescope_duts=list(set(align_duts) & set(select_telescope_duts)))
        actual_align_duts = align_duts
        actual_fit_duts = select_fit_duts
        # reqire hits in each DUT that will be aligned
        actual_hit_duts = [list(set(select_hit_duts) | set([dut_index])) for dut_index in actual_align_duts]
        actual_quality_duts = actual_hit_duts
        fit_quality_distances = np.zeros_like(quality_distances)
        for index, item in enumerate(quality_distances):
            if index in align_duts:
                fit_quality_distances[index, 0] = np.linspace(item[0] * 1.08447**max_iterations, item[0], max_iterations)[iteration_step]
                fit_quality_distances[index, 1] = np.linspace(item[1] * 1.08447**max_iterations, item[1], max_iterations)[iteration_step]
            else:
                fit_quality_distances[index, 0] = item[0]
                fit_quality_distances[index, 1] = item[1]
        fit_quality_distances = fit_quality_distances.tolist()
        if iteration_step > 0:
            logging.info('= Alignment step 1 - iteration %d: Finding tracks for %d DUTs =', iteration_step, len(align_duts))
            # remove temporary file
            if output_track_candidates_file is not None:
                os.remove(output_track_candidates_file)
            output_track_candidates_file = os.path.splitext(merged_file)[0] + '_track_candidates_aligned_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
            find_tracks(
                telescope_configuration=output_telescope_configuration,
                input_merged_file=merged_file,
                output_track_candidates_file=output_track_candidates_file,
                select_extrapolation_duts=select_extrapolation_duts,
                align_to_beam=True,
                max_events=max_events)

        # The quality flag of the actual align DUT depends on the alignment calculated
        # in the previous iteration, therefore this step has to be done every time
        logging.info('= Alignment step 2 - iteration %d: Fitting tracks for %d DUTs =', iteration_step, len(align_duts))
        output_tracks_file = os.path.splitext(merged_file)[0] + '_tracks_aligned_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
        fit_tracks(
            telescope_configuration=output_telescope_configuration,
            input_track_candidates_file=prealigned_track_candidates_file if iteration_step == 0 else output_track_candidates_file,
            output_tracks_file=output_tracks_file,
            max_events=None if iteration_step > 0 else max_events,
            select_duts=actual_align_duts,
            select_fit_duts=actual_fit_duts,
            select_hit_duts=actual_hit_duts,
            exclude_dut_hit=False,  # for biased residuals
            select_align_duts=actual_align_duts,  # correct residual offset for align DUTs
            method=fit_method,
            beam_energy=beam_energy,
            particle_mass=particle_mass,
            scattering_planes=scattering_planes,
            quality_distances=quality_distances,
            isolation_distances=isolation_distances,
            use_limits=use_limits,
            plot=plot,
            chunk_size=chunk_size)

        logging.info('= Alignment step 3a - iteration %d: Selecting tracks for %d DUTs =', iteration_step, len(align_duts))
        output_selected_tracks_file = os.path.splitext(merged_file)[0] + '_tracks_aligned_selected_tracks_duts_%s_tmp_%d.h5' % (alignment_duts, iteration_step)
        # generate query for select_tracks
        # generate default selection of cluster shapes: 1x1, 2x1, 1x2, 3-pixel cluster, 4-pixel cluster
        for index, shapes in enumerate(cluster_shapes):
            if shapes is None:
                cluster_shapes[index] = default_cluster_shapes
        query_string = [((('(track_chi_red < %f)' % track_chi2[index]) if track_chi2[index] else '') + (' & ' if (track_chi2[index] and cluster_shapes[index]) else '') + (('(' + ' | '.join([('(cluster_shape_dut_{0} == %d)' % cluster_shape) for cluster_shape in cluster_shapes[index]]).format(dut_index) + ')') if cluster_shapes[index] else '')) for index, dut_index in enumerate(actual_align_duts)]
        data_selection.select_tracks(
            telescope_configuration=output_telescope_configuration,
            input_tracks_file=output_tracks_file,
            output_tracks_file=output_selected_tracks_file,
            select_duts=actual_align_duts,
            select_hit_duts=actual_hit_duts,
            select_quality_duts=actual_quality_duts,
            select_isolated_track_duts=actual_quality_duts,
            select_isolated_hit_duts=actual_quality_duts,
            query=query_string,
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
                n_bins=100,
                plot=plot)
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

        logging.info('= Alignment step 4 - iteration %d: Calculating transformation matrix for %d DUTs =', iteration_step, len(align_duts))
        calculate_transformation(
            telescope_configuration=output_telescope_configuration,
            input_tracks_file=output_selected_tracks_file,
            select_duts=actual_align_duts,
            select_alignment_parameters=[(["translation_x", "translation_y", "rotation_alpha", "rotation_beta", "rotation_gamma"] if (dut_index in select_telescope_duts and (alignment_parameters is None or alignment_parameters[i] is None)) else (default_alignment_parameters if (alignment_parameters is None or alignment_parameters[i] is None) else alignment_parameters[i])) for i, dut_index in enumerate(actual_align_duts)],
            use_limits=use_limits,
            max_iterations=100,
            chunk_size=chunk_size)

        # Delete temporary files
        os.remove(output_tracks_file)
        os.remove(output_selected_tracks_file)

    # Delete temporary files
    if output_track_candidates_file is not None:
        os.remove(output_track_candidates_file)


def _duts_alignment_kalman(telescope_configuration, output_alignment_file, input_track_candidates_file, alignment_parameters, select_telescope_duts, select_duts=None, select_hit_duts=None, select_fit_duts=None, min_track_hits=None, beam_energy=2500, particle_mass=0.511, scattering_planes=None, track_chi2=25.0, iteration_index=0, exclude_dut_hit=False, annealing_factor=10000, annealing_tracks=5000, max_tracks=10000, alignment_parameters_errors=None, plot=True, chunk_size=1000):
    ''' Function which performs actual Kalman Filter alignment loop and calls plotting in the end.
    '''
    def _store_alignment_data(alignment_values, n_tracks_processed, chi2s, chi2s_probs, deviation_cuts):
        ''' Helper function to write alignment data to output file.
        '''
        # Do not forget to save configuration to .yaml file.
        telescope.save_configuration()

        # Store alignment results in file
        for dut_index, _ in enumerate(telescope):
            try:  # Check if table exists already, then append data
                alignment_table = out_file_h5.get_node('/Alignment_DUT%i' % dut_index)
            except tb.NoSuchNodeError:  # Table does not exist, thus create new
                alignment_table = out_file_h5.create_table(
                    where=out_file_h5.root,
                    name='Alignment_DUT%i' % dut_index,
                    description=alignment_values[dut_index].dtype,
                    title='Alignment_DUT%i' % dut_index,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))

            alignment_table.append(alignment_values[dut_index])
            alignment_table.attrs.deviation_cuts = deviation_cuts
            alignment_table.attrs.n_tracks_processed = n_tracks_processed[dut_index]
            alignment_table.flush()

        # Store chi2 values
        try:  # Check if table exists already, then append data
            out_chi2s = out_file_h5.get_node('/TrackChi2')
            out_chi2s_probs = out_file_h5.get_node('/TrackpValue')
        except tb.NoSuchNodeError:  # Table does not exist, thus create new
            out_chi2s = out_file_h5.create_earray(
                where=out_file_h5.root,
                name='TrackChi2',
                title='Track Chi2',
                atom=tb.Atom.from_dtype(chi2s.dtype),
                shape=(0,),
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))
            out_chi2s_probs = out_file_h5.create_earray(
                where=out_file_h5.root,
                name='TrackpValue',
                title='Track pValue',
                atom=tb.Atom.from_dtype(chi2s.dtype),
                shape=(0,),
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))

        out_chi2s.append(chi2s)
        out_chi2s.flush()
        out_chi2s_probs.append(chi2s_probs)
        out_chi2s_probs.flush()
        out_chi2s.attrs.max_track_chi2 = track_chi2

    def _alignment_loop(actual_align_state, actual_align_cov, initial_rotation_matrix, initial_position_vector):
        ''' Helper function which loops over track chunks and performs the alignnment.
        '''
        # Init progressbar
        n_tracks = in_file_h5.root.TrackCandidates.shape[0]
        pbar = tqdm(total=n_tracks, ncols=80)

        # Number of processed tracks for every DUT
        n_tracks_processed = np.zeros(shape=(len(telescope)), dtype=int)
        # Number of tracks fulfilling hit requirement
        total_n_tracks_valid_hits = 0

        # Maximum allowed relative change for each alignment parameter. Can be adjusted if needed.
        deviation_cuts = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        alpha = np.zeros(shape=len(telescope), dtype=np.float64)  # annealing factor

        # Loop in chunks over tracks. After each chunk, alignment values are stored.
        for track_candidates_chunk, index_chunk in analysis_utils.data_aligned_at_events(in_file_h5.root.TrackCandidates, chunk_size=10000):
            # Select only tracks for which hit requirement is fulfilled
            track_candidates_chunk_valid_hits = track_candidates_chunk[track_candidates_chunk['hit_flag'] & dut_hit_mask == dut_hit_mask]
            total_n_tracks_valid_hits_chunk = track_candidates_chunk_valid_hits.shape[0]
            total_n_tracks_valid_hits += total_n_tracks_valid_hits_chunk

            # Per chunk variables
            chi2s = np.zeros(shape=(total_n_tracks_valid_hits_chunk), dtype=np.float64)  # track chi2s
            chi2s_probs = np.zeros(shape=(total_n_tracks_valid_hits_chunk), dtype=np.float64)  # track pvalues
            alignment_values = np.full(shape=(len(telescope), total_n_tracks_valid_hits_chunk), dtype=kfa_alignment_descr, fill_value=np.nan)  # alignment values

            # Loop over tracks in chunk
            for track_index, track in enumerate(track_candidates_chunk_valid_hits):
                track_hits = np.full((1, n_duts, 6), fill_value=np.nan, dtype=np.float64)

                # Compute aligned position and apply the alignment
                for dut_index, dut in enumerate(telescope):
                    # Get local track hits
                    track_hits[:, dut_index, 0] = track['x_dut_%s' % dut_index]
                    track_hits[:, dut_index, 1] = track['y_dut_%s' % dut_index]
                    track_hits[:, dut_index, 2] = track['z_dut_%s' % dut_index]
                    track_hits[:, dut_index, 3] = track['x_err_dut_%s' % dut_index]
                    track_hits[:, dut_index, 4] = track['y_err_dut_%s' % dut_index]
                    track_hits[:, dut_index, 5] = track['z_err_dut_%s' % dut_index]

                    # Calculate new alignment (takes initial alignment and actual *change* of parameters)
                    new_rotation_matrix, new_position_vector = _update_alignment(initial_rotation_matrix[dut_index], initial_position_vector[dut_index], actual_align_state[dut_index])

                    # Get euler angles from rotation matrix
                    alpha_average, beta_average, gamma_average = geometry_utils.euler_angles(R=new_rotation_matrix)

                    # Set new alignment to DUT
                    dut._translation_x = float(new_position_vector[0])
                    dut._translation_y = float(new_position_vector[1])
                    dut._translation_z = float(new_position_vector[2])
                    dut._rotation_alpha = float(alpha_average)
                    dut._rotation_beta = float(beta_average)
                    dut._rotation_gamma = float(gamma_average)

                    alignment_values[dut_index, track_index]['translation_x'] = dut.translation_x
                    alignment_values[dut_index, track_index]['translation_y'] = dut.translation_y
                    alignment_values[dut_index, track_index]['translation_z'] = dut.translation_z
                    alignment_values[dut_index, track_index]['rotation_alpha'] = dut.rotation_alpha
                    alignment_values[dut_index, track_index]['rotation_beta'] = dut.rotation_beta
                    alignment_values[dut_index, track_index]['rotation_gamma'] = dut.rotation_gamma

                    C = actual_align_cov[dut_index]
                    alignment_values[dut_index, track_index]['translation_x_err'] = np.sqrt(C[0, 0])
                    alignment_values[dut_index, track_index]['translation_y_err'] = np.sqrt(C[1, 1])
                    alignment_values[dut_index, track_index]['translation_z_err'] = np.sqrt(C[2, 2])
                    alignment_values[dut_index, track_index]['rotation_alpha_err'] = np.sqrt(C[3, 3])
                    alignment_values[dut_index, track_index]['rotation_beta_err'] = np.sqrt(C[4, 4])
                    alignment_values[dut_index, track_index]['rotation_gamma_err'] = np.sqrt(C[5, 5])

                    # Calculate deterministic annealing (scaling factor for covariance matrix) in order to take into account misalignment
                    alpha[dut_index] = _calculate_annealing(k=n_tracks_processed[dut_index], annealing_factor=annealing_factor, annealing_tracks=annealing_tracks)
                    # Store annealing factor
                    alignment_values[dut_index, track_index]['annealing_factor'] = alpha[dut_index]

                # Run Kalman Filter
                try:
                    offsets, slopes, chi2s_reg, chi2s_red, chi2s_prob, x_err, y_err, cov, cov_obs, obs_mat = _fit_tracks_kalman_loop(track_hits, telescope, fit_duts, beam_energy, particle_mass, scattering_planes, alpha)
                except Exception as e:
                    print(e, 'TRACK FITTING')
                    continue

                # Store chi2 and pvalue
                chi2s[track_index] = chi2s_red
                chi2s_probs[track_index] = chi2s_prob

                # Data quality check I: Check chi2 of track
                if chi2s_red > track_chi2:
                    continue

                # Actual track states
                p0 = np.column_stack((offsets[0, :, 0], offsets[0, :, 1],
                                      slopes[0, :, 0], slopes[0, :, 1]))
                # Covariance matrix (x, y, dx, dy) of track estimates
                C0 = cov[0, :, :, :]
                # Covariance matrix (x, y, dx, dy) of observations
                V = cov_obs[0, :, :, :]
                # Measurement matrix
                H = obs_mat[0, :, :, :]

                # Actual alignment parameters and its covariance
                a0 = actual_align_state.copy()
                E0 = actual_align_cov.copy()

                # Updated alignment parameters and its covariance
                E1 = np.zeros_like(E0)
                a1 = np.zeros_like(a0)

                # Update all alignables
                actual_align_state, actual_align_cov, alignment_values, n_tracks_processed = _update_alignment_parameters(
                    telescope, H, V, C0, p0, a0, E0, track_hits, a1, E1,
                    alignment_values, deviation_cuts,
                    actual_align_state, actual_align_cov, n_tracks_processed, track_index)

                # Reached number of max. specified tracks. Stop alignment
                if n_tracks_processed.min() > max_tracks:
                    pbar.update(track_index)
                    pbar.write('Processed {0} tracks (per DUT) out of {1} tracks'.format(n_tracks_processed, total_n_tracks_valid_hits))
                    pbar.close()
                    logging.info('Maximum number of tracks reached! Stopping alignment...')
                    # Store alignment data
                    _store_alignment_data(alignment_values[:, :track_index + 1], n_tracks_processed, chi2s[:track_index + 1], chi2s_probs[:track_index + 1], deviation_cuts)
                    return

            pbar.update(track_candidates_chunk.shape[0])
            pbar.write('Processed {0} tracks (per DUT) out of {1} tracks'.format(n_tracks_processed, total_n_tracks_valid_hits))
            # Store alignment data
            _store_alignment_data(alignment_values, n_tracks_processed, chi2s, chi2s_probs, deviation_cuts)

        pbar.close()

    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)

    logging.info('= Alignment step 2 - Fitting tracks for %d DUTs =', len(select_duts))
    if iteration_index == 0:  # clean up before starting alignment. In case different sets of DUTs are aligned after each other only clean up once.
        if os.path.exists(output_alignment_file):
            os.remove(output_alignment_file)

    logging.info('=== Fitting tracks of %d DUTs ===' % n_duts)

    fitted_duts = []
    with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
        with tb.open_file(output_alignment_file, mode='a') as out_file_h5:
            for fit_dut_index, actual_fit_dut in enumerate(select_duts):  # Loop over the DUTs where tracks shall be fitted for
                # Test whether other DUTs have identical tracks
                # if yes, save some CPU time and fit only once.
                # This following list contains all DUT indices that will be fitted
                # during this step of the loop.
                if actual_fit_dut in fitted_duts:
                    continue
                # calculate all DUTs with identical tracks to save processing time
                actual_fit_duts = []
                for curr_fit_dut_index, curr_fit_dut in enumerate(select_duts):
                    if (curr_fit_dut == actual_fit_dut or
                        (((exclude_dut_hit[curr_fit_dut_index] is False and exclude_dut_hit[fit_dut_index] is False and set(select_fit_duts[curr_fit_dut_index]) == set(select_fit_duts[fit_dut_index])) or
                          (exclude_dut_hit[curr_fit_dut_index] is False and exclude_dut_hit[fit_dut_index] is True and set(select_fit_duts[curr_fit_dut_index]) == (set(select_fit_duts[fit_dut_index]) - set([actual_fit_dut]))) or
                          (exclude_dut_hit[curr_fit_dut_index] is True and exclude_dut_hit[fit_dut_index] is False and (set(select_fit_duts[curr_fit_dut_index]) - set([curr_fit_dut])) == set(select_fit_duts[fit_dut_index])) or
                          (exclude_dut_hit[curr_fit_dut_index] is True and exclude_dut_hit[fit_dut_index] is True and (set(select_fit_duts[curr_fit_dut_index]) - set([curr_fit_dut])) == (set(select_fit_duts[fit_dut_index]) - set([actual_fit_dut])))) and
                         set(select_hit_duts[curr_fit_dut_index]) == set(select_hit_duts[fit_dut_index]) and
                         min_track_hits[curr_fit_dut_index] == min_track_hits[fit_dut_index])):
                        actual_fit_duts.append(curr_fit_dut)
                # continue with fitting
                logging.info('== Fit tracks for %s ==', ', '.join([telescope[curr_dut].name for curr_dut in actual_fit_duts]))

                # select hit DUTs based on input parameters
                # hit DUTs are always enforced
                hit_duts = select_hit_duts[fit_dut_index]
                dut_hit_mask = 0  # DUTs required to have hits
                for dut_index in hit_duts:
                    dut_hit_mask |= ((1 << dut_index))
                logging.info('Require hits in %d DUTs for track selection: %s', len(hit_duts), ', '.join([telescope[curr_dut].name for curr_dut in hit_duts]))

                # select fit DUTs based on input parameters
                # exclude actual DUTs from fit DUTs if exclude_dut_hit parameter is set (for, e.g., unbiased residuals)
                fit_duts = list(set(select_fit_duts[fit_dut_index]) - set([actual_fit_dut])) if exclude_dut_hit[fit_dut_index] else select_fit_duts[fit_dut_index]
                if min_track_hits[fit_dut_index] is None:
                    actual_min_track_hits = len(fit_duts)
                else:
                    actual_min_track_hits = min_track_hits[fit_dut_index]
                if actual_min_track_hits < 2:
                    raise ValueError('The number of required hits is smaller than 2. Cannot fit tracks for %s.', telescope[actual_fit_dut].name)
                dut_fit_mask = 0  # DUTs to be used for the fit
                for dut_index in fit_duts:
                    dut_fit_mask |= ((1 << dut_index))
                if actual_min_track_hits > len(fit_duts):
                    raise RuntimeError("min_track_hits for DUT%d is larger than the number of fit DUTs" % (actual_fit_dut,))
                logging.info('Require at least %d hits in %d DUTs for track selection: %s', actual_min_track_hits, len(fit_duts), ', '.join([telescope[curr_dut].name for curr_dut in fit_duts]))

                if scattering_planes is not None:
                    logging.info('Adding the following scattering planes: %s', ', '.join([scp.name for scp in scattering_planes]))

                # Actual *change* of alignment parameters and covariance
                actual_align_state = np.zeros(shape=(len(telescope), 6), dtype=np.float64)  # No change at beginning
                actual_align_cov = np.zeros(shape=(len(telescope), 6, 6), dtype=np.float64)

                # Calculate initial alignment
                initial_rotation_matrix, initial_position_vector, actual_align_cov = _calculate_initial_alignment(telescope, select_duts, select_telescope_duts, alignment_parameters, actual_align_cov, alignment_parameters_errors)
                # Loop over tracks in chunks and perform alignment.
                _alignment_loop(actual_align_state, actual_align_cov, initial_rotation_matrix, initial_position_vector)

                fitted_duts.extend(actual_fit_duts)

    output_pdf_file = output_alignment_file[:-3] + '.pdf'
    # Plot alignment result
    plot_utils.plot_kf_alignment(output_alignment_file, telescope, output_pdf_file)

    # Delete tmp track candidates file
    os.remove(input_track_candidates_file)


def align_telescope(telescope_configuration, select_telescope_duts, reference_dut=None):
    telescope = Telescope(telescope_configuration)
    logging.info('= Beam-alignment of the telescope =')
    logging.info('Use %d DUTs for beam-alignment: %s', len(select_telescope_duts), ', '.join([telescope[index].name for index in select_telescope_duts]))

    telescope_duts_positions = np.full((len(select_telescope_duts), 3), fill_value=np.nan, dtype=np.float64)
    for index, dut_index in enumerate(select_telescope_duts):
        telescope_duts_positions[index, 0] = telescope[dut_index].translation_x
        telescope_duts_positions[index, 1] = telescope[dut_index].translation_y
        telescope_duts_positions[index, 2] = telescope[dut_index].translation_z
    # the x and y translation for the reference DUT will be set to 0
    if reference_dut is not None:
        first_telescope_dut_index = reference_dut
    else:
        # calculate reference DUT, use DUT with the smallest z position
        first_telescope_dut_index = select_telescope_duts[np.argmin(telescope_duts_positions[:, 2])]
    offset, slope = line_fit_3d(positions=telescope_duts_positions)
    first_telescope_dut = telescope[first_telescope_dut_index]
    logging.info('Reference DUT for beam-alignment: %s', first_telescope_dut.name)
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
        select_alignment_parameters = [default_alignment_parameters] * len(select_duts)
    if len(select_duts) != len(select_alignment_parameters):
        raise ValueError("Parameter select_alignment_parameters has the wrong length.")
    for index, actual_alignment_parameters in enumerate(select_alignment_parameters):
        if actual_alignment_parameters is None:
            select_alignment_parameters[index] = default_alignment_parameters
        else:
            non_valid_paramters = set(actual_alignment_parameters) - set(default_alignment_parameters)
            if non_valid_paramters:
                raise ValueError("Found invalid values in parameter select_alignment_parameters: %s." % ", ".join(non_valid_paramters))

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        for index, actual_dut_index in enumerate(select_duts):
            actual_dut = telescope[actual_dut_index]
            node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)
            logging.info('= Calculate transformation for %s =', actual_dut.name)
            logging.info("Modify alignment parameters: %s", ', '.join([alignment_paramter for alignment_paramter in select_alignment_parameters[index]]))

            if use_limits:
                limit_x_local = actual_dut.x_limit  # (lower limit, upper limit)
                limit_y_local = actual_dut.y_limit  # (lower limit, upper limit)
            else:
                limit_x_local = None
                limit_y_local = None

            rotation_average = None
            translation_average = None
            # euler_angles_average = None

            # calculate equal chunk size
            start_val = max(int(node.nrows / chunk_size), 2)
            while True:
                chunk_indices = np.linspace(0, node.nrows, start_val).astype(int)
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

                limit_xy_local_sel = np.ones_like(hit_x_local, dtype=bool)
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
                delta_t = 0.9  # TODO: optimize
                if max_iterations is None:
                    iterations = 100
                else:
                    iterations = max_iterations
                lin_alpha = 1.0

                initialize_angles = True
                translation_old = None
                rotation_old = None
                for i in range(iterations):
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
                    u, s, v = np.linalg.svd(tot_matr, full_matrices=False)

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
                        allclose_trans = np.allclose(translation, translation_old, rtol=0.0, atol=1e-4)
                        allclose_rot = np.allclose(rotation, rotation_old, rtol=0.0, atol=1e-5)
                        allclose_trans2 = np.allclose(translation, translation_old2, rtol=0.0, atol=1e-3)
                        allclose_rot2 = np.allclose(rotation, rotation_old2, rtol=0.0, atol=1e-4)
                        # exit if paramters are more or less constant
                        if (allclose_rot and allclose_trans):
                            break
                        # change to smaller step size for smaller transformation parameters
                        # and check for oscillating result (every second result is identical)
                        elif (allclose_rot2 or allclose_trans2 or allclose_rot or allclose_trans):
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

            actual_dut.translation_x = translation_average[0]
            actual_dut.translation_y = translation_average[1]
            actual_dut.translation_z = translation_average[2]
            actual_dut.rotation_alpha = alpha_average
            actual_dut.rotation_beta = beta_average
            actual_dut.rotation_gamma = gamma_average
    telescope.save_configuration()


def _calculate_initial_alignment(telescope, select_duts, select_telescope_duts, alignment_parameters, actual_align_cov, alignment_parameters_errors):
    ''' Calculate initial alignment parameters. Setting initial covariance to zero excludes alignment parameter from alignment.
    '''
    initial_rotation_matrix = np.zeros(shape=(len(telescope), 3, 3), dtype=np.float64)
    initial_position_vector = np.zeros(shape=(len(telescope), 3), dtype=np.float64)
    for dut_index, dut in enumerate(telescope):
        # Initial (global) postion
        initial_position_vector[dut_index, 0] = dut.translation_x
        initial_position_vector[dut_index, 1] = dut.translation_y
        initial_position_vector[dut_index, 2] = dut.translation_z
        # Initial rotation matrix
        initial_rotation_matrix[dut_index] = geometry_utils.rotation_matrix(
                    alpha=dut.rotation_alpha,
                    beta=dut.rotation_beta,
                    gamma=dut.rotation_gamma)

        # Errors on initial alignment parameters
        if dut_index in select_duts:
            if 'translation_x' in alignment_parameters[dut_index]:
                actual_align_cov[dut_index, 0, 0] = np.square(alignment_parameters_errors[0])  # 50 um error
            if 'translation_y' in alignment_parameters[dut_index]:
                actual_align_cov[dut_index, 1, 1] = np.square(alignment_parameters_errors[1])  # 50 um error
            if 'translation_z' in alignment_parameters[dut_index]:
                actual_align_cov[dut_index, 2, 2] = np.square(alignment_parameters_errors[2])  # 1 mm error
            if 'rotation_alpha' in alignment_parameters[dut_index]:
                actual_align_cov[dut_index, 3, 3] = np.square(alignment_parameters_errors[3])  # 20 mrad error
            if 'rotation_beta' in alignment_parameters[dut_index]:
                actual_align_cov[dut_index, 4, 4] = np.square(alignment_parameters_errors[4])  # 20 mrad error
            if 'rotation_gamma' in alignment_parameters[dut_index]:
                actual_align_cov[dut_index, 5, 5] = np.square(alignment_parameters_errors[5])  # 20 mrad error

    # Fix first and last telescope plane (only rotation_gamma is not fixed).
    # In principle only z needs to be fixed to avoid telescope stretching and very first plane (+ usage of beam alignment).
    for k in [0, 1, 2, 3, 4]:  # leave rotation_gamma floating
        actual_align_cov[select_telescope_duts[0], k, k] = 0.0
        actual_align_cov[select_telescope_duts[-1], k, k] = 0.0

    return initial_rotation_matrix, initial_position_vector, actual_align_cov


def _update_alignment_parameters(telescope, H, V, C0, p0, a0, E0, track_hits, a1, E1, alignment_values, deviation_cuts, actual_align_state, actual_align_cov, n_tracks_processed, track_index):
    ''' Update alignment parameters and check for change. If rel. change too large, change is rejected.
    '''
    for dut_index, dut in enumerate(telescope):
        R = geometry_utils.rotation_matrix(
                alpha=dut.rotation_alpha,
                beta=dut.rotation_beta,
                gamma=dut.rotation_gamma).T

        # Calculate update for alignment parameters
        a1[dut_index, :], E1[dut_index, :] = _calculate_alignment_parameters(H[dut_index], V[dut_index], C0[dut_index], p0[dut_index],
                                                                             a0[dut_index], E0[dut_index], R, track_hits[0, dut_index, :2])

        # Data quality check II: Check change of alignment parameters
        filter_track = False
        for par, par_name in enumerate(['translation_x', 'translation_y', 'translation_z', 'rotation_alpha', 'rotation_beta', 'rotation_gamma']):  # loop over all parameters for alignable
            if deviation_cuts[par] > 0.0:
                if np.sqrt(E0[dut_index, par, par]) == 0:
                    alignment_values[dut_index, track_index][par_name + '_delta'] = 0.0
                else:
                    alignment_values[dut_index, track_index][par_name + '_delta'] = np.abs(a1[dut_index, par] - a0[dut_index, par]) / np.sqrt(E0[dut_index, par, par])
                if np.abs(a1[dut_index, par] - a0[dut_index, par]) > deviation_cuts[par] * np.sqrt(E0[dut_index, par, par]):
                    filter_track = True
                    break

        if filter_track:
            continue

        # Update actual alignment state
        actual_align_state[dut_index] = a1[dut_index].copy()
        actual_align_cov[dut_index] = E1[dut_index].copy()
        n_tracks_processed[dut_index] += 1

    return actual_align_state, actual_align_cov, alignment_values, n_tracks_processed


def _update_alignment(initial_rotation_matrix, initial_position_vector, actual_align_state):
    ''' Update alignment by applying corrections to initial alignment.
    '''
    # Extract alignment parameters
    dx = actual_align_state[0]
    dy = actual_align_state[1]
    dz = actual_align_state[2]
    dalpha = actual_align_state[3]
    dbeta = actual_align_state[4]
    dgamma = actual_align_state[5]

    # Compute a 'delta' frame from corrections
    delta_frame = _create_karimaki_delta(dx, dy, dz, dalpha, dbeta, dgamma)

    # Merge initial alignment and corrections ('delta')
    return _combine_karimaki(initial_rotation_matrix, initial_position_vector, delta_frame)


def _create_karimaki_delta(dx, dy, dz, dalpha, dbeta, dgamma):
    ''' Reference: https://cds.cern.ch/record/619975/files/cr03_022.pdf, Eq. 4
        Use full rotation matrix here, instead of small angle approximation used in paper.
    '''

    # Small rotation by dalpha, dbeta, dgamma around
    delta_rot = geometry_utils.rotation_matrix(
                                alpha=dalpha,
                                beta=dbeta,
                                gamma=dgamma).T

    # Shift of sensor center by dx,dy,dz in global coord. system.
    delta_offset = np.array([dx, dy, dz])

    return delta_rot, delta_offset


def _combine_karimaki(initial_rotation_matrix, initial_position_vector, delta_frame):
    ''' Apply corrections ('delta') to initial alignment.
        Reference: https://cds.cern.ch/record/619975/files/cr03_022.pdf
    '''
    combined_rot = np.matmul(delta_frame[0], initial_rotation_matrix)
    combined_offset = delta_frame[1] + initial_position_vector
    return combined_rot, combined_offset


def _calculate_annealing(k, annealing_factor, annealing_tracks):
    '''Geometrical annealing scheme according to https://iopscience.iop.org/article/10.1088/0954-3899/29/3/309.
    '''
    if k < annealing_tracks:
        alpha = annealing_factor ** ((annealing_tracks - k) / annealing_tracks)
    else:
        alpha = 1.0

    return alpha


def _calculate_alignment_parameters(H, V, C0, p0, a0, E0, R, m):
    ''' Update formulars for alignment parameters and its covariance.
        Reference: https://iopscience.iop.org/article/10.1088/0954-3899/29/3/309
    '''
    # Jacobian of aligmnent parameters
    D = _jacobian_alignment(p0, R)

    # Weight matrix
    W = np.linalg.inv((V + np.matmul(H, np.matmul(C0, H.T)) + np.matmul(D, np.matmul(E0, D.T))))

    # Update alignment parameter states and covariance
    a1 = a0 + np.matmul(np.matmul(E0,  np.matmul(D.T, W)), (m - np.matmul(H, p0)))
    E1 = E0 - np.matmul(E0, np.matmul(D.T, np.matmul(W, np.matmul(D, E0.T))))

    return a1, E1


def _jacobian_alignment(p0, R):
    ''' Derivative of measuremnts with respect to global alignment parameters.
        Reference: https://cds.cern.ch/record/619975/files/cr03_022.pdf
    '''
    # Extract position and slope from track state
    u = p0[0]
    v = p0[1]
    tu = p0[2]
    tv = p0[3]

    # Jacobian matrix
    jaq = np.zeros(shape=(2, 6), dtype=np.float64)
    jaq[0, 0] = -1.0      # dfu / ddu
    jaq[1, 0] = 0.0       # dfv / ddu
    jaq[0, 1] = 0.0       # dfu / ddv
    jaq[1, 1] = -1.0      # dfv / ddv
    jaq[0, 2] = tu        # dfu / ddw
    jaq[1, 2] = tv        # dfv / ddw
    jaq[0, 3] = -v * tu   # dfu / ddalpha
    jaq[1, 3] = -v * tv   # dfv / ddalpha
    jaq[0, 4] = u * tu    # dfu / ddbeta
    jaq[1, 4] = u * tv    # dfv / ddbeta
    jaq[0, 5] = -v        # dfu / ddgamma
    jaq[1, 5] = u         # dfv / ddgamma

    A = np.eye(6, dtype=np.float64)
    A[:3, :3] = R

    # Apply chain rule to transform into global coordinate system
    return np.matmul(jaq, A)
