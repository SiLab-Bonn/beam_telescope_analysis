''' Helper functions for geometrical operations.
'''

from __future__ import division

import logging
from collections import Iterable
import math

import tables as tb
import numpy as np
from scipy.optimize import minimize, basinhopping
import scipy


def skew(v):
    matr = np.zeros(np.prod(np.r_[v.shape, 3]))
    matr[::9] = v.flatten()[::3]
    matr[4::9] = v.flatten()[1::3]
    matr[8::9] = v.flatten()[2::3]
    matr = matr.reshape(np.r_[v.shape, 3])
    
    if len(v.shape) == 1:
        matr_skv = np.roll(np.roll(matr, 1, 1), -1, 0)
        return matr_skv - matr_skv.T
    else:
        matr_skv = np.roll(np.roll(matr, 1, 2), -1, 1)
        return matr_skv - matr_skv.transpose((0, 2, 1))
    

def get_plane_normal(direction_vector_1, direction_vector_2):
    ''' Normal vector of a plane.

    Plane is define by two non parallel direction vectors within the plane.

    Parameters
    ----------
    direction_vector_1 : array
        Array with x, y and z.
    direction_vector_2 : array
        Array with x, y and z.

    Returns
    -------
    Array with x, y and z.
    '''
    return np.cross(direction_vector_1, direction_vector_2)


def get_line_intersections_with_plane(line_origins, line_directions,
                                      position_plane, normal_plane):
    ''' Calculates the intersection of n lines with one plane.

    If there is no intersection point (line is parallel to plane or the line is
    in the plane) the intersection point is set to nan.

    Notes
    -----
    Further information:
    http://stackoverflow.com/questions/4938332/line-plane-intersection-based-on-points

    Parameters
    ----------
    line_origins : array
        A point (x, y and z) on the line for each of the n lines.
    line_directions : array
        The direction vector of the line for n lines.
    position_plane : array
        A array (x, y and z) to the plane.
    normal_plane : array
        The normal vector (x, y and z) of the plane.


    Returns
    -------
    Array with shape (n, 3) with the intersection point.
    '''
    # Calculate offsets and extend in missing dimension
    offsets = position_plane[np.newaxis, :] - line_origins

    # Precalculate to be able to avoid division by 0
    # (line is parallel to the plane or in the plane)
    norm_dot_off = np.dot(normal_plane, offsets.T)
    # Dot product is transformed to be at least 1D for special n = 1
    norm_dot_dir = np.atleast_1d(np.dot(normal_plane,
                                        line_directions.T))

    # Initialize result to nan
    t = np.full_like(norm_dot_off, fill_value=np.nan)

    # Warn if some intersection cannot be calculated
    if np.any(norm_dot_dir == 0):
        logging.warning('Some line plane intersection could not be calculated')

    # Calculate t scalar for each line simultaniously, avoid division by 0
    sel = norm_dot_dir != 0
    t[sel] = norm_dot_off[sel] / norm_dot_dir[sel]

    # Calculate the intersections for each line with the plane
    intersections = line_origins + line_directions * t[:, np.newaxis]

    return intersections


def get_z_point_on_plane(point, position_plane, normal_plane):
    ''' Calculates the (missing) z coordinate of a point (x,y) which lies in the plane.

    Parameters
    ----------
    point : array
        A point (x, y) which lies in the plane.
    position_plane : array
        A array (x, y and z) to the plane.
    normal_plane : array
        The normal vector (x, y and z) of the plane.

    Returns
    -------
    Array with z coordiate of point in the plane.
    '''
    A1 = normal_plane[0] * (point[0] - position_plane[0])
    A2 = normal_plane[1] * (point[1] - position_plane[1])

    z_coord = (normal_plane[2] * position_plane[2] - A1 - A2) / normal_plane[2]
    A3 = normal_plane[2]*(z_coord - position_plane[2])

    if not np.allclose(A1 + A2 + A3, 0.0):
        logging.warning('Some points do not lie in specified plane!')
    return z_coord



def cartesian_to_spherical(x, y, z):
    ''' Does a transformation from cartesian to spherical coordinates.

    Convention: r = 0 --> phi = theta = 0

    Parameters
    ----------
    x, y, z : float
        Position in cartesian space.

    Returns
    -------
    Spherical coordinates phi, theta and r.
    '''
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.zeros_like(r)  # define phi = 0 for x = 0
    theta = np.zeros_like(r)  # theta = 0 for r = 0
    # Avoid division by zero
    # https://en.wikipedia.org/wiki/Atan2
    phi[x != 0] = np.arctan2(y[x != 0], x[x != 0])
    phi[phi < 0] += 2. * np.pi  # map to phi = [0 .. 2 pi[
    theta[r != 0] = np.arccos(z[r != 0] / r[r != 0])
    return phi, theta, r


def spherical_to_cartesian(phi, theta, r):
    ''' Transformation from spherical to cartesian coordinates.

    Including error checks.

    Parameters
    ----------
    phi, theta, r : float
        Position in spherical space.

    Returns
    -------
    Cartesian coordinates x, y and z.
    '''
    if np.any(r < 0):
        raise RuntimeError('Conversion from spherical to cartesian coordinates failed, because r < 0')
    if np.any(theta < 0) or np.any(theta >= np.pi):
        raise RuntimeError('Conversion from spherical to cartesian coordinates failed, because theta exceeds [0, Pi[')
    if np.any(phi < 0) or np.any(phi >= 2 * np.pi):
        raise RuntimeError('Conversion from spherical to cartesian coordinates failed, because phi exceeds [0, 2*Pi[')
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)

    return x, y, z


def rotation_matrix_x(alpha):
    ''' Calculates the rotation matrix for the rotation around the x axis by an angle alpha in a cartesian right-handed coordinate system.

    Note
    ----
    Rotation in a cartesian right-handed coordinate system.

    Parameters
    ----------
    alpha : float
        Angle in radians.

    Returns
    -------
    Array with shape (3, 3).
    '''
    return np.array([[1, 0, 0],
                     [0, np.cos(alpha), np.sin(alpha)],
                     [0, -np.sin(alpha), np.cos(alpha)]])


def rotation_matrix_y(beta):
    ''' Calculates the rotation matrix for the rotation around the y axis by an angle beta in a cartesian right-handed coordinate system.

    Note
    ----
    Rotation in a cartesian right-handed coordinate system.


    Parameters
    ----------
    beta : float
        Angle in radians.

    Returns
    -------
    Array with shape (3, 3).
    '''
    return np.array([[np.cos(beta), 0, -np.sin(beta)],
                     [0, 1, 0],
                     [np.sin(beta), 0, np.cos(beta)]])


def rotation_matrix_z(gamma):
    ''' Calculates the rotation matrix for the rotation around the z axis by an angle gamma in a cartesian right-handed coordinate system.

    Note
    ----
    Rotation in a cartesian right-handed coordinate system.


    Parameters
    ----------
    gamma : float
        Angle in radians.

    Returns
    -------
    Array with shape (3, 3).
    '''
    return np.array([[np.cos(gamma), np.sin(gamma), 0],
                     [-np.sin(gamma), np.cos(gamma), 0],
                     [0, 0, 1]])


def rotation_matrix(alpha, beta, gamma):
    ''' Calculates the rotation matrix for the rotation around the three cartesian axis x, y, z in a right-handed system.

    Note
    ----
    In a right-handed system. The rotation is done around x then y then z.

    Remember:
        - Transform to the locale coordinate system before applying rotations
        - Rotations are associative but not commutative

    Usage
    -----
        A rotation by (alpha, beta, gamma) of the vector (x, y, z) in the local
        coordinate system can be done by:
          np.dot(rotation_matrix(alpha, beta, gamma), np.array([x, y, z]))


    Parameters
    ----------
    alpha : float
        Angle in radians for rotation around x.
    beta : float
        Angle in radians for rotation around y.
    gamma : float
        Angle in radians for rotation around z.

    Returns
    -------
    Array with shape (3, 3).
    '''
    return np.linalg.multi_dot([rotation_matrix_x(alpha=alpha), rotation_matrix_y(beta=beta), rotation_matrix_z(gamma=gamma)])


def euler_angles(R):
    ''' Calculates the Euler angles from rotation matrix R.

    Note
    ----
    In a right-handed system. The rotation is done around x then y then z.

    Remember:
        - Transform to the locale coordinate system before applying rotations
        - Rotations are associative but not commutative

    Parameters
    ----------
    R : array
        Rotation matrix.

    Returns
    -------
    Returns Euler angles alpha, beta and gamma.
    '''
    def is_rotation_matrix(R):
        norm = np.linalg.norm(np.identity(3, dtype=R.dtype) - np.dot(R.T, R))
        return np.isclose(0.0, norm)

    if not is_rotation_matrix(R):
        raise ValueError("%s is not a rotation matrix" % str(R))
    
#     sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
#     
#     singular = np.isclose(0.0, sy)

#     if not singular:
#         alpha = -math.atan2(R[2,1], R[2,2])
#         beta = -math.atan2(-R[2,0], sy)
#         gamma = -math.atan2(R[1,0], R[0,0])
#     else:
#         alpha = -math.atan2(-R[1,2], R[1,1])
#         beta = -math.atan2(-R[2,0], sy)
#         gamma = 0.0


    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
     
    singular = np.isclose(0.0, sy)

    if not singular:
        alpha = math.atan2(R[1,0], R[0,0])
        beta = math.atan2(-R[2,0], sy)
        gamma = math.atan2(R[2,1], R[2,2])
    else:
        raise
        alpha = math.atan2(-R[1,2], R[1,1])
        beta = math.atan2(-R[2,0], sy)
        gamma = 0.0

    return alpha, beta, gamma


def euler_angles_minimizer(R, alpha_start=0.0, beta_start=0.0, gamma_start=0.0, limit=0.5, tol=None):
    print "R", R.dtype
    def _minimize_me(angles, R):
        calculated_R = rotation_matrix(alpha=angles[0],
                                       beta=angles[1],
                                       gamma=angles[2])

        return np.linalg.norm(np.identity(3, dtype=R.dtype) - np.dot(calculated_R.T, R), ord=None)

    bounds = [(alpha_start - limit * np.pi, alpha_start + limit * np.pi), (beta_start - limit * np.pi, beta_start + limit * np.pi), (gamma_start - limit * np.pi, gamma_start + limit * np.pi)]
#     result = minimize(fun=_minimize_me,
#                       x0=np.array([alpha_start, beta_start, gamma_start]),  # Start values from residual fit
#                       args=(R,),
#                       tol=tol,
#                       bounds=bounds,
#                       method='L-BFGS-B')
#     if not result.success:
#         raise ValueError("Couldn't find Euler angles for rotation matrix")
    result = basinhopping(func=_minimize_me,
                          x0=np.array([alpha_start, beta_start, gamma_start], dtype=np.double),  # Start values from residual fit
                          niter=1000,
                          T=1e-6,
                          stepsize=1e-3,
                          interval=10,
                          niter_success=50,
                          minimizer_kwargs={"args": (R,), "method": 'TNC', "bounds": bounds, "tol": tol})
#     result = minimize(fun=_minimize_me,
#                       x0=[alpha_start, beta_start, gamma_start],
#                       args=(R,),
#                       method='SLSQP',
#                       bounds=bounds,
#                       options={"ftol": 1e-8, "maxiter":1000, 'eps': 1e-10})
#     if not result.success:
#         raise ValueError("Couldn't find Euler angles for rotation matrix")
    alpha, beta, gamma = result.x
    return alpha, beta, gamma


def translation_matrix(x, y, z):
    ''' Calculates the translation matrix for the translation in x, y, z in a cartesian right-handed system.

    Note
    ----
    Remember: Translations are associative and commutative

    Usage
    -----
        A translation of a vector (x, y, z) by dx, dy, dz can be done by:
          np.dot(translation_matrix(dx, dy, dz), np.array([x, y, z, 1]))


    Parameters
    ----------
    x : float
        Translation in x.
    y : float
        Translation in y.
    z : float
        Translation in z.

    Returns
    -------
    Array with shape (4, 4).
    '''
    translation_matrix = np.eye(4, 4, 0)
    translation_matrix[3, :3] = np.array([x, y, z])

    return translation_matrix.T


def global_to_local_transformation_matrix(x, y, z, alpha, beta, gamma):
    ''' Transformation matrix that applies a translation and rotation.


    Translation is T=(-x, -y, -z) to the local coordinate system followed
    by a rotation = R(alpha, beta, gamma).T in the local coordinate system.

    Note
    ----
        - This function is the inverse of
          local_to_global_transformation_matrix()
        - The resulting transformation matrix is 4 x 4
        - Translation and Rotation operations are not commutative

    Parameters
    ----------
    x : float
        Translation in x.
    y : float
        Translation in y.
    z : float
        Translation in z.
    alpha : float
        Angle in radians for rotation around x.
    beta : float
        Angle in radians for rotation around y.
    gamma : float
        Angle in radians for rotation around z.

    Returns
    -------
    Array with shape (4, 4).
    '''
    # Extend rotation matrix R by one dimension
    R = np.eye(4, 4, 0)
    R[:3, :3] = rotation_matrix(alpha=alpha, beta=beta, gamma=gamma).T

    # Get translation matrix T
    T = translation_matrix(x=-x, y=-y, z=-z)

    return np.dot(R, T)


def local_to_global_transformation_matrix(x, y, z, alpha, beta, gamma):
    ''' Transformation matrix that applies a inverse translation and rotation.

    Inverse rotation in the local coordinate system followed by an inverse
    translation by x, y, z to the global coordinate system.

    Note
    ----
        - The resulting transformation matrix is 4 x 4
        - Translation and Rotation operations do not commutative

    Parameters
    ----------
    x : float
        Translation in x.
    y : float
        Translation in y.
    z : float
        Translation in z.
    alpha : float
        Angle in radians for rotation around x.
    beta : float
        Angle in radians for rotation around y.
    gamma : float
        Angle in radians for rotation around z.

    Returns
    -------
    Array with shape (4, 4).
    '''
    # Extend inverse rotation matrix R by one dimension
    R = np.eye(4, 4, 0)
    R[:3, :3] = rotation_matrix(alpha=alpha, beta=beta, gamma=gamma)

    # Get inverse translation matrix T
    T = translation_matrix(x=x, y=y, z=z)

    return np.dot(T, R)


def apply_transformation_matrix(x, y, z, transformation_matrix):
    ''' Takes arrays for x, y, z and applies a transformation matrix (4 x 4).

    Parameters
    ----------
    x : array
        Array of x coordinates.
    y : array
        Array of y coordinates.
    z : array
        Array of z coordinates.

    Returns
    -------
    Array with transformed coordinates.
    '''
    # Add extra 4th dimension
    pos = np.column_stack((x, y, z, np.ones_like(x))).T

    # Transform and delete extra dimension
    pos_transformed = np.dot(transformation_matrix, pos).T[:, :-1]

    return pos_transformed[:, 0], pos_transformed[:, 1], pos_transformed[:, 2]


def apply_rotation_matrix(x, y, z, rotation_matrix):
    ''' Takes array in x, y, z and applies a rotation matrix (3 x 3).

    Parameters
    ----------
    x : array
        Array of x coordinates.
    y : array
        Array of x coordinates.
    z : array
        Array of x coordinates.

    Returns
    -------
    Array with rotated coordinates.
    '''
    pos = np.column_stack((x, y, z)).T
    pos_transformed = np.dot(rotation_matrix, pos).T

    return pos_transformed[:, 0], pos_transformed[:, 1], pos_transformed[:, 2]


def apply_alignment(hits_x, hits_y, hits_z, dut_index,
                    hits_xerr=None, hits_yerr=None, hits_zerr=None,
                    alignment=None, prealignment=None, inverse=False, beam_alignment=None):
    ''' Takes hits with errors and applies a transformation according to the alignment data.

    If alignment data with rotations and translations are given the hits are
    transformed according to the rotations and translations.
    If pre-alignment data with offsets and slopes are given the hits are
    transformed according to the slopes and offsets.
    If both are given alignment data is taken.
    The transformation can be inverted.

    Parameters
    ---------
    hits_x, hits_y, hits_z : array
        Array(s) with hit positions.
    hits_xerr, hits_yerr, hits_zerr : array
        Array(s) with hit errors.
    dut_index : int
        Needed to select the corrct alignment info.
    alignment : array
        Alignment information with rotations and translations.
    prealignment : array
        Pre-alignment information with offsets and slopes.
    inverse : bool
        Apply inverse transformation if True.

    Returns
    -------
    hits_x, hits_y, hits_z : array
        Array with transformed hit positions.
    '''
    if (alignment is None and prealignment is None) or \
       (alignment is not None and prealignment is not None):
        raise RuntimeError('Neither pre-alignment or alignment data given.')

    if beam_alignment is not None and prealignment is not None:
        raise RuntimeError('Beam-alignment cannot be used with pre-alignment.')

    if alignment is not None:
        if inverse:
            logging.debug('Transform hit position into the local coordinate '
                          'system using alignment data')
            transformation_matrix = global_to_local_transformation_matrix(
                x=alignment[dut_index]['translation_x'],
                y=alignment[dut_index]['translation_y'],
                z=alignment[dut_index]['translation_z'],
                alpha=alignment[dut_index]['alpha'],
                beta=alignment[dut_index]['beta'],
                gamma=alignment[dut_index]['gamma'])

            rotation_matrix = global_to_local_transformation_matrix(
                x=0.0,
                y=0.0,
                z=0.0,
                alpha=alignment[dut_index]['alpha'],
                beta=alignment[dut_index]['beta'],
                gamma=alignment[dut_index]['gamma'])
            if beam_alignment is not None:
                beam_transformation_matrix = global_to_local_transformation_matrix(
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    alpha=beam_alignment[0]['beam_alpha'],
                    beta=beam_alignment[0]['beam_beta'],
                    gamma=0.0)
                beam_rotation_matrix = global_to_local_transformation_matrix(
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    alpha=beam_alignment[0]['beam_alpha'],
                    beta=beam_alignment[0]['beam_beta'],
                    gamma=0.0)

            # reverse order
            if beam_alignment is not None:
                hits_x, hits_y, hits_z = apply_transformation_matrix(
                    x=hits_x,
                    y=hits_y,
                    z=hits_z,
                    transformation_matrix=beam_transformation_matrix)

            hits_x, hits_y, hits_z = apply_transformation_matrix(
                x=hits_x,
                y=hits_y,
                z=hits_z,
                transformation_matrix=transformation_matrix)

            if hits_xerr is not None and hits_yerr is not None and hits_zerr is not None:
                # Errors need only rotation but no translation
                if beam_alignment is not None:
                    hits_xerr, hits_yerr, hits_zerr = apply_transformation_matrix(
                        x=hits_xerr,
                        y=hits_yerr,
                        z=hits_zerr,
                        transformation_matrix=beam_rotation_matrix)

                hits_xerr, hits_yerr, hits_zerr = apply_transformation_matrix(
                    x=hits_xerr,
                    y=hits_yerr,
                    z=hits_zerr,
                    transformation_matrix=rotation_matrix)
        else:
            logging.debug('Transform hit position into the global coordinate '
                          'system using alignment data')
            transformation_matrix = local_to_global_transformation_matrix(
                x=alignment[dut_index]['translation_x'],
                y=alignment[dut_index]['translation_y'],
                z=alignment[dut_index]['translation_z'],
                alpha=alignment[dut_index]['alpha'],
                beta=alignment[dut_index]['beta'],
                gamma=alignment[dut_index]['gamma'])
            rotation_matrix = local_to_global_transformation_matrix(
                x=0.0,
                y=0.0,
                z=0.0,
                alpha=alignment[dut_index]['alpha'],
                beta=alignment[dut_index]['beta'],
                gamma=alignment[dut_index]['gamma'])
            if beam_alignment is not None:
                beam_transformation_matrix = local_to_global_transformation_matrix(
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    alpha=beam_alignment[0]['beam_alpha'],
                    beta=beam_alignment[0]['beam_beta'],
                    gamma=0.0)
                beam_rotation_matrix = local_to_global_transformation_matrix(
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    alpha=beam_alignment[0]['beam_alpha'],
                    beta=beam_alignment[0]['beam_beta'],
                    gamma=0.0)

            hits_x, hits_y, hits_z = apply_transformation_matrix(
                x=hits_x,
                y=hits_y,
                z=hits_z,
                transformation_matrix=transformation_matrix)

            if beam_alignment is not None:
                hits_x, hits_y, hits_z = apply_transformation_matrix(
                    x=hits_x,
                    y=hits_y,
                    z=hits_z,
                    transformation_matrix=beam_transformation_matrix)

            if hits_xerr is not None and hits_yerr is not None and hits_zerr is not None:
                # Errors need only rotation but no translation
                hits_xerr, hits_yerr, hits_zerr = apply_transformation_matrix(
                    x=hits_xerr,
                    y=hits_yerr,
                    z=hits_zerr,
                    transformation_matrix=rotation_matrix)

                if beam_alignment is not None:
                    hits_xerr, hits_yerr, hits_zerr = apply_transformation_matrix(
                        x=hits_xerr,
                        y=hits_yerr,
                        z=hits_zerr,
                        transformation_matrix=beam_rotation_matrix)

    else:
        c0_column = prealignment[dut_index]['column_c0']
        c1_column = prealignment[dut_index]['column_c1']
        c0_row = prealignment[dut_index]['row_c0']
        c1_row = prealignment[dut_index]['row_c1']
        z = prealignment[dut_index]['z']

        if inverse:
            logging.debug('Transform hit position into the local coordinate '
                          'system using pre-alignment data')
            hits_x = (hits_x - c0_column) / c1_column
            hits_y = (hits_y - c0_row) / c1_row
            hits_z = hits_z - z

            if hits_xerr is not None and hits_yerr is not None and hits_zerr is not None:
                hits_xerr = hits_xerr / c1_column
                hits_yerr = hits_yerr / c1_row
        else:
            logging.debug('Transform hit position into the global coordinate '
                          'system using pre-alignment data')
            hits_x = c1_column * hits_x + c0_column
            hits_y = c1_row * hits_y + c0_row
            hits_z = hits_z + z

            if hits_xerr is not None and hits_yerr is not None and hits_zerr is not None:
                hits_xerr = c1_column * hits_xerr
                hits_yerr = c1_row * hits_yerr

    if hits_xerr is not None and hits_yerr is not None and hits_zerr is not None:
        return hits_x, hits_y, hits_z, hits_xerr, hits_yerr, hits_zerr

    return hits_x, hits_y, hits_z


def apply_alignment_to_hits(hits, dut_index, use_prealignment, alignment, inverse, no_z, beam_alignment=None):
    if use_prealignment:  # Apply transformation from pre-alignment information
        (hits['x_dut_%d' % dut_index],
         hits['y_dut_%d' % dut_index],
         hit_z,
         hits['xerr_dut_%d' % dut_index],
         hits['yerr_dut_%d' % dut_index],
         hits['zerr_dut_%d' % dut_index]) = apply_alignment(
            hits_x=hits['x_dut_%d' % dut_index],
            hits_y=hits['y_dut_%d' % dut_index],
            hits_z=hits['z_dut_%d' % dut_index],
            hits_xerr=hits['xerr_dut_%d' % dut_index],
            hits_yerr=hits['yerr_dut_%d' % dut_index],
            hits_zerr=hits['zerr_dut_%d' % dut_index],
            dut_index=dut_index,
            prealignment=alignment,
            beam_alignment=beam_alignment,
            inverse=inverse)
    else:  # Apply transformation from fine alignment information
        (hits['x_dut_%d' % dut_index],
         hits['y_dut_%d' % dut_index],
         hit_z,
         hits['xerr_dut_%d' % dut_index],
         hits['yerr_dut_%d' % dut_index],
         hits['zerr_dut_%d' % dut_index]) = apply_alignment(
            hits_x=hits['x_dut_%d' % dut_index],
            hits_y=hits['y_dut_%d' % dut_index],
            hits_z=hits['z_dut_%d' % dut_index],
            hits_xerr=hits['xerr_dut_%d' % dut_index],
            hits_yerr=hits['yerr_dut_%d' % dut_index],
            hits_zerr=hits['zerr_dut_%d' % dut_index],
            dut_index=dut_index,
            alignment=alignment,
            beam_alignment=beam_alignment,
            inverse=inverse)
    if not no_z:
        hits['z_dut_%d' % dut_index] = hit_z


def get_alignment_parameters(alignment, inverse=False):
    n_duts = alignment.shape[0]

    rotation = [None] * n_duts
    translation = [None] * n_duts

    for dut_index in range(n_duts):
        x=alignment[dut_index]['translation_x']
        y=alignment[dut_index]['translation_y']
        z=alignment[dut_index]['translation_z']
        alpha=alignment[dut_index]['alpha']
        beta=alignment[dut_index]['beta']
        gamma=alignment[dut_index]['gamma']

        if inverse:
            transformation_matrix = geometry_utils.global_to_local_transformation_matrix(
                x=alignment[dut_index]['translation_x'],
                y=alignment[dut_index]['translation_y'],
                z=alignment[dut_index]['translation_z'],
                alpha=alignment[dut_index]['alpha'],
                beta=alignment[dut_index]['beta'],
                gamma=alignment[dut_index]['gamma'])
    
            rotation_dut = geometry_utils.rotation_matrix(alpha=alpha, beta=beta, gamma=gamma).T
            translation_dut = np.array([-x, -y, -z])
        else:
            transformation_matrix = geometry_utils.local_to_global_transformation_matrix(
                x=alignment[dut_index]['translation_x'],
                y=alignment[dut_index]['translation_y'],
                z=alignment[dut_index]['translation_z'],
                alpha=alignment[dut_index]['alpha'],
                beta=alignment[dut_index]['beta'],
                gamma=alignment[dut_index]['gamma'])
    
            rotation_dut = geometry_utils.rotation_matrix(alpha=alpha, beta=beta, gamma=gamma)
            translation_dut = np.array([x, y, z])
            
        rotation[dut_index] = rotation_dut
        translation[dut_index] = translation_dut

    return rotation, translation


def merge_alignment_parameters(old_alignment, new_alignment, mode='absolute',
                               select_duts=None, parameters=None):
    if select_duts is None:  # Select all DUTs
        select_duts = range(old_alignment.shape[0])
    else:
        if not isinstance(select_duts, Iterable):
            select_duts = list(select_duts)

    all_parameters = set(["alpha", "beta", "gamma", "translation_x", "translation_y", "translation_z"])
    if parameters is None:
        parameters = all_parameters
    else:
        if not isinstance(parameters, Iterable):
            parameters = list(parameters)
        unknown_parameters = set(parameters) - all_parameters  # difference
        if unknown_parameters:
            raise ValueError("Unknown parameter(s): %s" % ", ".join(unknown_parameters))

    alignment_parameters = old_alignment.copy()  # Do not change input parameters

    for dut in select_duts:
        for parameter in parameters:
            if mode == 'absolute':
                alignment_parameters[dut][parameter] = new_alignment[dut][parameter]
            elif mode == 'relative':
                alignment_parameters[dut][parameter] += new_alignment[dut][parameter]
            else:
                raise RuntimeError('Unknown mode %s', str(mode))
    return alignment_parameters


def save_alignment_parameters(alignment_file, alignment_parameters,
                              mode='absolute', select_duts=None, parameters=None):
    ''' Saving alignment parameters (rotations, translations) to file.

    Absolute (overwriting) and relative (add angles, translations) supported.

    Parameters
    ----------
    alignment_file : string
        The pytables file name containing the alignment.
    alignment_parameters : recarray
        An array with the alignment values.
    mode : string
        Select relative or absolute alignment. The strings 'relative' and 'absolute' are supported.
    use_duts : iterable
        In relative mode only change specified DUTs.
    parameters : iterable
        Iterable of alignment parameters to be stored. Possible values are
        "alpha", "beta", "gamma", "translation_x", "translation_y" and "translation_z".
        If None, select all alignment parameters.
    '''
    # Open file with alignment data
    with tb.open_file(alignment_file, mode="r+") as out_file_h5:
        try:
            old_alignment = out_file_h5.root.Alignment[:]
        except tb.NodeError:
            if mode != 'absolute':
                raise ValueError("No existing alignment found. Mode %s will not work." % (mode,))
        else:
            logging.info('Overwrite existing alignment (Mode: %s).' % (mode,))
            out_file_h5.root.Alignment._f_remove()
            alignment_parameters = merge_alignment_parameters(
                old_alignment=old_alignment,
                new_alignment=alignment_parameters,
                mode=mode,
                select_duts=select_duts,
                parameters=parameters)
        finally:
            alignment_table = out_file_h5.create_table(out_file_h5.root, name='Alignment',
                                                       title='Transformation parameters',
                                                       description=alignment_parameters.dtype)
            alignment_table.append(alignment_parameters)


        string = "\n".join(['DUT%d: alpha/beta/gamma=%1.4f/%1.4f/%1.4f Rad, '
                            'x/y/z=%.2f/%.2f/%.2f um' % (dut_values['DUT'],
                                                         dut_values['alpha'],
                                                         dut_values['beta'],
                                                         dut_values['gamma'],
                                                         dut_values['translation_x'],
                                                         dut_values['translation_y'],
                                                         dut_values['translation_z'])
                            for dut_values in alignment_parameters])
        logging.info('Setting alignment parameters to:\n%s' % string)


def load_alignment_parameters(alignment_file):
    ''' Loading alignment parameters (rotations, translations) from file.

    Parameters
    ----------
    alignment_file : string
        The pytables file name containing the alignment.
    '''
    # Open file with alignment data
    with tb.open_file(alignment_file, mode="r") as out_file_h5:
        alignment = out_file_h5.root.Alignment[:]
    
    return alignment


def save_beam_alignment_parameters(alignment_file, beam_alignment):
    with tb.open_file(alignment_file, mode="r+") as out_file_h5:
        try:
            old_alignment = out_file_h5.root.BeamAlignment[:]
        except tb.NodeError:
            pass
        else:
            logging.info('Overwrite existing beam-alignment')
            out_file_h5.root.BeamAlignment._f_remove()
        finally:
            alignment_table = out_file_h5.create_table(out_file_h5.root, name='BeamAlignment',
                                                       title='Beam axis parameters',
                                                       description=beam_alignment.dtype)
            alignment_table.append(beam_alignment)


        logging.info('Setting beam-alignment parameters to: alpha/beta=%1.4f/%1.4f Rad' % (beam_alignment[0]['beam_alpha'],
                                                                                           beam_alignment[0]['beam_beta']))


def load_beam_alignment_parameters(alignment_file):
    with tb.open_file(alignment_file, mode="r") as out_file_h5:
        beam_alignment = out_file_h5.root.BeamAlignment[:]
    
    return beam_alignment