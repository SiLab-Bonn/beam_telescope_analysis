''' Helper functions for geometrical operations.
'''
from __future__ import division

import logging

import numpy as np


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


def get_line_intersections_with_dut(line_origins, line_directions, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma):
    ''' Calculates the intersection of lines with a DUT.

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
    translation_x, translation_y, translation_z : float
        The coordianates of the DUT (in um).
    rotation_alpha, rotation_beta, rotation_gamma : float
        The rotation angles of the DUT (in radians).

    Returns
    -------
    Array with shape (n, 3) with the intersection point.
    '''
    dut_position = np.array([
        translation_x,
        translation_y,
        translation_z],
        dtype=np.float64)
    dut_rotation_matrix = rotation_matrix(
        alpha=rotation_alpha,
        beta=rotation_beta,
        gamma=rotation_gamma)
    basis_global = dut_rotation_matrix.T.dot(np.eye(3, dtype=np.float64))
    dut_normal = basis_global[2]
    dut_normal /= np.sqrt(np.dot(dut_normal, dut_normal))
    if dut_normal[2] < 0:
        dut_normal = -dut_normal
    return get_line_intersections_with_plane(
        line_origins=line_origins,
        line_directions=line_directions,
        position_plane=dut_position,
        normal_plane=dut_normal)


def get_line_intersections_with_plane(line_origins, line_directions, position_plane, normal_plane):
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
    t = np.full_like(norm_dot_off, fill_value=np.nan, dtype=np.float64)

    # Warn if some intersection cannot be calculated
    if np.any(norm_dot_dir == 0):
        logging.warning('Some line plane intersection could not be calculated')

    # Calculate t scalar for each line simultaniously, avoid division by 0
    sel = norm_dot_dir != 0
    t[sel] = norm_dot_off[sel] / norm_dot_dir[sel]

    # Calculate the intersections for each line with the plane
    intersections = line_origins + line_directions * t[:, np.newaxis]

    return intersections


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
    phi = np.zeros_like(r, dtype=np.float64)  # define phi = 0 for x = 0
    theta = np.zeros_like(r, dtype=np.float64)  # theta = 0 for r = 0
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
                     [0, np.cos(alpha), -np.sin(alpha)],
                     [0, np.sin(alpha), np.cos(alpha)]],
                    dtype=np.float64)


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
    return np.array([[np.cos(beta), 0, np.sin(beta)],
                     [0, 1, 0],
                     [-np.sin(beta), 0, np.cos(beta)]],
                    dtype=np.float64)


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
    return np.array([[np.cos(gamma), -np.sin(gamma), 0],
                     [np.sin(gamma), np.cos(gamma), 0],
                     [0, 0, 1]],
                    dtype=np.float64)


def rotation_matrix(alpha, beta, gamma):
    ''' Calculates the rotation matrix for the rotation around the three cartesian
    axis x, y, z in a right-handed coordinate system.

    Note
    ----
    The matrix represents a rotation around the x axis, then the y axis,
    and then the z axis in a right-handed coordinate system.

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
    return np.linalg.multi_dot([rotation_matrix_z(gamma=gamma), rotation_matrix_y(beta=beta), rotation_matrix_x(alpha=alpha)])


def euler_angles(R):
    ''' Calculates the Euler angles from rotation matrix R.

    Note
    ----
    In a right-handed system. The rotation is done around x then y then z.

    Note:
        - Transform to the locale coordinate system before applying rotations
        - Rotations are associative but not commutative
        - In cases of beta = pi/2 and -pi/2, gamma and alpha are linked (gimbal lock).
          In this case, gamma is set to zero.

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

    if R[2, 0] == -1:
        gamma = 0.0  # gimbal lock
        alpha = gamma + np.arctan2(R[0, 1], R[0, 2])
        beta = np.pi / 2
    elif R[2, 0] == 1:
        gamma = 0.0  # gimbal lock
        alpha = -gamma + np.arctan2(-R[0, 1], -R[0, 2])
        beta = -np.pi / 2
    else:
        beta_1 = -np.arcsin(R[2, 0])
        beta_2 = np.pi - beta_1
        alpha_1 = np.arctan2(R[2, 1] / np.cos(beta_1), R[2, 2] / np.cos(beta_1))
        alpha_2 = np.arctan2(R[2, 1] / np.cos(beta_2), R[2, 2] / np.cos(beta_2))
        gamma_1 = np.arctan2(R[1, 0] / np.cos(beta_1), R[0, 0] / np.cos(beta_1))
        gamma_2 = np.arctan2(R[1, 0] / np.cos(beta_2), R[0, 0] / np.cos(beta_2))
        # chose the angles with smaller values
        if np.sum(np.abs([alpha_1, beta_1, gamma_1])) <= np.sum(np.abs([alpha_2, beta_2, gamma_2])):
            alpha, beta, gamma = alpha_1, beta_1, gamma_1
        else:
            alpha, beta, gamma = alpha_2, beta_2, gamma_2
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
    translation_matrix = np.eye(4, 4, 0, dtype=np.float64)
    translation_matrix[3, :3] = np.array([x, y, z], dtype=np.float64)

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
    R = np.eye(4, 4, 0, dtype=np.float64)
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
    R = np.eye(4, 4, 0, dtype=np.float64)
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
    pos = np.column_stack((x, y, z, np.ones_like(x, dtype=np.float64))).T

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
