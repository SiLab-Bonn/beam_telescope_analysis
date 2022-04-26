''' Helper functions for the unittests are defined here.
'''
from __future__ import division

import os
import inspect
import itertools
import yaml

import numpy as np
import tables as tb

FIXTURE_FOLDER = 'fixtures'


def nan_to_num(array, copy=False):
    ''' Like np.nan_to_num but also works on recarray

    Parameters
    ----------
    array : numpy.ndarray
    copy : boolean
        If True, return copy. If False, replace values in-place.

    Returns
    -------
    boolean
    '''
    if array.dtype.names is None:  # normal nd.array
        return np.nan_to_num(array, copy=copy)
    else:
        if copy:
            array = np.copy(array)
        for column_name in array.dtype.names:
            np.nan_to_num(array[column_name], copy=False)
        return array


def nan_equal(first_array, second_array):
    ''' Compares two arrays and test for equality.

    Works with array and recarrays.
    NaNs are considered equal.

    Parameters
    ----------
    first_array : numpy.ndarray
    second_array : numpy.ndarray

    Returns
    -------
    boolean
    '''
    # Check for shape, prevent broadcast
    if first_array.shape != second_array.shape:
        return False
    # Check if both are recarrays
    if (first_array.dtype.names is None and second_array.dtype.names is not None) or (first_array.dtype.names is not None and second_array.dtype.names is None):
        return False
    if first_array.dtype.names is None:  # Not a recarray
        # Check for same dtypes
        if first_array.dtype != second_array.dtype:
            return False
        # Check for equality
        try:
            np.testing.assert_equal(first_array, second_array)
        except AssertionError:
            return False
    else:
        # Check for same column names and same order
        if first_array.dtype.names != second_array.dtype.names:
            return False
        for column in first_array.dtype.names:
            # Check for same dtypes
            if first_array[column].dtype != second_array[column].dtype:
                return False
            # Check for equality
            try:
                np.testing.assert_equal(first_array[column], second_array[column])
            except AssertionError:
                return False
    return True


def nan_close(first_array, second_array, rtol=1e-5, atol=1e-8, equal_nan=True):
    ''' Compares two arrays and test for similarity.

    Works with recarrays.

    Parameters
    ----------
    first_array : numpy.ndarray
    second_array : numpy.ndarray
    rtol : float
    atol : float
    equal_nan : boolean
        If True, NaNs are considered equal.

    Returns
    -------
    boolean
    '''
    # Check for shape, prevent broadcast
    if first_array.shape != second_array.shape:
        return False
    # Check if both are recarrays
    if (first_array.dtype.names is None and second_array.dtype.names is not None) or (first_array.dtype.names is not None and second_array.dtype.names is None):
        return False
    if first_array.dtype.names is None:  # Not a recarray
        # Check for same dtypes
        if first_array.dtype != second_array.dtype:
            return False
        return np.allclose(a=first_array, b=second_array, rtol=rtol, atol=atol, equal_nan=equal_nan)
    else:
        # Check for same column names and same order
        if first_array.dtype.names != second_array.dtype.names:
            return False
        for column in first_array.dtype.names:
            # Check for same dtypes
            if first_array[column].dtype != second_array[column].dtype:
                return False
            # Workaround for string data types
            if first_array[column].dtype.type is np.string_:
                return np.all(first_array[column] == second_array[column])
            # Check for similarity
            if not np.allclose(a=first_array[column], b=second_array[column], rtol=rtol, atol=atol, equal_nan=equal_nan):
                return False
        return True


def get_array_differences(first_array, second_array, exact=True, rtol=1e-5, atol=1e-8, equal_nan=True):
    '''Takes two numpy.ndarrays and compares them on a column basis.
    Different column data types, missing columns and columns with different values are returned in a string.

    Parameters
    ----------
    first_array : numpy.ndarray
    second_array : numpy.ndarray

    Returns
    -------
    string
    '''
    def compare_arrays(actual, desired, exact, rtol, atol, equal_nan):
        compare_str = ''
        if actual.dtype != desired.dtype:
            compare_str += ' Type:\n  first: %s\n  second: %s\n' % (str(actual.dtype), str(desired.dtype))
        if actual.shape != desired.shape:
            compare_str += ' Shape:\n  first: %s\n  second: %s\n' % (str(actual.shape), str(desired.shape))
            try:  # Try reshaping, is possible when changed dimension has only one setting
                actual = actual.reshape(desired.shape)
            except ValueError:
                pass
        try:
            if np.nansum(actual) != np.nansum(desired):
                compare_str += ' Sum:\n  first: %s\n  second: %s\n' % (str(np.nansum(actual)), str(np.nansum(desired)))
        except TypeError:  # cannot perform reduce with flexible type
            pass
        if exact:
            try:
                np.testing.assert_equal(actual=actual, desired=desired)
            except AssertionError as e:
                compare_str += str(e) + "\n"
        else:
            try:
                np.testing.assert_allclose(actual=actual, desired=desired, rtol=rtol, atol=atol, equal_nan=equal_nan)
            except AssertionError as e:
                compare_str += str(e) + "\n"
        if compare_str:
            compare_str = ("Difference (%s):\n" % ("exact" if exact else "close")) + compare_str
        else:
            compare_str = "No Difference (%s)\n" % ("exact" if exact else "close")
        return compare_str

    # Check if both are recarrays
    if (first_array.dtype.names is None and second_array.dtype.names is not None) or (first_array.dtype.names is not None and second_array.dtype.names is None):
        return "Type mismatch: np.array and np.recarray"
    if first_array.dtype.names is None:  # Not a recarray
        return compare_arrays(actual=first_array, desired=second_array, exact=exact, rtol=rtol, atol=atol, equal_nan=equal_nan)
    else:
        return_str = ''
        first_array_column_names = first_array.dtype.names
        second_array_column_names = second_array.dtype.names
        additional_first_array_column_names = set(first_array_column_names) - set(second_array_column_names)
        additional_second_array_column_names = set(second_array_column_names) - set(first_array_column_names)
        if additional_first_array_column_names:
            return_str += 'First array has additional columns: %s\n' % ', '.join(additional_first_array_column_names)
        if additional_second_array_column_names:
            return_str += 'Second array has additional columns: %s\n' % ', '.join(additional_second_array_column_names)
        if not additional_first_array_column_names and not additional_second_array_column_names and first_array_column_names != second_array_column_names:
            return_str += 'Columns have different order:\nfirst: %s\nsecond: %s\n' % (first_array_column_names, second_array_column_names)
        common_columns = set(first_array_column_names) & set(second_array_column_names)
        for column_name in common_columns:  # loop over all nodes and compare each node, do not abort if one node is wrong
            first_column_data = first_array[column_name]
            second_column_data = second_array[column_name]
            col_compare_str = compare_arrays(actual=first_column_data, desired=second_column_data, exact=exact, rtol=rtol, atol=atol, equal_nan=equal_nan)
            return_str += "Column %s:\n%s" % (column_name, col_compare_str)
        return return_str


def compare_h5_files(first_file, second_file, node_names=None, ignore_nodes=None, detailed_comparison=True, exact=True, rtol=1e-5, atol=1e-8, chunk_size=1000000):
    '''Takes two hdf5 files and check for equality of all nodes.
    Returns true if the node data is equal and the number of nodes is the number of expected nodes.
    It also returns an error string containing the names of the nodes that are not equal.

    Parameters
    ----------
    first_file : string
        Path to the first file.
    second_file : string
        Path to the second file.
    node_names : list, tuple
        Iterable of node names that are required to exist and will be compared.
        If None, compare all existing nodes and fail if nodes are not existing.
        This is a white list of nodes to check.
    ignore_nodes : list, tuple
        Iterable of node names that are not required to exist and will not be compared.
        If None, no existing nodes is excluded. This is a black list of nodes to check.
    detailed_comparison : boolean
        Print reason why the comparison failed
    exact : boolean
        True if the results have to match exactly. E.g. False for fit results.
    rtol, atol: number
        From numpy.allclose:
        rtol : float
            The relative tolerance parameter (see Notes).
        atol : float
            The absolute tolerance parameter (see Notes).

    Returns
    -------
    (bool, string)
    '''

    checks_passed = True
    error_msg = ""
    with tb.open_file(first_file, 'r') as first_h5_file:
        with tb.open_file(second_file, 'r') as second_h5_file:

            def walk_nodes(f, n, g="/"):
                for item in f.get_node(f.root, g):
                    if isinstance(item, tb.group.Group):
                        walk_nodes(f=f, n=n, g=item._v_pathname)
                    else:
                        n.append(item._v_pathname)

            fist_file_nodes = []
            walk_nodes(f=first_h5_file, n=fist_file_nodes)  # get node names
            second_file_nodes = []
            walk_nodes(f=second_h5_file, n=second_file_nodes)  # get node names

            if ignore_nodes:
                fist_file_nodes = [node for node in fist_file_nodes if node not in ignore_nodes]
                second_file_nodes = [node for node in second_file_nodes if node not in ignore_nodes]

            if node_names is None:
                additional_first_file_nodes = set(fist_file_nodes) - set(second_file_nodes)
                additional_second_file_nodes = set(second_file_nodes) - set(fist_file_nodes)
                if additional_first_file_nodes:
                    checks_passed = False
                    if detailed_comparison:
                        error_msg += 'File %s has additional nodes: %s\n' % (first_file, ', '.join(additional_first_file_nodes))
                if additional_second_file_nodes:
                    checks_passed = False
                    if detailed_comparison:
                        error_msg += 'File %s has additional nodes: %s\n' % (second_file, ', '.join(additional_second_file_nodes))
                common_nodes = set(fist_file_nodes) & set(second_file_nodes)
            else:
                node_names = [(("/" + name) if (name and name[:1] != "/") else name) for name in node_names]
                missing_first_file_nodes = set(node_names) - set(fist_file_nodes)
                if missing_first_file_nodes:
                    checks_passed = False
                    if detailed_comparison:
                        error_msg += 'File %s is missing nodes: %s\n' % (first_file, ', '.join(missing_first_file_nodes))
                missing_second_file_nodes = set(node_names) - set(second_file_nodes)
                if missing_second_file_nodes:
                    checks_passed = False
                    if detailed_comparison:
                        error_msg += 'File %s is missing nodes: %s\n' % (second_file, ', '.join(missing_second_file_nodes))
                common_nodes = (set(fist_file_nodes) & set(second_file_nodes)) & set(node_names)
            for node_name in common_nodes:  # loop over all nodes and compare each node, do not abort if one node is wrong
                nrows = first_h5_file.get_node(first_h5_file.root, node_name).nrows
                index_start = 0
                while index_start < nrows:
                    # reduce memory footprint by taken array dimension into account
                    read_nrows = max(1, int(chunk_size / np.prod(first_h5_file.get_node(first_h5_file.root, node_name).shape[1:])))
                    index_stop = index_start + read_nrows
                    first_file_data = first_h5_file.get_node(first_h5_file.root, node_name).read(index_start, index_stop)
                    second_file_data = second_h5_file.get_node(second_h5_file.root, node_name).read(index_start, index_stop)
                    if exact:
                        if not nan_equal(first_array=first_file_data, second_array=second_file_data):
                            checks_passed = False
                            if detailed_comparison:
                                error_msg += ('Node %s:\n' % node_name) + get_array_differences(first_array=first_file_data, second_array=second_file_data, exact=True)
                            break
                    else:
                        if not nan_close(first_array=first_file_data, second_array=second_file_data, rtol=rtol, atol=atol, equal_nan=True):
                            checks_passed = False
                            if detailed_comparison:
                                error_msg += ('Node %s:\n' % node_name) + get_array_differences(first_array=first_file_data, second_array=second_file_data, exact=False, rtol=rtol, atol=atol, equal_nan=True)
                            break
                    index_start += read_nrows
    if checks_passed:
        error_msg = 'Comparing files %s and %s: OK\n%s' % (first_file, second_file, error_msg)
    else:
        error_msg = 'Comparing files %s and %s: FAILED\n%s' % (first_file, second_file, error_msg)
    return checks_passed, error_msg


def compare_yaml_files(first_file, second_file):
    ''' Compare two yaml files and check if they are equal.
    '''
    def yaml_as_dict(input_file):
        yaml_dict = {}
        with open(input_file, 'r') as fp:
            docs = yaml.safe_load_all(fp)
            for doc in docs:
                for key, value in doc.items():
                    yaml_dict[key] = value
        return yaml_dict

    dict_1 = yaml_as_dict(first_file)
    dict_2 = yaml_as_dict(second_file)

    return dict_1 == dict_2


def _call_function_with_args(function, **kwargs):
    ''' Calls the function with the given kwargs
    and returns the result in a numpy array. All combinations
    of functions arguments in a list are used for multiple
    function calls.'''

    # Create all combinations of arguments from list parameters
    # This is ugly but avoids recursion and does effectively
    # a nested loop of n parameters:
    # for par_1 in pars_1:
    #  for par_2 in pars_2:
    #    ...
    #    for par_n in pars_n:
    #      function(par_1, par_2, ..., par_n)

    call_values = []  # Arguments with permutations
    fixed_arguments = []  # Constant arguments
    fixed_arguments_pos = []
    for index, values in enumerate(kwargs.values()):
        if isinstance(values, list):
            call_values.extend([values])
        else:
            fixed_arguments.append(values)
            fixed_arguments_pos.append(index)
    call_values = list(itertools.product(*call_values))

    data = []

    # Call functions with all parameter combinations
    for call_value in call_values:
        actual_call_value = list(call_value)
        for index, fixed_arg_pos in enumerate(fixed_arguments_pos):
            actual_call_value.insert(fixed_arg_pos, fixed_arguments[index])
        call_args = {
            key: value for key, value in zip(kwargs.keys(), actual_call_value)}
        data.append(function(**call_args))

    return data


def create_fixture(function, **kwargs):
    ''' Calls the function with the given kwargs values and stores the result.

    Numpy arrays are given as one parameter, lists parameters are looped with repeated
    function calls.
    '''

    # Check if all parameters are defined
    func_args = inspect.getargspec(function)[0]
    if not all([a in kwargs for a in func_args]):
        raise RuntimeError('Not all function arguments values defined')

    data = _call_function_with_args(function, **kwargs)

    # Store function return values in compressed pytable array
    data = np.array(data)
    with tb.open_file(os.path.join(FIXTURE_FOLDER, '%s.h5' % str(function.__name__)), mode='w') as out_file_h5:
        data_array = out_file_h5.create_carray(
            where=out_file_h5.root,
            name='Data',
            title='%s return values' % function.__name__,
            atom=tb.Atom.from_dtype(data.dtype),
            shape=data.shape,
            filters=tb.Filters(
                complib='blosc',
                complevel=5,
                fletcher32=False))
        data_array[:] = data


def check_with_fixture(function, **kwargs):
    ''' Calls the function with the given kwargs values and compares the result with the fixture.

    Numpy arrays are given as one parameter, lists parameters are looped with repeated
    function calls.
    '''

    with tb.open_file(os.path.join(FIXTURE_FOLDER, '%s.h5' % str(function.__name__)), mode='r') as in_file_h5:
        data_fixture = in_file_h5.root.Data[:]

    data = _call_function_with_args(function, **kwargs)

    return np.allclose(data_fixture, data)


def create_folder(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
