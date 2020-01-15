import logging
import time
import os
from functools import wraps

import tables as tb

from beam_telescope_analysis import __version__ as bta_version


# Python 3 compatibility
try:
    basestring
except NameError:
    basestring = str


class NameValue(tb.IsDescription):
    name = tb.StringCol(256, pos=0)
    value = tb.StringCol(32 * 1024, pos=1)


def save_configuration_dict(output_file, table_name, dictionary, group_name="configuration", date_created=None, **kwargs):
    '''Stores any configuration dictionary to HDF5 file.

    Parameters
    ----------
    output_file : string, file
        Filename of the output pytables file or file object.
    table_name : str
        The name will be used as table name.
    dictionary : dict
        A dictionary with key/value pairs.
    date_created : float, time.struct_time
        If None (default), the local time is used.
    '''
    def save_conf():
        try:
            h5_file.remove_node(where="/%s" % group_name, name=table_name)
        except tb.NodeError:
            pass
        try:
            configuration_group = h5_file.create_group(where="/", name=group_name)
        except tb.NodeError:
            configuration_group = h5_file.root.configuration

        scan_param_table = h5_file.create_table(where=configuration_group, name=table_name, description=NameValue, title=table_name)
        row_scan_param = scan_param_table.row
        for key, value in dictionary.items():
            row_scan_param['name'] = key
            row_scan_param['value'] = str(value)
            row_scan_param.append()
        if isinstance(date_created, float):
            scan_param_table.attrs.date_created = time.asctime(time.localtime(date_created))
        elif isinstance(date_created, time.struct_time):
            time.asctime(date_created)
        else:
            scan_param_table.attrs.date_created = time.asctime()
        scan_param_table.attrs.bta_version = bta_version
        scan_param_table.flush()

    if isinstance(output_file, tb.file.File):
        h5_file = output_file
        save_conf()
    else:
        mode = kwargs.pop("mode", "a")
        with tb.open_file(output_file, mode=mode, **kwargs) as h5_file:
            save_conf()


def save_arguments(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        curr_time = time.time()
        ret_val = func(*args, **kwargs)
        func_name = func.__name__
        output_files = None
        if isinstance(ret_val, basestring) and os.path.isfile(ret_val):
            output_files = (ret_val,)
        elif isinstance(ret_val, (list, tuple)):  # allow multiple return values
            if all(map(lambda item: isinstance(item, basestring), ret_val)) and all(map(os.path.isfile, ret_val)):  # all return values are files
                output_files = ret_val
            elif isinstance(ret_val[0], basestring) and os.path.isfile(ret_val[0]):  # first item is file
                output_files = (ret_val[0],)
            elif isinstance(ret_val[0], (list, tuple)) and all(map(lambda item: isinstance(item, basestring), ret_val[0])) and all(map(os.path.isfile, ret_val[0])):  # first item is list of files
                output_files = ret_val[0]
        if output_files:
            for output_file in output_files:
                all_parameters = func.func_code.co_varnames[:func.func_code.co_argcount]
                all_kwargs = dict(zip(all_parameters, args))
                all_kwargs.update(kwargs)
                save_configuration_dict(output_file=output_file, table_name=func_name, dictionary=all_kwargs, group_name="arguments", date_created=curr_time, mode="a")
        else:
            logging.warning("Invalid value(s) returned by \"%s()\": function arguments were not saved." % func_name)
        return ret_val
    return wrapper
