''' Helper functions to select and combine data '''
from __future__ import division

import logging
import re
import os
from collections.abc import Iterable

import numpy as np
import tables as tb
import numexpr as ne

from tqdm import tqdm

from beam_telescope_analysis.telescope.telescope import Telescope
from beam_telescope_analysis.tools import analysis_utils
from beam_telescope_analysis.tools.storage_utils import save_arguments


@save_arguments
def combine_files(input_files, output_file=None, names=None, event_number_offsets=None, chunk_size=1000000):
    ''' Combine tables from different files and merge it into one single table.

    Some use cases:
    - Merging hit tables from different runs for combined analysis
      (under the assumption that telescope geometry has not changed between the runs)
    - Merging of tracks tables from different runs for combined efficiency analysis.
      (telescope geometry has changed between the runs and each run requires a separate alignment)

    Parameters
    ----------
    input_files : list
        Filenames of the input files containing a table.
    output_file : string
        Filename of the output file containing the merged table.
    names : list or string
        List of table names that will be merged. If None, all tables will be merged
    event_number_offsets : list
        Manually set start event number offset for each hit array.
        The event number is increased by the given number.
        If None, the event number will be generated automatically.
        If no "event_number" column is available, this parameter will be ignored.
    chunk_size : int
        Chunk size of the data when reading from the table.

    Returns
    -------
    applied_event_number_offsets : dict
        The dictinary contains the the lists of the event numbers offsets of each table.
    '''
    logging.info('=== Combining %d files ===' % len(input_files))

    if not output_file:
        prefix = os.path.commonprefix(input_files)
        output_file = os.path.splitext(prefix)[0] + '_combined.h5'

    # convert to list
    if names is not None and not isinstance(names, (list, tuple, set)):
        names = [names]

    out_tables = {}
    last_event_numbers = {}
    applied_event_number_offsets = {}
    with tb.open_file(filename=output_file, mode="w") as out_file_h5:
        for file_index, input_file in enumerate(input_files):
            with tb.open_file(filename=input_file, mode='r') as in_file_h5:
                # get all nodes of type 'table'
                in_tables = in_file_h5.list_nodes('/', classname='Table')
                for table in in_tables:
                    if names is not None and table.name not in names:
                        continue
                    if table.name not in out_tables:
                        out_tables[table.name] = out_file_h5.create_table(
                            where=out_file_h5.root,
                            name=table.name,
                            description=table.dtype,
                            title=table.title,
                            filters=tb.Filters(
                                complib='blosc',
                                complevel=5,
                                fletcher32=False))
                        if 'event_number' in table.dtype.names:
                            last_event_numbers[table.name] = -1
                            applied_event_number_offsets[table.name] = []
                        else:
                            last_event_numbers[table.name] = None
                            applied_event_number_offsets[table.name] = None

                    event_number_offset = 0
                    if last_event_numbers[table.name] is not None and event_number_offsets is not None and event_number_offsets[file_index] is not None:
                        event_number_offset = event_number_offsets[file_index]
                    elif last_event_numbers[table.name] is not None:
                        # increase by 1 to avoid duplicate event number
                        event_number_offset += last_event_numbers[table.name] + 1

                    for read_index in range(0, table.nrows, chunk_size):
                        data_chunk = table.read(start=read_index, stop=read_index + chunk_size)
                        if last_event_numbers[table.name] is not None and event_number_offset != 0:
                            data_chunk[:]['event_number'] += event_number_offset
                        out_tables[table.name].append(data_chunk)
                        out_tables[table.name].flush()
                    if last_event_numbers[table.name] is not None:
                        last_event_numbers[table.name] = data_chunk[-1]['event_number']
                        applied_event_number_offsets[table.name].append(event_number_offset)

    return output_file, applied_event_number_offsets


@save_arguments
def reduce_events(input_file, max_events, output_file=None, chunk_size=1000000):
    ''' Reducing the size of a file to a given number of events.

    Parameters
    ----------
    input_file : string
        Filename of the input file.
    output_file : string
        Filename of the output file.
    max_events : utint
        Maximum number of radomly selected events.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '_reduced.h5'

    with tb.open_file(input_file, mode='r') as in_file_h5:
        with tb.open_file(output_file, mode="w") as out_file_h5:
            for node in in_file_h5.root:
                if type(node) == tb.group.Group:
                    continue
                logging.info('Reducing events for node %s', node.name)

                tracks_table_out = out_file_h5.create_table(
                    where=out_file_h5.root,
                    name=node.name,
                    description=node.dtype,
                    title=node.title,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))

                total_n_tracks = node.shape[0]
                total_n_tracks_stored = 0
                total_n_events_stored = 0
                pbar = tqdm(total=max_events, ncols=80)

                total_n_events_stored_last = None
                # total_n_tracks_last = None
                last_index_chunk = None
                for data_chunk, index_chunk in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    n_tracks_chunk = data_chunk.shape[0]

                    unique_events = np.unique(data_chunk["event_number"])
                    n_events_chunk = unique_events.shape[0]

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
                    selected_tracks = np.in1d(data_chunk["event_number"], selected_events)
                    store_n_tracks = np.count_nonzero(selected_tracks)
                    total_n_tracks_stored += store_n_tracks
                    data_chunk = data_chunk[selected_tracks]

                    tracks_table_out.append(data_chunk)
                    tracks_table_out.flush()
                    total_n_events_stored_last = total_n_events_stored
                    # total_n_tracks_last = total_n_tracks
                    last_index_chunk = index_chunk
                    pbar.update(n_events_chunk)
                pbar.close()

    return output_file


@save_arguments
def select_tracks(telescope_configuration, input_tracks_file, select_duts, output_tracks_file=None, query=None, max_events=None, select_hit_duts=None, select_no_hit_duts=None, select_quality_duts=None, select_isolated_track_duts=None, select_isolated_hit_duts=None, chunk_size=1000000):
    ''' Selecting tracks that are matching the conditions and query strings.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_tracks_file : string
        Filename of the input tracks file.
    select_duts : list
        Selecting DUTs that will be processed.
    output_tracks_file : string
        Filename of the output tracks file.
    query : string or list
        List of query strings for each slected DUT.
        A query is a string that is processed and is used to select data from the table, e.g.,
        "track_chi2 <= 5", where "track_chi2" is a column in the table.
        The data in the output table contains only data with "track_chi2" smaller or equal to 5.
    max_events : uint
        Maximum number of radomly selected events.
    select_hit_duts : list
        List of DUTs for each slected DUT. The DUTs are required to have the hit flag set.
    select_no_hit_duts : list
        List of DUTs for each slected DUT. The DUTs are required to have hit flag not set.
    select_quality_duts : list
        List of DUTs for each slected DUT. The DUTs are required to have the quality flag set.
        The quality flag is only evaluated for DUTs where the hit flag is set.
    select_isolated_track_duts : list
        List of DUTs for each slected DUT. The DUTs are required to have the isolated track flag set.
        The isolated track flag is only evaluated for DUTs where the hit flag is set.
    select_isolated_hit_duts : list
        List of DUTs for each slected DUT. The DUTs are required to have the isolated hit flag set.
        The isolated hit flag is only evaluated for DUTs where the hit flag is set.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    logging.info('=== Selecting tracks of %d DUTs ===' % len(select_duts))

    if not output_tracks_file:
        output_tracks_file = os.path.splitext(input_tracks_file)[0] + '_selected.h5'

    # Check select_duts
    # Check for value errors
    if not isinstance(select_duts, Iterable):
        raise ValueError("Parameter select_duts is not an iterable.")
    elif not select_duts:  # empty iterable
        raise ValueError("Parameter select_duts has no items.")
    # Check if only non-iterable in iterable
    if not all(map(lambda val: isinstance(val, (int,)), select_duts)):
        raise ValueError("Not all items in parameter select_duts are integer.")

    # Create select_hit_duts
    if select_hit_duts is None:  # If None, use no selection
        select_hit_duts = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(select_hit_duts, Iterable):
        raise ValueError("Parameter select_hit_duts is not an iterable.")
    elif not select_hit_duts:  # empty iterable
        raise ValueError("Parameter select_hit_duts has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable) and val is not None, select_hit_duts)):
        select_hit_duts = [select_hit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, select_hit_duts)):
        raise ValueError("Not all items in parameter select_hit_duts are iterable or None.")
    # Finally check length of all arrays
    if len(select_hit_duts) != len(select_duts):  # empty iterable
        raise ValueError("Parameter select_hit_duts has the wrong length.")

    # Create select_no_hit_duts
    if select_no_hit_duts is None:  # If None, use no selection
        select_no_hit_duts = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(select_no_hit_duts, Iterable):
        raise ValueError("Parameter select_no_hit_duts is not an iterable.")
    elif not select_no_hit_duts:  # empty iterable
        raise ValueError("Parameter select_no_hit_duts has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable) and val is not None, select_no_hit_duts)):
        select_no_hit_duts = [select_no_hit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, select_no_hit_duts)):
        raise ValueError("Not all items in parameter select_no_hit_duts are iterable or None.")
    # Finally check length of all arrays
    if len(select_no_hit_duts) != len(select_duts):  # empty iterable
        raise ValueError("Parameter select_no_hit_duts has the wrong length.")
    for index, item in enumerate(select_no_hit_duts):
        if item is not None and select_hit_duts[index] is not None:
            if set(item) & set(select_hit_duts[index]):  # check for empty intersection
                raise ValueError("DUT%d cannot have select_hit_duts and select_no_hit_duts set for the same DUTs." % (select_duts[index],))

    # Create select_quality_duts
    if select_quality_duts is None:  # If None, use no selection
        select_quality_duts = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(select_quality_duts, Iterable):
        raise ValueError("Parameter select_quality_duts is not an iterable.")
    elif not select_quality_duts:  # empty iterable
        raise ValueError("Parameter select_quality_duts has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable) and val is not None, select_quality_duts)):
        select_quality_duts = [select_quality_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, select_quality_duts)):
        raise ValueError("Not all items in parameter select_quality_duts are iterable or None.")
    # Finally check length of all arrays
    if len(select_quality_duts) != len(select_duts):  # empty iterable
        raise ValueError("Parameter select_quality_duts has the wrong length.")

    # Create select_isolated_track_duts
    if select_isolated_track_duts is None:  # If None, use no selection
        select_isolated_track_duts = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(select_isolated_track_duts, Iterable):
        raise ValueError("Parameter select_isolated_track_duts is not an iterable.")
    elif not select_isolated_track_duts:  # empty iterable
        raise ValueError("Parameter select_isolated_track_duts has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable) and val is not None, select_isolated_track_duts)):
        select_isolated_track_duts = [select_isolated_track_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, select_isolated_track_duts)):
        raise ValueError("Not all items in parameter select_isolated_track_duts are iterable or None.")
    # Finally check length of all arrays
    if len(select_isolated_track_duts) != len(select_duts):  # empty iterable
        raise ValueError("Parameter select_isolated_track_duts has the wrong length.")

    # Create select_isolated_hit_duts
    if select_isolated_hit_duts is None:  # If None, use no selection
        select_isolated_hit_duts = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(select_isolated_hit_duts, Iterable):
        raise ValueError("Parameter select_isolated_hit_duts is not an iterable.")
    elif not select_isolated_hit_duts:  # empty iterable
        raise ValueError("Parameter select_isolated_hit_duts has no items.")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable) and val is not None, select_isolated_hit_duts)):
        select_isolated_hit_duts = [select_isolated_hit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable) or val is None, select_isolated_hit_duts)):
        raise ValueError("Not all items in parameter select_isolated_hit_duts are iterable or None.")
    # Finally check length of all arrays
    if len(select_isolated_hit_duts) != len(select_duts):  # empty iterable
        raise ValueError("Parameter select_isolated_hit_duts has the wrong length.")

    # Create query
    if query is None:  # If None, use empty strings for all DUTs
        query = ['' for _ in select_duts]
    # Check if iterable
    if isinstance(query, str):
        query = [query] * len(select_duts)
    # Check if only strings in iterable
    if not all(map(lambda val: isinstance(val, str), query)):
        raise ValueError("Not all items in parameter query are strings.")
    # Finally check length of all arrays
    if len(query) != len(select_duts):  # empty iterable
        raise ValueError("Parameter query has the wrong length.")

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_tracks_file, mode="w") as out_file_h5:
            for index, actual_dut_index in enumerate(select_duts):
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)
                logging.info('== Selecting tracks for %s ==', telescope[actual_dut_index].name)
                if query[index]:
                    logging.info('Query string: {}'.format(query[index]))
                hit_mask = 0
                if select_hit_duts[index]:
                    for dut in select_hit_duts[index]:
                        hit_mask |= (1 << dut)
                no_hit_mask = 0
                if select_no_hit_duts[index]:
                    for dut in select_no_hit_duts[index]:
                        no_hit_mask |= (1 << dut)
                quality_mask = 0
                if select_quality_duts[index]:
                    for dut in select_quality_duts[index]:
                        quality_mask |= (1 << dut)
                isolated_track_mask = 0
                if select_isolated_track_duts[index]:
                    for dut in select_isolated_track_duts[index]:
                        isolated_track_mask |= (1 << dut)
                isolated_hit_mask = 0
                if select_isolated_hit_duts[index]:
                    for dut in select_isolated_hit_duts[index]:
                        isolated_hit_mask |= (1 << dut)

                tracks_table_out = out_file_h5.create_table(
                    where=out_file_h5.root,
                    name=node.name,
                    description=node.dtype,
                    title=node.title,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))

                total_n_tracks = node.shape[0]
                total_n_tracks_stored = 0
                total_n_events_stored = 0
                if max_events:
                    pbar = tqdm(total=max_events, ncols=80)
                else:
                    pbar = tqdm(total=total_n_tracks, ncols=80)

                n_tracks_hit_mask = 0
                n_tracks_quality_mask = 0
                n_tracks_iso_track_mask = 0
                n_tracks_iso_hit_mask = 0
                n_tracks_query = 0

                total_n_events_stored_last = None
                # total_n_tracks_last = None
                last_index_chunk = None
                for tracks, index_chunk in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    n_tracks_chunk = tracks.shape[0]
                    if hit_mask != 0 or no_hit_mask != 0 or quality_mask != 0 or isolated_track_mask != 0 or isolated_hit_mask != 0:
                        select = np.ones(n_tracks_chunk, dtype=bool)
                        if hit_mask != 0:
                            select &= ((tracks['hit_flag'] & hit_mask) == hit_mask)
                        if no_hit_mask != 0:
                            select &= ((~tracks['hit_flag'] & no_hit_mask) == no_hit_mask)
                        n_tracks_hit_mask += np.count_nonzero(select)
                        if quality_mask != 0:
                            # Require only quality if have a valid hit
                            quality_mask_mod = quality_mask & tracks['hit_flag']
                            quality_flags_mod = quality_mask & tracks['hit_flag']
                            select &= ((tracks['quality_flag'] & quality_mask_mod) == quality_flags_mod)
                        n_tracks_quality_mask += np.count_nonzero(select)
                        if isolated_track_mask != 0:
                            select &= ((tracks['isolated_track_flag'] & isolated_track_mask) == isolated_track_mask)
                        n_tracks_iso_track_mask += np.count_nonzero(select)
                        if isolated_hit_mask != 0:
                            # Require only isolated hit if have a valid hit
                            isolated_hit_mask_mod = isolated_hit_mask & tracks['hit_flag']
                            isolated_hit_flags_mod = isolated_hit_mask & tracks['hit_flag']
                            select &= ((tracks['isolated_hit_flag'] & isolated_hit_mask_mod) == isolated_hit_flags_mod)
                        n_tracks_iso_hit_mask += np.count_nonzero(select)
                        tracks = tracks[select]
                    if query[index]:
                        tracks = table_where(
                            arr=tracks,
                            query_str=query[index])
                        n_tracks_query += tracks.shape[0]

                    if max_events:
                        unique_events = np.unique(tracks["event_number"])
                        n_events_chunk = unique_events.shape[0]
                        if total_n_tracks == index_chunk:  # last chunk, adding all remaining events
                            select_n_events = max_events - total_n_events_stored
                        elif total_n_events_stored == 0:  # first chunk
                            select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks)))
                        else:
                            # calculate correction of number of selected events
                            correction = (total_n_tracks - index_chunk) / total_n_tracks * 1 / (((total_n_tracks - last_index_chunk) / total_n_tracks) / ((max_events - total_n_events_stored_last) / max_events)) \
                                + (index_chunk) / total_n_tracks * 1 / (((last_index_chunk) / total_n_tracks) / ((total_n_events_stored_last) / max_events))
                            # select_n_events = np.ceil(n_events_chunk * correction)
                            # calculate correction of number of selected events
                            # correction = 1/(((total_n_tracks-last_index_chunk)/total_n_tracks_last)/((max_events-total_n_events_stored_last)/max_events))
                            select_n_events = int(round(max_events * (n_tracks_chunk / total_n_tracks) * correction))
                        # do not store more events than in current chunk
                        select_n_events = min(n_events_chunk, select_n_events)
                        # do not store more events than given by max_events
                        select_n_events = min(select_n_events, max_events - total_n_events_stored)
                        np.random.seed(seed=0)
                        selected_events = np.random.choice(unique_events, size=select_n_events, replace=False)
                        store_n_events = selected_events.shape[0]
                        total_n_events_stored += store_n_events
                        # print "store_n_events", store_n_events
                        selected_tracks = np.in1d(tracks["event_number"], selected_events)
                        store_n_tracks = np.count_nonzero(selected_tracks)
                        # TODO: total_n_tracks_stored not used...
                        total_n_tracks_stored += store_n_tracks
                        tracks = tracks[selected_tracks]
                        pbar.update(n_events_chunk)
                    else:
                        pbar.update(n_tracks_chunk)

                    tracks_table_out.append(tracks)
                    tracks_table_out.flush()
                    total_n_events_stored_last = total_n_events_stored
                    # total_n_tracks_last = total_n_tracks
                    last_index_chunk = index_chunk
                pbar.close()
                logging.info('Track selection summary: In total selected %.2f %% of tracks\nHit selection: %.2f\nQuality selection: %.2f %%\nIsolated tracks selection: %.2f %%\nIsolated hits selection: %.2f %%\nQuery string selection: %.2f %%' % (100.0 * n_tracks_query / total_n_tracks, 100.0 * n_tracks_hit_mask / total_n_tracks, 100.0 * n_tracks_quality_mask / n_tracks_hit_mask, 100.0 * n_tracks_iso_track_mask / n_tracks_quality_mask, 100.0 * n_tracks_iso_hit_mask / n_tracks_iso_track_mask, 100.0 * n_tracks_query / n_tracks_iso_hit_mask))

    return output_tracks_file


def table_where(arr, query_str):
    for variable in set(re.findall(r'(\d*[a-zA-Z_]+\d*)', query_str)):
        exec(variable + ' = arr[\'' + variable + '\']')  # expose variables; not a copy, this is just a reference

    return arr[ne.evaluate(query_str, casting="safe")]
