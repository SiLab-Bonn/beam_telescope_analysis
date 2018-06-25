''' Helper functions to select and combine data '''
from __future__ import division

import logging
import re
import os
from collections import Iterable

import progressbar
import numpy as np
import tables as tb
import numexpr as ne

from testbeam_analysis.telescope.telescope import Telescope
from testbeam_analysis.tools import analysis_utils


def combine_hit_files(input_hit_files, output_hit_file=None, event_number_offsets=None, chunk_size=1000000):
    ''' Combine hit files of runs with same parameters to increase statistics.

    Parameters
    ----------
    input_hit_files : iterable
        List of filenames of the input hit files containing a hit array.
    output_hit_file : string
        Filename of the output hit file containing the combined hit array.
    event_number_offsets : iterable
        Manually set start event number offset for each hit array.
        The event number is increased by the given number.
        If None, the event number will be generated automatically.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    if not output_hit_file:
        prefix = os.path.commonprefix(input_hit_files)
        output_hit_file = os.path.splitext(prefix)[0] + '_combined.h5'

    last_event_number = 0
    used_event_number_offsets = []
    with tb.open_file(output_hit_file, mode="w") as out_file_h5:
        hits_out = None
        for index, hit_file in enumerate(input_hit_files):
            if event_number_offsets and event_number_offsets[index] is not None:
                event_number_offset = event_number_offsets[index]
            elif index == 0:
                event_number_offset = 0  # by default no offset for the first file
            else:
                event_number_offset += last_event_number + 1  # increase by 1 to avoid duplicate numbers

            with tb.open_file(hit_file, mode='r') as in_file_h5:
                for hits, _ in analysis_utils.data_aligned_at_events(
                        in_file_h5.root.Hits, chunk_size=chunk_size):
                    hits[:]['event_number'] += event_number_offset
                    if hits_out is None:
                        hits_out = out_file_h5.create_table(
                            where=out_file_h5.root,
                            name='Hits',
                            description=in_file_h5.root.Hits.dtype,
                            title=in_file_h5.root.Hits.title,
                            filters=tb.Filters(
                                complib='blosc',
                                complevel=5,
                                fletcher32=False))
                    hits_out.append(hits)
                    hits_out.flush()
                last_event_number = hits[-1]['event_number']
                used_event_number_offsets.append(event_number_offset)

    return used_event_number_offsets


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
                logging.info('Reducing events for node %s', node.name)
                total_n_tracks = node.shape[0]
                total_n_tracks_stored = 0
                total_n_events_stored = 0
                widgets = ['', progressbar.Percentage(), ' ',
                           progressbar.Bar(marker='*', left='|', right='|'),
                           ' ', progressbar.AdaptiveETA()]
                progress_bar = progressbar.ProgressBar(widgets=widgets,
                                                       maxval=total_n_tracks,
                                                       term_width=80)
                progress_bar.start()

                tracks_table_out = out_file_h5.create_table(
                    where=out_file_h5.root,
                    name=node.name,
                    description=node.dtype,
                    title=node.title,
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))

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
                        correction = (total_n_tracks - index_chunk)/total_n_tracks * 1 / (((total_n_tracks-last_index_chunk)/total_n_tracks)/((max_events-total_n_events_stored_last)/max_events)) \
                                     + (index_chunk)/total_n_tracks * 1 / (((last_index_chunk)/total_n_tracks)/((total_n_events_stored_last)/max_events))
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
                    total_n_tracks_last = total_n_tracks
                    last_index_chunk = index_chunk
                    progress_bar.update(index_chunk)
                progress_bar.finish()


def select_tracks(telescope_configuration, input_tracks_file, select_duts, output_tracks_file=None, condition=None, max_events=None, select_hit_duts=None, select_no_hit_duts=None, select_quality_duts=None, select_no_quality_duts=None, chunk_size=1000000):
    ''' Selecting tracks that are matching the conditions.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_tracks_file : string
        Filename of the input tracks file.
    '''
    telescope = Telescope(telescope_configuration)
    logging.info('=== Selecting tracks of %d DUTs ===' % len(select_duts))

    if not output_tracks_file:
        output_tracks_file = os.path.splitext(input_tracks_file)[0] + '_selected.h5'

    # Check select_duts
    # Check for value errors
    if not isinstance(select_duts, Iterable):
        raise ValueError("select_duts is no iterable")
    elif not select_duts:  # empty iterable
        raise ValueError("select_duts has no items")
    # Check if only non-iterable in iterable
    if not all(map(lambda val: isinstance(val, (int, long)), select_duts)):
        raise ValueError("not all items in select_duts are integer")

    # Create select_hit_duts
    if select_hit_duts is None:  # If None, use no selection
        select_hit_duts = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(select_hit_duts, Iterable):
        raise ValueError("select_hit_duts is no iterable")
    elif not select_hit_duts:  # empty iterable
        raise ValueError("select_hit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_hit_duts)):
        select_hit_duts = [select_hit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_hit_duts)):
        raise ValueError("not all items in select_hit_duts are iterable")
    # Finally check length of all arrays
    if len(select_hit_duts) != len(select_duts):  # empty iterable
        raise ValueError("select_hit_duts has the wrong length")

    # Create select_no_hit_duts
    if select_no_hit_duts is None:  # If None, use no selection
        select_no_hit_duts = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(select_no_hit_duts, Iterable):
        raise ValueError("select_no_hit_duts is no iterable")
    elif not select_no_hit_duts:  # empty iterable
        raise ValueError("select_no_hit_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_no_hit_duts)):
        select_no_hit_duts = [select_no_hit_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_no_hit_duts)):
        raise ValueError("not all items in select_no_hit_duts are iterable")
    # Finally check length of all arrays
    if len(select_no_hit_duts) != len(select_duts):  # empty iterable
        raise ValueError("select_no_hit_duts has the wrong length")

    # Create select_quality_duts
    if select_quality_duts is None:  # If None, use no selection
        select_quality_duts = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(select_quality_duts, Iterable):
        raise ValueError("select_quality_duts is no iterable")
    elif not select_quality_duts:  # empty iterable
        raise ValueError("select_quality_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_quality_duts)):
        select_quality_duts = [select_quality_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_quality_duts)):
        raise ValueError("not all items in select_quality_duts are iterable")
    # Finally check length of all arrays
    if len(select_quality_duts) != len(select_duts):  # empty iterable
        raise ValueError("select_quality_duts has the wrong length")

    # Create select_no_quality_duts
    if select_no_quality_duts is None:  # If None, use no selection
        select_no_quality_duts = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(select_no_quality_duts, Iterable):
        raise ValueError("select_no_quality_duts is no iterable")
    elif not select_no_quality_duts:  # empty iterable
        raise ValueError("select_no_quality_duts has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), select_no_quality_duts)):
        select_no_quality_duts = [select_no_quality_duts[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), select_no_quality_duts)):
        raise ValueError("not all items in select_no_quality_duts are iterable")
    # Finally check length of all arrays
    if len(select_no_quality_duts) != len(select_duts):  # empty iterable
        raise ValueError("select_no_quality_duts has the wrong length")

    # Create condition
    if condition is None:  # If None, use empty strings for all DUTs
        condition = ['' for _ in select_duts]
    # Check if iterable
    if isinstance(condition, str):
        condition = [condition] * len(select_duts)
    # Check if only strings in iterable
    if not all(map(lambda val: isinstance(val, str), condition)):
        raise ValueError("not all items in condition are strings")
    # Finally check length of all arrays
    if len(condition) != len(select_duts):  # empty iterable
        raise ValueError("condition has the wrong length")

    with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
        with tb.open_file(output_tracks_file, mode="w") as out_file_h5:
            for index, actual_dut_index in enumerate(select_duts):
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)
                logging.info('== Selecting tracks for %s ==', telescope[actual_dut_index].name)

                hit_flags = 0
                hit_mask = 0
                for dut in select_hit_duts[index]:
                    hit_flags |= (1 << dut)
                    hit_mask |= (1 << dut)
                for dut in select_no_hit_duts[index]:
                    hit_mask |= (1 << dut)
                print hit_flags, bin(hit_flags)
                print hit_mask, bin(hit_mask)
                quality_flags = 0
                quality_mask = 0
                for dut in select_quality_duts[index]:
                    quality_flags |= (1 << dut)
                    quality_mask |= (1 << dut)
                for dut in select_no_quality_duts[index]:
                    quality_mask |= (1 << dut)
                print quality_flags, bin(quality_flags)
                print quality_mask, bin(quality_mask)

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
                widgets = ['', progressbar.Percentage(), ' ',
                           progressbar.Bar(marker='*', left='|', right='|'),
                           ' ', progressbar.AdaptiveETA()]
                progress_bar = progressbar.ProgressBar(widgets=widgets,
                                                       maxval=total_n_tracks,
                                                       term_width=80)
                progress_bar.start()
                for tracks, index_chunk in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    n_tracks_chunk = tracks.shape[0]

                    if hit_mask != 0 or quality_mask != 0:
                        select = np.ones(n_tracks_chunk, dtype=np.bool)
                        if hit_mask != 0:
                            select &= ((tracks['hit_flag'] & hit_mask) == hit_flags)
                        if quality_mask != 0:
                            select &= ((tracks['quality_flag'] & quality_mask) == quality_flags)
                        tracks = tracks[select]
                    if condition[index]:
                        tracks = _select_rows_with_condition(tracks, condition[index])

                    unique_events = np.unique(tracks["event_number"])
                    n_events_chunk = unique_events.shape[0]

                    # print "n_events_chunk", n_events_chunk
                    # print "n_tracks_chunk", n_tracks_chunk
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
                        selected_tracks = np.in1d(tracks["event_number"], selected_events)
                        store_n_tracks = np.count_nonzero(selected_tracks)
                        # TODO: total_n_tracks_stored not used...
                        total_n_tracks_stored += store_n_tracks
                        tracks = tracks[selected_tracks]

                    tracks_table_out.append(tracks)
                    tracks_table_out.flush()
                    total_n_events_stored_last = total_n_events_stored
                    total_n_tracks_last = total_n_tracks
                    last_index_chunk = index_chunk
                    progress_bar.update(index_chunk)
                progress_bar.finish()
                # print "***************"
                # print "total_n_tracks_stored", total_n_tracks_stored
                # print "total_n_events_stored", total_n_events_stored


def _select_rows_with_condition(rec_array, condition):
    for variable in set(re.findall(r'(\d*[a-zA-Z_]+\d*)', condition)):
        exec(variable + ' = rec_array[\'' + variable + '\']')  # expose variables; not a copy, this is just a reference

    return rec_array[ne.evaluate(condition, casting="safe")]
