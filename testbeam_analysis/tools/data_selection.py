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
from numba import njit

from testbeam_analysis.tools import analysis_utils

# Hit data dtype
hit_dcr = np.dtype([('event_number', np.int64), ('frame', np.uint8),
                    ('column', np.uint16), ('row', np.uint16),
                    ('charge', np.uint16)])


def combine_hit_files(hit_files, combined_file, event_number_offsets=None,
                      chunk_size=1000000):
    ''' Combine hit files of runs with same parameters to increase statistics.

    Parameters
    ----------
    hit_files : iterable
        Filenames of files containing the hit array.
    combined_file : string
        Filename of the output file containing the combined hit array.
    event_number_offsets : iterable
        Manually set start event number offset for each hit array.
        The event number is increased by the given number.
        If None, the event number will be generated automatically.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    last_event_number = 0
    used_event_number_offsets = []
    with tb.open_file(combined_file, mode="w") as out_file:
        hits_out = out_file.create_table(out_file.root, name='Hits',
                                         description=hit_dcr,
                                         title='Selected FE-I4 hits',
                                         filters=tb.Filters(complib='blosc',
                                                            complevel=5,
                                                            fletcher32=False))
        for index, hit_file in enumerate(hit_files):
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
                    hits_out.append(hits)
                last_event_number = hits[-1]['event_number']
                used_event_number_offsets.append(event_number_offset)

    return used_event_number_offsets


@njit()
def _delete_events(data, fraction):
    result = np.zeros_like(data)
    index_result = 0

    for index in range(data.shape[0]):
        if data[index]['event_number'] % fraction == 0:
            result[index_result] = data[index]
            index_result += 1
    return result[:index_result]


def reduce_hit_files(hit_files, fraction=10, chunk_size=1000000):
    ''' Delete a fraction of events to allow faster testing of analysis functions.

    Parameters
    ----------
    hit_files : iterable
        Filenames of files containing the hit array.
    fraction : uint
        The fraction of leftover events,
        e.g.: 10 would correspond to n_events = total_events / fraction.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''

    for hit_file in hit_files:
        with tb.open_file(hit_file, mode='r') as in_file:
            with tb.open_file(os.path.splitext(hit_file)[0] +
                              '_reduced.h5', mode="w") as out_file:
                filters = tb.Filters(complib='blosc', complevel=5,
                                     fletcher32=False)
                hits_out = out_file.create_table(out_file.root,
                                                 name='Hits',
                                                 description=hit_dcr,
                                                 title='Selected FE-I4 hits',
                                                 filters=filters)
                for hits, _ in analysis_utils.data_aligned_at_events(
                        in_file.root.Hits, chunk_size=chunk_size):
                    hits_out.append(_delete_events(hits, fraction))


def select_hits(hit_file, max_hits=None, condition=None,
                hit_flag=None, hit_mask=None, quality_flag=None, quality_mask=None,
                track_quality=None, track_quality_mask=None,
                output_file=None, chunk_size=1000000):
    ''' Function to select a fraction of hits fulfilling a given condition.

    Needed for analysis speed up, when very large runs are used.

    Parameters
    ----------
    hit_file : string
        Filename of the input tracks file.
    max_hits : uint
        Number of maximum hits with selection. For data reduction.
    condition : string
        A condition that is applied to the hits in numexpr. Only if the
        expression evaluates to True the hit is taken.
        E.g.: condition = 'track_quality == 2 & event_number < 1000'
    chunk_size : int
        Chunk size of the data when reading from file.
    '''

    with tb.open_file(hit_file, mode='r') as in_file:
        if not output_file:
            output_file = os.path.splitext(hit_file)[0] + '_reduced.h5'
        with tb.open_file(output_file, mode="w") as out_file:
            for node in in_file.root:
                total_hits = node.shape[0]
                widgets = ['', progressbar.Percentage(), ' ',
                           progressbar.Bar(marker='*', left='|', right='|'),
                           ' ', progressbar.AdaptiveETA()]
                progress_bar = progressbar.ProgressBar(widgets=widgets,
                                                       maxval=total_hits,
                                                       term_width=80)
                progress_bar.start()
                hits_out = out_file.create_table(out_file.root, name=node.name,
                                                 description=node.dtype,
                                                 title=node.title,
                                                 filters=tb.Filters(
                                                     complib='blosc',
                                                     complevel=5,
                                                     fletcher32=False))
                for hits, index in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    n_hits = hits.shape[0]
                    sel = np.ones((n_hits,), dtype=np.bool)
                    if hit_flag is not None:
                        if hit_mask is None:
                            hit_mask = hit_flag
                        sel &= ((hits['hit_flag'] & hit_mask) == hit_flag)
                    if quality_flag is not None:
                        if quality_mask is None:
                            quality_mask = quality_flag
                        sel &= ((hits['quality_flag'] & quality_mask) == quality_flag)
                    hits = hits[sel]
                    if condition:
                        hits = _select_rows_with_condition(hits, condition)

#                     if track_quality:
#                         # If no mask is defined select all quality bits
#                         if not track_quality_mask:
#                             track_quality_mask = int(0xFFFFFFFF)
#                         sel = (hits['track_quality'] &
#                                track_quality_mask) == (track_quality)
#                         hits = hits[sel]

                    if hits.shape[0] == 0:
                        logging.warning('No hits selected')

                    # Reduce the number of added hits of this chunk to not
                    # exeed max_hits
                    if max_hits:
                        # Calculate number of hits to add for this chunk
                        # Fraction of hits to add per chunk
                        hit_fraction = max_hits / total_hits
                        sel = np.ceil(np.linspace(0,
                                                  hits.shape[0],
                                                  int(hit_fraction * n_hits),
                                                  endpoint=False)).astype(np.int32)
                        sel = sel[sel < hits.shape[0]]
                        hits = hits[sel]

                    hits_out.append(hits)
                    progress_bar.update(index)
                progress_bar.finish()


def select_tracks(input_tracks_file, select_duts, output_tracks_file=None, condition=None, max_events=None, duts_hit_selection=None, duts_no_hit_selection=None, duts_quality_selection=None, duts_no_quality_selection=None, chunk_size=1000000):
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

    # Create duts_hit_selection
    if duts_hit_selection is None:  # If None, use no selection
        duts_hit_selection = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(duts_hit_selection, Iterable):
        raise ValueError("duts_hit_selection is no iterable")
    elif not duts_hit_selection:  # empty iterable
        raise ValueError("duts_hit_selection has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), duts_hit_selection)):
        duts_hit_selection = [duts_hit_selection[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), duts_hit_selection)):
        raise ValueError("not all items in duts_hit_selection are iterable")
    # Finally check length of all arrays
    if len(duts_hit_selection) != len(select_duts):  # empty iterable
        raise ValueError("duts_hit_selection has the wrong length")

    # Create duts_no_hit_selection
    if duts_no_hit_selection is None:  # If None, use no selection
        duts_no_hit_selection = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(duts_no_hit_selection, Iterable):
        raise ValueError("duts_no_hit_selection is no iterable")
    elif not duts_no_hit_selection:  # empty iterable
        raise ValueError("duts_no_hit_selection has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), duts_no_hit_selection)):
        duts_no_hit_selection = [duts_no_hit_selection[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), duts_no_hit_selection)):
        raise ValueError("not all items in duts_no_hit_selection are iterable")
    # Finally check length of all arrays
    if len(duts_no_hit_selection) != len(select_duts):  # empty iterable
        raise ValueError("duts_no_hit_selection has the wrong length")

    # Create duts_quality_selection
    if duts_quality_selection is None:  # If None, use no selection
        duts_quality_selection = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(duts_quality_selection, Iterable):
        raise ValueError("duts_quality_selection is no iterable")
    elif not duts_quality_selection:  # empty iterable
        raise ValueError("duts_quality_selection has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), duts_quality_selection)):
        duts_quality_selection = [duts_quality_selection[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), duts_quality_selection)):
        raise ValueError("not all items in duts_quality_selection are iterable")
    # Finally check length of all arrays
    if len(duts_quality_selection) != len(select_duts):  # empty iterable
        raise ValueError("duts_quality_selection has the wrong length")

    # Create duts_no_quality_selection
    if duts_no_quality_selection is None:  # If None, use no selection
        duts_no_quality_selection = [[] for _ in select_duts]
    # Check iterable and length
    if not isinstance(duts_no_quality_selection, Iterable):
        raise ValueError("duts_no_quality_selection is no iterable")
    elif not duts_no_quality_selection:  # empty iterable
        raise ValueError("duts_no_quality_selection has no items")
    # Check if only non-iterable in iterable
    if all(map(lambda val: not isinstance(val, Iterable), duts_no_quality_selection)):
        duts_no_quality_selection = [duts_no_quality_selection[:] for _ in select_duts]
    # Check if only iterable in iterable
    if not all(map(lambda val: isinstance(val, Iterable), duts_no_quality_selection)):
        raise ValueError("not all items in duts_no_quality_selection are iterable")
    # Finally check length of all arrays
    if len(duts_no_quality_selection) != len(select_duts):  # empty iterable
        raise ValueError("duts_no_quality_selection has the wrong length")

    # Create condition
    if condition is None:  # If None, use empty strings for all DUTs
        condition = ['' for _ in select_duts]
    # Check if iterable
    if isinstance(condition, str):
        condition = [condition] * select_duts
    # Check if only strings in iterable
    if not all(map(lambda val: isinstance(val, str), condition)):
        raise ValueError("not all items in condition are strings")
    # Finally check length of all arrays
    if len(condition) != len(select_duts):  # empty iterable
        raise ValueError("condition has the wrong length")

    with tb.open_file(input_tracks_file, mode='r') as in_file:
        with tb.open_file(output_tracks_file, mode="w") as out_file:
            for node in in_file.root:
                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                if (select_duts and actual_dut not in select_duts):
                    continue
                print "track selection for DUT%d" % actual_dut
                dut_index = np.where(np.array(select_duts) == actual_dut)[0][0]

                hit_flags = 0
                hit_mask = 0
                for dut in duts_hit_selection[dut_index]:
                    hit_flags |= (1 << dut)
                    hit_mask |= (1 << dut)
                for dut in duts_no_hit_selection[dut_index]:
                    hit_mask |= (1 << dut)
                print hit_flags, bin(hit_flags)
                print hit_mask, bin(hit_mask)
                quality_flags = 0
                quality_mask = 0
                for dut in duts_quality_selection[dut_index]:
                    quality_flags |= (1 << dut)
                    quality_mask |= (1 << dut)
                for dut in duts_no_quality_selection[dut_index]:
                    quality_mask |= (1 << dut)
                print quality_flags, bin(quality_flags)
                print quality_mask, bin(quality_mask)

                tracks_table_out = out_file.create_table(out_file.root, name=node.name,
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
                read = 0
                for tracks, index_chunk in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    n_tracks_chunk = tracks.shape[0]

                    if hit_mask != 0 or quality_mask != 0:
                        select = np.ones(n_tracks_chunk, dtype=np.bool)
                        if hit_mask != 0:
                            select &= ((tracks['hit_flag'] & hit_mask) == hit_flags)
                        if quality_mask != 0:
                            select &= ((tracks['quality_flag'] & quality_mask) == quality_flags)
                        tracks = tracks[select]
                    if condition[dut_index]:
                        tracks = _select_rows_with_condition(tracks, condition[dut_index])

                    read += 1
#                     if read == 5:
#                         tracks["event_number"] = 0
#                     n_tracks_chunk = tracks.shape[0]

                    unique_events = np.unique(tracks["event_number"])
                    n_events_chunk = unique_events.shape[0]

                    print "**read**", read, index_chunk, total_n_tracks
                    print "n_events_chunk", n_events_chunk
                    print "n_tracks_chunk", n_tracks_chunk
                    if max_events:
                        if total_n_tracks == index_chunk:  # last chunk, adding all remaining events
                            select_n_events = max_events - total_n_events_stored
                        elif total_n_events_stored == 0:  # first chunk
                            select_n_events = np.ceil(max_events * (n_tracks_chunk / total_n_tracks))
                        else:
                            # calculate correction of number of selected events
                            correction = 1/(((total_n_tracks-last_index_chunk)/total_n_tracks_last)/((max_events-total_n_events_stored_last)/max_events))
                            select_n_events = np.ceil(max_events * (n_tracks_chunk / total_n_tracks) * correction)
                        # do not store more events than in current chunk
                        select_n_events = min(n_events_chunk, select_n_events)
                        # do not store more events than given by max_events
                        select_n_events = min(int(select_n_events), max_events - total_n_events_stored)
                        np.random.seed(seed=0)
                        selected_events = np.random.choice(unique_events, size=select_n_events, replace=False)
                        store_n_events = selected_events.shape[0]
                        total_n_events_stored += store_n_events
                        print "store_n_events", store_n_events
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
                print "***************"
                print "total_n_tracks_stored", total_n_tracks_stored
                print "total_n_events_stored", total_n_events_stored


def _select_rows_with_condition(rec_array, condition):
    for variable in set(re.findall(r'(\d*[a-zA-Z_]+\d*)', condition)):
        exec(variable + ' = rec_array[\'' + variable + '\']')  # expose variables; not a copy, this is just a reference

    return rec_array[ne.evaluate(condition, casting="safe")]
