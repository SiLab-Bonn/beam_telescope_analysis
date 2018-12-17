''' Helper functions to select and combine data '''
import logging
import re
import os

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


def select_hits(hit_file, max_hits=None, condition=None, track_quality=None,
                track_quality_mask=None, output_file=None,
                chunk_size=1000000):
    ''' Function to select a fraction of hits fulfilling a given condition.

    Needed for analysis speed up, when very large runs are used.

    Parameters
    ----------
    hit_file : string
        Filename of the input hits file.
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
                for hits, i in analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    n_hits = hits.shape[0]
                    if condition:
                        hits = _select_hits_with_condition(hits, condition)

                    if track_quality:
                        # If no mask is defined select all quality bits
                        if not track_quality_mask:
                            track_quality_mask = int(0xFFFFFFFF)
                        sel = (hits['track_quality'] &
                               track_quality_mask) == (track_quality)
                        hits = hits[sel]

                    if hits.shape[0] == 0:
                        logging.warning('No hits selected')

                    # Reduce the number of added hits of this chunk to not
                    # exeed max_hits
                    if max_hits:
                        # Calculate number of hits to add for this chunk
                        # Fraction of hits to add per chunk
                        hit_fraction = max_hits / float(total_hits)
                        sel = np.ceil(np.linspace(0,
                                                  hits.shape[0],
                                                  int(hit_fraction * n_hits),
                                                  endpoint=False)).astype(np.int32)
                        sel = sel[sel < hits.shape[0]]
                        hits = hits[sel]

                    hits_out.append(hits)
                    progress_bar.update(i)
                progress_bar.finish()


def _select_hits_with_condition(hits_array, condition):
    for variable in set(re.findall(r'(\d*[a-zA-Z_]+\d*)', condition)):
        exec(variable + ' = hits_array[\'' + variable + '\']')  # expose variables; not a copy, this is just a reference

    return hits_array[ne.evaluate(condition, casting="safe")]
