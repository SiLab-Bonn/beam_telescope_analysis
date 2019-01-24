
import logging
import os
import inspect
import itertools

import numpy as np
import tables as tb
from numba import njit, jit
from tqdm import tqdm
import matplotlib.pyplot as plt

from beam_telescope_analysis.converter.pybar_ship_converter import get_plane_files, merge_dc_module_local


def sort_modules_by_timestamp(pybar_runs, partition_dirs):
    '''
    Takes interpreted files of single modules and merges hit tables by partition.
    Hit tables are sorted by spill and timestamp within the spill.
    This is needed to correct the event_number since different busy times for the
    partitions result in different numbers of events for each partition in a run.
    ----------
    input:
        pybar_runs: tuple of 3 ints, pyBAR runs belonging to one ship run
        partition_dirs: iterable of strings with partition data
    return:
        list with sorted hit files, one file per partition_dir
    '''

    partitions = []
    spills = []
    for partition, dir in enumerate(partition_dirs):
        module_hits = []
        first_plane, second_plane = get_plane_files(pyBARrun=pybar_runs[partition], subpartition_dir=dir, string='s_hi_p_interpreted.h5')
        for i, hit_file in enumerate(first_plane + second_plane):
            if partition == 0 and i >= 4:  # account for missing module 4 in partition 0800
                i += 1
            with tb.open_file(hit_file, mode='r') as in_file:
                hit_table = in_file.root.Hits[:]
                hit_dtype = []

                for j, name in enumerate(hit_table.dtype.names):
                    if name == 'trigger_time_stamp':
                        hit_dtype.append((hit_table.dtype.names[j], np.int64))
                    else:
                        hit_dtype.append((hit_table.dtype.names[j], hit_table[name].dtype))
                hit_dtype.extend([('module', np.int16), ('spill', np.int16), ('partition', np.int16), ('avg_time_stamp', np.int64)])
                hits = np.empty(shape=hit_table.shape, dtype=hit_dtype)
                for name in hits.dtype.names:
                    if name in hit_table.dtype.names:
                        hits[name] = hit_table[name]
                    elif name == 'module':
                        hits['module'] = i
                    elif name == 'partition':
                        hits['partition'] = partition
                spill_list = np.where(np.diff(hits['trigger_time_stamp']) < 0)[0]
                for spill, row in enumerate(spill_list):
                    if spill == 0:
                        hits['spill'][:row + 1] = spill
                    elif spill == 1:
                        hits['spill'][spill_list[0] + 1:row + 1] = spill
                    else:
                        hits['spill'][spill_list[spill - 1] + 1:row + 1] = spill
                        hits['spill'][spill_list[spill] + 1:] = spill + 1
            module_hits.append(hits)

        partitions.append(np.sort(np.concatenate(module_hits), order=('spill', 'trigger_time_stamp', 'module')))
    run = np.sort(np.concatenate(partitions), order=('spill', 'trigger_time_stamp', 'partition', 'module'))  # combine all partition arrays to one run array
    print "hits in run", run.shape
    spills.extend(np.split(run, np.where(np.diff(run['spill']) > 0)[0]))  # sort hits by spill rather than partition

    return partitions, spills


def plot_timestamps(partition_hit_tables, spill_hit_tables):
    '''
    In case different busy times of partitions have lead to out of sync events, the same event number needs to
    be assigned to hits in all 3 partitions based on the timestamp of the respective spill.
    Timestamps for the same hit can differ by a small number of clock cycles (O ~ 1).
    '''

#     d={}
#     for i in range(len(partition_hit_tables)):
#         d['part{0}'.format(i)] = [] # one list for each partition
#     for i,key in enumerate(d):
#         d[key].extend(np.split(partition_hit_tables[i], np.where(np.diff(partition_hit_tables[i]['spill'])>0)[0])) # one store spill arrays in list per partition
#         timestamps = [np.unique(spill['trigger_time_stamp']) for spill in d[key]]
#
#         n_timestamps = 0
#         timestamps = np.zeros(shape=(longest,), dtype = [('ts_part0', np.int64),('ts_part1', np.int64),('ts_part2', np.int64),('ts_avg', np.int64)])

    # get all unique timestamps per spill from partition and store the arrays in a list
#         print [np.average(np.diff(spill)) for spill in timestamps]
    last_event = 0
    timestamps = []
    edges = []
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, 10000, 100000]
    for spill in spill_hit_tables:
        hist, edge = np.histogram(np.diff(spill['trigger_time_stamp']), bins=bins)
#         print np.unique(spill['trigger_time_stamp']).shape
        timestamps.append(hist)
        edges.append(edge)
    timestamps = np.vstack(timestamps)
    ts = np.sum(timestamps, axis=0)
    print ts
    print edges[-1][:-1]
#         np.histogram(np.diff(timestamps), bins= 50, range=(1,1000))[0] #np.bincount(np.diff(ts),minlength = 10)
    plt.title('Difference of timestamps for run 2815')
    plt.xlabel("# clock cycles")
    plt.ylabel('#')
#     plt.xlim(0,120)
#     plt.xscale('log')
    plt.xticks([i for i in range(0, len(bins))], bins)
    plt.bar(bins[:-1], ts, align='center')
    plt.grid()
    plt.yscale('log')
    plt.show()
#     plt.savefig('/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/delta_timestamps_run2815.pdf')
#         if any(np.diff(timestamps))<=64:
#             print "same event"


def open_files(files):
    hit_tables = []
    for f in files:
        with tb.open_file(f, 'r') as in_file:
            hit_tables.append(in_file.root.Hits[:])
    return hit_tables

def save_new_tables(files, arrays):
    '''
    saves tables with new event numbers, also removes empty events!!
    '''
    out_file_names = []
    for i, array in enumerate(arrays):
        out_file_name = files[i][:-3] + '_corr_evts.h5'
        out_file_names.append(out_file_name)
#         if not os.path.exists(out_file_name):
        with tb.open_file(out_file_name, 'w') as out_file_h5:
            hit_table_description = array.dtype
            hit_table_out = out_file_h5.create_table(
                where=out_file_h5.root,
                name='Hits',
                description=hit_table_description, title='Selected hits for test beam analysis, event number aligned at trigger_time_stamp',
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))
            # remove emtpy events for testbeam analysis
            hit_table_out.append(array[np.where((array['column']!=0) & (array['row']!=0))])
#         elif os.path.exists(out_file_name):
#             with tb.open_file(out_file_name, 'a') as out_file_h5:
#                 hit_table_out = out_file_h5.root.Hits
#                 hit_table_out.append(array)
        logging.info('saved hit table with corrected events under %s' % out_file_name)
    return out_file_names

@njit
def compare_timestamps(min_ts, timestamps, index, max_row):
    module_finished = False
    while timestamps[index] == min_ts or timestamps[index] == (min_ts + 1):
        if not index == max_row:
            index +=1
        elif index == max_row:
            module_finished = True
            return module_finished
    return index

def merge_hits(hit_tables_in, chunk_size=None, files = None):
    '''
    merge event numbers for events with jittering trigger_time_stamp.
    To account for empty spills, use "create empty event" while formatting hit tables in "pybar_ship_converter.py"
    '''

    delta_t = 1
    n_planes = len(hit_tables_in)

    # Init variables
    tss = []  # actual time stamps of planes
    new_tables = []

    for i in range(n_planes):
        value = hit_tables_in[i][0]['trigger_time_stamp']
        evt = hit_tables_in[i][0]['event_number']
        tss.append(value)
        new_tables.append(hit_tables_in[i].copy())
        print hit_tables_in[i][[row+1 for row in np.where((np.diff(hit_tables_in[i]['trigger_time_stamp'])==0) & (np.diff(hit_tables_in[i]['event_number'])!=0))]]

    max_rows = [table.shape[0]-1 for table in hit_tables_in]

    tss = np.array(tss)
    min_ts = min(tss)

    indices = [0] * n_planes  # actual hit indices of planes
    module_finished = [False] * n_planes
    module_finished = np.array(module_finished)
    event_number = 0
    n_spills = 0

    while np.any(np.array(indices) <= min(max_rows)) and not np.all(np.array(module_finished)==True):
        # print '# event ', event_number, '## ts', min_ts
        for plane in range(n_planes):
            # print '  - plane', plane, '-', indices[plane]
            start_index = indices[plane]
#             if not module_finished[plane] ==True:
#                 indices[plane] = compare_timestamps(min_ts, hit_tables_in[plane][indices[plane]:]['trigger_time_stamp'],indices[plane],max_rows[plane])
            while not module_finished[plane] == True and (hit_tables_in[plane][indices[plane]]['trigger_time_stamp'] == min_ts or hit_tables_in[plane][indices[plane]]['trigger_time_stamp'] == (min_ts + delta_t)) :
                if not indices[plane] == max_rows[plane]:
                    indices[plane] += 1
                    # fill new table
                    new_tables[plane][start_index:indices[plane]]['event_number'] = event_number
                elif indices[plane] == max_rows[plane]:
                    module_finished[plane] = True
                    # fix for last row in table
                    new_tables[plane][start_index:indices[plane]+1]['event_number'] = event_number
                    logging.info("reached end of hit_table %s: row %s, event_number %s " % (plane,indices[plane],event_number))
                # print '     col/row', hit_tables_in[plane][indices[plane]]['column'], hit_tables_in[plane][indices[plane]]['row']
            else:
                tss[plane] = hit_tables_in[plane][indices[plane]]['trigger_time_stamp']

        sel = tss > min_ts # select time stamps that do not belong to new spill
        if np.any(sel):  # at least one plane spill data not added
            min_ts = min(tss[sel])
        else:  # new spill, all ts are smaller
            n_spills += 1
            logging.info("========  new spill - last event_number %s ========" % (event_number))
            min_ts = min(tss)
        event_number += 1

    for i, plane in enumerate(hit_tables_in):
        print "old event number of table", i, plane[indices[i]]['event_number'], "new event number", new_tables[i][indices[i]]['event_number'], "timestamp", plane[indices[i]]['trigger_time_stamp']
    logging.info("fixed event numbers for %s spills" % n_spills)
    logging.info("last event_number %s , timestamp %s" % (event_number, min_ts))
    return new_tables


if __name__ == "__main__":
    # TODO: to correctly reconstruct hit tables with no trigger in spill use "create_empty_events" in "pybar_ship_converter.py"

    pybar_runs ='376' , '270', '204' # '412' , '306', '240' # '394', '288', '222'
    dirs = ['/media/niko/data/SHiP/charm_exp_2018/data/part_0x0800',
            '/media/niko/data/SHiP/charm_exp_2018/data/part_0x0801',
            '/media/niko/data/SHiP/charm_exp_2018/data/part_0x0802'
            ]

    files = []
    for i, directory in enumerate(dirs):
        first_plane, second_plane = get_plane_files(pyBARrun=pybar_runs[i], subpartition_dir=directory, string='_s_hi_p_interpreted_formatted_fei4.h5')
        files.extend(first_plane)
        files.extend(second_plane)

    hit_tables_in = open_files(files)
    new_tables = merge_hits(hit_tables_in=hit_tables_in)
    new_files = save_new_tables(files, new_tables)
