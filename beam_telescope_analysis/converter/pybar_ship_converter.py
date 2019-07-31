"""This script prepares FE-I4 test beam raw data recorded by pyBAR to be analyzed by the simple python test beam analysis.
An installation of pyBAR is required: https://silab-redmine.physik.uni-bonn.de/projects/pybar
- This script does for each DUT in parallel
  - Create a hit tables from the raw data
  - Align the hit table event number to the trigger number to be able to correlate hits in time
  - Rename and select hit info needed for further analysis.
"""

import logging
import os
from multiprocessing import Pool

import numpy as np
import tables as tb

from tqdm import tqdm

from pybar.analysis import analysis_utils
from pybar.analysis.analyze_raw_data import AnalyzeRawData
from pybar_fei4_interpreter import data_struct


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

testbeam_analysis_dtype = np.dtype([
    ('event_number', np.int64),
    ('trigger_time_stamp', np.int64),
    ('frame', np.uint8),
    ('column', np.uint16),
    ('row', np.uint16),
    ('charge', np.uint16)])


def process_dut(raw_data_file, output_filename=None, trigger_data_format=0, do_corrections=False, empty_events=False):
    ''' Process and format raw data.

    Parameters
    ----------
    raw_data_file : string or list of strings
        Filename(s) of the raw data file(s).
    output_filename : string
        Filename of the output interpreted and formatted data file.
    trigger_data_format : int
        Trigger/TLU FSM data mode.

    Returns
    -------
    output_filename : string
        Filename of the output interpreted and formatted data file.
    '''
    if do_corrections is True:
        fix_trigger_number, fix_event_number = False, True
    else:
        fix_trigger_number, fix_event_number = False, False
    analyze_raw_data(input_filename=raw_data_file, trigger_data_format=trigger_data_format, empty_events = empty_events)
    if isinstance(raw_data_file, (list, tuple)):
        raw_data_filename = os.path.splitext(sorted(raw_data_file)[0])[0]  # get filename with the lowest index
    else:  # string
        raw_data_filename = os.path.splitext(raw_data_file)[0]
    if do_corrections:
        align_events(
            input_filename=raw_data_filename + '_interpreted.h5',
            output_filename=raw_data_filename + '_event_aligned.h5',
            fix_trigger_number=fix_trigger_number,
            fix_event_number=fix_event_number)
        output_filename = format_hit_table(input_filename=raw_data_filename + '_event_aligned.h5', output_filename=output_filename)
    else:
        output_filename = format_hit_table(input_filename=raw_data_filename + '_interpreted.h5', output_filename=output_filename)
    return output_filename


def analyze_raw_data(input_filename, output_filename=None, trigger_data_format=0, empty_events = False):  # FE-I4 raw data analysis
    '''Std. raw data analysis of FE-I4 data. A hit table is created for further analysis.

    Parameters
    ----------
    input_filename : string
        Filename of the input raw data file.
    output_filename : string
        Filename of the output interpreted data file.
    '''
    with AnalyzeRawData(raw_data_file=input_filename, analyzed_data_file=output_filename, create_pdf=True) as analyze_raw_data:
        # analyze_raw_data.align_at_trigger_number = True  # if trigger number is at the beginning of each event activate this for event alignment
        analyze_raw_data.trigger_data_format = trigger_data_format
        analyze_raw_data.use_tdc_word = False
        analyze_raw_data.create_hit_table = True
        analyze_raw_data.create_meta_event_index = True
        analyze_raw_data.create_trigger_error_hist = True
        analyze_raw_data.create_rel_bcid_hist = True
        analyze_raw_data.create_error_hist = True
        analyze_raw_data.create_service_record_hist = True
        analyze_raw_data.create_occupancy_hist = True
        analyze_raw_data.create_tot_hist = False
        analyze_raw_data.align_at_trigger = True
        analyze_raw_data.fei4b = False
        analyze_raw_data.create_empty_event_hits = empty_events
        # analyze_raw_data.n_bcid = 16
        # analyze_raw_data.max_tot_value = 13
        analyze_raw_data.interpreter.set_warning_output(False)
        analyze_raw_data.interpret_word_table()
        analyze_raw_data.interpreter.print_summary()
        analyze_raw_data.plot_histograms()


def align_events(input_filename, output_filename, fix_event_number=True, fix_trigger_number=True, chunk_size=1000000):
    ''' Selects only hits from good events and checks the distance between event number and trigger number for each hit.
    If the FE data allowed a successful event recognition the distance is always constant (besides the fact that the trigger number overflows).
    Otherwise the event number is corrected by the trigger number. How often an inconsistency occurs is counted as well as the number of events that had to be corrected.
    Remark: Only one event analyzed wrong shifts all event numbers leading to no correlation! But usually data does not have to be corrected.

    Parameters
    ----------
    input_filename : string
        Filename of the input interpreted data file.
    output_filename : string
        Filename of the output interpreted data file.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('Align events to trigger number in %s' % input_filename)

    with tb.open_file(filename=input_filename, mode='r') as in_file_h5:
        hit_table = in_file_h5.root.Hits
        jumps = []  # variable to determine the jumps in the event-number to trigger-number offset
        n_fixed_hits = 0  # events that were fixed

        with tb.open_file(filename=output_filename, mode='w') as out_file_h5:
            hit_table_description = data_struct.HitInfoTable().columns.copy()
            hit_table_out = out_file_h5.create_table(
                where=out_file_h5.root,
                name='Hits',
                description=hit_table_description, title='Selected hits for test beam analysis',
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))

            # Correct hit event number
            for hits, _ in analysis_utils.data_aligned_at_events(hit_table, chunk_size=chunk_size):

                if not np.all(np.diff(hits['event_number']) >= 0):
                    raise RuntimeError('The event number does not always increase. This data cannot be used like this!')

                if fix_trigger_number is True:
                    selection = np.logical_or((hits['trigger_status'] & 0b00000001) == 0b00000001,
                                              (hits['event_status'] & 0b0000000000000010) == 0b0000000000000010)
                    selected_te_hits = np.where(selection)[0]  # select both events with and without hit that have trigger error flag set

#                     assert selected_te_hits[0] > 0
                    tmp_trigger_number = hits['trigger_number'].astype(np.int32)

                    # save trigger and event number for plotting correlation between trigger number and event number
                    event_number, trigger_number = hits['event_number'].copy(), hits['trigger_number'].copy()

                    hits['trigger_number'][0] = 0

                    offset = (hits['trigger_number'][selected_te_hits] - hits['trigger_number'][selected_te_hits - 1] - hits['event_number'][selected_te_hits] + hits['event_number'][selected_te_hits - 1]).astype(np.int32)  # save jumps in trigger number
                    offset_tot = np.cumsum(offset)

                    offset_tot[offset_tot > 32768] = np.mod(offset_tot[offset_tot > 32768], 32768)
                    offset_tot[offset_tot < -32768] = np.mod(offset_tot[offset_tot < -32768], 32768)

                    for start_hit_index in range(len(selected_te_hits)):
                        start_hit = selected_te_hits[start_hit_index]
                        stop_hit = selected_te_hits[start_hit_index + 1] if start_hit_index < (len(selected_te_hits) - 1) else None
                        tmp_trigger_number[start_hit:stop_hit] -= offset_tot[start_hit_index]

                    tmp_trigger_number[tmp_trigger_number >= 32768] = np.mod(tmp_trigger_number[tmp_trigger_number >= 32768], 32768)
                    tmp_trigger_number[tmp_trigger_number < 0] = 32768 - np.mod(np.abs(tmp_trigger_number[tmp_trigger_number < 0]), 32768)

                    hits['trigger_number'] = tmp_trigger_number

                selected_hits = hits[(hits['event_status'] & 0b0000100000000000) == 0b0000000000000000]  # select not empty events

                if fix_event_number is True:
                    selector = (selected_hits['event_number'] != (np.divide(selected_hits['event_number'] + 1, 32768) * 32768 + selected_hits['trigger_number'] - 1))
                    n_fixed_hits += np.count_nonzero(selector)
                    selector = selected_hits['event_number'] > selected_hits['trigger_number']
                    selected_hits['event_number'] = np.divide(selected_hits['event_number'] + 1, 32768) * 32768 + selected_hits['trigger_number'] - 1
                    selected_hits['event_number'][selector] = np.divide(selected_hits['event_number'][selector] + 1, 32768) * 32768 + 32768 + selected_hits['trigger_number'][selector] - 1

#                 FIX FOR DIAMOND:
#                 selected_hits['event_number'] -= 1  # FIX FOR DIAMOND EVENT OFFSET

                hit_table_out.append(selected_hits)

        jumps = np.unique(np.array(jumps))
        logging.info('Corrected %d inconsistencies in the event number. %d hits corrected.' % (jumps[jumps != 0].shape[0], n_fixed_hits))

        if fix_trigger_number is True:
            return (output_filename, event_number, trigger_number, hits['trigger_number'])


def format_hit_table(input_filename, output_filename=None, chunk_size=1000000):
    ''' Selects and renames important columns for test beam analysis and stores them into a new file.

    Parameters
    ----------
    input_filename : string
        Filename of the input interpreted data file.
    output_filename : string
        Filename of the output interpreted and formatted data file.
        If None, the filename will be generated.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_filename : string
        Filename of the output interpreted and formatted data file.
    '''
    if output_filename is None:
        output_filename = os.path.splitext(input_filename)[0] + '_formatted_fei4.h5'

    logging.info('Format hit table in %s', input_filename)
    with tb.open_file(filename=input_filename, mode='r') as in_file_h5:
        last_event_number = np.zeros(shape=1, dtype=np.int64)
        input_hits_table = in_file_h5.root.Hits
        with tb.open_file(filename=output_filename, mode='w') as out_file_h5:
            output_hits_table = out_file_h5.create_table(
                where=out_file_h5.root,
                name='Hits',
                description=testbeam_analysis_dtype,
                title='Hits for test beam analysis',
                filters=tb.Filters(
                    complib='blosc',
                    complevel=5,
                    fletcher32=False))
            for read_index in tqdm(range(0, input_hits_table.nrows, chunk_size)):
                hits_chunk = input_hits_table.read(read_index, read_index + chunk_size)
                if np.any(np.diff(np.concatenate((last_event_number, hits_chunk['event_number']))) < 0):
                    raise RuntimeError('The event number does not increase.')
                last_event_number = hits_chunk['event_number'][-1:]
                hits_data_formatted = np.zeros(shape=hits_chunk.shape[0], dtype=testbeam_analysis_dtype)
                hits_data_formatted['event_number'] = hits_chunk['event_number']
                hits_data_formatted['frame'] = hits_chunk['relative_BCID']
                hits_data_formatted['column'] = hits_chunk['column']
                hits_data_formatted['row'] = hits_chunk['row']
                hits_data_formatted['charge'] = hits_chunk['tot']
                hits_data_formatted['trigger_time_stamp'] = hits_chunk['trigger_time_stamp']
                output_hits_table.append(hits_data_formatted)
                output_hits_table.flush()

    return output_filename

def get_plane_files(pyBARrun, subpartition_dir, string='s_hi_p.h5'):

    first_plane_modules = ('module_0','module_1','module_2','module_3')
    second_plane_modules = ('module_4','module_5','module_6','module_7')

    partID = subpartition_dir[subpartition_dir.find('0x08') + 5]

    first_plane, second_plane = [], []
    for dirpath,_,filenames in os.walk(subpartition_dir):
        for f in filenames:
            if pyBARrun in f and string in f: #f[-9:] == 's_hi_p.h5':
                if 'module_0' in f or 'module_1'in f or 'module_2'in f or 'module_3'in f :
                    first_plane.append(os.path.abspath(os.path.join(dirpath, f)))
                elif 'module_4' in f or 'module_5'in f or 'module_6'in f or 'module_7'in f :
                    second_plane.append(os.path.abspath(os.path.join(dirpath, f)))

    return sorted(first_plane), sorted(second_plane)


def merge_plane(output_dir, plane_files, pyBARrun, plane_number):
    ''' merges hits of 4 FEs in one plane to one hit_table
    input:
        plane_files : list of paths to files with converted hit_tables
        pyBARrun : string, run number for creating merged file
        plane_number: int, plane in telescope setup, increasing with increasing z.
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.abspath(os.path.join(output_dir, 'pyBARrun_%s' % pyBARrun + '_plane_%s' % plane_number +'_merged.h5'))
    single_hit_arrays = []
    nhits = 0
    with tb.open_file(output_file, 'w') as out_file:
        hits_out = out_file.create_table(out_file.root, name='Hits',
                                             description=np.dtype([('event_number', np.int64), ('trigger_time_stamp',np.int64),('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)]),
                                             title='Merged hits for plane %s'% plane_number,
                                             filters=tb.Filters(
                                                 complib='blosc',
                                                 complevel=5,
                                                 fletcher32=False))
        for hit_file in plane_files:
            moduleID = int(hit_file[hit_file.rfind('module_') + 7])
            with tb.open_file(hit_file[:-3] + '_aligned.h5', 'r') as in_file:
                hits = in_file.root.Hits[:]
                logging.info("merging front end %s" % frontEndID)
                if moduleID == 0 :
                    hits['row'] = -(hits['row'] - 81)
#                     hits['row'] = hits['row'] + 80
#                     hits['column'] = -hits['column'] + 337
                elif moduleID == 1:
                    hits['row'] = -(hits['row'] - 81) + 80
                elif moduleID == 2:
                    hits['column'] = (hits['column'] -337)*-1 + 336
                    hits['row'] = -(hits['row'] - 81)
                elif moduleID == 3:
                    hits['column'] = (hits['column'] -337)*-1 + 336
                    hits['row'] = -(hits['row'] - 81) + 80
                elif moduleID == 4:
                    hits['column'] = -(hits['column'] - 81)
                    hits['row'] = -(hits['row'] - 337) + 336
                elif moduleID == 5:
                    hits['column'] = -(hits['column'] - 81) + 80
                    hits['row'] = -(hits['row'] - 337) + 336
                elif moduleID == 6:
                    hits['column'] = -(hits['column'] - 81)
#                     hits['row'] = (hits['row'] -337)*-1
                elif moduleID == 7:
#                     hits['row'] = (hits['row'] -337)*-1
                    hits['column'] = -(hits['column'] - 81) + 80
                else:
                    logging.error('moduleID out of range')
                nhits += hits.shape[0]
                single_hit_arrays.append(hits)

        try:
            if len(single_hit_arrays) > 0:
                merged_hits = np.sort(np.concatenate(single_hit_arrays), order='event_number') #hits_out.read_sorted(sortby = 'event_number', checkCSI = True)
        except ValueError:
            logging.error('Empty hit array')

        logging.info("number of hits = %s" %nhits)
        logging.info("shape of new hit table: %s" % merged_hits.shape)
        hits_out.append(merged_hits)

        logging.info('merged run %s plane %s. File saved at: %s'%(pyBARrun, plane_number, output_file))


def merge_dc_module(output_dir, plane_files, pyBARrun, plane_number,output_file_list=None):
    ''' merges 2 FE data files, columnwise, i.e. the new object has 160 columms and 336 rows.
        Also transforms and rotates modules according to ship telescope layout. The resulting planes are aligned relative to the beam.
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for module_number in range(2) :
        dc_files = plane_files[module_number*2:(module_number*2)+2]
        if module_number == 0 and plane_number == 1 :
            dc_files = [plane_files[0]]
        elif module_number == 1 and plane_number == 1:
            dc_files = plane_files[-2:]
        output_file = os.path.abspath(os.path.join(output_dir, 'pyBARrun_%s' % pyBARrun + 'plane_%s' % plane_number + '_DC_module_%s.h5' % module_number))
        single_hit_arrays = []
        nhits = 0
        with tb.open_file(output_file, 'w') as out_file:
            hits_out = out_file.create_table(out_file.root, name='Hits',
                                                description= np.dtype([
                                                    ('event_number', np.int64),
                                                    ('trigger_time_stamp',np.int64),
                                                    ('frame', np.uint8),
                                                    ('column', np.uint16),
                                                    ('row', np.uint16),
                                                    ('charge', np.uint16)]),
                                                 title='Merged hits for plane %s DC module %s' % (plane_number, module_number),
                                                 filters=tb.Filters(
                                                     complib='blosc',
                                                     complevel=5,
                                                     fletcher32=False))
            for hit_file in dc_files:
                frontEndID = int(hit_file[hit_file.rfind('module_') + 7])
                with tb.open_file(hit_file[:-3] + '_interpreted_formatted_fei4.h5', 'r') as in_file:
                    hits = in_file.root.Hits[:]
                    logging.info("merging front end %s" % frontEndID)
                    if frontEndID == 0 :
                        hits['row'] = -(hits['row'] - 81)
    #                     hits['row'] = hits['row'] + 80
    #                     hits['column'] = -hits['column'] + 337
                    elif frontEndID == 1:
                        hits['row'] = -(hits['row'] - 81) + 80
                    elif frontEndID == 2:
                        hits['column'] = (hits['column'] -337)*-1
                        hits['row'] = -(hits['row'] - 81)
                    elif frontEndID == 3:
                        hits['column'] = (hits['column'] -337)*-1
                        hits['row'] = -(hits['row'] - 81) + 80
                    elif frontEndID == 4:
                        hits['column'] = -(hits['column'] - 81)
#                         hits['row'] = -(hits['row'] - 337) + 336
                    elif frontEndID == 5:
                        hits['column'] = -(hits['column'] - 81) + 80
#                         hits['row'] = -(hits['row'] - 337) + 336
                    elif frontEndID == 6:
                        hits['column'] = -(hits['column'] - 81)
    #                     hits['row'] = (hits['row'] -337)*-1
                    elif frontEndID == 7:
    #                     hits['row'] = (hits['row'] -337)*-1
                        hits['column'] = -(hits['column'] - 81) + 80
                    else:
                        logging.error('moduleID out of range')
                    nhits += hits.shape[0]
                    single_hit_arrays.append(hits)

            try:
#                 if len(single_hit_arrays) > 0:
                merged_hits = np.sort(np.concatenate(single_hit_arrays), order='event_number') #hits_out.read_sorted(sortby = 'event_number', checkCSI = True)
            except ValueError:
                logging.error('Empty hit array')

            logging.info("number of hits = %s" %nhits)
            logging.info("shape of new hit table: %s" % merged_hits.shape)
            hits_out.append(merged_hits)

            logging.info('merged run %s plane %s module %s. File saved at: %s'%(pyBARrun, plane_number, module_number, output_file))
            if isinstance(output_file_list , list):
                output_file_list.append(output_file)

def merge_dc_module_local(plane_files, pyBARrun, plane_number, output_dir=None, output_file_list = None):
    ''' merges 2 FE data files, columnwise, i.e. the new object has 160 columms and 336 rows.
        No translation or rotation is performed.
    '''
    if not output_dir:
        output_dir = "./"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for module_number in range(2) :
        dc_files = plane_files[module_number*2:(module_number*2)+2]
        if module_number == 0 and plane_number == 1 :
            dc_files = [plane_files[0]]
        elif module_number == 1 and plane_number == 1:
            dc_files = plane_files[-2:]
        output_file = os.path.abspath(os.path.join(output_dir, 'pyBARrun_%s' % pyBARrun + '_plane_%s' % plane_number + '_DC_module_%s_local.h5' % module_number))
        single_hit_arrays = []
        nhits = 0
        with tb.open_file(output_file, 'w') as out_file:
            hits_out = out_file.create_table(out_file.root, name='Hits',
                                                 description= np.dtype([
                                                     ('event_number', np.int64),
                                                     ('trigger_time_stamp',np.int64),
                                                     ('frame', np.uint8),
                                                     ('column', np.uint16),
                                                     ('row', np.uint16),
                                                     ('charge', np.uint16)]),

                                                 title='Merged hits for plane %s DC module %s' % (plane_number, module_number),
                                                 filters=tb.Filters(
                                                     complib='blosc',
                                                     complevel=5,
                                                     fletcher32=False))
            for hit_file in dc_files:
                frontEndID = int(hit_file[hit_file.rfind('module_') + 7])
                with tb.open_file(hit_file[:-3] + '_interpreted_formatted_fei4.h5', 'r') as in_file:
                    hits = in_file.root.Hits[:]
                    logging.info("merging front end %s" % frontEndID)
                    if frontEndID % 2 == 0 :
                        hits['column'] = hits['column'] + 80
    #                     hits['row'] = hits['row'] + 80
    #                     hits['column'] = -hits['column'] + 337
                    else:
                        hits['column'] = hits['column']
                    nhits += hits.shape[0]
                    single_hit_arrays.append(hits)

            try:
#                 if len(single_hit_arrays) > 0:
                merged_hits = np.sort(np.concatenate(single_hit_arrays), order='event_number') #hits_out.read_sorted(sortby = 'event_number', checkCSI = True)
            except ValueError:
                logging.error('Empty hit array')

            logging.info("number of hits = %s" %nhits)
            logging.info("shape of new hit table: %s" % merged_hits.shape)
            hits_out.append(merged_hits)

            logging.info('merged run %s plane %s module %s. File saved at: %s'%(pyBARrun, plane_number, module_number, output_file))
            if isinstance(output_file_list , list):
                output_file_list.append(output_file)


if __name__ == "__main__":

    pybar_runs = '412' , '306', '240' # '394', '288', '222' #  '439', '333', '267'
    # dirs = ['/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/single_FE_files/run_2836',
    #         '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/single_FE_files/run_2836',
    #         '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/single_FE_files/run_2836'
    #         ]

    dirs = ['/media/niko/data/SHiP/charm_exp_2018/data/part_0x0800',
            '/media/niko/data/SHiP/charm_exp_2018/data/part_0x0801',
            '/media/niko/data/SHiP/charm_exp_2018/data/part_0x0802'
            ]

    output_dir = '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/'
    for i, directory in enumerate(dirs):
        first_plane, second_plane = get_plane_files(pyBARrun = pybar_runs[i], subpartition_dir = directory, string = 's_hi_p.h5')
        for data_file in first_plane:
            process_dut(data_file, trigger_data_format=1, do_corrections=False)
        for data_file in second_plane:
            process_dut(data_file, trigger_data_format=1, do_corrections=False)
        merge_dc_module_local(output_dir = output_dir, plane_files = first_plane, pyBARrun = pybar_runs[i], plane_number = i*2)
        merge_dc_module_local(output_dir = output_dir, plane_files = second_plane, pyBARrun = pybar_runs[i], plane_number = (i*2)+1)
