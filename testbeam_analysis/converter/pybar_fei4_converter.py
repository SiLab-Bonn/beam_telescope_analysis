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
    ('frame', np.uint8),
    ('column', np.uint16),
    ('row', np.uint16),
    ('charge', np.uint16)])


def process_dut(raw_data_file, output_filename=None, trigger_data_format=0, do_corrections=False):
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
    analyze_raw_data(input_filename=raw_data_file, trigger_data_format=trigger_data_format)
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


def analyze_raw_data(input_filename, output_filename=None, trigger_data_format=0):  # FE-I4 raw data analysis
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
        analyze_raw_data.create_tot_hist = True
        analyze_raw_data.create_source_scan_hist = True
        analyze_raw_data.create_cluster_size_hist = True
        analyze_raw_data.create_cluster_tot_hist = True
        analyze_raw_data.align_at_trigger = True
        analyze_raw_data.fei4b = False
        analyze_raw_data.create_empty_event_hits = False
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
                output_hits_table.append(hits_data_formatted)
                output_hits_table.flush()

    return output_filename


if __name__ == "__main__":
    # Multi file processing
    # Input raw data filenames
    raw_data_files = ['pybar_raw_data.h5']
    trigger_data_format = 0
    # Simultaneous the processing of the files. The output is a formatted hit table.
    pool = Pool()
    results = [pool.apply_async(process_dut, kwds={'raw_data_file': raw_data_file, 'trigger_data_format': trigger_data_format}) for raw_data_file in raw_data_files]
    [result.get()[0] for result in results]
    pool.close()
    pool.join()
    # Single file processing
    # Input raw data filename
    # raw_data_file = 'pybar_raw_data.h5'
    # process_dut(input_file=raw_data_file, trigger_data_format=trigger_data_format)
