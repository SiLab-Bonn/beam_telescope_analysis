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

from pybar.analysis.analyze_raw_data import AnalyzeRawData

from beam_telescope_analysis.hit_analysis import default_hits_dtype


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def process_dut(raw_data_file, output_filename=None, trigger_data_format=0):
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
    analyze_raw_data(input_filename=raw_data_file, trigger_data_format=trigger_data_format)
    if isinstance(raw_data_file, (list, tuple)):
        raw_data_filename = os.path.splitext(sorted(raw_data_file)[0])[0]  # get filename with the lowest index
    else:  # string
        raw_data_filename = os.path.splitext(raw_data_file)[0]
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
        analyze_raw_data.fei4b = True
        analyze_raw_data.create_empty_event_hits = False
        # analyze_raw_data.n_bcid = 16
        # analyze_raw_data.max_tot_value = 13
        analyze_raw_data.interpreter.set_warning_output(False)
        analyze_raw_data.interpret_word_table()
        analyze_raw_data.interpreter.print_summary()
        analyze_raw_data.plot_histograms()


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
                description=default_hits_dtype,
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
                hits_data_formatted = np.zeros(shape=hits_chunk.shape[0], dtype=default_hits_dtype)
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
    # process_dut(raw_data_file=raw_data_file, trigger_data_format=trigger_data_format)
