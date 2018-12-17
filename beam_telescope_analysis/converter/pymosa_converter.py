'''Example how to use the M26 data interpreter. A hit table is created from raw data, additionally events are build using the TLU data words.
    At the end, the hit table is formatted into the correct data format needed for testbeam analysis.
'''

import logging
import os

import numpy as np
import tables as tb

from tqdm import tqdm

from contextlib2 import ExitStack

from pyBAR_mimosa26_interpreter import data_interpreter


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

beam_telescope_analysis_dtype = np.dtype([
    ('event_number', np.int64),
    ('frame', np.uint8),
    ('column', np.uint16),
    ('row', np.uint16),
    ('charge', np.uint16)])


def process_dut(raw_data_file, output_filenames=None, trigger_data_format=2, timing_offset=None):
    ''' Process and format raw data.

    Parameters
    ----------
    raw_data_file : string
        Filename of the raw data file.
    output_filenames : list of strings
        Filenames of the output interpreted and formatted data files.
    trigger_data_format : int
        Trigger/TLU FSM data mode.

    Returns
    -------
    output_filenames : list of strings
        Filenames of the output interpreted and formatted data files.
    '''
    analyzed_data_file = os.path.splitext(raw_data_file)[0] + '_interpreted.h5'
    with data_interpreter.DataInterpreter(raw_data_file=raw_data_file, analyzed_data_file=analyzed_data_file, trigger_data_format=trigger_data_format, timing_offset=timing_offset, create_pdf=True) as mimosa_data_interpreter:
        mimosa_data_interpreter.create_occupancy_hist = True
        mimosa_data_interpreter.create_error_hist = True
        mimosa_data_interpreter.create_hit_table = True
        mimosa_data_interpreter.interpret_word_table()  # interpret raw data
    output_filenames = format_hit_table(input_filename=analyzed_data_file, output_filenames=output_filenames)
    return output_filenames


def format_hit_table(input_filename, output_filenames=None, chunk_size=1000000):
    ''' Selects and renames important columns for test beam analysis and stores them into a new file.

    Parameters
    ----------
    input_filename : string
        Filename of the input interpreted data file.
    output_filenames : list
        Filenames of the output interpreted and formatted data files.
        If None, the filenames will be generated.
    chunk_size : uint
        Chunk size of the data when reading from file.

    Returns
    -------
    output_filenames : list
        Filenames of the output interpreted and formatted data files.
    '''
    if output_filenames is None:
        output_filenames = [(os.path.splitext(input_filename)[0] + '_formatted_telescope' + str(plane + 1) + '.h5') for plane in range(6)]
    else:
        if len(output_filenames) != 6:
            raise ValueError('Output filenames must be a list of length 6')
    with tb.open_file(filename=input_filename, mode='r') as in_file_h5:
        last_event_number = np.zeros(shape=1, dtype=np.int64)
        input_hits_table = in_file_h5.root.Hits
        with ExitStack() as output_files_stack:
            out_files_h5 = [output_files_stack.enter_context(tb.open_file(filename=filename, mode='w')) for filename in output_filenames]
            output_hits_tables = []
            for out_file_h5 in out_files_h5:
                output_hits_table = out_file_h5.create_table(
                    where=out_file_h5.root,
                    name='Hits',
                    description=beam_telescope_analysis_dtype,
                    title='Hits for test beam analysis',
                    filters=tb.Filters(
                        complib='blosc',
                        complevel=5,
                        fletcher32=False))
                output_hits_tables.append(output_hits_table)

            for read_index in tqdm(range(0, input_hits_table.nrows, chunk_size)):
                hits_chunk = input_hits_table.read(read_index, read_index + chunk_size)
                if np.any(np.diff(np.concatenate((last_event_number, hits_chunk['event_number']))) < 0):
                    raise RuntimeError('The event number does not increase.')
                last_event_number = hits_chunk['event_number'][-1:]
                for plane, output_hits_table in enumerate(output_hits_tables):
                    selected_hits = (hits_chunk['plane'] == plane)
                    hits_data_formatted = np.zeros(shape=np.count_nonzero(selected_hits), dtype=beam_telescope_analysis_dtype)
                    # Format data for testbeam analysis
                    hits_data_formatted['event_number'] = hits_chunk[selected_hits]['event_number']
                    hits_data_formatted['frame'] = 0
                    hits_data_formatted['column'] = hits_chunk[selected_hits]['column'] + 1
                    hits_data_formatted['row'] = hits_chunk[selected_hits]['row'] + 1
                    hits_data_formatted['charge'] = 0
                    # Append data to table
                    output_hits_table.append(hits_data_formatted)
                    output_hits_table.flush()

    return output_filenames


if __name__ == "__main__":
    # Single file processing
    # Input raw data filename
    raw_data_file = 'pymosa_raw_data.h5'
    process_dut(input_file=raw_data_file, trigger_data_format=2)
