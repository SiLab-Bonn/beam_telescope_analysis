"""This script prepares raw test beam data recorded by eudaq to be analyzed by the simple python test beam analysis.

"""
import logging
import numpy as np
import tables as tb

from eudaq2np import data_np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def process_dut(raw_data_file):
    ''' Process and formate the raw data.'''
    convert_eudaq_raw_data(raw_data_file)
    align_events(raw_data_file[:-4] + '.h5', raw_data_file[:-3] + '_event_aligned.h5')
    format_hit_table(raw_data_file[:-4] + '.h5')


def convert_eudaq_raw_data(input_file):
    logging.info('Convert EUDAQ raw data using eudaq2np')
    eudaq_data = data_np(input_file)
    with tb.open_file(input_file[:-4] + '.h5', 'w') as out_file_h5:
        for readout_type, hit_data in eudaq_data.iteritems():
            logging.info('Create hit table for %s' % readout_type)
            hit_table_out = out_file_h5.create_table(out_file_h5.root, name=readout_type, description=hit_data.dtype, title='Hits for %s readout from EUDAQ raw data' % readout_type, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            hit_table_out.append(hit_data)


def align_events(input_file, output_file, chunk_size=10000000):
    '''
    Parameters
    ----------
    input_file : pytables file
    output_file : pytables file
    chunk_size :  int
        How many events are read at once into RAM for correction.
    '''
    logging.warning('Aligning events and correcting timestamp / tlu trigger number is not implemented. We trust the EUDAQ event building now.')


def format_hit_table(input_file):
    ''' Selects and renames important columns for test beam analysis and stores them into a new file.

    Parameters
    ----------
    input_file : pytables file
    output_file : pytables file
    '''

    with tb.open_file(input_file, 'r') as in_file_h5:
        min_timestamp = min([node[0]['timestamp'] for node in in_file_h5.root])
        for node in in_file_h5.root:
            hits = node[:]
            for dut_index in np.unique(hits['plane']):
                with tb.open_file(input_file[:-3] + '_DUT%d.h5' % dut_index, 'w') as out_file_h5:
                    hits_actual_dut = hits[hits['plane'] == dut_index]
                    hits_formatted = np.zeros((hits_actual_dut.shape[0], ), dtype=[('event_number', np.int64), ('frame', np.uint8), ('column', np.uint16), ('row', np.uint16), ('charge', np.uint16)])
                    hit_table_out = out_file_h5.create_table(out_file_h5.root, name='Hits', description=hits_formatted.dtype, title='Selected FE-I4 hits for test beam analysis', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                    hits_formatted['event_number'] = hits_actual_dut['timestamp'] - min_timestamp  # we take the time stamp as a event number, this uses the EUDAQ event building wich was most reliable so far
                    hits_formatted['frame'] = hits_actual_dut['frame']
                    hits_formatted['column'] = hits_actual_dut['x'] + 1
                    hits_formatted['row'] = hits_actual_dut['y'] + 1
                    hits_formatted['charge'] = hits_actual_dut['val']
                    hit_table_out.append(hits_formatted)

if __name__ == "__main__":
    # Input raw data file names
    raw_data_files = [r'/home/davidlp/Desktop/run000033.raw']

    for raw_data_file in raw_data_files:
        process_dut(raw_data_file)
