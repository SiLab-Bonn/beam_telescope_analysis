import os
import numba

import numpy as np
import tables as tb
from tqdm import tqdm


beam_telescope_analysis_dtype = np.dtype([
    ('event_number', np.int64),
    ('frame', np.uint8),
    ('column', np.uint16),
    ('row', np.uint16),
    ('charge', np.uint16),
#    ('tdc_value', np.uint16),
#    ('tdc_timestamp', np.uint16),
#    ('tdc_status', np.uint8),
    ('tot', "<f4"),
    ('phase', "u1"),
    ('phase_quality', "u1"),
    ('veto_flg', "u1")])


CHUNKSIZE = 1000000

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def get_latest_file(working_dir, keyword, exclude=None):
    candidates = []
    for _, _, files in walklevel(working_dir, level=0):
        for f in files:
            if exclude is None:
                if keyword in f:
                    candidates.append({'name': f, 'modified': os.path.getmtime(os.path.join(working_dir, f))})
            else:
                if keyword in f and not exclude in f:
                    candidates.append({'name': f, 'modified': os.path.getmtime(os.path.join(working_dir, f))})

    if len(candidates) == 0:
        return None

    outfile = candidates[0]
    for candidate in candidates:
        if candidate['modified'] > outfile['modified']:
            outfile = candidate

    return os.path.join(working_dir, outfile['name'])

def get_hit_files(output_folder):
    hit_files = []
    hit_files.append(get_latest_file(working_dir=output_folder, keyword='header_id_1.h5'))  # M26 plane 1
    hit_files.append(get_latest_file(working_dir=output_folder, keyword='header_id_2.h5'))  # M26 plane 2
    hit_files.append(get_latest_file(working_dir=output_folder, keyword='header_id_3.h5'))  # M26 plane 3
    hit_files.append(get_latest_file(working_dir=output_folder, keyword='scan_source_ev.h5'))  # LF-Monopix2
    hit_files.append(get_latest_file(working_dir=output_folder, keyword='header_id_4.h5'))  # M26 plane 4
    # hit_files.append(get_latest_file(working_dir=output_folder, keyword='header_id_5.h5'))  # M26 plane 5
    hit_files.append(get_latest_file(working_dir=output_folder, keyword='header_id_6.h5'))  # M26 plane 6
    hit_files.append(get_latest_file(working_dir=output_folder, keyword='formatted_fei4.h5'))  # FEI4

    return hit_files

def open_hit_table(out_file, dtype=beam_telescope_analysis_dtype):
    '''Check if Hits table already exists. Load or create the table respectively.'''
    try:
        hit_table = out_file.root['Hits']
        # print('Loaded existing file')
    except:
        hit_table = out_file.create_table(out_file.root, name='Hits',
                                            description=dtype,
                                            title='Hits for test beam analysis',
                                            expectedrows=CHUNKSIZE,
                                            filters=tb.Filters(complib='blosc',
                                                                complevel=5,
                                                                fletcher32=False))

    return hit_table

def split_array(seq, chunksize):
    return [seq[i:i + chunksize] for i in range(0, len(seq), chunksize)]

@numba.njit
def copy_hit_file(data_out, hit_array, event_number_offset=0, idx=0):
    while idx < len(hit_array):
        data_out['event_number'][idx] = hit_array['event_number'][idx] + event_number_offset
        data_out['frame'][idx] = hit_array['frame'][idx]
        data_out['column'][idx] = hit_array['column'][idx]
        data_out['row'][idx] = hit_array['row'][idx]
        data_out['charge'][idx] = hit_array['charge'][idx]
        data_out['tot'][idx] = hit_array['tot'][idx]
        data_out['phase'][idx] = hit_array['phase'][idx]
        data_out['phase_quality'][idx] = hit_array['phase_quality'][idx]
        data_out['veto_flg'][idx] = hit_array['veto_flg'][idx]
        idx += 1

    return data_out[:idx]

def combine_hit_files(run_dir_list, output_folder):
    '''Loop through all runs and hit files. Combines the hit files of each DUT respectively.'''

    file_names = ['mimosa_header_id_1.h5',
                  'mimosa_header_id_2.h5',
                  'mimosa_header_id_3.h5',
                  'lfmonopix2_scan_source_ev.h5',
                  'mimosa_header_id_4.h5',
                #   'mimosa_header_id_5.h5',
                  'mimosa_header_id_6.h5',
                  'ext_trigger_scan_interpreted_valid_formatted_fei4.h5']
    event_number_offset = 0

    pbar = tqdm(total=len(run_dir_list) * len(file_names), unit='Files combined')
    for run_id in run_dir_list:
        hit_files = get_hit_files(run_id)
        max_event_numbers = []

        for pos, hit_file in enumerate(hit_files):
            with tb.open_file(hit_file, mode='r') as in_file:
                hit_array = in_file.root.Hits[:]
            with tb.open_file(os.path.join(output_folder, file_names[pos]), mode='a') as out_file:
                combined_hit_table = open_hit_table(out_file)
                for chunk in split_array(hit_array, CHUNKSIZE):
                    buffer = np.zeros(shape=2 * CHUNKSIZE, dtype=beam_telescope_analysis_dtype)
                    hit_data = copy_hit_file(buffer, chunk, event_number_offset=event_number_offset)
                    combined_hit_table.append(hit_data)
                    combined_hit_table.flush()

                max_event_numbers.append(out_file.root.Hits[-1][0])

            pbar.update(1)

        event_number_offset = np.max(max_event_numbers) + 1
        print(max_event_numbers, event_number_offset)

    pbar.close()
    print('Successfully combined hit files.')

    return


if __name__ == '__main__':
    run_list = ['33', '34']
    base_dir = '/home/lars/mnt/ceph/testbeam/lfmonopix2/202211_DESY/'
    output_dir = 'combined_run'

    run_dir_list = []
    for run_id in run_list:
        run_dir_list.append(base_dir + 'run_' + run_id)
        output_dir = output_dir + '-' + run_id

    output_folder = os.path.join(base_dir, output_dir)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print('Combining runs ', run_list)
    combine_hit_files(run_dir_list, output_folder)