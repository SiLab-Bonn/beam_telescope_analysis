'''Example script to run a full analysis with high resolution telescope data + a fast time reference plane + a small device under tests.

The telescope consists of 6 Mimosa26 planes and one FE-I4 with a full size planar n-in-n sensor as a timing reference.
The device under tests is a small passive sensor in LFoundry 150 nm CMOS process.

The Mimosa26 has an active area of 21.2mm x 10.6mm and the pixel matrix consists of 1152 columns and 576 rows (18.4um x 18.4um pixel size).
The total size of the chip is 21.5mm x 13.7mm x 0.036mm (radiation length 9.3660734)
The matrix is divided into 4 areas. For each area the threshold can be set up individually.
The quartes are from column 0-287, 288,575, 576-863 and 864-1151.

The timing reference is about 2 cm x 2 cm divided into 80 x 336 pixels. The time stamping happens with a 40 MHz clock (25 ns).
'''

import os
import logging
import numpy as np

from beam_telescope_analysis import hit_analysis
from beam_telescope_analysis import dut_alignment
from beam_telescope_analysis import track_analysis
from beam_telescope_analysis import result_analysis
from beam_telescope_analysis.tools import data_selection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def run_analysis():
    # The location of the example data files, one file per DUT
    data_files = [r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane0.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane1.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane2.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\fe_dut-converted-synchronized_plane0.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\fe_dut-converted-synchronized_plane1.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane3.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane4.h5',
                  r'H:\Testbeam_05052016_LFCMOS\Telescope_data\kartel-converted-synchronized_plane5.h5']  # The first device is the reference for the coordinate system

    # Pixel dimesions and matrix size of the DUTs
    pixel_size = [(18.4, 18.4), (18.4, 18.4), (18.4, 18.4), (250, 50), (250, 50), (18.4, 18.4), (18.4, 18.4), (18.4, 18.4)]  # (Column, row) pixel pitch in um
    n_pixels = [(1152, 576), (1152, 576), (1152, 576), (80, 336), (80, 336), (1152, 576), (1152, 576), (1152, 576)]  # (Column, row) dimensions of the pixel matrix
    z_positions = [0., 20000, 40000, 40000 + 101000, 40000 + 101000 + 23000, 247000, 267000, 287000]  # in um
    dut_names = ("Tel 0", "Tel 1", "Tel 2", "LFCMOS3", "FEI4 Reference", "Tel 3", "Tel 4", "Tel 5")  # Friendly names for plotting

    # Folder where all output data and plots are stored
    output_folder = r'H:\Testbeam_05052016_LFCMOS\output'

    # The following shows a complete test beam analysis by calling the seperate function in correct order

    # Generate noisy pixel mask for all DUTs
    threshold = [2, 2, 2, 10, 10, 2, 2, 2]
    for i, data_file in enumerate(data_files):
        hit_analysis.generate_pixel_mask(input_hits_file=data_file,
                                         n_pixel=n_pixels[i],
                                         pixel_mask_name='NoisyPixelMask',
                                         pixel_size=pixel_size[i],
                                         threshold=threshold[i],
                                         dut_name=dut_names[i])

    # Cluster hits from all DUTs
    column_cluster_distance = [3, 3, 3, 2, 2, 3, 3, 3]
    row_cluster_distance = [3, 3, 3, 3, 3, 3, 3, 3]
    frame_cluster_distance = [0, 0, 0, 0, 0, 0, 0, 0]
    for i, data_file in enumerate(data_files):
        hit_analysis.cluster_hits(input_hits_file=data_file,
                                  input_noisy_pixel_mask_file=os.path.splitext(data_files[i])[0] + '_noisy_pixel_mask.h5',
                                  min_hit_charge=0,
                                  max_hit_charge=13,
                                  column_cluster_distance=column_cluster_distance[i],
                                  row_cluster_distance=row_cluster_distance[i],
                                  frame_cluster_distance=frame_cluster_distance[i],
                                  dut_name=dut_names[i])

    # Generate filenames for cluster data
    input_cluster_files = [os.path.splitext(data_file)[0] + '_clustered.h5'
                           for data_file in data_files]

    # Correlate the row / column of each DUT
    dut_alignment.correlate_cluster(input_cluster_files=input_cluster_files,
                                    output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                                    n_pixels=n_pixels,
                                    pixel_size=pixel_size,
                                    dut_names=dut_names)

    # Create prealignment relative to the first DUT from the correlation data
    dut_alignment.prealignment(input_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                               output_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               z_positions=z_positions,
                               pixel_size=pixel_size,
                               dut_names=dut_names,
                               fit_background=True,
                               non_interactive=False)  # Tries to find cuts automatically; deactivate to do this manualy

    # Merge the cluster tables to one merged table aligned at the event number
    dut_alignment.merge_cluster_data(input_cluster_files=input_cluster_files,
                                     output_merged_file=os.path.join(output_folder, 'Merged.h5'),
                                     n_pixels=n_pixels,
                                     pixel_size=pixel_size)

    # Apply the prealignment to the merged cluster table to create tracklets
    dut_alignment.apply_alignment(input_hit_file=os.path.join(output_folder, 'Merged.h5'),
                                  input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                  output_hit_file=os.path.join(output_folder, 'Tracklets_prealigned.h5'),
                                  force_prealignment=True)

    # Find tracks from the prealigned tracklets and stores the with quality indicator into track candidates table
    track_analysis.find_tracks(input_tracklets_file=os.path.join(output_folder, 'Tracklets_prealigned.h5'),
                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealignment.h5'))

    # Select tracks with a hit in the time reference (DUT 4) and all position devices to increase analysis speed due to data reduction
    data_selection.select_hits(hit_file=os.path.join(output_folder, 'TrackCandidates_prealignment.h5'),
                               track_quality=0b11110111,
                               track_quality_mask=0b11110111)

    # Do an alignment step with the track candidates, corrects rotations and is therefore much more precise than simple prealignment
    dut_alignment.alignment(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealignment_reduced.h5'),
                            input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                            # Order of combinaions of planes to align, one should start with high resoultion planes (here: telescope planes)
                            align_duts=[[0, 1, 2, 5, 6, 7],  # align the telescope planes first
                                        [4],  # align the time reference after the telescope alignment
                                        [3]],  # align the DUT last and not with the reference since it is rather small and would make the time reference alinmnt worse
                            # The DUTs to be used in the fit, always just the high resolution Mimosa26 planes used
                            select_fit_duts=[0, 1, 2, 5, 6, 7],
                            # The DUTs to be required to have a hit for the alignment
                            select_hit_duts=[[0, 1, 2, 4, 5, 6, 7],  # Take tracks with time reference hit
                                                [0, 1, 2, 4, 5, 6, 7],  # Take tracks with time reference hit
                                                [0, 1, 2, 3, 4, 5, 6, 7]],  # Also require hit in the small DUT
                            # The required track quality per alignment step and DUT
                            selection_track_quality=[[1, 1, 1, 0, 1, 1, 1],  # Do not require a good hit in the time refernce
                                                     [1, 1, 1, 1, 1, 1, 1],
                                                     [1, 1, 1, 1, 0, 1, 1, 1]],  # Do not require a good hit in the small DUT
                            initial_rotation=[[0., 0., 0.],
                                              [0., 0., 0.],
                                              [0., 0., 0.],
                                              # Devices 3, 4 are heavily rotated (inverted), this is not implemented now
                                              # Thus one has to set the correct rotation angles here manually
                                              [np.pi - 0.05, -0.05, -0.005],
                                              [np.pi - 0.01, -0.02, -0.0005],
                                              [0., 0, 0.],
                                              [0., 0, 0.],
                                              [0., 0, 0.]],
                            initial_translation=[[0., 0, 0.],
                                                 [0., 0, 0.],
                                                 [0., 0, 0.],
                                                 # Devices 3, 4 are heavily rotated (inverted), this is not implemented now
                                                 # Thus one has to set the correct positions here manually
                                                 [11540, 18791, 0.],
                                                 [710., 9851., 0.],
                                                 [0., 0, 0.],
                                                 [0., 0, 0.],
                                                 [0., 0, 0.]],
                            n_pixels=n_pixels,
                            use_n_tracks=200000,  # Do the alignment only on a subset of data, needed for reasonable run time
                            pixel_size=pixel_size)

    # Apply new alignment to data
    # Revert alignment from track candidates. Usually one would just apply the alignment to the merged data.
    # Due to the large beam angle track finding fails on aligned data. Thus rely on the found tracks from prealignment.
    dut_alignment.apply_alignment(input_hit_file=os.path.join(output_folder, 'TrackCandidates_prealignment_reduced.h5'),
                                  input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                  output_hit_file=os.path.join(output_folder, 'Merged_small.h5'),  # This is the new not aligned but preselected merged data file to apply (pre-) alignment on
                                  inverse=True,
                                  force_prealignment=True)

    # Apply the alignment to the merged cluster table to create tracklets
    dut_alignment.apply_alignment(input_hit_file=os.path.join(output_folder, 'Merged_small.h5'),
                                  input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                  output_hit_file=os.path.join(output_folder, 'TrackCandidates.h5'))

    # Fit track using alignment
    track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'),
                              input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                              output_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                              select_hit_duts=[0, 1, 2, 4, 5, 6, 7],
                              select_fit_duts=[0, 1, 2, 5, 6, 7],
                              selection_track_quality=1)  # Take all tracks with good hits, do not care about time reference hit quality

    # Create unconstrained residuals with aligned data
    result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                                        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                        output_residuals_file=os.path.join(output_folder, 'Residuals.h5'),
                                        n_pixels=n_pixels,
                                        pixel_size=pixel_size)

    # Calculate efficiency with aligned data
    result_analysis.calculate_efficiency(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
                                         input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                         output_efficiency_file=os.path.join(output_folder, 'Efficiency.h5'),
                                         bin_size=(10, 10),
                                         use_duts=[3],
                                         sensor_size=[(20000, 10000),
                                                      (20000, 10000),
                                                      (20000, 10000),
                                                      (20000, 20000),
                                                      (20000, 10000),
                                                      (20000, 10000),
                                                      (20000, 10000)])

    # Fit tracks using prealignmend
    track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealignment_reduced.h5'),
                              input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                              output_tracks_file=os.path.join(output_folder, 'Tracks_prealignment.h5'),
                              force_prealignment=True,
                              select_hit_duts=[0, 1, 2, 4, 5, 6, 7],
                              select_fit_duts=[0, 1, 2, 5, 6, 7],
                              selection_track_quality=1)  # Take all tracks with good hits, do not care about time reference hit quality

    # Create unconstrained residuals with prealigned data
    result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks_prealignment.h5'),
                                        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                        output_residuals_file=os.path.join(output_folder, 'Residuals_prealignment.h5'),
                                        force_prealignment=True,
                                        n_pixels=n_pixels,
                                        pixel_size=pixel_size)

    # Create efficiency plot with prealigned data
    result_analysis.calculate_efficiency(input_tracks_file=os.path.join(output_folder, 'Tracks_prealignment.h5'),
                                         input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                         output_efficiency_file=os.path.join(output_folder, 'Efficiency_prealignment.h5'),
                                         force_prealignment=True,
                                         bin_size=(10, 10),
                                         use_duts=[3],
                                         sensor_size=[(20000, 10000),
                                                      (20000, 10000),
                                                      (20000, 10000),
                                                      (20000, 20000),
                                                      (20000, 10000),
                                                      (20000, 10000),
                                                      (20000, 10000)])


if __name__ == '__main__':  # Main entry point is needed for multiprocessing under windows
    run_analysis()
