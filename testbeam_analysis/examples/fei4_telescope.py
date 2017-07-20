''' The FE-I4 telescope data example shows how to run a full analysis on data
    taken with a FE-I4 telescope.

    .. NOTE::
       Only prealignment is done here, since the telescope data is too coarse
       to profit from an aligment step. The data was recorded at DESY with
       pyBar. The telescope consists of 6 DUTs with ~ 2 cm distance between the
       planes. Only the first two and last two planes were taken here. The
       first and last plane were IBL n-in-n planar sensors and the 2 devices in
       the center 3D CNM/FBK sensors.

'''

import os
import inspect
import logging

from testbeam_analysis import hit_analysis
from testbeam_analysis import dut_alignment
from testbeam_analysis import track_analysis
from testbeam_analysis import result_analysis
from testbeam_analysis.tools import plot_utils, data_selection, analysis_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def run_analysis(data_files):
    # Dimensions
    pixel_size = [(250.0, 50.0)] * 4  # in um
    n_pixels = [(80, 336)] * 4
    z_positions = [0.0, 19500.0, 108800.0, 128300.0]  # in um
    dut_names = ("Tel_0", "Tel_1", "Tel_2", "Tel_3")

    # Create output subfolder where all output data and plots are stored
    output_folder = os.path.join(os.path.split(data_files[0])[0], 'output_fei4')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # The following shows a complete test beam analysis by calling the seperate
    # function in correct order

    # Generate noisy pixel mask for all DUTs
    for i, data_file in enumerate(data_files):
        hit_analysis.generate_pixel_mask(input_hits_file=data_file,
                                         n_pixel=n_pixels[i],
                                         pixel_mask_name='NoisyPixelMask',
                                         pixel_size=pixel_size[i],
                                         threshold=7.5,
                                         dut_name=dut_names[i])

    # Cluster hits from all DUTs
    for i, data_file in enumerate(data_files):
        hit_analysis.cluster_hits(input_hits_file=data_file,
                                  input_noisy_pixel_mask_file=os.path.splitext(data_file)[0] + '_noisy_pixel_mask.h5',
                                  min_hit_charge=0,
                                  max_hit_charge=13,
                                  column_cluster_distance=1,
                                  row_cluster_distance=2,
                                  frame_cluster_distance=2,
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

    # Create prealignment data for the DUT positions to the first DUT from the correlations
    dut_alignment.prealignment(input_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
                               output_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               z_positions=z_positions,
                               pixel_size=pixel_size,
                               fit_background=False,
                               reduce_background=True,
                               dut_names=dut_names,
                               non_interactive=True)  # Tries to find cuts automatically; deactivate to do this manualy

    # Correct all DUT hits via alignment information and merge the cluster tables to one tracklets table aligned at the event number
    dut_alignment.merge_cluster_data(input_cluster_files=input_cluster_files,
                                     n_pixels=n_pixels,
                                     output_merged_file=os.path.join(output_folder, 'Merged.h5'),
                                     pixel_size=pixel_size)

    dut_alignment.alignment(input_merged_file=os.path.join(output_folder, 'Merged.h5'),
                            input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                            align_duts=[0, 1, 2, 3],
                            select_telescope_duts=[0, 3],
                            select_fit_duts=[0, 1, 2, 3],
                            select_hit_duts=[0, 1, 2, 3],
                            max_iterations=[5],
                            max_events=100000,
                            n_pixels=n_pixels,
                            pixel_size=pixel_size,
                            dut_names=dut_names,
                            use_fit_limits=True,
                            plot=True,
                            quality_sigma=5.0)

    # Find tracks from the tracklets and stores the with quality indicator into track candidates table
    track_analysis.find_tracks(input_merged_file=os.path.join(output_folder, 'Merged.h5'),
                               input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                               output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_aligned.h5'),
                               correct_beam_alignment=True,
                               use_prealignment=False)  # If there is already an alignment info in the alignment file this has to be set

    # Fit the track candidates and create new track table
    track_analysis.fit_tracks(input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_aligned.h5'),
                              input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                              output_tracks_file=os.path.join(output_folder, 'Tracks_aligned.h5'),
                              select_duts=[0, 1, 2, 3],
                              select_hit_duts=[0, 1, 2, 3],
                              select_fit_duts=[0, 1, 2, 3],
                              n_pixels=n_pixels,
                              pixel_size=pixel_size,
                              dut_names=dut_names,
                              exclude_dut_hit=True,
                              quality_sigma=5.0,
                              use_prealignment=False)

    data_selection.select_tracks(input_tracks_file=os.path.join(output_folder, 'Tracks_aligned.h5'),
                                 output_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
                                 select_duts=[0, 1, 2, 3],
                                 duts_hit_selection=None,
                                 duts_no_hit_selection=None,
                                 duts_quality_selection=[[1, 2, 3],
                                                         [0, 2, 3],
                                                         [0, 1, 3],
                                                         [0, 1, 2]],
                                 duts_no_quality_selection=None,
                                 condition=['(n_cluster_dut_1 == 1) & (n_cluster_dut_2 == 1) & (n_cluster_dut_3 == 1)',
                                            '(n_cluster_dut_0 == 1) & (n_cluster_dut_2 == 1) & (n_cluster_dut_3 == 1)',
                                            '(n_cluster_dut_0 == 1) & (n_cluster_dut_1 == 1) & (n_cluster_dut_3 == 1)',
                                            '(n_cluster_dut_0 == 1) & (n_cluster_dut_1 == 1) & (n_cluster_dut_2 == 1)'])

    # Optional: plot some tracks (or track candidates) of a selected event range
    plot_utils.plot_events(input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
                           input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                           output_pdf_file=os.path.join(output_folder, 'Event.pdf'),
                           n_pixels=n_pixels,
                           pixel_size=pixel_size,
                           dut_names=dut_names,
                           event_range=(0, 40),
                           select_duts=[1],
                           use_prealignment=False)

    # Calculate the unconstrained residuals to check the alignment
    result_analysis.calculate_residuals(input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
                                        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                        output_residuals_file=os.path.join(output_folder, 'Residuals_aligned.h5'),
                                        select_duts=[0, 1, 2, 3],
                                        n_pixels=n_pixels,
                                        pixel_size=pixel_size,
                                        use_prealignment=False)

    result_analysis.histogram_track_angle(input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
                                          output_track_angle_file=None,
                                          input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                          select_duts=[0, 1, 2, 3],
                                          n_bins=200,
                                          dut_names=dut_names,
                                          plot=True,
                                          chunk_size=499999)

    # Calculate the efficiency and mean hit/track hit distance
    # When needed, set included column and row range for each DUT as list of tuples
    result_analysis.calculate_efficiency(input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
                                         input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
                                         output_efficiency_file=os.path.join(output_folder, 'Efficiency.h5'),
                                         select_duts=[0, 1, 2, 3],
                                         bin_size=[(250, 50)],
                                         sensor_size=[(250. * 80, 50. * 336)],
                                         minimum_track_density=2,
                                         cut_distance=500,
                                         col_range=None,
                                         row_range=None,
                                         pixel_size=pixel_size,
                                         n_pixels=n_pixels,
                                         use_prealignment=False)


if __name__ == '__main__':  # Main entry point is needed for multiprocessing under windows
    # Get the absolute path of example data
    tests_data_folder = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')
    # The location of the data files, one file per DUT
    data_files = [analysis_utils.get_data(path='examples/TestBeamData_FEI4_DUT%d.h5' % i,
                                          output=os.path.join(tests_data_folder,
                                                                   'TestBeamData_FEI4_DUT%d.h5' % i)) for i in [0, 1, 4, 5]]  # The first device is the reference for the coordinate system

    run_analysis(data_files)
