''' The FE-I4 telescope data example shows how to run a full analysis on data
    taken with a FE-I4 telescope.

    .. NOTE::
       The data was recorded with pyBAR at DESY.
       The telescope consists of 6 DUTs with ~2cm distance between the
       planes. Only the first two and last two planes are taken here. The
       first and last plane were IBL n-in-n planar sensors and the 2 devices in
       the center are 3D CNM/FBK sensors.
'''

import os
import inspect
import logging

from beam_telescope_analysis import hit_analysis
from beam_telescope_analysis import dut_alignment
from beam_telescope_analysis import track_analysis
from beam_telescope_analysis import result_analysis
from beam_telescope_analysis.tools import plot_utils, data_selection, analysis_utils
from beam_telescope_analysis.telescope.telescope import Telescope


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def run_analysis(hit_files):
    # Create output subfolder where all output data and plots are stored
    output_folder = os.path.join(os.path.split(hit_files[0])[0], 'output_fei4_telescope')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mask_files = [(os.path.splitext(hit_file)[0] + '_mask.h5') for hit_file in hit_files]
    cluster_files = [(os.path.splitext(hit_file)[0] + '_clustered.h5') for hit_file in hit_files]

    z_positions = [0.0, 19500.0, 108800.0, 128300.0]  # in um
    initial_configuration = os.path.join(output_folder, 'telescope.yaml')
    telescope = Telescope()
    telescope.add_dut(dut_type="FEI4", dut_id=0, translation_x=0, translation_y=0, translation_z=z_positions[0], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, name="Telescope 1")
    telescope.add_dut(dut_type="FEI4", dut_id=1, translation_x=0, translation_y=0, translation_z=z_positions[1], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, name="Telescope 2")
    telescope.add_dut(dut_type="FEI4", dut_id=2, translation_x=0, translation_y=0, translation_z=z_positions[2], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, name="Telescope 3")
    telescope.add_dut(dut_type="FEI4", dut_id=3, translation_x=0, translation_y=0, translation_z=z_positions[3], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, name="Telescope 4")
    telescope.save_configuration(initial_configuration)
    prealigned_configuration = os.path.join(output_folder, 'telescope_prealigned.yaml')
    aligned_configuration = os.path.join(output_folder, 'telescope_aligned.yaml')

    check_files = hit_analysis.check(
        telescope_configuration=initial_configuration,
        input_hit_files=hit_files)

    # Generate noisy pixel mask for all DUTs
    thresholds = [100, 100, 100, 100]
    pixel_mask_names = ["NoisyPixelMask"] * len(thresholds)
    mask_files = hit_analysis.mask(
        telescope_configuration=initial_configuration,
        input_hit_files=hit_files,
        pixel_mask_names=pixel_mask_names,
        thresholds=thresholds)

    # Cluster hits from all DUTs
    use_positions = [False, False, False, False]
    min_hit_charges = [0, 0, 0, 0]
    max_hit_charges = [13, 13, 13, 13]
    column_cluster_distances = [1, 1, 1, 1]
    row_cluster_distances = [3, 3, 3, 3]
    frame_cluster_distances = [4, 4, 4, 4]
    cluster_files = hit_analysis.cluster(
        telescope_configuration=initial_configuration,
        select_duts=None,
        input_hit_files=hit_files,
        input_mask_files=[None if val else mask_files[i] for i, val in enumerate(use_positions)],
        use_positions=use_positions,
        min_hit_charges=min_hit_charges,
        max_hit_charges=max_hit_charges,
        column_cluster_distances=column_cluster_distances,
        row_cluster_distances=row_cluster_distances,
        frame_cluster_distances=frame_cluster_distances)

    # Correlate each DUT with the first DUT
    hit_analysis.correlate(
        telescope_configuration=initial_configuration,
        input_files=cluster_files,
        output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
        resolution=(250.0, 50.0),
        select_reference_duts=0)

    # Create pre-alignment, take first DUT as reference
    prealigned_configuration = dut_alignment.prealign(
        telescope_configuration=initial_configuration,
        input_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
        reduce_background=True,
        select_reference_dut=0)

    # Merge all cluster tables into a single table
    hit_analysis.merge_cluster_data(
        telescope_configuration=initial_configuration,
        input_cluster_files=cluster_files,
        output_merged_file=os.path.join(output_folder, 'Merged.h5'))

    # Create alignment, take first and last DUT as reference (telescope DUTs)
    aligned_configuration = dut_alignment.align(
        telescope_configuration=prealigned_configuration,
        input_merged_file=os.path.join(output_folder, 'Merged.h5'),
        select_duts=[[0, 1, 2, 3]],  # align all planes at once
        select_telescope_duts=[0, 3],  # add outermost planes, z-axis positions are fixed for telescope DUTs, if not stated otherwise (see select_alignment_parameters)
        select_fit_duts=[0, 1, 2, 3],  # use all DUTs for track fit
        select_hit_duts=[[0, 1, 2, 3]],  # require hits in all DUTs
        max_iterations=[7],  # number of alignment iterations, the higher the number the more precise
        max_events=(100000),  # limit number of events to speed up alignment
        quality_distances=[(250.0, 50.0), (250.0, 50.0), (250.0, 50.0), (250.0, 50.0)],
        reject_quality_distances=(1000.0, 1000.0),
        use_limits=True,
        plot=True)

    # Find tracks from the tracklets and create a track candidates table
    track_analysis.find_tracks(
        telescope_configuration=aligned_configuration,
        input_merged_file=os.path.join(output_folder, 'Merged.h5'),
        output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_aligned.h5'),
        align_to_beam=True)

    # Fit the track candidates, assign quality flags, and create a track table
    track_analysis.fit_tracks(
        telescope_configuration=aligned_configuration,
        input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_aligned.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_aligned.h5'),
        select_duts=[0, 1, 2, 3],
        select_fit_duts=(0, 1, 2, 3),
        select_hit_duts=(0, 1, 2, 3),
        exclude_dut_hit=True,
        quality_distances=[(250.0, 50.0), (250.0, 50.0), (250.0, 50.0), (250.0, 50.0)],
        reject_quality_distances=(1000.0, 1000.0),
        use_limits=False,
        plot=True)

    # Do additional track selection cuts on the tracks table
    data_selection.select_tracks(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
        select_duts=[0, 1, 2, 3],
        select_hit_duts=[[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        select_no_hit_duts=None,
        select_quality_duts=[[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        select_no_quality_duts=None,
        condition='(track_chi2 < 10)')

    # Calculate the unconstrained residuals from final tracks to check the alignment
    result_analysis.calculate_residuals(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
        output_residuals_file=os.path.join(output_folder, 'Residuals_aligned.h5'),
        select_duts=[0, 1, 2, 3],
        nbins_per_pixel=20,
        use_limits=True)

    # Plotting of the tracks angles of the final tracks
    result_analysis.histogram_track_angle(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
        output_track_angle_file=None,
        n_bins=200,
        select_duts=[0, 1, 2, 3],
        plot=True)

    # Plotting of the 2D tracks density of the final tracks
    plot_utils.plot_track_density(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
        select_duts=[0, 1, 2, 3])

    # Plotting of the 2D charge distribution of the final tracks
    plot_utils.plot_charge_distribution(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
        select_duts=[0, 1, 2, 3])

    # Plotting of some final tracks (or track candidates) from selected event range
    plot_utils.plot_events(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
        output_pdf_file=os.path.join(output_folder, 'Events.pdf'),
        select_duts=[1],
        event_range=(0, 40))

    # Create final efficiency plots from final tracks
    result_analysis.calculate_efficiency(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
        output_efficiency_file=os.path.join(output_folder, 'Efficiency.h5'),
        select_duts=[0, 1, 2, 3],
        resolutions=(250, 50),
        extend_areas=(2000, 2000),
        plot_ranges=None,
        efficiency_regions=None,
        minimum_track_density=1,
        cut_distances=(1000.0, 1000.0))


if __name__ == '__main__':  # Main entry point is needed for multiprocessing under windows
    # Get the absolute path of example data
    tests_data_folder = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')
    # Path of the data files, one file per DUT
    hit_files = [analysis_utils.get_data(path='examples/TestBeamData_FEI4_DUT%d.h5' % i,
                                         output=os.path.join(tests_data_folder, 'TestBeamData_FEI4_DUT%d.h5' % i)) for i in [0, 1, 4, 5]]
    run_analysis(hit_files)
