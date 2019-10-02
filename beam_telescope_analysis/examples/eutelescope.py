''' Example script to run a full analysis on telescope data. The original data
    can be found in the example folder of the EUTelescope framework.

    Data in https://github.com/eutelescope/eutelescope/tree/v1.0-tag/jobsub/examples/datura-150mm-DAF

    The residuals are calculated with different cuts on prealigned and aligned data
    for demonstration purpose:

    - When only prealigning the DUTs and using all DUT hits and cutting on the chi2:
      The residuals are very dependent if the prealignment is sufficient. Residuals
      are usually rather high (several 10um)

    - When aligning the DUTs and using all DUT hits and cutting on the chi2:
      The residuals and selected number of tracks are highly dependent on the
      chi2 cut. The innermost planes (Telescope 3/4) have biased/unbiased
      residuals of ~7um/~10um. The outermost planes (Telescope 1/6) have larger
      residuals of about ~8um/~20um. The two other planes (Telescope 2/5) have
      residuals of about ~5um/~7um. This is an effect of multiple scattering.

    - When aligning the DUTs and only interpolating the tracks from 2 DUTs:
      The residual for the planes 2 - 4 (DUT 1 - DUT 3) are about 6.5um in x/y and
      comparable to the residuals from the EuTelescope software (6um).

    - When using a Kalman Filter for track builing instead of an interpolation
      which takes no correlations between the measurements into account, the
      residuals can be improved by ~ 30 percent for the inner planes.

    Setup
    -----

    The telescope consists of 6 planes with 15cm clearance between the planes.
    The data was taken at DESY with ~ 5 GeV/c (Run number 36).

    The Mimosa26 has an active area of 21.2mm x 10.6mm and the pixel matrix
    consists of 1152 columns and 576 rows (18.4um x 18.4um pixel size).
    The total size of the chip is 21.5mm x 13.7mm x 0.036mm
    (radiation length 9.3660734)

    The matrix is divided into 4 areas. For each area the threshold can be set up
    individually. The quartes are from column 0-287, 288-575, 576-863 and 864-1151.

    The Mimosa26 detects ionizing particle with a density of up to
    10^6 hits/cm^2/s. The hit rate for a beam telescope is ~5 hits/frame.
'''

import os
import inspect
import logging

from beam_telescope_analysis import hit_analysis
from beam_telescope_analysis import dut_alignment
from beam_telescope_analysis import track_analysis
from beam_telescope_analysis import result_analysis
from beam_telescope_analysis.tools import data_selection, analysis_utils
from beam_telescope_analysis.telescope.telescope import Telescope


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")


def run_analysis(hit_files):
    # Create output subfolder where all output data and plots are stored
    output_folder = os.path.join(os.path.split(hit_files[0])[0], 'output_eutelescope')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mask_files = [(os.path.splitext(hit_file)[0] + '_mask.h5') for hit_file in hit_files]
    cluster_files = [(os.path.splitext(hit_file)[0] + '_clustered.h5') for hit_file in hit_files]

    z_positions = [0.0, 150000.0, 300000.0, 450000.0, 600000.0, 750000.0]  # in um
    material_budget = [100.0 / 125390.0, 100.0 / 125390.0, 100.0 / 125390.0, 100.0 / 125390.0, 100.0 / 125390.0, 100.0 / 125390.0]
    initial_configuration = os.path.join(output_folder, 'telescope.yaml')
    telescope = Telescope()
    telescope.add_dut(dut_type="Mimosa26", dut_id=0, translation_x=0, translation_y=0, translation_z=z_positions[0], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[0], name="Telescope 1")
    telescope.add_dut(dut_type="Mimosa26", dut_id=1, translation_x=0, translation_y=0, translation_z=z_positions[1], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[1], name="Telescope 2")
    telescope.add_dut(dut_type="Mimosa26", dut_id=2, translation_x=0, translation_y=0, translation_z=z_positions[2], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[2], name="Telescope 3")
    telescope.add_dut(dut_type="Mimosa26", dut_id=3, translation_x=0, translation_y=0, translation_z=z_positions[3], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[3], name="Telescope 4")
    telescope.add_dut(dut_type="Mimosa26", dut_id=4, translation_x=0, translation_y=0, translation_z=z_positions[4], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[4], name="Telescope 5")
    telescope.add_dut(dut_type="Mimosa26", dut_id=5, translation_x=0, translation_y=0, translation_z=z_positions[5], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, material_budget=material_budget[5], name="Telescope 6")
    telescope.save_configuration(initial_configuration)
    prealigned_configuration = os.path.join(output_folder, 'telescope_prealigned.yaml')
    aligned_configuration = os.path.join(output_folder, 'telescope_aligned.yaml')

    check_files = hit_analysis.check(
        telescope_configuration=initial_configuration,
        input_hit_files=hit_files)

    # Generate noisy pixel mask for all DUTs
    thresholds = [2, 2, 2, 2, 2, 2]
    # last plane has noisy cluster, use larger median filter to mask cluster
    pixel_mask_names = ["NoisyPixelMask", "NoisyPixelMask", "NoisyPixelMask", "NoisyPixelMask", "NoisyPixelMask", "DisabledPixelMask"]
    mask_files = hit_analysis.mask(
        telescope_configuration=initial_configuration,
        input_hit_files=hit_files,
        pixel_mask_names=pixel_mask_names,
        thresholds=thresholds)

    # Cluster hits from all DUTs
    use_positions = [False, False, False, False, False, False]
    min_hit_charges = [0, 0, 0, 0, 0, 0]
    max_hit_charges = [0, 0, 0, 0, 0, 0]
    column_cluster_distances = [3, 3, 3, 3, 3, 3]
    row_cluster_distances = [3, 3, 3, 3, 3, 3]
    frame_cluster_distances = [0, 0, 0, 0, 0, 0]
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
        resolution=(50.0, 50.0),
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

    # Example 1:
    # The following 4 steps are for demonstration purpose only.
    # They show track finding, fitting and selection, and residual calculation on pre-aligned hits.
    # Usually you would not do this and you would use fully aligned hits instead.

    # Step 1:
    # Find tracks from the tracklets and create a track candidates table
    track_analysis.find_tracks(
        telescope_configuration=prealigned_configuration,
        input_merged_file=os.path.join(output_folder, 'Merged.h5'),
        output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealigned.h5'),
        align_to_beam=True)

    # Step 2:
    # Fit the track candidates, assign quality flags, and create a track table
    track_analysis.fit_tracks(
        telescope_configuration=prealigned_configuration,
        input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_prealigned.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        select_fit_duts=[0, 1, 2, 3, 4, 5],
        select_hit_duts=[0, 1, 2, 3, 4, 5],
        exclude_dut_hit=True,
        isolation_distances=(1000.0, 1000.0),
        use_limits=False,
        plot=True)

    # Step 3:
    # Do additional track selection cuts on the tracks table
    data_selection.select_tracks(
        telescope_configuration=prealigned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_prealigned_selected.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        select_hit_duts=[0, 1, 2, 3, 4, 5],
        select_no_hit_duts=None,
        select_quality_duts=[[1, 2, 3, 4, 5], [0, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 1, 2, 4, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 4]],
        select_no_quality_duts=None,
        query='(track_chi2 < 500)')

    # Step 4:
    # Calculate the unconstrained residuals from pre-aligned tracks to check the pre-alignment
    result_analysis.calculate_residuals(
        telescope_configuration=prealigned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_prealigned_selected.h5'),
        output_residuals_file=os.path.join(output_folder, 'Residuals_prealigned.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        nbins_per_pixel=20,
        use_limits=True)

    # Create alignment, take first and last DUT as reference ("select_telescope_duts" parameter)
    # The position (translation and rotation) of telescope DUTs are not changed
    aligned_configuration = dut_alignment.align(
        telescope_configuration=prealigned_configuration,
        input_merged_file=os.path.join(output_folder, 'Merged.h5'),
        select_duts=[[0, 1, 2, 3, 4, 5]],  # align the telescope planes first
        select_telescope_duts=[0, 1, 2, 3, 4, 5],  # telescope planes
        select_fit_duts=[[0, 1, 2, 3, 4, 5]],
        select_hit_duts=[[0, 1, 2, 3, 4, 5]],
        max_iterations=[5],
        max_events=(100000),
        track_chi2=15.0,
        quality_distances=[(18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2)],
        isolation_distances=(1000.0, 1000.0),
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
        select_duts=[0, 1, 2, 3, 4, 5],
        select_fit_duts=[0, 1, 2, 3, 4, 5],
        select_hit_duts=[0, 1, 2, 3, 4, 5],
        exclude_dut_hit=True,
        quality_distances=[(18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2)],
        isolation_distances=(1000.0, 1000.0),
        use_limits=False,
        plot=True)

    # Calculate the unconstrained residuals from all tracks
    result_analysis.calculate_residuals(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned.h5'),
        output_residuals_file=os.path.join(output_folder, 'Residuals_aligned.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        nbins_per_pixel=20,
        use_limits=True)

    # Do additional track selection cuts on the tracks table
    data_selection.select_tracks(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        select_hit_duts=[0, 1, 2, 3, 4, 5],
        select_no_hit_duts=None,
        select_quality_duts=[[1, 2, 3, 4, 5], [0, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 1, 2, 4, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 4]],
        select_no_quality_duts=None,
        query='(track_chi2 < 15.0)')

    # Calculate the unconstrained residuals from final tracks (with chi^2 cut and quality selection)
    result_analysis.calculate_residuals(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
        output_residuals_file=os.path.join(output_folder, 'Residuals_aligned_selected.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        nbins_per_pixel=20,
        use_limits=True)

    # Example 2:
    # Use only 2 DUTs next to the fit DUT and cut on track quality.
    # Thus the track fit is just a track interpolation with chi2 = 0.
    # This is better here due to heavily scatterd tracks, where a straight line
    # assumption for all DUTs is wrong.
    # This leads to symmetric residuals in x and y for all DUTs between 2 DUTs
    # (= DUTs: 1, 2, 3, 4)
    track_analysis.fit_tracks(
        telescope_configuration=aligned_configuration,
        input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_aligned.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_pair.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        select_fit_duts=[[1, 2],  # Only select DUTs next to the DUT to fit
                         [0, 2],
                         [1, 3],
                         [2, 4],
                         [3, 5],
                         [3, 4]],
        exclude_dut_hit=True,
        quality_distances=[(18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2)],
        isolation_distances=(1000.0, 1000.0),
        use_limits=False,
        plot=True)

    # Do additional track selection cuts on the tracks table
    data_selection.select_tracks(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_pair.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_pair_selected.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        select_hit_duts=[0, 1, 2, 3, 4, 5],
        select_no_hit_duts=None,
        select_quality_duts=[[1, 2, 3, 4, 5], [0, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 1, 2, 4, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 4]],
        select_no_quality_duts=None,
        query='(track_chi2 < 5.0)')

    # Calculate the unconstrained residuals from final from final tracks (with chi^2 cut and quality selection)
    result_analysis.calculate_residuals(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_pair_selected.h5'),
        output_residuals_file=os.path.join(output_folder, 'Residuals_pair_selected.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        nbins_per_pixel=20,
        use_limits=True)

    # Example 3:
    # Use a Kalman filter to build tracks. This is the best way to build
    # tracks in case of heavily scattered tracks.
    # Fit the track candidates, assign quality flags, and create a track table
    track_analysis.fit_tracks(
        telescope_configuration=aligned_configuration,
        input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_aligned.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_aligned_kalman.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        select_fit_duts=[0, 1, 2, 3, 4, 5],
        select_hit_duts=[0, 1, 2, 3, 4, 5],
        exclude_dut_hit=True,
        quality_distances=[(18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2), (18.4 * 2, 18.4 * 2)],
        isolation_distances=(1000.0, 1000.0),
        method='kalman',
        beam_energy=5000.0,
        particle_mass=0.511,
        use_limits=False,
        plot=True)

    # Do additional track selection cuts on the tracks table
    data_selection.select_tracks(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_kalman.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_aligned_kalman_selected.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        select_hit_duts=[0, 1, 2, 3, 4, 5],
        select_no_hit_duts=None,
        select_quality_duts=[[1, 2, 3, 4, 5], [0, 2, 3, 4, 5], [0, 1, 3, 4, 5], [0, 1, 2, 4, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 4]],
        select_no_quality_duts=None,
        query='(track_chi2 < 5.0)')

    # Calculate the unconstrained residuals from final from final tracks (with chi^2 cut and quality selection)
    result_analysis.calculate_residuals(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_kalman_selected.h5'),
        output_residuals_file=os.path.join(output_folder, 'Residuals_aligned_kalman.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        nbins_per_pixel=20,
        use_limits=True)


# Main entry point is needed for multiprocessing under Windows
if __name__ == '__main__':
    # Get the absolute path of example data
    tests_data_folder = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')
    # The location of the data files, one file per DUT
    hit_files = [analysis_utils.get_data(
        path='examples/TestBeamData_Mimosa26_DUT%d.h5' % i,
        output=os.path.join(tests_data_folder, 'TestBeamData_Mimosa26_DUT%d.h5' % i)) for i in range(6)]

    run_analysis(hit_files=hit_files)
