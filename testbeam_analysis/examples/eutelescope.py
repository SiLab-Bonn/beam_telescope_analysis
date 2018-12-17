''' Example script to run a full analysis on telescope data. The original data
    can be found in the example folder of the EUTelescope framework.

    Data in https://github.com/eutelescope/eutelescope/tree/v1.0-tag/jobsub/examples/datura-150mm-DAF

    The residuals are calculated with different cuts on prealigned and aligned data
    for demonstration purpose:

    - When only prealigning the DUTs and using all DUT hits and cutting on the chi2:
      The residuals are very dependent if the prealignment is sufficient. Residuals
      are usually rather high (several 10 um)

    - When aligning the DUTs and only interpolating the tracks from 2 DUTs:
      The residual for the planes 2 - 4 (DUT 1 - DUT 3) are about 6.5 um in x/y and
      comparable to the residuals from the EuTelescope software (6 um).

    - When aligning the DUTs and using all DUT hits and cutting on the chi2:
      The residuals and selected number of tracks are highly dependent on the
      chi2 cut and are at least 6 um and usually < 10 um depending on the
      plane position. This is an effect of multiple scattering. The outer most plans
      have a rather high residual (~ 18 um)

    - When using a Kalman Filter for track builing instead of an interpolation
      which takes no correlations between the measurements into account, the
      residuals can be improved by ~ 30 percent for the inner planes.

    Setup
    -----

    The telescope consists of 6 planes with 15 cm clearance between the planes.
    The data was taken at Desy with ~ 5 GeV/c (Run number 36).

    The Mimosa26 has an active area of 21.2mm x 10.6mm and the pixel matrix
    consists of 1152 columns and 576 rows (18.4um x 18.4um pixel size).
    The total size of the chip is 21.5mm x 13.7mm x 0.036mm
    (radiation length 9.3660734)

    The matrix is divided into 4 areas. For each area the threshold can be set up
    individually. The quartes are from column 0-287, 288-575, 576-863 and 864-1151.

    The Mimosa26 detects ionizing particle with a density of up to
    10^6 hits / cm^2 / s. The hit rate for a beam telescope is ~5 hits / frame.
'''

import os
import inspect
import logging

from testbeam_analysis import (hit_analysis, dut_alignment, track_analysis,
                               result_analysis)
from testbeam_analysis.tools import analysis_utils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s]\
     (%(threadName)-10s) %(message)s")


def run_analysis(data_files):
    # Pixel dimesions and matrix size of the DUTs
    pixel_size = [(18.4, 18.4)] * 6  # Column, row pixel pitch in um
    n_pixels = [(1152, 576)] * 6  # Number of pixel on column, row

    z_positions = [0., 150000, 300000, 450000, 600000, 750000]  # z position in um
    # Friendly names for plotting
    dut_names = ("Tel_0", "Tel_1", "Tel_2", "Tel_3", "Tel_4", "Tel_5")

    # Create output subfolder where all output data and plots are stored
    output_folder = os.path.join(os.path.split(data_files[0])[0],
                                 'output_eutel')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # The following shows a complete test beam analysis by calling the
    # seperate function in correct order

    # Generate noisy pixel mask for all DUTs
    for i, data_file in enumerate(data_files):
        hit_analysis.generate_pixel_mask(input_hits_file=data_file,
                                         n_pixel=n_pixels[i],
                                         pixel_mask_name='NoisyPixelMask',
                                         pixel_size=pixel_size[i],
                                         threshold=0.5,
                                         dut_name=dut_names[i])

    # Cluster hits from all DUTs
    for i, data_file in enumerate(data_files):
        hit_analysis.cluster_hits(input_hits_file=data_file,
                                  input_noisy_pixel_mask_file=os.path.splitext(data_files[i])[0] + '_noisy_pixel_mask.h5',
                                  min_hit_charge=0,
                                  max_hit_charge=1,
                                  column_cluster_distance=3,
                                  row_cluster_distance=3,
                                  frame_cluster_distance=1,
                                  dut_name=dut_names[i])

    # Generate filenames for cluster data
    input_cluster_files = [os.path.splitext(data_file)[0] + '_clustered.h5'
                           for data_file in data_files]

    # Correlate the row / column of each DUT
    dut_alignment.correlate_cluster(input_cluster_files=input_cluster_files,
                                    output_correlation_file=os.path.join(
                                        output_folder, 'Correlation.h5'),
                                    n_pixels=n_pixels,
                                    pixel_size=pixel_size,
                                    dut_names=dut_names)

    # Create prealignment relative to the first DUT from the correlation data
    input_correlation_file = os.path.join(output_folder, 'Correlation.h5')
    dut_alignment.prealignment(input_correlation_file=input_correlation_file,
                               output_alignment_file=os.path.join(
                                   output_folder, 'Alignment.h5'),
                               z_positions=z_positions,
                               pixel_size=pixel_size,
                               dut_names=dut_names,
                               # This data has several tracks per event and
                               # noisy pixel, thus fit existing background
                               fit_background=True,
                               # Tries to find cuts automatically;
                               # deactivate to do this manualy
                               non_interactive=True)

    # Merge the cluster tables to one merged table aligned at the event number
    dut_alignment.merge_cluster_data(input_cluster_files=input_cluster_files,
                                     output_merged_file=os.path.join(
                                         output_folder, 'Merged.h5'),
                                     n_pixels=n_pixels,
                                     pixel_size=pixel_size)

    # Apply the prealignment to the merged cluster table to create tracklets
    dut_alignment.apply_alignment(
        input_hit_file=os.path.join(output_folder, 'Merged.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_hit_file=os.path.join(output_folder,
                                     'Tracklets_prealigned.h5'),
        force_prealignment=True)

    # Find tracks from the prealigned tracklets and stores them with quality
    # indicator into track candidates table
    track_analysis.find_tracks(
        input_tracklets_file=os.path.join(output_folder,
                                          'Tracklets_prealigned.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_track_candidates_file=os.path.join(
            output_folder, 'TrackCandidates_prealignment.h5')
    )

    # The following two steps are for demonstration only.
    # They show track fitting and residual calculation on
    # prealigned hits. Usually you are not interested in this and will use
    # the aligned hits directly.

    # Step 1.: Fit the track candidates and create new track table (using the
    # prealignment!)
    track_analysis.fit_tracks(
        input_track_candidates_file=os.path.join(
            output_folder, 'TrackCandidates_prealignment.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
        # To get unconstrained residuals do not use DUT
        # hit for track fit
        exclude_dut_hit=True,
        # This is just for demonstration purpose, usually
        # uses fully aligned hits
        force_prealignment=True,
        selection_track_quality=0)  # We will cut on chi2

    # Step 2.:  Calculate the residuals to check the alignment (using the
    # prealignment!)
    result_analysis.calculate_residuals(
        input_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_residuals_file=os.path.join(output_folder,
                                           'Residuals_prealigned.h5'),
        n_pixels=n_pixels,
        pixel_size=pixel_size,
        max_chi2=2000,
        # This is just for demonstration purpose
        # you usually use fully aligned hits
        force_prealignment=True)

    # Do an alignment step with the track candidates, corrects rotations and
    # is therefore much more precise than simple prealignment
    dut_alignment.alignment(
        input_track_candidates_file=os.path.join(
            output_folder, 'TrackCandidates_prealignment.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        n_pixels=n_pixels,
        pixel_size=pixel_size)

    # Apply the alignment to the merged cluster table to create tracklets
    dut_alignment.apply_alignment(
        input_hit_file=os.path.join(output_folder, 'Merged.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_hit_file=os.path.join(output_folder, 'Tracklets.h5')
    )

    # Find tracks from the tracklets and stores the with quality indicator
    # into track candidates table
    track_analysis.find_tracks(
        input_tracklets_file=os.path.join(output_folder, 'Tracklets.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_track_candidates_file=os.path.join(
            output_folder, 'TrackCandidates.h5')
    )

    # Example 1: use all DUTs in fit and cut on chi2
    track_analysis.fit_tracks(
        input_track_candidates_file=os.path.join(output_folder,
                                                 'TrackCandidates.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_all.h5'),
        # To get unconstrained residuals do not use DUT
        # hit for track fit
        exclude_dut_hit=True,
        # We do not cut on track quality but on chi2 later
        selection_track_quality=0)

    # Create unconstrained residuals with chi2 cut
    result_analysis.calculate_residuals(
        input_tracks_file=os.path.join(output_folder, 'Tracks_all.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_residuals_file=os.path.join(output_folder, 'Residuals_all_chi2_cut.h5'),
        # The chi2 cut has a large influence on
        # the residuals and number of tracks,
        # since the resolution is dominated by
        # multiple scattering
        max_chi2=500,
        n_pixels=n_pixels,
        pixel_size=pixel_size)

    # Create unconstrained residuals
    result_analysis.calculate_residuals(
        input_tracks_file=os.path.join(output_folder, 'Tracks_all.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_residuals_file=os.path.join(output_folder, 'Residuals_all.h5'),
        n_pixels=n_pixels,
        pixel_size=pixel_size)

    # Example 2: Use only 2 DUTs next to the fit DUT and cut on track quality.
    # Thus the track fit is just a track interpolation with chi2 = 0.
    # This is better here due to heavily scatterd tracks, where a straight line
    # assumption for all DUTs is wrong.
    # This leads to symmetric residuals in x and y for all DUTs between 2 DUTs
    # (= DUTs: 1, 2, 3, 4)
    track_analysis.fit_tracks(
        input_track_candidates_file=os.path.join(output_folder,
                                                 'TrackCandidates.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_some.h5'),
        selection_hit_duts=[[1, 2],  # Only select DUTs next to the DUT to fit
                            [0, 2],
                            [1, 3],
                            [2, 4],
                            [3, 5],
                            [3, 4]],
        selection_track_quality=1)  # We cut on track quality

    # Create unconstrained residuals
    result_analysis.calculate_residuals(
        input_tracks_file=os.path.join(output_folder, 'Tracks_some.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_residuals_file=os.path.join(output_folder, 'Residuals_some.h5'),
        n_pixels=n_pixels,
        pixel_size=pixel_size)

    # Example 3: Use a Kalman Filter to build tracks. This is the best way to build
    # tracks in case of heavily scattered tracks.
    track_analysis.fit_tracks(
        input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_all_Kalman.h5'),
        exclude_dut_hit=True,
        pixel_size=pixel_size,
        n_pixels=n_pixels,
        beam_energy=5000.,
        material_budget=[100. / 125390., 100. / 125390., 100. / 125390., 100. / 125390., 100. / 125390., 100. / 125390.],
        selection_track_quality=0,
        method='Kalman')

    # Create unconstrained residuals
    result_analysis.calculate_residuals(
        input_tracks_file=os.path.join(output_folder, 'Tracks_all_Kalman.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_residuals_file=os.path.join(output_folder, 'Residuals_all_Kalman.h5'),
        n_pixels=n_pixels,
        pixel_size=pixel_size,
        npixels_per_bin=10,
        nbins_per_pixel=50)


# Main entry point is needed for multiprocessing under windows
if __name__ == '__main__':
    # Get the absolute path of example data
    tests_data_folder = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), 'data')
    # The location of the data files, one file per DUT
    data_files = [analysis_utils.get_data(path='examples/TestBeamData_Mimosa26_DUT%d.h5' % i,
                                          output=os.path.join(tests_data_folder,
                                                              'TestBeamData_Mimosa26_DUT%d.h5' % i)) for i in range(6)]  # The first device is the reference for the coordinate system

    run_analysis(data_files)
