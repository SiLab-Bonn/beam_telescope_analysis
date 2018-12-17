''' Example script to run a full analysis on simulated data.
'''

import logging
import os

from beam_telescope_analysis import (
    hit_analysis, dut_alignment, track_analysis, result_analysis)

from beam_telescope_analysis.tools.simulate_data import SimulateData

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - "
                    "[%(levelname)-8s] (%(threadName)-10s) %(message)s")


def run_analysis(n_events):
    # Start simulator with random seed 0
    sim = SimulateData(random_seed=0)

    # All simulator std. settings are listed here and can be changed
    # Dimensions are in um, angles in mRad, temperatures in Kelvin
    # voltages in Volt

    # General setup
    sim.n_duts = 6  # Number of DUTs in the simulation
    sim.z_positions = [i * 10000 for i in range(sim.n_duts)]
    sim.offsets = [(-10000 + 111 * 0., -10000 + 111 * 0.)
                   for i in range(sim.n_duts)]
    sim.rotations = [(0, 0, 0)] * sim.n_duts  # in rotation around x, y, z axis
    sim.temperature = 300  # needed for charge sharing calculation

    # Beam related settings
    sim.beam_position = (0, 0)  # Average beam position in x, y at z = 0
    sim.beam_position_sigma = (2000, 2000)  # in x, y at z = 0
    sim.beam_momentum = 3200  # MeV
    sim.beam_angle = 0  # Average beam angle in theta at z = 0
    sim.beam_angle_sigma = 2  # Deviation of average beam angle in theta
    sim.tracks_per_event = 3  # Average number of tracks per event
    # Deviation from the average number of tracks
    # Allows for no track per event possible!
    sim.tracks_per_event_sigma = 1

    # Device specific settings
    sim.dut_bias = [50] * sim.n_duts  # Sensor bias voltage
    sim.dut_thickness = [200] * sim.n_duts  # Sensor thickness
    # Detection threshold for each device in electrons, influences efficiency!
    sim.dut_threshold = [0.] * sim.n_duts
    sim.dut_noise = [0.] * sim.n_duts  # Noise for each device in electrons
    sim.dut_pixel_size = [(50, 18.4)] * sim.n_duts  # Pixel size in x / y
    sim.dut_n_pixel = [(400, 1100)] * sim.n_duts  # Number of pixel in x / y
    # Efficiency for each device from 0. to 1. for hits above threshold
    sim.dut_efficiencies = [1.] * sim.n_duts
    # The effective material budget (sensor + passive compoonents) given in
    # total material distance / total radiation length
    # (https://cdsweb.cern.ch/record/1279627/files/PH-EP-Tech-Note-2010-013.pdf)
    # 0 means no multiple scattering; std. setting is the sensor thickness made
    # of silicon as material budget
    sim.dut_material_budget = [sim.dut_thickness[i] * 1e-4 / 9.370
                               for i in range(sim.n_duts)]
    # Digitization settings
    sim.digitization_charge_sharing = True
    # Shuffle hits per event to challenge track finding
    sim.digitization_shuffle_hits = True
    # Translate hit position on DUT plane to channel indices (column / row)
    sim.digitization_pixel_discretization = True

    # Create the data
    output_folder = 'simulation'  # Define a folder for output data
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    sim.create_data_and_store(os.path.join(output_folder, 'simulated_data'),
                              n_events=n_events)

    # The simulated data files, one file per DUT
    data_files = [os.path.join(output_folder, r'simulated_data_DUT%d.h5' % i)
                  for i in range(sim.n_duts)]

    # The following shows a complete test beam analysis by calling the separate
    # function in correct order

    # Cluster hits from all DUTs
    for i, data_file in enumerate(data_files):
        hit_analysis.cluster_hits(input_hits_file=data_file,
                                  min_hit_charge=1,
                                  max_hit_charge=2 ** 16,
                                  column_cluster_distance=1,
                                  row_cluster_distance=1,
                                  frame_cluster_distance=2,
                                  dut_name=data_files[i])

    # Generate filenames for cluster data
    input_cluster_files = [os.path.splitext(data_file)[0] + '_clustered.h5'
                           for data_file in data_files]

    # Correlate the row / column of each DUT
    dut_alignment.correlate_cluster(
        input_cluster_files=input_cluster_files,
        output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
        n_pixels=sim.dut_n_pixel,
        pixel_size=sim.dut_pixel_size)

    # Create alignment data for the DUT positions to the first DUT from the
    # correlation data. When needed, set offset and error cut for each DUT
    # as list of tuples
    dut_alignment.prealignment(
        input_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
        output_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        z_positions=sim.z_positions,
        pixel_size=sim.dut_pixel_size,
        # Deactivate if you have a large dataset, enhances alignment slightly
        no_fit=True,
        fit_background=False if not (sim.tracks_per_event or
                                     sim.tracks_per_event_sigma) else True,
        # Tries to find cuts automatically; deactivate to do this manualy
        non_interactive=True)

    # Correct all DUT hits via alignment information and merge the cluster
    # tables to one tracklets table aligned at the event number
    dut_alignment.merge_cluster_data(
        input_cluster_files=input_cluster_files,
        output_merged_file=os.path.join(output_folder, 'Merged.h5'),
        n_pixels=sim.dut_n_pixel,
        pixel_size=sim.dut_pixel_size)

    dut_alignment.apply_alignment(
        input_hit_file=os.path.join(output_folder, 'Merged.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_hit_file=os.path.join(output_folder,
                                             'Tracklets_prealigned.h5'),
        # If there is already an alignment info in the alignment file this has
        # to be set
        force_prealignment=True)

    # Find tracks from the tracklets and stores the with quality indicator
    # into track candidates table
    track_analysis.find_tracks(
        input_tracklets_file=os.path.join(
            output_folder, 'Tracklets_prealigned.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_track_candidates_file=os.path.join(
            output_folder, 'TrackCandidates_prealigned.h5'))

    # Fit the track candidates and create new track table
    track_analysis.fit_tracks(
        input_track_candidates_file=os.path.join(
            output_folder, 'TrackCandidates_prealigned.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
        # To get unconstrained residuals do not use DUT hit for track fit
        exclude_dut_hit=True,
        selection_track_quality=0,
        # To get close to excact efficiency heavily avoid merged tracks
        min_track_distance=1000,
        force_prealignment=True)

    result_analysis.calculate_efficiency(
        input_tracks_file=os.path.join(output_folder, 'Tracks_prealigned.h5'),
        input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
        output_efficiency_file=os.path.join(output_folder, 'Efficiency.h5'),
        bin_size=[(250, 50)],
        sensor_size=[(250. * 80, 50. * 336)],
        minimum_track_density=2,
        use_duts=None,
        cut_distance=500,
        max_distance=500,
        col_range=None,
        row_range=None,
        pixel_size=sim.dut_pixel_size,
        n_pixels=sim.dut_n_pixel,
        force_prealignment=True,
        show_inefficient_events=True)

if __name__ == '__main__':
    run_analysis(n_events=1000000)
