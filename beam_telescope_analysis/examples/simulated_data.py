''' Example script to run a full analysis on simulated data.
'''

import logging
import os

from beam_telescope_analysis import (
    hit_analysis, dut_alignment, track_analysis, result_analysis)

from beam_telescope_analysis.telescope.telescope import Telescope

from beam_telescope_analysis.tools.simulate_data import SimulateData

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - ""[%(levelname)-8s] (%(threadName)-10s) %(message)s")


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
    sim.dut_bias = [80] * sim.n_duts  # Sensor bias voltage
    sim.dut_thickness = [200] * sim.n_duts  # Sensor thickness
    # Detection threshold for each device in electrons, influences efficiency!
    sim.dut_threshold = [0.] * sim.n_duts
    sim.dut_noise = [0.] * sim.n_duts  # Noise for each device in electrons
    sim.dut_pixel_size = [(250.0, 50.0)] * sim.n_duts  # Pixel size in x / y
    sim.dut_n_pixel = [(80, 336)] * sim.n_duts  # Number of pixel in x / y
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

    initial_configuration = os.path.join(output_folder, 'telescope.yaml')
    telescope = Telescope()
    telescope.add_dut(dut_type="FEI4", dut_id=0, translation_x=0, translation_y=0,
                      translation_z=sim.z_positions[0], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, name="Telescope 1")
    telescope.add_dut(dut_type="FEI4", dut_id=1, translation_x=0, translation_y=0,
                      translation_z=sim.z_positions[1], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, name="Telescope 2")
    telescope.add_dut(dut_type="FEI4", dut_id=2, translation_x=0, translation_y=0,
                      translation_z=sim.z_positions[2], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, name="Telescope 3")
    telescope.add_dut(dut_type="FEI4", dut_id=3, translation_x=0, translation_y=0,
                      translation_z=sim.z_positions[3], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, name="Telescope 4")
    telescope.add_dut(dut_type="FEI4", dut_id=4, translation_x=0, translation_y=0,
                      translation_z=sim.z_positions[4], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, name="Telescope 5")
    telescope.add_dut(dut_type="FEI4", dut_id=5, translation_x=0, translation_y=0,
                      translation_z=sim.z_positions[5], rotation_alpha=0, rotation_beta=0, rotation_gamma=0, name="Telescope 6")
    telescope.save_configuration(initial_configuration)
    prealigned_configuration = os.path.join(
        output_folder, 'telescope_prealigned.yaml')
    aligned_configuration = os.path.join(
        output_folder, 'telescope_aligned.yaml')

    # The following shows a complete test beam analysis by calling the separate
    # function in correct order

    # Cluster hits from all DUTs
    cluster_files = hit_analysis.cluster(telescope_configuration=initial_configuration,
                                         input_hit_files=data_files,
                                         select_duts=None,
                                         input_mask_files=[None]*sim.n_duts,
                                         use_positions=[False]*sim.n_duts,
                                         min_hit_charges=[1]*sim.n_duts,
                                         max_hit_charges=[2 ** 16]*sim.n_duts,
                                         column_cluster_distances=[
                                             1]*sim.n_duts,
                                         row_cluster_distances=[1]*sim.n_duts,
                                         frame_cluster_distances=[
                                             2]*sim.n_duts,
                                         )

    # Generate filenames for cluster data
    # cluster_files = [os.path.splitext(data_file)[0] + '_clustered.h5'
    #                        for data_file in data_files]

    # Correlate the row / column of each DUT
    hit_analysis.correlate(
        telescope_configuration=initial_configuration,
        input_files=cluster_files,
        output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
        resolution=(250.0, 50.0),
        select_reference_duts=0)

    # Create alignment data for the DUT positions to the first DUT from the
    # correlation data. When needed, set offset and error cut for each DUT
    # as list of tuples
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
        select_duts=[[0, 1, 2, 3, 4, 5]],  # align all planes at once
        # add outermost planes, z-axis positions are fixed for telescope DUTs, if not stated otherwise (see select_alignment_parameters)
        select_telescope_duts=[0, 5],
        select_fit_duts=[0, 1, 2, 3, 4, 5],  # use all DUTs for track fit
        select_hit_duts=[[0, 1, 2, 3, 4, 5]],  # require hits in all DUTs
        # number of alignment iterations, the higher the number the more precise
        max_iterations=[3],
        max_events=(100000),  # limit number of events to speed up alignment
        quality_distances=[(250.0, 50.0), (250.0, 50.0), (250.0, 50.0),
                           (250.0, 50.0), (250.0, 50.0), (250.0, 50.0)],
        isolation_distances=(1000.0, 1000.0),
        use_limits=True,
        plot=True)

    # Find tracks from the tracklets and stores the with quality indicator
    # into track candidates table
    track_analysis.find_tracks(
        telescope_configuration=aligned_configuration,
        input_merged_file=os.path.join(output_folder, 'Merged.h5'),
        output_track_candidates_file=os.path.join(
            output_folder, 'TrackCandidates_aligned.h5'),
        align_to_beam=True)

    # Fit the track candidates and create new track table
    track_analysis.fit_tracks(
        telescope_configuration=aligned_configuration,
        input_track_candidates_file=os.path.join(
            output_folder, 'TrackCandidates_aligned.h5'),
        output_tracks_file=os.path.join(output_folder, 'Tracks_aligned.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        select_fit_duts=(0, 1, 2, 3, 4, 5),
        select_hit_duts=(0, 1, 2, 3, 4, 5),
        exclude_dut_hit=True,
        quality_distances=[(250.0, 50.0), (250.0, 50.0), (250.0, 50.0),
                           (250.0, 50.0), (250.0, 50.0), (250.0, 50.0)],
        isolation_distances=(1000.0, 1000.0),
        use_limits=False,
        plot=True)

    result_analysis.calculate_residuals(
        telescope_configuration=aligned_configuration,
        input_tracks_file=os.path.join(output_folder, 'Tracks_aligned.h5'),
        output_residuals_file=os.path.join(
            output_folder, 'Residuals_aligned.h5'),
        select_duts=[0, 1, 2, 3, 4, 5],
        nbins_per_pixel=20,
        use_limits=True)


if __name__ == '__main__':
    run_analysis(n_events=1000000)
