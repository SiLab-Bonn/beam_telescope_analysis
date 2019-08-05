import logging
import os

import numpy as np

from beam_telescope_analysis.converter import pybar_ship_converter
# from beam_telescope_analysis.converter import pymosa_converter
from beam_telescope_analysis import hit_analysis
from beam_telescope_analysis import dut_alignment
from beam_telescope_analysis import track_analysis
from beam_telescope_analysis import result_analysis
from beam_telescope_analysis.tools import data_selection
from beam_telescope_analysis.tools import plot_utils
from beam_telescope_analysis.telescope.telescope import Telescope
from beam_telescope_analysis.tools.merge_data import open_files, merge_hits, save_new_tables
# from beam_telescope_analysis.telescope import dut

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')

output_dir = "/media/niko/big_data/charm_testbeam_july18/analysis"
# data_folder = os.path.dirname(os.path.realpath(__file__))

runs = {
        2849 : ("425", "319", "253"),
        2850 : ("426", "320", "254"),
        2851 : ("427", "321", "255"),
        2852 : ("428", "322", "256"),
        2853 : ("429", "323", "257"),
        2854 : ("430", "324", "258"),
        2855 : ("431", "325", "259"),
        2856 : ("432", "326", "260"),
        2857 : ("433", "327", "261"),
        }

raw_data_dirs = ['/media/niko/data/SHiP/charm_exp_2018/data/part_0x0800',
                 '/media/niko/data/SHiP/charm_exp_2018/data/part_0x0801',
                 '/media/niko/data/SHiP/charm_exp_2018/data/part_0x0802'
                 ]

# for key in runs.keys():
#     run_number = key
#     pybar_runs = runs[key]
#     logger.info("============================== converting run %s ==============================" % run_number)
#     # logger.info("============================= using pybar files  ============================", pybar_runs )
#
#     data_files_folder = os.path.join(output_dir, 'run_%s'% run_number)
#     output_folder = os.path.join(output_dir, 'output_folder_run_%s' % run_number)
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     if not os.path.exists(data_files_folder):
#         os.makedirs(data_files_folder)
#
#     hit_files = []
#     for i, folder in enumerate(raw_data_dirs):
#         first_plane, second_plane = pybar_ship_converter.get_plane_files(pyBARrun = pybar_runs[i], subpartition_dir = folder)
#         for data_file in first_plane:
#             pybar_ship_converter.process_dut(data_file, trigger_data_format=1, do_corrections=False, empty_events=True)
#         for data_file in second_plane:
#             pybar_ship_converter.process_dut(data_file, trigger_data_format=1, do_corrections=False, empty_events=True)
#         pybar_ship_converter.merge_dc_module_local(output_dir = data_files_folder, plane_files = first_plane, pyBARrun = pybar_runs[i], plane_number = i*2, output_file_list = hit_files)
#         pybar_ship_converter.merge_dc_module_local(output_dir = data_files_folder, plane_files = second_plane, pyBARrun = pybar_runs[i], plane_number = (i*2)+1 , output_file_list = hit_files)
#
#     hit_tables_in = open_files(hit_files)
#     new_tables = merge_hits(hit_tables_in=hit_tables_in)
#     hit_files = save_new_tables(hit_files, new_tables)

# choose Hit files to be merged
merge_hit_files = {}
for plane in range(0,6):
    for module in range(0,2):
        plane_hit_files = []
        for key in runs.keys():
            run_number = key
            pybar_runs = runs[key]
            plane_hit_files.append('/media/niko/big_data/charm_testbeam_july18/analysis/run_%s/pyBARrun_%s_plane_%s_DC_module_%s_local_corr_evts.h5'
                                    % (run_number, pybar_runs[plane/2], plane, module))
        merge_hit_files.update({str(plane)+str(module):plane_hit_files})

#merge hit files and update list
output_folder = os.path.join(output_dir, 'one_brick')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

hit_files = []
logging.info("merging hit files of different runs")
for key in merge_hit_files.keys():
    output_name = os.path.join(output_dir, output_folder + '/plane%s_module_%s_local_corr_evts_merged.h5' % (key[0], key[1]))
    out_file_name = pybar_ship_converter.merge_runs(hit_files = merge_hit_files[key], table_name="Hits", max_event_size = 10, output_name = output_name )
    hit_files.append(out_file_name)
hit_files.sort()

# hit_files = ['/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane0_module_0_local_corr_evts_merged.h5',
#              '/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane0_module_1_local_corr_evts_merged.h5',
#              '/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane1_module_0_local_corr_evts_merged.h5',
#              '/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane1_module_1_local_corr_evts_merged.h5',
#              '/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane2_module_0_local_corr_evts_merged.h5',
#              '/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane2_module_1_local_corr_evts_merged.h5',
#              '/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane3_module_0_local_corr_evts_merged.h5',
#              '/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane3_module_1_local_corr_evts_merged.h5',
#              '/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane4_module_0_local_corr_evts_merged.h5',
#              '/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane4_module_1_local_corr_evts_merged.h5',
#              '/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane5_module_0_local_corr_evts_merged.h5',
#              '/media/niko/big_data/charm_testbeam_july18/analysis/no_brick/plane5_module_1_local_corr_evts_merged.h5'
#              ]

output_folder = os.path.join(output_dir, 'output_folder_one_brick')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

z_positions = [0., 6400., 26400., 32800., 52800., 59200., 79200., 85600., 105600., 112000.,  132000., 138400. ]# (in um)

shift = 0 #(336/2. * 50.)

''' rotations: 1. alpha (around x) 2. beta (around y) 3. gamma (around z)'''

telescope = Telescope()

telescope.add_dut(dut_type="FEI4DCModule", dut_id=0, translation_x= -shift, translation_y= 0, translation_z=z_positions[0],
                  rotation_alpha=np.pi, rotation_beta= 0, rotation_gamma= 0.5 * np.pi, name="Telescope 0")

telescope.add_dut(dut_type="FEI4DCModule", dut_id=1, translation_x= shift, translation_y= 0, translation_z=z_positions[1],
                  rotation_alpha=0, rotation_beta= 0, rotation_gamma= 0.5 * np.pi, name="Telescope 1")

telescope.add_dut(dut_type="FEI4DCModule", dut_id=2, translation_x=0, translation_y= -shift, translation_z=z_positions[2],
                  rotation_alpha=0, rotation_beta=np.pi , rotation_gamma=0, name="Telescope 2")

telescope.add_dut(dut_type="FEI4DCModule", dut_id=3, translation_x=0, translation_y= shift, translation_z=z_positions[3],
                  rotation_alpha=0, rotation_beta= 0 , rotation_gamma= np.pi, name="Telescope 3")

telescope.add_dut(dut_type="FEI4DCModule", dut_id=4, translation_x= -shift, translation_y= 0, translation_z=z_positions[4],
                  rotation_alpha=np.pi, rotation_beta=0, rotation_gamma= 0.5 * np.pi, name="Telescope 4")

telescope.add_dut(dut_type="FEI4DCModule", dut_id=5, translation_x= shift, translation_y= 0, translation_z=z_positions[5],
                  rotation_alpha= 0, rotation_beta=0, rotation_gamma= 0.5 * np.pi, name="Telescope 5")

telescope.add_dut(dut_type="FEI4DCModule", dut_id=6, translation_x=0, translation_y= -shift, translation_z=z_positions[6],
                  rotation_alpha=0, rotation_beta=np.pi, rotation_gamma=0, name="Telescope 6")

telescope.add_dut(dut_type="FEI4DCModule", dut_id=7, translation_x=0, translation_y= shift, translation_z=z_positions[7],
                  rotation_alpha=0, rotation_beta=0, rotation_gamma=np.pi, name="Telescope 7")

telescope.add_dut(dut_type="FEI4DCModule", dut_id=8, translation_x= -shift, translation_y= 0, translation_z=z_positions[8],
                  rotation_alpha=np.pi, rotation_beta=0, rotation_gamma=0.5 * np.pi, name="Telescope 8")

telescope.add_dut(dut_type="FEI4DCModule", dut_id=9, translation_x= shift, translation_y= 0, translation_z=z_positions[9],
                  rotation_alpha=0, rotation_beta=0, rotation_gamma= 0.5 * np.pi, name="Telescope 9")

telescope.add_dut(dut_type="FEI4DCModule", dut_id=10, translation_x=0, translation_y= -shift, translation_z=z_positions[10],
                  rotation_alpha=0, rotation_beta=np.pi, rotation_gamma=0, name="Telescope 10")

telescope.add_dut(dut_type="FEI4DCModule", dut_id=11, translation_x=0, translation_y= shift, translation_z=z_positions[11],
                  rotation_alpha=0, rotation_beta=0, rotation_gamma=np.pi, name="Telescope 11")


telescope.save_configuration(os.path.join(output_folder,'no_brick_telescope.yaml'))
# telescope.save_configuration('run%d_telescope_local_coordinates.yaml' % run_number)

pixel_size = [(250, 50)]*12
n_pixels = [(160, 336)]*12

# mixed coordinates
# check_files = hit_analysis.check(
#     telescope_configuration=os.path.join(output_folder,'run%d_telescope.yaml' % run_number),
#     input_hit_files=hit_files,
#     )

thresholds = [10,] * 12
pixel_mask_names = ["NoisyPixelMask"] * len(thresholds)
mask_files = hit_analysis.mask(
    telescope_configuration=os.path.join(output_folder,'no_brick_telescope.yaml'),
    input_hit_files=hit_files,
    pixel_mask_names=pixel_mask_names,
    thresholds=thresholds)

# mask_files = [(os.path.splitext(hit_file)[0] + '_mask.h5') for hit_file in hit_files]

use_positions = [False, ]*12

min_hit_charges = [0,] * 12
max_hit_charges = [13,] * 12
column_cluster_distances = [1,]*12
row_cluster_distances = [1,] * 12
frame_cluster_distances = [1,] * 12
cluster_files = hit_analysis.cluster(
   telescope_configuration=os.path.join(output_folder, 'no_brick_telescope.yaml'),
   select_duts=None,
   input_hit_files=hit_files,
   input_mask_files=[None]*12, #[None if val else mask_files[i] for i, val in enumerate(use_positions)],
   use_positions=use_positions,
   min_hit_charges=min_hit_charges,
   max_hit_charges=max_hit_charges,
   column_cluster_distances=column_cluster_distances,
   row_cluster_distances=row_cluster_distances,
   frame_cluster_distances=frame_cluster_distances)

# cluster_files = [(os.path.splitext(hit_file)[0] + '_clustered.h5') for hit_file in hit_files]

hit_analysis.correlate(
   telescope_configuration= os.path.join(output_folder,'no_brick_telescope.yaml'),
   input_files=cluster_files,
   output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
   resolution=(250.0, 250.0),
   select_reference_duts=1)

prealigned_configuration = dut_alignment.prealign(
   telescope_configuration=os.path.join(output_folder,'no_brick_telescope.yaml'),
   input_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
   reduce_background=True,
   select_reference_dut=1)

# prealigned_configuration = os.path.join(output_folder,'run%d_telescope_prealigned.yaml' % run_number)

# hit_analysis.correlate(
#     telescope_configuration= prealigned_configuration,
#     input_files=cluster_files,
#     output_correlation_file=os.path.join(output_folder, 'Correlation_prealign.h5'),
#     resolution=(250.0, 250.0),
#     select_reference_duts=None)

hit_analysis.merge_cluster_data(
    telescope_configuration=os.path.join(output_folder,'no_brick_telescope.yaml'),
    input_cluster_files=cluster_files,
    output_merged_file=os.path.join(output_folder, 'Merged.h5'))

aligned_configuration = dut_alignment.align(
    telescope_configuration=prealigned_configuration,
    input_merged_file=os.path.join(output_folder, 'Merged.h5'),
    select_duts=[[1, 3, 5, 7,9,11], [1, 2, 5, 6,],],
    select_telescope_duts=[1,9],
    select_fit_duts=[[1, 3, 5, 7,9,11], [1, 2, 5, 6,9,10],],
    select_hit_duts=[[1, 3, 5, 7,9,11], [1, 2, 5, 6,9,10],],
    max_iterations=[3, 3],
    max_events= None, #[1000000, 1000000],
    quality_distances=(250.0, 50.0),
    reject_quality_distances=(1000.0, 1000.0),
    use_limits=True,
    plot=True)

# aligned_configuration = os.path.join(output_folder,'run%d_telescope_aligned.yaml' % run_number)
# aligned_configuration = '../run%d/run%d_telescope_aligned.yaml' % tuple([125] * 2)

track_analysis.find_tracks(
    telescope_configuration=aligned_configuration,
    input_merged_file=os.path.join(output_folder, 'Merged.h5'),
    output_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_aligned.h5'),
    align_to_beam=True)


track_analysis.fit_tracks(
    telescope_configuration=aligned_configuration,
    input_track_candidates_file=os.path.join(output_folder, 'TrackCandidates_aligned.h5'),
    output_tracks_file=os.path.join(output_folder, 'Tracks_aligned.h5'),
    select_duts=[1, 2, 3, 5],
    select_fit_duts=[[1, 3, 5, 7],[1,2,5,6], [1, 3, 5, 7]],
    select_hit_duts=[[1, 3, 5, 7],[1,2,5,6], [1, 3, 5, 7]],
    exclude_dut_hit=True,
    quality_distances=(500.0, 500.0),
    use_limits=False,
    plot=True)

data_selection.select_tracks(
    telescope_configuration=aligned_configuration,
    input_tracks_file=os.path.join(output_folder, 'Tracks_aligned.h5'),
    output_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
    select_duts=[1,2,3],
    select_hit_duts=[[1, 3, 5, 7],[1,2,5,6], [1, 3, 5, 7]],
    select_no_hit_duts=None,
    select_quality_duts=[[1, 3, 5, 7],[1,2,5,6], [1, 3, 5, 7]],
    select_no_quality_duts=None,
    condition='(track_chi2 < 100)')

result_analysis.histogram_track_angle(
    telescope_configuration=aligned_configuration,
    input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
    output_track_angle_file=None,
    n_bins=200,
    select_duts=[1, 2, 3],
    plot=True)

result_analysis.calculate_residuals(
    telescope_configuration=aligned_configuration,
    input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
    output_residuals_file=os.path.join(output_folder, 'Residuals_aligned.h5'),
    select_duts=[1, 2, 3],
    use_limits=True)

plot_utils.plot_track_density(
    telescope_configuration=aligned_configuration,
    input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
    select_duts=[1, 2,3,])

# plot_utils.plot_charge_distribution(
#     telescope_configuration=aligned_configuration,
#     input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
#     select_duts=[3, 4, 5])

# result_analysis.calculate_efficiency(
#     telescope_configuration=aligned_configuration,
#     input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
#     input_cluster_files=[cluster_files[3]],
#     output_efficiency_file=os.path.join(output_folder, 'Efficiency.h5'),
#     select_duts=[3],
#     resolutions=[(5, 5)],
#     extend_areas=[2000],
#     plot_ranges=[[(5000, 11000), (-3000, -9000)]],
#     minimum_track_density=1,
#     cut_distances=[250])
# result_analysis.calculate_efficiency(
#     telescope_configuration=aligned_configuration,
#     input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
#     output_efficiency_file=os.path.join(output_folder, 'Efficiency.h5'),
#     select_duts=[3, 4, 5],
#     resolutions=[(10, 10), (10, 10), (10, 10)],
#     extend_areas=[2000, 200, 2000],
#     plot_ranges=[[(5000, 11000), (-3000, -9000)], [(4500, 10500), (-3000, -9000)], None],
#     minimum_track_density=1,
#     cut_distances=[250, 250, 250])

# result_analysis.calculate_purity(
#     telescope_configuration=aligned_configuration,
#     input_tracks_file=os.path.join(output_folder, 'Tracks_aligned_selected.h5'),
#     output_purity_file=os.path.join(output_folder, 'Purity.h5'),
#     select_duts=[2, 3, 4, 5],
#     bin_size=[(18.4, 18.4), (10, 10), (10, 10), (10, 10)],
#     minimum_hit_density=5,
#     cut_distance=500)

# import tables as tb
#
# raw_data_files = [None, "/media/big_data/testbeam-dbm-may-2016/lfcmos3/telescope_run/4005_telescope_run_ext_trigger_scan.h5", "/media/big_data/testbeam-dbm-may-2016/proto09/telescope_run/4005_telescope_run_ext_trigger_scan.h5"]
# dut_masks = [None] * len(raw_data_files)
# for index, file_name in enumerate(raw_data_files):
#     if file_name is not None:
#         with tb.open_file(file_name, 'r') as in_h5_file:
#             enable_mask = in_h5_file.root.configuration.Enable[:]
#             dut_masks[index] = 1 - enable_mask
#     else:
#         dut_masks[index] = None
#
# result_analysis.calculate_efficiency(input_tracks_file=os.path.join(output_folder, 'Tracks.h5'),
#                                      input_alignment_file=os.path.join(output_folder, 'Alignment.h5'),
#                                      output_efficiency_file=os.path.join(output_folder, 'Efficiency.h5'),
#                                      dut_names=dut_names,
#                                      pixel_size=pixel_size,
#                                      n_pixels=n_pixels,
#                                      bin_size=(pixel_size[2], (10, 10), (10, 10)),
#                                      use_duts=[2, 3, 4],
#                                      cut_distance=1000,
#                                      max_distance=1000,
#                                      charge_bins=[None, 16 * 2, 16 * 2],
#                                      sensor_sizes=None,
#                                      dut_masks=dut_masks,
#                                      efficiency_range=[None, [[750,1000], [370,750]], None])
