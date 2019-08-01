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

runs = {
        # 2746 : ("336", "230", "164"),
        # 2750 : ("340", "234", "168"),
        # 2763 : ("348", "242" ,"176"),
        # 2776 : ("360", "254" ,"188"),
        # 2777 : ("361", "255", "189"),
        # 2781 : ("364", "258", "192"),
        # 2783 : ("366", "260", "194"),
        # 2785 : ("368", "262", "196"),
        # 2788 : ("371", "265", "199"),
        # 2789 : ("372", "266", "200"),
        # 2790 : ("373", "267", "201"),
        # 2791 : ("374", "268", "202"),
        # 2792 : ("375", "269", "203"),
        # 2793 : ("376", "270", "204"),
        # 2794 : ("377", "271", "205"),
        # 2795 : ("378", "272", "206"),
        # 2796 : ("379", "273", "207"),
        # 2797 : ("380", "274", "208"),
        # 2798 : ("381", "275", "209")
        # 2799 : ("382", "276", "210"),
        #''' alignment runs '''
        2815 : ("394", "288", "222"),
        2817 : ("395", "289", "223"),
        2818 : ("396", "290", "224"),
        2825 : ("402", "296", "230"),
        2829 : ("405", "299", "233"),
        2830 : ("406", "300", "234"),
        2836 : ("412", "306", "244"),
        2837 : ("413", "307", "241"),

        # 2853 : ("429", "323", "257"),
        # 2854 : ("430", "324", "258"),
        # 2863 : ("439", "333", "267"),
        }

raw_data_dirs = ['/media/niko/data/SHiP/charm_exp_2018/data/part_0x0800',
                 '/media/niko/data/SHiP/charm_exp_2018/data/part_0x0801',
                 '/media/niko/data/SHiP/charm_exp_2018/data/part_0x0802'
                ]

output_dir = '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements'

# pybar_runs =  '376' , '270', '204' #'394', '288', '222'

# run_number = 2793
for key in runs.keys():
    run_number = key
    pybar_runs = runs[key]
    logger.info("============================== converting run %s ==============================" % run_number)
    # logger.info("============================= using pybar files  ============================", pybar_runs )

    data_files_folder = os.path.join(output_dir, 'run_%s'% run_number)
    output_folder = os.path.join(output_dir, 'output_folder_run_%s' % run_number)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(data_files_folder):
        os.makedirs(data_files_folder)

    hit_files = []
    for i, folder in enumerate(raw_data_dirs):
        first_plane, second_plane = pybar_ship_converter.get_plane_files(pyBARrun = pybar_runs[i], subpartition_dir = folder)
        for data_file in first_plane:
            pybar_ship_converter.process_dut(data_file, trigger_data_format=1, do_corrections=False, empty_events=True)
        for data_file in second_plane:
            pybar_ship_converter.process_dut(data_file, trigger_data_format=1, do_corrections=False, empty_events=True)
        pybar_ship_converter.merge_dc_module_local(output_dir = data_files_folder, plane_files = first_plane, pyBARrun = pybar_runs[i], plane_number = i*2, output_file_list = hit_files)
        pybar_ship_converter.merge_dc_module_local(output_dir = data_files_folder, plane_files = second_plane, pyBARrun = pybar_runs[i], plane_number = (i*2)+1 , output_file_list = hit_files)

    hit_tables_in = open_files(hit_files)
    new_tables = merge_hits(hit_tables_in=hit_tables_in)
    hit_files = save_new_tables(hit_files, new_tables)

    # hit_files= ['/media/niko/big_data/run_%s/pyBARrun_%s_plane_0_DC_module_0_local.h5' % (run_number, pybar_runs[0]),
    #             '/media/niko/big_data/run_%s/pyBARrun_%s_plane_0_DC_module_1_local.h5' % (run_number, pybar_runs[0]),
    #             '/media/niko/big_data/run_%s/pyBARrun_%s_plane_1_DC_module_0_local.h5' % (run_number, pybar_runs[0]),
    #             '/media/niko/big_data/run_%s/pyBARrun_%s_plane_1_DC_module_1_local.h5' % (run_number, pybar_runs[0]),
    #             '/media/niko/big_data/run_%s/pyBARrun_%s_plane_2_DC_module_0_local.h5' % (run_number, pybar_runs[1]),
    #             '/media/niko/big_data/run_%s/pyBARrun_%s_plane_2_DC_module_1_local.h5' % (run_number, pybar_runs[1]),
    #             '/media/niko/big_data/run_%s/pyBARrun_%s_plane_3_DC_module_0_local.h5' % (run_number, pybar_runs[1]),
    #             '/media/niko/big_data/run_%s/pyBARrun_%s_plane_3_DC_module_1_local.h5' % (run_number, pybar_runs[1]),
    #             '/media/niko/big_data/run_%s/pyBARrun_%s_plane_4_DC_module_0_local.h5' % (run_number, pybar_runs[2]),
    #             '/media/niko/big_data/run_%s/pyBARrun_%s_plane_4_DC_module_1_local.h5' % (run_number, pybar_runs[2]),
    #             '/media/niko/big_data/run_%s/pyBARrun_%s_plane_5_DC_module_0_local.h5' % (run_number, pybar_runs[2]),
    #             '/media/niko/big_data/run_%s/pyBARrun_%s_plane_5_DC_module_1_local.h5' % (run_number, pybar_runs[2]),
    #             ]
    # #
    # hit_files= ['/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_0_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[0]),
    #             '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_0_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[0]),
    #             '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_1_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[0]),
    #             '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_1_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[0]),
    #             '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_2_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[1]),
    #             '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_2_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[1]),
    #             '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_3_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[1]),
    #             '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_3_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[1]),
    #             '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_4_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[2]),
    #             '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_4_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[2]),
    #             '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_5_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[2]),
    #             '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_5_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[2]),
    #             ]

    z_positions = [0., 6400., 26800., 33000., 52200., 58400., 79300., 85900., 105800., 112200.,  132200., 138600. ]# (in um)

    shift = 0 #(336/2. * 50.)

    ''' rotations: 1. alpha (around x) 2. beta (around y) 3. gamma (around z)'''

    telescope = Telescope()

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=0, translation_x= -shift, translation_y= 0, translation_z=z_positions[0],
                      rotation_alpha=0.0, rotation_beta= 0.0, rotation_gamma= 0.0, name="Telescope 0")

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=1, translation_x= shift, translation_y= 0, translation_z=z_positions[1],
                      rotation_alpha=0, rotation_beta= 0, rotation_gamma= 0.0, name="Telescope 1")

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=2, translation_x=0, translation_y= -shift, translation_z=z_positions[2],
                      rotation_alpha=0, rotation_beta=0.0 , rotation_gamma=0, name="Telescope 2")

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=3, translation_x=0, translation_y= shift, translation_z=z_positions[3],
                      rotation_alpha=0, rotation_beta= 0 , rotation_gamma= 0.0, name="Telescope 3")

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=4, translation_x= -shift, translation_y= 0, translation_z=z_positions[4],
                      rotation_alpha=0.0, rotation_beta=0.0, rotation_gamma= 0.0, name="Telescope 4")

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=5, translation_x= shift, translation_y= 0, translation_z=z_positions[5],
                      rotation_alpha= 0.0, rotation_beta=0.0, rotation_gamma= 0.0, name="Telescope 5")

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=6, translation_x=0, translation_y= -shift, translation_z=z_positions[6],
                      rotation_alpha=0.0, rotation_beta=0.0, rotation_gamma=0.0, name="Telescope 6")

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=7, translation_x=0, translation_y= shift, translation_z=z_positions[7],
                      rotation_alpha=0.0, rotation_beta=0.0, rotation_gamma=0.0, name="Telescope 7")

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=8, translation_x= -shift, translation_y= 0, translation_z=z_positions[8],
                      rotation_alpha=0.0, rotation_beta=0.0, rotation_gamma=0.0, name="Telescope 8")

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=9, translation_x= shift, translation_y= 0, translation_z=z_positions[9],
                      rotation_alpha=0.0, rotation_beta=0.0, rotation_gamma= 0.0, name="Telescope 9")

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=10, translation_x=0, translation_y= -shift, translation_z=z_positions[10],
                      rotation_alpha=0.0, rotation_beta=0.0, rotation_gamma=0.0, name="Telescope 10")

    telescope.add_dut(dut_type="FEI4DCModule", dut_id=11, translation_x=0, translation_y= shift, translation_z=z_positions[11],
                      rotation_alpha=0.0, rotation_beta=0.0, rotation_gamma=0.0, name="Telescope 11")

    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=0, translation_x= -shift, translation_y= 0, translation_z=z_positions[0],
    #                   rotation_alpha=np.pi, rotation_beta= 0, rotation_gamma= 0.5 * np.pi, name="Telescope 0")
    #
    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=1, translation_x= shift, translation_y= 0, translation_z=z_positions[1],
    #                   rotation_alpha=0, rotation_beta= 0, rotation_gamma= -0.5 * np.pi, name="Telescope 1")
    #
    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=2, translation_x=0, translation_y= -shift, translation_z=z_positions[2],
    #                   rotation_alpha=0, rotation_beta=np.pi , rotation_gamma=0, name="Telescope 2")
    #
    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=3, translation_x=0, translation_y= shift, translation_z=z_positions[3],
    #                   rotation_alpha=0, rotation_beta= 0 , rotation_gamma= np.pi, name="Telescope 3")
    #
    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=4, translation_x= -shift, translation_y= 0, translation_z=z_positions[4],
    #                   rotation_alpha=np.pi, rotation_beta=0, rotation_gamma= 0.5 * np.pi, name="Telescope 4")
    #
    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=5, translation_x= shift, translation_y= 0, translation_z=z_positions[5],
    #                   rotation_alpha= 0, rotation_beta=0, rotation_gamma= -0.5 * np.pi, name="Telescope 5")
    #
    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=6, translation_x=0, translation_y= -shift, translation_z=z_positions[6],
    #                   rotation_alpha=0, rotation_beta=np.pi, rotation_gamma=0, name="Telescope 6")
    #
    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=7, translation_x=0, translation_y= shift, translation_z=z_positions[7],
    #                   rotation_alpha=0, rotation_beta=0, rotation_gamma=np.pi, name="Telescope 7")
    #
    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=8, translation_x= -shift, translation_y= 0, translation_z=z_positions[8],
    #                   rotation_alpha=np.pi, rotation_beta=0, rotation_gamma=0.5 * np.pi, name="Telescope 8")
    #
    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=9, translation_x= shift, translation_y= 0, translation_z=z_positions[9],
    #                   rotation_alpha=0, rotation_beta=0, rotation_gamma= -0.5 * np.pi, name="Telescope 9")
    #
    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=10, translation_x=0, translation_y= -shift, translation_z=z_positions[10],
    #                   rotation_alpha=0, rotation_beta=np.pi, rotation_gamma=0, name="Telescope 10")
    #
    # telescope.add_dut(dut_type="FEI4DCModule", dut_id=11, translation_x=0, translation_y= shift, translation_z=z_positions[11],
    #                   rotation_alpha=0, rotation_beta=0, rotation_gamma=np.pi, name="Telescope 11")

    telescope.save_configuration(os.path.join(output_folder,'run%d_telescope.yaml' % run_number))
    # telescope.save_configuration('run%d_telescope_local_coordinates.yaml' % run_number)

    pixel_size = [(250., 50.)]*12
    n_pixels = [(160, 336)]*12

    # mixed coordinates
    # check_files = hit_analysis.check(
    #     telescope_configuration='run%d_telescope.yaml' % run_number,
    #     input_hit_files=hit_files)

    thresholds = [10,] * 12
    pixel_mask_names = ["NoisyPixelMask"] * len(thresholds)
    mask_files = hit_analysis.mask(
        telescope_configuration=os.path.join(output_folder,'run%d_telescope.yaml' % run_number),
        input_hit_files=hit_files,
        pixel_mask_names=pixel_mask_names,
        thresholds=thresholds)

    # mask_files = [None]*12 #[(os.path.splitext(hit_file)[0] + '_mask.h5') for hit_file in hit_files]
    #
    min_hit_charges = [0,] * 12
    max_hit_charges = [13,] * 12
    column_cluster_distances = [1,]*12
    row_cluster_distances = [1,] * 12
    frame_cluster_distances = [1,] * 12
    cluster_files = hit_analysis.cluster(
        telescope_configuration=os.path.join(output_folder,'run%d_telescope.yaml' % run_number),
        select_duts=None,
        input_hit_files=hit_files,
        input_mask_files= mask_files, #[None]*12, #[None if val else mask_files[i] for i, val in enumerate(use_positions)],
        use_positions=[False,]*12,
        min_hit_charges=min_hit_charges,
        max_hit_charges=max_hit_charges,
        column_cluster_distances=column_cluster_distances,
        row_cluster_distances=row_cluster_distances,
        frame_cluster_distances=frame_cluster_distances)

    # cluster_files = [(os.path.splitext(hit_file)[0] + '_clustered.h5') for hit_file in hit_files]
    # TODO: correlation fails with DC module pixel sizes
    # hit_analysis.correlate(
    #     telescope_configuration=os.path.join(output_folder,'run%d_telescope.yaml' % run_number),
    #     input_files=cluster_files,
    #     output_correlation_file=os.path.join(output_folder, 'Correlation.h5'),
    #     resolution=(250., 250.),
    #     select_reference_duts=[1],
    #     chunk_size = 100000)

    hit_analysis.merge_cluster_data(
        telescope_configuration=os.path.join(output_folder,'run%d_telescope.yaml' % run_number),
        input_cluster_files=cluster_files,
        output_merged_file=os.path.join(output_folder, 'Merged.h5'))
