
import numpy as np
import tables as tb

from beam_telescope_analysis.tools import geometry_utils
from beam_telescope_analysis.result_analysis import histogram_track_angle
from beam_telescope_analysis.cpp import data_struct
from beam_telescope_analysis.tools.plot_utils import plot_2d_pixel_hist
from beam_telescope_analysis.telescope.dut import Dut
from beam_telescope_analysis.tools import analysis_utils
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages

def local_to_global_position(x, y, translation_x=None, translation_y=None, translation_z=None, rotation_alpha=None, rotation_beta=None, rotation_gamma=None):
    if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
    z = np.zeros_like(x)
    # check for valid z coordinates
    if translation_x is None and translation_y is None and translation_z is None and rotation_alpha is None and rotation_beta is None and rotation_gamma is None and not np.allclose(np.nan_to_num(z), 0.0):
        raise RuntimeError('The local z positions contain values z!=0.')
    # apply DUT alignment
    transformation_matrix = geometry_utils.local_to_global_transformation_matrix(
        x=0. if translation_x is None else float(translation_x),
        y=0. if translation_y is None else float(translation_y),
        z=0. if translation_z is None else float(translation_z),
        alpha=0. if rotation_alpha is None else float(rotation_alpha),
        beta=0. if rotation_beta is None else float(rotation_beta),
        gamma=0. if rotation_gamma is None else float(rotation_gamma))
    return geometry_utils.apply_transformation_matrix(
        x=x,
        y=y,
        z=z,
        transformation_matrix=transformation_matrix)


def plot_cluster_2dhist(cluster_file, hit_file, edge_pixels=True, plot_large_cluster = False, cluster_size_threshold = 10):
    ''' take merged cluster file and plot 2d hist of clusters '''
    with tb.open_file(cluster_file,'r') as cluster_file, tb.open_file(hit_file,'r') as hit_file :
        clusters = cluster_file.root.MergedClusters[:]
        # clusters = clusters[["x_dut_1","y_dut_1", "n_hits_dut_1","event_number"]]
        hits = hit_file.root.ClusterHits[:]
        # clusters = clusters[np.where(clusters==clusters)]
        if edge_pixels:
            xbins = [-20400] + [-19950 + i*250 for i in range(0,79)] + [0] + [200 + i*250 for i in range(1,80)] + [20400]
            # print len(xbins)
            # print xbins
            ybins = [-8400 + i*50 for i in range(0,168)] + [i*50 for i in range(0,168)]
            # print len(ybins)
            # print ybins
            bins = [xbins, ybins]
        else:
            bins = [80,336]
        # hist2d, xedges, yedges = np.histogram2d(clusters["x_dut_0"],clusters['y_dut_0'],range(-20000,20000))
        hist ,_ ,_ ,_ = plt.hist2d(clusters["x_dut_1"],clusters["y_dut_1"], bins =bins, cmap = 'viridis')
        zmin = np.min(hist)
        zmax = np.max(hist)
        bounds = np.linspace(start=zmin, stop=zmax, num=256, endpoint=True)
        plt.colorbar(boundaries=bounds, ticks=np.linspace(start=zmin, stop=zmax, num=9, endpoint=True), fraction=0.04, pad=0.05)
        plt.title("Cluster histogram for plane 1")
        plt.xlabel("x position")
        plt.ylabel("y position")
        # fig = Figure()
        # _ = FigureCanvas(fig)
        # ax = fig.add_subplot(111)
        # plot_2d_pixel_hist(fig, ax, hist2d,plot_range=[-20000,20000])
        plt.show()
        plt.savefig("/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2793/clusters_plane1.pdf")

        if plot_large_cluster:
            clusters = clusters[clusters["n_hits_dut_1"]>cluster_size_threshold]
            events = np.unique(clusters["event_number"])
            # events = np.unique(clusters["event_number"][clusters["n_hits_dut_1"]>30])
            print events
            for i, event in enumerate(events):
                # if i == 15:
                #     break

                print "cluster_size : ", clusters[clusters["event_number"]==event]["n_hits_dut_1"]
                # print hits[np.where(hits["event_number"]==event)][0]["n_cluster"]
                ids = clusters[clusters["event_number"]==event]["cluster_ID_dut_1"]
                print "event_number = ", event
                print "cluster_IDs with n_hits > %s: %s " %(cluster_size_threshold,ids)
                # single_event = clusters[clusters["event_number"]==event][["x_dut_1","y_dut_1"]]
                for cluster in ids:
                    plt.clf()
                    single_cluster = hits[np.where((hits["event_number"]==event) & (hits["cluster_ID"]==cluster))]
                    print "clusterID = ", cluster
                    print "cluster_size = ", single_cluster[0]["cluster_size"]
                    x = single_cluster['column']
                    y = single_cluster['row']
                    bins = [160,336]
                    hist2 ,_ ,_ ,_ = plt.hist2d(x,y, bins =bins, cmap = 'viridis',range = [[1,161],[1,337]])
                    zmin = np.min(hist2)
                    zmax = np.max(hist2)
                    bounds = np.linspace(start=zmin, stop=zmax, num=256, endpoint=True)
                    plt.colorbar(boundaries=bounds, ticks=np.linspace(start=zmin, stop=zmax, num=9, endpoint=True), fraction=0.04, pad=0.05)
                    # plt.savefig("/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2793/single_event%s.pdf" %event)
                    plt.show()


def plot_cluster_2dhist_global(cluster_file, hit_file, plot_large_cluster = False, cluster_size_threshold = 10):
    ''' take merged cluster file and plot 2d hist of clusters '''
    with tb.open_file(cluster_file,'r') as cluster_file_in:
        clusters = cluster_file_in.root.MergedClusters[:]
        # scale_columns = clusters[np.where((clusters["x_dut_1"]<=450) & np.where(clusters["x_dut_1"]>= - 450))]
        # scale_columns[""]
        x0,y0,z0 = local_to_global_position(x = clusters["x_dut_0"], y = clusters["y_dut_0"], translation_x=-168*50 - 20, translation_y= None, translation_z=None, rotation_alpha=0, rotation_beta=np.pi, rotation_gamma=-np.pi/2)
        x1,y1,z1 = local_to_global_position(x = clusters["x_dut_1"], y = clusters["y_dut_1"], translation_x=+168*50 + 20, translation_y= None, translation_z=None, rotation_alpha=0, rotation_beta=None, rotation_gamma=np.pi/2)

        # x0,y0,z0 = local_to_global_position(x = clusters["x_dut_2"], y = clusters["y_dut_2"], translation_x=0, translation_y= -168*50 - 20, translation_z=None, rotation_alpha=0, rotation_beta=np.pi, rotation_gamma=None)
        # x1,y1,z1 = local_to_global_position(x = clusters["x_dut_3"], y = clusters["y_dut_3"], translation_x=0, translation_y= +168*50, translation_z=None, rotation_alpha=0, rotation_beta=None, rotation_gamma=np.pi)

        xbins = [-20400] + [-19950 + i*250 for i in range(0,79)] + [0] + [200 + i*250 for i in range(1,80)] + [20400]
        # xbins = [-20400] + [-19950 + i*250 for i in range(0,79)] + [-125] + [0] + [125] + [200 + i*250 for i in range(1,80)] + [20400]
        ybins = [-16800 + i*50 for i in range(0,336)] + [i*50 for i in range(0,336)]

        # ybins = [-16800 + i*250 for i in range(0,136)] #+ [i*250 for i in range(0,80)]
        # ybins = [-20400 + i*250 for i in range(0,178)] #+ [i*250 for i in range(0,80)]
        # bins = [xbins, ybins]
        bins = [ybins, xbins]
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        x = np.concatenate((x0,x1))
        y = np.concatenate((y0,y1))

        hist, xbins, ybins = np.histogram2d(x, y, bins = bins)
        #scale larger pixel occupancy
        hist[:,79:81] = hist[:,79:81] * 250/450.
        # hist[79:81,:] = hist[79:81,:] * 250/450.
        # hist ,_ ,_ ,_ = plt.hist2d(x,y, bins = bins, cmap = 'viridis', cmin=1, normed = False)
        zmin = np.min(hist[hist!=0])
        zmax = np.max(hist)
        cmap = cm.get_cmap("viridis")
        cmap.set_bad('w')
        # im = ax.hist2d(x, y, bins = bins)
        im = ax.imshow(np.ma.masked_where(hist <=0, hist).T, interpolation='none', origin='lower', aspect="auto", extent = [xbins.min(),xbins.max(),ybins.max(),ybins.min()], cmap=cmap, clim=(zmin, zmax))
        bounds = np.linspace(start=zmin, stop=zmax, num=256, endpoint=True)
        fig.colorbar(im, boundaries=bounds, ticks=np.linspace(start=zmin, stop=zmax, num=9, endpoint=True), fraction=0.04, pad=0.05)
        ax.set_title("Cluster seed histogram CH1R6 - modules 0 and 1")
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")

        # plot_2d_pixel_hist(fig, ax, hist2d,plot_range=[-20000,20000])
        fig.tight_layout()
        fig.savefig(cluster_file.replace("Merged.h5", "clusters_plane01.pdf"))

        if plot_large_cluster:
            with tb.open_file(hit_file,'r') as hit_file:
                hits = hit_file.root.ClusterHits[:]
                clusters = clusters[clusters["n_hits_dut_1"]>cluster_size_threshold]
                events = np.unique(clusters["event_number"])
                # events = np.unique(clusters["event_number"][clusters["n_hits_dut_1"]>30])
                print events
                for i, event in enumerate(events):
                    print "cluster_size : ", clusters[clusters["event_number"]==event]["n_hits_dut_1"]
                    # print hits[np.where(hits["event_number"]==event)][0]["n_cluster"]
                    ids = clusters[clusters["event_number"]==event]["cluster_ID_dut_1"]
                    print "event_number = ", event
                    print "cluster_IDs with n_hits > %s: %s " %(cluster_size_threshold,ids)
                    # single_event = clusters[clusters["event_number"]==event][["x_dut_1","y_dut_1"]]
                    for cluster in ids:
                        plt.clf()
                        single_cluster = hits[np.where((hits["event_number"]==event) & (hits["cluster_ID"]==cluster))]
                        print "clusterID = ", cluster
                        print "cluster_size = ", single_cluster[0]["cluster_size"]
                        x = single_cluster['column']
                        y = single_cluster['row']
                        bins = [160,336]
                        hist2 ,_ ,_ ,_ = plt.hist2d(x,y, bins =bins, cmap = 'viridis',range = [[1,161],[1,337]])
                        zmin = np.min(hist2)
                        zmax = np.max(hist2)
                        bounds = np.linspace(start=zmin, stop=zmax, num=256, endpoint=True)
                        plt.colorbar(boundaries=bounds, ticks=np.linspace(start=zmin, stop=zmax, num=9, endpoint=True), fraction=0.04, pad=0.05)
                        # plt.savefig("/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2793/single_event%s.pdf" %event)
                        plt.show()


def plot_timestamps(merged_file_in):
    with tb.open_file(merged_file_in, "r") as merged_file:
        merged_clusters = merged_file.root.Hits[:]
    events,indices = np.unique(merged_clusters["event_number"],return_index=True)
    merged_clusters = merged_clusters[indices]
    delta_t = np.diff(merged_clusters["trigger_time_stamp"][np.where(merged_clusters["spill"]==0)])

    for spill in range(1,merged_clusters["spill"].max()):
        delta_t = np.concatenate((delta_t, np.diff(merged_clusters["trigger_time_stamp"][np.where(merged_clusters["spill"]==spill)])))

    # delta_t = delta_t * 25e-9 * 1e6
    print np.median(delta_t)
    plt.clf()
    # plt.hist(delta_t * 25e-9 * 1e3, bins = np.arange(0,3.5,3.5/100),edgecolor='black', linewidth=0.3) #
    plt.hist(delta_t * 25e-9 * 1e6, bins = np.arange(0,105,1),edgecolor='black', linewidth=0.3)
    plt.grid()
    plt.title("trigger time distance")
    plt.ylabel("#")
    # plt.xlabel(r"$\Delta$ t in ms")
    plt.xlabel(r"$\Delta$ t in $\mu$s")
    # plt.yscale('log')
    # plt.show()
    plt.savefig(merged_file_in.replace(".h5","_delta_t_us.pdf"))

def plot_pos_vs_time(merged_file_in):
    # with tb.open_file(merged_file_in, "r") as merged_file:
    #     merged_clusters = merged_file.root.Hits[:]
    # events,indices = np.unique(merged_clusters["event_number"],return_index=True)
    # merged_clusters = merged_clusters[indices]
    # spill_data = merged_clusters[np.where(merged_clusters["spill"]==8)]
    # spill_data["trigger_time_stamp"] = spill_data["trigger_time_stamp"] * 25e-9  *1e3
    #
    # tbins = np.linspace(spill_data["trigger_time_stamp"].min(), spill_data["trigger_time_stamp"].max(), 1000)
    # xbin = [-20400] + [-19950 + i*250 for i in range(0,79)] + [0] + [200 + i*250 for i in range(1,80)] + [20400]
    # ybin = [-8400 + i*50 for i in range(0,168)] + [i*50 for i in range(0,168)]
    # fig = Figure()
    # _ = FigureCanvas(fig)
    # ax = fig.add_subplot(111)
    # hist, xbins, ybins = np.histogram2d(spill_data["trigger_time_stamp"], spill_data['y_dut_1'], bins = [tbins,ybin])
    # zmin = np.min(hist)
    # zmax = np.max(hist)
    # cmap = cm.get_cmap("viridis")
    # cmap.set_bad('w')
    # # ax.plot(spill_data["trigger_time_stamp"],spill_data['y_dut_1'], linestyle = "None", marker = "o", markersize = 0.1)
    # im = ax.imshow(np.ma.masked_where(hist <= 1., hist).T, interpolation='none', origin='lower', aspect="auto", extent = [xbins.min(),xbins.max(),ybins.max(),ybins.min()], cmap=cmap, clim=(zmin, zmax))
    # bounds = np.linspace(start=zmin, stop=zmax, num=256, endpoint=True)
    # fig.colorbar(im, boundaries=bounds, ticks=np.linspace(start=zmin, stop=zmax, num=9, endpoint=True), fraction=0.04, pad=0.05)
    # ax.set_title("Cluster seed histogram CH1R6 - modules 8 and 9")
    # ax.set_xlabel("x position")
    # ax.set_ylabel("y position")
    # fig.tight_layout()
    # # fig.show()
    # fig.savefig(merged_file_in.replace("Merged_spills.h5", "x_vs_time.pdf"))

    with tb.open_file(merged_file_in,'r') as cluster_file_in:
        clusters1 = cluster_file_in.root.MergedClusters[:]
        t_slices = np.linspace(5e5,clusters1["trigger_time_stamp"].max(),15)
        meansx = []
        meansy = []
        xerr = []
        yerr = []
        with PdfPages(merged_file_in.replace(".h5","_cluster_pos_vs_time.pdf")) as output_pdf:
            for i,time in enumerate(t_slices[:-1]):
                # print t_slices[i]
                # print t_slices[i+1]
                clusters = clusters1[np.where((t_slices[i] <= clusters1['trigger_time_stamp']) & (clusters1['trigger_time_stamp'] <= t_slices[i+1]))]
                # print clusters.shape
                x0,y0,z0 = local_to_global_position(x = clusters["x_dut_0"], y = clusters["y_dut_0"], translation_x=-168*50 - 20, translation_y= None, translation_z=None, rotation_alpha=0, rotation_beta=np.pi, rotation_gamma=-np.pi/2)
                x1,y1,z1 = local_to_global_position(x = clusters["x_dut_1"], y = clusters["y_dut_1"], translation_x=+168*50 + 20, translation_y= None, translation_z=None, rotation_alpha=0, rotation_beta=None, rotation_gamma=np.pi/2)

                # x0,y0,z0 = local_to_global_position(x = clusters["x_dut_10"], y = clusters["y_dut_10"], translation_x=0, translation_y= -168*50 - 20, translation_z=None, rotation_alpha=0, rotation_beta=np.pi, rotation_gamma=None)
                # x1,y1,z1 = local_to_global_position(x = clusters["x_dut_11"], y = clusters["y_dut_11"], translation_x=0, translation_y= +168*50, translation_z=None, rotation_alpha=0, rotation_beta=None, rotation_gamma=np.pi)

                xbins = [-20400] + [-19950 + i*250 for i in range(0,79)] + [0] + [200 + i*250 for i in range(1,80)] + [20400]
                # print len(xbins)
                # print xbins
                ybins = [-16800 + i*50 for i in range(0,336)] + [i*50 for i in range(0,336)]
                # print len(ybins)
                # print ybins
                bins = [ybins, xbins]
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                x = np.concatenate((x0,x1))
                y = np.concatenate((y0,y1))
                x = x[~np.isnan(x)]
                y = y[~np.isnan(y)]
                # mean_x = np.nanmean(np.ma.masked_where(x <= 1, x))
                # mean_y = np.nanmean(np.ma.masked_where(y <= 1, y))
                # xfit, xcov = curve_fit(analysis_utils.gauss, x, 4500, p0=[0,0,0])
                # yfit, ycov = curve_fit(analysis_utils.gauss, y, 0, p0=[0,0,0])
                # meansx.append(np.mean(x[np.where((x>=-8000) & (x<8000))]))
                # meansy.append(np.mean(y[np.where((y>=-10000) & (y<10000))]))
                meansx.append(np.mean(x))
                meansy.append(np.mean(y))
                # xerr.append(np.std(x)/np.sqrt(x[np.where((x>=-8000) & (x<8000))].shape[0]))
                # yerr.append(np.std(y)/np.sqrt(y[np.where((y>=-10000) & (y<10000))].shape[0]))
                xerr.append(np.std(x))
                yerr.append(np.std(y))
                hist, xbins, ybins = np.histogram2d(x, y, bins = bins)
                # hist ,_ ,_ ,_ = plt.hist2d(x,y, bins = bins, cmap = 'viridis', cmin=1, normed = False)
                hist[:,79:81] = hist[:,79:81] * 250/450.
                zmin = np.min(hist[hist >= 1])
                zmax = np.max(hist)
                cmap = cm.get_cmap("viridis")
                cmap.set_bad('w')
                # im = ax.hist(y, range = [-10000,10000])
                im = ax.imshow(np.ma.masked_where(hist <= 1, hist).T, interpolation='none', origin='lower', aspect="auto", extent = [xbins.min(),xbins.max(),ybins.max(),ybins.min()], cmap=cmap, clim=(zmin, zmax))
                bounds = np.linspace(start=zmin, stop=zmax, num=256, endpoint=True)
                fig.colorbar(im, boundaries=bounds, ticks=np.linspace(start=zmin, stop=zmax, num=9, endpoint=True), fraction=0.04, pad=0.05)
                ax.set_title("Beam spot run 2793")
                ax.set_xlabel("x position $\mu$m")
                ax.set_ylabel("y position $\mu$m")

                # plot_2d_pixel_hist(fig, ax, hist2d,plot_range=[-20000,20000])
                fig.tight_layout()
                output_pdf.savefig(fig)
                # print "saved pdf at %s" % merged_file_in.replace(".h5","cluster_pos_vs_time.pdf")

            cmap = cm.get_cmap("viridis")

            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            # ax.plot(np.array(meansx), np.array(meansy))
            t_slices = t_slices * 25e-9 * 1e3
            cmin = t_slices.min()
            cmax = t_slices.max()
            im = ax.scatter(np.array(meansx), np.array(meansy), c =t_slices[:-1])
            for i, _ in enumerate(t_slices[:-1]):
                color = cmap(1.0*i/len(t_slices[:-1]))
                ax.errorbar(np.array(meansx[i]), np.array(meansy[i]), xerr = xerr[i]/np.sqrt(14), yerr = yerr[i]/np.sqrt(14), capsize = 2, linestyle="None" , c= color)
            # ax.invert_xaxis()
            # ax.invert_yaxis()
            bounds = np.linspace(start=cmin, stop=cmax, num=256, endpoint=True)
            cbar = fig.colorbar(im, boundaries=bounds, ticks=np.linspace(start=cmin, stop=cmax, num=9, endpoint=True), fraction=0.04, pad=0.05)
            cbar.set_label("time in ms")
            ax.grid()
            ax.set_title("mean beam position over time run 2793")
            ax.set_xlabel("x position $\mu$m")
            ax.set_ylabel("y position $\mu$m")
            output_pdf.savefig(fig)

            # fig = Figure()
            # _ = FigureCanvas(fig)
            # ax = fig.add_subplot(111)
            # r = np.sqrt(np.array(meansx)**2 + np.array(meansy)**2)
            # ax.plot(t_slices[:-1], r)
            # ax.grid()
            # ax.set_title("mean beam position vs time")
            # ax.set_xlabel("x position $\mu$m")
            # ax.set_ylabel("y position $\mu$m")
            # output_pdf.savefig(fig)

def transform_to_emulsion_frame(clusters,duts=(0,1),emulsion_speed=2.6, y_offset=10000):
    '''
    input
    ------------------
        clusters: cluster table
        duts: dut planes to plot
        emulsion_speed: movement speed in x of emulsion in cm/s
        y_offset: stepsize of emulsion between spills in y, in micrometer

    output
    ------------------
    x and y numpy arrays with transformed coordinates in emulsion rest frame
    '''
    clusters["x_dut_0"],clusters["y_dut_0"],z0 = local_to_global_position(x = clusters["x_dut_0"], y = clusters["y_dut_0"], translation_x=336/2*50, translation_y= 80*250 , translation_z=None, rotation_alpha=0, rotation_beta=np.pi, rotation_gamma=-np.pi/2)
    clusters["x_dut_1"],clusters["y_dut_1"],z1 = local_to_global_position(x = clusters["x_dut_1"], y = clusters["y_dut_1"], translation_x=336*1.5*50, translation_y= 80*250, translation_z=None, rotation_alpha=0, rotation_beta=None, rotation_gamma=np.pi/2)

    clustersx1 = clusters["x_dut_1"] + (-1)**(clusters["spill"]%2) * 10000*emulsion_speed * clusters["trigger_time_stamp"]*25e-9
    clustersx0 = clusters["x_dut_0"] + (-1)**(clusters["spill"]%2) * 10000*emulsion_speed * clusters["trigger_time_stamp"]*25e-9
    clustersy0 = clusters["y_dut_0"] + clusters["spill"]*y_offset
    clustersy1 = clusters["y_dut_1"] + clusters["spill"]*y_offset

    x = np.concatenate((clustersx0,clustersx1))
    y = np.concatenate((clustersy0,clustersy1))

    x = x[~np.isnan(x)]
    x = x - 24000
    x[x<0] += 124000
    y = y[~np.isnan(y)]
    return x,y


def plot_spill(output_pdf,x,y,bins=[360,160],spill=None):
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    hist, xbins, ybins = np.histogram2d(x, y, bins = bins)
    zmin = np.min(hist[hist >= 1])
    zmax = np.max(hist)
    cmap = cm.get_cmap("viridis")
    cmap.set_bad('w')

    im = ax.imshow(hist.T, interpolation='none', origin='lower', aspect="auto", extent = [xbins.min(),xbins.max(),ybins.min(),ybins.max()], cmap=cmap, clim=(zmin, zmax))
    bounds = np.linspace(start=zmin, stop=zmax, num=256, endpoint=True)
    fig.colorbar(im, boundaries=bounds, ticks=np.linspace(start=zmin, stop=zmax, num=9, endpoint=True), fraction=0.04, pad=0.05)
    ax.grid()
    if spill:
        ax.set_title("Spill %s DUT 0/1 in emulsion rest frame" %spill)
        ax.set_ylim(10000+10000*spill,220000)
    else:
        ax.set_title("DUT 0/1 in emulsion rest frame")
    ax.set_ylabel("y position [$\mu$m]")
    ax.set_xlabel("x position [$\mu$m]")
    output_pdf.savefig(fig)


def transform_2d_hist(merged_file_in, spills=None, duts=(0,1)):
    ''' transform clusters in rest frame of emulsion, movement speed was 2.6cm/s
    input
    ---------------
        merged_file_in: string pointing to cluster file with merged clusters
        spills: tuple or list with first and last spill for single spill plotting
        duts :tuple of duts to plot
    '''
    with tb.open_file(merged_file_in,'r') as cluster_file_in:
        clustertable = cluster_file_in.root.MergedClusters[:]
        output_pdf = PdfPages(merged_file_in.replace(".h5","_emulsion_rest_frame_per_spill.pdf"))
        if spills:
            for spill in range(spills[0],spills[1]+1):
                print "plotting spill %s" %spill
                clusters = clustertable[np.where(clustertable["spill"]==spill)]
                x,y = transform_to_emulsion_frame(clusters,duts)
                plot_spill(output_pdf,x,y,spill=spill)
        ''' now plot all spills'''
        print "plotting all spills"
        clusters = clustertable
        x,y = transform_to_emulsion_frame(clusters,duts)
        plot_spill(output_pdf,x,y,bins=[360,160*2])
        output_pdf.close()


if __name__ == '__main__':

    # plot_cluster_hist(cluster_file = "/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2793/Merged.h5",
    #                     hit_file = "/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_2793/pyBARrun_376_plane_0_DC_module_1_local_corr_evts_clustered.h5",
    #                     cluster_size_threshold = 10)
    transform_2d_hist("/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2781/Merged_spills.h5", spills=[8,18])
    plot_cluster_2dhist_global(cluster_file = "/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2781/Merged.h5",
                            hit_file = None)
    # plot_timestamps("/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2793/Merged_spills.h5")
    # plot_pos_vs_time("/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2793/Merged.h5")
