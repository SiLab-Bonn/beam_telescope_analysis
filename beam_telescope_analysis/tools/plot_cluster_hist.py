
import numpy as np
import tables as tb
import logging

from beam_telescope_analysis.tools import geometry_utils
from beam_telescope_analysis.result_analysis import histogram_track_angle
from beam_telescope_analysis.cpp import data_struct
from beam_telescope_analysis.tools.plot_utils import plot_2d_pixel_hist
from beam_telescope_analysis.telescope.dut import Dut
from beam_telescope_analysis.tools import analysis_utils
from scipy.optimize import curve_fit
from scipy.special import factorial, wofz
import pylandau
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib import colors, cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')

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
        merged_clusters = merged_file.root.MergedClusters[:]
    events,indices = np.unique(merged_clusters["event_number"],return_index=True)
    merged_clusters = merged_clusters[indices]
    delta_t = np.diff(merged_clusters["trigger_time_stamp"][np.where(merged_clusters["spill"]==0)])

    for spill in range(1,merged_clusters["spill"].max()):
        delta_t = np.concatenate((delta_t, np.diff(merged_clusters["trigger_time_stamp"][np.where(merged_clusters["spill"]==spill)])))

    delta_t = delta_t * 25e-9 * 1e6
    print delta_t.max()
    # hist, bins = np.histogram(delta_t, bins = 100, range=[0,100] ) #np.logspace(-1,7,10))
    hist, bins = np.histogram(delta_t, bins = np.logspace(-1,7,10))
    plt.clf()
    # plt.hist(delta_t * 25e-9 * 1e3, bins = np.arange(0,3.5,3.5/100),edgecolor='black', linewidth=0.3) #
    plt.bar(bins[:-1],hist, width = np.diff(bins),align = "edge", edgecolor ="k", linewidth=0.3, log=True, label="data")
    # plt.hist(delta_t * 25e-9 * 1e6, bins = np.arange(0,105,1),edgecolor='black', linewidth=0.3)
    plt.grid()
    plt.title("trigger time distance run 2793")
    plt.ylabel("#")
    # plt.xlabel(r"$\Delta$ t in ms")
    plt.xlabel(r"$\Delta$ t in $\mu$s")
    plt.xscale("log")
    plt.xlim(0.1,1e7)
    # plt.show()
    plt.savefig(merged_file_in.replace("Merged_spills_global.h5","_delta_t_us_log.pdf"))


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

def plot_1d_hist(output_pdf, hist, title, axis, fitfunction,  bins, fit_p0=None, cut_bins=None):

    if cut_bins :
        fit_hist = hist[cut_bins[0]:cut_bins[1]]
        fit_bins = bins[cut_bins[0]:cut_bins[1]]
    else:
        fit_hist = hist
        fit_bins = bins

    if fit_p0 :
        p = fit_p0
    else:
        p = [np.max(fit_hist), np.mean(fit_hist), np.std(fit_hist)]
    if fitfunction is not None:
        fit , pcovs = curve_fit(fitfunction, fit_bins[:-1], fit_hist,p0=p)
        perr = np.sqrt(np.diag(pcovs))
        fit_dummy = np.linspace(fit_bins[0],fit_bins[-1],1000)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.grid()
    if fitfunction is not None:
        ax.plot(fit_dummy, fitfunction(fit_dummy, *fit), color = "crimson", label = "fit")
    ax.bar(bins[:-1], hist, width = np.diff(bins),align = "edge", label="data")
    ax.set_title(title + " - %s entries" %int(hist.sum()))
    ax.set_xlabel("%s position $\mu$m" % axis)
    box = AnchoredText('A = %.f $\pm$ %.f\n$\mu$ = %.f $\pm$ %.f $\mu$m \n$\sigma$ = %.3f $\pm$ %.2f mm' %(fit[0], perr[0], fit[1], perr[1], fit[2]/1000, perr[2]/1000), loc=6)
    ax.add_artist(box)
    ax.legend(loc = "upper left")
    output_pdf.savefig(fig)


def plot_2d_hist(output_pdf, hist, xbins, ybins, title, xlim, ylim, cmap_bad = "w"):

    zmin = np.min(hist)
    zmax = np.max(hist)
    cmap = cm.get_cmap("viridis")
    cmap.set_bad(cmap_bad)
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.ma.masked_where(hist < 1, hist).T, interpolation='none', origin='lower', aspect="auto", extent = [xbins.min(),xbins.max(),ybins.min(),ybins.max()], cmap=cmap, clim=(zmin, zmax))
    bounds = np.linspace(start=zmin, stop=zmax, num=256, endpoint=True)
    fig.colorbar(im, boundaries=bounds, ticks=np.linspace(start=zmin, stop=zmax, num=9, endpoint=True), fraction=0.04, pad=0.05)
    ax.grid()
    if xlim!=None:
        ax.set_xlim(xlim[0],xlim[1])
    if ylim!=None:
        ax.set_ylim(ylim[0],ylim[1])
    ax.set_title(title)
    ax.set_xlabel("x position $\mu$m")
    ax.set_ylabel("y position $\mu$m")
    fig.tight_layout()
    output_pdf.savefig(fig)


def plot_material_dependence(file_name):
    x_mus = [5621,5443,4960,5338,5084]
    x_sigmas = [1252,2239,2458,2532,2745]
    x_errs = [10,30,30,50,50]
    y_mus = [71,379,2057,454,1016]
    y_sigmas = [4287,5636,6125,6098,6320]
    y_errs = [40,50,40,50,40]
    material = [0,1,2,3,9]
    with PdfPages(file_name) as output_pdf:

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # ax.plot(material, x_mus, color = "crimson", label = "mean x")
        # ax.plot(material, y_mus, color = "blue", label = "mean y")
        ax.errorbar(material, x_sigmas, yerr = x_errs, color = "crimson", label = "x width", capsize = 2)
        ax.errorbar(material, y_sigmas, yerr = y_errs, color = "blue", label = "y width", capsize = 2)
        ax.set_title("Material dependence of beam width")
        # box = AnchoredText('A = %.f $\pm$ %.f\n$\mu$ = %.2f $\pm$ %.2f mm\n$\sigma$ = %.2f $\pm$ %.2f mm\na = %.3f $\pm$ %.3f\nb = %.2e $\pm$ %.1e\nc = %.f $\pm$ %.f' %(fit[0], perr[0], fit[1]/1000, perr[1]/1000, fit[2]/1000, perr[2]/1000, fit[3], perr[3], fit[4], perr[4], fit[5], perr[5],), loc=2, prop={'size': 7.5})
        # ax.add_artist(box)
        ax.set_xlim(0,10)
        ax.grid()
        ax.legend(loc = "best", prop={'size': 7.5})
        ax.set_xlabel("material thickness in units of 28 mm")
        ax.set_ylabel("$\mu$m")
        output_pdf.savefig(fig)


def plot_gauss_poly2(output_pdf, hist, title, axis, bins, fit_p0=None, cut_bins=None):

    if cut_bins is None:
        cut_bins = [0,-1]

    fitfunction = gauss_poly2
    fit_hist = hist[cut_bins[0]:cut_bins[1]]
    if fit_p0 is None:
        p = [np.max(hist), np.mean(hist), np.std(hist), 0, 1000, -15000 ]
    else:
        p = fit_p0
    fit , pcovs = curve_fit(fitfunction, bins[cut_bins[0]:cut_bins[1]-1], fit_hist,p0=p)
    perr = np.sqrt(np.diag(pcovs))
    fit_dummy = np.linspace(bins[cut_bins[0]],bins[cut_bins[1]],1000)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.bar(bins[:-1], hist, width = np.diff(bins), align = "edge", label="data") # ec='k', linewidth = 0.05,
    ax.plot(fit_dummy, fitfunction(fit_dummy, *fit), color = "crimson", label = "$A \cdot e^{\\frac{-(x-\mu)^2}{2 \sigma^2}}$\n $ + a \cdot x + b \cdot x^2 + c$")
    ax.set_title(title)
    box = AnchoredText('A = %.f $\pm$ %.f\n$\mu$ = %.2f $\pm$ %.2f mm\n$\sigma$ = %.2f $\pm$ %.2f mm\na = %.3f $\pm$ %.3f\nb = %.2e $\pm$ %.1e\nc = %.f $\pm$ %.f' %(fit[0], perr[0], fit[1]/1000, perr[1]/1000, fit[2]/1000, perr[2]/1000, fit[3], perr[3], fit[4], perr[4], fit[5], perr[5],), loc=2, prop={'size': 7.5})
    ax.add_artist(box)
    ax.grid()
    ax.legend(loc = "upper right", prop={'size': 7.5})
    ax.set_xlabel("%s position $\mu$m" % axis)
    output_pdf.savefig(fig)
    return (fit[1],perr[1], fit[2], perr[2])

def plot_gauss_poly3(output_pdf, hist, title, axis, bins, fit_p0=None, cut_bins=None):

    if cut_bins is None:
        cut_bins = [0,-1]
    fit_hist = hist[cut_bins[0]:cut_bins[1]]
    if fit_p0 is None:
        p = [np.max(hist), np.mean(hist), np.std(hist),0,1000,10,-10000 ]
    else:
        p = fit_p0
    fit_dummy = np.linspace(bins[cut_bins[0]],bins[cut_bins[1]],1000)
    fitfunction = gauss_poly3
    fit , pcovs = curve_fit(fitfunction, bins[cut_bins[0]:cut_bins[1]-1], fit_hist, p0=p)
    perr = np.sqrt(np.diag(pcovs))

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.bar(bins[:-1], hist, width = np.diff(bins), align = "edge")
    ax.plot(fit_dummy, fitfunction(fit_dummy, *fit), color = "crimson", label = "$A \cdot e^{\\frac{-(x-\mu)^2}{2 \sigma^2}}$\n $ + a \cdot x + b \cdot x^2 + c\cdot x^3 + d$")#  label = "$A \cdot e^{\\frac{-(x-\mu)^2}{2 \sigma^2}}$\n $ + a \cdot x + b \cdot x^2 + c \cdot x^3 + d$")
    box = AnchoredText('A = %.f $\pm$ %.f\n$\mu$ = %.2f $\pm$ %.2f mm\n$\sigma$ = %.2f $\pm$ %.2f mm\na = %.3f $\pm$ %.3f\nb = %.2E $\pm$ %.1E\nc = %.2E $\pm$ %.1E\nd =%.f $\pm$ %.f' %(fit[0], perr[0], fit[1]/1000, perr[1]/1000, fit[2]/1000, perr[2]/1000, fit[3], perr[3], fit[4], perr[4], fit[5], perr[5], fit[6], perr[6]), loc="center left", prop={'size': 7.5})
    ax.add_artist(box)
    ax.set_title(title)
    ax.grid()
    ax.legend(loc = "upper left", prop={'size': 7.5})
    ax.set_xlabel("%s position $\mu$m" % axis)
    output_pdf.savefig(fig)
    return (fit[1],perr[1], fit[2], perr[2])


def plot_cauchy_poly2(output_pdf, hist, title, axis, bins, fit_p0=None, cut_bins=None):

    if fit_p0 is None:
        p = [2000,5000,1e8,5000,10,0,5000]
    else:
        p = fit_p0
    if cut_bins is None:
        cut_bins = [0,-1]

    fitfunction = cauchy_poly2
    fit_histx = hist[cut_bins[0]:cut_bins[1]]

    fit , pcovs = curve_fit(fitfunction, bins[cut_bins[0]:cut_bins[1]-1], fit_histx, p0=p)
    perr = np.sqrt(np.diag(pcovs))
    gauss_x = np.linspace(bins[cut_bins[0]],bins[cut_bins[1]],1000)
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.bar(bins[:-1], hist, width = np.diff(bins), align = "edge")
    ax.plot(gauss_x, fitfunction(gauss_x, *fit), color = "crimson", label = "$\\frac{1}{\pi \cdot \gamma \cdot (1+\\frac{x-x0}{\gamma})^2} \cdot a + b + c\cdot x + d\cdot x^2 + e $")#  label = "$A \cdot e^{\\frac{-(x-\mu)^2}{2 \sigma^2}}$\n $ + a \cdot x + b \cdot x^2 + c \cdot x^3 + d$")
    box = AnchoredText('$\gamma$ = %.f $\pm$%.f\nx0 = %.f $\pm$ %.f\na = %.2E $\pm$ %.2E\nb = %.2E $\pm$ %.2E\nc = %.3f $\pm$ %.3f\nd = %.2E $\pm$ %.2E\ne = %.2E $\pm$ %.2E' %(fit[0], perr[0], fit[1], perr[1], fit[2], perr[2], fit[3], perr[3], fit[4], perr[4], fit[5], perr[5], fit[6], perr[6]), loc="center left", prop={'size': 7.5})
    ax.add_artist(box)
    ax.set_title(title)
    ax.grid()
    ax.legend(loc = "upper left", prop={'size': 7.5})
    ax.set_xlabel("%s position $\mu$m" % axis)
    output_pdf.savefig(fig)


def plot_pos_vs_time_per_spill(merged_file_in,no_target=False):
    run_number = int(merged_file_in[merged_file_in.find("run_")+4:merged_file_in.find("run_")+8])
    logging.info("opening file for run %s" % run_number)
    with tb.open_file(merged_file_in,'r') as cluster_file_in:
        clusters = cluster_file_in.root.MergedClusters[:]
        clusters = clusters[["x_dut_1","x_dut_0","y_dut_1", "y_dut_0", "trigger_time_stamp", "spill"]]
        spillx = []
        spilly = []
        spillxerr = []
        spillyerr = []
        used_spills = []
        xcorrs = []
        ycorrs = []

        nspills = clusters["spill"].max()
        logging.info("found %s spills" %nspills)
        if no_target ==True:
            out_file_path = merged_file_in.replace("Merged_spills_global.h5","run%s_no_target_cluster_pos_per_spill.pdf" %run_number)
        else:
            out_file_path = merged_file_in.replace("Merged_spills_global.h5","run%s_with_target_cluster_pos_per_spill.pdf" %run_number)
        with PdfPages(out_file_path) as output_pdf:
            if no_target==True:
                if run_number == 2781:
                    nspills = 9
                if run_number == 2785:
                    nspills= nspills
                if run_number == 2793:
                    nspills = 8
                if run_number == 2798:
                    nspills = nspills
                clusters = clusters[clusters["spill"]<nspills]
                start = 0
            else:
                if run_number == 2781:
                    start = 9
                if run_number == 2785:
                    start = 0
                if run_number == 2793:
                    start = 8
                if run_number == 2798:
                    start = 0
            x0 = clusters["x_dut_0"]
            y0 = clusters["y_dut_0"]
            x1 = clusters["x_dut_1"]
            y1 = clusters["y_dut_1"]

            x = np.concatenate((x0,x1))
            y = np.concatenate((y0,y1))
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]

            #weighting larger pixels acoording to their size (250/450 = 0.56 , 250/500 = 0.5)
            weights_all_spills = np.ones_like(y)
            weights_all_spills[np.logical_and(y>-451, y<451)] = 0.56
            weights_all_spills[y<-19950] = 0.5
            weights_all_spills[y>19950] = 0.5

            histbinsy = np.concatenate((np.array([-20450]),np.arange(-19950,-200,250),np.array([0]),np.arange(450,20200,250),np.array([20450])))
            histy, binsy = np.histogram(y, bins = histbinsy, weights = weights_all_spills)
            histx, binsx = np.histogram(x, bins = 336)

            if no_target:
                plot_1d_hist(output_pdf, hist = histx, title="Clusters in x - run %s" %run_number, axis= "x", fitfunction = gauss, fit_p0=None, bins = binsx, cut_bins=None)
                plot_1d_hist(output_pdf, hist = histy, title="Clusters in y - run %s" %run_number, axis= "y" , fitfunction = gauss, fit_p0=None, bins = binsy, cut_bins=None)
            else:
                ''' fit x cluster histogram '''
                cut_bins = [50,-20]
                fit_histx = histx[cut_bins[0]:cut_bins[1]]
                plot_cauchy_poly2(output_pdf=output_pdf, hist=histx, title = "Histogram of clusters in x run %s" %run_number, axis = "x", bins = binsx, fit_p0=None, cut_bins = cut_bins)

                p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),0,1000,10,-10000 ]
                plot_gauss_poly3(output_pdf=output_pdf, hist = histx, title = "Histogram of clusters in x run %s" %run_number, axis = "x", bins = binsx, fit_p0=p, cut_bins = cut_bins)

                ''' fit y cluster histogram '''
                cut_bins = [0,-1]
                fit_histy = histy[cut_bins[0]:cut_bins[1]]
                p = [np.max(fit_histy), np.mean(fit_histy), np.std(fit_histy), 0, 1000, -15000 ]
                if run_number == 2781:
                    p = [np.max(fit_histy), np.mean(fit_histy), np.std(fit_histy), 0, 1000, -15000 ]
                if run_number == 2785:
                    p = [np.max(fit_histy), np.mean(fit_histy), np.std(fit_histy), 100, 1000, -10 ]
                if run_number==2798:
                    p = [np.max(fit_histy), np.mean(fit_histy), np.std(fit_histy), 100, 1000, -10 ]

                plot_gauss_poly2(output_pdf=output_pdf, hist = histy, title = "Histogram of clusters in y run %s" %run_number, axis = "y", bins = binsy, fit_p0 = p, cut_bins = cut_bins)


            for spill in range(start,nspills):
                no_target = False
                meansx = []
                meansy = []
                xerr = []
                yerr = []
                plot_cluster = clusters[clusters["spill"]==spill]
                if plot_cluster.shape[0]<1000:
                    continue
                if plot_cluster.shape[0]<30000:
                    no_target = True
                used_spills.append(spill)
                logging.info("analyzing spill %s" %spill)
                t_slices = np.linspace(0,plot_cluster["trigger_time_stamp"].max(),15)
                # if not global_coordinates:
                #     plot_cluster["x_dut_0"],plot_cluster["y_dut_0"],z0 = local_to_global_position(x = plot_cluster["x_dut_0"], y = plot_cluster["y_dut_0"], translation_x=-168*50 - 20, translation_y= None, translation_z=None, rotation_alpha=0, rotation_beta=np.pi, rotation_gamma=-np.pi/2)
                #     plot_cluster["x_dut_1"],plot_cluster["y_dut_1"],z1 = local_to_global_position(x = plot_cluster["x_dut_1"], y = plot_cluster["y_dut_1"], translation_x=+168*50 + 20, translation_y= None, translation_z=None, rotation_alpha=0, rotation_beta=None, rotation_gamma=np.pi/2)

                x0 = plot_cluster["x_dut_0"]
                y0 = plot_cluster["y_dut_0"]
                x1 = plot_cluster["x_dut_1"]
                y1 = plot_cluster["y_dut_1"]

                xbins = histbinsy
                ybins = [-16800 + i*50 for i in range(0,336)] + [i*50 for i in range(0,336)]
                bins = [ybins, xbins]

                x = np.concatenate((x0,x1))
                y = np.concatenate((y0,y1))
                x = x[~np.isnan(x)]
                y = y[~np.isnan(y)]

                spillx.append(np.mean(x))
                spilly.append(np.mean(y))
                spillxerr.append(np.std(x))
                spillyerr.append(np.std(y))

                weights = np.ones_like(y)
                weights[np.logical_and(y>-451, y<451)] = 0.56
                weights[y<-19950] = 0.5
                weights[y>19950] = 0.5

                hist, xbins, ybins = np.histogram2d(x, y, bins = bins, weights=weights )
                nentries = int(hist.flatten().sum())
                hist[:,79:81] = hist[:,79:81] * 0.57 #250/450.
                if no_target:
                    xlim=(0,12000)
                    ylim=(-12000,15000)
                else:
                    xlim=None
                    ylim=None
                plot_2d_hist(output_pdf=output_pdf, hist=hist, xbins = xbins, ybins = ybins, title="Beam spot run %s - spill %s - %s entries" %(run_number,spill, nentries), xlim=xlim, ylim=ylim)

                histx, binsx = np.histogram(x, bins =336)
                histy, binsy = np.histogram(y,bins=histbinsy, weights = weights)
                cut_bins = [None, None]
                if run_number==2793:
                    cut_bins = [[100,-20],None]
                    if spill == 7:
                        cut_bins[1] = [10,-10]
                if no_target==True:
                    plot_1d_hist(output_pdf, hist = histx, title="Clusters in x - run %s spill %s" % (run_number, spill), axis= "x", fitfunction = gauss, fit_p0=None, bins = binsx, cut_bins=cut_bins[0])
                    plot_1d_hist(output_pdf, hist = histy, title="Clusters in y - run %s spill %s" % (run_number, spill), axis= "y", fitfunction = gauss, fit_p0=None, bins = binsy, cut_bins=cut_bins[1])
                else:
                    ''' fit y cluster histogram '''
                    cut_bins = [1,-1]
                    fit_histy = histy[cut_bins[0]:cut_bins[1]]
                    p = [np.max(histy), np.mean(histy), np.std(histy),0,1000,-15000 ]
                    if run_number == 2798:
                        cut_bins = [20,-10]
                        p = [1000, 2900, np.std(fit_histy),0,-1e-6,1000 ]
                    plot_gauss_poly2(output_pdf=output_pdf, hist = histy, title = "clusters in y run %s - spill %s" %(run_number,spill), axis= "y", bins = binsy, fit_p0=p, cut_bins = cut_bins)

                    ''' fit x cluster histogram '''
                    cut_bins = [50,-20]
                    fit_histx=histx[cut_bins[0]:cut_bins[1]]
                    p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),0,1000,10,-10000 ]

                    if run_number == 2781:
                        if spill==16 :
                            cut_bins = [40,-20]
                            fit_histx=histx[cut_bins[0]:cut_bins[1]]
                            p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),0,100,10,-1000 ]
                    if run_number == 2785:
                        cut_bins = [30,-20]
                        fit_histx = histx[cut_bins[0]:cut_bins[1]]
                        p = [np.max(fit_histx), 5000, np.std(fit_histx),0,100,10,-1000 ]
                        # if spill == 5:
                        #     cut_bins = [50,-25]
                        #     fit_histx = histx[cut_bins[0]:cut_bins[1]]
                        #     p = [np.max(fit_histx), 5000, np.std(fit_histx),0,100,10,-1000 ]
                        if spill == 16:
                            p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),0,100,10,-1000 ]
                    if run_number == 2793:
                        p = [np.max(fit_histx), np.mean(fit_histx)*4, np.std(fit_histx),0.01,-4e-6,-3e-10,2000 ]
                    if run_number == 2798:
                        cut_bins = [80,-20]
                        fit_histx = histx[cut_bins[0]:cut_bins[1]]
                        p = [np.max(fit_histx), np.mean(fit_histx)*2, np.std(fit_histx),0.01,-4e-6,-3e-10,2000]
                        if spill==2:
                            cut_bins = [20,-20]
                            fit_histx = histx[cut_bins[0]:cut_bins[1]]
                            p = [np.max(fit_histx), 5000, np.std(fit_histx),0.01,-4e-6,-3e-10,2000]

                    plot_gauss_poly3(output_pdf=output_pdf, hist = histx, title = "clusters in x run %s - spill %s" %(run_number,spill), axis= "x", bins = binsx, fit_p0=p, cut_bins = cut_bins)

                masks = []
                for i, time in enumerate(t_slices[:-1]):
                    mask = np.logical_and((t_slices[i] <= plot_cluster['trigger_time_stamp']), (plot_cluster['trigger_time_stamp'] <= t_slices[i+1]))
                    masks.append(mask)
                    slice_x = np.concatenate((plot_cluster["x_dut_0"][mask],plot_cluster["x_dut_1"][mask]))
                    slice_y = np.concatenate((plot_cluster["y_dut_0"][mask],plot_cluster["y_dut_1"][mask]))
                    slice_x = slice_x[~np.isnan(slice_x)]
                    slice_y = slice_y[~np.isnan(slice_y)]
                    meansx.append(np.mean(slice_x))
                    meansy.append(np.mean(slice_y))
                    xerr.append(np.std(meansx[-1]))
                    yerr.append(np.std(meansy[-1]))

                    if no_target:
                        hist, xbins, ybins = np.histogram2d(slice_x, slice_y, bins = bins)
                        nentries = int(hist.flatten().sum())
                        hist[:,79:81] = hist[:,79:81] * 0.57 #250/450.
                        xlim = (0,12000)
                        ylim = (-12000,15000)
                        plot_2d_hist(output_pdf=output_pdf, hist=hist, xbins = xbins,ybins = ybins, title="Beam spot run %s - spill %s - %.2E ms" %(run_number,spill, t_slices[i+1]), xlim=xlim, ylim=ylim, cmap_bad = "w")

                xcenter = np.median(meansx)
                ycenter = np.median(meansy)
                xcorr = [meanx - xcenter for meanx in meansx]
                ycorr = [meany - ycenter for meany in meansy]
                xcorrs.append(np.mean(xcorr))
                ycorrs.append(np.mean(ycorr))

                for i, mask in enumerate(masks):
                    plot_cluster["x_dut_0"][mask] = plot_cluster["x_dut_0"][mask] - xcorr[i]
                    plot_cluster["x_dut_1"][mask] = plot_cluster["x_dut_1"][mask] - xcorr[i]
                    plot_cluster["y_dut_0"][mask] = plot_cluster["y_dut_0"][mask] - ycorr[i]
                    plot_cluster["y_dut_1"][mask] = plot_cluster["y_dut_1"][mask] - ycorr[i]

                x_corrected = np.concatenate((plot_cluster["x_dut_0"], plot_cluster["x_dut_1"]))
                y_corrected = np.concatenate((plot_cluster["y_dut_0"], plot_cluster["y_dut_1"]))
                x_corrected = x_corrected[~np.isnan(x_corrected)]
                y_corrected = y_corrected[~np.isnan(y_corrected)]

                # weights = np.ones_like(y_corrected)
                # weights[np.logical_and(y_corrected>-451, y_corrected<451)] = 0.56
                # weights[y_corrected<-19950] = 0.5
                # weights[y_corrected>19950] = 0.5

                t_slices = t_slices * 25e-9 * 1e3
                cmin2 = t_slices.min()
                cmax2 = t_slices.max()
                cmap = cm.get_cmap("viridis")
                fig2 = Figure()
                _ = FigureCanvas(fig2)
                ax2 = fig2.add_subplot(111)
                im2 = ax2.scatter(np.array(meansx), np.array(meansy), c =t_slices[:-1], s = 50 , marker = "x")
                for j, _ in enumerate(t_slices[:-1]):
                    color = cmap(1.0*j/len(t_slices[:-1]))
                    ax2.errorbar(np.array(meansx[j]), np.array(meansy[j]), xerr = xerr[j], yerr = yerr[j], capsize = 0, linestyle="None" , c= color)
                bounds2 = np.linspace(start=cmin2, stop=cmax2, num=256, endpoint=True)
                cbar2 = fig2.colorbar(im2, boundaries=bounds2, ticks=np.linspace(start=cmin2, stop=cmax2, num=9, endpoint=True), fraction=0.04, pad=0.05)
                cbar2.set_label("time in ms")
                ax2.grid()
                ax2.set_title("Beam positions in run %s - spill %s" %(run_number,spill))
                ax2.set_xlabel("x position $\mu$m")
                ax2.set_ylabel("y position $\mu$m")
                fig2.tight_layout()
                output_pdf.savefig(fig2)

                histx, binsx = np.histogram(x_corrected, bins =336)
                histy, binsy = np.histogram(y_corrected, bins=161, weights = weights)
                if spill == 7:
                    cut_bins = [[100,-20],None]
                else:
                    cut_bins = [None, None]
                if no_target:
                    xlim = (0,12000)
                    ylim = (-12000,15000)
                    hist, xbins, ybins = np.histogram2d(x_corrected, y_corrected, bins = bins)
                    nentries = int(hist.flatten().sum())
                    plot_2d_hist(output_pdf=output_pdf, hist=hist, xbins = xbins,ybins = ybins, title="corrected spot run %s - spill %s - %s entries" %(run_number,spill, nentries), xlim=xlim, ylim=ylim)

                    plot_1d_hist(output_pdf, hist = histx, title="Clusters in x corrected - run %s spill %s" % (run_number, spill), axis= "x", fitfunction = gauss, fit_p0=None, bins = binsx, cut_bins=cut_bins[0])
                    plot_1d_hist(output_pdf, hist = histy, title="Clusters in y corrected - run %s spill %s" % (run_number, spill), axis= "y", fitfunction = gauss, fit_p0=None, bins = binsy, cut_bins=cut_bins[1])
                else:
                    ''' fit y cluster histogram '''
                    cut_bins = [1,-1]
                    p = [np.max(histy), np.mean(histy), np.std(histy),0,1000,-15000 ]
                    plot_gauss_poly2(output_pdf=output_pdf, hist = histy, title = "corrected clusters in y run %s - spill %s" %(run_number,spill), axis= "y", fit_p0=p, bins = binsy, cut_bins = cut_bins)

                    ''' fit x cluster histogram '''
                    cut_bins = [30,-20]
                    fit_histx=histx[cut_bins[0]:cut_bins[1]]
                    p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),0,1000,10,-10000 ]

                    if run_number == 2781:
                        p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),0,1000,10,-10000 ]
                        if spill==12 :
                            cut_bins = [40,-20]
                            fit_histx=histx[cut_bins[0]:cut_bins[1]]
                            p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),0,100,10,-1000 ]
                    if run_number == 2785:
                        cut_bins = [30,-20]
                        fit_histx = histx[cut_bins[0]:cut_bins[1]]
                        p = [np.max(fit_histx), 5000, np.std(fit_histx),0,100,10,-1000 ]
                        if spill==5:
                            cut_bins = [40,-20]
                            fit_histx = histx[cut_bins[0]:cut_bins[1]]
                            p = [np.max(fit_histx), 5000, np.std(fit_histx),0,100,10,-1000 ]
                        # p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),100,1000,10,-10000 ]
                        # if spill== 9:
                        #     p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),100,1000,10,-1000 ]
                        # if spill == 10:
                        #     p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),100,1000,1,-10000 ]
                        # if spill >= 11: # or spill==12 or spill==13:
                        #     cut_bins=[1,-1]
                        # if spill == 16:
                            # p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),0,100,10,-1000 ]
                    if run_number== 2793:
                        cut_bins = [50,-20]
                        p = [np.max(fit_histx), np.mean(fit_histx)*4, np.std(fit_histx),0.01,-4e-6,-3e-10,2000 ]
                    if run_number == 2799:
                        cut_bins = [80,-20]
                        fit_histx = histx[cut_bins[0]:cut_bins[1]]
                        p = [np.max(fit_histx), np.mean(fit_histx)*2, np.std(fit_histx),0.01,-4e-6,-3e-10,2000]
                    if run_number == 2798:
                        cut_bins = [80,-20]
                        fit_histx = histx[cut_bins[0]:cut_bins[1]]
                        p = [np.max(fit_histx), np.mean(fit_histx)*2, np.std(fit_histx),0.01,-4e-6,-3e-10,2000]
                        if spill ==2:
                            cut_bins = [20,-20]
                            fit_histx = histx[cut_bins[0]:cut_bins[1]]
                            p = [np.max(fit_histx), 5000, np.std(fit_histx),0.01,-4e-6,-3e-10,2000]

                    plot_gauss_poly3(output_pdf=output_pdf, hist = histx, title = "corrected clusters in x run %s - spill %s" %(run_number,spill), axis= "x", bins = binsx, fit_p0=p, cut_bins = cut_bins)

            for i, spill in enumerate(used_spills):
                clusters["x_dut_0"][clusters["spill"]==spill] -= xcorrs[i]
                clusters["x_dut_1"][clusters["spill"]==spill] -= xcorrs[i]
                clusters["y_dut_0"][clusters["spill"]==spill] -= ycorrs[i]
                clusters["y_dut_1"][clusters["spill"]==spill] -= ycorrs[i]

            x0 = clusters["x_dut_0"]
            y0 = clusters["y_dut_0"]
            x1 = clusters["x_dut_1"]
            y1 = clusters["y_dut_1"]

            x = np.concatenate((x0,x1))
            y = np.concatenate((y0,y1))
            x = x[~np.isnan(x)]
            y = y[~np.isnan(y)]

            weights = np.ones_like(y)
            weights[np.logical_and(y>-451, y<451)] = 0.56
            weights[y<-19950] = 0.5
            weights[y>19950] = 0.5

            histy, binsy = np.histogram(y, bins = 161, weights = weights)
            histx, binsx = np.histogram(x, bins = 336)

            if no_target:
                plot_1d_hist(output_pdf, hist = histx, title="Clusters in x corrected - run %s" % (run_number), axis= "x", fitfunction = gauss, fit_p0=None, bins = binsx, cut_bins=None)
                plot_1d_hist(output_pdf, hist = histy, title="Clusters in y corrected - run %s" % (run_number), axis= "y", fitfunction = gauss, fit_p0=None, bins = binsy, cut_bins=None)
            else:
                ''' fit x cluster histogram '''
                cut_bins = [50,-20]
                fit_histx = histx[cut_bins[0]:cut_bins[1]]
                p = [2000,5000,1e8,5000,10,0,5000]
                plot_cauchy_poly2(output_pdf=output_pdf, hist = histx, title = "Corrected clusters in x run %s" %(run_number), axis= "x", fit_p0=p, bins = binsx, cut_bins = cut_bins)

                fitfunction = gauss_poly3
                p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),0,1000,10,-10000 ]
                if run_number == 2793:
                    cut_bins = [120,-5]
                    fit_histx = histx[cut_bins[0]:cut_bins[1]]
                    p = [np.max(fit_histx), np.mean(fit_histx), np.std(fit_histx),0,1000,10,-15000 ]

                plot_gauss_poly3(output_pdf=output_pdf, hist = histx, title = "Corrected clusters in x run %s" %(run_number), axis= "x", fit_p0=p, bins = binsx, cut_bins = cut_bins)

                ''' fit y cluster histogram '''
                cut_bins = [1,-1]
                fit_histy = histy[cut_bins[0]:cut_bins[1]][cut_bins[0]:cut_bins[1]]
                p = [np.max(fit_histy), np.mean(fit_histy), np.std(fit_histy),0,1000,-15000 ]
                if run_number == 2785 or run_number == 2781:
                    cut_bins = [5,-5]
                    p = [np.max(fit_histy), np.mean(fit_histy), np.std(fit_histy), 100, 1000, -10 ]

                plot_gauss_poly2(output_pdf=output_pdf, hist = histy, title = "Corrected clusters in y run %s" %(run_number), axis= "y", bins = binsy, fit_p0=p, cut_bins = cut_bins)


def plot_tot_histograms(run_number, pybar_runs,out_file_path):
    hit_files= ['/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_0_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[0]),
                '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_0_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[0]),
                '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_1_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[0]),
                '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_1_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[0]),
                '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_2_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[1]),
                '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_2_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[1]),
                '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_3_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[1]),
                '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_3_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[1]),
                '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_4_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[2]),
                '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_4_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[2]),
                '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_5_DC_module_0_local_corr_evts.h5' % (run_number, pybar_runs[2]),
                '/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_%s/pyBARrun_%s_plane_5_DC_module_1_local_corr_evts.h5' % (run_number, pybar_runs[2]),
                ]
    with PdfPages(out_file_path) as output_pdf:
        fig = Figure(figsize=(30,15))
        _ = FigureCanvas(fig)
        for i, hit_file in enumerate(hit_files):
            with tb.open_file(hit_file) as in_file_h5:
                hits = in_file_h5.root.Hits[:]
                charge = hits["charge"]
                nhits = charge.shape[0]

            ax = fig.add_subplot(3,4,i+1)
            ax.set_title("ToT for hits in run %s plane %s - %s hits" %(run_number, i, nhits))
            ax.set_xlabel("ToT [25ns]")
            ax.hist(charge, bins = np.arange(0,16,1))
            ax.grid()

            # ax.subplots_adjust(hspace = 0.33)
            # ax.subplots_adjust(wspace = 0.20)
        # plt.tight_layout()
        # plt.show()
        # plt.savefig(fig)
        output_pdf.savefig(fig)



def transform_to_emulsion_frame(clusters,duts=(0,1),emulsion_speed=2.6, y_offset=10000, start_spill=0):
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

    clustersx1 = clusters["x_dut_1"] + (-1)**(clusters["spill"]%2+1) * 10000*emulsion_speed * clusters["trigger_time_stamp"]*25e-9
    clustersx0 = clusters["x_dut_0"] + (-1)**(clusters["spill"]%2+1) * 10000*emulsion_speed * clusters["trigger_time_stamp"]*25e-9

    spill_mask = clusters["spill"]%2==0
    if np.any(spill_mask):
        x_offset = max(np.abs(clustersx0[np.logical_and(spill_mask , (~np.isnan(clustersx0)))]).max() , np.abs(clustersx1[np.logical_and(spill_mask , (~np.isnan(clustersx1)))]).max())
        print "x offset: ", x_offset
        clustersx0[np.where(spill_mask)] += x_offset
        clustersx1[np.where(spill_mask)] += x_offset

    clustersy0 = clusters["y_dut_0"] + (clusters["spill"]-start_spill)*y_offset
    clustersy1 = clusters["y_dut_1"] + (clusters["spill"]-start_spill)*y_offset

    x = np.concatenate((clustersx0,clustersx1))
    y = np.concatenate((clustersy0,clustersy1))
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    x = x[y>0]
    y = y[y>0]
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
    ax.grid(linestyle="--", linewidth=0.2, alpha = 0.5)
    if spill:
        ax.set_title("Spill %s DUT 0/1 in emulsion rest frame" %spill)
    else:
        ax.set_title("DUT 0/1 in emulsion rest frame - %s entries" %y[~np.isnan(y)].shape[0])
    ax.set_ylim(0,120000)
    ax.set_aspect(1/1.2)
    ax.set_ylabel("y position [$\mu$m]")
    ax.set_xlabel("x position [$\mu$m]")
    fig.tight_layout()

    output_pdf.savefig(fig)


# def plot_1d_hist(output_pdf, x, bins=None, title= None, fit=False):
#     fig = Figure()
#     _ = FigureCanvas(fig)
#     ax = fig.add_subplot(111)
#     n, bins, patches = ax.hist(x, bins = bins)
#     ax.grid(linestyle="--", linewidth=0.2, alpha = 0.5)
#     ax.set_title(title + " - %s entries" %x.shape[0])
#     # max_index, = np.where(n==n.max())
#     # plt.bar(bins[:-1],n,width=np.diff(bins),align="edge")
#     # n[max_index-1] += 3500
#     # n[max_index+2] += 3300
#     # n[max_index+1] -= 2500
#     # n[max_index] -= 4500
#     # plt.bar(bins[:-1],n,width=np.diff(bins),align="edge")
#     # plt.show()
#     if fit:
#         # x = x[np.where((x<50000) & (x>30000))]
#         mask = np.where((bins>25000)&(bins<50000))
#         xdata = bins[mask]
#         fit, xcov = curve_fit(f=analysis_utils.gauss,xdata=xdata, ydata=n[mask], p0=[9000,38000, np.std(n[mask])])
#         # fit = fit_multigauss(x)
#         # fit_y = fit.values()
#         ax.plot(xdata, analysis_utils.gauss(xdata,*fit), linestyle="--")
#
#     output_pdf.savefig(fig)


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(- (x - mu)**2.0 / (2.0 * sigma**2.0))

def gauss_poly2(x, *p):
    A, mu, sigma, a, b, offset = p
    return gauss(x, A, mu, sigma) + poly2(x,a,b,offset)

def gauss_poly3(x, *p):
    A, mu, sigma, a, b, c, offset = p
    gauss = A * np.exp(- (x - mu)**2.0 / (2.0 * sigma**2.0))
    return gauss + poly3(x,a,b,c,offset)

def gauss_poly4(x, *p):
    A, mu, sigma, a, b, c, d, offset = p
    gauss = A * np.exp(- (x - mu)**2.0 / (2.0 * sigma**2.0))
    return gauss + poly4(x,a,b,c,d,offset)

def poly2 (x,*p):
    a,b, offset = p
    return a*x+b*(x)**2 + offset

def poly3 (x,*p):
    a,b,c, offset = p
    return a*x + b*x**2 + c*x**3 + offset

def poly4 (x,*p):
    a,b,c,d, offset = p
    return a*x + b*x**2 + c*x**3 + d*x**4 + offset

def cauchy(x, *p):
    gamma, x0, factor, offset = p
    return 1/(np.pi*gamma*(1+((x-x0)/gamma)**2)) * factor + offset

def cauchy_poly2(x, *p):
    gamma, x0, factor, offset, a, b, offset2 = p
    return cauchy(x, gamma, x0, factor, offset) + poly2(x, a, b, offset2)

def voigt_offset(x,*p):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.
    see: https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
    CAREFUL!!!!!! added offset to gamma (offset2)

    """
    alpha, gamma, offset, offset2, offset3= p
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma + offset2)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi) * offset

def voigt(x,*p):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.
    see: https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/

    """
    alpha, gamma, offset = p
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma /np.sqrt(2*np.pi)


def plot_spill_histograms(merged_file_in, spills=None, duts=(0,1), start_spill=0):
    ''' transform clusters in rest frame of emulsion, movement speed was 2.6cm/s
    input
    ---------------
        merged_file_in: string pointing to cluster file with merged clusters
        spills: tuple or list with first and last spill for single spill plotting
        duts :tuple of duts to plot
    '''
    xbins = [0] + [500 + i*250 for i in range(0,79)] + [20400] + [20800 + i*250 for i in range(0,79)] + [40800] #[-20400] + [-19950 + i*250 for i in range(0,79)] + [0] + [200 + i*250 for i in range(1,80)] + [20400]
    xbins = [0] + [250*i for i in range(0,162)]
    # xbins = 160
    ybins = 250
    with tb.open_file(merged_file_in,'r') as cluster_file_in:
        clustertable = cluster_file_in.root.MergedClusters[:]
        output_pdf = PdfPages(merged_file_in.replace(".h5","_emulsion_rest_frame_per_spill.pdf"))
        if spills:
            for spill in range(spills[0],spills[1]+1):
                print "plotting spill %s" %spill
                clusters = clustertable[np.where(clustertable["spill"]==spill)]
                x,y = transform_to_emulsion_frame(clusters,duts)
                plot_spill(output_pdf, x, y, bins=[ybins,np.array(xbins)+(10000*(spill-spills[0]))], spill=spill)
        ''' now plot all spills'''
        print "plotting all spills"
        clusters = clustertable
        x,y = transform_to_emulsion_frame(clusters,duts,y_offset=20000, start_spill=start_spill)
        plot_spill(output_pdf,x,y,bins=[ybins,520])
        # plot_1d_hist(output_pdf,y[x<x.max()/2], bins = 520, title="left half", fitfunction=None)
        # plot_1d_hist(output_pdf,y[x>x.max()/2], bins = 520, title="right half", fitfunction=None)
        output_pdf.close()



if __name__ == '__main__':
    # 2793 : ("376", "270", "204")
    # plot_tot_histograms(2815,( '394', '288', '222'), "/home/niko/Desktop/tot_hist_run2815.pdf" )
    # raise
    plot_cluster_hist(cluster_file = "/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2793/Merged.h5",
                        hit_file = "/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/run_2793/pyBARrun_376_plane_0_DC_module_1_local_corr_evts_clustered.h5",
                        cluster_size_threshold = 10)
    plot_spill_histograms(merged_file_in = "/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2793/Merged_spills.h5", start_spill=8) # , spills = [8,18]
    # plot_cluster_2dhist_global(cluster_file = "/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2781/Merged.h5",
    #                         hit_file = None)
    # plot_timestamps("/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2793/Merged_spills_global.h5")
    # plot_pos_vs_time("/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2793/Merged.h5")
    raise
    run_list = [
                "/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2781/Merged_spills_global.h5",
                # "/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2785/Merged_spills_global.h5",
                # "/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2793/Merged_spills_global.h5",
                # "/media/niko/data/SHiP/charm_exp_2018/data/tba_improvements/output_folder_run_2798/Merged_spills_global.h5",
                ]
    for run_file in run_list:
        plot_pos_vs_time_per_spill(merged_file_in= run_file,no_target=False)
        plot_pos_vs_time_per_spill(merged_file_in= run_file,no_target=True)
    # plot_material_dependence("/home/niko/cernbox/talks/SHiP/clb_meeting_Mar19/material_dependence.pdf")
