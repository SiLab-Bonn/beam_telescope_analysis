from __future__ import division

import logging
import re
import os.path
import warnings
from math import ceil

import numpy as np
import tables as tb
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import colors, cm

from mpl_toolkits.mplot3d import Axes3D  # needed for 3d plotting although it is shown as not used
from matplotlib.widgets import Slider, Button
from scipy.optimize import curve_fit
from scipy import stats

import testbeam_analysis.tools.analysis_utils
import testbeam_analysis.tools.geometry_utils

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")  # Plot backend error not important


def plot_2d_map(hist2d, plot_range, title=None, x_axis_title=None, y_axis_title=None, z_min=0, z_max=None, output_pdf=None, gui=False, figs=None):
    if not output_pdf and not gui:
        return
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig=fig, ax=ax, hist2d=hist2d, plot_range=plot_range, title=title, x_axis_title=x_axis_title, y_axis_title=y_axis_title, z_min=z_min, z_max=z_max)
    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)


def plot_2d_pixel_hist(fig, ax, hist2d, plot_range, title=None, x_axis_title=None, y_axis_title=None, z_min=0, z_max=None):
#     extent = [0.5, plot_range[0] + .5, plot_range[1] + .5, 0.5]
    if z_max is None:
        if hist2d.all() is np.ma.masked:  # check if masked array is fully masked
            z_max = 1
        else:
            z_max = ceil(hist2d.max())
    bounds = np.linspace(start=z_min, stop=z_max, num=255, endpoint=True)
    cmap = cm.get_cmap('viridis')
    cmap.set_bad('w')
    im = ax.imshow(hist2d, interpolation='none', aspect="auto", extent=plot_range, cmap=cmap, clim=(0, z_max))
    if title is not None:
        ax.set_title(title)
    if x_axis_title is not None:
        ax.set_xlabel(x_axis_title)
    if y_axis_title is not None:
        ax.set_ylabel(y_axis_title)
    fig.colorbar(im, boundaries=bounds, ticks=np.linspace(start=z_min, stop=z_max, num=9, endpoint=True), fraction=0.04, pad=0.05)


def plot_masked_pixels(input_mask_file, pixel_size=None, dut_name=None, output_pdf_file=None, gui=False):
    with tb.open_file(input_mask_file, 'r') as input_file_h5:
        try:
            noisy_pixels = np.dstack(np.nonzero(input_file_h5.root.NoisyPixelMask[:].T))[0]
            n_noisy_pixels = np.count_nonzero(input_file_h5.root.NoisyPixelMask[:])
        except tb.NodeError:
            noisy_pixels = None
            n_noisy_pixels = 0
        try:
            disabled_pixels = np.dstack(np.nonzero(input_file_h5.root.DisabledPixelMask[:].T))[0]
            n_disabled_pixels = np.count_nonzero(input_file_h5.root.DisabledPixelMask[:])
        except tb.NodeError:
            disabled_pixels = None
            n_disabled_pixels = 0
        occupancy = input_file_h5.root.HistOcc[:].T

    if pixel_size:
        aspect = pixel_size[1] / pixel_size[0]
    else:
        aspect = "auto"

    if dut_name is None:
        dut_name = os.path.split(input_mask_file)[1]

    if output_pdf_file is None:
        output_pdf_file = os.path.splitext(input_mask_file)[0] + '_masked_pixels.pdf'

    if gui:
        figs = []

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:  # if gui, we dont want to safe empty pdf
        cmap = cm.get_cmap('viridis')
        cmap.set_bad('w')
        c_max = np.percentile(occupancy, 99)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_title('%s' % (dut_name, ))
        # plot noisy pixels
        if noisy_pixels is not None:
            ax.plot(noisy_pixels[:, 1], noisy_pixels[:, 0], 'ro', mfc='none', mec='c', ms=10, label='Noisy pixels')
            ax.set_title(ax.get_title() + ',\n%d noisy pixels' % (n_noisy_pixels,))
        # plot disabled pixels
        if disabled_pixels is not None:
            ax.plot(disabled_pixels[:, 1], disabled_pixels[:, 0], 'ro', mfc='none', mec='r', ms=10, label='Disabled pixels')
            ax.set_title(ax.get_title() + ',\n%d disabled pixels' % (n_disabled_pixels,))
        ax.imshow(np.ma.getdata(occupancy), aspect=aspect, cmap=cmap, interpolation='none', origin='lower', clim=(0, c_max))
        ax.set_xlim(-0.5, occupancy.shape[1] - 0.5)
        ax.set_ylim(-0.5, occupancy.shape[0] - 0.5)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        if gui:
            leg = ax.legend(numpoints=1, bbox_to_anchor=(1.015, 1.135), loc='upper right')
            leg.get_frame().set_facecolor('none')
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_title('%s\n occupancy' % (dut_name, ))
        ax.imshow(occupancy, aspect=aspect, cmap=cmap, interpolation='none', origin='lower', clim=(0, c_max))
    #     np.ma.filled(occupancy, fill_value=0)
        ax.set_xlim(-0.5, occupancy.shape[1] - 0.5)
        ax.set_ylim(-0.5, occupancy.shape[0] - 0.5)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

    if gui:
        return figs


def plot_cluster_size(input_cluster_file, dut_name=None, output_pdf_file=None, chunk_size=1000000, gui=False):
    '''Plotting cluster size histogram.

    Parameters
    ----------
    input_cluster_file : string
        Filename of the input cluster file.
    dut_name : string
        Name of the DUT. If None, the filename of the input cluster file will be used.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    if not dut_name:
        dut_name = os.path.split(input_cluster_file)[1]

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_cluster_file)[0] + '_cluster_size.pdf'

    if gui:
        figs = []

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:  # if gui, we dont want to safe empty pdf
        with tb.open_file(input_cluster_file, 'r') as input_file_h5:
            hight = None
            n_hits = 0
            n_clusters = input_file_h5.root.Cluster.nrows
            for start_index in range(0, n_clusters, chunk_size):
                cluster_n_hits = input_file_h5.root.Cluster[start_index:start_index + chunk_size]['n_hits']
                # calculate cluster size histogram
                if hight is None:
                    max_cluster_size = np.amax(cluster_n_hits)
                    hight = testbeam_analysis.tools.analysis_utils.hist_1d_index(cluster_n_hits, shape=(max_cluster_size + 1,))
                elif max_cluster_size < np.amax(cluster_n_hits):
                    max_cluster_size = np.amax(cluster_n_hits)
                    hight.resize(max_cluster_size + 1)
                    hight += testbeam_analysis.tools.analysis_utils.hist_1d_index(cluster_n_hits, shape=(max_cluster_size + 1,))
                else:
                    hight += testbeam_analysis.tools.analysis_utils.hist_1d_index(cluster_n_hits, shape=(max_cluster_size + 1,))
                n_hits += np.sum(cluster_n_hits)

        left = np.arange(max_cluster_size + 1)
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.bar(left, hight, align='center')
        ax.set_title('Cluster size of %s\n(%i hits in %i clusters)' % (dut_name, n_hits, n_clusters))
        ax.set_xlabel('Cluster size')
        ax.set_ylabel('#')
        ax.grid()
        ax.set_yscale('log')
        ax.set_xlim(xmin=0.5)
        ax.set_ylim(ymin=1e-1)

        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.bar(left, hight, align='center')
        ax.set_title('Cluster size of %s\n(%i hits in %i clusters)' % (dut_name, n_hits, n_clusters))
        ax.set_xlabel('Cluster size')
        ax.set_ylabel('#')
        ax.grid()
        ax.set_yscale('linear')
        ax.set_ylim(ymax=np.amax(hight))
        ax.set_xlim(0.5, min(10, max_cluster_size) + 0.5)

        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

    if gui:
        return figs


def plot_tracks_per_event(input_tracks_file, output_pdf_file=None, gui=False):
    """Plotting tracks per event
    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    chunk_size : int
        Chunk size of the data when reading from file.
    gui: bool
        Whether or not to plot directly into gui
    """
    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_tracks_per_event.pdf'

    if gui:
        figs = []

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:  # if gui, we dont want to safe empty pdf
        with tb.open_file(input_tracks_file, 'r') as input_file_h5:
            fitted_tracks = False
            try:  # data has track candidates
                _ = input_file_h5.root.TrackCandidates
            except tb.NoSuchNodeError:  # data has fitted tracks
                fitted_tracks = True

            for node in input_file_h5.root:

                table_events = node[:]['event_number']

                events, event_count = np.unique(table_events, return_counts=True)
                tracks_per_event, tracks_count = np.unique(event_count, return_counts=True)

                if not fitted_tracks:
                    title = 'Track candidates per event number\n for %d events' % events.shape[0]
                else:
                    title = 'Tracks per event number of Tel_%s\n for %d events' % \
                            (str(node.name).split('_')[-1], events.shape[0])

                xlabel = 'Track candidates per event' if not fitted_tracks else 'Tracks per event'

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.bar(tracks_per_event, tracks_count, align='center')
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('#')
                ax.grid()
                ax.set_yscale('log')

                if gui:
                    figs.append(fig)
                else:
                    output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.bar(tracks_per_event, tracks_count, align='center')
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('#')
                ax.grid()
                ax.set_yscale('linear')

                if gui:
                    figs.append(fig)
                else:
                    output_pdf.savefig(fig)

    if gui:
        return figs


def plot_cluster_hists(input_cluster_file=None, input_tracks_file=None, dut_name=None, dut_names=None, select_duts=None, output_pdf_file=None, gui=False, chunk_size=1000000):
    '''Plotting cluster histograms.

    Parameters
    ----------
    input_cluster_file : string
        Filename of the input cluster file.
    input_tracks_file : string
        Filename of the input tracks file.
    dut_name : string
        TBD
    dut_names : iterable
        TBD
    select_duts : iterable
        TBD
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    gui : boolean
        Whether or not to plot directly into gui
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    if gui:
        figs = []

    if (input_cluster_file and input_tracks_file) or (not input_cluster_file and not input_tracks_file):
        raise ValueError("A single input file must be given")

    if input_cluster_file and dut_names:
        raise ValueError("\"dut_name\" parameter must be used")

    if input_tracks_file and dut_name:
        raise ValueError("\"dut_names\" parameter must be used")

    if input_tracks_file and not select_duts:
        raise ValueError("\"select_duts\" parameter must be given")

    if input_cluster_file:
        input_file = input_cluster_file
        select_duts = [None]
    else:
        input_file = input_tracks_file

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_file)[0] + '_cluster_hists.pdf'

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_file, "r") as in_file_h5:
            for actual_dut in select_duts:
                if actual_dut is None:
                    node = in_file_h5.get_node(in_file_h5.root, 'Cluster')
                else:
                    node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT_%d' % actual_dut)

                    logging.info('Plotting cluster histograms for DUT%d', actual_dut)

                    if dut_names is not None:
                        dut_name = dut_names[actual_dut]
                    else:
                        dut_name = "DUT%d" % actual_dut

                initialize = True  # initialize the histograms
                n_hits = 0
                n_clusters = 0
                for chunk, _ in testbeam_analysis.tools.analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):

                    if actual_dut is None:
                        cluster_n_hits = chunk['n_hits']
                        cluster_shape = chunk['cluster_shape']
                    else:
                        cluster_n_hits = chunk['n_hits_dut_%d' % actual_dut]
                        cluster_shape = chunk['cluster_shape_dut_%d' % actual_dut]

                    max_cluster_size = np.max(cluster_n_hits)
                    n_hits += np.sum(cluster_n_hits)
                    n_clusters += chunk.shape[0]
                    edges = np.arange(2**16)
                    if initialize:
                        initialize = False

                        cluster_size_hist = np.bincount(cluster_n_hits, minlength=max_cluster_size + 1)
                        cluster_shapes_hist, _ = np.histogram(a=cluster_shape, bins=edges)
                    else:
                        if cluster_size_hist.size - 1 < max_cluster_size:
                            max_cluster_size = np.max(cluster_n_hits)
                            cluster_size_hist.resize(max_cluster_size + 1)
                            cluster_size_hist += np.bincount(cluster_n_hits, minlength=max_cluster_size + 1)
                        else:
                            cluster_size_hist += np.bincount(cluster_n_hits, minlength=cluster_size_hist.size)
                        cluster_shapes_hist += np.histogram(a=cluster_shape, bins=edges)[0]

                x = np.arange(cluster_size_hist.size)
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.bar(x, cluster_size_hist, align='center')
                ax.set_title('Cluster sizes%s\n(%i hits in %i clusters)' % ((" of %s" % dut_name) if dut_name else "", n_hits, n_clusters))
                ax.set_xlabel('Cluster size')
                ax.set_ylabel('#')
                ax.grid()
                ax.set_yscale('log')
                ax.set_xlim(xmin=0.5)
                ax.set_ylim(ymin=1e-1)
                if gui:
                    figs.append(fig)
                else:
                    output_pdf.savefig(fig)
                ax.set_yscale('linear')
                ax.set_ylim(ymin=0.0, ymax=np.max(cluster_size_hist))
                ax.set_xlim(0.5, min(10.0, cluster_size_hist.size - 1) + 0.5)
                if gui:
                    figs.append(fig)
                else:
                    output_pdf.savefig(fig)

                x = np.arange(12)
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                selected_clusters = cluster_shapes_hist[[1, 3, 5, 6, 9, 13, 14, 7, 11, 19, 261, 15]]
                ax.bar(x, selected_clusters, align='center')
                ax.xaxis.set_ticks(x)
                fig.subplots_adjust(bottom=0.2)
                ax.set_xticklabels([u"\u2004\u2596",
                                    u"\u2597\u2009\u2596", # 2 hit cluster, horizontal
                                    u"\u2004\u2596\n\u2004\u2598", # 2 hit cluster, vertical
                                    u"\u259e", # 2 hit cluster
                                    u"\u259a", # 2 hit cluster
                                    u"\u2599", # 3 hit cluster, L
                                    u"\u259f", # 3 hit cluster
                                    u"\u259b", # 3 hit cluster
                                    u"\u259c", # 3 hit cluster
                                    u"\u2004\u2596\u2596\u2596", # 3 hit cluster, horizontal
                                    u"\u2004\u2596\n\u2004\u2596\n\u2004\u2596", # 3 hit cluster, vertical
                                    u"\u2597\u2009\u2596\n\u259d\u2009\u2598"]) # 4 hit cluster
                ax.set_title('Cluster shapes%s\n(%i hits in %i clusters)' % ((" of %s" % dut_name) if dut_name else "", n_hits, n_clusters))
                ax.set_xlabel('Cluster shape')
                ax.set_ylabel('#')
                ax.grid()
                ax.set_yscale('log')
                ax.set_ylim(ymin=1e-1)
                if gui:
                    figs.append(fig)
                else:
                    output_pdf.savefig(fig)
                ax.set_yscale('linear')
                ax.set_ylim(ymin=0.0, ymax=np.max(selected_clusters))
                if gui:
                    figs.append(fig)
                else:
                    output_pdf.savefig(fig)
    if gui:
        return figs


def plot_correlation_fit(x, y, x_fit, y_fit, xlabel, fit_label, title, output_pdf=None, gui=False, figs=None):
    if not output_pdf and not gui:
        return
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'r.-', label='Data')
    ax.plot(x_fit, y_fit, 'g-', linewidth=2, label='Fit: %s' % fit_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('#')
    ax.set_xlim((np.min(x), np.max(x)))
    ax.grid()
    ax.legend(loc=0)

    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)


def plot_prealignments(x, mean_fitted, mean_error_fitted, n_cluster, ref_name, dut_name, prefix, pre_fit=None, non_interactive=False):
    '''PLots the correlation and lets the user cut on the data in an interactive way.

    Parameters
    ----------
    mean_fitted : array like
        The fitted peaks of one column / row correlation
    mean_error_fitted : array like
        The error of the fitted peaks of one column / row correlation
    n_cluster : array like
        The number of hits per column / row
    ref_name : string
        Reference name
    dut_name : string
        DUT name
    title : string
        Plot title
    pre_fit : iterable
        Tuple of offset and slope, e.g. (offset, slope). If proper values are provided, the
        automatic pre-alignment precision can be improved. If None, the data will be fitted.
    non_interactive : bool
        Deactivate user interaction to apply cuts
    '''
    # Global variables needed to manipulate them within a matplotlib QT slot function
    global selected_data
    global fit
    global do_refit
    global error_limit
    global offset_limit
    global left_limit
    global right_limit
    global offset
    global fit
    global fit_fn

    do_refit = True  # True as long as not the Refit button is pressed, needed to signal calling function that the fit is ok or not

    def update_offset(offset_limit_new):  # Function called when offset slider is moved
        global selected_data
        global offset_limit
        offset_limit_tmp = offset_limit
        offset_limit = offset_limit_new
        update_selected_data()
        if np.count_nonzero(selected_data) < 2:
            logging.warning("Offset limit: less than 2 data points are left")
            offset_limit = offset_limit_tmp
            update_selected_data()
        update_plot()

    def update_error(error_limit_new):  # Function called when error slider is moved
        global selected_data
        global error_limit
        error_limit_tmp = error_limit
        error_limit = error_limit_new / 10.0
        update_selected_data()
        if np.count_nonzero(selected_data) < 2:
            logging.warning("Error limit: less than 2 data points are left")
            error_limit = error_limit_tmp
            update_selected_data()
        update_plot()

    def update_left_limit(left_limit_new):  # Function called when left limit slider is moved
        global selected_data
        global left_limit
        left_limit_tmp = left_limit
        left_limit = left_limit_new
        update_selected_data()
        if np.count_nonzero(selected_data) < 2:
            logging.warning("Left limit: less than 2 data points are left")
            left_limit = left_limit_tmp
            update_selected_data()
        update_plot()

    def update_right_limit(right_limit_new):  # Function called when right limit slider is moved
        global selected_data
        global right_limit
        right_limit_tmp = right_limit
        right_limit = right_limit_new
        update_selected_data()
        if np.count_nonzero(selected_data) < 2:
            logging.warning("Right limit: less than 2 data points are left")
            right_limit = right_limit_tmp
            update_selected_data()
        update_plot()

    def update_selected_data():
        global selected_data
        global offset
        global error_limit
        global offset_limit
        global left_limit
        global right_limit
        #init_selected_data()
        selected_data = initial_select.copy()
#         selected_data &= np.logical_and(np.logical_and(np.logical_and(np.abs(offset) <= offset_limit, np.abs(mean_error_fitted) <= error_limit), x >= left_limit), x <= right_limit)
        selected_data[selected_data] = (np.abs(offset[selected_data]) <= offset_limit)
        selected_data[selected_data] = (np.abs(mean_error_fitted[selected_data]) <= error_limit)
        selected_data &= (x >= left_limit)
        selected_data &= (x <= right_limit)

    def update_auto(event):  # Function called when auto button is pressed
        global selected_data
        global offset
        global error_limit
        global offset_limit
        global left_limit
        global right_limit

        selected_data_tmp = selected_data.copy()
        error_limit_tmp = error_limit
        offset_limit_tmp = offset_limit
        left_limit_tmp = left_limit
        right_limit_tmp = right_limit

        # This function automatically applies cuts according to these percentiles
        n_hit_percentile = 1
        mean_error_percentile = 95
        offset_percentile = 99

        error_median = np.nanmedian(mean_error_fitted[selected_data])
        error_std = np.nanstd(mean_error_fitted[selected_data])
        error_limit = max(error_median + error_std * 2, np.percentile(np.abs(mean_error_fitted[selected_data]), mean_error_percentile))
        offset_median = np.nanmedian(offset[selected_data])
        offset_std = np.nanstd(offset[selected_data])
        offset_limit = max(offset_median + offset_std * 2, np.percentile(np.abs(offset[selected_data]), offset_percentile))  # Do not cut too much on the offset, it depends on the fit that might be off

        n_hit_cut = np.percentile(n_cluster[selected_data], n_hit_percentile)  # Cut off low/high % of the hits
        n_hit_cut_index = np.zeros_like(n_cluster, dtype=np.bool)
        n_hit_cut_index |= (n_cluster <= n_hit_cut)
        n_hit_cut_index[selected_data] |= (np.abs(offset[selected_data]) > offset_limit)
        n_hit_cut_index[~np.isfinite(offset)] = 1
        n_hit_cut_index[selected_data] |= (np.abs(mean_error_fitted[selected_data]) > error_limit)
        n_hit_cut_index[~np.isfinite(mean_error_fitted)] = 1
        n_hit_cut_index = np.where(n_hit_cut_index == 1)[0]
        left_index = np.where(x <= left_limit)[0][-1]
        right_index = np.where(x >= right_limit)[0][0]

        # update plot and selected data
        n_hit_cut_index = n_hit_cut_index[n_hit_cut_index >= left_index]
        n_hit_cut_index = n_hit_cut_index[n_hit_cut_index <= right_index]
        if not np.any(n_hit_cut_index == left_index):
            n_hit_cut_index = np.r_[[left_index], n_hit_cut_index]
        if not np.any(n_hit_cut_index == right_index):
            n_hit_cut_index = np.r_[n_hit_cut_index, [right_index]]

        if np.any(n_hit_cut_index.shape):  # If data has no anomalies n_hit_cut_index is empty
            def consecutive(data, max_stepsize=1):  # Returns group of consecutive increasing values
                return np.split(data, np.where(np.diff(data) > max_stepsize)[0] + 1)
            cons = consecutive(n_hit_cut_index, max_stepsize=10)
            left_cut = left_index if cons[0].shape[0] == 1 else cons[0][-1]
            right_cut = right_index if cons[-1].shape[0] == 1 else cons[-1][0] - 1
            left_limit = x[left_cut]
            right_limit = x[right_cut]

        update_selected_data()
        if np.count_nonzero(selected_data) < 2:
            logging.info("Automatic pre-alignment: less than 2 data points are left, discard new limits")
            selected_data = selected_data_tmp
            error_limit = error_limit_tmp
            offset_limit = offset_limit_tmp
            left_limit = left_limit_tmp
            right_limit = right_limit_tmp
            update_selected_data()
        fit_data()
        if not non_interactive:
            offset_limit = np.max(np.abs(offset[selected_data]))
            error_limit = np.max(np.abs(mean_error_fitted[selected_data]))
            update_plot()

    def update_plot():  # Replot correlation data with new selection
        global selected_data
        global offset
        global error_limit
        global offset_limit
        global left_limit
        global right_limit
        if np.count_nonzero(selected_data) > 1:
            left_index = np.where(x <= left_limit)[0][-1]
            right_index = np.where(x >= right_limit)[0][0]
            # set ymax to maximum of either error or offset within the left and right limit, and increase by 10%
            ax2.set_ylim(ymax=max(np.max(np.abs(mean_error_fitted[selected_data])) * 10.0, np.max(np.abs(offset[selected_data])) * 1.0) * 1.1)
            offset_limit_plot.set_ydata([offset_limit, offset_limit])
            error_limit_plot.set_ydata([error_limit * 10.0, error_limit * 10.0])
            left_limit_plot.set_xdata([left_limit, left_limit])
            right_limit_plot.set_xdata([right_limit, right_limit])
            # setting calculated offset data
            offset_plot.set_data(x[initial_select], np.abs(offset[initial_select]))
            # update offset slider
            offset_range = offset[left_index:right_index]
            offset_range = offset_range[np.isfinite(offset_range)]
            offset_max = np.max(np.abs(offset_range))
            ax_offset.set_xlim(xmax=offset_max)
            offset_slider.valmax = offset_max
            cid = offset_slider.cnt - 1
            offset_slider.disconnect(cid)
            offset_slider.set_val(offset_limit)
            offset_slider.on_changed(update_offset)
            # update error slider
            error_range = mean_error_fitted[left_index:right_index]
            error_range = error_range[np.isfinite(error_range)]
            error_max = np.max(np.abs(error_range)) * 10.0
            ax_error.set_xlim(xmax=error_max)
            error_slider.valmax = error_max
            cid = error_slider.cnt - 1
            error_slider.disconnect(cid)
            error_slider.set_val(error_limit * 10.0)
            error_slider.on_changed(update_error)
            # update left slider
            cid = left_slider.cnt - 1
            left_slider.disconnect(cid)
            left_slider.set_val(left_limit)
            left_slider.on_changed(update_left_limit)
            # update right slider
            cid = right_slider.cnt - 1
            right_slider.disconnect(cid)
            right_slider.set_val(right_limit)
            right_slider.on_changed(update_right_limit)
            # setting calculated fit line
            line_plot.set_data(x, fit_fn(x))
        else:
            if non_interactive:
                raise RuntimeError('Coarse alignment in non-interactive mode failed. Rerun with less iterations or in interactive mode!')
            else:
                logging.info('Cuts are too tight. Not enough data to fit')

    def init_selected_data():
        global selected_data
        selected_data = np.ones_like(mean_fitted, dtype=np.bool)
        selected_data &= np.isfinite(mean_fitted)
        selected_data &= np.isfinite(mean_error_fitted)

    def finish(event):  # Fit result is ok
        global do_refit
        do_refit = False  # Set to signal that no refit is required anymore
        update_selected_data()
        fit_data()
        plt.close()  # Close the plot to let the program continue (blocking)

    def refit(event):
        fit_data()
        update_plot()

    def fit_data():
        global selected_data
        global offset
        global fit
        global fit_fn
        try:
            fit, _ = curve_fit(testbeam_analysis.tools.analysis_utils.linear, x[selected_data], mean_fitted[selected_data])  # Fit straight line
        except TypeError:  # if number of points < 2
            raise RuntimeError('Cannot find any correlation, please check data!')
        fit_fn = np.poly1d(fit[::-1])
        offset = fit_fn(x) - mean_fitted  # Calculate straight line fit offset

    def pre_fit_data():
        global offset
        global fit_fn
        fit_fn = np.poly1d(pre_fit[::-1])
        offset = fit_fn(x) - mean_fitted  # Calculate straight line fit offset

    # Require the gaussian fit error to be reasonable
#     selected_data = (mean_error_fitted < 1e-2)
    # Check for nan's and inf's
    init_selected_data()
    initial_select = selected_data.copy()

    # Calculate and plot selected data + fit + fit offset and gauss fit error
    if pre_fit is None:
        fit_data()
    else:
        pre_fit_data()
    offset_limit = np.max(np.abs(offset[selected_data]))  # Calculate starting offset cut
    error_limit = np.max(np.abs(mean_error_fitted[selected_data]))  # Calculate starting fit error cut
    left_limit = np.min(x[selected_data])  # Calculate starting left cut
    right_limit = np.max(x[selected_data])  # Calculate starting right cut

    # setup plotting
    if non_interactive:
        pass
#         fig = Figure()
#         _ = FigureCanvas(fig)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        # Setup plot
        mean_plot, = ax.plot(x[selected_data], mean_fitted[selected_data], 'o-', label='Data prefit')  # Plot correlation
        line_plot, = ax.plot(x[selected_data], fit_fn(x[selected_data]), '-', label='Line fit')  # Plot line fit
        error_plot, = ax2.plot(x[selected_data], np.abs(mean_error_fitted[selected_data]) * 10.0, 'ro-', label='Error x10')  # Plot gaussian fit error
        offset_plot, = ax2.plot(x[selected_data], np.abs(offset[selected_data]), 'go-', label='Offset')  # Plot line fit offset
        offset_limit_plot = ax2.axhline(offset_limit, linestyle='--', color='g', linewidth=2)  # Plot offset cut as a line
        error_limit_plot = ax2.axhline(error_limit * 10.0, linestyle='--', color='r', linewidth=2)  # Plot error cut as a line
        left_limit_plot = ax2.axvline(left_limit, linestyle='-', color='r', linewidth=2)  # Plot left cut as a vertical line
        right_limit_plot = ax2.axvline(right_limit, linestyle='-', color='r', linewidth=2)  # Plot right cut as a vertical line
        ncluster_plot = ax.bar(x[selected_data], n_cluster[selected_data] / np.max(n_cluster[selected_data]).astype(np.float) * abs(np.diff(ax.get_ylim())[0]), bottom=ax.get_ylim()[0], align='center', alpha=0.1, label='#Cluster [a.u.]', width=np.min(np.diff(x[selected_data])))  # Plot number of hits for each correlation point
        ax.set_ylim(ymin=np.min(mean_fitted[selected_data]), ymax=np.max(mean_fitted[selected_data]))
        ax2.set_ylim(ymin=0.0, ymax=max(np.max(np.abs(mean_error_fitted[selected_data])) * 10.0, np.max(np.abs(offset[selected_data])) * 1.0) * 1.1)
        ax.set_xlim((np.nanmin(x), np.nanmax(x)))
        ax.set_title("Correlation of %s: %s vs. %s" % (prefix + "s", ref_name, dut_name))
        ax.set_xlabel("%s [um]" % dut_name)
        ax.set_ylabel("%s [um]" % ref_name)
        ax2.set_ylabel("Error / Offset")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        ax.grid()
        # Setup interactive sliders/buttons
        ax_offset = plt.axes([0.410, 0.04, 0.2, 0.02], facecolor='white')
        ax_error = plt.axes([0.410, 0.01, 0.2, 0.02], facecolor='white')
        ax_left_limit = plt.axes([0.125, 0.04, 0.2, 0.02], facecolor='white')
        ax_right_limit = plt.axes([0.125, 0.01, 0.2, 0.02], facecolor='white')
        ax_button_auto = plt.axes([0.670, 0.01, 0.06, 0.05], facecolor='black')
        ax_button_refit = plt.axes([0.735, 0.01, 0.08, 0.05], facecolor='black')
        ax_button_ok = plt.axes([0.82, 0.01, 0.08, 0.05], facecolor='black')
        # Create widgets
        offset_slider = Slider(ax=ax_offset, label='Offset limit', valmin=0.0, valmax=offset_limit, valinit=offset_limit, closedmin=True, closedmax=True)
        error_slider = Slider(ax=ax_error, label='Error limit', valmin=0.0, valmax=error_limit * 10.0, valinit=error_limit * 10.0, closedmin=True, closedmax=True)
        left_slider = Slider(ax=ax_left_limit, label='Left limit', valmin=left_limit, valmax=right_limit, valinit=left_limit, closedmin=True, closedmax=True)
        right_slider = Slider(ax=ax_right_limit, label='Right limit', valmin=left_limit, valmax=right_limit, valinit=right_limit, closedmin=True, closedmax=True)
        auto_button = Button(ax_button_auto, 'Auto')
        refit_button = Button(ax_button_refit, 'Refit')
        ok_button = Button(ax_button_ok, 'OK')
        # Connect slots
        offset_slider.on_changed(update_offset)
        error_slider.on_changed(update_error)
        left_slider.on_changed(update_left_limit)
        right_slider.on_changed(update_right_limit)
        auto_button.on_clicked(update_auto)
        refit_button.on_clicked(refit)
        ok_button.on_clicked(finish)
        # refit on pressing close button, same effect as OK button
        fig.canvas.mpl_connect(s='close_event', func=finish)

    if non_interactive:
        update_auto(None)
    else:
        plt.get_current_fig_manager().window.showMaximized()  # Plot needs to be large, so maximize
        plt.show()

    return selected_data, fit, do_refit  # Return cut data for further processing


def plot_prealignment_fit(x, mean_fitted, mask, fit_fn, fit, fit_limit, pcov, chi2, mean_error_fitted, n_cluster, n_pixel_ref, n_pixel_dut, pixel_size_ref, pixel_size_dut, ref_name, dut_name, prefix, output_pdf=None, gui=False, figs=None):
    if not output_pdf and not gui:
        return
    fig = Figure()
    _ = FigureCanvas(fig)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.errorbar(x[mask], mean_fitted[mask], yerr=mean_error_fitted[mask], linestyle='', color="blue", fmt='.', label='Correlation', zorder=9)
    ax2.plot(x[mask], mean_error_fitted[mask] * 10.0, linestyle='', color="red", marker='o', label='Error x10', zorder=2)
    ax2.errorbar(x[mask], np.abs(fit_fn(x[mask]) - mean_fitted[mask]), mean_error_fitted[mask], linestyle='', color="lightgreen", marker='o', label='Offset', zorder=3)
    ax2.plot(x, chi2 / 1e5, 'r--', label="Chi$^2$")
    # Plot masked data points, but they should not influence the ylimit
    y_limits = ax2.get_ylim()
    ax1.plot(x[~mask], mean_fitted[~mask], linestyle='', color="darkblue", marker='.', zorder=10)
    ax2.plot(x[~mask], mean_error_fitted[~mask] * 10.0, linestyle='', color="darkred", marker='o', zorder=4)
    ax2.errorbar(x[~mask], np.abs(fit_fn(x[~mask]) - mean_fitted[~mask]), mean_error_fitted[~mask], linestyle='', color="darkgreen", marker='o', zorder=5)
    ax2.set_ylim(y_limits)
    ax2.set_ylim(ymin=0.0)
    ax1.set_ylim((-n_pixel_ref * pixel_size_ref / 2.0, n_pixel_ref * pixel_size_ref / 2.0))
    ax1.set_xlim((-n_pixel_dut * pixel_size_dut / 2.0, n_pixel_dut * pixel_size_dut / 2.0))
    ax2.bar(x, n_cluster / np.max(n_cluster).astype(np.float) * ax2.get_ylim()[1], align='center', alpha=0.5, label='# Cluster [a.u.]', width=np.min(np.diff(x)), zorder=1)  # Plot number of hits for each correlation point
    # Plot again to draw line above the markers
    if len(pcov) > 1:
        fit_legend_entry = 'Fit: $c_0+c_1*x$\n$c_0=%.1e \pm %.1e$\n$c_1=%.1e \pm %.1e$' % (fit[0], np.absolute(pcov[0][0]) ** 0.5, fit[1], np.absolute(pcov[1][1]) ** 0.5)
    else:
        fit_legend_entry = 'Fit: $c_0+x$\n$c_0=%.1e \pm %.1e$' % (fit[0], np.absolute(pcov[0][0]) ** 0.5)
    ax1.plot(x, fit_fn(x), linestyle='-', color="darkorange", label=fit_legend_entry, zorder=11)
    if fit_limit is not None:
        if np.isfinite(fit_limit[0]):
            ax1.axvline(x=fit_limit[0], linewidth=2, color='r', zorder=12)
        if np.isfinite(fit_limit[1]):
            ax1.axvline(x=fit_limit[1], linewidth=2, color='r', zorder=13)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc=0)
    ax1.set_title("Correlation of %s: %s vs. %s" % (prefix + "s", ref_name, dut_name))
    ax1.set_xlabel("%s %s [um]" % (prefix.title(), dut_name))
    ax1.set_ylabel("%s %s [um]" % (prefix.title(), ref_name))
    ax2.set_ylabel("Error / Offset [a.u.]")
    ax1.grid()
    # put ax in front of ax2
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)  # hide the canvas

    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)


def plot_hough(x, data, accumulator, offset, slope, theta_edges, rho_edges, n_pixel_ref, n_pixel_dut, pixel_size_ref, pixel_size_dut, ref_name, dut_name, prefix, output_pdf=None, gui=False, figs=None):
    if not output_pdf and not gui:
        return
    capital_prefix = prefix
    if prefix:
        capital_prefix = prefix.title()
    if pixel_size_dut and pixel_size_ref:
        aspect = pixel_size_ref / pixel_size_dut
    else:
        aspect = "auto"

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap('viridis')
    cmap.set_bad('w')
    ax.imshow(accumulator, interpolation="none", origin="lower", aspect="auto", cmap=cmap, extent=[np.rad2deg(theta_edges[0]), np.rad2deg(theta_edges[-1]), rho_edges[0], rho_edges[-1]])
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_title("Accumulator plot of %s correlations: %s vs. %s" % (prefix, ref_name, dut_name))
    ax.set_xlabel(r'$\theta$ [degree]')
    ax.set_ylabel(r'$\rho$ [um]')

    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    fit_legend_entry = 'Hough: $c_0+c_1*x$\n$c_0=%.1e$\n$c_1=%.1e$' % (offset, slope)
    ax.plot(x, testbeam_analysis.tools.analysis_utils.linear(x, offset, slope), linestyle=':', color="darkorange", label=fit_legend_entry)
    ax.imshow(data, interpolation="none", origin="lower", aspect=aspect, cmap='Greys')
    ax.set_title("Correlation of %s: %s vs. %s" % (prefix + "s", ref_name, dut_name))
    ax.set_xlabel("%s %s" % (capital_prefix, dut_name))
    ax.set_ylabel("%s %s" % (capital_prefix, ref_name))
    ax.legend(loc=0)

    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)



def plot_correlations(input_correlation_file, output_pdf_file=None, pixel_size=None, dut_names=None, gui=False):
    '''Takes the correlation histograms and plots them.

    Parameters
    ----------
    input_correlation_file : string
        Filename of the input correlation file.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    '''

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_correlation_file)[0] + '_correlation.pdf'

    if gui:
        figs = []

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:  # if gui, we dont want to safe empty pdf
        with tb.open_file(input_correlation_file, mode="r") as in_file_h5:
            for node in in_file_h5.root:
                try:
                    indices = re.findall(r'\d+', node.name)
                    ref_index = int(indices[0])
                    dut_index = int(indices[1])
                    if "column" in node.name:
                        column = True
                    else:
                        column = False
                    if "background" in node.name:
                        reduced_background = True
                    else:
                        reduced_background = False
                except AttributeError:
                    continue
                data = node[:]

                if np.all(data <= 0):
                    logging.warning('All correlation entries for %s are zero, do not create plots', str(node.name))
                    continue

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                cmap = cm.get_cmap('viridis')
                cmap.set_bad('w')
                norm = colors.LogNorm()
                if pixel_size:
                    aspect = pixel_size[ref_index][0 if column else 1] / (pixel_size[dut_index][0 if column else 1])
                else:
                    aspect = "auto"
                im = ax.imshow(data.T, origin="lower", cmap=cmap, norm=norm, aspect=aspect, interpolation='none')
                dut_name = dut_names[dut_index] if dut_names else ("DUT " + str(dut_index))
                ref_name = dut_names[ref_index] if dut_names else ("DUT " + str(ref_index))
                ax.set_title("Correlation of %s:\n%s vs. %s%s" % ("columns" if "column" in node.title.lower() else "rows", ref_name, dut_name, " (reduced background)" if reduced_background else ""))
                ax.set_xlabel('%s %s' % ("Column" if "column" in node.title.lower() else "Row", dut_name))
                ax.set_ylabel('%s %s' % ("Column" if "column" in node.title.lower() else "Row", ref_name))
                # do not append to axis to preserve aspect ratio
                fig.colorbar(im, cmap=cmap, norm=norm, fraction=0.04, pad=0.05)

                if gui:
                    figs.append(fig)
                else:
                    output_pdf.savefig(fig)

    if gui:
        return figs


def plot_checks(input_corr_file, output_pdf_file=None):
    '''Takes the hit check histograms and plots them.
    Parameters
    ----------
    input_corr_file : string
        Filename of the input correlation file.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from
        the input file.
    '''

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_corr_file)[0] + '.pdf'

    with PdfPages(output_pdf_file) as output_pdf:
        with tb.open_file(input_corr_file, mode="r") as in_file_h5:
            for node in in_file_h5.root:
                data = node[:]

                if np.all(data <= 0):
                    logging.warning('All correlation entries for %s are zero, do not create plots', str(node.name))
                    continue

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                cmap = cm.get_cmap('viridis')
                cmap.set_bad('w')
                norm = colors.LogNorm()
                if len(data.shape) == 1:  # 1 d data (event delta)
                    ax.plot(data, '-')
                if len(data.shape) == 2:  # 2 d data (correlation)
                    im = ax.imshow(data.T, origin="lower", cmap=cmap, norm=norm, aspect="auto", interpolation='none')
                ax.set_title("%s" % node.title)
                # do not append to axis to preserve aspect ratio
                fig.colorbar(im, cmap=cmap, norm=norm, fraction=0.04, pad=0.05)
                output_pdf.savefig(fig)


def plot_events(input_tracks_file, event_range=(0, 100), dut=None, n_tracks=None, max_chi2=None, output_pdf_file=None, gui=False):
    '''Plots the tracks (or track candidates) of the events in the given event range.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    event_range : iterable
        Tuple of start event number and stop event number (excluding), e.g. (0, 100).
    dut : uint
        Take data from DUT with the given number. If None, plot all DUTs
    max_chi2 : uint
        Plot events with track chi2 smaller than the gven number.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    gui: bool
        Determines whether to plot directly onto gui
    n_tracks: uint
        plots all tracks from first to n_tracks, if amount of tracks less than n_tracks, plot all
        if not None, event_range has no effect
    '''

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_events.pdf'

    if gui:
        figs = []

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:  # if gui, we dont want to safe empty pdf
        with tb.open_file(input_tracks_file, "r") as in_file_h5:
            fitted_tracks = False
            try:  # data has track candidates
                _ = in_file_h5.root.TrackCandidates
            except tb.NoSuchNodeError:  # data has fitted tracks
                fitted_tracks = True

            n_duts = sum(['charge' in col for col in table.dtype.names])
            array = table[:]
            tracks = testbeam_analysis.tools.analysis_utils.get_data_in_event_range(array, event_range[0], event_range[-1])
            if tracks.shape[0] == 0:
                logging.warning('No tracks in event selection, cannot plot events!')
                return
            if max_chi2:
                tracks = tracks[tracks['track_chi2'] <= max_chi2]
            mpl.rcParams['legend.fontsize'] = 10
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.gca(projection='3d')
            for track in tracks:
                x, y, z = [], [], []
                for dut_index in range(0, n_duts):
                    if track['x_dut_%d' % dut_index] != 0:  # No hit has x = 0
                        x.append(track['x_dut_%d' % dut_index] * 1.e-3)  # in mm
                        y.append(track['y_dut_%d' % dut_index] * 1.e-3)  # in mm
                        z.append(track['z_dut_%d' % dut_index] * 1.e-3)  # in mm

                if fitted_tracks:
                    offset = np.array((track['offset_0'], track['offset_1'], track['offset_2']))
                    slope = np.array((track['slope_0'], track['slope_1'], track['slope_2']))
                    linepts = offset * 1.e-3 + slope * 1.e-3 * np.mgrid[-150000:150000:2000j][:, np.newaxis]

                n_hits = np.binary_repr(track['hit_flag']).count('1')
                n_very_good_hits = np.binary_repr(track['quality_flag']).count('1')

                if n_hits > 2:  # only plot tracks with more than 2 hits
                    if fitted_tracks:
                        ax.plot(x, y, z, '.' if n_hits == n_very_good_hits else 'o')
                        ax.plot3D(*linepts.T)
                    else:
                        ax.plot(x, y, z, '.-' if n_hits == n_very_good_hits else '.--')

            ax.set_zlim(np.amin(np.array(z)), np.amax(np.array(z)))
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.set_zlabel('z [mm]')
            title_prefix = '%d tracks of %d events' if fitted_tracks else '%d track candidates of %d events'
            title_prefix += ' of DUT %d' % dut if dut else ''
            ax.set_title(title_prefix % (tracks.shape[0], np.unique(tracks['event_number']).shape[0]))

            if gui:
                figs.append(fig)
            else:
                output_pdf.savefig(fig)

    if gui:
        return figs


def plot_track_chi2(input_tracks_file, output_pdf_file=None, dut_names=None, chunk_size=1000000):
    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_chi2.pdf'

    with PdfPages(output_pdf_file) as output_pdf:
        with tb.open_file(input_tracks_file, "r") as in_file_h5:
            for node in in_file_h5.root:
                try:
                    actual_dut = int(re.findall(r'\d+', node.name)[-1])
                    if dut_names is not None:
                        dut_name = dut_names[actual_dut]
                    else:
                        dut_name = "DUT%d" % actual_dut
                except AttributeError:
                    continue

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in testbeam_analysis.tools.analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    chi2s = tracks_chunk["track_chi2"]
                    # Plot track chi2 and angular distribution
                    chi2s = chi2s[np.isfinite(chi2s)]
                    if initialize:
                        initialize = False
                        try:
                            # Plot up to 3 sigma of the chi2 range
                            range_full = [0.0, np.ceil(np.percentile(chi2s, q=99.73))]
                        except IndexError:  # array empty
                            range_full = [0.0, 1.0]
                        hist_full, edges_full = np.histogram(chi2s, range=range_full, bins=200)
                        hist_narrow, edges_narrow = np.histogram(chi2s, range=(0, 2500), bins=200)
                    else:
                        hist_full += np.histogram(chi2s, bins=edges_full)[0]
                        hist_narrow += np.histogram(chi2s, bins=edges_narrow)[0]

                plot_log = np.any(chi2s)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                x = (edges_full[1:] + edges_full[:-1]) / 2.0
                width = (edges_full[1:] - edges_full[:-1])
                ax.bar(x, hist_full, width=width, log=plot_log, align='center')
                ax.grid()
                ax.set_xlim([edges_full[0], edges_full[-1]])
                ax.set_xlabel('Track Chi2 [um*um]')
                ax.set_ylabel('#')
                ax.set_yscale('log')
                ax.set_title('Track Chi2 for %s' % dut_name)
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                x = (edges_narrow[1:] + edges_narrow[:-1]) / 2.0
                width = (edges_narrow[1:] - edges_narrow[:-1])
                ax.bar(x, hist_narrow, width=width, log=plot_log, align='center')
                ax.grid()
                ax.set_xlim([edges_narrow[0], edges_narrow[-1]])
                ax.set_xlabel('Track Chi2 [um*um]')
                ax.set_ylabel('#')
                ax.set_yscale('log')
                ax.set_title('Track Chi2 for %s' % dut_name)
                output_pdf.savefig(fig)


def plot_residuals(histogram, edges, fit, cov, xlabel, title, output_pdf=None, gui=False, figs=None):
    if not output_pdf and not gui:
        return

    for plot_log in [False, True]:  # plot with log y or not
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # Calculate bin centers
        x = (edges[1:] + edges[:-1]) / 2.0
        plot_range = (testbeam_analysis.tools.analysis_utils.get_mean_from_histogram(histogram, x) - 5 * testbeam_analysis.tools.analysis_utils.get_rms_from_histogram(histogram, x),
                      testbeam_analysis.tools.analysis_utils.get_mean_from_histogram(histogram, x) + 5 * testbeam_analysis.tools.analysis_utils.get_rms_from_histogram(histogram, x))
        ax.set_xlim(plot_range)
        ax.grid()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('#')

        if plot_log:
            ax.set_ylim(1, int(ceil(np.amax(histogram) / 10.0)) * 100)

        # fixing bin width in plotting
        width = (edges[1:] - edges[:-1])
        ax.bar(x, histogram, width=width, log=plot_log, align='center')
        if np.any(fit):
            ax.plot([fit[1], fit[1]], [0, ax.get_ylim()[1]], color='red', label='Entries %d\nRMS %.1f um' % (histogram.sum(), testbeam_analysis.tools.analysis_utils.get_rms_from_histogram(histogram, x)))
            gauss_fit_legend_entry = 'Gauss fit: \nA=$%.1f\pm %.1f$\nmu=$%.1f\pm %.1f$\nsigma=$%.1f\pm %.1f$' % (fit[0], np.absolute(cov[0][0] ** 0.5), fit[1], np.absolute(cov[1][1] ** 0.5), np.absolute(fit[2]), np.absolute(cov[2][2] ** 0.5))
            x_gauss = np.arange(np.floor(np.min(edges)), np.ceil(np.max(edges)), step=0.1)
            ax.plot(x_gauss, testbeam_analysis.tools.analysis_utils.gauss(x_gauss, *fit), 'r--', label=gauss_fit_legend_entry, linewidth=2)
            ax.legend(loc=0)
        ax.set_xlim([edges[0], edges[-1]])

        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)


def plot_residuals_vs_position(hist, xedges, yedges, xlabel, ylabel, title, res_mean=None, select=None, fit=None, cov=None, fit_limit=None, output_pdf=None, gui=False, figs=None):
    '''Plot the residuals as a function of the position.
    '''
    if not output_pdf and not gui:
        return

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.imshow(np.ma.masked_equal(hist, 0).T, extent=[xedges[0], xedges[-1] , yedges[0], yedges[-1]], origin='low', aspect='auto', interpolation='none')
    if res_mean is not None:
        res_pos = (xedges[1:] + xedges[:-1]) / 2.0
        if select is None:
            select = np.full_like(res_pos, True, dtype=np.bool)
        ax.plot(res_pos[select], res_mean[select], linestyle='', color="blue", marker='o',  label='Mean residual')
        ax.plot(res_pos[~select], res_mean[~select], linestyle='', color="darkblue", marker='o')
    if fit is not None:
        x_lim = np.array(ax.get_xlim(), dtype=np.float)
        ax.plot(x_lim, testbeam_analysis.tools.analysis_utils.linear(x_lim, *fit), linestyle='-', color="darkorange", linewidth=2, label='Mean residual fit\n%.2e + %.2e x' % (fit[0], fit[1]))
    if fit_limit is not None:
        if np.isfinite(fit_limit[0]):
            ax.axvline(x=fit_limit[0], linewidth=2, color='r')
        if np.isfinite(fit_limit[1]):
            ax.axvline(x=fit_limit[1], linewidth=2, color='r')
    ax.set_xlim([xedges[0], xedges[-1]])
    ax.set_ylim([yedges[0], yedges[-1]])
    ax.legend(loc=0)

    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)


def plot_track_density(input_tracks_file, input_alignment_file, n_pixels, pixel_size, select_duts, use_prealignment, output_pdf_file=None, gui=False, chunk_size=1000000):
    '''Plotting of the track and hit density for selected DUTs.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    input_alignment_file : string
        Filename of the input aligment file.
    n_pixels : iterable of tuples
        One tuple per DUT describing the number of pixels in column, row direction
        e.g. for 2 DUTs: n_pixels = [(80, 336), (80, 336)]
    pixel_size : iterable
        Tuple of the pixel size for column and row for every plane, e.g. [[250, 50], [250, 50]].
    select_duts : iterable
        Selecting DUTs that will be processed.
    use_prealignment : bool
        If True, use pre-alignment; if False, use alignment.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    gui: bool
        Determines whether to plot directly onto gui
    '''
    logging.info('Plotting track density')

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_track_density.pdf'

    if gui:
        figs = []

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        if use_prealignment:
            logging.info('Use pre-alignment data')
            prealignment = in_file_h5.root.PreAlignment[:]
        else:
            logging.info('Use alignment data')
            alignment = in_file_h5.root.Alignment[:]

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            for actual_dut in select_duts:
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT_%d' % actual_dut)

                logging.info('Calculating track density for DUT%d', actual_dut)

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in testbeam_analysis.tools.analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):

                    # Coordinates in global coordinate system (x, y, z)
                    hit_x_local, hit_y_local, hit_z_local = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
                    intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']

                    if use_prealignment:
                        intersection_x_local, intersection_y_local, intersection_z_local = testbeam_analysis.tools.geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                                                  dut_index=actual_dut,
                                                                                                                                                  prealignment=prealignment,
                                                                                                                                                  inverse=True)
                    else:
                        intersection_x_local, intersection_y_local, intersection_z_local = testbeam_analysis.tools.geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                                                  dut_index=actual_dut,
                                                                                                                                                  alignment=alignment,
                                                                                                                                                  inverse=True)

                    if not np.allclose(intersection_z_local, 0.0):
                        raise RuntimeError("Transformation into local coordinate system gives z != 0")

                    if initialize:
                        initialize = False

                        dut_x_size = n_pixels[actual_dut][0] * pixel_size[actual_dut][0]
                        dut_y_size = n_pixels[actual_dut][1] * pixel_size[actual_dut][1]
                        hist_2d_res_x_edges = np.linspace(-dut_x_size / 2.0, dut_x_size / 2.0, n_pixels[actual_dut][0] + 1, endpoint=True)
                        hist_2d_res_y_edges = np.linspace(-dut_y_size / 2.0, dut_y_size / 2.0, n_pixels[actual_dut][1] + 1, endpoint=True)
                        hist_2d_edges = [hist_2d_res_x_edges, hist_2d_res_y_edges]

                        hist_tracks, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=None, statistic='count', bins=hist_2d_edges)
                        hist_hits, _, _, _ = stats.binned_statistic_2d(x=hit_x_local, y=hit_y_local, values=None, statistic='count', bins=hist_2d_edges)
                    else:
                        hist_tracks += stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=None, statistic='count', bins=hist_2d_edges)[0]
                        hist_hits += stats.binned_statistic_2d(x=hit_x_local, y=hit_y_local, values=None, statistic='count', bins=hist_2d_edges)[0]

                # For better readability allow masking of entries that are zero
                hist_tracks = np.ma.masked_equal(hist_tracks, 0)
                hist_hits = np.ma.masked_equal(hist_hits, 0)

                # Get number of hits / tracks
                n_tracks = np.sum(hist_tracks)
                n_hits = np.sum(hist_hits)

                plot_2d_map(hist2d=hist_tracks.T,
                            plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                            title='Track density for DUT%d (%d Tracks)' % (actual_dut, n_tracks),
                            x_axis_title='Column position [um]',
                            y_axis_title='Row position [um]',
                            z_min=0,
                            z_max=None,
                            output_pdf=output_pdf,
                            gui=gui,
                            figs=figs)

                plot_2d_map(hist2d=hist_hits.T,
                            plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                            title='Hit density for DUT%d (%d Hits)' % (actual_dut, n_hits),
                            x_axis_title='Column position [um]',
                            y_axis_title='Row position [um]',
                            z_min=0,
                            z_max=None,
                            output_pdf=output_pdf,
                            gui=gui,
                            figs=figs)

    if gui:
        return figs


def plot_charge_distribution(input_tracks_file, input_alignment_file, n_pixels, pixel_size, select_duts, use_prealignment, output_pdf_file=None, chunk_size=1000000):
    '''Plotting of the charge distribution for selected DUTs.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    input_alignment_file : string
        Filename of the input aligment file.
    n_pixels : iterable of tuples
        One tuple per DUT describing the number of pixels in column, row direction
        e.g. for 2 DUTs: n_pixels = [(80, 336), (80, 336)]
    pixel_size : iterable
        Tuple of the pixel size for column and row for every plane, e.g. [[250, 50], [250, 50]].
    select_duts : iterable
        Selecting DUTs that will be processed.
    use_prealignment : bool
        If True, use pre-alignment; if False, use alignment.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    '''
    logging.info('Plotting mean charge distribution')

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_mean_charge.pdf'

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        if use_prealignment:
            logging.info('Use pre-alignment data')
            prealignment = in_file_h5.root.PreAlignment[:]
        else:
            logging.info('Use alignment data')
            alignment = in_file_h5.root.Alignment[:]

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            for actual_dut in select_duts:
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT_%d' % actual_dut)

                logging.info('Calculating mean charge for DUT%d', actual_dut)

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in testbeam_analysis.tools.analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):

                    # Coordinates in global coordinate system (x, y, z)
                    hit_x_local, hit_y_local, hit_z_local = tracks_chunk['x_dut_%d' % actual_dut], tracks_chunk['y_dut_%d' % actual_dut], tracks_chunk['z_dut_%d' % actual_dut]
                    intersection_x, intersection_y, intersection_z = tracks_chunk['offset_0'], tracks_chunk['offset_1'], tracks_chunk['offset_2']
                    charge = tracks_chunk['charge_dut_%d' % actual_dut]

                    # select tracks and hits with charge
                    select_hits = np.isfinite(charge)

                    hit_x_local, hit_y_local, hit_z_local = hit_x_local[select_hits], hit_y_local[select_hits], hit_z_local[select_hits]
                    intersection_x, intersection_y, intersection_z = intersection_x[select_hits], intersection_y[select_hits], intersection_z[select_hits]
                    charge = charge[select_hits]

                    if use_prealignment:
                        intersection_x_local, intersection_y_local, intersection_z_local = testbeam_analysis.tools.geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                                                  dut_index=actual_dut,
                                                                                                                                                  prealignment=prealignment,
                                                                                                                                                  inverse=True)
                    else:
                        intersection_x_local, intersection_y_local, intersection_z_local = testbeam_analysis.tools.geometry_utils.apply_alignment(intersection_x, intersection_y, intersection_z,
                                                                                                                                                  dut_index=actual_dut,
                                                                                                                                                  alignment=alignment,
                                                                                                                                                  inverse=True)

                    if not np.allclose(intersection_z_local, 0.0):
                        raise RuntimeError("Transformation into local coordinate system gives z != 0")

                    if initialize:
                        initialize = False

                        dut_x_size = n_pixels[actual_dut][0] * pixel_size[actual_dut][0]
                        dut_y_size = n_pixels[actual_dut][1] * pixel_size[actual_dut][1]
                        hist_2d_res_x_edges = np.linspace(-dut_x_size / 2.0, dut_x_size / 2.0, n_pixels[actual_dut][0] + 1, endpoint=True)
                        hist_2d_res_y_edges = np.linspace(-dut_y_size / 2.0, dut_y_size / 2.0, n_pixels[actual_dut][1] + 1, endpoint=True)
                        hist_2d_edges = [hist_2d_res_x_edges, hist_2d_res_y_edges]

                        stat_track_charge_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=charge, statistic='mean', bins=hist_2d_edges)
                        stat_track_charge_hist = np.nan_to_num(stat_track_charge_hist)
                        count_track_charge_hist, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=charge, statistic='count', bins=hist_2d_edges)

                        stat_hits_charge_hist, _, _, _ = stats.binned_statistic_2d(x=hit_x_local, y=hit_y_local, values=charge, statistic='mean', bins=hist_2d_edges)
                        stat_hits_charge_hist = np.nan_to_num(stat_hits_charge_hist)
                        count_hits_charge_hist, _, _, _ = stats.binned_statistic_2d(x=hit_x_local, y=hit_y_local, values=charge, statistic='count', bins=hist_2d_edges)
                    else:
                        stat_track_charge_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=charge, statistic='mean', bins=hist_2d_edges)
                        stat_track_charge_hist_tmp = np.nan_to_num(stat_track_charge_hist_tmp)
                        count_track_charge_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=intersection_x_local, y=intersection_y_local, values=charge, statistic='count', bins=hist_2d_edges)
                        stat_track_charge_hist, count_track_charge_hist = np.ma.average(a=np.stack([stat_track_charge_hist, stat_track_charge_hist_tmp]), axis=0, weights=np.stack([count_track_charge_hist, count_track_charge_hist_tmp]), returned=True)

                        stat_hits_charge_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=hit_x_local, y=hit_y_local, values=charge, statistic='mean', bins=hist_2d_edges)
                        stat_hits_charge_hist_tmp = np.nan_to_num(stat_hits_charge_hist_tmp)
                        count_hits_charge_hist_tmp, _, _, _ = stats.binned_statistic_2d(x=hit_x_local, y=hit_y_local, values=charge, statistic='count', bins=hist_2d_edges)
                        stat_hits_charge_hist, count_hits_charge_hist = np.ma.average(a=np.stack([stat_hits_charge_hist, stat_hits_charge_hist_tmp]), axis=0, weights=np.stack([count_hits_charge_hist, count_hits_charge_hist_tmp]), returned=True)

                # For better readability allow masking of entries that are zero
                stat_track_charge_hist = np.ma.masked_where(count_track_charge_hist == 0, stat_track_charge_hist)
                stat_hits_charge_hist = np.ma.masked_where(count_hits_charge_hist == 0, stat_hits_charge_hist)

                # Get number of hits / tracks
                n_tracks = np.sum(count_track_charge_hist)
                n_hits = np.sum(count_hits_charge_hist)

                charge_max_tracks = 2.0 * np.ma.median(stat_track_charge_hist)
                charge_max_hits = 2.0 * np.ma.median(stat_hits_charge_hist)

                plot_2d_map(hist2d=stat_track_charge_hist.T,
                            plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                            title='Mean charge of tracks for DUT%d (%d Tracks)' % (actual_dut, n_tracks),
                            x_axis_title='Column position [um]',
                            y_axis_title='Row position [um]',
                            z_min=0,
                            z_max=charge_max_tracks,
                            output_pdf=output_pdf)

                plot_2d_map(hist2d=stat_hits_charge_hist.T,
                            plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                            title='Mean charge of hits for DUT%d (%d Hits)' % (actual_dut, n_hits),
                            x_axis_title='Column position [um]',
                            y_axis_title='Row position [um]',
                            z_min=0,
                            z_max=charge_max_hits,
                            output_pdf=output_pdf)


def efficiency_plots(hit_hist, track_density, track_density_with_DUT_hit, efficiency, actual_dut, minimum_track_density, plot_range, cut_distance, mask_zero=True, output_pdf=None, gui=False, figs=None):
    if not output_pdf and not gui:
        return
    # get number of entries for every histogram
    n_hits_hit_hist = np.count_nonzero(hit_hist)
    n_tracks_track_density = np.count_nonzero(track_density)
    n_tracks_track_density_with_DUT_hit = np.count_nonzero(track_density_with_DUT_hit)
    n_hits_efficiency = np.count_nonzero(efficiency)

    # for better readability allow masking of entries that are zero
    if mask_zero:
        hit_hist = np.ma.array(hit_hist, mask=(hit_hist == 0))
        track_density = np.ma.array(track_density, mask=(track_density == 0))
        track_density_with_DUT_hit = np.ma.array(track_density_with_DUT_hit, mask=(track_density_with_DUT_hit == 0))
        efficiency = np.ma.array(efficiency, mask=(efficiency == 0))

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, hit_hist.T, plot_range, title='Hit density for DUT%d (%d Hits)' % (actual_dut, n_hits_hit_hist), x_axis_title="column [um]", y_axis_title="row [um]")
    fig.tight_layout()

    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, track_density.T, plot_range, title='Track density for DUT%d (%d Tracks)' % (actual_dut, n_tracks_track_density), x_axis_title="column [um]", y_axis_title="row [um]")
    fig.tight_layout()

    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, track_density_with_DUT_hit.T, plot_range, title='Density of tracks with DUT hit for DUT%d (%d Tracks)' % (actual_dut, n_tracks_track_density_with_DUT_hit), x_axis_title="column [um]", y_axis_title="row [um]")
    fig.tight_layout()

    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    if np.any(~efficiency.mask):
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        z_min = np.ma.min(efficiency)
        if z_min == 100.:  # One cannot plot with 0 z axis range
            z_min = 90.
        plot_2d_pixel_hist(fig, ax, efficiency.T, plot_range, title='Efficiency for DUT%d (%d Entries)' % (actual_dut, n_hits_efficiency), x_axis_title="column [um]", y_axis_title="row [um]", z_min=z_min, z_max=100.)
        fig.tight_layout()

        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title('Efficiency per pixel for DUT%d: %1.4f +- %1.4f' % (actual_dut, np.ma.mean(efficiency), np.ma.std(efficiency)))
        ax.set_xlabel('Efficiency [%]')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_xlim([-0.5, 101.5])
        ax.hist(efficiency.ravel()[efficiency.ravel().mask != 1], bins=101, range=(0, 100))  # Histogram not masked pixel efficiency
        fig.tight_layout()
        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)
    else:
        logging.warning('Cannot create efficiency plots, all pixels are masked')


def purity_plots(pure_hit_hist, hit_hist, purity, actual_dut, minimum_hit_density, plot_range, cut_distance, mask_zero=True, output_pdf=None, gui=False, figs=None):
    if not output_pdf and not gui:
        return
    # get number of entries for every histogram
    n_pure_hit_hist = np.count_nonzero(pure_hit_hist)
    n_hits_hit_density = np.sum(hit_hist)
    n_hits_purity = np.count_nonzero(purity)

    # for better readability allow masking of entries that are zero
    if mask_zero:
        pure_hit_hist = np.ma.array(pure_hit_hist, mask=(pure_hit_hist == 0))
        hit_hist = np.ma.array(hit_hist, mask=hit_hist == 0)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, pure_hit_hist.T, plot_range, title='Pure hit density for DUT%d (%d Pure Hits)' % (actual_dut, n_pure_hit_hist), x_axis_title="column [um]", y_axis_title="row [um]")
    fig.tight_layout()
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, hit_hist.T, plot_range, title='Hit density for DUT%d (%d Hits)' % (actual_dut, n_hits_hit_density), x_axis_title="column [um]", y_axis_title="row [um]")
    fig.tight_layout()
    output_pdf.savefig(fig)

    if np.any(~purity.mask):
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        z_min = np.ma.min(purity)
        if z_min == 100.:  # One cannot plot with 0 z axis range
            z_min = 90.
        plot_2d_pixel_hist(fig, ax, purity.T, plot_range, title='Purity for DUT%d (%d Entries)' % (actual_dut, n_hits_purity), x_axis_title="column [um]", y_axis_title="row [um]", z_min=z_min, z_max=100.)
        fig.tight_layout()
        output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title('Purity per pixel for DUT%d: %1.4f +- %1.4f' % (actual_dut, np.ma.mean(purity), np.ma.std(purity)))
        ax.set_xlabel('Purity [%]')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_xlim([-0.5, 101.5])
        ax.hist(purity.ravel()[purity.ravel().mask != 1], bins=101, range=(0, 100))  # Histogram not masked pixel purity
        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)
    else:
        logging.warning('Cannot create purity plots, since all pixels are masked')

def plot_track_angle(input_track_angle_file, output_pdf_file=None, dut_names=None):
    ''' Plotting track slopes.

    Parameters
    ----------
    input_track_angle_file : string
        Filename of the track angle file.
    output_pdf_file: string
        Filename of the output PDF file.
        If None, deduce filename from input track angle file.
    dut_names : iterable of strings
        Names of the DUTs. If None, the DUT index will be used.
    '''
    logging.info('Plotting track angle histogram')
    if output_pdf_file is None:
        output_pdf_file = os.path.splitext(input_track_angle_file)[0] + '.pdf'

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_track_angle_file, mode="r") as in_file_h5:
            for node in in_file_h5.root:
                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                if dut_names is not None:
                    dut_name = dut_names[actual_dut]
                else:
                    dut_name = "DUT%d" % actual_dut
                track_angle_hist = node[:]
                edges = node._v_attrs.edges * 1000  # conversion to mrad
                mean = node._v_attrs.mean * 1000  # conversion to mrad
                sigma = node._v_attrs.sigma * 1000  # conversion to mrad
                amplitude = node._v_attrs.amplitude
                bin_center = (edges[1:] + edges[:-1]) / 2.0
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                # fixing bin width in plotting
                width = (edges[1:] - edges[:-1])
                ax.bar(bin_center, track_angle_hist, label='Angular Distribution%s' % ((" for %s" % dut_name) if dut_name else ""), width=width, color='b', align='center')
                x_gauss = np.arange(np.min(edges), np.max(edges), step=0.00001)
                ax.plot(x_gauss, testbeam_analysis.tools.analysis_utils.gauss(x_gauss, amplitude, mean, sigma), color='r', label='Gauss-Fit:\nMean: %.5f mrad,\nSigma: %.5f mrad' % (mean, sigma))
                ax.set_ylabel('#')
                if 'x' in node.name:
                    direction = "X"
                else:
                    direction = "Y"
                ax.set_title('Angular distribution of fitted tracks for %s in %s-direction (beta)' % (dut_name, direction))
                ax.set_xlabel('Track angle [mrad]')
                ax.legend(loc=1, fancybox=True, frameon=True)
                ax.grid()
                ax.set_xlim(min(edges), max(edges))
                output_pdf.savefig(fig)


def plot_residual_correlation(input_residual_correlation_file, select_duts, pixel_size, output_pdf_file=None, dut_names=None, chunk_size=1000000):
    '''Plotting residual correlation.

    Parameters
    ----------
    input_residual_correlation_file : string
        Filename of the residual correlation file.
    select_duts : iterable
        Selecting DUTs that will be processed.
    output_pdf_file: string
        Filename of the output PDF file.
        If None, deduce filename from input track angle file.
    dut_names : iterable of strings
        Names of the DUTs. If None, the DUT index will be used.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    logging.info('Plotting residual correlation')
    if output_pdf_file is None:
        output_pdf_file = os.path.splitext(input_residual_correlation_file)[0] + '.pdf'

    with PdfPages(output_pdf_file) as output_pdf:
        with tb.open_file(input_residual_correlation_file, mode='r') as in_file_h5:
            fig_col = Figure()
            _ = FigureCanvas(fig_col)
            ax_col = fig_col.add_subplot(111)
            ax_col.set_title("Column residual correlation")
            ax_col.set_ylabel("Correlation of column residuals")
            ax_col.set_xlabel("Track distance [um]")
            fig_row = Figure()
            _ = FigureCanvas(fig_row)
            ax_row = fig_row.add_subplot(111)
            ax_row.set_title("Row residual correlation")
            ax_row.set_ylabel("Correlation of row residuals")
            ax_row.set_xlabel("Track distance [um]")
            for actual_dut in select_duts:
                dut_name = dut_names[actual_dut] if dut_names else ("DUT" + str(actual_dut))
                for direction in ["column", "row"]:
                    correlations = []
                    ref_res_node = in_file_h5.get_node(in_file_h5.root, '%s_residuals_reference_DUT_%d' % (direction.title(), actual_dut))
                    res_node = in_file_h5.get_node(in_file_h5.root, '%s_residuals_DUT_%d' % (direction.title(), actual_dut))
                    edges = res_node.attrs.edges
                    bin_centers = (edges[1:] +  edges[:-1]) / 2.0
                    res_count = []
                    # iterating over bins
                    for index, _ in enumerate(bin_centers):
                        res = None
                        ref_res = None
                        for index_read in np.arange(0, ref_res_node.nrows, step=chunk_size):
                            res_chunk = res_node.read(start=index_read, stop=index_read + chunk_size)[:, index]
                            select = np.isfinite(res_chunk)
                            res_chunk = res_chunk[select]
                            if res is None:
                                res = res_chunk
                            else:
                                res = np.append(res, res_chunk)
                            ref_res_chunk = ref_res_node.read(start=index_read, stop=index_read + chunk_size)[select]
                            if ref_res is None:
                                ref_res = ref_res_chunk
                            else:
                                ref_res = np.append(ref_res, ref_res_chunk)
                        res_count.append(ref_res.shape[0])
                        correlations.append(np.corrcoef(ref_res, res)[0, 1])

                    # fit: see https://doi.org/10.1016/j.nima.2004.08.069
                    def corr_f(x, a, b, c, x_0, p):
                        return a * np.exp(-x / x_0) + b * np.sin(x * 2.0 * np.pi / p + c)
                    fit, pcov = curve_fit(f=corr_f, xdata=bin_centers, ydata=correlations, p0=[0.1, 0.5, np.pi / 2.0, 1.0, pixel_size[actual_dut][0 if direction == "column" else 1]], bounds=[[0.0, 0.0, 0.0, -np.inf, 0.0], [1.0, 1.0, 2 * np.pi, np.inf, bin_centers[-1]]], sigma=np.sqrt(np.array(res_count)), absolute_sigma=False)
                    x = np.linspace(start=bin_centers[0], stop=bin_centers[-1], num=1000, endpoint=True)
                    fitted_correlations = corr_f(x, *fit)

                    fig = Figure()
                    _ = FigureCanvas(fig)
                    ax = fig.add_subplot(111)
                    fit_label = 'Fit: $a*\exp(x/x_0)+b*\sin(2*\pi*x/p+c)$\n$a=%.3f \pm %.3f$\n$b=%.3f \pm %.3f$\n$c=%.1f \pm %.1f$\n$x_0=%.1f \pm %.1f$\n$p=%.1f \pm %.1f$' % (fit[0],
                                                                                                                                                                               np.absolute(pcov[0][0]) ** 0.5,
                                                                                                                                                                               fit[1],
                                                                                                                                                                               np.absolute(pcov[1][1]) ** 0.5,
                                                                                                                                                                               fit[2],
                                                                                                                                                                               np.absolute(pcov[2][2]) ** 0.5,
                                                                                                                                                                               fit[3],
                                                                                                                                                                               np.absolute(pcov[3][3]) ** 0.5,
                                                                                                                                                                               fit[4],
                                                                                                                                                                               np.absolute(pcov[4][4]) ** 0.5)
                    ax.plot(x, fitted_correlations, color='k', label=fit_label)
                    data_label = 'Data'
                    ax.plot(bin_centers, correlations, marker='s', linestyle='None', label=data_label)
#                     yerr = correlations/np.sqrt(np.array(res_count))
#                     ax.errorbar(bin_centers, correlations, yerr=yerr, marker='s', linestyle='None')
                    ax.set_title("%s residual correlation of %s" % (direction.title(), dut_name))
                    ax.set_ylabel("Correlation of %s residuals" % (direction,))
                    ax.set_xlabel("Track distance [um]")
                    ax.legend(loc="upper right")
                    output_pdf.savefig(fig)

                    if direction == "column":
                        ax_col.plot(bin_centers, correlations, label='%s' % (dut_name,), marker='s', linestyle='None')
                    else:
                        ax_row.plot(bin_centers, correlations, label='%s' % (dut_name,), marker='s', linestyle='None')

            ax_col.legend(loc="upper right")
            ax_row.legend(loc="upper right")
            output_pdf.savefig(fig_col)
            output_pdf.savefig(fig_row)
