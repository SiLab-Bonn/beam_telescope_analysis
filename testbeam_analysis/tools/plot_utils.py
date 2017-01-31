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

import testbeam_analysis.tools.analysis_utils

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")  # Plot backend error not important


def plot_2d_pixel_hist(fig, ax, hist2d, plot_range, title=None, x_axis_title=None, y_axis_title=None, z_min=0, z_max=None):
    extent = [0.5, plot_range[0] + .5, plot_range[1] + .5, 0.5]
    if z_max is None:
        if hist2d.all() is np.ma.masked:  # check if masked array is fully masked
            z_max = 1
        else:
            z_max = ceil(hist2d.max())
    bounds = np.linspace(start=z_min, stop=z_max, num=255, endpoint=True)
    cmap = cm.get_cmap('viridis')
    cmap.set_bad('w')
    im = ax.imshow(hist2d, interpolation='none', aspect="auto", extent=extent, cmap=cmap, clim=(0, z_max))
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
        ax1.axvline(x=fit_limit[0], linewidth=2, color='r', zorder=12)
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
                    dut_idx = int(indices[0])
                    ref_idx = int(indices[1])
                    if "column" in node.name.lower():
                        column = True
                    else:
                        column = False
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
                    aspect = pixel_size[ref_idx][0 if column else 1] / (pixel_size[dut_idx][0 if column else 1])
                else:
                    aspect = "auto"
                im = ax.imshow(data.T, origin="lower", cmap=cmap, norm=norm, aspect=aspect, interpolation='none')
                dut_name = dut_names[dut_idx] if dut_names else ("DUT " + str(dut_idx))
                ref_name = dut_names[ref_idx] if dut_names else ("DUT " + str(ref_idx))
                ax.set_title("Correlation of %s: %s vs. %s" % ("columns" if "column" in node.title.lower() else "rows", ref_name, dut_name))
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

            for node in in_file_h5.root:

                # If a DUT is given skip others
                if dut and fitted_tracks:
                    if node.name != 'Tracks_DUT_%d' % dut:
                        continue

                table = node[:]

                n_duts = sum(['charge' in col for col in table.dtype.names])
                array = table[:]
                if n_tracks is not None:
                    index_stop = 0
                    event_start = array['event_number'][0]
                    while index_stop <= n_tracks:
                        try:
                            event_stop = array['event_number'][index_stop]
                            index_stop += 1
                        except IndexError:
                            if index_stop:
                                index_stop -= 1
                            break
                    tracks = testbeam_analysis.tools.analysis_utils.get_data_in_event_range(array, event_start, event_stop)
                else:
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

                    n_hits = bin(track['track_quality'] & 0xFF).count('1')
                    n_very_good_hits = bin(track['track_quality'] & 0xFF0000).count('1')

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


def plot_track_chi2(chi2s, fit_dut, output_pdf=None):
    if not output_pdf:
        return
    # Plot track chi2 and angular distribution
    chi2s = chi2s[np.isfinite(chi2s)]
    try:
        # Plot up to 3 sigma of the chi2 range
        x_limits = [np.ceil(np.percentile(chi2s, q=99.73))]
    except IndexError:  # array empty
        x_limits = [1]
    x_limits.append(2500)  # plot fixed narrow range
    for x_limit in x_limits:
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.hist(chi2s, bins=100, range=(0, x_limit))
        ax.set_xlim(0, x_limit)
        ax.grid()
        ax.set_xlabel('Track Chi2 [um*um]')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_title('Track Chi2 for DUT%d tracks' % fit_dut)
        output_pdf.savefig(fig)


def plot_residuals(histogram, edges, fit, fit_errors, x_label, title, output_pdf=None, gui=False, figs=None):
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
        ax.set_xlabel(x_label)
        ax.set_ylabel('#')

        if plot_log:
            ax.set_ylim(1, int(ceil(np.amax(histogram) / 10.0)) * 100)

        ax.bar(x, histogram, log=plot_log, align='center')
        if np.any(fit):
            ax.plot([fit[1], fit[1]], [0, ax.get_ylim()[1]], color='red', label='Entries %d\nRMS %.1f um' % (histogram.sum(), testbeam_analysis.tools.analysis_utils.get_rms_from_histogram(histogram, x)))
            gauss_fit_legend_entry = 'Gauss fit: \nA=$%.1f\pm %.1f$\nmu=$%.1f\pm %.1f$\nsigma=$%.1f\pm %.1f$' % (fit[0], np.absolute(fit_errors[0][0] ** 0.5), fit[1], np.absolute(fit_errors[1][1] ** 0.5), np.absolute(fit[2]), np.absolute(fit_errors[2][2] ** 0.5))
            x_gauss = np.arange(np.floor(np.min(edges)), np.ceil(np.max(edges)), step=0.1)
            ax.plot(x_gauss, testbeam_analysis.tools.analysis_utils.gauss(x_gauss, *fit), 'r--', label=gauss_fit_legend_entry, linewidth=2)
            ax.legend(loc=0)
        ax.set_xlim([edges[0], edges[-1]])

        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)


def plot_residuals_vs_position(hist, xedges, yedges, xlabel, ylabel, res_mean=None, res_pos=None, selection=None, title=None, fit=None, cov=None, output_pdf=None, gui=False, figs=None):
    '''Plot the residuals as a function of the position.
    '''
    if not output_pdf and not gui:
        return

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.grid()
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Residual vs. Position")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.imshow(np.ma.masked_equal(hist, 0).T, extent=[xedges[0], xedges[-1] , yedges[0], yedges[-1]], origin='low', aspect='auto', interpolation='none')
    if res_mean is not None and res_pos is not None:
        if selection is None:
            selection = np.full_like(res_pos, True, dtype=np.bool)
        ax.plot(res_pos[selection], res_mean[selection], linestyle='', color="blue", marker='o',  label='Mean residual')
        ax.plot(res_pos[~selection], res_mean[~selection], linestyle='', color="darkblue", marker='o')
    if fit is not None:
        x_lim = np.array(ax.get_xlim(), dtype=np.float)
        ax.plot(x_lim, testbeam_analysis.tools.analysis_utils.linear(x_lim, *fit), linestyle='-', color="darkorange", linewidth=2, label='Mean residual fit\n%.2e + %.2e x' % (fit[0], fit[1]))
    ax.set_xlim([xedges[0], xedges[-1]])
    ax.set_ylim([yedges[0], yedges[-1]])
    ax.legend(loc=0)

    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)


def plot_track_density(input_tracks_file, z_positions, dim_x, dim_y, pixel_size, mask_zero=True, use_duts=None, max_chi2=None, output_pdf_file=None, gui=False):
    '''Takes the tracks and calculates the track density projected on selected DUTs.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    z_positions : iterable
        Iterable with z-positions of all DUTs.
    dim_x, dim_y : int FIXME: iterables
        Number of pixels FIXME: Iterable of number of pixels in each dimension for each DUT.
    pixel_size : iterable
        Tuple of the pixel size for column and row for every plane, e.g. [[250, 50], [250, 50]].
    mask_zero : bool
        Mask heatmap entries = 0 for plotting.
    use_duts : iterable
        DUTs that will be used for plotting. If None, all DUTs are used.
    max_chi2 : uint
        Plot events with track chi2 smaller than the gven number.
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

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:  # if gui, we dont want to safe empty pdf
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            plot_ref_dut = False
            dimensions = []

            for index, node in enumerate(in_file_h5.root):
                # Bins define (virtual) pixel size for histogramming
                bin_x, bin_y = dim_x[index], dim_y[index]

                # Calculate dimensions in um for every plane
                dimensions.append((dim_x[index] * pixel_size[index][0], dim_y[index] * pixel_size[index][1]))

                plot_range = (dimensions[index][0], dimensions[index][1])

                actual_dut = int(re.findall(r'\d+', node.name)[-1])
                if use_duts and actual_dut not in use_duts:
                    continue
                logging.info('Plot track density for DUT%d', actual_dut)

                track_array = node[:]

                # If set, select only converged fits
                if max_chi2:
                    track_array = track_array[track_array['track_chi2'] <= max_chi2]

                if plot_ref_dut:  # Plot first and last device
                    heatmap_ref_hits, _, _ = np.histogram2d(track_array['x_dut_0'], track_array['y_dut_0'], bins=(bin_x, bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])
                    if mask_zero:
                        heatmap_ref_hits = np.ma.array(heatmap_ref_hits, mask=(heatmap_ref_hits == 0))

                    # Get number of hits in DUT0
                    n_ref_hits = np.count_nonzero(heatmap_ref_hits)

                    fig = Figure()
                    _ = FigureCanvas(fig)
                    ax = fig.add_subplot(111)
                    plot_2d_pixel_hist(fig, ax, heatmap_ref_hits.T, plot_range, title='Hit density for DUT 0 (%d Hits)' % n_ref_hits, x_axis_title="column [um]", y_axis_title="row [um]")
                    fig.tight_layout()
                    output_pdf.savefig(fig)

                    plot_ref_dut = False

                offset, slope = np.column_stack((track_array['offset_0'], track_array['offset_1'], track_array['offset_2'])), np.column_stack((track_array['slope_0'], track_array['slope_1'], track_array['slope_2']))
                intersection = offset + slope / slope[:, 2, np.newaxis] * (z_positions[actual_dut] - offset[:, 2, np.newaxis])  # intersection track with DUT plane

                heatmap, _, _ = np.histogram2d(intersection[:, 0], intersection[:, 1], bins=(bin_x, bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])
                heatmap_hits, _, _ = np.histogram2d(track_array['x_dut_%d' % actual_dut], track_array['y_dut_%d' % actual_dut], bins=(bin_x, bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])

                # For better readability allow masking of entries that are zero
                if mask_zero:
                    heatmap = np.ma.array(heatmap, mask=(heatmap == 0))
                    heatmap_hits = np.ma.array(heatmap_hits, mask=(heatmap_hits == 0))

                # Get number of hits / tracks
                n_hits_heatmap = np.count_nonzero(heatmap)
                n_hits_heatmap_hits = np.count_nonzero(heatmap_hits)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                plot_2d_pixel_hist(fig, ax, heatmap.T, plot_range, title='Track density for DUT%d tracks (%d Tracks)' % (actual_dut, n_hits_heatmap), x_axis_title="column [um]", y_axis_title="row [um]")
                fig.tight_layout()

                if gui:
                    figs.append(fig)
                else:
                    output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                plot_2d_pixel_hist(fig, ax, heatmap_hits.T, plot_range, title='Hit density for DUT%d (%d Hits)' % (actual_dut, n_hits_heatmap_hits), x_axis_title="column [um]", y_axis_title="row [um]")
                fig.tight_layout()

                if gui:
                    figs.append(fig)
                else:
                    output_pdf.savefig(fig)

    if gui:
        return figs


def plot_charge_distribution(input_track_candidates_file, dim_x, dim_y, pixel_size, mask_zero=True, use_duts=None, output_pdf_file=None):
    '''Takes the data and plots the charge distribution for selected DUTs.

    Parameters
    ----------
    input_track_candidates_file : string
        Filename of the input track candidates file.
    dim_x, dim_y : int
        Number of pixels.
    pixel_size : iterable
        Tuple of the pixel size for column and row for every plane, e.g. [[250, 50], [250, 50]].
    mask_zero : bool
        Masking heatmap entries = 0 for plotting.
    use_duts : iterable
        DUTs that will be used for plotting. If None, all DUTs are used.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    '''
    logging.info('Plotting charge distribution')
    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_track_candidates_file)[0] + '_charge_distribution.pdf'

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_track_candidates_file, mode='r') as in_file_h5:
            dimensions = []
            for table_column in in_file_h5.root.TrackCandidates.dtype.names:
                if 'charge' in table_column:
                    actual_dut = int(re.findall(r'\d+', table_column)[-1])
                    index = actual_dut

                    # allow one channel value for all planes or one value for each plane
                    channels_x = [dim_x, ] if not isinstance(dim_x, tuple) else dim_x
                    channels_y = [dim_y, ] if not isinstance(dim_y, tuple) else dim_y
                    if len(channels_x) == 1:  # if one value for all planes
                        n_bin_x, n_bin_y = channels_x, channels_y  # Bins define (virtual) pixel size for histogramming
                        dimensions.append((channels_x * pixel_size[index][0], channels_y * pixel_size[index][1]))  # Calculate dimensions in um for every plane

                    else:  # if one value for each plane
                        n_bin_x, n_bin_y = channels_x[index], channels_y[index]  # Bins define (virtual) pixel size for histogramming
                        dimensions.append((channels_x[index] * pixel_size[index][0], channels_y[index] * pixel_size[index][1]))  # Calculate dimensions in um for every plane

                    plot_range = (dimensions[index][0], dimensions[index][1])

                    if use_duts and actual_dut not in use_duts:
                        continue
                    logging.info('Plot charge distribution for DUT%d', actual_dut)

                    track_array = in_file_h5.root.TrackCandidates[:]

                    n_bins_charge = int(np.amax(track_array['charge_dut_%d' % actual_dut]))

                    x_y_charge = np.column_stack((track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], track_array['charge_dut_%d' % actual_dut]))
                    hit_hist, _, _ = np.histogram2d(track_array['column_dut_%d' % actual_dut], track_array['row_dut_%d' % actual_dut], bins=(n_bin_x, n_bin_y), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5]])
                    charge_distribution = np.histogramdd(x_y_charge, bins=(n_bin_x, n_bin_y, n_bins_charge), range=[[1.5, dimensions[index][0] + 0.5], [1.5, dimensions[index][1] + 0.5], [0, n_bins_charge]])[0]

                    charge_density = np.average(charge_distribution, axis=2, weights=range(0, n_bins_charge)) * sum(range(0, n_bins_charge)) / hit_hist.astype(float)
                    charge_density = np.ma.masked_invalid(charge_density)

                    fig = Figure()
                    _ = FigureCanvas(fig)
                    ax = fig.add_subplot(111)
                    plot_2d_pixel_hist(fig, ax, charge_density.T, plot_range, title='Charge density for DUT%d' % actual_dut, x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=int(np.ma.average(charge_density) * 1.5))
                    fig.tight_layout()
                    output_pdf.savefig(fig)


def plot_track_distances(distance_min_array, distance_max_array, distance_mean_array, actual_dut, plot_range, cut_distance, output_pdf=None):
    if not output_pdf:
        return
    # get number of entries for every histogram
    n_hits_distance_min_array = distance_min_array.count()
    n_hits_distance_max_array = distance_max_array.count()
    n_hits_distance_mean_array = distance_mean_array.count()

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, distance_min_array.T, plot_range, title='Minimal distance for DUT%d (%d Hits)' % (actual_dut, n_hits_distance_min_array), x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=125000)
    fig.tight_layout()
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, distance_max_array.T, plot_range, title='Maximal distance for DUT%d (%d Hits)' % (actual_dut, n_hits_distance_max_array), x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=125000)
    fig.tight_layout()
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, distance_mean_array.T, plot_range, title='Weighted distance for DUT%d (%d Hits)' % (actual_dut, n_hits_distance_mean_array), x_axis_title="column [um]", y_axis_title="row [um]", z_min=0, z_max=cut_distance)
    fig.tight_layout()
    output_pdf.savefig(fig)


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


def plot_track_angle(input_track_angle_file, output_pdf_file=None, dut_names=None):
    ''' Plot track slopes.

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
        output_pdf_file = os.path.splitext(input_track_angle_file)[0] + '_track_angle.pdf'

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
                amp = node._v_attrs.amp
                bin_center = (edges[1:] + edges[:-1]) / 2.0
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.bar(bin_center, track_angle_hist, label=('Angular Distribution for %s' % dut_name), width=(edges[0]-edges[-1])/len(edges), color='b', align='center')
                x_gauss = np.arange(np.min(edges), np.max(edges), step=0.00001)
                ax.plot(x_gauss, testbeam_analysis.tools.analysis_utils.gauss(x_gauss, amp, mean, sigma), color='r', label='Gauss-Fit:\nMean: %.5f mrad,\nSigma: %.5f mrad' % (mean, sigma))
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
