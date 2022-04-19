from __future__ import division

import logging
import re
import os.path
import warnings
from copy import copy
from math import ceil
from math import floor
from itertools import cycle

import numpy as np
import tables as tb
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.artist import setp
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import matplotlib.patches
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib import colors, cm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3d plotting
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial import cKDTree

from beam_telescope_analysis.telescope.telescope import Telescope
import beam_telescope_analysis.tools.analysis_utils
import beam_telescope_analysis.tools.geometry_utils


warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")  # Plot backend error not important


# Python 3 compatibility
try:
    basestring
except NameError:
    basestring = str


cluster_shape_strings = {
    -1: 'others',
    0: r'$\sum$',
    1: u'\u2004\u2596',  # CS 1
    3: u'\u2597\u2009\u2596',  # CS 3, 2 hit cluster, horizontal
    5: u'\u2004\u2596\n\u2004\u2598',  # CS 5, 2 hit cluster, vertical
    6: u'\u259e',  # CS 6, 2 hit cluster
    7: u'\u259b',  # CS 7, 3 hit cluster
    9: u'\u259a',  # CS 9, 2 hit cluster
    11: u'\u259c',  # CS 11, 3 hit cluster
    13: u'\u2599',  # CS 13, 3 hit cluster, L
    14: u'\u259f',  # CS 14, 3 hit cluster
    15: u'\u2597\u2009\u2596\n\u259d\u2009\u2598',  # CS 15, 4 hit cluster
    19: u'\u2004\u2596\u2596\u2596',  # CS 19, 3 hit cluster, horizontal
    95: u'\u2004\u2596\u2596\u2596\n\u2004\u2598\u2598\u2598',  # CS 95, 3x2 hit cluster, horizontal
    261: u'\u2004\u2596\n\u2004\u2596\n\u2004\u2596',  # CS 261, 3 hit cluster, vertical
    783: u'\u2597\u2009\u2596\n\u2597\u2009\u2596\n\u2597\u2009\u2596',  # CS 783, 2x3 hit cluster, vertical
    4959: u'\u2004\u2596\u2596\u2596\n\u2004\u2596\u2596\u2596\n\u2004\u2596\u2596\u2596',  # CS 4959, 3x3 hit cluster, horizontal
}


def plot_2d_map(hist2d, plot_range, title=None, x_axis_title=None, y_axis_title=None, z_min=0, z_max=None, cmap='viridis', aspect='auto', show_colorbar=True, output_pdf=None):
    if not output_pdf:
        return
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ret = plot_2d_pixel_hist(
        fig=fig,
        ax=ax,
        hist2d=hist2d,
        plot_range=plot_range,
        title=title, x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        z_min=z_min,
        z_max=z_max,
        cmap=cmap,
        aspect=aspect,
        show_colorbar=show_colorbar)
    output_pdf.savefig(fig)
    return ret


def plot_2d_pixel_hist(fig, ax, hist2d, plot_range, title=None, x_axis_title=None, y_axis_title=None, z_min=0, z_max=None, cmap='viridis', aspect='auto', show_colorbar=True, plot_projection=False, n_bins_projections=(10, 10)):
    if z_max is None:
        if hist2d.all() is np.ma.masked or np.allclose(0, hist2d):  # check if masked array is fully masked
            z_max = 1
        else:
            z_max = ceil(hist2d.max())
    if isinstance(cmap, basestring):
        cmap = copy(cm.get_cmap(cmap))
        cmap.set_bad('w')
    else:
        cmap = cmap
    # Check if z_max and z_min are not the same (since this causes plotting error)
    if z_max == z_min:
        z_min = 0.0
    im = ax.imshow(hist2d, interpolation='none', origin='lower', aspect=aspect, extent=plot_range, cmap=cmap, clim=(z_min, z_max))
    if plot_projection:  # Plot projection on top of axes
        # Calculate slices
        center_bin_x = int(hist2d.shape[0] / 2.0)
        center_bin_y = int(hist2d.shape[1] / 2.0)
        indices_x = [center_bin_x - int(n_bins_projections[0] / 2.0), center_bin_x + int(n_bins_projections[0] / 2.0)]
        indices_y = [center_bin_y - int(n_bins_projections[1] / 2.0), center_bin_y + int(n_bins_projections[1] / 2.0)]
        slice_x = np.mean(hist2d[:, indices_y[0]:indices_y[1]], axis=1)
        slice_y = np.mean(hist2d[indices_x[0]:indices_x[1], :], axis=0)
        # Plot slices
        divider = make_axes_locatable(ax)
        axHistx = divider.append_axes("top", 1.05, pad=0.2, sharex=ax)
        axHisty = divider.append_axes("right", 1.05, pad=0.2, sharey=ax)
        bin_centers_x = (np.linspace(plot_range[0], plot_range[1], hist2d.shape[0] + 1, endpoint=True)[:-1] + np.linspace(plot_range[0], plot_range[1], hist2d.shape[0] + 1, endpoint=True)[1:]) / 2.0
        bin_centers_y = (np.linspace(plot_range[2], plot_range[3], hist2d.shape[1] + 1, endpoint=True)[:-1] + np.linspace(plot_range[2], plot_range[3], hist2d.shape[1] + 1, endpoint=True)[1:]) / 2.0
        axHistx.plot(bin_centers_x, slice_x, color='grey', ls='--', marker='.', lw=1)
        axHisty.plot(slice_y, bin_centers_y, color='grey', ls='--', marker='.', lw=1)
        ax.axvline(x=bin_centers_x[indices_x[0]], color='red', linestyle='--', lw=1)
        ax.axvline(x=bin_centers_x[indices_x[1]], color='red', linestyle='--', lw=1)
        ax.axhline(y=bin_centers_y[indices_y[0]], color='red', linestyle='--', lw=1)
        ax.axhline(y=bin_centers_y[indices_y[1]], color='red', linestyle='--', lw=1)
        setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(), visible=False)
    if title is not None:
        if plot_projection:
            ax.set_title(title, x=0.5, y=1.5, fontsize=10)
        else:
            ax.set_title(title)
    if x_axis_title is not None:
        ax.set_xlabel(x_axis_title)
    if y_axis_title is not None:
        ax.set_ylabel(y_axis_title)
    if show_colorbar:
        bounds = np.linspace(start=z_min, stop=z_max, num=256, endpoint=True)
        if plot_projection:
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax, boundaries=bounds, ticks=np.linspace(start=z_min, stop=z_max, num=9, endpoint=True), fraction=0.04, pad=0.05)
        else:
            cbar = fig.colorbar(im, boundaries=bounds, ticks=np.linspace(start=z_min, stop=z_max, num=9, endpoint=True), fraction=0.04, pad=0.05)
        return im, cbar
    return im


def add_value_labels(ax, spacing=5):
    ''' Adding labels to the end of each bar in a bar chart.

    Shamelessly stolen from: https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart

    ax : matplotlib.axes.Axes
        The matplotlib object containing the axes of the plot to annotate.
    spacing : int
        The distance between the labels and the bars.
    '''

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with 3 decimal places
        label = "{:.3G}".format(y_value)

        # Create annotation
        ax.annotate(
            label,  # Use 'label' as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(0, space),  # Vertically shift label by 'space'
            textcoords="offset points",  # Interpret 'xytext' as offset in points
            ha='center',  # Horizontally center label
            va=va,  # Vertically align label differently for positive and negative values
            fontsize=5)


def plot_masked_pixels(input_mask_file, pixel_size=None, dut_name=None, output_pdf_file=None):
    with tb.open_file(input_mask_file, mode='r') as input_file_h5:
        try:
            noisy_pixels = np.dstack(np.nonzero(input_file_h5.root.NoisyPixelMask[:].T))[0] + 1  # index starts at 1
            n_noisy_pixels = np.count_nonzero(input_file_h5.root.NoisyPixelMask[:])
        except tb.NodeError:
            noisy_pixels = None
            n_noisy_pixels = 0
        try:
            disabled_pixels = np.dstack(np.nonzero(input_file_h5.root.DisabledPixelMask[:].T))[0] + 1  # index starts at 1
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
        output_pdf_file = os.path.splitext(input_mask_file)[0] + '.pdf'

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        cmap = copy(cm.get_cmap('viridis'))
        cmap.set_bad('w')
        c_max = np.ceil(np.percentile(occupancy, 99.0))

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_title('Occupancy for %s' % (dut_name,))
        ax.imshow(occupancy, aspect=aspect, cmap=cmap, interpolation='none', origin='lower', clim=(0, c_max), extent=[0.5, occupancy.shape[1] + 0.5, 0.5, occupancy.shape[0] + 0.5])
        ax.set_xlim(0.5, occupancy.shape[1] + 0.5)
        ax.set_ylim(0.5, occupancy.shape[0] + 0.5)
        # plot noisy pixels
        if noisy_pixels is not None:
            ax.plot(noisy_pixels[:, 1], noisy_pixels[:, 0], 'ro', mfc='none', mec='c', ms=10, label='Noisy pixels')
            ax.set_title(ax.get_title() + ',\n%d noisy pixels' % (n_noisy_pixels,))
        # plot disabled pixels
        if disabled_pixels is not None:
            ax.plot(disabled_pixels[:, 1], disabled_pixels[:, 0], 'ro', mfc='none', mec='r', ms=10, label='Disabled pixels')
            ax.set_title(ax.get_title() + ',\n%d disabled pixels' % (n_disabled_pixels,))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.legend()
        output_pdf.savefig(fig)

        for masked_hits in [False, True]:
            if masked_hits:
                if noisy_pixels is None and disabled_pixels is None:
                    continue
                else:
                    occupancy_mask = np.zeros_like(occupancy, dtype=np.bool)
                    if noisy_pixels is not None:
                        occupancy_mask[noisy_pixels.T[0] - 1, noisy_pixels.T[1] - 1] = 1
                    if disabled_pixels is not None:
                        occupancy_mask[disabled_pixels.T[0] - 1, disabled_pixels.T[1] - 1] = 1
                    occupancy[occupancy_mask > 0] = 0

            c_max = np.ceil(np.percentile(occupancy, 99.0))  # update maximum after masking pixels

            # Fancy occupancy plot
            fig = Figure()
            _ = FigureCanvas(fig)
            # title of the page
            fig.suptitle('Occupancy for %s%s' % (dut_name, '\n(noisy and disabled pixels masked)' if masked_hits else ''))
            ax = fig.add_subplot(111)
            # ax.set_title('Occupancy for %s' % (dut_name,))
            im = ax.imshow(np.ma.getdata(occupancy), aspect='auto', cmap=cmap, interpolation='none', origin='lower', clim=(0, c_max), extent=[0.5, occupancy.shape[1] + 0.5, 0.5, occupancy.shape[0] + 0.5])
            # np.ma.filled(occupancy, fill_value=0)
            ax.set_xlim(0.5, occupancy.shape[1] + 0.5)
            ax.set_ylim(0.5, occupancy.shape[0] + 0.5)
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")

            # create new axes on the right and on the top of the current axes
            # The first argument of the new_vertical(new_horizontal) method is
            # the height (width) of the axes to be created in inches.
            divider = make_axes_locatable(ax)
            axHistx = divider.append_axes("top", 1.2, pad=0.2, sharex=ax)
            axHisty = divider.append_axes("right", 1.2, pad=0.2, sharey=ax)

            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = fig.colorbar(im, cax=cax, ticks=np.linspace(start=0, stop=c_max, num=9, endpoint=True))
            cb.set_label("#")
            # make some labels invisible
            setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(), visible=False)
            hight = np.ma.sum(np.ma.getdata(occupancy), axis=0)

            axHistx.bar(x=range(1, occupancy.shape[1] + 1), height=hight, align='center', linewidth=0)
            axHistx.set_xlim(0.5, occupancy.shape[1] + 0.5)
            if np.ma.getdata(occupancy).all() is np.ma.masked:
                axHistx.set_ylim((0, 1))
            else:
                x_c_max = np.ceil(np.percentile(hight, 99.0))
                axHistx.set_ylim(0, max(1, x_c_max))
            axHistx.locator_params(axis='y', nbins=3)
            axHistx.ticklabel_format(style='sci', scilimits=(0, 4), axis='y')
            axHistx.set_ylabel('#')
            width = np.ma.sum(np.ma.getdata(occupancy), axis=1)

            axHisty.barh(y=range(1, occupancy.shape[0] + 1), width=width, align='center', linewidth=0)
            axHisty.set_ylim(0.5, occupancy.shape[0] + 0.5)
            if np.ma.getdata(occupancy).all() is np.ma.masked:
                axHisty.set_xlim((0, 1))
            else:
                y_c_max = np.ceil(np.percentile(width, 99.0))
                axHisty.set_xlim(0, max(1, y_c_max))
            axHisty.locator_params(axis='x', nbins=3)
            axHisty.ticklabel_format(style='sci', scilimits=(0, 4), axis='x')
            axHisty.set_xlabel('#')
            output_pdf.savefig(fig)


def plot_tracks_per_event(input_tracks_file, output_pdf_file=None):
    """Plotting tracks per event
    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    chunk_size : int
        Chunk size of the data when reading from file.
    """
    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_tracks_per_event.pdf'

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_tracks_file, mode='r') as input_file_h5:
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
                output_pdf.savefig(fig)


def plot_cluster_hists(telescope_configuration=None, input_cluster_file=None, input_merged_file=None, input_track_candidates_file=None, input_tracks_file=None, dut_name=None, select_duts=None, output_pdf_file=None, chunk_size=1000000):
    '''Plotting cluster histograms.

    Parameters
    ----------
    input_cluster_file : string
        Filename of the input cluster file.
    input_merged_file : string
        Filename of the input merged file.
    input_track_candidates_file : string
        Filename of the input track candiates file.
    input_tracks_file : string
        Filename of the input tracks file.
    dut_name : string
        Name of the DUT. Only needed when input_cluster_file file is used.
        Otherwise dut_name is used from telescope object.
    select_duts : iterable
        Selecting DUTs that will be processed.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    chunk_size : int
        Chunk size of the data when reading from file.
    '''
    if (input_cluster_file and (input_merged_file or input_track_candidates_file or input_tracks_file)) or (not input_cluster_file and not input_merged_file and not input_track_candidates_file and not input_tracks_file):
        raise ValueError("A single input file must be given")

    if input_cluster_file and telescope_configuration:
        raise ValueError("\"dut_name\" parameter must be used instead of \"telescope_configuration\" parameter")

    if input_cluster_file and select_duts:
        raise ValueError("\"select_duts\" parameter not supported when using input_cluster_file")

    if (input_merged_file or input_track_candidates_file or input_tracks_file) and dut_name:
        raise ValueError("\"telescope_configuration\" parameter must be used instead of \"dut_name\" parameter")

    if telescope_configuration is not None:
        telescope = Telescope(telescope_configuration)

    if input_cluster_file:
        input_file = input_cluster_file
        select_duts = [None]
    elif input_merged_file:
        input_file = input_merged_file
        if select_duts is None:
            select_duts = range(len(telescope))
    elif input_track_candidates_file:
        input_file = input_track_candidates_file
        if select_duts is None:
            select_duts = range(len(telescope))
    elif input_tracks_file:
        input_file = input_tracks_file
        if select_duts is None:
            select_duts = range(len(telescope))

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_file)[0] + '_cluster_hists.pdf'

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_file, mode='r') as in_file_h5:
            for actual_dut_index in select_duts:
                if actual_dut_index is None:  # File containing Clusters node
                    node = in_file_h5.get_node(in_file_h5.root, 'Clusters')
                else:
                    try:  # Track file
                        node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)
                    except tb.NoSuchNodeError:
                        try:  # Merged file
                            node = in_file_h5.get_node(in_file_h5.root, 'MergedClusters')
                        except tb.NoSuchNodeError:  # Track candidates file
                            node = in_file_h5.get_node(in_file_h5.root, 'TrackCandidates')

                    actual_dut = telescope[actual_dut_index]
                    dut_name = actual_dut.name
                    logging.info('= Plotting cluster histograms for %s =', dut_name)

                initialize = True  # initialize the histograms
                try:
                    cluster_size_hist = in_file_h5.root.HistClusterSize[:]
                    total_n_hits = np.sum(np.multiply(cluster_size_hist, range(cluster_size_hist.shape[0])))
                    total_n_clusters = np.sum(cluster_size_hist)
                    cluster_shapes_hist = in_file_h5.root.HistClusterShape[:]
                except tb.NoSuchNodeError:
                    total_n_hits = 0
                    total_n_clusters = 0
                    for chunk, _ in beam_telescope_analysis.tools.analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                        if actual_dut_index is None:
                            cluster_n_hits = chunk['n_hits']
                            cluster_shape = chunk['cluster_shape']
                        else:
                            cluster_n_hits = chunk['n_hits_dut_%d' % actual_dut_index]
                            cluster_shape = chunk['cluster_shape_dut_%d' % actual_dut_index]
                            valid_hits = ~np.isnan(chunk['x_dut_%d' % actual_dut_index])
                            # Select events with valid hits in DUT
                            cluster_n_hits = cluster_n_hits[valid_hits]
                            cluster_shape = cluster_shape[valid_hits]

                        max_cluster_size = np.max(cluster_n_hits)
                        total_n_hits += np.sum(cluster_n_hits)
                        total_n_clusters += cluster_n_hits.shape[0]
                        # limit cluster shape histogram to cluster size 4x4
                        edges = np.arange(2**(4 * 4))
                        if initialize:
                            initialize = False

                            cluster_size_hist = np.bincount(cluster_n_hits, minlength=max_cluster_size + 1)
                            cluster_shapes_hist, _ = np.histogram(a=cluster_shape, bins=edges)
                        else:
                            if cluster_size_hist.size - 1 < max_cluster_size:
                                cluster_size_hist.resize(max_cluster_size + 1)
                                cluster_size_hist += np.bincount(cluster_n_hits, minlength=max_cluster_size + 1)
                            else:
                                cluster_size_hist += np.bincount(cluster_n_hits, minlength=cluster_size_hist.size)
                            cluster_shapes_hist += np.histogram(a=cluster_shape, bins=edges)[0]

                x = np.arange(cluster_size_hist.shape[0] - 1) + 1
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.bar(x, cluster_size_hist[1:], align='center')
                ax.set_title('Cluster size distribution%s\n(%d hits, %d clusters)' % (("\nfor %s" % dut_name) if dut_name else "", total_n_hits, total_n_clusters))
                ax.set_xlabel('Cluster size')
                ax.set_ylabel('#')
                ax.grid()
                ax.set_yscale('log')
                ax.set_ylim(ymin=1e-1)
                output_pdf.savefig(fig)

                max_bins = min(10, cluster_size_hist.shape[0] - 1)
                x = np.arange(max_bins) + 1
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.bar(x, cluster_size_hist[1:max_bins + 1], align='center')
                ax.set_title('Cluster size distribution%s\n(%d hits, %d clusters)' % (("\nfor %s" % dut_name) if dut_name else "", total_n_hits, total_n_clusters))
                ax.set_xlabel('Cluster size')
                ax.set_ylabel('#')
                ax.xaxis.set_ticks(x)
                ax.grid()
                ax.set_yscale('linear')
                ax.set_ylim(ymin=0.0)
                output_pdf.savefig(fig)
                ax.autoscale()
                ax.set_yscale('log')
                ax.set_ylim(ymin=1e-1)
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                analyze_cluster_shapes = [1, 3, 5, 6, 9, 13, 14, 7, 11, 19, 261, 15, 95, 783, 4959]
                cluster_shape_hist = cluster_shapes_hist[analyze_cluster_shapes]
                remaining_clusters = total_n_clusters - np.sum(cluster_shape_hist)
                cluster_shape_hist = np.r_[cluster_shape_hist, remaining_clusters]
                analyze_cluster_shapes = np.r_[analyze_cluster_shapes, -1]
                x = np.arange(analyze_cluster_shapes.shape[0])
                ax.bar(x, cluster_shape_hist, align='center')
                ax.xaxis.set_ticks(x)
                fig.subplots_adjust(bottom=0.2)
                ax.set_xticklabels([cluster_shape_strings[i] for i in analyze_cluster_shapes])
                ax.tick_params(axis='x', labelsize=7)
                ax.set_title('Cluster shape distribution%s\n(%d hits, %d clusters)' % (("\nfor %s" % dut_name) if dut_name else "", total_n_hits, total_n_clusters))
                ax.set_xlabel('Cluster shape')
                ax.set_ylabel('#')
                ax.grid()
                ax.set_yscale('linear')
                ax.set_ylim(ymin=0.0)
                output_pdf.savefig(fig)
                ax.autoscale()
                ax.set_yscale('log')
                ax.set_ylim(ymin=1e-1)
                output_pdf.savefig(fig)


def plot_correlations(input_correlation_file, output_pdf_file=None, dut_names=None):
    '''Plotting the correlation histograms.

    Parameters
    ----------
    input_correlation_file : string
        Filename of the input correlation file.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    dut_names : list
        Names of the DUTs. If None, generic DUT names will be used.
    '''

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_correlation_file)[0] + '.pdf'

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_correlation_file, mode="r") as in_file_h5:
            for node in in_file_h5.root:
                try:
                    indices = re.findall(r'\d+', node.name)
                    ref_index = int(indices[0])
                    dut_index = int(indices[1])
                    if "Correlation_x" in node.name:
                        x_direction = True
                    else:
                        x_direction = False
                    if "background" in node.name:
                        reduced_background = True
                    else:
                        reduced_background = False
                except AttributeError:
                    continue
                ref_hist_extent = node.attrs.ref_hist_extent
                dut_hist_extent = node.attrs.dut_hist_extent

                data = node[:]

                if np.all(data <= 0):
                    logging.warning('Cannot create correlation plots: all entries from %s are zero' % str(node.name))
                    continue

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                cmap = copy(cm.get_cmap('viridis'))
                cmap.set_bad('w')
                norm = colors.LogNorm()
                aspect = 1.0  # "auto"
                im = ax.imshow(data.T, interpolation='none', origin="lower", norm=norm, aspect=aspect, cmap=cmap, extent=[dut_hist_extent[0], dut_hist_extent[1], ref_hist_extent[0], ref_hist_extent[1]])
                dut_name = dut_names[dut_index] if dut_names else ("DUT" + str(dut_index))
                ref_name = dut_names[ref_index] if dut_names else ("DUT" + str(ref_index))
                ax.set_title("%s correlation%s:\n%s vs. %s" % ("X" if x_direction else "Y", " (reduced background)" if reduced_background else "", ref_name, dut_name))
                ax.set_xlabel('%s %s [$\mathrm{\mu}$m]' % (dut_name, "x" if x_direction else "y"))
                ax.set_ylabel('%s %s [$\mathrm{\mu}$m]' % (ref_name, "x" if x_direction else "y"))
                # do not append to axis to preserve aspect ratio
                fig.colorbar(im, norm=norm, cmap=cmap, fraction=0.04, pad=0.05)
                output_pdf.savefig(fig)


def plot_hough(dut_pos, data, accumulator, offset, slope, dut_pos_limit, theta_edges, rho_edges, ref_hist_extent, dut_hist_extent, ref_name, dut_name, x_direction, reduce_background, output_pdf=None):
    if not output_pdf:
        return

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    cmap = copy(cm.get_cmap('viridis'))
    cmap.set_bad('w')
    ax.imshow(np.flip(np.flip(accumulator, 0), 1), interpolation="none", origin="lower", aspect="auto", cmap=cmap, extent=[np.rad2deg(theta_edges[0]), np.rad2deg(theta_edges[-1]), rho_edges[0], rho_edges[-1]])
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_title("%s correlation accumulator%s:\n%s vs. %s" % ('X' if x_direction else 'Y', " (reduced background)" if reduce_background else "", ref_name, dut_name))
    ax.set_xlabel(r'$\theta$ [degree]')
    ax.set_ylabel(r'$\rho$ [$\mathrm{\mu}$m]')
    output_pdf.savefig(fig)

    aspect = 1.0  # "auto"
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    fit_legend_entry = 'Hough: $c_0+c_1*x$\n$c_0=%.1e$\n$c_1=%.1e$' % (offset, slope)
    ax.plot(dut_pos, beam_telescope_analysis.tools.analysis_utils.linear(dut_pos, offset, slope), linestyle='-', alpha=0.7, color="darkorange", label=fit_legend_entry)
    ax.axvline(x=dut_pos_limit[0])
    ax.axvline(x=dut_pos_limit[1])
    ax.imshow(data, interpolation="none", origin="lower", aspect=aspect, cmap='Greys', extent=[dut_hist_extent[0], dut_hist_extent[1], ref_hist_extent[0], ref_hist_extent[1]])
    ax.set_title("%s correlation%s:\n%s vs. %s" % ('X' if x_direction else 'Y', " (reduced background)" if reduce_background else "", ref_name, dut_name))
    ax.set_xlabel("%s %s [$\mathrm{\mu}$m]" % (dut_name, "x" if x_direction else "y"))
    ax.set_ylabel("%s %s [$\mathrm{\mu}$m]" % (ref_name, "x" if x_direction else "y"))
    ax.legend(loc=0)
    output_pdf.savefig(fig)


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
                    logging.warning('Cannot create check plots: all entries from %s are zero' % str(node.name))
                    continue

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                cmap = copy(cm.get_cmap('viridis'))
                cmap.set_bad('w')
                norm = colors.LogNorm()
                if len(data.shape) == 1:  # 1 d data (event delta)
                    ax.plot(data, '-')
                if len(data.shape) == 2:  # 2 d data (correlation)
                    im = ax.imshow(data.T, origin="lower", cmap=cmap, norm=norm, aspect="auto", interpolation='none')
                    # do not append to axis to preserve aspect ratio
                    fig.colorbar(im, cmap=cmap, norm=norm, fraction=0.04, pad=0.05)
                ax.set_title("%s" % node.title)
                output_pdf.savefig(fig)


def plot_events(telescope_configuration, input_tracks_file, select_duts, event_range, output_pdf_file=None, show=False, chunk_size=1000000):
    '''Plots the tracks (or track candidates) of the events in the given event range.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_tracks_file : string
        Filename of the input tracks file.
    select_duts : iterable
        Selecting DUTs that will be processed.
    event_range : 2-tuple
        Tuple of start event number and stop event number (excluding), e.g. (0, 100).
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    plot : bool
        If True, show interactive plot.
    '''
    telescope = Telescope(telescope_configuration)
    n_duts = len(telescope)

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_events.pdf'

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            for actual_dut_index in select_duts:
                dut_name = telescope[actual_dut_index].name
                logging.info('== Plotting events for %s ==', dut_name)

                if show:
                    import matplotlib.pyplot as plt
                    fig = plt.gcf()
                else:
                    fig = Figure()
                    _ = FigureCanvas(fig)
                ax = fig.add_subplot(111, projection='3d')
                colors = cycle('bgrcmyk')

                min_z_dut_index = 0
                max_z_dut_index = 0
                for dut_index in range(0, n_duts):
                    actual_dut = telescope[dut_index]
                    plane_x, plane_y, plane_z = actual_dut.index_to_global_position(
                        column=[0.5, 0.5, actual_dut.n_columns + 0.5, actual_dut.n_columns + 0.5],
                        row=[0.5, actual_dut.n_rows + 0.5, 0.5, actual_dut.n_rows + 0.5],
                        # reduce plotting clutter
                        rotation_alpha=0.0,
                        rotation_beta=0.0)
                    plane_x = plane_x * 1.e-3  # in mm
                    plane_y = plane_y * 1.e-3  # in mm
                    plane_z = plane_z * 1.e-3  # in mm
                    plane_x = plane_x.reshape(2, -1)
                    plane_y = plane_y.reshape(2, -1)
                    plane_z = plane_z.reshape(2, -1)
                    ax.plot_surface(plane_x, plane_y, plane_z, color='lightgray', alpha=0.3, linewidth=1.0, zorder=-1)
                    if telescope[min_z_dut_index].translation_z > actual_dut.translation_z:
                        min_z_dut_index = dut_index
                    if telescope[max_z_dut_index].translation_z < actual_dut.translation_z:
                        max_z_dut_index = dut_index

                # Loop over the tracks
                events = 0
                tracks = 0
                tracks_node = in_file_h5.get_node(in_file_h5.root, name='Tracks_DUT%d' % actual_dut_index)
                for tracks_chunk, index in beam_telescope_analysis.tools.analysis_utils.data_aligned_at_events(tracks_node, start_event_number=event_range[0], stop_event_number=event_range[1], fail_on_missing_events=False, chunk_size=chunk_size):
                    events += len(set(tracks_chunk["event_number"]))
                    for track in tracks_chunk:
                        color = next(colors)
                        x, y, z = [], [], []
                        for dut_index in range(0, n_duts):
                            actual_dut = telescope[dut_index]
                            # Coordinates in global coordinate system (x, y, z)
                            hit_x_local, hit_y_local, hit_z_local = track['x_dut_%d' % dut_index], track['y_dut_%d' % dut_index], track['z_dut_%d' % dut_index]
                            hit_x, hit_y, hit_z = actual_dut.local_to_global_position(
                                x=hit_x_local,
                                y=hit_y_local,
                                z=hit_z_local)
                            hit_x = hit_x * 1.e-3  # in mm
                            hit_y = hit_y * 1.e-3  # in mm
                            hit_z = hit_z * 1.e-3  # in mm
                            x.extend(hit_x)
                            y.extend(hit_y)
                            z.extend(hit_z)

                        if np.isfinite(track['offset_x']):
                            tracks += 1
                            track_offset_x, track_offset_y, track_offset_z = telescope[actual_dut_index].local_to_global_position(
                                x=track['offset_x'],
                                y=track['offset_y'],
                                z=track['offset_z'])
                            # convert to mm
                            offset = np.column_stack((track_offset_x, track_offset_y, track_offset_z))
                            track_slope_x, track_slope_y, track_slope_z = telescope[actual_dut_index].local_to_global_position(
                                x=track['slope_x'],
                                y=track['slope_y'],
                                z=track['slope_z'],
                                # manually set translation and rotation to avoid z=0 error check and no translation for the slopes
                                translation_x=0.0,
                                translation_y=0.0,
                                translation_z=0.0,
                                rotation_alpha=telescope[actual_dut_index].rotation_alpha,
                                rotation_beta=telescope[actual_dut_index].rotation_beta,
                                rotation_gamma=telescope[actual_dut_index].rotation_gamma)
                            slope = np.column_stack((track_slope_x, track_slope_y, track_slope_z))
                            offse_min_z_dut = beam_telescope_analysis.tools.geometry_utils.get_line_intersections_with_dut(
                                line_origins=offset,
                                line_directions=slope,
                                translation_x=telescope[min_z_dut_index].translation_x,
                                translation_y=telescope[min_z_dut_index].translation_y,
                                translation_z=telescope[min_z_dut_index].translation_z,
                                rotation_alpha=telescope[min_z_dut_index].rotation_alpha,
                                rotation_beta=telescope[min_z_dut_index].rotation_beta,
                                rotation_gamma=telescope[min_z_dut_index].rotation_gamma) * 1.e-3
                            offset_max_z_dut = beam_telescope_analysis.tools.geometry_utils.get_line_intersections_with_dut(
                                line_origins=offset,
                                line_directions=slope,
                                translation_x=telescope[max_z_dut_index].translation_x,
                                translation_y=telescope[max_z_dut_index].translation_y,
                                translation_z=telescope[max_z_dut_index].translation_z,
                                rotation_alpha=telescope[max_z_dut_index].rotation_alpha,
                                rotation_beta=telescope[max_z_dut_index].rotation_beta,
                                rotation_gamma=telescope[max_z_dut_index].rotation_gamma) * 1.e-3
                            linepts = offse_min_z_dut + slope * np.mgrid[-offse_min_z_dut[0, 2] + 10:-offset_max_z_dut[0, 2] - 10:2j][:, np.newaxis]
                            no_fit = False
                        else:
                            no_fit = True

                        if no_fit:
                            ax.plot(x, y, z, 'x', color=color)
                        else:
                            ax.plot(x, y, z, 's' if track['hit_flag'] == track['quality_flag'] else 'o', color=color)
                            ax.plot3D(*linepts.T, color=color)

                ax.set_xlabel('x [mm]')
                ax.set_ylabel('y [mm]')
                ax.set_zlabel('z [mm]')
                ax.set_title('%d tracks in %d events for %s' % (tracks, events, dut_name))

                if show:
                    plt.show()
                else:
                    output_pdf.savefig(fig)


def plot_fit_tracks_statistics(telescope, fit_duts, chunk_indices, chunk_stats, dut_stats, output_pdf=None):
    '''Plot the residuals as a function of the position.
    '''
    if not output_pdf:
        return

    for actual_dut_index, actual_dut in enumerate(telescope):
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.plot(chunk_indices, chunk_stats, label="Share of tracks fulfilling hit requirements")
        curr_stats = []
        if dut_stats and len(dut_stats[-1]) > actual_dut_index and dut_stats[-1][actual_dut_index]:
            for item in dut_stats:
                curr_stats.append(item[actual_dut_index])
        stat_labels = ["... of which the share of DUT with hit", "   ... of which the share of tracks with quality flag set", "   ... of which the share of tracks with isolated tracks flag set", "   ... of which the share of tracks with isolated hits flag set", "Share of tracks fulfilling hit requirements and\nhave quality/isolated track/isolated hit flag set"]
        for index, stat in enumerate(zip(*curr_stats)):
            for i in range(3)[::-1]:
                if np.all(np.less_equal(stat, 10**-i)):
                    factor = 10**i
                    label_extent = r' ($\times %d$)' % factor
                    break
            ax.plot(chunk_indices, np.array(stat) * factor, label=stat_labels[index] + ("" if factor == 1 else label_extent))
        ax.set_title("Fit statistics for %s\nFit DUTs: %s" % (actual_dut.name, ', '.join([telescope[dut_index].name for dut_index in fit_duts])))
        ax.set_xlim(left=0.0)
        ax.set_ylim([0.0, 1.1])
        ax.set_xlabel("Index")
        ax.set_ylabel("#")
        ax.legend(loc=1, prop={'size': 6})
        output_pdf.savefig(fig)


def plot_track_chi2(input_tracks_file, output_pdf_file=None, dut_names=None, chunk_size=1000000):
    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_chi2.pdf'

    with PdfPages(output_pdf_file) as output_pdf:
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            for node in in_file_h5.root:
                try:
                    actual_dut_index = int(re.findall(r'\d+', node.name)[-1])
                    if dut_names is not None:
                        dut_name = dut_names[actual_dut_index]
                    else:
                        dut_name = "DUT%d" % actual_dut_index
                except AttributeError:
                    continue

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in beam_telescope_analysis.tools.analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):
                    chi2s = tracks_chunk["track_chi2"]
                    chi2s_red = tracks_chunk["track_chi_red"]  # reduced chi2
                    track_pvalue = tracks_chunk["track_chi_prob"]  # pvalue
                    # Plot track chi2 and angular distribution
                    chi2s = chi2s[np.isfinite(chi2s)]
                    if initialize:
                        initialize = False
                        try:
                            # Plot up to 3 sigma of the chi2 range
                            range_full = [0.0, np.ceil(np.percentile(chi2s, q=99.0))]
                        except IndexError:  # array empty
                            range_full = [0.0, 1.0]
                        hist_full, edges_full = np.histogram(chi2s, range=range_full, bins=250)
                        hist_narrow, edges_narrow = np.histogram(chi2s, range=[0, 20], bins=100)
                        hist_chi2_red, edges_chi2_red = np.histogram(chi2s_red, range=[0, 10], bins=50)
                        hist_pvalue, edges_pvalue = np.histogram(track_pvalue, range=[0, 1], bins=50)
                    else:
                        hist_full += np.histogram(chi2s, bins=edges_full)[0]
                        hist_narrow += np.histogram(chi2s, bins=edges_narrow)[0]
                        hist_very_narrow += np.histogram(chi2s, bins=edges_very_narrow)[0]
                        hist_chi2_red += np.histogram(chi2s_red, bins=edges_chi2_red)[0]
                        hist_pvalue += np.histogram(track_pvalue, bins=edges_pvalue)[0]

                plot_log = np.any(chi2s)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                x = (edges_full[1:] + edges_full[:-1]) / 2.0
                width = (edges_full[1:] - edges_full[:-1])
                ax.bar(x, hist_full, width=width, log=plot_log, align='center')
                ax.grid()
                ax.set_xlim([edges_full[0], edges_full[-1]])
                ax.set_xlabel('$\mathrm{\chi}^2$')
                ax.set_ylabel('#')
                ax.set_yscale('log')
                ax.set_title('Track $\mathrm{\chi}^2$ for %s' % dut_name)
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                x = (edges_narrow[1:] + edges_narrow[:-1]) / 2.0
                width = (edges_narrow[1:] - edges_narrow[:-1])
                ax.bar(x, hist_narrow, width=width, log=plot_log, align='center')
                ax.grid()
                ax.set_xlim([edges_narrow[0], edges_narrow[-1]])
                ax.set_xlabel('$\mathrm{\chi}^2$')
                ax.set_ylabel('#')
                ax.set_yscale('log')
                ax.set_title('Track $\mathrm{\chi}^2$ for %s' % dut_name)
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                x = (edges_narrow[1:] + edges_narrow[:-1]) / 2.0
                width = (edges_narrow[1:] - edges_narrow[:-1])
                ax.bar(x, hist_narrow / hist_narrow.sum() / width, width=width, log=False, align='center')
                x = np.arange(0, 250, 0.001)
                ax.grid()
                ax.set_xlim([edges_narrow[0], edges_narrow[-1]])
                ax.set_xlabel('$\mathrm{\chi}^2$')
                ax.set_ylabel('#')
                ax.set_title('Track $\mathrm{\chi}^2$ for %s' % dut_name)
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                x = (edges_chi2_red[1:] + edges_chi2_red[:-1]) / 2.0
                width = (edges_chi2_red[1:] - edges_chi2_red[:-1])
                ax.bar(x, hist_chi2_red, width=width, log=False, align='center')
                ax.grid()
                ax.set_xlim([edges_chi2_red[0], edges_chi2_red[-1]])
                ax.set_xlabel('$\mathrm{\chi}^2_{\mathrm{red}}$')
                ax.set_ylabel('#')
                ax.set_title('Track $\mathrm{\chi}^2_{\mathrm{red}}$ for %s' % dut_name)
                output_pdf.savefig(fig)

                # Plot pvalue distribution
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                x = (edges_pvalue[1:] + edges_pvalue[:-1]) / 2.0
                width = (edges_pvalue[1:] - edges_pvalue[:-1])
                ax.bar(x, hist_pvalue, width=width, align='center')
                ax.set_xlabel('Track p-value')
                ax.set_ylabel('#')
                ax.grid()
                ax.set_title('p-value distribution for %s' % dut_name)
                output_pdf.savefig(fig)


def plot_residuals(histogram, edges, fit, cov, xlabel, title, output_pdf=None):
    if not output_pdf:
        return

    for plot_log in [False, True]:  # plot with log y or not
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # Calculate bin centers
        x = (edges[1:] + edges[:-1]) / 2.0
        plot_range = (beam_telescope_analysis.tools.analysis_utils.get_mean_from_histogram(histogram, x) - 5 * beam_telescope_analysis.tools.analysis_utils.get_rms_from_histogram(histogram, x),
                      beam_telescope_analysis.tools.analysis_utils.get_mean_from_histogram(histogram, x) + 5 * beam_telescope_analysis.tools.analysis_utils.get_rms_from_histogram(histogram, x))
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
            ax.plot([fit[1], fit[1]], [0, ax.get_ylim()[1]], color='r', label='Entries: %d\n$\mathrm{RMS}=%.1f$ [$\mathrm{\mu}$m]' % (histogram.sum(), beam_telescope_analysis.tools.analysis_utils.get_rms_from_histogram(histogram, x)))
            gauss_fit_legend_entry = 'Gauss fit: \n$A=%.1f\pm %.1f$\n$\mathrm{\mu}=%.1f\pm %.1f$ [$\mathrm{\mu}$m]\n$\mathrm{\sigma}=%.1f\pm %.1f$ [$\mathrm{\mu}$m]' % (fit[0], np.absolute(cov[0][0] ** 0.5), fit[1], np.absolute(cov[1][1] ** 0.5), np.absolute(fit[2]), np.absolute(cov[2][2] ** 0.5))
            x_gauss = np.arange(np.floor(np.min(edges)), np.ceil(np.max(edges)), step=0.1)
            ax.plot(x_gauss, beam_telescope_analysis.tools.analysis_utils.gauss(x_gauss, *fit), 'r--', label=gauss_fit_legend_entry, linewidth=2)
            ax.legend(loc=0)
        ax.set_xlim([edges[0], edges[-1]])
        output_pdf.savefig(fig)


def plot_residuals_vs_position(hist, xedges, yedges, xlabel, ylabel, title, residuals_mean=None, select=None, fit=None, cov=None, limit=None, output_pdf=None):
    '''Plot the residuals as a function of the position.
    '''
    if not output_pdf:
        return

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.imshow(np.ma.masked_equal(hist, 0).T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', aspect='auto', interpolation='none')
    if residuals_mean is not None:
        res_pos = (xedges[1:] + xedges[:-1]) / 2.0
        if select is None:
            select = np.full_like(res_pos, True, dtype=np.bool)
        ax.scatter(res_pos[select], residuals_mean[select], color="blue", marker='o', label='Mean residual')
        ax.scatter(res_pos[~select], residuals_mean[~select], color="r", marker='o')
    if fit is not None:
        x_lim = np.array(ax.get_xlim(), dtype=np.float64)
        ax.plot(x_lim, beam_telescope_analysis.tools.analysis_utils.linear(x_lim, *fit), linestyle='-', color="darkorange", linewidth=2, label='Mean residual fit\n%.2e + %.2e x' % (fit[0], fit[1]))
    if limit is not None:
        if np.isfinite(limit[0]):
            ax.axvline(x=limit[0], linewidth=2, color='r')
        if np.isfinite(limit[1]):
            ax.axvline(x=limit[1], linewidth=2, color='r')
    ax.set_xlim([xedges[0], xedges[-1]])
    ax.set_ylim([yedges[0], yedges[-1]])
    ax.legend(loc=0)
    output_pdf.savefig(fig)


def plot_track_density(telescope_configuration, input_tracks_file, select_duts, output_pdf_file=None, chunk_size=1000000):
    '''Plotting of the track and hit density for selected DUTs.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_tracks_file : string
        Filename of the input tracks file.
    select_duts : iterable
        Selecting DUTs that will be processed.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    logging.info('=== Plotting track density for %d DUTs ===' % len(select_duts))

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_track_density.pdf'

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            for actual_dut_index in select_duts:
                actual_dut = telescope[actual_dut_index]
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)

                logging.info('Calculating track density for %s', actual_dut.name)

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in beam_telescope_analysis.tools.analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):

                    # Coordinates in global coordinate system (x, y, z)
                    hit_x_local, hit_y_local = tracks_chunk['x_dut_%d' % actual_dut_index], tracks_chunk['y_dut_%d' % actual_dut_index]
                    intersection_x_local, intersection_y_local = tracks_chunk['offset_x'], tracks_chunk['offset_y']

                    if initialize:
                        initialize = False

                        dut_x_size = np.abs(np.diff(actual_dut.x_extent()))[0]
                        dut_y_size = np.abs(np.diff(actual_dut.y_extent()))[0]
                        hist_2d_res_x_edges = np.linspace(-dut_x_size / 2.0, dut_x_size / 2.0, actual_dut.n_columns + 1, endpoint=True)
                        hist_2d_res_y_edges = np.linspace(-dut_y_size / 2.0, dut_y_size / 2.0, actual_dut.n_rows + 1, endpoint=True)
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

                plot_2d_map(
                    hist2d=hist_tracks.T,
                    plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                    title='Track density for %s (%d Tracks)' % (actual_dut.name, n_tracks),
                    x_axis_title='Column position [$\mathrm{\mu}$m]',
                    y_axis_title='Row position [$\mathrm{\mu}$m]',
                    z_min=0,
                    z_max=None,
                    output_pdf=output_pdf)

                plot_2d_map(
                    hist2d=hist_hits.T,
                    plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                    title='Hit density for %s (%d Hits)' % (actual_dut.name, n_hits),
                    x_axis_title='Column position [$\mathrm{\mu}$m]',
                    y_axis_title='Row position [$\mathrm{\mu}$m]',
                    z_min=0,
                    z_max=None,
                    output_pdf=output_pdf)


def plot_charge_distribution(telescope_configuration, input_tracks_file, select_duts, output_pdf_file=None, chunk_size=1000000):
    '''Plotting of the charge distribution for selected DUTs.

    Parameters
    ----------
    telescope_configuration : string
        Filename of the telescope configuration file.
    input_tracks_file : string
        Filename of the input tracks file.
    select_duts : iterable
        Selecting DUTs that will be processed.
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    logging.info('=== Plotting mean charge distribution for %d DUTs ===' % len(select_duts))

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_mean_charge.pdf'

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            for actual_dut_index in select_duts:
                actual_dut = telescope[actual_dut_index]
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)

                logging.info('Calculating mean charge for %s', actual_dut.name)

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in beam_telescope_analysis.tools.analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):

                    # Coordinates in global coordinate system (x, y, z)
                    hit_x_local, hit_y_local = tracks_chunk['x_dut_%d' % actual_dut_index], tracks_chunk['y_dut_%d' % actual_dut_index]
                    intersection_x_local, intersection_y_local = tracks_chunk['offset_x'], tracks_chunk['offset_y']
                    charge = tracks_chunk['charge_dut_%d' % actual_dut_index]

                    # select tracks and hits with charge
                    select_hits = np.isfinite(charge)

                    hit_x_local, hit_y_local = hit_x_local[select_hits], hit_y_local[select_hits]
                    intersection_x_local, intersection_y_local = intersection_x_local[select_hits], intersection_y_local[select_hits]
                    charge = charge[select_hits]

                    if initialize:
                        initialize = False

                        dut_x_size = np.abs(np.diff(actual_dut.x_extent()))[0]
                        dut_y_size = np.abs(np.diff(actual_dut.y_extent()))[0]
                        hist_2d_res_x_edges = np.linspace(-dut_x_size / 2.0, dut_x_size / 2.0, actual_dut.n_columns + 1, endpoint=True)
                        hist_2d_res_y_edges = np.linspace(-dut_y_size / 2.0, dut_y_size / 2.0, actual_dut.n_rows + 1, endpoint=True)
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

                plot_2d_map(
                    hist2d=stat_track_charge_hist.T,
                    plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                    title='Mean charge of tracks for %s (%d Tracks)' % (actual_dut.name, n_tracks),
                    x_axis_title='Column position [$\mathrm{\mu}$m]',
                    y_axis_title='Row position [$\mathrm{\mu}$m]',
                    z_min=0,
                    z_max=charge_max_tracks,
                    output_pdf=output_pdf)

                plot_2d_map(
                    hist2d=stat_hits_charge_hist.T,
                    plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                    title='Mean charge of hits for %s (%d Hits)' % (actual_dut.name, n_hits),
                    x_axis_title='Column position [$\mathrm{\mu}$m]',
                    y_axis_title='Row position [$\mathrm{\mu}$m]',
                    z_min=0,
                    z_max=charge_max_hits,
                    output_pdf=output_pdf)


def voronoi_plot_2d(ax, ridge_vertices, vertices, points=None, show_points=False, line_width=1.0, line_alpha=1.0, line_style='solid', line_color='k', point_size=1.0, point_alpha=1.0, point_marker='.', point_color='k'):
    '''
    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot
    '''
    ridge_vertices = np.array(ridge_vertices)
    select = np.all(ridge_vertices >= 0, axis=1)
    if np.count_nonzero(select) > 10**6 or (points is not None and points.shape[0] > 10**5):
        logging.warning("Omitting voronoi mesh: too many vertices and/or points.")
    else:
        if show_points:
            ax.plot(points[:, 0], points[:, 1], linestyle='None', markersize=point_size, alpha=point_alpha, marker=point_marker, color=point_color)
        ax.add_collection(
            LineCollection(
                segments=vertices[ridge_vertices[select]],
                colors=line_color,
                linewidth=line_width,
                alpha=line_alpha,
                linestyle=line_style))
    return ax.figure


def pixels_plot_2d(fig, ax, regions, vertices, values, z_min=0, z_max=None):
    '''
    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot
    '''
    if z_max is None:
        if values.all() is np.ma.masked or np.allclose(0, values):  # check if masked array is fully masked
            z_max = 1
        else:
            z_max = ceil(values.max())
    verts = []
    for region in regions:
        if -1 not in region:
            vert = vertices[region]
            verts.append(vert)
    cmap = cm.get_cmap('viridis')
    cmap.set_bad('w')
    norm = colors.Normalize(vmin=z_min, vmax=z_max)
    p = PolyCollection(
        verts=verts,
        cmap=cmap,
        norm=norm)
    p.set_array(values)
    ax.add_collection(p)
    bounds = np.linspace(start=z_min, stop=z_max, num=255, endpoint=True)
    fig.colorbar(p, boundaries=bounds, ticks=np.linspace(start=z_min, stop=z_max, num=9, endpoint=True), fraction=0.04, pad=0.05)
    return ax.figure


def efficiency_plots(telescope, hist_2d_edges, count_hits_2d_hist, count_tracks_2d_hist, count_tracks_with_hit_2d_hist, stat_2d_x_residuals_hist, stat_2d_y_residuals_hist, stat_2d_residuals_hist, count_1d_charge_hist, stat_2d_charge_hist, count_1d_frame_hist, stat_2d_cluster_size_hist, count_1d_total_angle_hist, count_1d_total_angle_hist_edges, count_1d_alpha_angle_hist, count_1d_alpha_angle_hist_edges, count_1d_beta_angle_hist, count_1d_beta_angle_hist_edges, stat_2d_frame_hist, stat_2d_total_angle_hist, stat_2d_alpha_angle_hist, stat_2d_beta_angle_hist, stat_2d_efficiency_hist, stat_pixel_efficiency_hist, count_pixel_hits_2d_hist, efficiency, efficiency_chunks, actual_dut_index, dut_extent, hist_extent, plot_range, efficiency_regions, efficiency_regions_names, efficiency_regions_efficiencies, efficiency_regions_count_1d_charge_hist, efficiency_regions_count_1d_frame_hist, efficiency_regions_count_1d_cluster_size_hist, efficiency_regions_count_1d_total_angle_hist, efficiency_regions_count_1d_total_angle_hist_edges, efficiency_regions_count_1d_alpha_angle_hist, efficiency_regions_count_1d_alpha_angle_hist_edges, efficiency_regions_count_1d_beta_angle_hist, efficiency_regions_count_1d_beta_angle_hist_edges, efficiency_regions_count_1d_cluster_shape_hist, efficiency_regions_stat_pixel_efficiency_hist, efficiency_regions_count_in_pixel_hits_2d_hist, efficiency_regions_count_in_pixel_tracks_2d_hist, efficiency_regions_count_in_pixel_tracks_with_hit_2d_hist, efficiency_regions_stat_in_pixel_efficiency_2d_hist, efficiency_regions_stat_in_pixel_x_residuals_2d_hist, efficiency_regions_stat_in_pixel_y_residuals_2d_hist, efficiency_regions_stat_in_pixel_residuals_2d_hist, efficiency_regions_stat_in_pixel_charge_2d_hist, efficiency_regions_stat_in_pixel_frame_2d_hist, efficiency_regions_stat_in_pixel_cluster_size_2d_hist, efficiency_regions_count_in_pixel_cluster_shape_2d_hist, efficiency_regions_stat_in_pixel_cluster_shape_2d_hist, chunk_indices, efficiency_regions_in_pixel_hist_extent, efficiency_regions_in_pixel_plot_range, efficiency_regions_analyze_cluster_shapes, mask_zero=True, output_pdf=None, z_limits_charge=(0, 255)):
    actual_dut = telescope[actual_dut_index]
    if not output_pdf:
        return
    # get number of entries for every histogram
    n_hits = np.sum(count_hits_2d_hist)
    n_tracks = np.sum(count_tracks_2d_hist)
    n_tracks_with_hit = np.sum(count_tracks_with_hit_2d_hist)

    pixel_indices = np.indices((actual_dut.n_columns, actual_dut.n_rows)).reshape(2, -1).T
    local_x, local_y, _ = actual_dut.index_to_local_position(
        column=pixel_indices[:, 0] + 1,
        row=pixel_indices[:, 1] + 1)
    pixel_center_data = np.column_stack((local_x, local_y))
    _, regions, ridge_vertices, vertices = beam_telescope_analysis.tools.analysis_utils.voronoi_finite_polygons_2d(points=pixel_center_data, dut_extent=dut_extent)

    # for better readability allow masking of entries that are zero
    count_hits_2d_hist_masked = np.ma.array(count_hits_2d_hist, mask=(count_hits_2d_hist == 0))
    count_tracks_2d_hist_masked = np.ma.array(count_tracks_2d_hist, mask=(count_tracks_2d_hist == 0))
    count_tracks_with_hit_2d_hist_masked = np.ma.array(count_tracks_with_hit_2d_hist, mask=(count_tracks_with_hit_2d_hist == 0))

    mesh_color = 'red'
    mesh_line_width = 0.5
    mesh_point_size = 0.75
    mesh_alpha = 0.5

    in_pixel_mesh_color = 'grey'
    in_pixel_mesh_line_width = 2.0
    in_pixel_mesh_point_size = 1.0
    in_pixel_mesh_alpha = 1.0
    in_pixel_mesh_line_style = '--'

    widths_in_pixel_regions = [[20.0, 20.0], [10.0, 20.0], [10.0, 10.0]]  # in um
    center_location_in_pixel_regions = [[25.0, 25.0], [0.0, 25.0], [0.0, 50.0]]  # in um

    # widths_in_pixel_regions = [[50.0, 50.0]]  # in um
    # center_location_in_pixel_regions = [[25.0, 25.0]]  # in um

    fig = Figure()
    text = 'DUT%d:\n%s' % (actual_dut_index, actual_dut.name)
    fig.text(0.5, 0.5, text, transform=fig.transFigure, size=24, ha="center")
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color='r')
    _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices, vertices=vertices, points=pixel_center_data, show_points=True, line_width=mesh_line_width, line_alpha=1.0, line_color=mesh_color, point_size=mesh_point_size, point_alpha=1.0, point_color=mesh_color)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    ax.set_title('Pixel locations\nfor %s' % actual_dut.name)
    ax.set_xlabel("column [$\mathrm{\mu}$m]")
    ax.set_ylabel("row [$\mathrm{\mu}$m]")
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    z_max = np.ceil(np.percentile(count_hits_2d_hist_masked.compressed(), q=95.0))
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    plot_2d_pixel_hist(fig, ax, count_hits_2d_hist_masked.T, hist_extent, title='Hit density\nfor %s\n(%d Hits)' % (actual_dut.name, n_hits), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_max=z_max)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    z_max = np.ceil(np.percentile(count_tracks_2d_hist_masked.compressed(), q=95.0))
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    plot_2d_pixel_hist(fig, ax, count_tracks_2d_hist_masked.T, hist_extent, title='Track density\nfor %s\n(%d Tracks)' % (actual_dut.name, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_max=z_max)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    z_max = np.ceil(np.percentile(count_tracks_with_hit_2d_hist_masked.compressed(), q=95.0))
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    plot_2d_pixel_hist(fig, ax, count_tracks_with_hit_2d_hist_masked.T, hist_extent, title='Track density with associated hit\nfor %s\n(%d Tracks)' % (actual_dut.name, n_tracks_with_hit), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_max=z_max)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    output_pdf.savefig(fig)

    if pixel_center_data.shape[0] > 10**5:
        logging.warning("Omitting plots: too many pixels")
    else:
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        bin_indices = np.indices(stat_2d_efficiency_hist.shape).reshape(2, -1).T
        x_bin_centers = (hist_2d_edges[0][1:] + hist_2d_edges[0][:-1]) / 2
        y_bin_centers = (hist_2d_edges[1][1:] + hist_2d_edges[1][:-1]) / 2
        y_meshgrid, x_meshgrid = np.meshgrid(y_bin_centers, x_bin_centers)
        select_bins = np.zeros(x_meshgrid.shape, dtype=np.bool)
        # reduce number of arrows
        select_bins[::4, ::4] = 1
        # speed up plotting
        select_bins &= x_meshgrid >= min(plot_range[0])
        select_bins &= x_meshgrid <= max(plot_range[0])
        select_bins &= y_meshgrid >= min(plot_range[1])
        select_bins &= y_meshgrid <= max(plot_range[1])
        _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices, vertices=vertices, points=pixel_center_data, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
        ax.quiver(x_meshgrid[select_bins], y_meshgrid[select_bins], np.nan_to_num(stat_2d_x_residuals_hist[select_bins]), np.nan_to_num(stat_2d_y_residuals_hist[select_bins]), angles='xy', scale_units='xy', scale=1.0, pivot='tail', minshaft=2.0, width=0.001)
        rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
        ax.add_patch(rect)
        ax.set_title('Residuals\nfor %s' % actual_dut.name)
        ax.set_xlabel("column [$\mathrm{\mu}$m]")
        ax.set_ylabel("row [$\mathrm{\mu}$m]")
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        output_pdf.savefig(fig)

        select_bins = (count_tracks_with_hit_2d_hist > 0)
        residual_vectors_x = x_meshgrid[select_bins] + stat_2d_x_residuals_hist[select_bins]
        residual_vectors_y = y_meshgrid[select_bins] + stat_2d_y_residuals_hist[select_bins]
        residual_vectors = np.column_stack((np.ravel(residual_vectors_x), np.ravel(residual_vectors_y)))
        bin_center_data_sel = np.column_stack((np.ravel(x_meshgrid[select_bins]), np.ravel(y_meshgrid[select_bins])))
        bin_indices_sel = bin_indices[np.ravel(select_bins)]
        pixel_center_kd_tree = cKDTree(pixel_center_data)
        _, residual_to_pixel_center_sel = pixel_center_kd_tree.query(residual_vectors)
        _, bin_center_to_pixel_center_sel = pixel_center_kd_tree.query(bin_center_data_sel)
        color_indices = np.linspace(0.0, 1.0, num=100)
        cmap = copy(cm.get_cmap('tab20'))
        cmap.set_bad('w')
        rgb_colors = np.array([cmap(color) for color in color_indices])
        valid_color_indices = np.r_[0, np.unique(np.where(rgb_colors[:-1] != rgb_colors[1:])[0]) + 1]
        num_colors = len(valid_color_indices)
        # select and reorder colors
        color_indices = color_indices[valid_color_indices]
        color_indices[::2] = np.roll(color_indices[::2], int(num_colors / 4))
        rgb_colors = rgb_colors[valid_color_indices]
        rgb_colors[::2, :] = np.roll(rgb_colors[::2, :], int(num_colors / 4), axis=0)
        color_index_array = np.full(shape=pixel_center_data.shape[0], dtype=np.int8, fill_value=-1)
        count_index_array = np.zeros(shape=pixel_center_data.shape[0], dtype=np.uint16)
        effective_pixels_2d = np.full(shape=stat_2d_efficiency_hist.shape, dtype=np.int32, fill_value=-1)
        color_index = 0
        # x_res = (hist_2d_edges[0][1] - hist_2d_edges[0][0])
        # y_res = (hist_2d_edges[1][1] - hist_2d_edges[1][0])
        for pixel_index, pixel_position in enumerate(pixel_center_data):
            res_center_col_row_pair_data_indices = np.where(residual_to_pixel_center_sel == pixel_index)[0]
            # res_center_col_row_pair_data_positions = bin_center_data_sel[res_center_col_row_pair_data_indices]
            actual_bin_col_row_indices = bin_indices_sel[res_center_col_row_pair_data_indices]
            bin_center_data_indices = np.where(bin_center_to_pixel_center_sel == pixel_index)[0]
            count_index_array[pixel_index] = np.count_nonzero(bin_center_data_indices)
            # bin_center_data_positions = bin_center_data_sel[bin_center_data_indices]
            # index_0_pixel = np.array((bin_center_data_positions[:, 0] - hist_2d_edges[0][0] - x_res / 2) / x_res, dtype=np.int)
            # index_1_pixel = np.array((bin_center_data_positions[:, 1] - hist_2d_edges[1][0] - y_res / 2) / y_res, dtype=np.int)
            actual_pixel_bin_indices = bin_indices_sel[bin_center_data_indices]
            other_pixel_indices = effective_pixels_2d[actual_pixel_bin_indices[:, 0], actual_pixel_bin_indices[:, 1]]
            other_colors = color_index_array[other_pixel_indices]
            # change color if same color is already occurring inside pixel region
            num_repeats = 0
            while color_index % num_colors in other_colors and num_repeats < num_colors - 1:
                color_index += 1
                num_repeats += 1
            # index_0 = np.array((res_center_col_row_pair_data_positions[:, 0] - hist_2d_edges[0][0] - x_res / 2) / x_res, dtype=np.int)
            # index_1 = np.array((res_center_col_row_pair_data_positions[:, 1] - hist_2d_edges[1][0] - y_res / 2) / y_res, dtype=np.int)
            color_index_array[pixel_index] = color_index % num_colors
            effective_pixels_2d[actual_bin_col_row_indices[:, 0], actual_bin_col_row_indices[:, 1]] = pixel_index
            color_index += 1
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        for pixel_index, pixel_position in enumerate(pixel_center_data):
            if count_index_array[pixel_index]:
                ax.plot(pixel_position[0], pixel_position[1], markersize=1.0, marker='o', alpha=1.0, color=rgb_colors[color_index_array[pixel_index]], markeredgecolor='k', markeredgewidth=0.1)
        _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices, vertices=vertices, points=pixel_center_data, show_points=False, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
        effective_color_2d = color_indices[color_index_array[effective_pixels_2d]]
        effective_color_2d = np.ma.masked_where(effective_pixels_2d == -1, effective_color_2d)
        plot_2d_pixel_hist(fig, ax, effective_color_2d.T, hist_extent, title='Effective pixel locations\nfor %s' % actual_dut.name, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=1.0, cmap=cmap, show_colorbar=False)
        rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
        ax.add_patch(rect)
        ax.set_xlabel("column [$\mathrm{\mu}$m]")
        ax.set_ylabel("row [$\mathrm{\mu}$m]")
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        output_pdf.savefig(fig)

        if count_pixel_hits_2d_hist is not None:
            x_resolution = np.diff(hist_2d_edges[0])[0]
            y_resolution = np.diff(hist_2d_edges[1])[0]
            pixel_sizes = np.full(pixel_center_data.shape[0], dtype=np.float64, fill_value=np.nan)
            select_bins = np.sum(count_pixel_hits_2d_hist.reshape(count_pixel_hits_2d_hist.shape[0], count_pixel_hits_2d_hist.shape[1], -1), axis=2) != 0
            # different from
            # select_bins = count_tracks_with_hit_2d_hist[:].astype(np.int64) != 0
            # due to hits outside the relative pixel index array
            bin_center_data_sel = np.column_stack((np.ravel(x_meshgrid[select_bins]), np.ravel(y_meshgrid[select_bins])))
            bin_indices_sel = bin_indices[np.ravel(select_bins)]
            _, bin_center_to_pixel_center_sel = pixel_center_kd_tree.query(bin_center_data_sel)
            max_pixel_index_hist = np.column_stack(np.unravel_index(np.argmax(count_pixel_hits_2d_hist.reshape(count_pixel_hits_2d_hist.shape[0] * count_pixel_hits_2d_hist.shape[1], -1), axis=1), dims=count_pixel_hits_2d_hist.shape[2:]))
            max_pixel_index_hist[np.ravel(select_bins), 0] -= int(count_pixel_hits_2d_hist.shape[2] / 2)
            max_pixel_index_hist[np.ravel(select_bins), 1] -= int(count_pixel_hits_2d_hist.shape[3] / 2)
            pixel_indices = np.indices((actual_dut.n_columns, actual_dut.n_rows)).reshape(2, -1).T
            bin_center_data = np.column_stack((np.ravel(x_meshgrid), np.ravel(y_meshgrid)))
            _, bin_center_to_pixel_center = pixel_center_kd_tree.query(bin_center_data)
            max_hits_pixel_col_row = pixel_indices[bin_center_to_pixel_center] + max_pixel_index_hist
            max_hits_pixel_index = np.ravel_multi_index(max_hits_pixel_col_row.T, dims=(actual_dut.n_columns, actual_dut.n_rows))
            # generate array with pixel index of the pixel with the most hits for each bin
            max_hits_pixel_index_sel = max_hits_pixel_index[np.ravel(select_bins)]
            color_indices = np.linspace(0.0, 1.0, num=100)
            cmap = copy(cm.get_cmap('tab20'))
            cmap.set_bad('w')
            rgb_colors = np.array([cmap(color) for color in color_indices])
            valid_color_indices = np.r_[0, np.unique(np.where(rgb_colors[:-1] != rgb_colors[1:])[0]) + 1]
            num_colors = len(valid_color_indices)
            # select and reorder colors
            color_indices = color_indices[valid_color_indices]
            color_indices[::2] = np.roll(color_indices[::2], int(num_colors / 4))
            rgb_colors = rgb_colors[valid_color_indices]
            rgb_colors[::2, :] = np.roll(rgb_colors[::2, :], int(num_colors / 4), axis=0)
            color_index_array = np.full(shape=pixel_center_data.shape[0], dtype=np.int8, fill_value=-1)
            count_index_array = np.zeros(shape=pixel_center_data.shape[0], dtype=np.uint16)
            effective_pixels_2d = np.full(shape=stat_2d_efficiency_hist.shape, dtype=np.int32, fill_value=-1)
            color_index = 0
            for pixel_index, pixel_position in enumerate(pixel_center_data):
                actual_bin_indices_sel = np.where((max_hits_pixel_index_sel == pixel_index))[0]
                actual_bin_col_row_indices = bin_indices_sel[actual_bin_indices_sel]
                # alternative:
                # actual_bin_indices = np.where((max_hits_pixel_index == pixel_index) & select_bins.reshape(-1))[0]
                # actual_bin_col_row_indices = np.column_stack(np.unravel_index(actual_bin_indices, dims=count_pixel_hits_2d_hist.shape[:2]))
                pixel_sizes[pixel_index] = actual_bin_col_row_indices.shape[0] * x_resolution * y_resolution
                bin_center_data_indices = np.where(bin_center_to_pixel_center_sel == pixel_index)[0]
                count_index_array[pixel_index] = np.count_nonzero(bin_center_data_indices)
                actual_pixel_bin_indices = bin_indices_sel[bin_center_data_indices]
                other_pixel_indices = effective_pixels_2d[actual_pixel_bin_indices[:, 0], actual_pixel_bin_indices[:, 1]]
                other_colors = color_index_array[other_pixel_indices]
                # change color if same color is already occurring inside pixel region
                num_repeats = 0
                while color_index % num_colors in other_colors and num_repeats < num_colors - 1:
                    color_index += 1
                    num_repeats += 1
                color_index_array[pixel_index] = color_index % num_colors
                effective_pixels_2d[actual_bin_col_row_indices[:, 0], actual_bin_col_row_indices[:, 1]] = pixel_index
                color_index += 1
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            for pixel_index, pixel_position in enumerate(pixel_center_data):
                if count_index_array[pixel_index]:
                    ax.plot(pixel_position[0], pixel_position[1], markersize=1.0, marker='o', alpha=1.0, color=rgb_colors[color_index_array[pixel_index]], markeredgecolor='k', markeredgewidth=0.1)
            _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices, vertices=vertices, points=pixel_center_data, show_points=False, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
            effective_color_2d = colors[color_index_array[effective_pixels_2d]]
            effective_color_2d = np.ma.masked_where(effective_pixels_2d == -1, effective_color_2d)
            plot_2d_pixel_hist(fig, ax, effective_color_2d.T, hist_extent, title='Effective pixel locations\nfor %s' % actual_dut.name, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=1.0, cmap=cmap, show_colorbar=False)
            rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
            ax.add_patch(rect)
            ax.set_xlabel("column [$\mathrm{\mu}$m]")
            ax.set_ylabel("row [$\mathrm{\mu}$m]")
            ax.set_xlim(plot_range[0])
            ax.set_ylim(plot_range[1])
            output_pdf.savefig(fig)

            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            max_region_size = 0
            for region in regions:
                max_region_size = max(max_region_size, len(region))
            region_array = np.zeros((len(regions), max_region_size), dtype=np.int32)
            for region_index, region in enumerate(regions):
                region_array[region_index] = region + [region[-1]] * (max_region_size - len(region))
            pixel_vertices = vertices[region_array]
            calculated_pixel_sizes = beam_telescope_analysis.tools.analysis_utils.polygon_area_multi(pixel_vertices[:, :, 0], pixel_vertices[:, :, 1])
            _, bin_edges, _ = ax.hist(pixel_sizes[np.isfinite(pixel_sizes)], bins=100, range=(0, np.ceil(max(np.max(pixel_sizes[np.isfinite(pixel_sizes)]), np.max(calculated_pixel_sizes)))), alpha=0.5, label='Measured pixel size')
            ax.hist(calculated_pixel_sizes[np.isfinite(pixel_sizes)], bins=bin_edges, alpha=0.5, label='Calculated pixel size')
            ax.set_yscale('log')
            ax.set_title('Effective pixel sizes\nfor %s' % actual_dut.name)
            ax.set_xlabel("Pixel size [$\mathrm{\mu}$m$^2$]")
            ax.set_ylabel("#")
            ax.legend()
            output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    residual_hist, _ = np.histogram(stat_2d_residuals_hist.compressed(), bins=np.arange(np.ceil(np.max(stat_2d_residuals_hist))))
    hist_residuals_masked, hist_residuals_indices = beam_telescope_analysis.tools.analysis_utils.hist_quantiles(hist=residual_hist, prob=(0.0, 0.95), return_indices=True)
    z_max = hist_residuals_indices[-1] + 1
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    plot_2d_pixel_hist(fig, ax, stat_2d_residuals_hist.T, hist_extent, title='Mean 2D residuals\nfor %s' % (actual_dut.name,), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_max=z_max)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    hist_charge_masked, hist_charge_indices = beam_telescope_analysis.tools.analysis_utils.hist_quantiles(hist=count_1d_charge_hist, prob=(0.0, 0.95), return_indices=True)
    ax.bar(x=range(hist_charge_indices[-1] + 1), height=hist_charge_masked[:hist_charge_indices[-1] + 1], align='center')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    add_value_labels(ax=ax)
    ax.set_title('Charge distribution\nfor %s' % actual_dut.name)
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    z_max = hist_charge_indices[-1] + 1
    plot_2d_pixel_hist(fig, ax, stat_2d_charge_hist.T, hist_extent, title='Mean charge\nfor %s' % (actual_dut.name,), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=z_max)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    count_1d_total_angle_hist_edges = count_1d_total_angle_hist_edges * 1000.0  # convert to mrad
    bin_centers = (count_1d_total_angle_hist_edges[1:] + count_1d_total_angle_hist_edges[:-1]) / 2.0
    width = np.diff(bin_centers)[0]
    ax.bar(x=bin_centers, height=count_1d_total_angle_hist, width=width, align='center')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Total track angle distribution\nfor %s' % actual_dut.name)
    ax.set_yscale('log')
    ax.set_xlabel('Track angle [mrad]')
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    max_frame = count_1d_frame_hist.shape[0]
    ax.bar(x=range(max_frame), height=count_1d_frame_hist, align='center')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    add_value_labels(ax=ax)
    ax.set_title('Frame distribution\nfor %s' % actual_dut.name)
    ax.set_yscale('log')
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    z_max = stat_2d_frame_hist.max()
    z_min = stat_2d_frame_hist.min()
    plot_2d_pixel_hist(fig, ax, stat_2d_frame_hist.T, hist_extent, title='Mean frame\nfor %s' % (actual_dut.name,), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=z_max)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # plot cluster size average (cluster sizes 1 - 4 only)
    z_min = 1.0
    z_max = 4.0
    # set1_cmap = copy(m.get_cmap("Set1", 9))
    # new_colors = set1_cmap(np.linspace(0, 1, 9))
    # new_cmap = colors.ListedColormap(new_colors[1:5], name="cluster_colormap")
    # new_cmap.set_over('k')
    cmap = copy(cm.get_cmap("viridis", 256))
    cmap.set_over('magenta')
    _, cbar = plot_2d_pixel_hist(fig, ax, stat_2d_cluster_size_hist.T, hist_extent, title='Mean cluster size\nfor %s' % (actual_dut.name,), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=z_max, cmap=cmap)
    cbar.set_ticks(range(1, 5))
    cbar.set_ticklabels(['1', '2', '3', '4'])
    cbar.set_label("cluster size")
    # cbar.ax.tick_params(length=0)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    count_1d_total_angle_hist_edges = count_1d_total_angle_hist_edges * 1000.0  # convert to mrad
    bin_centers = (count_1d_total_angle_hist_edges[1:] + count_1d_total_angle_hist_edges[:-1]) / 2.0
    width = np.diff(bin_centers)[0]
    ax.bar(x=bin_centers, height=count_1d_total_angle_hist, width=width, align='center')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Total track angle distribution\nfor %s' % actual_dut.name)
    ax.set_yscale('log')
    ax.set_xlabel('Track angle [mrad]')
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    stat_2d_total_angle_hist = stat_2d_total_angle_hist * 1000.0  # convert to mrad
    z_max = stat_2d_total_angle_hist.max()
    z_min = stat_2d_total_angle_hist.min()
    plot_2d_pixel_hist(fig, ax, stat_2d_total_angle_hist.T, hist_extent, title='Mean total track angle\nfor %s' % (actual_dut.name,), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=z_max)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    count_1d_alpha_angle_hist_edges = count_1d_alpha_angle_hist_edges * 1000.0  # convert to mrad
    bin_centers = (count_1d_alpha_angle_hist_edges[1:] + count_1d_alpha_angle_hist_edges[:-1]) / 2.0
    width = np.diff(bin_centers)[0]
    ax.bar(x=bin_centers, height=count_1d_alpha_angle_hist, width=width, align='center')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Alpha track angle distribution\nfor %s' % actual_dut.name)
    ax.set_yscale('log')
    ax.set_xlabel('Track angle [mrad]')
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    stat_2d_alpha_angle_hist = stat_2d_alpha_angle_hist * 1000.0  # convert to mrad
    z_max = stat_2d_alpha_angle_hist.max()
    z_min = stat_2d_alpha_angle_hist.min()
    plot_2d_pixel_hist(fig, ax, stat_2d_alpha_angle_hist.T, hist_extent, title='Mean alpha track angle\nfor %s' % (actual_dut.name,), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=z_max)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    count_1d_beta_angle_hist_edges = count_1d_beta_angle_hist_edges * 1000.0  # convert to mrad
    bin_centers = (count_1d_beta_angle_hist_edges[1:] + count_1d_beta_angle_hist_edges[:-1]) / 2.0
    width = np.diff(bin_centers)[0]
    ax.bar(x=bin_centers, height=count_1d_beta_angle_hist, width=width, align='center')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Beta track angle distribution\nfor %s' % actual_dut.name)
    ax.set_yscale('log')
    ax.set_xlabel('Track angle [mrad]')
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    stat_2d_beta_angle_hist = stat_2d_beta_angle_hist * 1000.0  # convert to mrad
    z_max = stat_2d_beta_angle_hist.max()
    z_min = stat_2d_beta_angle_hist.min()
    plot_2d_pixel_hist(fig, ax, stat_2d_beta_angle_hist.T, hist_extent, title='Mean beta track angle\nfor %s' % (actual_dut.name,), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=z_max)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    output_pdf.savefig(fig)

    if np.any(~stat_2d_efficiency_hist.mask):
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        z_min = 0.0
        plot_2d_pixel_hist(fig, ax, stat_2d_efficiency_hist.T, hist_extent, title='Efficiency\nfor %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)
        _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices, vertices=vertices, show_points=False, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color)
        rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
        ax.add_patch(rect)
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
        z_min = 0.0
        plot_2d_pixel_hist(fig, ax, stat_2d_efficiency_hist.T, hist_extent, title='Efficiency\nfor %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)
        rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
        ax.add_patch(rect)
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title('Efficiency per bin\nfor %s\n(%1.2f (+%1.2f/%1.2f)%%)' % (actual_dut.name, efficiency[0] * 100.0, efficiency[1] * 100.0, efficiency[2] * 100.0))
        ax.set_xlabel('Efficiency [%]')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_xlim([-0.5, 100.5])
        ax.hist(stat_2d_efficiency_hist.ravel()[stat_2d_efficiency_hist.ravel().mask != 1], bins=100, range=(0, 100))  # Histogram not masked pixel stat_2d_efficiency_hist
        output_pdf.savefig(fig)

        # x_mesh = (hist_2d_edges[0][1:] + hist_2d_edges[0][:-1]) / 2
        # y_mesh = (hist_2d_edges[1][1:] + hist_2d_edges[1][:-1]) / 2
        # # select = x_mesh >= min(dut_extent[:2]) & x_mesh <= max(dut_extent[:2])
        # # select &= y_mesh >= min(dut_extent[2:]) & y_mesh <= max(dut_extent[2:])
        # bin_center_data = np.array(np.meshgrid(x_mesh, y_mesh)).T.reshape(-1, 2)
        # select = bin_center_data[:, 0] >= min(dut_extent[:2])
        # select &= bin_center_data[:, 0] <= max(dut_extent[:2])
        # select &= bin_center_data[:, 1] >= min(dut_extent[2:])
        # select &= bin_center_data[:, 1] <= max(dut_extent[2:])
        # bin_center_col_row_pair_dut = bin_center_data[select]
        # _, pixel_center_col_row_pair_index = cKDTree(pixel_center_data).query(bin_center_col_row_pair_dut)
        # pixel_efficiencies = []
        # pixel_efficiencies_bins = np.zeros(shape=stat_2d_efficiency_hist.shape, dtype=np.float64)
        # for pixel_index, pixel in enumerate(pixel_center_data):
        #     bin_center_data_indices = np.where(pixel_center_col_row_pair_index == pixel_index)[0]
        #     bin_center_data_positions = bin_center_col_row_pair_dut[bin_center_data_indices]
        #     # select = bin_center_data_positions[:, 0] >= min(dut_extent[:2]) & bin_center_data_positions[:, 0] <= max(dut_extent[:2])
        #     # select &= bin_center_data_positions[:, 1] >= min(dut_extent[2:]) & bin_center_data_positions[:, 1] <= max(dut_extent[2:])
        #     x_res = (hist_2d_edges[0][1] - hist_2d_edges[0][0])
        #     y_res = (hist_2d_edges[1][1] - hist_2d_edges[1][0])
        #     index_0 = np.array((bin_center_data_positions[:, 0] - hist_2d_edges[0][0] - x_res / 2) / x_res, dtype=np.int)
        #     index_1 = np.array((bin_center_data_positions[:, 1] - hist_2d_edges[1][0] - y_res / 2) / y_res, dtype=np.int)
        #     tracks = np.sum(count_tracks_2d_hist[index_0, index_1])
        #     if tracks == 0:
        #         continue
        #     else:
        #         pixel_efficiency = np.sum(count_tracks_with_hit_2d_hist[index_0, index_1].astype(np.float64)) / np.sum(count_tracks_2d_hist[index_0, index_1].astype(np.float64)) * 100.0
        #     pixel_efficiencies.append(pixel_efficiency)
        #     pixel_efficiencies_bins[index_0, index_1] = pixel_efficiency

        # fig = Figure()
        # _ = FigureCanvas(fig)
        # ax = fig.add_subplot(111)
        # # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
        # z_min = 0.0
        # plot_2d_pixel_hist(fig, ax, pixel_efficiencies_bins.T, hist_extent, title='Efficiency per pixel\nfor %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)
        # rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
        # ax.add_patch(rect)
        # ax.set_xlim(plot_range[0])
        # ax.set_ylim(plot_range[1])
        # output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
        _ = pixels_plot_2d(fig=fig, ax=ax, regions=regions, vertices=vertices, values=stat_pixel_efficiency_hist, z_max=100.0)
        ax.set_title('Efficiency per pixel\nfor %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks))
        ax.set_xlabel("column [$\mathrm{\mu}$m]")
        ax.set_ylabel("row [$\mathrm{\mu}$m]")
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        _ = pixels_plot_2d(fig=fig, ax=ax, regions=regions, vertices=vertices, values=stat_pixel_efficiency_hist, z_min=90, z_max=100.0)
        ax.set_title('Efficiency per pixel for %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks))
        ax.set_xlabel("column [$\mathrm{\mu}$m]")
        ax.set_ylabel("row [$\mathrm{\mu}$m]")
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        _ = pixels_plot_2d(fig=fig, ax=ax, regions=regions, vertices=vertices, values=stat_pixel_efficiency_hist, z_min=95, z_max=100.0)
        ax.set_title('Efficiency per pixel for %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks))
        ax.set_xlabel("column [$\mathrm{\mu}$m]")
        ax.set_ylabel("row [$\mathrm{\mu}$m]")
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        _ = pixels_plot_2d(fig=fig, ax=ax, regions=regions, vertices=vertices, values=stat_pixel_efficiency_hist, z_min=97, z_max=100.0)
        ax.set_title('Efficiency per pixel for %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks))
        ax.set_xlabel("column [$\mathrm{\mu}$m]")
        ax.set_ylabel("row [$\mathrm{\mu}$m]")
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        _ = pixels_plot_2d(fig=fig, ax=ax, regions=regions, vertices=vertices, values=stat_pixel_efficiency_hist, z_min=98, z_max=100.0)
        ax.set_title('Efficiency per pixel for %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks))
        ax.set_xlabel("column [$\mathrm{\mu}$m]")
        ax.set_ylabel("row [$\mathrm{\mu}$m]")
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        output_pdf.savefig(fig)

        logging.info("Efficient pixels (>=97%%): %d" % np.count_nonzero(stat_pixel_efficiency_hist.compressed() >= 97.0))
        logging.info("Efficient pixels (>=95%%): %d" % np.count_nonzero(stat_pixel_efficiency_hist.compressed() >= 95.0))
        logging.info("Efficient pixels (>=80%%): %d" % np.count_nonzero(stat_pixel_efficiency_hist.compressed() >= 80.0))
        logging.info("Efficient pixels (>=50%%): %d" % np.count_nonzero(stat_pixel_efficiency_hist.compressed() >= 50.0))

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title('Efficiency per pixel\nfor %s' % (actual_dut.name,))
        ax.set_xlabel('Efficiency [%]')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_xlim([-0.5, 100.5])
        ax.hist(stat_pixel_efficiency_hist.compressed(), bins=100, range=(0, 100))  # Histogram not masked pixel stat_2d_efficiency_hist
        output_pdf.savefig(fig)

        if efficiency_regions:
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
            z_min = 0.0
            plot_2d_pixel_hist(fig, ax, stat_2d_efficiency_hist.T, hist_extent, title='Efficiency\nfor %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)
            for region_index, region in enumerate(efficiency_regions):
                rect = matplotlib.patches.Rectangle(xy=(min(region[0]), min(region[1])), width=np.abs(np.diff(region[0])), height=np.abs(np.diff(region[1])), linewidth=2.0, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
                ax.add_patch(rect)
                text = 'Region %d%s:\nEfficiency=%.2f%%' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", efficiency_regions_efficiencies[region_index] * 100.0)
                widths = [max(region[0]) - min(region[0]), max(region[1]) - min(region[1])]
                if widths[1] * 2.0 < widths[0]:
                    rotation = 'horizontal'
                elif widths[0] * 2.0 < widths[1]:
                    rotation = 'vertical'
                else:
                    rotation = 'horizontal'
                ax.text(np.sum(region[0]) / 2.0, np.sum(region[1]) / 2.0, text, horizontalalignment='center', verticalalignment='center', fontsize=9, rotation=rotation)
            rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
            ax.add_patch(rect)
            ax.set_xlim(plot_range[0])
            ax.set_ylim(plot_range[1])
            output_pdf.savefig(fig)

            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
            z_min = 0.0
            plot_2d_pixel_hist(fig, ax, stat_2d_efficiency_hist.T, hist_extent, title='Efficiency\nfor %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)
            for region_index, region in enumerate(efficiency_regions):
                rect = matplotlib.patches.Rectangle(xy=(min(region[0]), min(region[1])), width=np.abs(np.diff(region[0])), height=np.abs(np.diff(region[1])), linewidth=2.0, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
                ax.add_patch(rect)
                text = 'Region %d%s:\nEfficiency=%.2f%%' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", efficiency_regions_efficiencies[region_index] * 100.0)
                widths = [max(region[0]) - min(region[0]), max(region[1]) - min(region[1])]
                if widths[1] * 2.0 < widths[0]:
                    rotation = 'horizontal'
                elif widths[0] * 2.0 < widths[1]:
                    rotation = 'vertical'
                else:
                    rotation = 'horizontal'
                ax.text(np.sum(region[0]) / 2.0, np.sum(region[1]) / 2.0, text, horizontalalignment='center', verticalalignment='center', fontsize=9, rotation=rotation)
            _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices, vertices=vertices, show_points=False, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color)
            rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
            ax.add_patch(rect)
            ax.set_xlim(plot_range[0])
            ax.set_ylim(plot_range[1])
            output_pdf.savefig(fig)

            for region_index in range(len(efficiency_regions)):
                fig = Figure()
                text = 'Region %d%s' % (region_index + 1, (":\n" + efficiency_regions_names[region_index]) if efficiency_regions_names[region_index] else "")
                fig.text(0.5, 0.5, text, transform=fig.transFigure, size=24, ha="center")
                output_pdf.savefig(fig)

                region_n_pixels = np.count_nonzero(np.isfinite(efficiency_regions_stat_pixel_efficiency_hist[region_index]))

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
                z_min = 0.0
                plot_2d_pixel_hist(fig, ax, stat_2d_efficiency_hist.T, hist_extent, title='Region %d%s: Efficiency\nfor %s\n(%d Hits, %d Tracks)' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name, n_hits, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)
                region = efficiency_regions[region_index]
                rect = matplotlib.patches.Rectangle(xy=(min(region[0]), min(region[1])), width=np.abs(np.diff(region[0])), height=np.abs(np.diff(region[1])), linewidth=2.0, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
                ax.add_patch(rect)
                text = 'Region %d%s:\nEfficiency=%.2f%%' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", efficiency_regions_efficiencies[region_index] * 100.0)
                widths = [max(region[0]) - min(region[0]), max(region[1]) - min(region[1])]
                if widths[1] * 2.0 < widths[0]:
                    rotation = 'horizontal'
                elif widths[0] * 2.0 < widths[1]:
                    rotation = 'vertical'
                else:
                    rotation = 'horizontal'
                ax.text(np.sum(region[0]) / 2.0, np.sum(region[1]) / 2.0, text, horizontalalignment='center', verticalalignment='center', fontsize=9, rotation=rotation)
                rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
                ax.add_patch(rect)
                ax.set_xlim(plot_range[0])
                ax.set_ylim(plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.grid()
                title = 'Region %d%s: Efficiency per pixel\nfor %s\n(%d Pixels)' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name, region_n_pixels)
                ax.set_title(title)
                ax.set_xlabel('Efficiency [%]')
                ax.set_ylabel('#')
                ax.set_yscale('log')
                ax.set_xlim([-0.5, 100.5])
                ax.hist(efficiency_regions_stat_pixel_efficiency_hist[region_index], bins=100, range=(0, 100))  # Histogram not masked pixel stat_2d_efficiency_hist
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.bar(x=range(hist_charge_indices[-1] + 1), height=efficiency_regions_count_1d_charge_hist[region_index][:hist_charge_indices[-1] + 1] / np.sum(efficiency_regions_count_1d_charge_hist[region_index][:hist_charge_indices[-1] + 1]).astype(np.float32), align='center')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                add_value_labels(ax=ax)
                title = 'Region %d%s: Charge distribution\nfor %s\n(%d Pixels)' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name, region_n_pixels)
                ax.set_title(title)
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.bar(x=range(max_frame), height=efficiency_regions_count_1d_frame_hist[region_index] / np.sum(efficiency_regions_count_1d_frame_hist[region_index]).astype(np.float32), align='center')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                add_value_labels(ax=ax)
                title = 'Region %d%s: Frame distribution\nfor %s\n(%d Pixels)' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name, region_n_pixels)
                ax.set_title(title)
                ax.set_yscale('log')
                output_pdf.savefig(fig)

                total_n_hits = np.sum(np.multiply(efficiency_regions_count_1d_cluster_size_hist[region_index], range(efficiency_regions_count_1d_cluster_size_hist[region_index].shape[0])))
                total_n_clusters = np.sum(efficiency_regions_count_1d_cluster_size_hist[region_index])

                x = np.arange(efficiency_regions_count_1d_cluster_size_hist[region_index].shape[0] - 1) + 1
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.bar(x, efficiency_regions_count_1d_cluster_size_hist[region_index][1:], align='center')
                title = 'Region %d%s: Cluster size distribution\nfor %s\n(%d Pixels)' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name, region_n_pixels)
                ax.set_title(title)
                ax.set_xlabel('Cluster size')
                ax.set_ylabel('#')
                ax.grid()
                ax.set_yscale('log')
                ax.set_ylim(ymin=1e-1)
                output_pdf.savefig(fig)

                max_bins = min(10, efficiency_regions_count_1d_cluster_size_hist[region_index].shape[0] - 1)
                x = np.arange(max_bins) + 1
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.bar(x, efficiency_regions_count_1d_cluster_size_hist[region_index][1:max_bins + 1], align='center')
                title = 'Region %d%s: Cluster size distribution\nfor %s\n(%d Pixels)' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name, region_n_pixels)
                ax.set_title(title)
                ax.set_xlabel('Cluster size')
                ax.set_ylabel('#')
                ax.xaxis.set_ticks(x)
                ax.grid()
                ax.set_yscale('linear')
                ax.set_ylim(ymin=0.0)
                output_pdf.savefig(fig)
                ax.autoscale()
                ax.set_yscale('log')
                ax.set_ylim(ymin=1e-1)
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                analyze_cluster_shapes_1d = [1, 3, 5, 6, 9, 13, 14, 7, 11, 19, 261, 15, 95, 783, 4959]
                cluster_shape_hist = efficiency_regions_count_1d_cluster_shape_hist[region_index][analyze_cluster_shapes_1d]
                remaining_clusters = total_n_clusters - np.sum(cluster_shape_hist)
                cluster_shape_hist = np.r_[cluster_shape_hist, remaining_clusters]
                analyze_cluster_shapes_1d = np.r_[analyze_cluster_shapes_1d, -1]
                x = np.arange(analyze_cluster_shapes_1d.shape[0])
                ax.bar(x, cluster_shape_hist, align='center')
                ax.xaxis.set_ticks(x)
                fig.subplots_adjust(bottom=0.2)
                ax.set_xticklabels([cluster_shape_strings[i] for i in analyze_cluster_shapes_1d])
                ax.tick_params(axis='x', labelsize=7)
                title = 'Region %d%s: Cluster shape distribution\nfor %s\n(%d Pixels)' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name, region_n_pixels)
                ax.set_title(title)
                ax.set_xlabel('Cluster shape')
                ax.set_ylabel('#')
                ax.grid()
                ax.set_yscale('linear')
                ax.set_ylim(ymin=0.0)
                output_pdf.savefig(fig)
                ax.autoscale()
                ax.set_yscale('log')
                ax.set_ylim(ymin=1e-1)
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                efficiency_regions_count_1d_total_angle_hist_edges[region_index] = efficiency_regions_count_1d_total_angle_hist_edges[region_index] * 1000.0  # convert to mrad
                bin_centers = (efficiency_regions_count_1d_total_angle_hist_edges[region_index][1:] + efficiency_regions_count_1d_total_angle_hist_edges[region_index][:-1]) / 2.0
                width = np.diff(bin_centers)[0]
                ax.bar(x=bin_centers, height=efficiency_regions_count_1d_total_angle_hist[region_index] / np.sum(efficiency_regions_count_1d_total_angle_hist[region_index]).astype(np.float32), width=width, align='center')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                title = 'Region %d%s: Total track angle distribution\nfor %s\n(%d Pixels)' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name, region_n_pixels)
                ax.set_title(title)
                ax.set_yscale('log')
                ax.set_xlabel('Track angle [mrad]')
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                efficiency_regions_count_1d_alpha_angle_hist_edges[region_index] = efficiency_regions_count_1d_alpha_angle_hist_edges[region_index] * 1000.0  # convert to mrad
                bin_centers = (efficiency_regions_count_1d_alpha_angle_hist_edges[region_index][1:] + efficiency_regions_count_1d_alpha_angle_hist_edges[region_index][:-1]) / 2.0
                width = np.diff(bin_centers)[0]
                ax.bar(x=bin_centers, height=efficiency_regions_count_1d_alpha_angle_hist[region_index] / np.sum(efficiency_regions_count_1d_alpha_angle_hist[region_index]).astype(np.float32), align='center', width=width)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                title = 'Region %d%s: Alpha track angle distribution\nfor %s\n(%d Pixels)' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name, region_n_pixels)
                ax.set_title(title)
                ax.set_yscale('log')
                ax.set_xlabel('Track angle [mrad]')
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                efficiency_regions_count_1d_beta_angle_hist_edges[region_index] = efficiency_regions_count_1d_beta_angle_hist_edges[region_index] * 1000.0  # convert to mrad
                bin_centers = (efficiency_regions_count_1d_beta_angle_hist_edges[region_index][1:] + efficiency_regions_count_1d_beta_angle_hist_edges[region_index][:-1]) / 2.0
                width = np.diff(bin_centers)[0]
                ax.bar(x=bin_centers, height=efficiency_regions_count_1d_beta_angle_hist[region_index] / np.sum(efficiency_regions_count_1d_beta_angle_hist[region_index]).astype(np.float32), width=width, align='center')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                title = 'Region %d%s: Beta track angle distribution\nfor %s\n(%d Pixels)' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name, region_n_pixels)
                ax.set_title(title)
                ax.set_yscale('log')
                ax.set_xlabel('Track angle [mrad]')
                output_pdf.savefig(fig)

                # in-pixel plots
                local_x_map, local_y_map = actual_dut.map_to_primitive_cell(x=local_x, y=local_y)
                primitive_cell_local_pos = np.unique(np.column_stack((local_x_map, local_y_map)), axis=0)
                primitive_cell_x_size = (np.divmod(np.max(primitive_cell_local_pos[:, 0]), actual_dut.column_size)[0] + 1) * actual_dut.column_size
                primitive_cell_y_size = (np.divmod(np.max(primitive_cell_local_pos[:, 1]), actual_dut.row_size)[0] + 1) * actual_dut.row_size
                # x = np.unique(np.r_[primitive_cell_local_pos[:, 0], primitive_cell_local_pos[:, 0] + primitive_cell_x_size, primitive_cell_local_pos[:, 0] - primitive_cell_x_size])
                # y = np.unique(np.r_[primitive_cell_local_pos[:, 1], primitive_cell_local_pos[:, 1] + primitive_cell_y_size, primitive_cell_local_pos[:, 1] - primitive_cell_y_size])
                # xv_enclosed, yv_enclosed = np.meshgrid(x, y, sparse=False)
                # xv_enclosed = np.ravel(xv_enclosed)
                # yv_enclosed = np.ravel(yv_enclosed)
                x = np.unique(np.r_[primitive_cell_local_pos[:, 0], primitive_cell_local_pos[:, 0] + primitive_cell_x_size, primitive_cell_local_pos[:, 0] - primitive_cell_x_size, primitive_cell_local_pos[:, 0] + 2 * primitive_cell_x_size, primitive_cell_local_pos[:, 0] - 2 * primitive_cell_x_size].round(3))
                y = np.unique(np.r_[primitive_cell_local_pos[:, 1], primitive_cell_local_pos[:, 1] + primitive_cell_y_size, primitive_cell_local_pos[:, 1] - primitive_cell_y_size, primitive_cell_local_pos[:, 1] + 2 * primitive_cell_y_size, primitive_cell_local_pos[:, 1] - 2 * primitive_cell_y_size].round(3))
                xv, yv = np.meshgrid(x, y, sparse=False)
                xv = np.ravel(xv)
                yv = np.ravel(yv)
                pixel_center_data_in_pixel = np.column_stack((xv, yv))
                # is_neighbor = np.zeros_like(xv, dtype=np.bool)
                # for i in range(xv.shape[0]):
                #     for j in range(yv_enclosed.shape[0]):
                #         if xv_enclosed[j] == xv[i] and yv_enclosed[j] == yv[i]:
                #             is_neighbor[i] = True
                #             break
                bin_indices_in_pixel = np.indices(efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index].shape).reshape(2, -1).T
                hist_2d_x_edges_in_pixel = np.linspace(efficiency_regions_in_pixel_plot_range[0][0], efficiency_regions_in_pixel_plot_range[0][1], efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index].shape[0] + 1)
                hist_2d_y_edges_in_pixel = np.linspace(efficiency_regions_in_pixel_plot_range[1][0], efficiency_regions_in_pixel_plot_range[1][1], efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index].shape[1] + 1)
                x_bin_centers_in_pixel = (hist_2d_x_edges_in_pixel[1:] + hist_2d_x_edges_in_pixel[:-1]) / 2
                y_bin_centers_in_pixel = (hist_2d_y_edges_in_pixel[1:] + hist_2d_y_edges_in_pixel[:-1]) / 2
                y_meshgrid_in_pixel, x_meshgrid_in_pixel = np.meshgrid(y_bin_centers_in_pixel, x_bin_centers_in_pixel)

                _, regions_in_pixel, ridge_vertices_in_pixel, vertices_in_pixel = beam_telescope_analysis.tools.analysis_utils.voronoi_finite_polygons_2d(points=pixel_center_data_in_pixel, dut_extent=[-2 * primitive_cell_x_size, 3 * primitive_cell_x_size, -2 * primitive_cell_y_size, 3 * primitive_cell_y_size])

                select_bins_in_pixel = (efficiency_regions_count_in_pixel_tracks_with_hit_2d_hist[region_index] > 0)
                residual_vectors_x_in_pixel = x_meshgrid_in_pixel[select_bins_in_pixel] + efficiency_regions_stat_in_pixel_x_residuals_2d_hist[region_index][select_bins_in_pixel]
                residual_vectors_y_in_pixel = y_meshgrid_in_pixel[select_bins_in_pixel] + efficiency_regions_stat_in_pixel_y_residuals_2d_hist[region_index][select_bins_in_pixel]
                residual_vectors_in_pixel = np.column_stack((np.ravel(residual_vectors_x_in_pixel), np.ravel(residual_vectors_y_in_pixel)))
                bin_center_data_in_pixel_sel = np.column_stack((np.ravel(x_meshgrid_in_pixel[select_bins_in_pixel]), np.ravel(y_meshgrid_in_pixel[select_bins_in_pixel])))
                bin_indices_in_pixel_sel = bin_indices_in_pixel[np.ravel(select_bins_in_pixel)]
                pixel_center_in_pixel_kd_tree = cKDTree(pixel_center_data_in_pixel)
                _, residual_to_pixel_center_in_pixel_sel = pixel_center_in_pixel_kd_tree.query(residual_vectors_in_pixel)
                _, bin_center_to_pixel_center_in_pixel_sel = pixel_center_in_pixel_kd_tree.query(bin_center_data_in_pixel_sel)
                color_indices = np.linspace(0.0, 1.0, num=100)
                cmap = copy(cm.get_cmap('tab20'))
                cmap.set_bad('w')
                rgb_colors = np.array([cmap(color) for color in color_indices])
                valid_color_indices = np.r_[0, np.unique(np.where(rgb_colors[:-1] != rgb_colors[1:])[0]) + 1]
                num_colors = len(valid_color_indices)
                # select and reorder colors
                color_indices = color_indices[valid_color_indices]
                color_indices[::2] = np.roll(color_indices[::2], int(num_colors / 4))
                rgb_colors = rgb_colors[valid_color_indices]
                rgb_colors[::2, :] = np.roll(rgb_colors[::2, :], int(num_colors / 4), axis=0)
                color_index_array = np.full(shape=pixel_center_data_in_pixel.shape[0], dtype=np.int8, fill_value=-1)
                count_index_array = np.zeros(shape=pixel_center_data_in_pixel.shape[0], dtype=np.uint32)
                effective_pixels_2d = np.full(shape=efficiency_regions_count_in_pixel_tracks_with_hit_2d_hist[region_index].shape, dtype=np.int32, fill_value=-1)
                color_index = 0
                # x_res = (hist_2d_edges[0][1] - hist_2d_edges[0][0])
                # y_res = (hist_2d_edges[1][1] - hist_2d_edges[1][0])
                for pixel_index, pixel_position in enumerate(pixel_center_data_in_pixel):
                    res_center_col_row_pair_data_indices = np.where(residual_to_pixel_center_in_pixel_sel == pixel_index)[0]
                    # res_center_col_row_pair_data_positions = bin_center_data_sel[res_center_col_row_pair_data_indices]
                    actual_bin_col_row_indices = bin_indices_in_pixel_sel[res_center_col_row_pair_data_indices]
                    bin_center_data_indices = np.where(bin_center_to_pixel_center_in_pixel_sel == pixel_index)[0]
                    count_index_array[pixel_index] = np.count_nonzero(bin_center_data_indices)
                    # bin_center_data_positions = bin_center_data_sel[bin_center_data_indices]
                    # index_0_pixel = np.array((bin_center_data_positions[:, 0] - hist_2d_edges[0][0] - x_res / 2) / x_res, dtype=np.int)
                    # index_1_pixel = np.array((bin_center_data_positions[:, 1] - hist_2d_edges[1][0] - y_res / 2) / y_res, dtype=np.int)
                    actual_pixel_bin_indices_in_pixel = bin_indices_in_pixel_sel[bin_center_data_indices]
                    other_pixel_indices = effective_pixels_2d[actual_pixel_bin_indices_in_pixel[:, 0], actual_pixel_bin_indices_in_pixel[:, 1]]
                    other_colors = color_index_array[other_pixel_indices]
                    # change color if same color is already occurring inside pixel region
                    num_repeats = 0
                    while color_index % num_colors in other_colors and num_repeats < num_colors - 1:
                        color_index += 1
                        num_repeats += 1
                    # index_0 = np.array((res_center_col_row_pair_data_positions[:, 0] - hist_2d_edges[0][0] - x_res / 2) / x_res, dtype=np.int)
                    # index_1 = np.array((res_center_col_row_pair_data_positions[:, 1] - hist_2d_edges[1][0] - y_res / 2) / y_res, dtype=np.int)
                    color_index_array[pixel_index] = color_index % num_colors
                    effective_pixels_2d[actual_bin_col_row_indices[:, 0], actual_bin_col_row_indices[:, 1]] = pixel_index
                    color_index += 1
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                for pixel_index, pixel_position in enumerate(pixel_center_data_in_pixel):
                    if count_index_array[pixel_index]:
                        ax.plot(pixel_position[0], pixel_position[1], markersize=1.0, marker='o', alpha=1.0, color=rgb_colors[color_index_array[pixel_index]], markeredgecolor='k', markeredgewidth=0.1)
                effective_color_2d = color_indices[color_index_array[effective_pixels_2d]]
                effective_color_2d = np.ma.masked_where(effective_pixels_2d == -1, effective_color_2d)
                title = 'Region %d%s: Effective pixel locations\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, effective_color_2d.T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=1.0, cmap=cmap, aspect=1.0, show_colorbar=False)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlabel("column [$\mathrm{\mu}$m]")
                ax.set_ylabel("row [$\mathrm{\mu}$m]")
                # for x_val in vlines:
                #     ax.axvline(x=x_val, color='r', alpha=mesh_alpha, linewidth=1.0)
                # for y_val in hlines:
                #     ax.axhline(y=y_val, color='r', alpha=mesh_alpha, linewidth=1.0)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                # for better readability allow masking of entries that are zero
                count_in_pixel_hits_2d_hist_masked_tmp = np.ma.array(efficiency_regions_count_in_pixel_hits_2d_hist[region_index], mask=(efficiency_regions_count_in_pixel_hits_2d_hist[region_index] == 0))
                count_in_pixel_tracks_2d_hist_masked_tmp = np.ma.array(efficiency_regions_count_in_pixel_tracks_2d_hist[region_index], mask=(efficiency_regions_count_in_pixel_tracks_2d_hist[region_index] == 0))
                count_in_pixel_tracks_with_hit_2d_hist_masked_tmp = np.ma.array(efficiency_regions_count_in_pixel_tracks_with_hit_2d_hist[region_index], mask=(efficiency_regions_count_in_pixel_tracks_with_hit_2d_hist[region_index] == 0))

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                try:
                    z_max = np.ceil(np.percentile(count_in_pixel_hits_2d_hist_masked_tmp.compressed(), q=95.00))
                except IndexError:
                    z_max = 1
                title = 'Region %d%s: In-pixel hit density\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, count_in_pixel_hits_2d_hist_masked_tmp.T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=z_max, aspect=1.0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                try:
                    z_max = np.ceil(np.percentile(count_in_pixel_tracks_2d_hist_masked_tmp.compressed(), q=95.00))
                except IndexError:
                    z_max = 1
                title = 'Region %d%s: In-pixel track density\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, count_in_pixel_tracks_2d_hist_masked_tmp.T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=z_max, aspect=1.0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                try:
                    z_max = np.ceil(np.percentile(count_in_pixel_tracks_with_hit_2d_hist_masked_tmp.compressed(), q=95.00))
                except IndexError:
                    z_max = 1
                title = 'Region %d%s: In-pixel track density with associated hit\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, count_in_pixel_tracks_with_hit_2d_hist_masked_tmp.T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=z_max, aspect=1.0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                z_min = 0.0
                title = 'Region %d%s: In-pixel efficiency\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index].T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0, aspect=1.0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                z_min = 0.0
                title = 'Region %d%s: In-pixel efficiency\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index].T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0, aspect=1.0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                # Add in pixel regions
                in_pixel_resolution = np.absolute(np.diff(efficiency_regions_in_pixel_hist_extent)[:-1]) / np.array(efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index].shape)
                for w, loc in zip(widths_in_pixel_regions, center_location_in_pixel_regions):
                    center_region_indices = np.ceil((np.array(loc) - np.array([efficiency_regions_in_pixel_hist_extent[0], efficiency_regions_in_pixel_hist_extent[2]])) / in_pixel_resolution)
                    center_region_extent = np.array(w) / in_pixel_resolution
                    center_region_selection = [np.int(np.ceil(center_region_indices[0]) - np.ceil(center_region_extent[0] / 2.0)),
                                               np.int(np.ceil(center_region_indices[0]) + np.ceil(center_region_extent[0] / 2.0)),
                                               np.int(np.ceil(center_region_indices[1]) - np.ceil(center_region_extent[1] / 2.0)),
                                               np.int(np.ceil(center_region_indices[1]) + np.ceil(center_region_extent[1] / 2.0))]
                    rect = matplotlib.patches.Rectangle((loc[0] - w[0] / 2.0, loc[1] - w[1] / 2.0), w[0], w[1], linewidth=1.2, edgecolor=in_pixel_mesh_color, facecolor='none')
                    ax.add_patch(rect)
                    mean = np.mean(efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index][center_region_selection[0]:center_region_selection[1], center_region_selection[2]:center_region_selection[3]].compressed())
                    ax.text(loc[0], loc[1], '%.2f%%' % mean, horizontalalignment='center', verticalalignment='center', fontsize=9)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                z_min = 0.0
                title = 'Region %d%s: In-pixel efficiency\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index].T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=95.0, z_max=100.0, aspect=1.0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                z_min = 0.0
                title = 'Region %d%s: In-pixel efficiency\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index].T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=95.0, z_max=100.0, aspect=1.0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                # Add in pixel regions
                in_pixel_resolution = np.absolute(np.diff(efficiency_regions_in_pixel_hist_extent)[:-1]) / np.array(efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index].shape)
                for w, loc in zip(widths_in_pixel_regions, center_location_in_pixel_regions):
                    center_region_indices = np.ceil((np.array(loc) - np.array([efficiency_regions_in_pixel_hist_extent[0], efficiency_regions_in_pixel_hist_extent[2]])) / in_pixel_resolution)
                    center_region_extent = np.array(w) / in_pixel_resolution
                    center_region_selection = [np.int(np.ceil(center_region_indices[0]) - np.ceil(center_region_extent[0] / 2.0)),
                                               np.int(np.ceil(center_region_indices[0]) + np.ceil(center_region_extent[0] / 2.0)),
                                               np.int(np.ceil(center_region_indices[1]) - np.ceil(center_region_extent[1] / 2.0)),
                                               np.int(np.ceil(center_region_indices[1]) + np.ceil(center_region_extent[1] / 2.0))]
                    rect = matplotlib.patches.Rectangle((loc[0] - w[0] / 2.0, loc[1] - w[1] / 2.0), w[0], w[1], linewidth=1.2, edgecolor=in_pixel_mesh_color, facecolor='none')
                    ax.add_patch(rect)
                    mean = np.mean(efficiency_regions_stat_in_pixel_efficiency_2d_hist[region_index][center_region_selection[0]:center_region_selection[1], center_region_selection[2]:center_region_selection[3]].compressed())
                    ax.text(loc[0], loc[1], '%.2f%%' % mean, horizontalalignment='center', verticalalignment='center', fontsize=9)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                z_max = hist_residuals_indices[-1] + 1
                title = 'Region %d%s: In-pixel mean residuals\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, efficiency_regions_stat_in_pixel_residuals_2d_hist[region_index].T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=z_max, aspect=1.0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                z_max = hist_charge_indices[-1] + 1
                title = 'Region %d%s: In-pixel mean charge\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, efficiency_regions_stat_in_pixel_charge_2d_hist[region_index].T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=z_max, aspect=1.0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                z_max = hist_charge_indices[-1] + 1
                title = 'Region %d%s: In-pixel mean charge\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, efficiency_regions_stat_in_pixel_charge_2d_hist[region_index].T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=z_max, aspect=1.0, plot_projection=True, n_bins_projections=(10, 10))
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                z_max = max_frame
                title = 'Region %d%s: In-pixel mean frame\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                plot_2d_pixel_hist(fig, ax, efficiency_regions_stat_in_pixel_frame_2d_hist[region_index].T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=z_max, aspect=1.0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                # plot cluster size average (cluster sizes 1 - 4 only)
                z_min = 1.0
                z_max = 4.0
                # set1_cmap = copy(cm.get_cmap("Set1", 9))
                # new_colors = set1_cmap(np.linspace(0, 1, 9))
                # new_cmap = colors.ListedColormap(new_colors[1:5], name="cluster_colormap")
                # new_cmap.set_over('k')
                cmap = copy(cm.get_cmap("viridis", 256))
                cmap.set_over('magenta')
                title = 'Region %d%s: In-pixel mean cluster size\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                _, cbar = plot_2d_pixel_hist(fig, ax, efficiency_regions_stat_in_pixel_cluster_size_2d_hist[region_index].T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=z_max, cmap=cmap, aspect=1.0)
                cbar.set_ticks(range(1, 5))
                cbar.set_ticklabels(['1', '2', '3', '4'])
                cbar.set_label("cluster size")
                # cbar.ax.tick_params(length=0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                z_min = -0.5
                z_max = len(efficiency_regions_analyze_cluster_shapes) - 0.5
                set1_cmap = copy(cm.get_cmap("Set1", 9))
                new_colors = set1_cmap(np.linspace(0, 1, 9))
                new_cmap = colors.ListedColormap(new_colors[1:len(efficiency_regions_analyze_cluster_shapes) + 1], name="cluster_colormap")
                new_cmap.set_over('k')
                title = 'Region %d%s: Most probable cluster shape\nfor %s' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", actual_dut.name)
                _, cbar = plot_2d_pixel_hist(fig, ax, efficiency_regions_stat_in_pixel_cluster_shape_2d_hist[region_index].T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=z_max, cmap=new_cmap, aspect=1.0)
                cbar.set_ticks(range(len(efficiency_regions_analyze_cluster_shapes)))
                cbar.set_ticklabels([cluster_shape_strings[i] for i in efficiency_regions_analyze_cluster_shapes])
                cbar.set_label("cluster shape")
                cbar.ax.tick_params(length=0)
                # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                output_pdf.savefig(fig)

                plot_cluster_shapes = [[1], [3], [5], [3, 5], [13, 14, 7, 11], [15]]
                for shapes in plot_cluster_shapes:
                    if not np.all(np.in1d(shapes, efficiency_regions_analyze_cluster_shapes)):
                        continue
                    count_in_pixel_cluster_shape_2d_hist_tmp = np.zeros_like(efficiency_regions_count_in_pixel_cluster_shape_2d_hist[region_index][:, :, 0])
                    for shape in shapes:
                        count_in_pixel_cluster_shape_2d_hist_tmp += efficiency_regions_count_in_pixel_cluster_shape_2d_hist[region_index][:, :, np.where(shape == np.array(efficiency_regions_analyze_cluster_shapes))[0][0]]
                    count_in_pixel_cluster_shape_2d_hist_masked_tmp = np.ma.array(count_in_pixel_cluster_shape_2d_hist_tmp, mask=(count_in_pixel_cluster_shape_2d_hist_tmp == 0))
                    count_in_pixel_cluster_shape_2d_hist_masked_tmp /= efficiency_regions_count_in_pixel_tracks_with_hit_2d_hist[region_index]

                    fig = Figure()
                    _ = FigureCanvas(fig)
                    ax = fig.add_subplot(111)
                    try:
                        z_max = np.percentile(count_in_pixel_cluster_shape_2d_hist_masked_tmp.compressed(), q=99.0)
                    except IndexError:
                        z_max = 1
                    title = 'Region %d%s: In-pixel density for cluster shapes %s\nfor %s\n(%d Pixels)' % (region_index + 1, (" (" + efficiency_regions_names[region_index] + ")") if efficiency_regions_names[region_index] else "", ', '.join([str(shape) for shape in shapes]), actual_dut.name, region_n_pixels)
                    plot_2d_pixel_hist(fig, ax, count_in_pixel_cluster_shape_2d_hist_masked_tmp.T, efficiency_regions_in_pixel_hist_extent, title=title, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=z_max, aspect=1.0)
                    # _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color, point_size=mesh_point_size, point_alpha=mesh_alpha, point_color=mesh_color)
                    _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices_in_pixel, vertices=vertices_in_pixel, points=pixel_center_data_in_pixel, show_points=True, line_width=in_pixel_mesh_line_width, line_style=in_pixel_mesh_line_style, line_alpha=in_pixel_mesh_alpha, line_color=in_pixel_mesh_color, point_size=in_pixel_mesh_point_size, point_alpha=in_pixel_mesh_alpha, point_color=in_pixel_mesh_color)
                    ax.set_xlim(efficiency_regions_in_pixel_plot_range[0])
                    ax.set_ylim(efficiency_regions_in_pixel_plot_range[1])
                    output_pdf.savefig(fig)

    else:
        logging.warning('Cannot create efficiency plots: all pixels are masked.')


def purity_plots(telescope, pure_hit_hist, hit_hist, purity, purity_sensor, actual_dut_index, minimum_hit_density, hist_extent, cut_distance, mask_zero=True, output_pdf=None):
    actual_dut = telescope[actual_dut_index]
    if not output_pdf:
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
    plot_2d_pixel_hist(fig, ax, pure_hit_hist.T, hist_extent, title='Pure hit density for %s (%d Pure Hits)' % (actual_dut.name, n_pure_hit_hist), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]")
    output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig, ax, hit_hist.T, hist_extent, title='Hit density for %s (%d Hits)' % (actual_dut.name, n_hits_hit_density), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]")
    output_pdf.savefig(fig)

    if np.any(~purity.mask):
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        z_min = np.ma.min(purity)
        if z_min == 100.0:  # One cannot plot with 0 z axis range
            z_min = 90.0
        plot_2d_pixel_hist(fig, ax, purity.T, hist_extent, title='Purity for %s (%d Entries)' % (actual_dut.name, n_hits_purity), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)
        output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title('Purity per pixel for %s: %1.4f +- %1.4f%%' % (actual_dut.name, np.ma.mean(purity), np.ma.std(purity)))
        ax.set_xlabel('Purity [%]')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_xlim([-0.5, 100.5])
        ax.hist(purity.ravel()[purity.ravel().mask != 1], bins=100, range=(0, 100))  # Histogram not masked pixel purity
        output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title('Sensor Purity for %s: %1.4f +- %1.4f%%' % (actual_dut.name, np.ma.mean(purity_sensor), np.ma.std(purity_sensor)))
        ax.set_xlabel('Purity [%]')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_xlim([-0.5, 100.5])
        ax.hist(purity_sensor.ravel()[purity_sensor.ravel().mask != 1], bins=100, range=(0, 100))  # Histogram not masked pixel purity
        output_pdf.savefig(fig)

    else:
        logging.warning('Cannot create purity plots: all pixels are masked')


def plot_track_angle(input_track_angle_file, select_duts, output_pdf_file=None, dut_names=None):
    ''' Plotting track slopes.

    Parameters
    ----------
    input_track_angle_file : string
        Filename of the track angle file.
    select_duts : iterable
        Selecting DUTs that will be processed.
    output_pdf_file: string
        Filename of the output PDF file.
        If None, deduce filename from input track angle file.
    dut_names : iterable of strings
        Names of the DUTs. If None, the DUT index will be used.
    '''
    logging.info('Plotting track angle histogram')
    if output_pdf_file is None:
        output_pdf_file = os.path.splitext(input_track_angle_file)[0] + '.pdf'

    select_duts_mod = select_duts[:]
    select_duts_mod.insert(0, None)

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_track_angle_file, mode="r") as in_file_h5:
            for actual_dut_index in select_duts_mod:
                if actual_dut_index is not None:
                    if dut_names is not None:
                        dut_name = dut_names[actual_dut_index]
                    else:
                        dut_name = "DUT%d" % actual_dut_index
                else:
                    dut_name = None
                for angle in ["Global_total", "Global_alpha", "Global_beta", "Local_total", "Local_alpha", "Local_beta"]:
                    if actual_dut_index is None and "Local" in angle:
                        continue
                    node = in_file_h5.get_node(in_file_h5.root, '%s_track_angle_hist%s' % (angle, ("_DUT%d" % actual_dut_index) if actual_dut_index is not None else ""))
                    track_angle_hist = node[:]
                    edges = in_file_h5.get_node(in_file_h5.root, '%s_track_angle_edges%s' % (angle, ("_DUT%d" % actual_dut_index) if actual_dut_index is not None else ""))[:]
                    edges = edges * 1000  # conversion to mrad
                    mean = node._v_attrs.mean * 1000  # conversion to mrad
                    sigma = node._v_attrs.sigma * 1000  # conversion to mrad
                    amplitude = node._v_attrs.amplitude
                    bin_center = (edges[1:] + edges[:-1]) / 2.0
                    fig = Figure()
                    _ = FigureCanvas(fig)
                    ax = fig.add_subplot(111)
                    # fixing bin width in plotting
                    width = (edges[1:] - edges[:-1])
                    ax.bar(bin_center, track_angle_hist, label='Angular Distribution%s' % ((" for %s" % dut_name) if actual_dut_index is not None else ""), width=width, color='b', align='center')
                    x_gauss = np.linspace(np.min(edges), np.max(edges), num=1000)
                    ax.plot(x_gauss, beam_telescope_analysis.tools.analysis_utils.gauss(x_gauss, amplitude, mean, sigma), color='r', label='Gauss-Fit:\nMean: %.5f mrad,\nSigma: %.5f mrad' % (mean, sigma))
                    ax.set_ylabel('#')
                    ax.set_title('%s angular distribution of fitted tracks%s' % (angle.replace("_", " "), (" for %s" % dut_name) if actual_dut_index is not None else ""))
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
            ax_col.set_xlabel("Track distance [$\mathrm{\mu}$m]")
            fig_row = Figure()
            _ = FigureCanvas(fig_row)
            ax_row = fig_row.add_subplot(111)
            ax_row.set_title("Row residual correlation")
            ax_row.set_ylabel("Correlation of row residuals")
            ax_row.set_xlabel("Track distance [$\mathrm{\mu}$m]")
            for dut_index, actual_dut_index in enumerate(select_duts):
                dut_name = dut_names[actual_dut_index] if dut_names else ("DUT" + str(actual_dut_index))
                for direction in ["column", "row"]:
                    correlations = []
                    ref_res_node = in_file_h5.get_node(in_file_h5.root, '%s_residuals_reference_DUT%d' % (direction.title(), actual_dut_index))
                    res_node = in_file_h5.get_node(in_file_h5.root, '%s_residuals_DUT%d' % (direction.title(), actual_dut_index))
                    edges = res_node.attrs.edges
                    bin_centers = (edges[1:] + edges[:-1]) / 2.0
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
                    fit, pcov = curve_fit(f=corr_f, xdata=bin_centers, ydata=correlations, p0=[0.1, 0.5, np.pi / 2.0, 1.0, pixel_size[actual_dut_index][0 if direction == "column" else 1]], bounds=[[0.0, 0.0, 0.0, -np.inf, 0.0], [1.0, 1.0, 2 * np.pi, np.inf, bin_centers[-1]]], sigma=np.sqrt(np.array(res_count)), absolute_sigma=False)
                    x = np.linspace(start=bin_centers[0], stop=bin_centers[-1], num=1000, endpoint=True)
                    fitted_correlations = corr_f(x, *fit)

                    fig = Figure()
                    _ = FigureCanvas(fig)
                    ax = fig.add_subplot(111)
                    data_label = 'Data'
                    ax.scatter(bin_centers, correlations, marker='s', label=data_label)
#                     yerr = correlations/np.sqrt(np.array(res_count))
#                     ax.errorbar(bin_centers, correlations, yerr=yerr, marker='s', linestyle='None')
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
                    ax.set_title("%s residual correlation for %s" % (direction.title(), dut_name))
                    ax.set_ylabel("Correlation of %s residuals" % (direction,))
                    ax.set_xlabel("Track distance [$\mathrm{\mu}$m]")
                    ax.legend(loc="upper right")
                    output_pdf.savefig(fig)

                    if direction == "column":
                        seleced_axis = ax_col
                    else:
                        seleced_axis = ax_row
                    seleced_axis.plot(x, fitted_correlations, color='k', label='Fit: %s' % (dut_name,), zorder=2 * len(select_duts) + dut_index)
                    seleced_axis.scatter(bin_centers, correlations, label='Data: %s' % (dut_name,), marker='s', zorder=len(select_duts) + dut_index)

            ax_col.legend(loc="upper right")
            ax_row.legend(loc="upper right")
            output_pdf.savefig(fig_col)
            output_pdf.savefig(fig_row)


def plot_kf_alignment(output_alignment_file, telescope, output_pdf_file):
    aligment_parameter_names = ['translation_x', 'translation_y', 'translation_z', 'rotation_alpha', 'rotation_beta', 'rotation_gamma']
    with tb.open_file(output_alignment_file, mode='r') as in_file_h5:
        with PdfPages(output_pdf_file) as output_pdf:
            import matplotlib.pyplot as plt
            from matplotlib import colors, cm
            cmap = cm.get_cmap('tab10')

            # Read Chi2 and p-value
            track_chi2_table = in_file_h5.get_node('/TrackChi2')
            track_chi2 = track_chi2_table[:]
            pvalue_table = in_file_h5.get_node('/TrackpValue')
            track_pvalue = pvalue_table[:]
            max_track_chi2 = track_chi2_table._v_attrs.max_track_chi2

            n_tracks_processed = []  # number of procssed tracks for all DUTs

            for par in range(6):
                par_name = aligment_parameter_names[par]
                for dut_index, dut in enumerate(telescope):
                    try:
                        aligment_parameter_table = in_file_h5.get_node('/Alignment_DUT%i' % dut_index)
                        deviation_cuts = aligment_parameter_table._v_attrs.deviation_cuts
                        aligment_parameters = aligment_parameter_table[:]
                        x = np.arange(aligment_parameters[:][par_name].shape[0])
                        y = aligment_parameters[:][par_name]
                        yerr = aligment_parameters[:][par_name + '_err']
                        alpha = aligment_parameters[:]['annealing_factor']
                        y_delta = aligment_parameters[:][par_name + '_delta']
                        y_delta = y_delta[~np.isnan(y_delta)]
                        fig = Figure()
                        _ = FigureCanvas(fig)
                        ax = fig.add_subplot(111)
                        ax.plot(x, y, label='%s for %s' % (par_name, dut.name), color=cmap(dut_index))
                        ax.fill_between(x, y - yerr, y + yerr, color=cmap(dut_index), alpha=0.4)
                        ax.legend()
                        ax.set_title('%s for %s' % (par_name, dut.name))
                        ax.set_ylabel('%s' % par_name)
                        ax.set_xlabel('# processed tracks')
                        ax.grid()
                        output_pdf.savefig(fig)

                        fig = Figure()
                        _ = FigureCanvas(fig)
                        ax = fig.add_subplot(111)
                        ax.plot(x, y - y[0], label='%s for %s' % (par_name, dut.name), color=cmap(dut_index))
                        ax.legend()
                        ax.set_title('%s for %s' % (par_name, dut.name))
                        ax.set_ylabel('Rel. Change of %s' % par_name)
                        ax.set_xlabel('# processed tracks')
                        ax.grid()
                        output_pdf.savefig(fig)

                        fig = Figure()
                        _ = FigureCanvas(fig)
                        ax = fig.add_subplot(111)
                        ax.plot(x, yerr, label='%s for %s' % (par_name, dut.name), color=cmap(dut_index))
                        ax.legend()
                        ax.set_title('%s for %s' % (par_name, dut.name))
                        ax.set_ylabel('Error of %s' % par_name)
                        ax.set_xlabel('# processed tracks')
                        ax.grid()
                        output_pdf.savefig(fig)

                        fig = Figure()
                        _ = FigureCanvas(fig)
                        ax = fig.add_subplot(111)
                        ax.hist(y_delta, bins=np.linspace(y_delta.min(), np.percentile(y_delta, q=97.0), 100), color=cmap(dut_index))
                        ax.axvline(x=deviation_cuts[par], ls='--', color='grey')
                        ax.set_title('%s for %s' % (par_name, dut.name))
                        ax.set_ylabel('Deviation of %s for %s' % (par_name, dut.name))
                        ax.set_ylabel('#')
                        output_pdf.savefig(fig)

                        fig = Figure()
                        _ = FigureCanvas(fig)
                        ax = fig.add_subplot(111)
                        ax.plot(np.arange(y_delta.shape[0]), y_delta, label='%s for %s' % (par_name, dut.name), color=cmap(dut_index))
                        ax.axhline(y=deviation_cuts[par], ls='--', color='grey')
                        ax.set_ylabel('#')
                        ax.set_ylabel('Deviation of %s for %s' % (par_name, dut.name))
                        ax.grid()
                        output_pdf.savefig(fig)

                        fig = Figure()
                        _ = FigureCanvas(fig)
                        ax = fig.add_subplot(111)
                        ax.set_title('Annealing Factor for %s' % dut.name)
                        ax.plot(x, alpha, label='Annealing Factor for %s' % dut.name, color=cmap(dut_index))
                        ax.axhline(y=1.0, ls='--', color='grey')
                        ax.set_xlabel('# processed tracks')
                        ax.set_xlabel('Annealing Factor')
                        ax.grid()
                        output_pdf.savefig(fig)

                    except tb.NoSuchNodeError: # in case DUT has not been aligned do not plot
                        continue

            # Plot chi2 distribution
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.hist(track_chi2, bins=np.linspace(0.0, 100.0, 200))
            ax.set_xlim(0.0, 100.0)
            ax.axvline(x=max_track_chi2, ls='--', color='grey')
            ax.set_xlabel('Track $\chi^2$/ndf')
            ax.set_ylabel('#')
            ax.grid()
            output_pdf.savefig(fig)

            # Plot chi2 distribution, narrow
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.hist(track_chi2, bins=np.linspace(0.0, 20.0, 200))
            ax.set_xlim(0.0, 20.0)
            ax.axvline(x=max_track_chi2, ls='--', color='grey')
            ax.set_xlabel('Track $\chi^2$/ndf')
            ax.set_ylabel('#')
            ax.grid()
            output_pdf.savefig(fig)

            # Plot pvalue distribution
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.hist(track_pvalue, bins=np.linspace(0.0, 1.0,100))
            ax.set_xlabel('Track pValue')
            ax.set_ylabel('#')
            ax.grid()
            output_pdf.savefig(fig)

            # Plot number of processed tracks
            for dut_index, dut in enumerate(telescope):
                try:
                    aligment_parameter_table = in_file_h5.get_node('/Alignment_DUT%i' % dut_index)
                    n_tracks_processed.append(aligment_parameter_table._v_attrs.n_tracks_processed)
                except tb.NoSuchNodeError:
                    n_tracks_processed.append(0)
            xtick_labels = ['DUT %i' % i for i in range(len(telescope))]
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.bar(x=np.arange(len(telescope)), height=n_tracks_processed, width=0.8, align='center')
            ax.set_ylabel('# processed Tracks')
            ax.set_xticks(range(len(telescope)))
            ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
            output_pdf.savefig(fig)
