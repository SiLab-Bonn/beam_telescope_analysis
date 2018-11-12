from __future__ import division

import logging
import re
import os.path
import warnings
from math import ceil
from itertools import cycle

import numpy as np
import tables as tb
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.artist import setp
from matplotlib.figure import Figure
import matplotlib.patches
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib import colors, cm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3d plotting
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial import cKDTree

from testbeam_analysis.telescope.telescope import Telescope
import testbeam_analysis.tools.analysis_utils
import testbeam_analysis.tools.geometry_utils

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")  # Plot backend error not important


def plot_2d_map(hist2d, plot_range, title=None, x_axis_title=None, y_axis_title=None, z_min=0, z_max=None, cmap='viridis', show_colorbar=True, output_pdf=None, gui=False, figs=None):
    if not output_pdf and not gui:
        return
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    plot_2d_pixel_hist(fig=fig, ax=ax, hist2d=hist2d, plot_range=plot_range, title=title, x_axis_title=x_axis_title, y_axis_title=y_axis_title, z_min=z_min, z_max=z_max, cmap=cmap, show_colorbar=show_colorbar)
    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)


def plot_2d_pixel_hist(fig, ax, hist2d, plot_range, title=None, x_axis_title=None, y_axis_title=None, z_min=0, z_max=None, cmap='viridis', show_colorbar=True):
    if z_max is None:
        if hist2d.all() is np.ma.masked or np.allclose(0, hist2d):  # check if masked array is fully masked
            z_max = 1
        else:
            z_max = ceil(hist2d.max())
    if isinstance(cmap, basestring):
        cmap = cm.get_cmap(cmap)
        cmap.set_bad('w')
    else:
        cmap = cmap
    im = ax.imshow(hist2d, interpolation='none', origin='lower', aspect="auto", extent=plot_range, cmap=cmap, clim=(z_min, z_max))
    if title is not None:
        ax.set_title(title)
    if x_axis_title is not None:
        ax.set_xlabel(x_axis_title)
    if y_axis_title is not None:
        ax.set_ylabel(y_axis_title)
    if show_colorbar:
        bounds = np.linspace(start=z_min, stop=z_max, num=256, endpoint=True)
        fig.colorbar(im, boundaries=bounds, ticks=np.linspace(start=z_min, stop=z_max, num=9, endpoint=True), fraction=0.04, pad=0.05)


def plot_masked_pixels(input_mask_file, pixel_size=None, dut_name=None, output_pdf_file=None, gui=False):
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

    if gui:
        figs = []

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:  # if gui, we dont want to safe empty pdf
        cmap = cm.get_cmap('viridis')
        cmap.set_bad('w')
        c_max = np.ceil(np.percentile(occupancy, 99))

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_title('Occupancy for %s' % (dut_name, ))
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
        if gui:
            leg = ax.legend(numpoints=1, bbox_to_anchor=(1.015, 1.135), loc='upper right')
            leg.get_frame().set_facecolor('none')
            figs.append(fig)
        else:
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

            # Fancy occupancy plot
            fig = Figure()
            _ = FigureCanvas(fig)
            # title of the page
            fig.suptitle('Occupancy for %s%s' % (dut_name, '\n(noisy and disabled pixels masked)' if masked_hits else ''))
            ax = fig.add_subplot(111)
            # ax.set_title('Occupancy for %s' % (dut_name, ))
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
                x_c_max = np.ceil(np.percentile(hight, 99))
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
                y_c_max = np.ceil(np.percentile(width, 99))
                axHisty.set_xlim(0, max(1, y_c_max))
            axHisty.locator_params(axis='x', nbins=3)
            axHisty.ticklabel_format(style='sci', scilimits=(0, 4), axis='x')
            axHisty.set_xlabel('#')

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
        with tb.open_file(input_cluster_file, mode='r') as input_file_h5:
            hight = None
            n_hits = 0
            n_clusters = input_file_h5.root.Clusters.nrows
            for start_index in range(0, n_clusters, chunk_size):
                cluster_n_hits = input_file_h5.root.Clusters[start_index:start_index + chunk_size]['n_hits']
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
        ax.set_title('Cluster size for %s\n(%i hits in %i clusters)' % (dut_name, n_hits, n_clusters))
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
        ax.set_title('Cluster size for %s\n(%i hits in %i clusters)' % (dut_name, n_hits, n_clusters))
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
        output_pdf_file = os.path.splitext(input_file)[0] + '.pdf'

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_file, mode='r') as in_file_h5:
            for actual_dut_index in select_duts:
                if actual_dut_index is None:
                    node = in_file_h5.get_node(in_file_h5.root, 'Clusters')
                else:
                    node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)

                    logging.info('Plotting cluster histograms for DUT%d', actual_dut_index)

                    if dut_names is not None:
                        dut_name = dut_names[actual_dut_index]
                    else:
                        dut_name = "DUT%d" % actual_dut_index

                initialize = True  # initialize the histograms
                try:
                    cluster_size_hist = in_file_h5.root.HistClusterSize[:]
                    n_clusters = np.sum(cluster_size_hist)
                    cluster_shapes_hist = in_file_h5.root.HistClusterShape[:]
                except tb.NoSuchNodeError:
                    raise ValueError()
                    n_hits = 0
                    n_clusters = 0
                    for chunk, _ in testbeam_analysis.tools.analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):

                        if actual_dut_index is None:
                            cluster_n_hits = chunk['n_hits']
                            cluster_shape = chunk['cluster_shape']
                        else:
                            cluster_n_hits = chunk['n_hits_dut_%d' % actual_dut_index]
                            cluster_shape = chunk['cluster_shape_dut_%d' % actual_dut_index]

                        max_cluster_size = np.max(cluster_n_hits)
                        # n_hits += np.sum(cluster_n_hits)
                        n_clusters += chunk.shape[0]
                        edges = np.arange(2**16)
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

                x = np.arange(cluster_size_hist.size)
                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                ax.bar(x, cluster_size_hist, align='center')
                # ax.set_title('Cluster sizes%s\n(%d hits in %d clusters)' % ((" for %s" % dut_name) if dut_name else "", n_hits, n_clusters))
                ax.set_title('Cluster sizes%s\n(%d clusters)' % ((" for %s" % dut_name) if dut_name else "", n_clusters))
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
                                    u"\u2597\u2009\u2596",  # 2 hit cluster, horizontal
                                    u"\u2004\u2596\n\u2004\u2598",  # 2 hit cluster, vertical
                                    u"\u259e",  # 2 hit cluster
                                    u"\u259a",  # 2 hit cluster
                                    u"\u2599",  # 3 hit cluster, L
                                    u"\u259f",  # 3 hit cluster
                                    u"\u259b",  # 3 hit cluster
                                    u"\u259c",  # 3 hit cluster
                                    u"\u2004\u2596\u2596\u2596",  # 3 hit cluster, horizontal
                                    u"\u2004\u2596\n\u2004\u2596\n\u2004\u2596",  # 3 hit cluster, vertical
                                    u"\u2597\u2009\u2596\n\u259d\u2009\u2598"])  # 4 hit cluster
                # ax.set_title('Cluster shapes%s\n(%d hits in %d clusters)' % ((" for %s" % dut_name) if dut_name else "", n_hits, n_clusters))
                ax.set_title('Cluster shapes%s\n(%d clusters)' % ((" for %s" % dut_name) if dut_name else "", n_clusters))
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


def plot_correlations(input_correlation_file, output_pdf_file=None, dut_names=None, gui=False):
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

    if gui:
        figs = []

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:  # if gui, we dont want to safe empty pdf
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
                    logging.warning('All correlation entries for %s are zero, do not create plots', str(node.name))
                    continue

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                cmap = cm.get_cmap('viridis')
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

                if gui:
                    figs.append(fig)
                else:
                    output_pdf.savefig(fig)

    if gui:
        return figs


def plot_hough(dut_pos, data, accumulator, offset, slope, dut_pos_limit, theta_edges, rho_edges, ref_hist_extent, dut_hist_extent, ref_name, dut_name, x_direction, reduce_background, output_pdf=None, gui=False, figs=None):
    if not output_pdf and not gui:
        return

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap('viridis')
    cmap.set_bad('w')
    ax.imshow(np.flip(np.flip(accumulator, 0), 1), interpolation="none", origin="lower", aspect="auto", cmap=cmap, extent=[np.rad2deg(theta_edges[0]), np.rad2deg(theta_edges[-1]), rho_edges[0], rho_edges[-1]])
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_title("%s correlation accumulator%s:\n%s vs. %s" % ('X' if x_direction else 'Y', " (reduced background)" if reduce_background else "", ref_name, dut_name))
    ax.set_xlabel(r'$\theta$ [degree]')
    ax.set_ylabel(r'$\rho$ [$\mathrm{\mu}$m]')

    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    aspect = 1.0  # "auto"
    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    fit_legend_entry = 'Hough: $c_0+c_1*x$\n$c_0=%.1e$\n$c_1=%.1e$' % (offset, slope)
    ax.plot(dut_pos, testbeam_analysis.tools.analysis_utils.linear(dut_pos, offset, slope), linestyle='-', alpha=0.7, color="darkorange", label=fit_legend_entry)
    ax.axvline(x=dut_pos_limit[0])
    ax.axvline(x=dut_pos_limit[1])
    ax.imshow(data, interpolation="none", origin="lower", aspect=aspect, cmap='Greys', extent=[dut_hist_extent[0], dut_hist_extent[1], ref_hist_extent[0], ref_hist_extent[1]])
    ax.set_title("%s correlation%s:\n%s vs. %s" % ('X' if x_direction else 'Y', " (reduced background)" if reduce_background else "", ref_name, dut_name))
    ax.set_xlabel("%s %s [$\mathrm{\mu}$m]" % (dut_name, "x" if x_direction else "y"))
    ax.set_ylabel("%s %s [$\mathrm{\mu}$m]" % (ref_name, "x" if x_direction else "y"))
    ax.legend(loc=0)

    if gui:
        figs.append(fig)
    else:
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


def plot_events(input_tracks_file, input_alignment_file, event_range, use_prealignment, select_duts, n_pixels, pixel_size, dut_names=None, output_pdf_file=None, gui=False):
    '''Plots the tracks (or track candidates) of the events in the given event range.

    Parameters
    ----------
    input_tracks_file : string
        Filename of the input tracks file.
    input_alignment_file : string
        Filename of the input aligment file.
    event_range : iterable
        Tuple of start event number and stop event number (excluding), e.g. (0, 100).
    use_prealignment : bool
        If True, use pre-alignment; if False, use alignment.
    select_duts : iterable
        Selecting DUTs that will be processed.
    dut_names : iterable
        Names of the DUTs. If None, generic DUT names will be used.
    n_pixels : iterable of tuples
        One tuple per DUT describing the number of pixels in column, row direction
        e.g. for 2 DUTs: n_pixels = [(80, 336), (80, 336)]
    pixel_size : iterable
        Tuple of the pixel size for column and row for every plane, e.g. [[250, 50], [250, 50]].
    output_pdf_file : string
        Filename of the output PDF file. If None, the filename is derived from the input file.
    gui: bool
        Determines whether to plot directly onto gui
    n_tracks: uint
        plots all tracks from first to n_tracks, if amount of tracks less than n_tracks, plot all
        if not None, event_range has no effect.
    '''
    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_events.pdf'

    if gui:
        figs = []

    with tb.open_file(input_alignment_file, mode="r") as in_file_h5:  # Open file with alignment data
        if use_prealignment:
            logging.info('Use pre-alignment data')
            prealignment = in_file_h5.root.PreAlignment[:]
            n_duts = prealignment.shape[0]
        else:
            logging.info('Use alignment data')
            alignment = in_file_h5.root.Alignment[:]
            n_duts = alignment.shape[0]

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            for actual_dut_index in select_duts:
                logging.info('Plotting events for DUT%d', actual_dut_index)

                dut_name = dut_names[actual_dut_index] if dut_names else ("DUT" + str(actual_dut_index))

                array = in_file_h5.get_node(in_file_h5.root, name='Tracks_DUT%d' % actual_dut_index)[:]
                tracks_chunk = testbeam_analysis.tools.analysis_utils.get_data_in_event_range(array, event_range[0], event_range[-1])
                if tracks_chunk.shape[0] == 0:
                    raise ValueError('No events found in the given range, cannot plot events!')

                fig = Figure()
                _ = FigureCanvas(fig)
                ax = fig.add_subplot(111, projection='3d')

                for dut_index in range(0, n_duts):
                    sensor_size = np.array(pixel_size[actual_dut_index]) * n_pixels[actual_dut_index]
                    x, y = np.meshgrid([-sensor_size[0] / 2.0, sensor_size[0] / 2.0], [-sensor_size[1] / 2.0, sensor_size[1] / 2.0])
                    alignment_no_rot = alignment.copy()
                    # change alpha, beta to 0 to make plotting of DUTs nicer
                    alignment_no_rot['alpha'] = 0.0
                    alignment_no_rot['beta'] = 0.0
                    plane_x, plane_y, plane_z = testbeam_analysis.tools.geometry_utils.apply_alignment(x.flatten(), y.flatten(), np.zeros(x.size),
                                                                                                       dut_index=dut_index,
                                                                                                       alignment=alignment_no_rot,
                                                                                                       inverse=False)
                    plane_x = plane_x * 1.e-3  # in mm
                    plane_y = plane_y * 1.e-3  # in mm
                    plane_z = plane_z * 1.e-3  # in mm
                    plane_x = plane_x.reshape(2, -1)
                    plane_y = plane_y.reshape(2, -1)
                    plane_z = plane_z.reshape(2, -1)
                    ax.plot_surface(plane_x, plane_y, plane_z, color='lightgray', alpha=0.3, linewidth=1.0, zorder=-1)

                colors = cycle('bgrcmyk')
                for track in tracks_chunk:
                    color = next(colors)
                    x, y, z = [], [], []
                    for dut_index in range(0, n_duts):
                        # Coordinates in global coordinate system (x, y, z)
                        hit_x_local, hit_y_local, hit_z_local = track['x_dut_%d' % dut_index], track['y_dut_%d' % dut_index], track['z_dut_%d' % dut_index]

                        if use_prealignment:
                            hit_x, hit_y, hit_z = testbeam_analysis.tools.geometry_utils.apply_alignment(hit_x_local, hit_y_local, hit_z_local,
                                                                                                         dut_index=dut_index,
                                                                                                         prealignment=prealignment,
                                                                                                         inverse=False)
                        else:
                            hit_x, hit_y, hit_z = testbeam_analysis.tools.geometry_utils.apply_alignment(hit_x_local, hit_y_local, hit_z_local,
                                                                                                         dut_index=dut_index,
                                                                                                         alignment=alignment,
                                                                                                         inverse=False)

                        hit_x = hit_x * 1.e-3  # in mm
                        hit_y = hit_y * 1.e-3  # in mm
                        hit_z = hit_z * 1.e-3  # in mm
                        x.extend(hit_x)
                        y.extend(hit_y)
                        z.extend(hit_z)

                    if np.isfinite(track['offset_x']):
                        offset = np.array((track['offset_x'], track['offset_y'], track['offset_z']))
                        slope = np.array((track['slope_x'], track['slope_y'], track['slope_z']))
                        linepts = offset * 1.e-3 + slope * 1.e-3 * np.mgrid[-150000:150000:2000j][:, np.newaxis]
                        no_fit = False
                    else:
                        no_fit = True

                    if no_fit is False:
                        ax.plot(x, y, z, 's' if track['hit_flag'] == track['quality_flag'] else 'o', color=color)
                        ax.plot3D(*linepts.T, color=color)
                    else:
                        ax.plot(x, y, z, 'x', color=color)

#                 ax.set_zlim(min(z), max(z))
                ax.set_xlabel('x [mm]')
                ax.set_ylabel('y [mm]')
                ax.set_zlabel('z [mm]')
                ax.set_title('%d tracks of %d events for %s' % (tracks_chunk.shape[0], np.unique(tracks_chunk['event_number']).shape[0], dut_name))

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
                        hist_full, edges_full = np.histogram(chi2s, range=range_full, bins=250)
                        hist_narrow, edges_narrow = np.histogram(chi2s, range=[0, 250], bins=250)
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
                ax.set_xlabel('Track Chi2 [$\mathrm{\mu m}^2$]')
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
                ax.set_xlabel('Track Chi2 [$\mathrm{\mu m}^2$]')
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
            ax.plot([fit[1], fit[1]], [0, ax.get_ylim()[1]], color='r', label='Entries: %d\n$\mathrm{RMS}=%.1f$ [$\mathrm{\mu}$m]' % (histogram.sum(), testbeam_analysis.tools.analysis_utils.get_rms_from_histogram(histogram, x)))
            gauss_fit_legend_entry = 'Gauss fit: \n$A=%.1f\pm %.1f$\n$\mathrm{\mu}=%.1f\pm %.1f$ [$\mathrm{\mu}$m]\n$\mathrm{\sigma}=%.1f\pm %.1f$ [$\mathrm{\mu}$m]' % (fit[0], np.absolute(cov[0][0] ** 0.5), fit[1], np.absolute(cov[1][1] ** 0.5), np.absolute(fit[2]), np.absolute(cov[2][2] ** 0.5))
            x_gauss = np.arange(np.floor(np.min(edges)), np.ceil(np.max(edges)), step=0.1)
            ax.plot(x_gauss, testbeam_analysis.tools.analysis_utils.gauss(x_gauss, *fit), 'r--', label=gauss_fit_legend_entry, linewidth=2)
            ax.legend(loc=0)
        ax.set_xlim([edges[0], edges[-1]])

        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)


def plot_residuals_vs_position(hist, xedges, yedges, xlabel, ylabel, title, residuals_mean=None, select=None, fit=None, cov=None, limit=None, output_pdf=None, gui=False, figs=None):
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
    ax.imshow(np.ma.masked_equal(hist, 0).T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', aspect='auto', interpolation='none')
    if residuals_mean is not None:
        res_pos = (xedges[1:] + xedges[:-1]) / 2.0
        if select is None:
            select = np.full_like(res_pos, True, dtype=np.bool)
        ax.scatter(res_pos[select], residuals_mean[select], color="blue", marker='o', label='Mean residual')
        ax.scatter(res_pos[~select], residuals_mean[~select], color="r", marker='o')
    if fit is not None:
        x_lim = np.array(ax.get_xlim(), dtype=np.float32)
        ax.plot(x_lim, testbeam_analysis.tools.analysis_utils.linear(x_lim, *fit), linestyle='-', color="darkorange", linewidth=2, label='Mean residual fit\n%.2e + %.2e x' % (fit[0], fit[1]))
    if limit is not None:
        if np.isfinite(limit[0]):
            ax.axvline(x=limit[0], linewidth=2, color='r')
        if np.isfinite(limit[1]):
            ax.axvline(x=limit[1], linewidth=2, color='r')
    ax.set_xlim([xedges[0], xedges[-1]])
    ax.set_ylim([yedges[0], yedges[-1]])
    ax.legend(loc=0)

    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)


def plot_track_density(telescope_configuration, input_tracks_file, select_duts, output_pdf_file=None, gui=False, chunk_size=1000000):
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
    gui: bool
        Determines whether to plot directly onto gui.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    logging.info('=== Plotting track density for %d DUTs ===' % len(select_duts))

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_track_density.pdf'

    if gui:
        figs = []

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            for actual_dut_index in select_duts:
                actual_dut = telescope[actual_dut_index]
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)

                logging.info('Calculating track density for %s', actual_dut.name)

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in testbeam_analysis.tools.analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):

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
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs if gui else None)

                plot_2d_map(
                    hist2d=hist_hits.T,
                    plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                    title='Hit density for %s (%d Hits)' % (actual_dut.name, n_hits),
                    x_axis_title='Column position [$\mathrm{\mu}$m]',
                    y_axis_title='Row position [$\mathrm{\mu}$m]',
                    z_min=0,
                    z_max=None,
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs if gui else None)

    if gui:
        return figs


def plot_charge_distribution(telescope_configuration, input_tracks_file, select_duts, output_pdf_file=None, gui=False, chunk_size=1000000):
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
    gui: bool
        Determines whether to plot directly onto gui.
    chunk_size : uint
        Chunk size of the data when reading from file.
    '''
    telescope = Telescope(telescope_configuration)
    logging.info('=== Plotting mean charge distribution for %d DUTs ===' % len(select_duts))

    if not output_pdf_file:
        output_pdf_file = os.path.splitext(input_tracks_file)[0] + '_mean_charge.pdf'

    if gui:
        figs = []

    with PdfPages(output_pdf_file, keep_empty=False) as output_pdf:
        with tb.open_file(input_tracks_file, mode='r') as in_file_h5:
            for actual_dut_index in select_duts:
                actual_dut = telescope[actual_dut_index]
                node = in_file_h5.get_node(in_file_h5.root, 'Tracks_DUT%d' % actual_dut_index)

                logging.info('Calculating mean charge for %s', actual_dut.name)

                initialize = True  # initialize the histograms
                for tracks_chunk, _ in testbeam_analysis.tools.analysis_utils.data_aligned_at_events(node, chunk_size=chunk_size):

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
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs if gui else None)

                plot_2d_map(
                    hist2d=stat_hits_charge_hist.T,
                    plot_range=[-dut_x_size / 2.0, dut_x_size / 2.0, dut_y_size / 2.0, -dut_y_size / 2.0],
                    title='Mean charge of hits for %s (%d Hits)' % (actual_dut.name, n_hits),
                    x_axis_title='Column position [$\mathrm{\mu}$m]',
                    y_axis_title='Row position [$\mathrm{\mu}$m]',
                    z_min=0,
                    z_max=charge_max_hits,
                    output_pdf=output_pdf,
                    gui=gui,
                    figs=figs if gui else None)


def voronoi_plot_2d(ax, ridge_vertices, vertices, points=None, show_points=False, line_width=1.0, line_alpha=1.0, line_style='solid', line_color='k', point_size=1.0, point_alpha=1.0, point_marker='.', point_color='k'):
    '''
    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot
    '''
    if show_points:
        ax.plot(points[:, 0], points[:, 1], linestyle='None', markersize=point_size, alpha=point_alpha, marker=point_marker, color=point_color)
    ridge_vertices = np.array(ridge_vertices)
    select = np.all(ridge_vertices >= 0, axis=1)
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
    p = PolyCollection(
        verts=verts,
        cmap=cmap)
    p.set_array(values)
    ax.add_collection(p)
    bounds = np.linspace(start=z_min, stop=z_max, num=255, endpoint=True)
    fig.colorbar(p, boundaries=bounds, ticks=np.linspace(start=z_min, stop=z_max, num=9, endpoint=True), fraction=0.04, pad=0.05)
    return ax.figure


def efficiency_plots(telescope, hist_2d_edges, count_hits_2d_hist, count_tracks_2d_hist, count_tracks_with_hit_2d_hist, stat_2d_x_residuals_hist, stat_2d_y_residuals_hist, stat_2d_residuals_hist, count_1d_charge_hist, stat_2d_charge_hist, stat_2d_efficiency_hist, stat_pixel_efficiency_hist, count_pixel_hits_2d_hist, efficiency, actual_dut_index, dut_extent, hist_extent, plot_range, efficiency_region, efficiency_region_efficiency, in_pixel_efficiency=None, plot_range_in_pixel=None, mask_zero=True, output_pdf=None, gui=False, figs=None):
    actual_dut = telescope[actual_dut_index]
    if not output_pdf and not gui:
        return
    # get number of entries for every histogram
    n_hits = np.sum(count_hits_2d_hist)
    n_tracks = np.sum(count_tracks_2d_hist)
    n_tracks_with_hit = np.sum(count_tracks_with_hit_2d_hist)

    # for better readability allow masking of entries that are zero
    count_hits_2d_hist_masked = np.ma.array(count_hits_2d_hist, mask=(count_hits_2d_hist == 0))
    count_tracks_2d_hist_masked = np.ma.array(count_tracks_2d_hist, mask=(count_tracks_2d_hist == 0))
    count_tracks_with_hit_2d_hist_masked = np.ma.array(count_tracks_with_hit_2d_hist, mask=(count_tracks_with_hit_2d_hist == 0))

    pixel_indices = np.indices((actual_dut.n_columns, actual_dut.n_rows)).reshape(2, -1).T
    local_x, local_y, _ = actual_dut.index_to_local_position(
        column=pixel_indices[:, 0] + 1,
        row=pixel_indices[:, 1] + 1)
    pixel_center_data = np.column_stack((local_x, local_y))
    _, regions, ridge_vertices, vertices = testbeam_analysis.tools.analysis_utils.voronoi_finite_polygons_2d(points=pixel_center_data, dut_extent=dut_extent)

    mesh_color = 'red'
    mesh_line_width = 0.5
    mesh_point_size = 0.75
    mesh_alpha = 0.5

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color='r')
    _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices, vertices=vertices, points=pixel_center_data, show_points=True, line_width=mesh_line_width, line_alpha=1.0, line_color=mesh_color, point_size=mesh_point_size, point_alpha=1.0, point_color=mesh_color)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    ax.set_title('Pixel locations for %s' % actual_dut.name)
    ax.set_xlabel("column [$\mathrm{\mu}$m]")
    ax.set_ylabel("row [$\mathrm{\mu}$m]")
    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    plot_2d_pixel_hist(fig, ax, count_hits_2d_hist_masked.T, hist_extent, title='Hit density for %s\n(%d Hits)' % (actual_dut.name, n_hits), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]")
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    plot_2d_pixel_hist(fig, ax, count_tracks_2d_hist_masked.T, hist_extent, title='Track density for %s\n(%d Tracks)' % (actual_dut.name, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]")
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    plot_2d_pixel_hist(fig, ax, count_tracks_with_hit_2d_hist_masked.T, hist_extent, title='Track density with associated hit for %s\n(%d Tracks)' % (actual_dut.name, n_tracks_with_hit), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]")
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

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
    ax.set_title('Residuals for %s' % actual_dut.name)
    ax.set_xlabel("column [$\mathrm{\mu}$m]")
    ax.set_ylabel("row [$\mathrm{\mu}$m]")
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    select_bins = (count_tracks_with_hit_2d_hist > 0)
    residual_vetors_x = x_meshgrid[select_bins] + stat_2d_x_residuals_hist[select_bins]
    residual_vetors_y = y_meshgrid[select_bins] + stat_2d_y_residuals_hist[select_bins]
    residual_vetors = np.column_stack((np.ravel(residual_vetors_x), np.ravel(residual_vetors_y)))
    bin_center_data_sel = np.column_stack((np.ravel(x_meshgrid[select_bins]), np.ravel(y_meshgrid[select_bins])))
    bin_indices_sel = bin_indices[np.ravel(select_bins)]
    pixel_center_kd_tree = cKDTree(pixel_center_data)
    _, residual_to_pixel_center_sel = pixel_center_kd_tree.query(residual_vetors)
    _, bin_center_to_pixel_center_sel = pixel_center_kd_tree.query(bin_center_data_sel)
    colors = np.linspace(0.0, 1.0, num=100)
    cmap = cm.get_cmap('tab20')
    cmap.set_bad('w')
    rgb_colors = np.array([cmap(color) for color in colors])
    valid_color_indices = np.r_[0, np.unique(np.where(rgb_colors[:-1] != rgb_colors[1:])[0]) + 1]
    num_colors = len(valid_color_indices)
    # select and reorder colors
    colors = colors[valid_color_indices]
    colors[::2] = np.roll(colors[::2], int(num_colors / 4))
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
    effective_color_2d = colors[color_index_array[effective_pixels_2d]]
    effective_color_2d = np.ma.masked_where(effective_pixels_2d == -1, effective_color_2d)
    plot_2d_pixel_hist(fig, ax, effective_color_2d.T, hist_extent, title='Effective pixel locations for %s' % actual_dut.name, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=1.0, cmap=cmap, show_colorbar=False)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_title('Effective pixel locations for %s' % actual_dut.name)
    ax.set_xlabel("column [$\mathrm{\mu}$m]")
    ax.set_ylabel("row [$\mathrm{\mu}$m]")
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    if count_pixel_hits_2d_hist is not None:
        x_resolution = np.diff(hist_2d_edges[0])[0]
        y_resolution = np.diff(hist_2d_edges[1])[0]
        pixel_sizes = np.full(pixel_center_data.shape[0], dtype=np.float32, fill_value=np.nan)
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
        colors = np.linspace(0.0, 1.0, num=100)
        cmap = cm.get_cmap('tab20')
        cmap.set_bad('w')
        rgb_colors = np.array([cmap(color) for color in colors])
        valid_color_indices = np.r_[0, np.unique(np.where(rgb_colors[:-1] != rgb_colors[1:])[0]) + 1]
        num_colors = len(valid_color_indices)
        # select and reorder colors
        colors = colors[valid_color_indices]
        colors[::2] = np.roll(colors[::2], int(num_colors / 4))
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
        plot_2d_pixel_hist(fig, ax, effective_color_2d.T, hist_extent, title='Effective pixel locations for %s' % actual_dut.name, x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=1.0, cmap=cmap, show_colorbar=False)
        rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
        ax.add_patch(rect)
        ax.set_title('Effective pixel locations for %s' % actual_dut.name)
        ax.set_xlabel("column [$\mathrm{\mu}$m]")
        ax.set_ylabel("row [$\mathrm{\mu}$m]")
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        max_region_size = 0
        for region in regions:
            max_region_size = max(max_region_size, len(region))
        region_array = np.zeros((len(regions), max_region_size), dtype=np.int32)
        for index, region in enumerate(regions):
            region_array[index] = region + [region[-1]] * (max_region_size - len(region))
        pixel_vertices = vertices[region_array]
        calculated_pixel_sizes = testbeam_analysis.tools.analysis_utils.polygon_area_multi(pixel_vertices[:, :, 0], pixel_vertices[:, :, 1])
        _, bin_edges, _ = ax.hist(pixel_sizes[np.isfinite(pixel_sizes)], bins=100, range=(0, np.ceil(max(np.max(pixel_sizes[np.isfinite(pixel_sizes)]), np.max(calculated_pixel_sizes)))), alpha=0.5, label='Measured pixel size')
        ax.hist(calculated_pixel_sizes[np.isfinite(pixel_sizes)], bins=bin_edges, alpha=0.5, label='Calculated pixel size')
        ax.set_yscale('log')
        ax.set_title('Effective pixel sizes for %s' % actual_dut.name)
        ax.set_xlabel("Pixel size [$\mathrm{\mu}$m$^2$]")
        ax.set_ylabel("#")
        ax.legend()
        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    plot_2d_pixel_hist(fig, ax, stat_2d_residuals_hist.T, hist_extent, title='Mean 2D residuals for %s' % (actual_dut.name,), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]")
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    hist_charge_95, indices = testbeam_analysis.tools.analysis_utils.hist_quantiles(hist=count_1d_charge_hist, prob=(0.0, 0.99), return_indices=True)
    ax.bar(x=range(indices[1] + 1), height=hist_charge_95[:indices[1] + 1], align='center')
    ax.set_title('Charge distribution for %s' % actual_dut.name)
    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    fig = Figure()
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
    z_max = indices[1] + 1
    plot_2d_pixel_hist(fig, ax, stat_2d_charge_hist.T, hist_extent, title='Mean charge for %s' % (actual_dut.name,), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=0.0, z_max=z_max)
    rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
    ax.add_patch(rect)
    ax.set_xlim(plot_range[0])
    ax.set_ylim(plot_range[1])
    if gui:
        figs.append(fig)
    else:
        output_pdf.savefig(fig)

    if np.any(~stat_2d_efficiency_hist.mask):
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
        z_min = 0.0
        plot_2d_pixel_hist(fig, ax, stat_2d_efficiency_hist.T, hist_extent, title='Efficiency for %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)
        rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
        ax.add_patch(rect)
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

        if efficiency_region:
            fig = Figure()
            _ = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
            z_min = 0.0
            plot_2d_pixel_hist(fig, ax, stat_2d_efficiency_hist.T, hist_extent, title='Efficiency for %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)

            for index, region in enumerate(efficiency_region):
                rect = matplotlib.patches.Rectangle(xy=(min(region[0]), min(region[1])), width=np.abs(np.diff(region[0])), height=np.abs(np.diff(region[1])), linewidth=2.0, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
                ax.add_patch(rect)
                ax.text(np.sum(region[0]) / 2.0, np.sum(region[1]) / 2.0, 'Efficiency %.2f%%' % (efficiency_region_efficiency[index] * 100.0), horizontalalignment='center', verticalalignment='center', fontsize=8)

            ax.set_xlim(plot_range[0])
            ax.set_ylim(plot_range[1])
            if gui:
                figs.append(fig)
            else:
                output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        z_min = 0.0
        plot_2d_pixel_hist(fig, ax, stat_2d_efficiency_hist.T, hist_extent, title='Efficiency for %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)
        _ = voronoi_plot_2d(ax=ax, ridge_vertices=ridge_vertices, vertices=vertices, show_points=False, line_width=mesh_line_width, line_alpha=mesh_alpha, line_color=mesh_color)
        rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
        ax.add_patch(rect)
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title('Efficiency per bin for %s: %1.2f (+%1.2f/%1.2f)%%' % (actual_dut.name, efficiency[0], efficiency[1], efficiency[2]))
        ax.set_xlabel('Efficiency [%]')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_xlim([-0.5, 100.5])
        ax.hist(stat_2d_efficiency_hist.ravel()[stat_2d_efficiency_hist.ravel().mask != 1], bins=101, range=(0, 100), align='left')  # Histogram not masked pixel stat_2d_efficiency_hist
        if gui:
            figs.append(fig)
        else:
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
        # pixel_efficiencies_bins = np.zeros(shape=stat_2d_efficiency_hist.shape, dtype=np.float32)
        # for pixel_index, pixel in enumerate(pixel_center_data):
        #     bin_center_data_indices = np.where(pixel_center_col_row_pair_index == pixel_index)[0]
        #     bin_center_data_positions = bin_center_col_row_pair_dut[bin_center_data_indices]
        #     # print bin_center_data_positions
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
        #         pixel_efficiency = np.sum(count_tracks_with_hit_2d_hist[index_0, index_1].astype(np.float32)) / np.sum(count_tracks_2d_hist[index_0, index_1].astype(np.float32)) * 100.0
        #     pixel_efficiencies.append(pixel_efficiency)
        #     pixel_efficiencies_bins[index_0, index_1] = pixel_efficiency

        # fig = Figure()
        # _ = FigureCanvas(fig)
        # ax = fig.add_subplot(111)
        # # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
        # z_min = 0.0
        # plot_2d_pixel_hist(fig, ax, pixel_efficiencies_bins.T, hist_extent, title='Efficiency per pixel for %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)
        # rect = matplotlib.patches.Rectangle(xy=(min(dut_extent[:2]), min(dut_extent[2:])), width=np.abs(np.diff(dut_extent[:2])), height=np.abs(np.diff(dut_extent[2:])), linewidth=mesh_line_width, edgecolor=mesh_color, facecolor='none', alpha=mesh_alpha)
        # ax.add_patch(rect)
        # ax.set_xlim(plot_range[0])
        # ax.set_ylim(plot_range[1])
        # if gui:
        #     figs.append(fig)
        # else:
        #     output_pdf.savefig(fig)

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        # ax.scatter(local_x, local_y, marker='.', s=mesh_point_size, alpha=mesh_alpha, color=mesh_color)
        _ = pixels_plot_2d(fig=fig, ax=ax, regions=regions, vertices=vertices, values=stat_pixel_efficiency_hist, z_max=100.0)
        ax.set_title('Efficiency per pixel for %s\n(%d Hits, %d Tracks)' % (actual_dut.name, n_hits, n_tracks))
        ax.set_xlabel("column [$\mathrm{\mu}$m]")
        ax.set_ylabel("row [$\mathrm{\mu}$m]")
        ax.set_xlim(plot_range[0])
        ax.set_ylim(plot_range[1])
        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

        logging.info("Efficient pixels (>=97%%): %d" % np.count_nonzero(stat_pixel_efficiency_hist.compressed() >= 97.0))
        logging.info("Efficient pixels (>=95%%): %d" % np.count_nonzero(stat_pixel_efficiency_hist.compressed() >= 95.0))
        logging.info("Efficient pixels (>=80%%): %d" % np.count_nonzero(stat_pixel_efficiency_hist.compressed() >= 80.0))
        logging.info("Efficient pixels (>=50%%): %d" % np.count_nonzero(stat_pixel_efficiency_hist.compressed() >= 50.0))

        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.set_title('Efficiency per pixel for %s' % (actual_dut.name,))
        ax.set_xlabel('Efficiency [%]')
        ax.set_ylabel('#')
        ax.set_yscale('log')
        ax.set_xlim([-0.5, 100.5])
        ax.hist(stat_pixel_efficiency_hist.compressed(), bins=101, range=(0, 100), align='left')  # Histogram not masked pixel stat_2d_efficiency_hist
        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

    else:
        logging.warning('Cannot create stat_2d_efficiency_hist plots, all pixels are masked')

    if in_pixel_efficiency is not None:
        fig = Figure()
        _ = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        z_min = np.ma.min(in_pixel_efficiency)
        if z_min == 100.0:  # One cannot plot with 0 z axis range
            z_min = 90.0
        plot_2d_pixel_hist(fig, ax, in_pixel_efficiency.T, plot_range_in_pixel, title='In-Pixel-Efficiency for %s' % (actual_dut.name), x_axis_title="column [$\mathrm{\mu}$m]", y_axis_title="row [$\mathrm{\mu}$m]", z_min=z_min, z_max=100.0)

        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)


def purity_plots(telescope, pure_hit_hist, hit_hist, purity, purity_sensor, actual_dut_index, minimum_hit_density, hist_extent, cut_distance, mask_zero=True, output_pdf=None, gui=False, figs=None):
    actual_dut = telescope[actual_dut_index]
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
        ax.hist(purity.ravel()[purity.ravel().mask != 1], bins=101, range=(0, 100), align='left')  # Histogram not masked pixel purity
        if gui:
            figs.append(fig)
        else:
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
        ax.hist(purity_sensor.ravel()[purity_sensor.ravel().mask != 1], bins=101, range=(0, 100), align='left')  # Histogram not masked pixel purity
        if gui:
            figs.append(fig)
        else:
            output_pdf.savefig(fig)

    else:
        logging.warning('Cannot create purity plots, since all pixels are masked')


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
                    ax.plot(x_gauss, testbeam_analysis.tools.analysis_utils.gauss(x_gauss, amplitude, mean, sigma), color='r', label='Gauss-Fit:\nMean: %.5f mrad,\nSigma: %.5f mrad' % (mean, sigma))
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
