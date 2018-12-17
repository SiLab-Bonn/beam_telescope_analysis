""" Defines all analysis tabs

    Each tab is for one analysis function and has function
    gui options and plotting outputs
"""

import os
import queue

from collections import OrderedDict
from PyQt5 import QtCore, QtWidgets
from testbeam_analysis.gui.gui_widgets.analysis_widgets import AnalysisWidget, ParallelAnalysisWidget
from testbeam_analysis.gui.gui_widgets.sub_windows import IPrealignmentWindow
from testbeam_analysis.hit_analysis import generate_pixel_mask, cluster_hits
from testbeam_analysis.dut_alignment import correlate_cluster, prealignment, merge_cluster_data, apply_alignment, alignment
from testbeam_analysis.track_analysis import find_tracks, fit_tracks
from testbeam_analysis.result_analysis import calculate_efficiency, calculate_residuals

# Plot related import
from testbeam_analysis.tools.plot_utils import plot_masked_pixels, plot_cluster_size, plot_correlations, plot_tracks_per_event, plot_events, plot_track_density


class NoisyPixelsTab(ParallelAnalysisWidget):
    """
    Implements the noisy pixel analysis gui
    """

    def __init__(self, parent, setup, options, name, tab_list):
        super(NoisyPixelsTab, self).__init__(parent, setup, options, name, tab_list)

        # Make options and setup class variables
        self.options = options
        self.setup = setup

        # Plotting function
        self.plot_func = plot_masked_pixels
        self.plot_kwargs = {'gui': True}

        # Make variables for input of noisy pixel function
        self.output_file = [os.path.join(options['output_path'], dut + options['noisy_suffix']) for dut in setup['dut_names']]

        self.add_parallel_function(func=generate_pixel_mask)

        self.add_parallel_option(option='input_hits_file',
                                 default_value=options['input_files'],
                                 func=generate_pixel_mask,
                                 fixed=True)
        self.add_parallel_option(option='output_mask_file',
                                 default_value=self.output_file,
                                 func=generate_pixel_mask,
                                 fixed=True)
        self.add_parallel_option(option='n_pixel',
                                 default_value=setup['n_pixels'],
                                 func=generate_pixel_mask,
                                 fixed=True)
        self.add_parallel_option(option='dut_name',
                                 default_value=setup['dut_names'],
                                 func=generate_pixel_mask,
                                 fixed=True)
        self.add_parallel_option(option='filter_size',
                                 dtype='int',
                                 func=generate_pixel_mask)

        for x in [lambda: self._connect_vitables(files=self.output_file),
                  lambda: self.plot(input_file=self.output_file,
                                    plot_func=self.plot_func,
                                    gui=True)]:
            self.analysisFinished.connect(x)

        # Add checkbox to each tab to enable skipping noisy pixel removal individually
        self.check_boxes = {}
        for dut in setup['dut_names']:
            self.check_boxes[dut] = QtWidgets.QCheckBox('Skip noisy pixel removal for %s' % dut)
            self.tw[dut].layout_options.addWidget(self.check_boxes[dut])

        self.btn_ok.disconnect()
        self.btn_ok.clicked.connect(self.proceed)

    def proceed(self):
        """
        Called when ok button is clicked. Either does analysis or skips if all noisy pixel cbs are checked 
        """
        self.check_skip()

        for key in self.tw.keys():
            self.tw[key].container.setDisabled(True)

        if self.duts:
            self._call_parallel_funcs()
        else:
            self.btn_ok.setDisabled(True)
            self.p_bar.setVisible(True)
            self.analysisFinished.emit(self.name, self.tab_list)
            self.plottingFinished.emit(self.name)
            self.p_bar.setFinished()
            self.isFinished = True

    def check_skip(self):
        """
        Checks whether noisy pixel analysis is skipped for certain DUTs, adjusts respective data and starts analysis 
        """

        # Make array with bools whether noisy pixel removal is skipped
        self.options['skip_noisy_pixel'] = [False] * self.setup['n_duts']

        # Loop over duts
        for dut in self.setup['dut_names']:
            if self.check_boxes[dut].isChecked():
                self.duts = filter(lambda a: a != dut, self.duts)
                self.output_file = filter(lambda a: dut not in a, self.output_file)
                self.options['skip_noisy_pixel'][self.setup['dut_names'].index(dut)] = True
            else:
                if dut not in self.duts:
                    self.duts.insert(self.setup['dut_names'].index(dut), dut)
                    self.output_file.insert(self.setup['dut_names'].index(dut),
                                            os.path.join(self.options['output_path'],
                                                         dut + self.options['noisy_suffix']))
                self.options['skip_noisy_pixel'][self.setup['dut_names'].index(dut)] = False


class ClusterPixelsTab(ParallelAnalysisWidget):
    """
    Implements the pixel clustering gui
    """

    def __init__(self, parent, setup, options, name, tab_list):
        super(ClusterPixelsTab, self).__init__(parent, setup, options, name, tab_list)

        self.plot_func = plot_cluster_size
        self.plot_kwargs = {'gui': True}

        self.output_file = [os.path.join(options['output_path'], dut + options['cluster_suffix']) for dut in setup['dut_names']]

        self.add_parallel_function(func=cluster_hits)

        self.add_parallel_option(option='input_hits_file',
                                 default_value=options['input_files'],
                                 func=cluster_hits,
                                 fixed=True)

        self.add_parallel_option(option='output_cluster_file',
                                 default_value=self.output_file,
                                 func=cluster_hits,
                                 fixed=True)

        self.add_parallel_option(option='dut_name',
                                 default_value=setup['dut_names'],
                                 func=cluster_hits,
                                 fixed=True)

        if options['skip_noisy_pixel']:
            for dut in setup['dut_names']:
                if not options['skip_noisy_pixel'][setup['dut_names'].index(dut)]:
                    self.tw[dut].add_option(option='input_noisy_pixel_mask_file',
                                            default_value=os.path.join(options['output_path'], dut + options['noisy_suffix']),
                                            func=cluster_hits,
                                            fixed=True)
                else:
                    pass

        for x in [lambda: self._connect_vitables(files=self.output_file),
                  lambda: self.plot(input_file=self.output_file,
                                    plot_func=self.plot_func,
                                    gui=True)]:
            self.analysisFinished.connect(x)


class PrealignmentTab(AnalysisWidget):
    """
    Implements the prealignment gui. Prealignment uses 4 functions of test beam analysis:
        - correlate cluster
        - fit correlations (prealignment)
        - merge cluster data of duts
        - apply prealignment
    """

    def __init__(self, parent, setup, options, name, tab_list):
        super(PrealignmentTab, self).__init__(parent, setup, options, name, tab_list)

        self.output_file = {'correlation': os.path.join(options['output_path'], 'Correlation.h5'),
                            'alignment': os.path.join(options['output_path'], 'Alignment.h5'),
                            'merged': os.path.join(options['output_path'], 'Merged.h5'),
                            'tracklets': os.path.join(options['output_path'], 'Tracklets_prealigned.h5')}

        self.add_function(func=correlate_cluster)
        self.add_function(func=prealignment)
        self.add_function(func=merge_cluster_data)
        self.add_function(func=apply_alignment)

        self.add_option(option='input_cluster_files',
                        default_value=[os.path.join(options['output_path'], dut + options['cluster_suffix']) for dut in setup['dut_names']],
                        func=correlate_cluster,
                        fixed=True)

        self.add_option(option='output_correlation_file',
                        default_value=self.output_file['correlation'],
                        func=correlate_cluster,
                        fixed=True)

        self.add_option(option='input_correlation_file',
                        default_value=self.output_file['correlation'],
                        func=prealignment,
                        fixed=True)

        self.add_option(option='output_alignment_file',
                        default_value=self.output_file['alignment'],
                        func=prealignment,
                        fixed=True)

        self.add_option(option='input_cluster_files',
                        default_value=[os.path.join(options['output_path'], dut + options['cluster_suffix']) for dut in setup['dut_names']],
                        func=merge_cluster_data,
                        fixed=True)

        self.add_option(option='output_merged_file',
                        default_value=self.output_file['merged'],
                        func=merge_cluster_data,
                        fixed=True)

        self.add_option(option='input_hit_file',
                        default_value=self.output_file['merged'],
                        func=apply_alignment,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=self.output_file['alignment'],
                        func=apply_alignment,
                        fixed=True)

        self.add_option(option='output_hit_file',
                        default_value=self.output_file['tracklets'],
                        func=apply_alignment,
                        fixed=True)

        self.add_option(option='use_duts',
                        func=apply_alignment,
                        default_value=list(range(setup['n_duts'])),  # Python3
                        optional=True)

        self.add_option(option='gui',
                        default_value=True,
                        func=prealignment,
                        fixed=True,
                        hidden=True)

        # Important in order to keep underlying analysis widget from making an option widget.
        # This option is changed automatically if interactive prealignment is selected.
        self.add_option(option='queue',
                        default_value=False,
                        func=prealignment,
                        fixed=True,
                        hidden=True)

        # Fix options that should not be changed
        self.add_option(option='inverse', func=apply_alignment, fixed=True)
        self.add_option(option='force_prealignment', func=apply_alignment,
                        default_value=True, fixed=True)
        self.add_option(option='no_z', func=apply_alignment, fixed=True)

        for x in [lambda: self._connect_vitables(files=self.output_file.values()),
                  lambda: self._make_plots()]:  # kwargs for correlation
            self.analysisFinished.connect(x)

        # Disconect and reconnect to check for interactive prealignment
        self.btn_ok.disconnect()
        self.btn_ok.clicked.connect(self.proceed)

        # Variable for interactive prealignment window
        self.ip_win = None

    def proceed(self):

        if not self.calls[prealignment]['non_interactive']:

            class Listener(QtCore.QObject):

                dataReceived = QtCore.pyqtSignal(list)
                closeSignal = QtCore.pyqtSignal()

                def __init__(self, q):
                    super(Listener, self).__init__()

                    self.queue = q

                def listen(self):
                    while True:
                        try:
                            a, b, c, d, e, f, g = self.queue.get()
                            self.dataReceived.emit([a, b, c, d, e, f, g])
                        except ValueError:
                            self.closeSignal.emit()
                            break

            io = {'in': queue.Queue(), 'out': queue.Queue()}
            self.ip_win = IPrealignmentWindow(io['out'], parent=self)
            listener = Listener(io['in'])
            self.thread = QtCore.QThread()  # Make class variable to avoid getting garbage collected when out of scope
            listener.moveToThread(self.thread)
            self.thread.started.connect(listener.listen)
            listener.dataReceived.connect(lambda data: self.iprealignment(data))
            listener.closeSignal.connect(self.ip_win.close)
            listener.closeSignal.connect(self.thread.quit)
            self.thread.finished.connect(listener.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            # Important to delete to not safe in calls because queue has thread lock
            self.thread.finished.connect(lambda: self._delete_option(option='queue', func=prealignment))

            self.add_option(option='queue',
                            default_value=io,
                            func=prealignment,
                            fixed=True,
                            hidden=True)

            self.thread.start()

        self._call_funcs()

    def _make_plots(self):

        # Determine the order of plotting tabs with OrderedDict
        multiple_plotting_data = OrderedDict([('correlation', self.output_file['correlation']),
                                              ('prealignment', None)])
        multiple_plotting_func = {'correlation': plot_correlations, 'prealignment': None}
        multiple_plotting_figs = {'correlation': None, 'prealignment': self.return_values}

        self.plot(input_file=multiple_plotting_data, plot_func=multiple_plotting_func,
                  figures=multiple_plotting_figs, correlation={'dut_names': self.setup['dut_names'], 'gui': True})

    def iprealignment(self, data):
        self.ip_win.update_plots(*data)
        if not self.ip_win.isActiveWindow():
            self.ip_win.showMaximized()


class TrackFindingTab(AnalysisWidget):
    """
    Implements the track finding gui
    """

    def __init__(self, parent, setup, options, name, tab_list):
        super(TrackFindingTab, self).__init__(parent, setup, options, name, tab_list)

        self.plot_func = plot_tracks_per_event
        self.plot_kwargs = {'gui': True}

        self.output_file = os.path.join(options['output_path'], 'TrackCandidates_prealignment.h5')

        self.add_function(func=find_tracks)

        self.add_option(option='input_tracklets_file',
                        default_value=os.path.join(options['output_path'], 'Tracklets_prealigned.h5'),
                        func=find_tracks,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=os.path.join(options['output_path'], 'Alignment.h5'),
                        func=find_tracks,
                        fixed=True)

        self.add_option(option='output_track_candidates_file',
                        default_value=self.output_file,
                        func=find_tracks,
                        fixed=True)

        self.add_option(option='min_cluster_distance',
                        default_value=[200.]*setup['n_duts'],
                        func=find_tracks,
                        fixed=False)

        for x in [lambda: self._connect_vitables(files=self.output_file),
                  lambda: self.plot(input_file=self.output_file, plot_func=self.plot_func, gui=True)]:
            self.analysisFinished.connect(x)


class AlignmentTab(AnalysisWidget):
    """
    Implements the alignment gui
    """

    skipAlignment = QtCore.pyqtSignal()

    def __init__(self, parent, setup, options, name, tab_list):
        super(AlignmentTab, self).__init__(parent, setup, options, name, tab_list)

        self.output_file = os.path.join(options['output_path'], 'Tracklets.h5')

        self.add_function(func=alignment)
        self.add_function(func=apply_alignment)

        self.add_option(option='input_track_candidates_file',
                        default_value=os.path.join(options['output_path'], 'TrackCandidates_prealignment.h5'),
                        func=alignment,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=os.path.join(options['output_path'], 'Alignment.h5'),
                        func=alignment,
                        fixed=True)

        self.add_option(option='align_duts',
                        func=alignment,
                        optional=True)

        self.add_option(option='selection_hit_duts',
                        func=alignment,
                        optional=True)

        self.add_option(option='selection_fit_duts',
                        func=alignment,
                        optional=True)

        self.add_option(option='selection_track_quality',
                        func=alignment,
                        optional=True)

        self.add_option(option='initial_translation',
                        default_value=False,
                        func=alignment,
                        fixed=True)

        self.add_option(option='initial_rotation',
                        default_value=setup['rotations'],
                        func=alignment,
                        fixed=True)

        self.add_option(option='input_hit_file',
                        default_value=os.path.join(options['output_path'], 'Merged.h5'),
                        func=apply_alignment,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=os.path.join(options['output_path'], 'Alignment.h5'),
                        func=apply_alignment,
                        fixed=True)

        self.add_option(option='output_hit_file',
                        default_value=self.output_file,
                        func=apply_alignment,
                        fixed=True)

        self.add_option(option='use_duts',
                        default_value=list(range(setup['n_duts'])),  # Python3
                        func=apply_alignment,
                        optional=True)

        # Connect options widgets depending on each other
        self.option_widgets['align_duts'].selectionChanged.connect(lambda sel:
                                                                   self.option_widgets[
                                                                       'selection_hit_duts'].enable_selection(
                                                                       dict(zip(sel.keys(),
                                                                                list(range(setup['n_duts']))  # Python3
                                                                                * len(sel)))))
        self.option_widgets['selection_hit_duts'].selectionChanged.connect(lambda sel:
                                                                           self.option_widgets[
                                                                               'selection_fit_duts'].enable_selection(sel))
        self.option_widgets['selection_hit_duts'].selectionChanged.connect(lambda sel:
                                                                           self.option_widgets[
                                                                               'selection_track_quality'].enable_selection(sel))

        self.btn_skip = QtWidgets.QPushButton('Skip')
        self.btn_skip.setToolTip('Skip alignment and use pre-alignment for further analysis')
        self.btn_skip.clicked.connect(lambda: self._skip_alignment())
        self.layout_options.addWidget(self.btn_skip)
        self.btn_ok.clicked.connect(lambda: self.btn_skip.setDisabled(True))

        # When global settings are updated, recreate state of alignment tab
        # If alignment is skipped
        if options['skip_alignment']:
            self._skip_alignment(ask=False)
        # If not, make connections
        else:
            for x in [lambda: self._connect_vitables(files=self.output_file),
                      lambda: self.btn_skip.deleteLater()]:
                self.analysisFinished.connect(x)

    def _skip_alignment(self, ask=True):

        if ask:
            msg = 'Do you want to skip alignment and use pre-alignment for further analysis?'
            reply = QtWidgets.QMessageBox.question(self, 'Skip alignment', msg, QtWidgets.QMessageBox.Yes,
                                                   QtWidgets.QMessageBox.Cancel)
        else:
            reply = QtWidgets.QMessageBox.Yes

        if reply == QtWidgets.QMessageBox.Yes:

            self.btn_skip.setText('Alignment skipped')
            self.btn_ok.deleteLater()
            self.container.setDisabled(True)

            if ask:
                self.skipAlignment.emit()
                self.analysisFinished.emit(self.name, self.tab_list)
        else:
            pass


class TrackFittingTab(AnalysisWidget):
    """
    Implements the track fitting gui
    """

    def __init__(self, parent, setup, options, name, tab_list):
        super(TrackFittingTab, self).__init__(parent, setup, options, name, tab_list)

        if options['skip_alignment']:
            input_tracks = os.path.join(options['output_path'], 'TrackCandidates_prealignment.h5')
            self.output_file = os.path.join(options['output_path'], 'Tracks_prealigned.h5')
        else:
            self.output_file = os.path.join(options['output_path'], 'Tracks_aligned.h5')

            self.add_function(func=find_tracks)

            self.add_option(option='input_tracklets_file',
                            default_value=os.path.join(options['output_path'], 'Tracklets.h5'),  # from alignment
                            func=find_tracks,
                            fixed=True)

            self.add_option(option='input_alignment_file',
                            default_value=os.path.join(options['output_path'], 'Alignment.h5'),
                            func=find_tracks,
                            fixed=True)

            self.add_option(option='output_track_candidates_file',
                            default_value=os.path.join(options['output_path'], 'TrackCandidates.h5'),
                            func=find_tracks,
                            fixed=True)

            self.add_option(option='min_cluster_distance',
                            default_value=[200.] * setup['n_duts'],
                            func=find_tracks,
                            fixed=False)

            input_tracks = os.path.join(options['output_path'], 'TrackCandidates.h5')

        self.add_function(func=fit_tracks)

        self.add_option(option='input_track_candidates_file',
                        default_value=input_tracks,
                        func=fit_tracks,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=os.path.join(options['output_path'], 'Alignment.h5'),
                        func=fit_tracks,
                        fixed=True)

        self.add_option(option='output_tracks_file',
                        default_value=self.output_file,
                        func=fit_tracks,
                        fixed=True)

        self.add_option(option='fit_duts',
                        func=fit_tracks,
                        optional=True)

        self.add_option(option='selection_hit_duts',
                        func=fit_tracks,
                        optional=True)

        self.add_option(option='selection_fit_duts',
                        func=fit_tracks,
                        optional=True)

        self.add_option(option='selection_track_quality',
                        func=fit_tracks,
                        optional=True)

        self.add_option(option='exclude_dut_hit',
                        func=fit_tracks,
                        default_value=True,
                        fixed=False)

        # Connect options widgets depending on each other
        self.option_widgets['fit_duts'].selectionChanged.connect(lambda sel:
                                                                 self.option_widgets[
                                                                     'selection_hit_duts'].enable_selection(sel))
        self.option_widgets['selection_hit_duts'].selectionChanged.connect(lambda sel:
                                                                           self.option_widgets[
                                                                               'selection_fit_duts'].enable_selection(sel))
        self.option_widgets['selection_hit_duts'].selectionChanged.connect(lambda sel:
                                                                           self.option_widgets[
                                                                               'selection_track_quality'].enable_selection(sel))

        # Set and fix options
        self.add_option(option='force_prealignment', func=fit_tracks,
                        default_value=options['skip_alignment'], fixed=True)
        self.add_option(option='use_correlated', func=fit_tracks,
                        default_value=False, fixed=True)
        self.add_option(option='min_track_distance', func=fit_tracks,
                        default_value=[200.] * setup['n_duts'], optional=False)

        # Check whether scatter planes in setup
        if setup['scatter_planes']:
            self.add_option(option='add_scattering_plane',
                            default_value=setup['scatter_planes'],
                            func=fit_tracks,
                            fixed=True)
        else:
            self.add_option(option='add_scattering_plane',
                            default_value=False,
                            func=fit_tracks,
                            fixed=True)

        # Determine the order of plotting tabs with OrderedDict
        multiple_plotting_data = OrderedDict([('Tracks', self.output_file), ('Tracks_per_event', self.output_file),
                                              ('Track_density', self.output_file)])

        multiple_plotting_func = {'Tracks': plot_events, 'Tracks_per_event': plot_tracks_per_event,
                                  'Track_density': plot_track_density}

        multiple_plotting_kwargs = {'Tracks': {'n_tracks': 20, 'max_chi2': 100000, 'gui': True},
                                    'Track_density': {'z_positions': setup['z_positions'],
                                                      'dim_x': [setup['n_pixels'][i][0] for i in range(setup['n_duts'])],
                                                      'dim_y': [setup['n_pixels'][i][1] for i in range(setup['n_duts'])],
                                                      'pixel_size': setup['pixel_size'],
                                                      'max_chi2': 100000, 'gui': True},
                                    'Tracks_per_event': {'gui': True}}

        for x in [lambda: self._connect_vitables(files=self.output_file),
                  lambda: self.plot(input_file=multiple_plotting_data,
                                    plot_func=multiple_plotting_func,
                                    **multiple_plotting_kwargs)]:
            self.analysisFinished.connect(x)


class ResidualTab(AnalysisWidget):
    """
    Implements the result analysis gui
    """

    def __init__(self, parent, setup, options, name, tab_list):
        super(ResidualTab, self).__init__(parent, setup, options, name, tab_list)

        if options['skip_alignment']:
            input_tracks = os.path.join(options['output_path'], 'Tracks_prealigned.h5')
        else:
            input_tracks = os.path.join(options['output_path'], 'Tracks_aligned.h5')

        self.add_function(func=calculate_residuals)

        self.output_file = os.path.join(options['output_path'], 'Residuals.h5')

        self.add_option(option='input_tracks_file',
                        default_value=input_tracks,
                        func=calculate_residuals,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=os.path.join(options['output_path'], 'Alignment.h5'),
                        func=calculate_residuals,
                        fixed=True)

        self.add_option(option='output_residuals_file',
                        default_value=self.output_file,
                        func=calculate_residuals,
                        fixed=True)

        self.add_option(option='force_prealignment',
                        default_value=options['skip_alignment'],
                        func=calculate_residuals,
                        fixed=True)

        self.add_option(option='use_duts',
                        default_value=list(range(setup['n_duts'])),  # Python3
                        func=calculate_residuals,
                        optional=True)

        self.add_option(option='gui',
                        default_value=True,
                        func=calculate_residuals,
                        fixed=True,
                        hidden=True)

        for x in [lambda: self._connect_vitables(files=self.output_file),
                  lambda: self._make_plots()]:
            self.analysisFinished.connect(x)

    def _make_plots(self):

        input_files = OrderedDict()
        plot_func = {}
        figs = {}
        ppd = int(len(self.return_values)/self.setup['n_duts'])  # plots per dut

        for i, dut in enumerate(self.setup['dut_names']):
            input_files[dut] = None
            plot_func[dut] = None
            figs[dut] = self.return_values[ppd * i: ppd * (i + 1)]  # 16 figures per DUT

        # gui=True not needed and not possible since no function is called whose args can be inspected.
        self.plot(input_file=input_files, plot_func=plot_func, figures=figs)


class EfficiencyTab(AnalysisWidget):
    """
    Implements the efficiency results tab
    """

    def __init__(self, parent, setup, options, name, tab_list):
        super(EfficiencyTab, self).__init__(parent, setup, options, name, tab_list)

        if options['skip_alignment']:
            input_tracks = os.path.join(options['output_path'], 'Tracks_prealigned.h5')
        else:
            input_tracks = os.path.join(options['output_path'], 'Tracks_aligned.h5')

        self.add_function(func=calculate_efficiency)

        self.output_file = os.path.join(options['output_path'], 'Efficiency.h5')

        self.add_option(option='input_tracks_file',
                        default_value=input_tracks,
                        func=calculate_efficiency,
                        fixed=True)

        self.add_option(option='input_alignment_file',
                        default_value=os.path.join(options['output_path'], 'Alignment.h5'),
                        func=calculate_efficiency,
                        fixed=True)

        self.add_option(option='output_efficiency_file',
                        default_value=self.output_file,
                        func=calculate_efficiency,
                        fixed=True)

        self.add_option(option='bin_size',
                        default_value=setup['pixel_size'],
                        func=calculate_efficiency,
                        fixed=True)

        self.add_option(option='sensor_size',
                        default_value=[(setup['pixel_size'][i][0] * setup['n_pixels'][i][0],
                                        setup['pixel_size'][i][1] * setup['n_pixels'][i][1])
                                       for i in range(len(setup['dut_names']))],
                        func=calculate_efficiency,
                        fixed=True)

        self.add_option(option='force_prealignment',
                        default_value=options['skip_alignment'],
                        func=calculate_efficiency,
                        fixed=True)

        self.add_option(option='use_duts',
                        default_value=list(range(setup['n_duts'])),  # Python3
                        func=calculate_efficiency,
                        optional=True)

        self.add_option(option='col_range',
                        default_value=[[0, setup['n_pixels'][i][0]] for i in range(setup['n_duts'])],
                        func=calculate_efficiency,
                        optional=True)

        self.add_option(option='row_range',
                        default_value=[[0, setup['n_pixels'][i][1]] for i in range(setup['n_duts'])],
                        func=calculate_efficiency,
                        optional=True)

        self.add_option(option='gui',
                        default_value=True,
                        func=calculate_efficiency,
                        fixed=True,
                        hidden=True)

        for x in [lambda: self._connect_vitables(files=self.output_file),
                  lambda: self._make_plots()]:
            self.analysisFinished.connect(x)

    def _make_plots(self):

        input_files = OrderedDict()
        plot_func = {}
        figs = {}
        ppd = int(len(self.return_values)/self.setup['n_duts'])  # plots per dut

        for i, dut in enumerate(self.setup['dut_names']):
            input_files[dut] = None
            plot_func[dut] = None
            figs[dut] = self.return_values[ppd * i: ppd * (i + 1)]  # 5 figures per DUT

        # gui=True not needed and not possible since no function is called whose args can be inspected.
        self.plot(input_file=input_files, plot_func=plot_func, figures=figs)
