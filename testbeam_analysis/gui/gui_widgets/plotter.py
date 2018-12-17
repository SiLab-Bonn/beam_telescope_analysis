"""
Implements a generic plotting interface which plots given figures or from plotting functions and respective data.
AnalysisPlotter is a sub-class of QWidget and can be added to any QLayout.
"""

import matplotlib
import inspect
import logging

from testbeam_analysis.gui.gui_widgets.worker import AnalysisWorker
from PyQt5 import QtWidgets, QtCore

matplotlib.use('Qt5Agg')  # Make sure that we are using QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class AnalysisPlotter(QtWidgets.QWidget):
    """
    Implements generic plotting area widget. Takes one or multiple plotting functions and their input files
    and displays figures from their return values. Supports single and multiple figures as return values.
    Also supports plotting from multiple functions at once and input of predefined figures. If figures are plotted
    from provided plotting functions, the functions are executed on an extra thread
    """

    startedPlotting = QtCore.pyqtSignal()
    finishedPlotting = QtCore.pyqtSignal()
    exceptionSignal = QtCore.pyqtSignal(Exception, str)

    def __init__(self, input_file=None, plot_func=None, figures=None, thread=None, parent=None, **kwargs):

        super(AnalysisPlotter, self).__init__(parent)

        # Main layout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)

        # Input arguments
        self.input_file = input_file
        self.plot_func = plot_func
        self.figures = figures
        self.kwargs = kwargs

        # External thread provided for plotting; must be QtCore.QThread() or None, if None, no threading.
        # After finishing, thread is deleted
        self.plotting_thread = thread

        if input_file is None and plot_func is None and figures is None:
            msg = 'Need input file and plotting function or figures to do plotting!'
            raise ValueError(msg)

        # Bool whether to plot from multiple functions at once
        multi_plot = False

        # Threading related
        if self.plotting_thread:
            self.plotting_thread.started.connect(lambda: self.startedPlotting.emit())
            self.plotting_thread.finished.connect(lambda: self.finishedPlotting.emit())
            self.plotting_thread.finished.connect(self.plotting_thread.deleteLater)

        # Multiple plot_functions with respective input_data; dicts of plotting functions and input files
        # must have same keys. If figures are given, they must be given as a dict with a key that is in the
        # plot_functions keys. If kwargs are given for the plotting functions, keyword must be in plot_functions keys.
        # Value must be dict with actual kwarg for plot function. Full example for multi plotting WITH kwargs
        # for each plotting function would look like:
        #
        #    self.input_file={'event': input_file_event, 'correlation': input_file_correlations}
        #    self.plot_func={'event': plot_events, 'correlation': plot_correlations}
        #    self.kwargs={'event': {'event_range': 40}, 'correlation':{'pixel_size':(250,50), 'dut_names':'Tel_0'}}
        #
        # which is equivalent to:
        #
        #    AnalysisPlotter(self.input_files, self.plot_func, event={'event_range': 40}, correlation={'pixel_size':(250,50), 'dut_names':'Tel_0'})

        if isinstance(self.input_file, dict) and isinstance(self.plot_func, dict):
            if sorted(self.input_file.keys()) != sorted(self.plot_func.keys()):
                msg = 'Different sets of keys! Can not assign input data to respective plotting function!'
                raise KeyError(msg)
            else:
                if self.kwargs:
                    for key in self.kwargs.keys():
                        if key not in self.plot_func.keys():
                            msg = 'Can not assign keyword %s with argument %s to any plotting function.' \
                                  ' Keyword must be in keys of plotting function dictionary: %s.' \
                                  % (key, str(self.kwargs[key]), ''.join(str(self.plot_func.keys())))
                            raise KeyError(msg)

                multi_plot = True

        # Whether to plot a single or multiple functions
        if not multi_plot:

            # Init resulting figures and worker
            self.result_figs = None
            self.plotting_worker = None

            # Check whether kwargs are are args in plot_func
            if self.kwargs:
                self.check_kwargs(self.plot_func, self.kwargs)

            if self.figures:
                # Figures are already there, just add to widget
                self.result_figs = self.figures

            # Create respective worker instance
            self._spawn_worker()

        else:

            # Init resulting figures and workers as dict, init counter
            self.result_figs = {}
            self.plotting_worker = {}
            self._finished_workers = 0

            # Check whether kwargs are are args in plot_func
            if self.kwargs:
                for key in self.kwargs.keys():
                    self.check_kwargs(self.plot_func[key], self.kwargs[key])

            self._add_multi_figs()

    def plot(self):
        """
        Starts plotting by starting self.plotting_thread or emitting self.startedPlotting signal
        """
        if self.plotting_thread:
            self.plotting_thread.start()
        else:
            self.startedPlotting.emit()

    def _spawn_worker(self, multi_plot_key=None, dummy_widget=None):
        """
        Method to create a worker for plotting and move it to self.plotting_thread. Workers are created
        with regard to whether multiple or a single plot is created. 
        
        :param multi_plot_key: Whether worker is created for specific multi_plot_key in self.plot_func.keys() or single plot
        :param dummy_widget: External widget to be plotted on for multi_plot
        """

        # Single plot
        if multi_plot_key is None:
            self.plotting_worker = AnalysisWorker(func=self._get_figs, args=multi_plot_key)

            if self.plotting_thread:
                self.plotting_worker.moveToThread(self.plotting_thread)
                self.plotting_thread.started.connect(self.plotting_worker.work)
            else:
                self.startedPlotting.connect(self.plotting_worker.work)

            if dummy_widget is None:
                self.plotting_worker.finished.connect(lambda: self._add_figs(figures=self.result_figs))
            else:
                self.plotting_worker.finished.connect(lambda: self._add_figs(figures=self.result_figs,
                                                                              external_widget=dummy_widget))

            # Connect exceptions signal
            self.plotting_worker.exceptionSignal.connect(lambda e, trc_bck: self.emit_exception(exception=e,
                                                                                                trace_back=trc_bck))

            # Connect to slot for quitting thread and clean-up
            self.plotting_worker.finished.connect(self._finish_plotting)
            self.plotting_worker.finished.connect(self.plotting_worker.deleteLater)

        # Multiple plots
        else:
            self.plotting_worker[multi_plot_key] = AnalysisWorker(func=self._get_figs, args=multi_plot_key)

            if self.plotting_thread:
                self.plotting_worker[multi_plot_key].moveToThread(self.plotting_thread)
                self.plotting_thread.started.connect(self.plotting_worker[multi_plot_key].work)
            else:
                self.startedPlotting.connect(self.plotting_worker[multi_plot_key].work)

            if dummy_widget is None:
                self.plotting_worker[multi_plot_key].finished.connect(
                    lambda: self._add_figs(figures=self.result_figs[multi_plot_key]))
            else:
                self.plotting_worker[multi_plot_key].finished.connect(
                    lambda: self._add_figs(figures=self.result_figs[multi_plot_key], external_widget=dummy_widget))

            # Connect exceptions signal
            self.plotting_worker[multi_plot_key].exceptionSignal.connect(
                lambda e, trc_bck: self.emit_exception(exception=e, trace_back=trc_bck))

            # Connect to slot for quitting thread and clean-up
            self.plotting_worker[multi_plot_key].finished.connect(self._finish_plotting)
            self.plotting_worker[multi_plot_key].finished.connect(self.plotting_worker[multi_plot_key].deleteLater)

    def _finish_plotting(self):
        """
        Quits self.plotting_thread with regard to multiple or single plot if plotting thread is provided.
        Otherwise emits finished signal
        """
        if isinstance(self.input_file, dict):
            self._finished_workers += 1
            if self._finished_workers == len(self.input_file.keys()):
                if self.plotting_thread:
                    self.plotting_thread.quit()
                else:
                    self.finishedPlotting.emit()
        else:
            if self.plotting_thread:
                self.plotting_thread.quit()
            else:
                self.finishedPlotting.emit()

    def _get_figs(self, multi_plot_key):
        """
        Actual function that is run in the worker on self.plotting_thread. Saves the result figures in self.figures
        
        :param multi_plot_key: Whether to get figures for specific muli_plot_key in self.plot_func.keys() or single plot
        """

        # Single plot
        if multi_plot_key is None:
            if not self.result_figs:
                self.result_figs = self.plot_func(self.input_file, **self.kwargs)
            else:
                pass

        # Multiple plots
        else:
            if multi_plot_key not in self.result_figs.keys():
                if multi_plot_key in self.kwargs.keys():
                    self.result_figs[multi_plot_key] = self.plot_func[multi_plot_key](self.input_file[multi_plot_key],
                                                                                      **self.kwargs[multi_plot_key])
                else:
                    self.result_figs[multi_plot_key] = self.plot_func[multi_plot_key](self.input_file[multi_plot_key])
            else:
                pass

    def check_kwargs(self, plot_func, kwargs):
        """
        Takes a function and keyword arguments passed to the init of this class and checks whether
        or not the function takes these as arguments. If not, raise TypeError with message naming function and kwarg

        :param plot_func: function
        :param kwargs: dict of keyword arguments
        """

        # Get plot_func's args
        args = inspect.getargspec(plot_func)[0]

        for kw in kwargs.keys():
            if kw not in args:
                msg = 'Plotting function %s got unexpected argument %s' % (plot_func.__name__, kw)
                raise TypeError(msg)
            else:
                pass

    def _add_figs(self, figures, external_widget=None):
        """
        Function for plotting one or multiple plots from a single plot_func.
        If the function returns multiple plots, respective widgets for navigation
        through plots are created.

        :param external_widget: None or QWidget; if None figs are plotted on self (single fig) or an internal
                                plot_widget. If QWidget figs are plotted on this widget (must have layout)

        :param figures: matplotlib.Figure() or list of such figures; adds figures to plot widget
        """

        if figures is None:
            logging.warning('No figures returned by %s. No plotting possible' % self.plot_func.__name__)
            return

        # Make list of figures if not already
        if isinstance(figures, list):
            fig_list = figures
        else:
            fig_list = [figures]

        # Check for multiple plots and init plot widget
        if len(fig_list) > 1:
            plot_widget = QtWidgets.QStackedWidget()
        else:
            # Plots will be on self or external_widget
            plot_widget = None

        # Create a dummy widget and add a figure canvas and a toolbar for each plot
        for f in fig_list:
            dummy_widget = QtWidgets.QWidget()
            dummy_layout = QtWidgets.QVBoxLayout()
            dummy_widget.setLayout(dummy_layout)
            f.set_facecolor('0.99')
            canvas = FigureCanvas(f)
            canvas.setParent(self)
            toolbar = NavigationToolbar(canvas, self)
            dummy_layout.addWidget(toolbar)
            dummy_layout.addWidget(canvas)

            # Handle plot_widget and amount of figs
            if isinstance(plot_widget, QtWidgets.QStackedWidget):  # Multiple figs
                plot_widget.addWidget(dummy_widget)
            else:  # Single fig
                if external_widget is None:  # Plot on self
                    self.main_layout.addWidget(dummy_widget)
                else:  # Plot on external_widget
                    external_widget.layout().addWidget(dummy_widget)

        # If more than one fig make navigation widgets and add everything to respective widgets
        if isinstance(plot_widget, QtWidgets.QStackedWidget):

            # Add plot widget to external widget or self
            if external_widget is None:
                self.main_layout.addWidget(plot_widget)
            else:
                external_widget.layout().addWidget(plot_widget)

            # Create buttons to navigate through different plots
            layout_btn = QtWidgets.QHBoxLayout()
            btn_forward = QtWidgets.QPushButton()
            btn_back = QtWidgets.QPushButton()
            icon_forward = btn_forward.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward)
            icon_back = btn_back.style().standardIcon(QtWidgets.QStyle.SP_ArrowBack)
            btn_forward.setIcon(icon_forward)
            btn_back.setIcon(icon_back)
            btn_forward.setIconSize(QtCore.QSize(40, 40))
            btn_back.setIconSize(QtCore.QSize(40, 40))
            label_count = QtWidgets.QLabel('1 of %d' % plot_widget.count())
            # Connect buttons
            btn_forward.clicked.connect(lambda: navigate(val=1))
            btn_back.clicked.connect(lambda: navigate(val=-1))
            # Add buttons to layout
            layout_btn.addStretch()
            layout_btn.addWidget(btn_back)
            layout_btn.addSpacing(20)
            layout_btn.addWidget(label_count)
            layout_btn.addSpacing(20)
            layout_btn.addWidget(btn_forward)
            layout_btn.addStretch()

            # Disable back button when at first plot
            if plot_widget.currentIndex() == 0:
                btn_back.setDisabled(True)

            # Add all to main or external layout
            if external_widget is None:
                self.main_layout.addLayout(layout_btn)
            else:
                external_widget.layout().addLayout(layout_btn)

            # button slot to change plots
            def navigate(val):

                if 0 <= (plot_widget.currentIndex() + val) <= plot_widget.count():
                    index = plot_widget.currentIndex() + val
                    plot_widget.setCurrentIndex(index)

                    if index == plot_widget.count() - 1:
                        btn_back.setDisabled(False)
                        btn_forward.setDisabled(True)
                    elif index == 0:
                        btn_back.setDisabled(True)
                        btn_forward.setDisabled(False)
                    else:
                        btn_forward.setDisabled(False)
                        btn_back.setDisabled(False)

                    label_count.setText('%d of %d' % (index + 1, plot_widget.count()))

                else:
                    pass

    def _add_multi_figs(self):
        """
        Function that allows plotting from multiple plot functions at once.
        Creates a tab widget and one tab for every plot function. Uses self._add_figs() to add plots
        """

        if self.figures is not None:

            if isinstance(self.figures, dict):
                pass
            else:
                msg = 'Input figures must be in dictionary! Can not assign figure(s) to respective plotting function!'
                raise KeyError(msg)

        tabs = QtWidgets.QTabWidget()

        for key in self.input_file.keys():

            dummy_widget = QtWidgets.QWidget()
            dummy_widget.setLayout(QtWidgets.QVBoxLayout())

            if self.figures is not None and key in self.figures.keys():

                # If one of the multi_plot functions already has figures, add to result figures
                if self.figures[key] is not None:
                    self.result_figs[key] = self.figures[key]

                # Create respective worker instance
                self._spawn_worker(multi_plot_key=key, dummy_widget=dummy_widget)

            else:

                # Create respective worker instance
                self._spawn_worker(multi_plot_key=key, dummy_widget=dummy_widget)

            tabs.addTab(dummy_widget, str(key).capitalize())

        self.main_layout.addWidget(tabs)

    def emit_exception(self, exception, trace_back):
        """
        Emits exception signal

        :param exception: Any Exception
        :param trace_back: traceback of the exception or error
        """

        self.exceptionSignal.emit(exception, trace_back)
