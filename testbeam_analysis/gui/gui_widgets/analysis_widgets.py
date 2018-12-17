"""
Implements generic analysis widgets to apply analysis functions on test beam data. Two separate
widgets to apply one analysis function in parallel to all input data and to apply multiple analysis functions
to multiple input data
"""

import os
import inspect
import logging
import math

from subprocess import call
from collections import OrderedDict, defaultdict
from numpydoc.docscrape import FunctionDoc
from PyQt5 import QtWidgets, QtCore, QtGui

from testbeam_analysis.gui.gui_widgets import option_widgets
from testbeam_analysis.gui.gui_widgets.plotter import AnalysisPlotter
from testbeam_analysis.gui.gui_widgets.worker import AnalysisWorker
from testbeam_analysis.gui.gui_widgets.progbar import AnalysisBar


def get_default_args(func):
    """
    Returns a dictionary of arg_name:default_values for the input function
    """
    args, _, _, defaults = inspect.getargspec(func)
    return dict(zip(args[-len(defaults):], defaults))


def get_parameter_doc(func, dtype=False):
    """
    Returns a dictionary of paramerter:pardoc for the input function
    Pardoc is either the parameter description (dtype=False) or the data type (dtype=False)
    """
    doc = FunctionDoc(func)
    pars = {}
    for par, datatype, descr in doc['Parameters']:
        if not dtype:
            pars[par] = '\n'.join(descr)
        else:
            pars[par] = datatype
    return pars


def where(name):
    """
    Finds and returns a list with system paths to an executable name
    """
    result = []
    paths = os.defpath.split(os.pathsep)
    for outerpath in paths:
        for innerpath, _, _ in os.walk(outerpath):
            path = os.path.join(innerpath, name)
    if os.path.isfile(path) and os.access(path, os.X_OK):
        result.append(os.path.normpath(path))
    return result


class AnalysisWidget(QtWidgets.QWidget):
    """
    Implements a generic analysis gui.

    There are two separated widget areas. One the left one for plotting
    and on the right for function parameter options.
    There are 3 kind of options:
      - needed ones on top
      - optional options that can be deactivated below
      - fixed option that cannot be changed
    Below this is a button to call the underlying function with given
    keyword arguments from the options.

    Introprospection is used to determine function argument types and
    documentation from the function implementation automatically.
    """

    # Signal emitted after all funcs are called. First argument is the finished analysis step, second
    # is a list of analysis steps to be enabled next
    analysisFinished = QtCore.pyqtSignal(str, list)

    # Signal emitted if exceptions occur
    exceptionSignal = QtCore.pyqtSignal(Exception, str, str, str)

    # Signal emitted when plotting is finished
    plottingFinished = QtCore.pyqtSignal(str)

    # Signal emitted when user wants to re-run analysis of respective widget
    rerunSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent, setup, options, name, tab_list=None):
        super(AnalysisWidget, self).__init__(parent)
        self.setup = setup
        self.options = options
        self.option_widgets = {}
        self.splitter_size = [parent.width() / 2, parent.width() / 2]
        self._setup()
        # Multi-threading related inits
        self.analysis_thread = QtCore.QThread()  # no parent
        self.analysis_worker = None
        self.plotting_thread = QtCore.QThread()
        self.vitables_thread = QtCore.QThread()  # no parent
        self.vitables_worker = None
        # Holds functions with kwargs
        self.calls = OrderedDict()
        # List of tabs which will be enabled after analysis
        if isinstance(tab_list, list):
            self.tab_list = tab_list
        else:
            self.tab_list = [tab_list]
        # Name of the analysis step performed in this analysis widget
        self.name = name
        # Store state of analysis widget
        self.isFinished = False
        # Store return values of functions
        self.return_values = None

    def _setup(self):
        # Plot area
        self.left_widget = QtWidgets.QWidget()
        self.plt = QtWidgets.QVBoxLayout()
        self.left_widget.setLayout(self.plt)
        # Options
        self.opt_needed = QtWidgets.QVBoxLayout()
        self.opt_optional = QtWidgets.QVBoxLayout()
        self.opt_fixed = QtWidgets.QVBoxLayout()
        # Option area
        self.layout_options = QtWidgets.QVBoxLayout()
        self.label_option = QtWidgets.QLabel('Options')
        self.layout_options.addWidget(self.label_option)
        self.layout_options.addLayout(self.opt_needed)
        self.layout_options.addLayout(self.opt_optional)
        self.layout_options.addLayout(self.opt_fixed)
        self.layout_options.addStretch(0)

        # Layout for proceed and rerun button
        self.layout_buttons = QtWidgets.QHBoxLayout()
        # Rerun, proceed button and progressbar
        self.btn_rerun = QtWidgets.QPushButton(' Re-run')
        self.btn_rerun.clicked.connect(lambda: self.rerunSignal.emit(self.name))
        self.btn_rerun.setToolTip('Re-runs current tab and resets all following, dependent tabs')
        self.btn_rerun.setVisible(False)
        icon_rerun = QtWidgets.qApp.style().standardIcon(QtWidgets.qApp.style().SP_BrowserReload)
        self.btn_rerun.setIcon(icon_rerun)
        self.analysisFinished.connect(lambda: self.btn_rerun.setVisible(True))
        self.btn_ok = QtWidgets.QPushButton('Ok')
        self.btn_ok.clicked.connect(lambda: self._call_funcs())
        self.p_bar = AnalysisBar()
        self.p_bar.setVisible(False)

        # Add buttons to layout
        self.layout_buttons.addWidget(self.btn_rerun)
        self.layout_buttons.addWidget(self.btn_ok)

        # Container widget to disable all but ok button after perfoming analysis
        self.container = QtWidgets.QWidget()
        self.container.setLayout(self.layout_options)

        # Scroll area widget
        self.scroll_widget = QtWidgets.QWidget()
        self.scroll_widget.setLayout(QtWidgets.QVBoxLayout())

        # Add container and ok button to right widget
        self.scroll_widget.layout().addWidget(self.container)
        self.scroll_widget.layout().addLayout(self.layout_buttons)

        # Make right widget scroll able
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setBackgroundRole(QtGui.QPalette.Light)
        self.scroll_area.setWidget(self.scroll_widget)

        # Make widget to hold scroll area and progressbar
        self.right_widget = QtWidgets.QWidget()
        self.right_widget.setLayout(QtWidgets.QVBoxLayout())

        self.right_widget.layout().addWidget(self.scroll_area)
        self.right_widget.layout().addWidget(self.p_bar)

        # Split plot and option area
        self.widget_splitter = QtWidgets.QSplitter(parent=self)
        self.widget_splitter.addWidget(self.left_widget)
        self.widget_splitter.addWidget(self.right_widget)
        self.widget_splitter.setSizes(self.splitter_size)
        self.widget_splitter.setChildrenCollapsible(False)

        # Add complete layout to this widget
        layout_widget = QtWidgets.QVBoxLayout()
        layout_widget.addWidget(self.widget_splitter)
        self.setLayout(layout_widget)

    def _option_exists(self, option):
        """
        Check if option is already defined
        """
        for call in self.calls.values():
            for kwarg in call:
                if option == kwarg:
                    return True
        return False

    def add_options_auto(self, func):
        """
        Inspect a function to create options for kwargs
        """

        for name in get_default_args(func):
            # Only add as function parameter if the info is not
            # given in setup/option data structures
            if name in self.setup:
                if not self._option_exists(option=name):
                    self.add_option(option=name, default_value=self.setup[name],
                                    func=func, fixed=True)
                else:
                    self.calls[func][name] = self.setup[name]
            elif name in self.options:
                if not self._option_exists(option=name):
                    self.add_option(option=name, default_value=self.options[name],
                                    func=func, fixed=True)
                else:
                    self.calls[func][name] = self.options[name]
            else:
                self.add_option(func=func, option=name)

    def add_option(self, option, func, dtype=None, name=None, optional=None, default_value=None, fixed=False,
                   tooltip=None, hidden=False):
        """
        Add an option to the gui to set function arguments

        option: str
            Function argument name
        func: function
            Function to be used for the option
        dtype: str
            Type string to select proper input method, if None determined from default parameter type
        name: str
            Name shown in gui
        optional: bool
            Show as optional option, If optional is not defined all parameters with default value
            None are set as optional. The common behavior is that None deactivates a parameter
        default_value : object
            Default value for option
        fixed : bool
            Fix option value  default value
        hidden : bool
            Whether or not to hide the option on GUI. Only available for fixed options
        """

        # Check if option exists already
        if option in self.calls[func]:
            self._delete_option(option=option, func=func)

        # Get name from argument name
        if not name:
            name = option.replace("_", " ").capitalize()

        # Get default argument value
        if default_value is None:
            default_value = get_default_args(func)[option]

        # Get parameter description from numpy style docstring
        if not tooltip:
            try:
                tooltip = get_parameter_doc(func)[option]
            except KeyError:  # No parameter docu available
                logging.warning(
                    'Parameter %s in function %s not documented', option, func.__name__)
                tooltip = None

        # Get parameter dtype from numpy style docstring
        if not dtype:
            try:
                dtype = get_parameter_doc(func, dtype=True)[option]
            except KeyError:  # No parameter docu available
                pass

        # Get dtype from default arg
        if not dtype:
            if default_value is not None:
                dtype = str(type(default_value).__name__)
            else:
                raise RuntimeError(
                    'Cannot deduce data type for %s in function %s, because no default parameter exists', option,
                    func.__name__)

        # Get optional argument from default function argument
        if optional is None and default_value is None:
            optional = True

        if not fixed:  # Option value can be changed
            try:
                widget = self._select_widget(dtype, name, default_value, optional, tooltip, func)
            except NotImplementedError:
                logging.warning('Cannot create option %s for dtype "%s" for function %s', option, dtype, func.__name__)
                return

            self._set_argument(func, option, default_value)
            self.option_widgets[option] = widget
            self.option_widgets[option].valueChanged.connect(lambda value: self._set_argument(func, option, value))

            if optional:
                self.opt_optional.addWidget(self.option_widgets[option])
            else:
                self.opt_needed.addWidget(self.option_widgets[option])
        else:  # Fixed value
            if default_value is None:
                raise RuntimeError(
                    'Cannot create fixed option without default value')
            text = QtWidgets.QLabel()
            text.setWordWrap(True)
            # Fixed options cannot be changed --> grey color
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.darkGray)
            text.setPalette(palette)
            text.setToolTip(tooltip)
            # Handle displaying list entries of default_value
            if isinstance(default_value, list):
                # Needed to get width of text
                metrics = QtGui.QFontMetrics(self.font())
                # If width of default_value as str is greater than widget, make new line for each entry
                if metrics.width(str(default_value)) > self.scroll_widget.width():
                    d_v = ['\n' + str(v) for v in default_value]
                    t = name + ':' + ''.join(d_v) + '\n'
                # If not, write list in one line
                else:
                    t = (name + ':\n' + str(default_value) + '\n')
                text.setText(t)
            else:
                text.setText(name + ':\n' + str(default_value) + '\n')
            if not hidden:
                self.opt_fixed.addWidget(text)
            self.calls[func][option] = default_value

    def _select_widget(self, dtype, name, default_value, optional, tooltip, func):
        # Create widget according to data type
        if ('scalar' in dtype and ('tuple' in dtype or 'iterable' in dtype) or
                        'int' in dtype and ('tuple' in dtype or 'iterable' in dtype) or
                ('iterable' in dtype and 'iterable of iterable' not in dtype and 'duts' not in name)) and 'quality' not in name:
            if 'range' not in name:
                widget = option_widgets.OptionMultiSlider(name=name, labels=self.setup['dut_names'],
                                                          default_value=default_value, optional=optional,
                                                          dtype=dtype, tooltip=tooltip, parent=self)
            else:
                widget = option_widgets.OptionMultiRangeBox(name=name, labels=self.setup['dut_names'],
                                                            default_value=default_value, optional=optional,
                                                            dtype=dtype, tooltip=tooltip, parent=self)

        elif ('iterable of iterable' in dtype or 'iterable' in dtype) and ('duts' in name or 'quality' in name):

            # Init labels
            labels_x = self.setup['dut_names']
            labels_y = self.setup['dut_names']

            # determine whether "iterable of iterable" or "iterable of duts"
            if 'iterable of iterable' not in dtype and 'duts' in name:
                labels_y = None

            if func.__name__ == 'alignment' and 'selection' in name.lower() or 'align' in name.lower():
                labels_x = ['Align %i.' % (i + 1) for i in range(self.setup['n_duts'])]

            elif func.__name__ == 'fit_tracks' and 'selection' in name.lower():
                labels_x = ['Fit ' + dut for dut in labels_x]

            if 'duts' in name:
                widget = option_widgets.OptionMultiCheckBox(
                    name=name, labels_x=labels_x, default_value=default_value, optional=optional,
                    tooltip=tooltip, labels_y=labels_y, parent=self)
            elif 'quality' in name:
                widget = option_widgets.OptionMultiSpinBox(
                    name=name, labels_x=labels_x, default_value=default_value, optional=optional,
                    tooltip=tooltip, labels_y=labels_y, parent=self)
        elif 'str' in dtype:
            widget = option_widgets.OptionText(
                name, default_value, optional, tooltip, parent=self)
        elif 'int' in dtype or 'float' in dtype:
            widget = option_widgets.OptionSlider(
                name, default_value, optional, tooltip, dtype, parent=self)
        elif 'bool' in dtype:
            widget = option_widgets.OptionBool(
                name, default_value, optional, tooltip, parent=self)
        else:
            raise NotImplementedError('Cannot use type %s', dtype)

        return widget

    def _delete_option(self, option, func):
        """
        Delete existing option. Needed if option is set manually.
        """

        # If option is not fixed, delete corresponding widget
        if option in self.option_widgets:
            # Delete option widget
            self.option_widgets[option].close()
            del self.option_widgets[option]
        # Update widgets
        self.opt_optional.update()
        self.opt_needed.update()
        # Delete kwarg
        del self.calls[func][option]

    def add_function(self, func):
        """
        Add an analysis function
        """

        self.calls[func] = {}
        # Add tooltip from function docstring
        doc = FunctionDoc(func)
        label_option = self.label_option.toolTip()
        self.label_option.setToolTip(label_option + '\n'.join(doc['Summary']))
        # Add function options to gui
        self.add_options_auto(func)

    def _set_argument(self, func, name, value):
        # Workaround for https://www.riverbankcomputing.com/pipermail/pyqt/2016-June/037662.html
        # Cannot transmit None for signals with string (likely also float)
        if type(value) == str and 'None' in value:
            value = None
        if type(value) == float and math.isnan(value):
            value = None
        if type(value) == list and None in value:
            value = None
        self.calls[func][name] = value

    def _call_func(self, func, kwargs):
        """
        Call an analysis function with given kwargs
        Setup info and generic options are added if needed.
        """

        # Set missing kwargs from setting data structures
        args = inspect.getargspec(func)[0]
        for arg in args:
            if arg not in self.calls[func]:
                if arg in self.setup:
                    kwargs[arg] = self.setup[arg]
                elif arg in self.options or 'file' in arg:
                    try:
                        if 'input' in arg or 'output' in arg:
                            kwargs[arg] = os.path.join(self.options['output_path'],  # self.options['working_directory']
                                                       self.options[arg])
                        else:
                            kwargs[arg] = self.options[arg]
                    except KeyError:
                        logging.error(
                            'File I/O %s not defined in settings', arg)
                else:
                    raise RuntimeError('Function argument %s not defined', arg)

        # Get functions return value
        val = func(**kwargs)

        # Most functions return None. If not None, store value
        if val is not None:
            self.return_values = val

    def _call_funcs(self):
        """
        Call all functions in a row
        """

        # Disable ok button and show progressbar
        self.btn_ok.setDisabled(True)
        self.p_bar.setVisible(True)
        self.p_bar.setBusy('Running analysis...')

        # Go to bottom of scroll area and disable widgets
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
        self.container.setDisabled(True)

        # Create worker for vitables and move to thread
        self.analysis_worker = AnalysisWorker(func=self._call_func, funcs_args=self.calls.iteritems())
        self.analysis_worker.moveToThread(self.analysis_thread)

        # Connect worker's status
        self.analysis_worker.progressSignal.connect(lambda: self.p_bar.setRange(0, len(self.calls.keys())))
        self.analysis_worker.progressSignal.connect(lambda: self.p_bar.setValue(self.p_bar.value() + 1))

        # Connect exceptions signal
        self.analysis_worker.exceptionSignal.connect(lambda e, trc_bck: self.emit_exception(exception=e,
                                                                                            trace_back=trc_bck,
                                                                                            name=self.name,
                                                                                            cause='analysis'))

        # Connect workers work method to the start of the thread, quit thread when worker finishes and clean-up
        self.analysis_thread.started.connect(self.analysis_worker.work)
        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        self.analysis_thread.finished.connect(self.emit_analysis_done)
        self.analysis_thread.finished.connect(self.analysis_worker.deleteLater)
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)

        # Start thread
        self.analysis_thread.start()

    def _connect_vitables(self, files):
        """
        Disconnects ok button from running analysis and connects to calling "ViTables".

        :param files: HDF5-file or list of HDF5-files
        """

        self.btn_ok.setDisabled(False)
        self.btn_ok.setText('Open output file(s) via ViTables')
        self.btn_ok.clicked.disconnect()
        self.btn_ok.clicked.connect(lambda: self._call_vitables(files=files))

    def _call_vitables(self, files):
        """
        Calls "ViTables" using subprocess' call.

        :param files: HDF5-file or list of HDF5-files
        """

        vitables_path = [vitables for vitables in where('vitables') if 'conda' not in vitables]

        if isinstance(files, list):
            for f in files:
                vitables_path.append(str(f))
        else:
            vitables_path.append(str(files))

        # Create worker for vitables and move to thread
        self.vitables_worker = AnalysisWorker(func=call, args=vitables_path)
        self.vitables_worker.moveToThread(self.vitables_thread)

        # Connect exceptions signal from worker on different thread to main thread
        self.vitables_worker.exceptionSignal.connect(lambda e, trc_bck: self.emit_exception(exception=e,
                                                                                            trace_back=trc_bck,
                                                                                            name=self.name,
                                                                                            cause='vitables'))
        self.vitables_worker.exceptionSignal.connect(self.vitables_thread.quit)

        # Connect workers work method to the start of the thread, quit thread when worker finishes
        self.vitables_worker.finished.connect(self.vitables_thread.quit)
        self.vitables_thread.started.connect(self.vitables_worker.work)

        # Start thread
        self.vitables_thread.start()

    def plot(self, input_file=None, plot_func=None, figures=None, **kwargs):
        """
        Function that creates the plot for the plotting area of the AnalysisWidget using AnalysisPlotter.
        See AnalysisPlotters docstring for info on how plots are created.

        :param input_file: HDF5-file or dict with HDF5-files if plotting for multiple functions or None if figures not None
        :param plot_func: function or dict of functions if plotting for multiple functions or None if figures not None
        :param figures: None, matplotlib.Figure() or list of such figures or dict of both if plotting for multiple functions
        :param kwargs: keyword arguments or keyword from dicts keys with another dict as argument if plotting for multiple functions
        """
        plot = AnalysisPlotter(input_file=input_file, plot_func=plot_func, figures=figures,
                               parent=self.left_widget, thread=self.plotting_thread, **kwargs)
        plot.startedPlotting.connect(lambda: self.p_bar.setBusy('Plotting'))
        plot.finishedPlotting.connect(self._plotting_finished)
        plot.exceptionSignal.connect(lambda e, trc_bck: self.emit_exception(exception=e, trace_back=trc_bck,
                                                                            name=self.name, cause='plotting'))
        # Start plotting
        plot.plot()

        self.plt.addWidget(plot)

    def _plotting_finished(self):
        """
        Emits plottingFinished signal
        """

        self.plottingFinished.emit(self.name)
        self.p_bar.setFinished()

    def emit_exception(self, exception, trace_back, name, cause):
        """
        Emits exception signal

        :param exception: Any Exception
        :param trace_back: traceback of the exception or error
        :param name: string of this widgets name
        :param cause: "vitables" or "analysis" or None
        """

        self.exceptionSignal.emit(exception, trace_back, name, cause)

    def emit_analysis_done(self):
        """
        Set the status of the AnalysisWidget to finished and emit corresponding signal
        """

        self.isFinished = True
        self.analysisFinished.emit(self.name, self.tab_list)


class ParallelAnalysisWidget(QtWidgets.QWidget):
    """
    AnalysisWidget for functions that need to run for every input data file.
    Creates UI with one tab widget per respective input file
    """

    # Signal emitted after all funcs are called. First argument is the finished analysis step, second
    # is a list of analysis steps to be enabled next
    analysisFinished = QtCore.pyqtSignal(str, list)

    # Signal emitted if exceptions occur
    exceptionSignal = QtCore.pyqtSignal(Exception, str, str, str)

    # Signal emitted when plotting is finished
    plottingFinished = QtCore.pyqtSignal(str)

    # Signal emitted when user wants to re-run analysis of respective widget
    rerunSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent, setup, options, name, tab_list=None):

        super(ParallelAnalysisWidget, self).__init__(parent)

        # Make main layout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)

        # Add sub-layout and rerun / ok button and progressbar
        self.sub_layout = QtWidgets.QHBoxLayout()
        self.btn_rerun = QtWidgets.QPushButton(' Re-run')
        self.btn_rerun.clicked.connect(lambda: self.rerunSignal.emit(self.name))
        self.btn_rerun.setToolTip('Re-runs current tab and resets all following, dependent tabs')
        self.btn_rerun.setVisible(False)
        icon_rerun = QtWidgets.qApp.style().standardIcon(QtWidgets.qApp.style().SP_BrowserReload)
        self.btn_rerun.setIcon(icon_rerun)
        self.analysisFinished.connect(lambda: self.btn_rerun.setVisible(True))
        self.analysisFinished.connect(lambda: self.handle_sub_layout(tab=self.tabs.currentIndex()))
        self.btn_ok = QtWidgets.QPushButton('Ok')
        self.btn_ok.clicked.connect(lambda: self._call_parallel_funcs())
        self.p_bar = AnalysisBar()
        self.p_bar.setVisible(False)

        # Set alignment in sub-layout
        self.sub_layout.addWidget(self.p_bar)
        self.sub_layout.addWidget(self.btn_rerun)
        self.sub_layout.addWidget(self.btn_ok)
        self.sub_layout.setAlignment(self.p_bar, QtCore.Qt.AlignLeading)
        self.sub_layout.setAlignment(self.btn_rerun, QtCore.Qt.AlignLeading)
        self.sub_layout.setAlignment(self.btn_ok, QtCore.Qt.AlignTrailing)

        # Tab related widgets
        self.tabs = QtWidgets.QTabWidget()
        self.tw = {}

        # Add to main layout
        self.main_layout.addWidget(self.tabs)
        self.main_layout.addLayout(self.sub_layout)

        # Initialize options and setup
        self.setup = setup
        self.options = options

        # Initialize thread and worker
        self.analysis_thread = QtCore.QThread()  # no parent
        self.analysis_worker = {}
        self._n_workers_finished = 0
        self.plotting_thread = QtCore.QThread()

        # Make dict to store all tabs calls.values() (dict) in a list with parallel function as key
        self.parallel_calls = defaultdict(list)

        # Store state of ParallelAnalysisWidget
        self.isFinished = False

        # Additional thread for vitables
        self.vitables_thread = QtCore.QThread()  # no parent
        self.vitables_worker = None

        # List in which DUTs are stored, COPY important, will eventually be altered due to user
        self.duts = setup['dut_names'][:]

        # List of tabs which will be enabled after analysis
        if isinstance(tab_list, list):
            self.tab_list = tab_list
        else:
            self.tab_list = [tab_list]
        # Name of analysis step performed in this parallel analysis widget
        self.name = name

        self._init_tabs()
        self.connect_tabs()

    def _init_tabs(self):
        """
        Initialises a tab per DUT for whose data the analysis is performed. Each tab is a AnalysisWidget instance
        """

        # Loop over number of DUTs and create tmp setup and options for each DUT
        for i in range(self.setup['n_duts']):

            tmp_setup = {}
            tmp_options = {}

            # Fill setup for i_th DUT
            for s_key in self.setup.keys():
                # Tuples and lists have DUT specific info; loop and assign the respective i_th entry to the tmp setup
                if isinstance(self.setup[s_key], list) or isinstance(self.setup[s_key], tuple):
                    tmp_setup[s_key] = self.setup[s_key][i]
                # General info valid for all DUTs
                else:
                    tmp_setup[s_key] = self.setup[s_key]

            # Fill options for i_th DUT
            for o_key in self.options.keys():
                # Tuples and lists have DUT specific info; loop and assign the respective i_th entry to the tmp options
                if isinstance(self.options[o_key], list) or isinstance(self.options[o_key], tuple):
                    tmp_options[o_key] = self.options[o_key][i]
                # General info valid for all DUTs
                else:
                    tmp_options[o_key] = self.options[o_key]

            # Create widget
            widget = AnalysisWidget(parent=self.tabs, setup=tmp_setup, options=tmp_options, name=self.name,
                                    tab_list=self.tab_list)

            # Remove buttons and progressbar from AnalysisWidget instance; ParallelAnalysisWidget has one for all
            widget.btn_ok.deleteLater()
            widget.btn_rerun.deleteLater()
            widget.p_bar.deleteLater()

            # Add to tab widget
            self.tw[self.setup['dut_names'][i]] = widget
            self.tabs.addTab(self.tw[self.setup['dut_names'][i]], self.setup['dut_names'][i])

    def connect_tabs(self):
        """
        Make connections that handle the dynamic layout of the widget
        """

        self.tabs.currentChanged.connect(lambda tab: self.handle_sub_layout(tab=tab))

        for tab_name in self.tw.keys():
            self.tw[tab_name].widget_splitter.splitterMoved.connect(
                lambda: self.handle_sub_layout(tab=self.tabs.currentIndex()))

    def resizeEvent(self, QResizeEvent):
        """
        Handle layout of widgets when re-sized

        :param QResizeEvent:
        """

        self.handle_sub_layout(tab=self.tabs.currentIndex())

    def showEvent(self, QShowEvent):
        """
        Handle layout of widgets when shown (on start-up)

        :param QShowEvent:
        """
        self.handle_sub_layout(tab=self.tabs.currentIndex())

    def handle_sub_layout(self, tab):
        """
        Handles the layout of tab; sets sizes according to splitter of underlying AnalysisWidget in tab

        :param tab: int position of tab in self.tabs whose layout is handled
        """

        # Offset in between buttons and progressbar
        offset = 10 if not self.btn_rerun.isVisible() else 5

        # Widths of tab splitter widget
        sub_widths = self.tw[self.tabs.tabText(tab)].widget_splitter.sizes()

        # Set sizes of buttons and progressbar
        self.btn_rerun.setFixedWidth(sub_widths[1] / 2 + offset)
        self.p_bar.setFixedWidth(sub_widths[0] + offset)

        # Set size of ok button according to visibility of rerun button
        if not self.btn_rerun.isVisible():
            self.btn_ok.setFixedWidth(sub_widths[1] + offset)
        else:
            self.btn_ok.setFixedWidth(sub_widths[1] / 2 + offset)

    def add_parallel_function(self, func):
        """
        Adds function func to each tab

        :param func: function to be added to the tab widgets (AnalysisWidget instances) in parallel
        """
        for i in range(self.setup['n_duts']):
            self.tw[self.setup['dut_names'][i]].add_function(func=func)

    def add_parallel_option(self, option, func, default_value=None, name=None, dtype=None, optional=None, fixed=False,
                            tooltip=None):

        """
        Add an option to the gui of each tab to set function arguments in parallel

        :param option: str
            Function argument name
        :param func: function
            Function to be used for the option
        :param dtype: str
            Type string to select proper input method, if None determined from default parameter type
        :param name: str
            Name shown in gui
        :param optional: bool
            Show as optional option, If optional is not defined all parameters with default value
            None are set as optional. The common behavior is that None deactivates a parameter
        :param default_value : object
            Default value for option
        :param fixed : bool
            Fix option value  default value
        """

        # Loop over DUTs
        for i in range(self.setup['n_duts']):

            # Whether a specific option per DUT is added or a general option valid for all DUTs
            if isinstance(default_value, list) or isinstance(default_value, tuple):
                default_value_tmp = default_value[i]
            else:
                default_value_tmp = default_value

            self.tw[self.setup['dut_names'][i]].add_option(option=option, func=func, dtype=dtype, name=name,
                                                           optional=optional, default_value=default_value_tmp,
                                                           fixed=fixed, tooltip=tooltip)

    def _call_parallel_funcs(self):
        """
        Calls the respective call_funcs method of each of the AnalysisWidgets and disables all input widgets
        """

        # Disable ok button and show progressbar
        self.btn_ok.setDisabled(True)
        self.p_bar.setVisible(True)
        self.p_bar.setBusy('Running analysis...')

        for dut in self.duts:

            # Disable widgets
            self.tw[dut].container.setDisabled(True)

            # Create worker for each analysis function and move to thread
            self.analysis_worker[dut] = AnalysisWorker(func=self.tw[dut]._call_func,
                                                       funcs_args=self.tw[dut].calls.iteritems())
            self.analysis_worker[dut].moveToThread(self.analysis_thread)

            # Connect worker's status
            self.analysis_worker[dut].progressSignal.connect(lambda: self.p_bar.setRange(0, len(self.duts)))
            self.analysis_worker[dut].progressSignal.connect(lambda: self.p_bar.setValue(self.p_bar.value() + 1))

            # Connect exceptions signal
            self.analysis_worker[dut].exceptionSignal.connect(lambda e, trc_bck: self.emit_exception(exception=e,
                                                                                                     trace_back=trc_bck,
                                                                                                     name=self.name,
                                                                                                     cause='analysis'))

            # Connect workers work method to the start of the thread, quit thread when worker finishes and clean-up
            self.analysis_thread.started.connect(self.analysis_worker[dut].work)
            self.analysis_thread.finished.connect(self.analysis_worker[dut].deleteLater)
            self.analysis_worker[dut].finished.connect(self._quit_thread)

        self.analysis_thread.finished.connect(self.emit_parallel_analysis_done)
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)

        # Start thread
        self.analysis_thread.start()

    def _quit_thread(self):
        """
        Increments the worker finished counter and finishes analysis_thread when all workers have finished
        """

        self._n_workers_finished += 1
        if self._n_workers_finished == len(self.duts):
            self.analysis_thread.quit()

    def emit_parallel_analysis_done(self):
        """
        Set the status of the ParallelAnalysisWidget to finished and emit corresponding signal
        """
        self.isFinished = True
        self.analysisFinished.emit(self.name, self.tab_list)

    def _connect_vitables(self, files):
        """
        Disconnects ok button from running analysis and connects to calling "ViTables".

        :param files: HDF5-file or list of HDF5-files
        """
        self.btn_ok.setDisabled(False)
        self.btn_ok.setText('Open output file(s) via ViTables')
        self.btn_ok.clicked.disconnect()
        self.btn_ok.clicked.connect(lambda: self._call_vitables(files=files))

    def _call_vitables(self, files):
        """
        Calls "ViTables" using subprocess' call.

        :param files: HDF5-file or list of HDF5-files
        """

#        def helper(f):
#            """Find vitables on OS and call"""
#            if platform.system() in ('Linux', 'Darwin'):  # Darwin == Mac OS
#                vitables_path = [vitables for vitables in check_output(('whereis', 'vitables')).split(' ')
#                                  if
#                                  os.path.isfile(vitables) and os.access(vitables, os.X_OK) and 'conda' not in vitables]
#            elif platform.system() == 'Windows':
#                vitables_path = [vitables for vitables in check_output(('where', 'vitables')).split(' ')
#                                  if
#                                  os.path.isfile(vitables) and os.access(vitables, os.X_OK) and 'conda' not in vitables]
#
#            if isinstance(f, list):
#                for f_ in f:
#                    vitables_path.append(str(f_))
#            else:
#                vitables_path.append(str(f))
#
#            call(vitables_path)

        vitables_path = [vitables for vitables in where('vitables') if 'conda' not in vitables]

        if isinstance(files, list):
            for f in files:
                vitables_path.append(str(f))
        else:
            vitables_path.append(str(files))

        # Create worker for vitables and move to thread
        self.vitables_worker = AnalysisWorker(func=call, args=vitables_path)
        self.vitables_worker.moveToThread(self.vitables_thread)

        # Connect exceptions signal from worker on different thread to main thread
        self.vitables_worker.exceptionSignal.connect(lambda e, trc_bck: self.emit_exception(exception=e,
                                                                                            trace_back=trc_bck,
                                                                                            name=self.name,
                                                                                            cause='vitables'))
        self.vitables_worker.exceptionSignal.connect(self.vitables_thread.quit)

        # Connect workers work method to the start of the thread, quit thread when worker finishes
        self.vitables_worker.finished.connect(self.vitables_thread.quit)
        self.vitables_thread.started.connect(self.vitables_worker.work)

        # Start thread
        self.vitables_thread.start()

    def plot(self, input_file, plot_func, dut_names=None, **kwargs):
        """
        Function that creates the plots for several input files with same plotting function for the plotting area
        of the ParallelAnalysisWidget using AnalysisPlotter. See AnalysisPlotters docstring for info on how
        plots are created.

        :param input_file: list of HDF5-files or dict of lists
        :param plot_func: function or dict of functions if plotting for multiple functions
        :param dut_names: list of dut names
        :param kwargs: keyword arguments or keyword from dicts keys with another dict as argument if plotting for multiple functions
        """

        if dut_names:
            names = dut_names
        else:
            names = self.duts

        # Counter for plots
        self._n_plots = 0
        self._n_plots_finished = 0
        self.p_bar.setBusy('Plotting')

        # Make plot widget for each DUT if DUT has exactly one input file
        for dut in names:
            input_file_tmp = [in_file for in_file in input_file if dut in in_file]
            if len(input_file_tmp) != 1:
                continue
            else:
                self._n_plots += 1
                input_file_tmp = input_file_tmp[0]
            plot = AnalysisPlotter(input_file=input_file_tmp, plot_func=plot_func,
                                   thread=self.plotting_thread, dut_name=dut, **kwargs)
            plot.finishedPlotting.connect(self._plotting_finished)
            plot.exceptionSignal.connect(lambda e, trc_bck: self.emit_exception(exception=e, trace_back=trc_bck,
                                                                                name=self.name, cause='plotting'))
            # If no thread is provided, plot instantly
            if not self.plotting_thread:
                plot.plot()

            self.tw[dut].plt.addWidget(plot)

        # If plotting thread is provided, start thread. Note that the plotting thread is quit and deleted automatically
        if self.plotting_thread:
            self.plotting_thread.start()

    def _plotting_finished(self):
        """
        Increments the plot counter and emits plottingFinished signal when counter reaches number of DUTs
        """
        self._n_plots_finished += 1
        if self._n_plots_finished == self._n_plots:
            self.p_bar.setFinished()
            self.plottingFinished.emit(self.name)

    def emit_exception(self, exception, trace_back, name, cause):
        """
        Emits exception signal

        :param exception: Any Exception
        :param trace_back: traceback of the exception or error
        :param name: string of this widgets name
        :param cause: "vitables" or "analysis" or None
        """

        self.exceptionSignal.emit(exception, trace_back, name, cause)
