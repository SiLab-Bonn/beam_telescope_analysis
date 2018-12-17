"""
All sub-windows of the AnalysisWindow are implemented here. One for global settings and one for exceptions
"""

import logging
import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from testbeam_analysis.gui.gui_widgets.sliders import FloatSlider
from testbeam_analysis.gui.gui_widgets.progbar import AnalysisBar
from testbeam_analysis.tools.analysis_utils import linear
from scipy.optimize import curve_fit
from PyQt5 import QtCore, QtWidgets, QtGui
from copy import deepcopy


class SettingsWindow(QtWidgets.QMainWindow):

    settingsUpdated = QtCore.pyqtSignal()

    def __init__(self, setup=None, options=None, parent=None):
        """
        Create window to set global settings for analysis
        """

        super(SettingsWindow, self).__init__(parent)

        self.window_title = 'Global settings'

        self.default_setup = {'dut_names': None,
                              'n_duts': None,
                              'n_pixels': None,
                              'pixel_size': None,
                              'z_positions': None,
                              'rotations': None,
                              'material_budget': None,
                              'scatter_planes': None}

        self.default_options = {'input_files': None,
                                'output_path': None,
                                'chunk_size': 1000000,
                                'plot': False,
                                'noisy_suffix': '_noisy.h5',  # fixed since fixed in function
                                'cluster_suffix': '_clustered.h5',  # fixed since fixed in function
                                'skip_alignment': False,
                                'skip_noisy_pixel': False}

        # Make copy of defaults to change values but don't change defaults
        if setup is None:
            self.setup = deepcopy(self.default_setup)
        else:
            self.setup = setup

        if options is None:
            self.options = deepcopy(self.default_options)
        else:
            self.options = options

        self._init_UI()

    def _init_UI(self):
        """
        Create user interface
        """

        # Settings window
        self.setWindowTitle(self.window_title)
        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.resize(0.25 * self.screen.width(), 0.25 * self.screen.height())
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Widgets and layout
        # Spacing related
        v_space = 30
        h_space = 15

        # Make central widget
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setSpacing(v_space)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Make QGridLayout for options
        layout_options = QtWidgets.QGridLayout()
        layout_options.setSpacing(h_space)

        # Make widgets for plot option
        label_plot = QtWidgets.QLabel('Plot:')
        self.rb_t = QtWidgets.QRadioButton('True')
        self.rb_f = QtWidgets.QRadioButton('False')
        self.group_plot = QtWidgets.QButtonGroup()
        self.group_plot.addButton(self.rb_t)
        self.group_plot.addButton(self.rb_f)

        if self.options['plot']:
            self.rb_t.setChecked(True)
        else:
            self.rb_f.setChecked(True)

        # Make widgets for skip alignment option
        label_align = QtWidgets.QLabel('Skip alignment:')
        self.rb_t_align = QtWidgets.QRadioButton('True')
        self.rb_f_align = QtWidgets.QRadioButton('False')
        self.group_align = QtWidgets.QButtonGroup()
        self.group_align.addButton(self.rb_t_align)
        self.group_align.addButton(self.rb_f_align)

        if self.options['skip_alignment']:
            self.rb_t_align.setChecked(True)
        else:
            self.rb_f_align.setChecked(True)

        # Make widgets for chunk size option
        label_chunk = QtWidgets.QLabel('Chunk size:')
        self.edit_chunk = QtWidgets.QLineEdit()
        valid_chunk = QtGui.QIntValidator()
        valid_chunk.setBottom(0)
        self.edit_chunk.setValidator(valid_chunk)
        self.edit_chunk.setText(str(self.options['chunk_size']))

        # Add all  option widgets to layout_options, add spacers
        layout_options.addWidget(label_plot, 0, 0, 1, 1)
        layout_options.addItem(QtWidgets.QSpacerItem(7*h_space, v_space), 0, 1, 1, 1)
        layout_options.addWidget(self.rb_t, 0, 2, 1, 1)
        layout_options.addWidget(self.rb_f, 0, 3, 1, 1)
        layout_options.addWidget(label_align, 1, 0, 1, 1)
        layout_options.addItem(QtWidgets.QSpacerItem(7 * h_space, v_space), 1, 1, 1, 1)
        layout_options.addWidget(self.rb_t_align, 1, 2, 1, 1)
        layout_options.addWidget(self.rb_f_align, 1, 3, 1, 1)
        layout_options.addWidget(label_chunk, 2, 0, 1, 1)
        layout_options.addItem(QtWidgets.QSpacerItem(7*h_space, v_space), 2, 1, 1, 1)
        layout_options.addWidget(self.edit_chunk, 2, 2, 1, 2)

        # Make buttons for apply settings and cancel and button layout
        layout_buttons = QtWidgets.QHBoxLayout()
        button_ok = QtWidgets.QPushButton('Ok')
        button_ok.clicked.connect(lambda: self._update_settings())
        button_cancel = QtWidgets.QPushButton('Cancel')
        button_cancel.clicked.connect(lambda: self.close())
        layout_buttons.addStretch(1)
        layout_buttons.addWidget(button_ok)
        layout_buttons.addWidget(button_cancel)

        # Add all layouts to main layout
        main_layout.addSpacing(v_space)
        main_layout.addLayout(layout_options)
        main_layout.addStretch(1)
        main_layout.addLayout(layout_buttons)

    def _update_settings(self):

        palette = QtGui.QPalette()

        try:
            n = int(self.edit_chunk.text())
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.black)
            self.edit_chunk.setPalette(palette)
            if self.options['chunk_size'] != n:
                self.options['chunk_size'] = n

        except (TypeError, ValueError):
            palette.setColor(QtGui.QPalette.Text, QtCore.Qt.darkGray)
            self.edit_chunk.setPalette(palette)
            n = self.edit_chunk.text()
            self.statusBar().showMessage('Chunk size must be an integer, is type %s !' % type(n), 2000)
            return

        self.options['plot'] = self.rb_t.isChecked()
        self.options['skip_alignment'] = self.rb_t_align.isChecked()

        self.settingsUpdated.emit()
        self.close()


class ExceptionWindow(QtWidgets.QMainWindow):

    resetTab = QtCore.pyqtSignal()
    exceptionRead = QtCore.pyqtSignal()

    def __init__(self, exception, trace_back, tab=None, cause=None, parent=None):

        super(ExceptionWindow, self).__init__(parent)

        # Make this window blocking parent window
        self.setWindowModality(QtCore.Qt.ApplicationModal)

        # Get important information of the exception
        self.exception = exception
        self.traceback = trace_back
        self.exc_type = type(self.exception).__name__

        # Make main message and label
        msg = "The following exception occurred during %s: %s.\n" \
              "Try changing the input parameters. To reset %s tab press 'Reset tab'," \
              " to keep the current selection press 'Ok' !" % (cause, self.exc_type, tab)

        self.label = QtWidgets.QLabel(msg)
        self.label.setWordWrap(True)

        # Make warning icon via pixmap on QLabel
        self.pix_map = QtWidgets.qApp.style().standardIcon(QtWidgets.qApp.style().SP_MessageBoxWarning).pixmap(40, 40)
        self.label_icon = QtWidgets.QLabel()
        self.label_icon.setPixmap(self.pix_map)
        self.label_icon.setFixedSize(40, 40)

        self._init_UI()

    def _init_UI(self):

        # Exceptions window
        self.setWindowTitle(self.exc_type)
        self.screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.setMinimumSize(0.3 * self.screen.width(), 0.3 * self.screen.height())
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Widgets and layout
        # Spacing related
        v_space = 30
        h_space = 15

        # Make central widget
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Layout for icon and label
        layout_labels = QtWidgets.QHBoxLayout()
        layout_labels.addWidget(self.label_icon)
        layout_labels.addWidget(self.label)

        # Layout for buttons
        layout_buttons = QtWidgets.QHBoxLayout()

        # Textbrowser to display traceback
        self.browser_traceback = QtWidgets.QTextBrowser()
        self.browser_traceback.setText(str(self.exception))

        # Button to switch between traceback and exception message
        self.btn_switch = QtWidgets.QPushButton('Traceback')
        self.btn_switch.setToolTip('Switch between exception message and full traceback')
        self.btn_switch.clicked.connect(self.switch_text)

        # Button to safe traceback to file
        btn_safe = QtWidgets.QPushButton('Save')
        btn_safe.setToolTip('Safe traceback to file')
        btn_safe.clicked.connect(self.safe_traceback)

        # Reset button
        btn_reset = QtWidgets.QPushButton('Reset tab')
        btn_reset.setToolTip('Reset current analysis tab')
        btn_reset.clicked.connect(self.resetTab.emit)
        btn_reset.clicked.connect(self.close)

        # Ok button
        btn_ok = QtWidgets.QPushButton('Ok')
        btn_ok.setToolTip('Restore current analysis tab (No reset).')
        btn_ok.clicked.connect(self.close)

        # Add buttons to layout
        layout_buttons.addWidget(self.btn_switch)
        layout_buttons.addWidget(btn_safe)
        layout_buttons.addStretch(1)
        layout_buttons.addWidget(btn_reset)
        layout_buttons.addWidget(btn_ok)

        # Dock in which text browser is placed
        self.browser_dock = QtWidgets.QDockWidget()
        self.browser_dock.setWidget(self.browser_traceback)
        self.browser_dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea)
        self.browser_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.browser_dock.setWindowTitle(str(self.exc_type) + ' message:')

        # Add to main layout
        main_layout.addLayout(layout_labels)
        main_layout.addSpacing(v_space)
        main_layout.addWidget(self.browser_dock)
        main_layout.addLayout(layout_buttons)

    def safe_traceback(self):

        caption = 'Save traceback to file'
        trcbck_path = QtWidgets.QFileDialog.getSaveFileName(parent=self,
                                                            caption=caption,
                                                            directory='./',
                                                            filter='*.txt')[0]

        if trcbck_path:

            if 'txt' not in trcbck_path.split('.'):
                trcbck_path += '.txt'

            with open(trcbck_path, 'w') as f_write:
                f_write.write('{}'.format(self.traceback))

        else:
            pass

    def switch_text(self):

        if self.browser_dock.windowTitle() == str(self.exc_type) + ' message:':
            self.browser_dock.setWindowTitle('Traceback:')
            self.browser_traceback.setText(self.traceback)
            self.btn_switch.setText('Exception')
        else:
            self.browser_dock.setWindowTitle(str(self.exc_type) + ' message:')
            self.browser_traceback.setText(str(self.exception))
            self.btn_switch.setText('Traceback')

    def closeEvent(self, QCloseEvent):

        self.exceptionRead.emit()


class IPrealignmentWindow(QtWidgets.QMainWindow):

    def __init__(self, queue, parent=None):

        super(IPrealignmentWindow, self).__init__(parent)

        # Make this window blocking parent window
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        # Hide this windows close button
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowTitleHint |
                            QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Interactive Pre-alignment')

        self.queue = queue

        # Input
        self.x = None
        self.mean_fitted = None
        self.mean_error_fitted = None
        self.n_cluster = None
        self.ref_name = None
        self.dut_name = None
        self.prefix = None

        # Global variables needed to manipulate them within a matplotlib QT slot function
        self.selected_data = None
        self.fit = None
        self.error_limit = None
        self.offset_limit = None
        self.left_limit = None
        self.right_limit = None
        self.offset = None
        self.fit = None
        self.fit_fn = None
        self.initial_select = None

        self.do_refit = True  # True as long as not the Refit button is pressed, needed to signal calling function that the fit is ok or not

        self.init_UI()

    def init_UI(self):

        # Main widgets and layout
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # Matplotlib
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addWidget(self.canvas)

        # Labels
        self.offset_label = QtWidgets.QLabel()
        self.error_label = QtWidgets.QLabel()
        self.left_label = QtWidgets.QLabel()
        self.right_label = QtWidgets.QLabel()

        # Sliders
        self.offset_slider = FloatSlider(QtCore.Qt.Horizontal)
        self.offset_slider.setAutoScale(False)
        self.error_slider = FloatSlider(QtCore.Qt.Horizontal)
        self.error_slider.setAutoScale(False)
        self.left_slider = FloatSlider(QtCore.Qt.Horizontal)
        self.left_slider.setAutoScale(False)
        self.right_slider = FloatSlider(QtCore.Qt.Horizontal)
        self.right_slider.setAutoScale(False)

        # Buttons
        self.ok_button = QtWidgets.QPushButton('Ok')
        self.ok_button.clicked.connect(self.finish)
        self.ok_button.clicked.connect(lambda: self.enable_widgets(False))
        self.ok_button.clicked.connect(lambda: self.pbar.setBusy('Processing'))
        self.auto_button = QtWidgets.QPushButton('Auto')
        self.auto_button.clicked.connect(self.update_auto)
        self.refit_button = QtWidgets.QPushButton('Refit')
        self.refit_button.clicked.connect(self.refit)

        label_offset = QtWidgets.QLabel('Offset')
        label_offset.setFixedWidth(100)
        label_error = QtWidgets.QLabel('Error')
        label_error.setFixedWidth(100)
        label_left = QtWidgets.QLabel('Left limit')
        label_left.setFixedWidth(100)
        label_right = QtWidgets.QLabel('Right limit')
        label_right.setFixedWidth(100)

        self.offset_label.setFixedWidth(100)
        self.error_label.setFixedWidth(100)
        self.left_label.setFixedWidth(100)
        self.right_label.setFixedWidth(100)

        self.pbar = AnalysisBar()
        # self.pbar.setVisible(False)
        self.pbar.setFormat('Idle')
        self.pbar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        # Grid
        grid = QtWidgets.QGridLayout()
        grid.addWidget(label_offset, 0, 0, 1, 1)
        grid.addWidget(self.offset_slider, 0, 1, 1, 1)
        grid.addWidget(self.offset_label, 0, 2, 1, 1)
        grid.addWidget(label_error, 0, 3, 1, 1)
        grid.addWidget(self.error_slider, 0, 4, 1, 1)
        grid.addWidget(self.error_label, 0, 5, 1, 1)
        grid.addWidget(label_left, 1, 0, 1, 1)
        grid.addWidget(self.left_slider, 1, 1, 1, 1)
        grid.addWidget(self.left_label, 1, 2, 1, 1)
        grid.addWidget(label_right, 1, 3, 1, 1)
        grid.addWidget(self.right_slider, 1, 4, 1, 1)
        grid.addWidget(self.right_label, 1, 5, 1, 1)
        grid.addWidget(self.refit_button, 0, 6, 2, 1)
        grid.addWidget(self.auto_button, 0, 7, 2, 1)
        grid.addWidget(self.ok_button, 0, 8, 2, 1)
        grid.addWidget(self.pbar, 2, 0, 1, 9)

        self.main_layout.addLayout(grid)

    def enable_widgets(self, enable=True):

        for w in [self.ok_button, self.refit_button, self.auto_button,
                  self.offset_slider, self.error_slider, self.left_slider, self.right_slider]:
            w.setDisabled(not enable)

    def update_offset(self, offset_limit_new):  # Function called when offset slider is moved
        offset_limit_tmp = self.offset_limit
        self.offset_limit = offset_limit_new
        self.update_selected_data()
        if np.count_nonzero(self.selected_data) < 2:
            logging.warning("Offset limit: less than 2 data points are left")
            self.offset_limit = offset_limit_tmp
            self.update_selected_data()
        self.update_plot()

    def update_error(self, error_limit_new):  # Function called when error slider is moved
        error_limit_tmp = self.error_limit
        self.error_limit = error_limit_new / 10.0
        self.update_selected_data()
        if np.count_nonzero(self.selected_data) < 2:
            logging.warning("Error limit: less than 2 data points are left")
            self.error_limit = error_limit_tmp
            self.update_selected_data()
        self.update_plot()

    def update_left_limit(self, left_limit_new):  # Function called when left limit slider is moved
        left_limit_tmp = self.left_limit
        self.left_limit = left_limit_new
        self.update_selected_data()
        if np.count_nonzero(self.selected_data) < 2:
            logging.warning("Left limit: less than 2 data points are left")
            self.left_limit = left_limit_tmp
            self.update_selected_data()
        self.update_plot()

    def update_right_limit(self, right_limit_new):  # Function called when right limit slider is moved
        right_limit_tmp = self.right_limit
        self.right_limit = right_limit_new
        self.update_selected_data()
        if np.count_nonzero(self.selected_data) < 2:
            logging.warning("Right limit: less than 2 data points are left")
            self.right_limit = right_limit_tmp
            self.update_selected_data()
        self.update_plot()

    def update_selected_data(self):
        self.selected_data = self.initial_select.copy()
        self.selected_data[self.selected_data] = (np.abs(self.offset[self.selected_data]) <= self.offset_limit)
        self.selected_data[self.selected_data] = (np.abs(self.mean_error_fitted[self.selected_data]) <= self.error_limit)
        self.selected_data &= (self.x >= self.left_limit)
        self.selected_data &= (self.x <= self.right_limit)

    def update_auto(self):  # Function called when auto button is pressed

        selected_data_tmp = self.selected_data.copy()
        error_limit_tmp = self.error_limit
        offset_limit_tmp = self.offset_limit
        left_limit_tmp = self.left_limit
        right_limit_tmp = self.right_limit

        # This function automatically applies cuts according to these percentiles
        n_hit_percentile = 1
        mean_error_percentile = 95
        offset_percentile = 99

        error_median = np.nanmedian(self.mean_error_fitted[self.selected_data])
        error_std = np.nanstd(self.mean_error_fitted[self.selected_data])
        self.error_limit = max(error_median + error_std * 2,
                          np.percentile(np.abs(self.mean_error_fitted[self.selected_data]), mean_error_percentile))
        offset_median = np.nanmedian(self.offset[self.selected_data])
        offset_std = np.nanstd(self.offset[self.selected_data])
        self.offset_limit = max(offset_median + offset_std * 2, np.percentile(np.abs(self.offset[self.selected_data]),
                                                                         offset_percentile))  # Do not cut too much on the offset, it depends on the fit that might be off

        n_hit_cut = np.percentile(self.n_cluster[self.selected_data], n_hit_percentile)  # Cut off low/high % of the hits
        n_hit_cut_index = np.zeros_like(self.n_cluster, dtype=np.bool)
        n_hit_cut_index |= (self.n_cluster <= n_hit_cut)
        n_hit_cut_index[self.selected_data] |= (np.abs(self.offset[self.selected_data]) > self.offset_limit)
        n_hit_cut_index[~np.isfinite(self.offset)] = 1
        n_hit_cut_index[self.selected_data] |= (np.abs(self.mean_error_fitted[self.selected_data]) > self.error_limit)
        n_hit_cut_index[~np.isfinite(self.mean_error_fitted)] = 1
        n_hit_cut_index = np.where(n_hit_cut_index == 1)[0]
        left_index = np.where(self.x <= self.left_limit)[0][-1]
        right_index = np.where(self.x >= self.right_limit)[0][0]

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
            self.left_limit = self.x[left_cut]
            self.right_limit = self.x[right_cut]

        self.update_selected_data()
        if np.count_nonzero(self.selected_data) < 2:
            logging.info("Automatic pre-alignment: less than 2 data points are left, discard new limits")
            self.selected_data = selected_data_tmp
            self.error_limit = error_limit_tmp
            self.offset_limit = offset_limit_tmp
            self.left_limit = left_limit_tmp
            self.right_limit = right_limit_tmp
            self.update_selected_data()
        self.fit_data()
        self.offset_limit = np.max(np.abs(self.offset[self.selected_data]))
        self.error_limit = np.max(np.abs(self.mean_error_fitted[self.selected_data]))
        self.update_plot()

    def update_plot(self):  # Replot correlation data with new selection

        if np.count_nonzero(self.selected_data) > 1:
            left_index = np.where(self.x <= self.left_limit)[0][-1]
            right_index = np.where(self.x >= self.right_limit)[0][0]
            # set ymax to maximum of either error or offset within the left and right limit, and increase by 10%
            self.ax2.set_ylim(ymax=max(np.max(np.abs(self.mean_error_fitted[self.selected_data])) * 10.0,
                                  np.max(np.abs(self.offset[self.selected_data])) * 1.0) * 1.1)
            self.offset_limit_plot.set_ydata([self.offset_limit, self.offset_limit])
            self.error_limit_plot.set_ydata([self.error_limit * 10.0, self.error_limit * 10.0])
            self.left_limit_plot.set_xdata([self.left_limit, self.left_limit])
            self.right_limit_plot.set_xdata([self.right_limit, self.right_limit])
            # setting calculated offset data
            self.offset_plot.set_data(self.x[self.initial_select], np.abs(self.offset[self.initial_select]))
            # update offset slider
            offset_range = self.offset[left_index:right_index]
            offset_range = offset_range[np.isfinite(offset_range)]
            offset_max = np.max(np.abs(offset_range))
            self.offset_slider.setMaximum(offset_max)
            self.offset_slider.setValue(self.offset_limit)
            # update error slider
            error_range = self.mean_error_fitted[left_index:right_index]
            error_range = error_range[np.isfinite(error_range)]
            error_max = np.max(np.abs(error_range)) * 10.0
            self.error_slider.setMaximum(error_max)
            self.error_slider.setValue(self.error_limit * 10.0)
            # update left slider
            self.left_slider.setValue(self.left_limit)
            # update right slider
            self.right_slider.setValue(self.right_limit)
            # setting calculated fit line
            self.line_plot.set_data(self.x, self.fit_fn(self.x))
            self.canvas.draw()  # Needed to update figure
        else:
            logging.info('Cuts are too tight. Not enough data to fit')

    def finish(self):  # Fit result is ok
        self.do_refit = False  # Set to signal that no refit is required anymore
        self.update_selected_data()
        self.fit_data()
        self.queue.put([self.selected_data, self.fit, self.do_refit])

    def refit(self):
        self.fit_data()
        self.update_plot()

    def fit_data(self):
        self.fit, _ = curve_fit(linear, self.x[self.selected_data],
                                self.mean_fitted[self.selected_data])  # Fit straight line
        self.fit_fn = np.poly1d(self.fit[::-1])
        self.offset = self.fit_fn(self.x) - self.mean_fitted  # Calculate straight line fit offset

    def update_plots(self, x, mean_fitted, mean_error_fitted, n_cluster, ref_name, dut_name, prefix):

        self.fig.clf()

        self.x = x
        self.mean_fitted = mean_fitted
        self.mean_error_fitted = mean_error_fitted
        self.n_cluster = n_cluster
        self.ref_name = ref_name
        self.dut_name = dut_name
        self.prefix = prefix

        self.selected_data = np.ones_like(self.mean_fitted, dtype=np.bool)
        self.selected_data &= np.isfinite(self.mean_fitted)
        self.selected_data &= np.isfinite(self.mean_error_fitted)

        self.initial_select = self.selected_data.copy()

        # Calculate and plot selected data + fit + fit offset and gauss fit error
        self.fit_data()
        self.offset_limit = np.max(np.abs(self.offset[self.selected_data]))  # Calculate starting offset cut
        self.error_limit = np.max(np.abs(self.mean_error_fitted[self.selected_data]))  # Calculate starting fit error cut
        self.left_limit = np.min(self.x[self.selected_data])  # Calculate starting left cut
        self.right_limit = np.max(self.x[self.selected_data])  # Calculate starting right cut

        ax = self.fig.add_subplot(111)
        self.ax2 = ax.twinx()
        # Setup plot
        self.mean_plot, = ax.plot(self.x[self.selected_data], self.mean_fitted[self.selected_data], 'o-',
                             label='Data prefit')  # Plot correlation
        self.line_plot, = ax.plot(self.x[self.selected_data], self.fit_fn(self.x[self.selected_data]), '-', label='Line fit')  # Plot line fit
        self.error_plot, = self.ax2.plot(self.x[self.selected_data], np.abs(self.mean_error_fitted[self.selected_data]) * 10.0, 'ro-',
                               label='Error x10')  # Plot gaussian fit error
        self.offset_plot, = self.ax2.plot(self.x[self.selected_data], np.abs(self.offset[self.selected_data]), 'go-',
                                label='Offset')  # Plot line fit offset
        self.offset_limit_plot = self.ax2.axhline(self.offset_limit, linestyle='--', color='g',
                                        linewidth=2)  # Plot offset cut as a line
        self.error_limit_plot = self.ax2.axhline(self.error_limit * 10.0, linestyle='--', color='r',
                                       linewidth=2)  # Plot error cut as a line
        self.left_limit_plot = self.ax2.axvline(self.left_limit, linestyle='-', color='r',
                                      linewidth=2)  # Plot left cut as a vertical line
        self.right_limit_plot = self.ax2.axvline(self.right_limit, linestyle='-', color='r',
                                       linewidth=2)  # Plot right cut as a vertical line
        self.ncluster_plot = ax.bar(self.x[self.selected_data],
                                    self.n_cluster[self.selected_data] / np.max(self.n_cluster[self.selected_data]).astype(np.float) * abs(
                                   np.diff(ax.get_ylim())[0]), bottom=ax.get_ylim()[0], align='center', alpha=0.1,
                               label='#Cluster [a.u.]', width=np.min(
                np.diff(self.x[self.selected_data])))  # Plot number of hits for each correlation point
        ax.set_ylim(ymin=np.min(self.mean_fitted[self.selected_data]), ymax=np.max(self.mean_fitted[self.selected_data]))
        self.ax2.set_ylim(ymin=0.0, ymax=max(np.max(np.abs(self.mean_error_fitted[self.selected_data])) * 10.0,
                                        np.max(np.abs(self.offset[self.selected_data])) * 1.0) * 1.1)
        ax.set_xlim((np.nanmin(self.x), np.nanmax(self.x)))
        ax.set_title("Correlation of %s: %s vs. %s" % (prefix + "s", self.ref_name, self.dut_name))
        ax.set_xlabel("%s [um]" % self.dut_name)
        ax.set_ylabel("%s [um]" % self.ref_name)
        self.ax2.set_ylabel("Error / Offset")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax2.legend(lines + lines2, labels + labels2, loc=0)
        ax.grid()

        # connect buttons
        self.offset_slider.valueChanged.connect(lambda val: self.update_offset(val))
        self.offset_slider.valueChanged.connect(lambda val: self.offset_label.setText(str(val)))
        self.error_slider.valueChanged.connect(lambda val: self.update_error(val))
        self.error_slider.valueChanged.connect(lambda val: self.error_label.setText(str(val)))
        self.left_slider.valueChanged.connect(lambda val: self.update_left_limit(val))
        self.left_slider.valueChanged.connect(lambda val: self.left_label.setText(str(val)))
        self.right_slider.valueChanged.connect(lambda val: self.update_right_limit(val))
        self.right_slider.valueChanged.connect(lambda val: self.right_label.setText(str(val)))

        self.offset_slider.setMaximum(self.offset_limit)
        self.error_slider.setMaximum(self.error_limit * 10.0)
        self.left_slider.setMinimum(self.left_limit)
        self.left_slider.setMaximum(self.right_limit)
        self.left_slider.setValue(self.left_limit)
        self.right_slider.setMinimum(self.left_limit)
        self.right_slider.setMaximum(self.right_limit)
        self.right_slider.setValue(self.right_limit)

        self.canvas.draw()  # Update figure
        self.enable_widgets()
        self.pbar.setFinished()
        self.pbar.setFormat('Ready')