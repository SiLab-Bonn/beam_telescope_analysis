import sys
import os
import logging
import unittest

from queue import Queue
from PyQt5 import QtWidgets, QtCore, QtGui

from testbeam_analysis.gui.main import AnalysisWindow
from testbeam_analysis.gui.tab_widgets.files_tab import FilesTab
from testbeam_analysis.gui.tab_widgets.setup_tab import SetupTab
from testbeam_analysis.gui.tab_widgets import analysis_tabs
from testbeam_analysis.gui.gui_widgets.analysis_widgets import AnalysisWidget, ParallelAnalysisWidget
from testbeam_analysis.gui.gui_widgets.option_widgets import *
from testbeam_analysis.gui.gui_widgets.sub_windows import IPrealignmentWindow, ExceptionWindow, SettingsWindow


class TestGui(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Make QApplication which starts event loop in order to create widgets
        cls.test_app = QtWidgets.QApplication(sys.argv)

        # Main widget to parent all other widgets
        cls.main_widget = QtWidgets.QTabWidget()

        # Make test setup and options
        cls.test_setup = {'dut_names': ['Tel_%i' % i for i in range(4)],
                          'n_duts': 4,
                          'n_pixels': [(80, 336)] * 4,
                          'pixel_size': [(250.0, 50.0)] * 4,
                          'z_positions': [0.0, 19500.0, 108800.0, 126800.0],
                          'rotations': [(0.0, 0.0, 0.0)] * 4,
                          'scatter_planes': None}

        cls.test_options = {'input_files': ['test_input_file_Tel_%i' % i for i in range(4)],
                            'output_path': 'test_output_path',
                            'chunk_size': 1000000,
                            'plot': False,
                            'noisy_suffix': '_noisy.h5',  # fixed since fixed in function
                            'cluster_suffix': '_clustered.h5',  # fixed since fixed in function
                            'skip_alignment': False,
                            'skip_noisy_pixel': False}

        # Dut types
        cls.dut_types = {'FE-I4': {'material_budget': 0.001067236, 'n_cols': 80, 'n_rows': 336, 'pitch_col': 250.0,
                                   'pitch_row': 50.0},
                         'Mimosa26': {'material_budget': 0.000797512, 'n_cols': 1152, 'n_rows': 576, 'pitch_col': 18.4,
                                      'pitch_row': 18.4}}

        # Create tab widgets
        cls.files_tab = FilesTab(parent=cls.main_widget)
        cls.setup_tab = SetupTab(parent=cls.main_widget)
        cls.noisy_pixel_tab = analysis_tabs.NoisyPixelsTab(parent=cls.main_widget, setup=cls.test_setup,
                                                           options=cls.test_options, name='Noisy Pixel',
                                                           tab_list='Clustering')
        cls.clustering_tab = analysis_tabs.ClusterPixelsTab(parent=cls.main_widget, setup=cls.test_setup,
                                                            options=cls.test_options, name='Clustering',
                                                            tab_list='Prealignment')
        cls.prealignment_tab = analysis_tabs.PrealignmentTab(parent=cls.main_widget, setup=cls.test_setup,
                                                             options=cls.test_options, name='Prealignment',
                                                             tab_list='Track Finding')
        cls.track_finding_tab = analysis_tabs.TrackFindingTab(parent=cls.main_widget, setup=cls.test_setup,
                                                              options=cls.test_options, name='Track Finding',
                                                              tab_list='Alignment')
        cls.alignment_tab = analysis_tabs.AlignmentTab(parent=cls.main_widget, setup=cls.test_setup,
                                                       options=cls.test_options, name='Alignment',
                                                       tab_list='Track Fitting')
        cls.track_fitting_tab = analysis_tabs.TrackFittingTab(parent=cls.main_widget, setup=cls.test_setup,
                                                              options=cls.test_options, name='Track Fitting',
                                                              tab_list=['Residuals', 'Efficiency'])
        cls.residual_tab = analysis_tabs.ResidualTab(parent=cls.main_widget, setup=cls.test_setup,
                                                     options=cls.test_options, name='Residuals',
                                                     tab_list='Efficiency')
        cls.efficiency_tab = analysis_tabs.EfficiencyTab(parent=cls.main_widget, setup=cls.test_setup,
                                                         options=cls.test_options, name='Efficiency',
                                                         tab_list='Last')
        # Create single analysis and parallel analysis widget
        cls.analysis_widget = AnalysisWidget(parent=cls.main_widget, setup=cls.test_setup, options=cls.test_options,
                                             name='Test')
        cls.parallel_analysis_widget = ParallelAnalysisWidget(parent=cls.main_widget, setup=cls.test_setup,
                                                              options=cls.test_options, name='Test')
        # Create complete window
        cls.analysis_window = AnalysisWindow()

        # Create sub-windows
        cls.iprealignment_window = IPrealignmentWindow(Queue())
        cls.exception_window = ExceptionWindow(AttributeError('Random exception'), 'Random traceback')
        cls.settings_window = SettingsWindow(setup=cls.test_setup, options=cls.test_options)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_files_tab(self):
        """Test FilesTab"""

        # Test default settings
        self.assertEqual(self.files_tab.isFinished, False)
        self.assertEqual(self.files_tab.edit_output.toPlainText(), os.getcwd())

        # Set DUT names and check
        self.files_tab._data_table.setRowCount(len(self.test_setup['dut_names']))
        self.files_tab._data_table.column_labels = ['Path', 'Name', 'Status', 'Navigation']
        self.files_tab._data_table.set_dut_names()

        self.assertListEqual(self.files_tab._data_table.dut_names, self.test_setup['dut_names'])

    def test_setup_tab(self):
        """Test SetupTab"""

        # Test default settings
        self.assertEqual(self.setup_tab.isFinished, False)

        # Init all tabs and the setup painter, check tabs are created
        self.setup_tab.input_data(data=self.test_setup)
        self.setup_tab._dut_types = self.dut_types

        self.assertListEqual(sorted(self.test_setup['dut_names']), sorted(self.setup_tab.tw.keys()))

        # Set props of Tel_0 and check
        self.setup_tab._set_properties(self.test_setup['dut_names'][0], 'FE-I4')

        properties = {}
        for p in self.setup_tab._dut_widgets[self.test_setup['dut_names'][0]].keys():
            if p in self.dut_types['FE-I4'].keys():
                properties[p] = str(self.setup_tab._dut_widgets[self.test_setup['dut_names'][0]][p].text())

        self.assertListEqual(sorted(properties.values()), sorted([str(r) for r in self.dut_types['FE-I4'].values()]))

        # Load test setup again
        self.setup_tab.load_setup(setup=self.test_setup)

        # Add scatter plane and check if tab is created
        self.setup_tab._add_dut('Scatter_plane', scatter_plane=True)
        dut_names = self.test_setup['dut_names'][:]
        dut_names.append('Scatter_plane')

        self.assertListEqual(sorted(dut_names), sorted(self.setup_tab.tw.keys()))

    def test_analysis_widget(self):
        """Test analysis widget"""

        # Make arbitrary function
        def some_func(a=1, b='String', c=3.14159265, d=True, col_range=None, some_iterable=None):
            ''' Some docstring of some func

                Parameters
                ----------
                col_range : iterable
                    Iterable of ranges.
                some_iterable : iterable
                    Its an iterable
            '''
            if d:
                return b*(a * int(c)), col_range, some_iterable

        # Test default settings
        self.assertEqual(self.analysis_widget.isFinished, False)
        self.assertEqual(self.analysis_widget.name, 'Test')

        # Add function and check
        self.analysis_widget.add_function(some_func)

        # Add option and check
        self.analysis_widget.add_option(option='col_range', func=some_func,
                                        default_value=[[0, 80]] * self.test_setup['n_duts'],
                                        optional=False)

        self.assertListEqual(list(self.analysis_widget.calls.values()), [{'a': 1, 'b': 'String', 'c': 3.14159265, 'd': True,
                                                                          'col_range':[[0, 80]] * self.test_setup['n_duts'],
                                                                          'some_iterable': None}])

        # Change existing option and check
        self.analysis_widget.add_option(option='a', func=some_func, default_value=2)

        # Check
        self.assertListEqual(list(self.analysis_widget.calls.values()), [{'a': 2, 'b': 'String', 'c': 3.14159265, 'd': True,
                                                                          'col_range': [[0, 80]] * self.test_setup['n_duts'],
                                                                          'some_iterable': None}])

        # Load some values in OptionMultiRangeBox
        self.analysis_widget.option_widgets['col_range'].load_value([[0, 10]] * self.test_setup['n_duts'])

        # Get these values again
        values = [[self.analysis_widget.option_widgets['col_range'].range_boxes[key][0].value(),
                   self.analysis_widget.option_widgets['col_range'].range_boxes[key][-1].value()] for key in self.test_setup['dut_names']]

        # Check values again
        self.assertListEqual(values, [[0, 10]] * self.test_setup['n_duts'])

        # Check for if correct widgets were created
        self.assertEqual(isinstance(self.analysis_widget.option_widgets['a'], OptionSlider), True)
        self.assertEqual(isinstance(self.analysis_widget.option_widgets['b'], OptionText), True)
        self.assertEqual(isinstance(self.analysis_widget.option_widgets['c'], OptionSlider), True)
        self.assertEqual(isinstance(self.analysis_widget.option_widgets['d'], OptionBool), True)
        self.assertEqual(isinstance(self.analysis_widget.option_widgets['col_range'], OptionMultiRangeBox), True)
        self.assertEqual(isinstance(self.analysis_widget.option_widgets['some_iterable'], OptionMultiSlider), True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGui)
    unittest.TextTestRunner(verbosity=2).run(suite)
