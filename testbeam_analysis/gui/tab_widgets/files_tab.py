"""
Implements a FilesTab that allows to select and check test beam data for further analysis
"""

import os
import tables as tb
import logging

from PyQt5 import QtCore, QtWidgets, QtGui


class FilesTab(QtWidgets.QWidget):
    """
    Implements the tab content for data file handling
    """

    statusMessage = QtCore.pyqtSignal(str)
    analysisFinished = QtCore.pyqtSignal(str, list)

    def __init__(self, parent=None):
        super(FilesTab, self).__init__(parent)

        # Add output data
        self.data = {}
        self.output_path = os.getcwd()

        # Store state of FilesTab
        self.isFinished = False

        self._setup()

    def _setup(self):
        # Add layout/spacing related stuff
        h_space = 10
        v_space = 30
        sub_v_space = 15

        # Table area
        left_widget = QtWidgets.QWidget()
        tab_layout = QtWidgets.QHBoxLayout()
        left_widget.setLayout(tab_layout)
        self._data_table = FilesTable(parent=left_widget)
        self._data_table.dragAndDropText.connect(lambda msg: self._emit_message(msg))
        tab_layout.addWidget(self._data_table)

        # Make option area
        layout_options = QtWidgets.QVBoxLayout()
        layout_options.setSpacing(v_space)
        label_option = QtWidgets.QLabel('Options')
        layout_options.addWidget(label_option)

        # Add sub layout for horizontal spacing
        sl = QtWidgets.QHBoxLayout()
        sl.addSpacing(h_space)

        # Make buttons and add layout for option buttons
        self.layout_buttons = QtWidgets.QVBoxLayout()
        self.layout_buttons.setSpacing(sub_v_space)

        # Make a select button to select input files and connect
        button_select = QtWidgets.QPushButton('Select data of DUTs')
        button_select.clicked.connect(lambda: self._data_table.get_data())

        # Make button to reset dut names and connect
        button_names = QtWidgets.QPushButton('Reset names')
        button_names.setToolTip('Set default DUT names')
        button_names.clicked.connect(lambda: self._data_table.set_dut_names())

        # Make button to clear the table content and connect
        button_clear = QtWidgets.QPushButton('Clear')
        button_clear.setToolTip('Clears table')
        button_clear.clicked.connect(lambda: self._data_table.clear_table())

        # Add buttons to main layout
        self.layout_buttons.addWidget(button_select)
        self.layout_buttons.addWidget(button_names)
        self.layout_buttons.addWidget(button_clear)
        sl.addLayout(self.layout_buttons)
        layout_options.addLayout(sl)

        # Make button to select output folder and connect
        # Add sub layout for horizontal spacing
        sl_1 = QtWidgets.QHBoxLayout()
        sl_1.addSpacing(h_space)
        self.layout_out = QtWidgets.QVBoxLayout()
        self.edit_output = QtWidgets.QTextEdit()
        self.edit_output.setReadOnly(True)
        self.edit_output.show()
        self.edit_output.document().contentsChanged.connect(self._set_edit_height)
        self.edit_output.setText(self.output_path)
        self.edit_output.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        button_out = QtWidgets.QPushButton('Set output folder')
        button_out.setToolTip('Set output older')
        button_out.clicked.connect(lambda: self._get_output_folder())
        self.layout_out.addWidget(self.edit_output)
        self.layout_out.addWidget(button_out)
        sl_1.addLayout(self.layout_out)
        label_output = QtWidgets.QLabel('Output folder')

        # Add to main layout
        layout_options.addWidget(label_output)
        layout_options.addLayout(sl_1)

        # Make proceed button
        self.btn_ok = QtWidgets.QPushButton('Ok')
        self.btn_ok.setDisabled(True)
        self.btn_ok.setToolTip('Select data of DUTs')

        # Connect proceed button and inputFilesChanged signal
        message_ok = "Configuration for %d DUT(s) set."
        for x in [lambda: self._data_table.update_setup(),
                  lambda: self._update_data(),
                  lambda: self.set_read_only(),
                  lambda: self._emit_message(message_ok % (len(self._data_table.input_files)))]:
            self.btn_ok.clicked.connect(x)

        self._data_table.inputFilesChanged.connect(lambda: self._analysis_check())

        # Add to main layout
        layout_options.addStretch(0)

        # Add container widget to disable widgets after ok is pressed
        self.container = QtWidgets.QWidget()
        self.container.setLayout(layout_options)

        # Add main layout to widget
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(QtWidgets.QVBoxLayout())

        right_widget.layout().addWidget(self.container)
        right_widget.layout().addWidget(self.btn_ok)

        # Split table and option area
        widget_splitter = QtWidgets.QSplitter()
        widget_splitter.addWidget(left_widget)
        widget_splitter.addWidget(right_widget)
        widget_splitter.setStretchFactor(0, 10)
        widget_splitter.setStretchFactor(1, 2.5)
        widget_splitter.setChildrenCollapsible(False)

        # Add complete layout to this widget
        layout_widget = QtWidgets.QVBoxLayout()
        layout_widget.addWidget(widget_splitter)
        self.setLayout(layout_widget)

    def _set_edit_height(self):
        """
        Dynamically change size of edit up to 1/3 of self.height
        """

        # Set size first
        self.edit_output.setFixedHeight(
            self.edit_output.document().size().height() +
            self.edit_output.contentsMargins().top() +
            self.edit_output.contentsMargins().bottom()
        )

        # Set fixed size if gets too big
        if self.edit_output.height() >= self.height() / 3 >= 100:  # 100 arbitrary value for first size setting
            self.edit_output.setFixedHeight(self.height() / 3)

    def _get_output_folder(self):
        """
        Get output folder and display path in QTextEdit
        """
        caption = 'Select output folder'
        path = QtWidgets.QFileDialog.getExistingDirectory(caption=caption, directory='./')

        if path != self.output_path and len(path) != 0:
            self.output_path = path
            self.edit_output.setText(self.output_path)

    def _analysis_check(self):
        """
        Handles  whether the proceed 'OK' button is clickable or not in regard to the input data.
        If not, respective messages are shown in QMainWindows statusBar
        """
        if self._data_table.input_files and not self._data_table.incompatible_data:
            self.btn_ok.setDisabled(False)
            self.btn_ok.setToolTip('Proceed')

        else:
            self.btn_ok.setDisabled(True)
            self.btn_ok.setToolTip('Select data of DUTs')

            if self._data_table.incompatible_data:
                broken = []
                for key in self._data_table.incompatible_data.keys():
                    broken.append(self._data_table.dut_names[key])
                message = "Data of %s is broken. Analysis impossible." % str(',').join(broken)
                self._emit_message(message)

            if not self._data_table.input_files:
                message = "No data. Analysis impossible."
                self._emit_message(message)

    def _emit_message(self, message):
        """
        Emits statusMessage signal with message
        """

        self.statusMessage.emit(message)

    def _update_data(self):
        """
        Updates the data returned by FilesTab
        """
        self.data['output_path'] = self.output_path
        self.data['input_files'] = self._data_table.input_files
        self.data['dut_names'] = self._data_table.dut_names
        self.data['n_duts'] = len(self._data_table.dut_names)

        self.isFinished = True
        self.analysisFinished.emit('Files', ['Setup'])

    def set_read_only(self, read_only=True):
        """
        Method to disable the tab while leaving some widgets active in order to review file selection
        """

        # Disable all buttons in the button and output layout
        for layout in [self.layout_buttons, self.layout_out]:
            for i in reversed(range(layout.count())):
                if isinstance(layout.itemAt(i), QtWidgets.QWidgetItem):
                    w = layout.itemAt(i).widget()
                    if isinstance(w, QtWidgets.QPushButton):
                        w.setDisabled(read_only)

        self._data_table.set_read_only(read_only)
        self.btn_ok.setDisabled(read_only)

    def load_files(self, session):
        """
        Loads file names and dut names from saved session

        :param session: dict
        """

        if 'options' not in session and 'setup' not in session:
            return

        self._data_table.input_files = session['options']['input_files']
        self._data_table.dut_names = session['setup']['dut_names']
        self.edit_output.setText(session['options']['output_path'])
        self._data_table.handle_data()


class FilesTable(QtWidgets.QTableWidget):
    """
    Class to get, display and handle the input data of the DUTs
    for which a testbeam analysis will be performed
    """

    inputFilesChanged = QtCore.pyqtSignal()
    dragAndDropText = QtCore.pyqtSignal('QString')

    def __init__(self, parent=None):
        super(FilesTable, self).__init__(parent)

        # Lists for dut names, input files
        self.dut_names = []
        self.input_files = []

        # store indices and status of incompatible data might occurring in check_data
        self.incompatible_data = {}

        # Appearance
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.setWordWrap(True)
        self.setTextElideMode(QtCore.Qt.ElideLeft)
        self.showGrid()
        self.setSortingEnabled(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        # Drag and drop related
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        if self.underMouse() and not self.input_files:
            msg = 'Select files via the button on the right or "Drag & Drop" files directly onto table area'
            self.dragAndDropText.emit(msg)
        else:
            self.setMouseTracking(False)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            for url in event.mimeData().urls():
                if '.h5' in url.toLocalFile():
                    self.input_files.append(os.path.join(url.toLocalFile()))
                else:
                    msg = 'Files must be HDF5-format'
                    logging.warning(msg)
            self.handle_data()
        else:
            event.ignore()

    def get_data(self):
        """
        Open file dialog and select data files. Only *.h5 files are allowed
        """

        caption = 'Select data of DUTs'
        for path in QtWidgets.QFileDialog.getOpenFileNames(parent=self,
                                                           caption=caption,
                                                           directory='./',
                                                           filter='*.h5')[0]:
            self.input_files.append(os.path.join(path))

        if self.input_files:
            self.handle_data()

    def handle_data(self):
        """
        Arranges input_data in the table and re-news table if DUT amount/order has been updated
        """

        self.row_labels = [('DUT ' + '%d' % i) for i, _ in enumerate(self.input_files)]
        self.column_labels = ['Path', 'Name', 'Status', 'Navigation']

        self.setColumnCount(len(self.column_labels))
        self.setRowCount(len(self.row_labels))
        self.setHorizontalHeaderLabels(self.column_labels)
        self.setVerticalHeaderLabels(self.row_labels)

        for row, dut in enumerate(self.input_files):
            # TODO: replace with QTextEdit to show full text if needed
            # edit_dut = QtWidgets.QTextEdit(dut)
            # edit_dut.setLineWrapMode(True)
            # edit_dut.setFrameStyle(0)
            # edit_dut.setReadOnly(True)
            # edit_dut.setVerticalScrollBarPolicy(1)
            # edit_dut.show()
            # edit_width = edit_dut.document().size().width()
            # edit_height = edit_dut.document().size().height()
            # edit_dut.setFixedSize(edit_width, edit_height)
            # self.setCellWidget(row, self.column_labels.index('Path'), edit_dut)

            path_item = QtWidgets.QTableWidgetItem()
            path_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            path_item.setTextAlignment(QtCore.Qt.AlignLeft)
            path_item.setText(dut)
            path_item.setToolTip(dut)
            self.setItem(row, self.column_labels.index('Path'), path_item)

        self.update_dut_names()
        self.check_data()
        self._make_nav_buttons()
        self.inputFilesChanged.emit()

    def check_data(self):
        """
        Checks if given input_files contain the necessary information like
        event_number, column, row, etc.; visualizes broken input
        """

        field_req = ('event_number', 'frame', 'column', 'row', 'charge')
        self.incompatible_data = dict()

        for i, path in enumerate(self.input_files):
            with tb.open_file(path, mode='r') as f:
                try:
                    fields = f.root.Hits.colnames
                    missing = []
                    for req in field_req:
                        if req in fields:
                            pass
                        else:
                            missing.append(req)
                    if len(missing) != 0:
                        self.incompatible_data[i] = 'Data does not contain required field(s):\n' + ', '.join(missing)
                    if f.root.Hits.shape[0] == 0:
                        self.incompatible_data[i] = 'Hit data is empty!'
                except tb.exceptions.NoSuchNodeError:
                    self.incompatible_data[i] = 'No node named Hits! Found nodes:\n' + ', '.join([node._v_name for node in f.root])

        font = QtGui.QFont()
        font.setBold(True)

        for row in range(self.rowCount()):
            status_item = QtWidgets.QTableWidgetItem()
            status_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            status_item.setTextAlignment(QtCore.Qt.AlignCenter)

            if row in self.incompatible_data.keys():
                error_font = font
                error_font.setUnderline(True)
                status_item.setText(self.incompatible_data[row])
                self.setItem(row, self.column_labels.index('Status'), status_item)

                for col in range(self.columnCount()):
                    try:
                        self.item(row, col).setFont(error_font)
                        self.item(row, col).setForeground(QtGui.QColor('red'))
                    except AttributeError:
                        pass
            else:
                status_item.setText('Okay')
                self.setItem(row, self.column_labels.index('Status'), status_item)
                self.item(row, self.column_labels.index('Status')).setFont(font)
                self.item(row, self.column_labels.index('Status')).setForeground(QtGui.QColor('green'))

    def update_data(self):
        """
        Updates the data/DUT content/order by re-reading the filespaths
        from the table and updating self.input_files
        """

        new = []
        try:
            for row in range(self.rowCount()):
                # FIXME: replace with QTextEdit to show full text if needed
                # new.append(self.cellWidget(row, self.column_labels.index('Path')).toPlainText())
                new.append(self.item(row, self.column_labels.index('Path')).text())
        except AttributeError:
            pass

        if new != self.input_files:
            self.input_files = new
            self.inputFilesChanged.emit()

    def set_dut_names(self, name='Tel'):
        """
        Set DUT names for further analysis. Std. setting is Tel_i  and i is index
        """

        for row in range(self.rowCount()):
            dut_name = name + '_%d' % row if not isinstance(name, list) else name[row]
            dut_item = QtWidgets.QTableWidgetItem()
            dut_item.setTextAlignment(QtCore.Qt.AlignCenter)
            dut_item.setText(dut_name)
            if row in self.incompatible_data.keys():
                font = dut_item.font()
                font.setBold(True)
                font.setUnderline(True)
                dut_item.setFont(font)
                dut_item.setForeground(QtGui.QColor('red'))
            self.setItem(row, self.column_labels.index('Name'), dut_item)
            self.dut_names.append(str(dut_name))

    def update_dut_names(self, name='Tel'):
        """
        Read list of DUT names from table and update dut names. Also add new DUT names and update
        """

        new = []
        try:
            for row in range(self.rowCount()):
                try:
                    new.append(str(self.item(row, self.column_labels.index('Name')).text()))
                except AttributeError:  # no QTableWidgetItem for new input data
                    add_dut_item = QtWidgets.QTableWidgetItem()
                    add_dut_item.setTextAlignment(QtCore.Qt.AlignCenter)
                    add_dut_item.setText(name + '_%d' % row)
                    self.setItem(row, self.column_labels.index('Name'), add_dut_item)
                    new.append(str(self.item(row, self.column_labels.index('Name')).text()))
        except AttributeError: # no QTableWidgetItem has been created at all
            self.set_dut_names()

        if new != self.dut_names:  # and len(new) != 0:
            self.dut_names = new
            for row in range(self.rowCount()):
                self.item(row, self.column_labels.index('Name')).setText(self.dut_names[row])

#        self.resizeRowsToContents()
#        self.resizeColumnsToContents()

    def update_setup(self):
        """
        Updating all relevant lists for further analysis
        """

        self.update_data()
        self.handle_data()
#        self.update_dut_names()

    def clear_table(self):
        """
        Clear table of all its contents
        """

        self.setRowCount(0)
        self.update_data()

    def set_read_only(self, read_only=True):
        """
        Make all cells of the table only readable
        """

        # Loop over all cells and disable widgets / cell items
        for i in range(self.columnCount()):
            for j in range(self.rowCount()):
                # Widgets
                w = self.cellWidget(i, j)
                if w:
                    w.setDisabled(read_only)
                # Items
                item = self.item(i, j)
                if item and read_only:
                    item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)

        # Disable sorting
        self.setSortingEnabled(not read_only)

    def _make_nav_buttons(self):
        """
        Make buttons to navigate through table and delete entries
        """

        for row in range(self.rowCount()):
            widget_but = QtWidgets.QWidget()
            layout_but = QtWidgets.QHBoxLayout()
            layout_but.setAlignment(QtCore.Qt.AlignCenter)
            self.button_up = QtWidgets.QPushButton()
            self.button_down = QtWidgets.QPushButton()
            self.button_del = QtWidgets.QPushButton()
            button_size = QtCore.QSize(40,40)
            icon_up = self.button_up.style().standardIcon(QtWidgets.QStyle.SP_ArrowUp)
            icon_down = self.button_down.style().standardIcon(QtWidgets.QStyle.SP_ArrowDown)
            icon_del = self.button_del.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon)
            icon_size = QtCore.QSize(30, 30)
            self.button_up.setIcon(icon_up)
            self.button_down.setIcon(icon_down)
            self.button_del.setIcon(icon_del)
            self.button_up.setIconSize(icon_size)
            self.button_down.setIconSize(icon_size)
            self.button_del.setIconSize(icon_size)
            self.button_up.setFixedSize(button_size)
            self.button_down.setFixedSize(button_size)
            self.button_del.setFixedSize(button_size)
            self.button_del.setToolTip('Delete')
            self.button_up.setToolTip('Move up')
            self.button_down.setToolTip('Move down')

            for x in [lambda: self._move_up(), lambda: self.update_setup()]:
                self.button_up.clicked.connect(x)

            for x in [lambda: self._move_down(), lambda: self.update_setup()]:
                self.button_down.clicked.connect(x)

            for x in [lambda: self._delete_data(), lambda: self.handle_data()]:
                self.button_del.clicked.connect(x)

            layout_but.addWidget(self.button_up)
            layout_but.addWidget(self.button_down)
            layout_but.addWidget(self.button_del)
            widget_but.setLayout(layout_but)
            self.setCellWidget(row, self.column_labels.index('Navigation'), widget_but)

    def _delete_data(self):
        """
        Deletes row at sending button position
        """

        button = self.sender()
        index = self.indexAt(button.parentWidget().pos())
        if index.isValid():
            row = index.row()
            self.removeRow(row)
            self.input_files.pop(row)
            if row in self.incompatible_data.keys():
                self.incompatible_data.pop(row)
            self.inputFilesChanged.emit()

    def _move_down(self):
        """
        Move row at sending button position one place down
        """

        button= self.sender()
        index = self.indexAt(button.parentWidget().pos())
        row = index.row()
        column = index.column()
        if row < self.rowCount() - 1:
            self.insertRow(row + 2)
            for i in range(self.columnCount()):
                self.setItem(row + 2, i, self.takeItem(row, i))
                self.setCurrentCell(row + 2, column)
            self.removeRow(row)
            self.setVerticalHeaderLabels(self.row_labels)

    def _move_up(self):
        """
        Move row at sending button position one place up
        """

        button = self.sender()
        index = self.indexAt(button.parentWidget().pos())
        row = index.row()
        column = index.column()
        if row > 0:
            self.insertRow(row - 1)
            for i in range(self.columnCount()):
                self.setItem(row - 1, i, self.takeItem(row + 1, i))
                self.setCurrentCell(row - 1, column)
            self.removeRow(row + 1)
            self.setVerticalHeaderLabels(self.row_labels)
