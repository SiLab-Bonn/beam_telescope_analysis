from PyQt5 import QtWidgets, QtCore


class AnalysisBar(QtWidgets.QWidget):
    """
    Custom progress bar that allows showing text on a busy progress bar and automatically hides text when progressing.
    """

    def __init__(self, parent=None):
        super(AnalysisBar, self).__init__(parent)

        # Init progress bar
        self.bar = QtWidgets.QProgressBar()
        self.setValue(0)

        # Connect progress to hiding text
        self.bar.valueChanged.connect(self._hide_label)

        # Init label over progress bar
        self.label = QtWidgets.QLabel()
        color = 'white' if str(self.bar.palette().highlight().color().name()) == '#308cc6' else 'black'
        self.label.setStyleSheet('color: %s' % color)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        # Init layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.bar, 0, 0)
        layout.addWidget(self.label, 0, 0)

        self.setLayout(layout)

    def value(self):
        return self.bar.value()

    def setRange(self, min_, max_):
        self.bar.setRange(min_, max_)

    def setMaximum(self, value):
        self.bar.setMaximum = value

    def setMinimum(self, value):
        self.bar.setMinimum = value

    def setValue(self, value):
        self.bar.setValue(value)

    def setFormat(self, text):
        if self.bar.maximum() == self.bar.minimum() == 0:
            self.label.setText(text)
            self.label.show()
        else:
            self.bar.setFormat(text)

    def setBusy(self, text=None):
        self.setRange(0, 0)
        if text is not None:
            self.setFormat(text=text)

    def setFinished(self):
        self.bar.setRange(0, 1)
        self.bar.setValue(1)

    def _hide_label(self):
        if self.bar.text():
            self.label.hide()
        else:
            pass
