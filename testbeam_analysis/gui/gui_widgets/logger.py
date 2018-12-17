import sys
import logging

from PyQt5 import QtCore


class AnalysisStream(QtCore.QObject):
    """
    Class to handle the stdout stream which is used to do thread safe logging
    since QtWidgets are not thread safe and therefore one can not directly log to GUIs
    widgets when performing analysis on different thread than main thread
    """

    _stdout = None
    _stderr = None
    messageWritten = QtCore.pyqtSignal(str)

    def flush(self):
        pass

    def fileno(self):
        return -1

    def write(self, msg):
        if not self.signalsBlocked():
            self.messageWritten.emit(str(msg))  # Python 3, was unicode before

    @staticmethod
    def stdout():
        if not AnalysisStream._stdout:
            AnalysisStream._stdout = AnalysisStream()
            sys.stdout = AnalysisStream._stdout
        return AnalysisStream._stdout

    @staticmethod
    def stderr():
        if not AnalysisStream._stderr:
            AnalysisStream._stderr = AnalysisStream()
            sys.stderr = AnalysisStream._stderr
        return AnalysisStream._stderr


class AnalysisLogger(logging.Handler):
    """
    Implements a logging handler which allows redirecting log thread-safe
    """

    def __init__(self, parent):
        super(AnalysisLogger, self).__init__()

    def emit(self, record):
        msg = self.format(record)
        if msg:
            AnalysisStream.stdout().write(msg)
