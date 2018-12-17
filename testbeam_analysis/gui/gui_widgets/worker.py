"""
Implements a worker object on which analysis can be done. The worker is then moved to a separate QThread
via the QObject.moveToThread() method.
"""

import traceback

from PyQt5 import QtCore


class AnalysisWorker(QtCore.QObject):
    """
    Implements a worker class which allows the worker to perform analysis / start vitables
    while being moved to an extra thread to keep the GUI responsive during analysis / vitables
    """

    finished = QtCore.pyqtSignal()
    exceptionSignal = QtCore.pyqtSignal(Exception, str)
    progressSignal = QtCore.pyqtSignal()

    def __init__(self, func, args=None, funcs_args=None):
        super(AnalysisWorker, self).__init__()

        # Main function which will be executed on this thread
        self.main_func = func
        # Arguments of main function
        self.args = args
        # Functions and arguments to perform analysis function;
        # if not None, main function is then AnalysisWidget.call_funcs()
        self.funcs_args = funcs_args

    def work(self):
        """
        Runs the function func with given argument args. If funcs_args is not None, it contains
        functions and corresponding arguments which are looped over and run. If errors or exceptions
        occur, a signal sends the exception to main thread. Most recent traceback wil be dumped in yaml file.
        """

        try:

            # Do analysis functions
            if self.funcs_args is not None:

                for func, kwargs in self.funcs_args:

                    # Each func has unique kwargs; used for analysis
                    self.main_func(func, kwargs)

                    # Emit progress signal
                    self.progressSignal.emit()

            # Do some arbitrary function
            else:
                self.main_func(self.args)

            self.finished.emit()

        except Exception as e:

            # Format traceback and send
            trc_bck = traceback.format_exc()

            # Emit exception signal
            self.exceptionSignal.emit(e, trc_bck)
