"""
Enhanced slider widgets for floats and integers
"""

from PyQt5 import QtWidgets, QtCore


class IntSlider(QtWidgets.QSlider):
    def __init__(self, *args, **kwargs):
        super(IntSlider, self).__init__(*args, **kwargs)

        # Whether or not to de/increase slider range automatically on MouseReleaseEvent on max or min value of slider
        self.auto_scale_flag = True

    def setMaximum(self, p_int):
        super(IntSlider, self).setMaximum(p_int)
        # Scale step size
        #super(IntSlider, self).setSingleStep(int(0.01 * p_int) if int(0.01 * p_int) > 0 else 1)
        #super(IntSlider, self).setTickInterval(int(0.01 * p_int) if int(0.01 * p_int) > 0 else 1)
        #super(IntSlider, self).setPageStep(int(0.02*p_int) if int(0.02*p_int) > 0 else 2)

    def mousePressEvent(self, e):
        """ Jump to click position """
        self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), e.x(), self.width()))

    def mouseMoveEvent(self, e):
        """ Jump to pointer position while moving """
        self.setValue(QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), e.x(), self.width()))

    def mouseReleaseEvent(self, e):
        """ De/increase slider range when set to min/max """
        if self.auto_scale_flag:
            if self.value() == self.maximum():
                self.setMaximum(2 * self.maximum())
                self.setValue(int(0.5 * self.maximum()))
            elif self.value() == self.minimum():
                if self.maximum() % 2 == 0 and self.maximum() > 0:
                    self.setMaximum(int(0.5 * self.maximum()))
                else:
                    self.setMaximum(int(0.5 * (self.maximum() + 1)))
                self.setValue(int(0.5 * self.maximum()))

    def setAutoScale(self, v=True):
        self.auto_scale_flag = v


class FloatSlider(QtWidgets.QSlider):

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super(FloatSlider, self).__init__(*args, **kwargs)
        self.decimals = 2  # Equivalent to decimal place from leading decimal which changes on step
        self._max_int = 10 ** (self.decimals + 1)

        super(FloatSlider, self).setMinimum(0)
        super(FloatSlider, self).setMaximum(self._max_int)
        super(FloatSlider, self).valueChanged.connect(lambda _: self.valueChanged.emit(self.value()))

        # Whether or not to de/increase slider range automatically on MouseReleaseEvent on max or min value of slider
        self.auto_scale_flag = True

        self._min_value = 0.0
        self._max_value = 1.0

    @property
    def _value_range(self):
        return self._max_value - self._min_value

    def value(self):
        return float(super(FloatSlider, self).value()) / self._max_int * self._value_range + self._min_value

    def setValue(self, value):
        super(FloatSlider, self).setValue(int((value - self._min_value) / self._value_range * self._max_int))

    def setMinimum(self, value):
        if value > self._max_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._min_value = value
        self.setValue(self.value())

    def setMaximum(self, value):
        if value < self._min_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._max_value = value
        self.setValue(self.value())

    def setDecimals(self, decimals):
        self.decimals = decimals
        self._max_int = 10 ** (self.decimals + 1)
        super(FloatSlider, self).setMaximum(self._max_int)

    def minimum(self):
        return self._min_value

    def maximum(self):
        return self._max_value

    def mousePressEvent(self, e):
        """ Jump to click position """
        self.setValue(float(e.x())/self.width() * self._value_range + self.minimum())

    def mouseMoveEvent(self, e):
        """ Jump to pointer position while moving """
        self.setValue(float(e.x()) / self.width() * self._value_range + self.minimum())

    def mouseReleaseEvent(self, e):
        """ De/increase slider range when set to min/max """
        if self.auto_scale_flag:
            if self.value() == self.maximum():
                self.setMaximum(2.0 * self.maximum())
                self.setValue(0.5 * self.maximum())
            elif self.value() == self.minimum():
                self.setMaximum(0.5 * self.maximum())
                self.setValue(0.5 * self.maximum())

    def setAutoScale(self, v=True):
        self.auto_scale_flag = v
