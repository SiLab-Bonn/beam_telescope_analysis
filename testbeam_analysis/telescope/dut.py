import logging
import inspect

import numpy as np

from testbeam_analysis.tools import geometry_utils


class Dut(object):
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "material_budget"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, material_budget=None):
        self.name = name
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.translation_z = translation_z
        self.rotation_alpha = rotation_alpha
        self.rotation_beta = rotation_beta
        self.rotation_gamma = rotation_gamma
        self.material_budget = 0.0 if material_budget is None else material_budget

    def __setattr__(self, name, value):
        ''' Only allow the change of attributes that are in the class attribute 'dut_attributes' or during init.
        '''
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        init = False
        for item in calframe:
            if "__init__" in item[3]:
                for function in item[4]:
                    if self.__class__.__name__ in function:
                        init = True
                        break
        if (name[0] == '_' and name[1:] in self.dut_attributes) or name in self.dut_attributes or init:
            super(Dut, self).__setattr__(name, value)
        else:
            raise ValueError("Attribute '%s' not allowed to be changed." % name)

    def __str__(self):
        return ("DUT %s: " % self.__class__.__name__) + ", ".join([(name + ": " + str(getattr(self, name))) for name in self.dut_attributes])

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def translation_x(self):
        return self._translation_x

    @translation_x.setter
    def translation_x(self, translation_x):
        self._translation_x = float(translation_x)

    @property
    def translation_y(self):
        return self._translation_y

    @translation_y.setter
    def translation_y(self, translation_y):
        self._translation_y = float(translation_y)

    @property
    def translation_z(self):
        return self._translation_z

    @translation_z.setter
    def translation_z(self, translation_z):
        self._translation_z = float(translation_z)

    @property
    def rotation_alpha(self):
        return self._rotation_alpha

    @rotation_alpha.setter
    def rotation_alpha(self, rotation_alpha):
        self._rotation_alpha = float(rotation_alpha)

    @property
    def rotation_beta(self):
        return self._rotation_beta

    @rotation_beta.setter
    def rotation_beta(self, rotation_beta):
        self._rotation_beta = float(rotation_beta)

    @property
    def rotation_gamma(self):
        return self._rotation_gamma

    @rotation_gamma.setter
    def rotation_gamma(self, rotation_gamma):
        self._rotation_gamma = float(rotation_gamma)

    @property
    def material_budget(self):
        return self._material_budget

    @material_budget.setter
    def material_budget(self, material_budget):
        self._material_budget = float(material_budget)

    def x_limit(self):
        raise NotImplementedError

    def y_limit(self):
        raise NotImplementedError

    def z_limit(self):
        raise NotImplementedError

    def x_size(self):
        raise NotImplementedError

    def y_size(self):
        raise NotImplementedError

    def z_size(self):
        raise NotImplementedError

    def index_to_position(self, column, row):
        raise NotImplementedError

    def position_to_index(self, x, y, z):
        raise NotImplementedError


class RectangularPixelDut(Dut):
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "material_budget", "column_size", "row_size", "n_columns", "n_rows"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, column_size, row_size, n_columns, n_rows, material_budget=None):
        super(RectangularPixelDut, self).__init__(name=name, material_budget=material_budget, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma)
        self.column_size = column_size
        self.row_size = row_size
        self.n_columns = n_columns
        self.n_rows = n_rows

    @property
    def column_size(self):
        return self._column_size

    @column_size.setter
    def column_size(self, column_size):
        self._column_size = float(column_size)

    @property
    def row_size(self):
        return self._row_size

    @row_size.setter
    def row_size(self, row_size):
        self._row_size = float(row_size)

    @property
    def n_columns(self):
        return self._n_columns

    @n_columns.setter
    def n_columns(self, n_columns):
        self._n_columns = int(n_columns)

    @property
    def n_rows(self):
        return self._n_rows

    @n_rows.setter
    def n_rows(self, n_rows):
        self._n_rows = int(n_rows)

    def x_limit(self, global_position=False):
        if global_position:
            conv = self.index_to_global_position
        else:
            conv = self.index_to_local_position
        x_values = conv([0.5, 0.5, self.n_columns + 0.5, self.n_columns + 0.5], [0.5, self.n_rows + 0.5, 0.5, self.n_rows + 0.5])[0]
        return min(x_values), max(x_values)

    def y_limit(self, global_position=False):
        if global_position:
            conv = self.index_to_global_position
        else:
            conv = self.index_to_local_position
        y_values = conv([0.5, 0.5, self.n_columns + 0.5, self.n_columns + 0.5], [0.5, self.n_rows + 0.5, 0.5, self.n_rows + 0.5])[1]
        return min(y_values), max(y_values)

    def z_limit(self, global_position=False):
        if global_position:
            conv = self.index_to_global_position
        else:
            conv = self.index_to_local_position
        z_values = conv([0.5, 0.5, self.n_columns + 0.5, self.n_columns + 0.5], [0.5, self.n_rows + 0.5, 0.5, self.n_rows + 0.5])[2]
        return min(z_values), max(z_values)

    def x_size(self, global_position=False):
        return np.squeeze(np.diff(self.x_limit(global_position=global_position)))

    def y_size(self, global_position=False):
        return np.squeeze(np.diff(self.y_limit(global_position=global_position)))

    def z_size(self, global_position=False):
        return np.squeeze(np.diff(self.z_limit(global_position=global_position)))

    def index_to_local_position(self, column, row):
        column = np.array(column, dtype=np.float64)
        row = np.array(row, dtype=np.float64)
        # from index to local coordinates
        x = np.full_like(column, fill_value=np.nan, dtype=np.float64)
        y = np.full_like(column, fill_value=np.nan, dtype=np.float64)
        z = np.full_like(column, fill_value=np.nan, dtype=np.float64)
        # check for hit index or cluster index is out of range
        hit_selection = np.logical_and(
            np.logical_and(column >= 0.5, column <= self.n_columns + 0.5),
            np.logical_and(row >= 0.5, row <= self.n_rows + 0.5))
        if not np.all(hit_selection):
            logging.warning("Column/row out of limits.")
        x[hit_selection] = self.column_size * (column[hit_selection] - 0.5 - (0.5 * self.n_columns))
        y[hit_selection] = self.row_size * (row[hit_selection] - 0.5 - (0.5 * self.n_rows))
        z[hit_selection] = 0.0  # all DUTs have their origin in 0, 0, 0
        return x, y, z

    def local_position_to_index(self, x, y, z):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        z = np.array(z, dtype=np.float64)
        # check for valid z coordinates
        if not np.allclose(np.nan_to_num(z), 0.0):
            raise RuntimeError('The local z coordinate is z!=0.')
        column = np.full_like(x, fill_value=np.nan, dtype=np.float64)
        row = np.full_like(x, fill_value=np.nan, dtype=np.float64)
        # check for hit index or cluster index is out of range
        hit_selection = np.logical_and(
            np.logical_and(x >= -0.5 * self.n_columns * self.column_size, x <= 0.5 * self.n_columns * self.column_size),
            np.logical_and(x >= -0.5 * self.n_rows * self.row_size, x <= 0.5 * self.n_rows * self.row_size))
        if not np.all(hit_selection):
            logging.warning("x/y position out of limits.")
        column[hit_selection] = (x[hit_selection] / self.column_size) + 0.5 + (0.5 * self.n_columns)
        row[hit_selection] = (y[hit_selection] / self.row_size) + 0.5 + (0.5 * self.n_rows)
        return column, row

    def local_to_global_position(self, x, y, z):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        z = np.array(z, dtype=np.float64)
        # apply DUT alignment
        transformation_matrix = geometry_utils.local_to_global_transformation_matrix(
            x=self.translation_x,
            y=self.translation_y,
            z=self.translation_z,
            alpha=self.rotation_alpha,
            beta=self.rotation_beta,
            gamma=self.rotation_gamma)
        return geometry_utils.apply_transformation_matrix(
            x=x,
            y=y,
            z=z,
            transformation_matrix=transformation_matrix)

    def global_to_local_position(self, x, y, z):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        z = np.array(z, dtype=np.float64)
        # apply DUT inverse alignment
        transformation_matrix = geometry_utils.global_to_local_transformation_matrix(
            x=self.translation_x,
            y=self.translation_y,
            z=self.translation_z,
            alpha=self.rotation_alpha,
            beta=self.rotation_beta,
            gamma=self.rotation_gamma)
        return geometry_utils.apply_transformation_matrix(
            x=x,
            y=y,
            z=z,
            transformation_matrix=transformation_matrix)

    def index_to_global_position(self, column, row):
        return self.local_to_global_position(*self.index_to_local_position(column=column, row=row))

    def global_position_to_index(self, x, y, z):
        return self.local_position_to_index(*self.global_to_local_position(x=x, y=y, z=z))


class FEI4(RectangularPixelDut):
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "material_budget"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, material_budget=None):
        super(FEI4, self).__init__(name=name, material_budget=material_budget, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma, column_size=250.0, row_size=50.0, n_columns=80, n_rows=336)


class Mimosa26(RectangularPixelDut):
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "material_budget"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, material_budget=None):
        super(Mimosa26, self).__init__(name=name, material_budget=material_budget, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma, column_size=18.4, row_size=18.4, n_columns=1152, n_rows=576)
