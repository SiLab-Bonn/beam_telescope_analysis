from __future__ import division

import inspect

import numpy as np

from beam_telescope_analysis.tools import geometry_utils


class Dut(object):
    ''' DUT base class.
    '''

    # List of member variables that are allowed to be changed/set (e.g., during initialization).
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "material_budget"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, material_budget=None):
        self.name = name  # string
        self.translation_x = translation_x  # in um
        self.translation_y = translation_y  # in um
        self.translation_z = translation_z  # in um
        self.rotation_alpha = rotation_alpha  # in rad
        self.rotation_beta = rotation_beta  # in rad
        self.rotation_gamma = rotation_gamma  # in rad
        self.material_budget = 0.0 if material_budget is None else material_budget  # the material budget is defined as the thickness devided by the radiation length

    def __setattr__(self, name, value):
        ''' Only allow the change of attributes that are listed in the class attribute 'dut_attributes' or during init.
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

    # Various properties of the DUT; check for correct input type and cast to proper type.
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

    # DUT methods
    @classmethod
    def from_dut(cls, dut, **kwargs):
        ''' Get new DUT from existing DUT. Copy all properties to new DUT.
        '''
        init_variables = list(set(cls.dut_attributes) & set(dut.dut_attributes))
        init_dict = {key: getattr(dut, key) for key in init_variables}
        init_dict.update(kwargs)
        return cls(**init_dict)

    def x_extent(self, global_position=False):
        ''' Size of the DUT in X dimension.
        '''
        raise NotImplementedError

    def y_extent(self, global_position=False):
        ''' Size of the DUT in Z dimension.
        '''
        raise NotImplementedError

    def z_extent(self, global_position=False):
        ''' Size of the DUT in Z dimension.
        '''
        raise NotImplementedError

    def index_to_local_position(self, index):
        ''' Transform index to local position.
        '''
        raise NotImplementedError

    def local_position_to_index(self, x, y, z=None):
        ''' Transform local position to index.
        '''
        raise NotImplementedError

    def local_to_global_position(self, x, y, z=None, translation_x=None, translation_y=None, translation_z=None, rotation_alpha=None, rotation_beta=None, rotation_gamma=None):
        ''' Transform local position to global position.
        '''
        raise NotImplementedError

    def global_to_local_position(self, x, y, z, translation_x=None, translation_y=None, translation_z=None, rotation_alpha=None, rotation_beta=None, rotation_gamma=None):
        ''' Transform global position to local position.
        '''
        raise NotImplementedError

    def index_to_global_position(self, index, translation_x=None, translation_y=None, translation_z=None, rotation_alpha=None, rotation_beta=None, rotation_gamma=None):
        ''' Transform index to global position.
        '''
        raise NotImplementedError

    def global_position_to_index(self, x, y, z, translation_x=None, translation_y=None, translation_z=None, rotation_alpha=None, rotation_beta=None, rotation_gamma=None):
        ''' Transform global position to index.
        '''
        raise NotImplementedError


class RectangularPixelDut(Dut):
    ''' DUT with rectangular pixels.
    '''
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "material_budget", "column_size", "row_size", "n_columns", "n_rows", "column_limit", "row_limit"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, column_size, row_size, n_columns, n_rows, column_limit=None, row_limit=None, material_budget=None):
        super(RectangularPixelDut, self).__init__(name=name, material_budget=material_budget, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma)
        self.column_size = column_size
        self.row_size = row_size
        self.n_columns = n_columns
        self.n_rows = n_rows
        if column_limit is None:
            self.column_limit = self.x_extent()
        else:
            self.column_limit = column_limit
        if row_limit is None:
            self.row_limit = self.y_extent()
        else:
            self.row_limit = row_limit

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
    def pixel_size(self):
        return (self.column_size, self.row_size)

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

    @property
    def n_pixel(self):
        return (self.n_columns, self.n_rows)

    @property
    def column_limit(self):
        return self._column_limit

    @column_limit.setter
    def column_limit(self, limit):
        self._column_limit = (float(limit[0]), float(limit[1]))

    @property
    def row_limit(self):
        return self._row_limit

    @row_limit.setter
    def row_limit(self, limit):
        self._row_limit = (float(limit[0]), float(limit[1]))

    def x_extent(self, global_position=False):
        if global_position:
            conv = self.index_to_global_position
        else:
            conv = self.index_to_local_position
        x_values = conv([0.5, 0.5, self.n_columns + 0.5, self.n_columns + 0.5], [0.5, self.n_rows + 0.5, 0.5, self.n_rows + 0.5])[0]
        return min(x_values), max(x_values)

    def y_extent(self, global_position=False):
        if global_position:
            conv = self.index_to_global_position
        else:
            conv = self.index_to_local_position
        y_values = conv([0.5, 0.5, self.n_columns + 0.5, self.n_columns + 0.5], [0.5, self.n_rows + 0.5, 0.5, self.n_rows + 0.5])[1]
        return min(y_values), max(y_values)

    def z_extent(self, global_position=False):
        if global_position:
            conv = self.index_to_global_position
        else:
            conv = self.index_to_local_position
        z_values = conv([0.5, 0.5, self.n_columns + 0.5, self.n_columns + 0.5], [0.5, self.n_rows + 0.5, 0.5, self.n_rows + 0.5])[2]
        return min(z_values), max(z_values)

    def index_to_local_position(self, column, row):
        if isinstance(column, (list, tuple)) or isinstance(row, (list, tuple)):
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
            raise ValueError("Column/row out of limits.")
        x = self.column_size * (column - 0.5 - (0.5 * self.n_columns))
        y = self.row_size * (row - 0.5 - (0.5 * self.n_rows))
        z = np.zeros_like(x)  # all DUTs have their origin in x=y=z=0
        return x, y, z

    def local_position_to_index(self, x, y, z=None):
        if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
            x = np.array(x, dtype=np.float64)
            y = np.array(y, dtype=np.float64)
        if z is not None:
            if isinstance(z, (list, tuple)):
                z = np.array(z, dtype=np.float64)
            # check for valid z coordinates
            if not np.allclose(np.nan_to_num(z), 0.0):
                raise RuntimeError('The local z positions contain values z!=0.')
        column = np.full_like(x, fill_value=np.nan, dtype=np.float64)
        row = np.full_like(x, fill_value=np.nan, dtype=np.float64)
        # check for hit index or cluster index is out of range
        hit_selection = np.logical_and(
            np.logical_and(x >= (-0.5 * self.n_columns) * self.column_size, x <= (0.5 * self.n_columns) * self.column_size),
            np.logical_and(y >= (-0.5 * self.n_rows) * self.row_size, y <= (0.5 * self.n_rows) * self.row_size))
        if not np.all(hit_selection):
            raise ValueError("x/y position out of limits.")
        column = (x / self.column_size) + 0.5 + (0.5 * self.n_columns)
        row = (y / self.row_size) + 0.5 + (0.5 * self.n_rows)
        return column, row

    def local_to_global_position(self, x, y, z=None, translation_x=None, translation_y=None, translation_z=None, rotation_alpha=None, rotation_beta=None, rotation_gamma=None):
        if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
            x = np.array(x, dtype=np.float64)
            y = np.array(y, dtype=np.float64)
        if z is None:
            z = np.zeros_like(x)
        elif isinstance(z, (list, tuple)):
            z = np.array(z, dtype=np.float64)
        # check for valid z coordinates
        if translation_x is None and translation_y is None and translation_z is None and rotation_alpha is None and rotation_beta is None and rotation_gamma is None and not np.allclose(np.nan_to_num(z), 0.0):
            raise RuntimeError('The local z positions contain values z!=0.')
        # apply DUT alignment
        transformation_matrix = geometry_utils.local_to_global_transformation_matrix(
            x=self.translation_x if translation_x is None else float(translation_x),
            y=self.translation_y if translation_y is None else float(translation_y),
            z=self.translation_z if translation_z is None else float(translation_z),
            alpha=self.rotation_alpha if rotation_alpha is None else float(rotation_alpha),
            beta=self.rotation_beta if rotation_beta is None else float(rotation_beta),
            gamma=self.rotation_gamma if rotation_gamma is None else float(rotation_gamma))
        return geometry_utils.apply_transformation_matrix(
            x=x,
            y=y,
            z=z,
            transformation_matrix=transformation_matrix)

    def global_to_local_position(self, x, y, z, translation_x=None, translation_y=None, translation_z=None, rotation_alpha=None, rotation_beta=None, rotation_gamma=None):
        if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)) or isinstance(z, (list, tuple)):
            x = np.array(x, dtype=np.float64)
            y = np.array(y, dtype=np.float64)
            z = np.array(z, dtype=np.float64)
        # apply DUT inverse alignment
        transformation_matrix = geometry_utils.global_to_local_transformation_matrix(
            x=self.translation_x if translation_x is None else float(translation_x),
            y=self.translation_y if translation_y is None else float(translation_y),
            z=self.translation_z if translation_z is None else float(translation_z),
            alpha=self.rotation_alpha if rotation_alpha is None else float(rotation_alpha),
            beta=self.rotation_beta if rotation_beta is None else float(rotation_beta),
            gamma=self.rotation_gamma if rotation_gamma is None else float(rotation_gamma))
        x, y, z = geometry_utils.apply_transformation_matrix(
            x=x,
            y=y,
            z=z,
            transformation_matrix=transformation_matrix)
        # check for valid z coordinates
        if translation_x is None and translation_y is None and translation_z is None and rotation_alpha is None and rotation_beta is None and rotation_gamma is None and not np.allclose(np.nan_to_num(z), 0.0):
            raise RuntimeError('The local z positions contain values z!=0.')
        return x, y, z

    def index_to_global_position(self, column, row, translation_x=None, translation_y=None, translation_z=None, rotation_alpha=None, rotation_beta=None, rotation_gamma=None):
        return self.local_to_global_position(*self.index_to_local_position(column=column, row=row), translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma)

    def global_position_to_index(self, x, y, z, translation_x=None, translation_y=None, translation_z=None, rotation_alpha=None, rotation_beta=None, rotation_gamma=None):
        return self.local_position_to_index(*self.global_to_local_position(x=x, y=y, z=z, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma))


class FEI4(RectangularPixelDut):
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "column_limit", "row_limit", "material_budget"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, column_limit=None, row_limit=None, material_budget=None):
        super(FEI4, self).__init__(name=name, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma, column_limit=column_limit, row_limit=row_limit, material_budget=material_budget, column_size=250.0, row_size=50.0, n_columns=80, n_rows=336)


class RD53A(RectangularPixelDut):
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "column_limit", "row_limit", "material_budget"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, column_limit=None, row_limit=None, material_budget=None):
        super(RD53A, self).__init__(name=name, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma, column_limit=column_limit, row_limit=row_limit, material_budget=material_budget, column_size=50.0, row_size=50.0, n_columns=400, n_rows=192)


class PSI46(RectangularPixelDut):
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "column_limit", "row_limit", "material_budget"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, column_limit=None, row_limit=None, material_budget=None):
        super(PSI46, self).__init__(name=name, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma, column_limit=column_limit, row_limit=row_limit, material_budget=material_budget, column_size=150.0, row_size=100.0, n_columns=52, n_rows=80)


class Mimosa26(RectangularPixelDut):
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "column_limit", "row_limit", "material_budget"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, column_limit=None, row_limit=None, material_budget=None):
        super(Mimosa26, self).__init__(name=name, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma, column_limit=column_limit, row_limit=row_limit, material_budget=material_budget, column_size=18.4, row_size=18.4, n_columns=1152, n_rows=576)


class Diamond3DpCVD(FEI4):
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "column_limit", "row_limit", "material_budget", "sensor_position"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, sensor_position, column_limit=None, row_limit=None, material_budget=None):
        self.sensor_position = sensor_position
        super(Diamond3DpCVD, self).__init__(name=name, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma, column_limit=column_limit, row_limit=row_limit, material_budget=material_budget)

    @property
    def sensor_position(self):
        return self._sensor_position

    @sensor_position.setter
    def sensor_position(self, position):
        self._sensor_position = (int(position[0]), int(position[1]))

    def index_to_local_position(self, column, row):
        column = np.array(column, dtype=np.float64)
        row = np.array(row, dtype=np.float64)
        x, y, z = super(Diamond3DpCVD, self).index_to_local_position(column=column, row=row)
        # select all pixels, move positions where the bump bonds are
        hit_selection = np.mod(column, 2) == 1
        x[hit_selection] -= 100
        hit_selection = np.mod(column, 2) == 0
        x[hit_selection] += 100
        # select square pixels (125um x 100um)
        hit_selection = (column >= (self.sensor_position[0])) & (column < (self.sensor_position[0] + 12)) & (row >= (self.sensor_position[1])) & (row < (self.sensor_position[1] + 30))
        # defects
        hit_selection &= ~((column == (self.sensor_position[0] + 1)) & (row >= (self.sensor_position[1] + 8)) & (row < (self.sensor_position[1] + 12)))
        hit_selection &= ~((column == (self.sensor_position[0] + 4)) & (row >= (self.sensor_position[1] + 12)) & (row < (self.sensor_position[1] + 16)))
        hit_selection &= ~((column == (self.sensor_position[0] + 3)) & (row >= (self.sensor_position[1] + 24)) & (row < (self.sensor_position[1] + 30)))
        hit_selection &= ~((column == (self.sensor_position[0] + 8)) & (row >= (self.sensor_position[1])) & (row < (self.sensor_position[1] + 8)))
        hit_selection &= ~((column >= (self.sensor_position[0] + 10)) & (column < (self.sensor_position[0] + 12)) & (row >= (self.sensor_position[1] + 8)) & (row < (self.sensor_position[1] + 18)))
        hit_selection_0 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 0)
        hit_selection_0_0 = hit_selection_0 & (np.mod(row - self.sensor_position[1], 2) == 0)
        x[hit_selection_0_0] += 100
        y[hit_selection_0_0] += 25
        hit_selection_0_1 = hit_selection_0 & (np.mod(row - self.sensor_position[1], 2) == 1)
        x[hit_selection_0_1] += 225
        y[hit_selection_0_1] -= 25
        hit_selection_1 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 1)
        # hit_selection_1_0 = hit_selection_1 & (np.mod(row - self.sensor_position[1], 2) == 0)
        # x[hit_selection_1_0] -= 100
        # y[hit_selection_1_0] += 25
        hit_selection_1_1 = hit_selection_1 & (np.mod(row - self.sensor_position[1], 2) == 1)
        x[hit_selection_1_1] -= 100
        y[hit_selection_1_1] -= 25
        hit_selection_2 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 2)
        # hit_selection_2_0 = hit_selection_2 & (np.mod(row - self.sensor_position[1], 2) == 0)
        # x[hit_selection_2_0] += 100
        # y[hit_selection_2_0] += 25
        hit_selection_2_1 = hit_selection_2 & (np.mod(row - self.sensor_position[1], 2) == 1)
        x[hit_selection_2_1] += 100
        y[hit_selection_2_1] -= 25
        hit_selection_3 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 3)
        hit_selection_3_0 = hit_selection_3 & (np.mod(row - self.sensor_position[1], 2) == 0)
        x[hit_selection_3_0] -= 100
        y[hit_selection_3_0] += 25
        hit_selection_3_1 = hit_selection_3 & (np.mod(row - self.sensor_position[1], 2) == 1)
        x[hit_selection_3_1] -= 225
        y[hit_selection_3_1] -= 25
        # select hexagonal pixels (115.5um x 133.3um)
        hit_selection = (column >= (self.sensor_position[0])) & (column < (self.sensor_position[0] + 12)) & (row >= (self.sensor_position[1] + 30)) & (row < (self.sensor_position[1] + 62))
        # defects
        hit_selection &= ~((column >= (self.sensor_position[0] + 10)) & (column < (self.sensor_position[0] + 12)) & (row == (self.sensor_position[1] + 30)))
        hit_selection &= ~((column >= (self.sensor_position[0] + 10)) & (column < (self.sensor_position[0] + 12)) & (row >= (self.sensor_position[1] + 32)) & (row < (self.sensor_position[1] + 35)))
        hit_selection_0 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 0)
        hit_selection_0_0 = hit_selection_0 & (np.mod(row - self.sensor_position[1] - 30, 4) == 0)
        x[hit_selection_0_0] += 167.265
        y[hit_selection_0_0] += 75
        hit_selection_0_1 = hit_selection_0 & (np.mod(row - self.sensor_position[1] - 30, 4) == 1)
        x[hit_selection_0_1] += 51.785
        y[hit_selection_0_1] += 25
        hit_selection_0_2 = hit_selection_0 & (np.mod(row - self.sensor_position[1] - 30, 4) == 2)
        x[hit_selection_0_2] += 109.525
        y[hit_selection_0_2] += 75
        # hit_selection_0_3 = hit_selection_0 & (np.mod(row - self.sensor_position[1] - 30, 4) == 3)
        # x[hit_selection_0_3] += 225
        # y[hit_selection_0_3] += 25
        hit_selection_1 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 1)
        hit_selection_1_0 = hit_selection_1 & (np.mod(row - self.sensor_position[1] - 30, 4) == 0)
        x[hit_selection_1_0] -= 167.265
        y[hit_selection_1_0] += 75
        hit_selection_1_1 = hit_selection_1 & (np.mod(row - self.sensor_position[1] - 30, 4) == 1)
        x[hit_selection_1_1] -= 51.785
        y[hit_selection_1_1] += 25
        hit_selection_1_2 = hit_selection_1 & (np.mod(row - self.sensor_position[1] - 30, 4) == 2)
        x[hit_selection_1_2] -= 109.525
        y[hit_selection_1_2] += 75
        hit_selection_1_3 = hit_selection_1 & (np.mod(row - self.sensor_position[1] - 30, 4) == 3)
        x[hit_selection_1_3] -= 225
        y[hit_selection_1_3] += 25
        hit_selection_2 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 2)
        hit_selection_2_0 = hit_selection_2 & (np.mod(row - self.sensor_position[1] - 30, 4) == 0)
        x[hit_selection_2_0] += 167.265
        y[hit_selection_2_0] += 75
        hit_selection_2_1 = hit_selection_2 & (np.mod(row - self.sensor_position[1] - 30, 4) == 1)
        x[hit_selection_2_1] += 51.785
        y[hit_selection_2_1] += 25
        hit_selection_2_2 = hit_selection_2 & (np.mod(row - self.sensor_position[1] - 30, 4) == 2)
        x[hit_selection_2_2] += 109.525
        y[hit_selection_2_2] += 75
        hit_selection_2_3 = hit_selection_2 & (np.mod(row - self.sensor_position[1] - 30, 4) == 3)
        x[hit_selection_2_3] += 225
        y[hit_selection_2_3] += 25
        hit_selection_3 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 3)
        hit_selection_3_0 = hit_selection_3 & (np.mod(row - self.sensor_position[1] - 30, 4) == 0)
        x[hit_selection_3_0] -= 167.265
        y[hit_selection_3_0] += 75
        hit_selection_3_1 = hit_selection_3 & (np.mod(row - self.sensor_position[1] - 30, 4) == 1)
        x[hit_selection_3_1] -= 51.785
        y[hit_selection_3_1] += 25
        hit_selection_3_2 = hit_selection_3 & (np.mod(row - self.sensor_position[1] - 30, 4) == 2)
        x[hit_selection_3_2] -= 109.525
        y[hit_selection_3_2] += 75
        # hit_selection_3_3 = hit_selection_3 & (np.mod(row - self.sensor_position[1] - 30, 4) == 3)
        # x[hit_selection_3_3] -= 225
        # y[hit_selection_3_3] += 25
        return x, y, z


class DiamondPseudo3DpCVD(FEI4):
    dut_attributes = ["name", "translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma", "column_limit", "row_limit", "material_budget", "sensor_position"]

    def __init__(self, name, translation_x, translation_y, translation_z, rotation_alpha, rotation_beta, rotation_gamma, sensor_position, column_limit=None, row_limit=None, material_budget=None):
        self.sensor_position = sensor_position
        super(DiamondPseudo3DpCVD, self).__init__(name=name, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, rotation_alpha=rotation_alpha, rotation_beta=rotation_beta, rotation_gamma=rotation_gamma, column_limit=column_limit, row_limit=row_limit, material_budget=material_budget)

    @property
    def sensor_position(self):
        return self._sensor_position

    @sensor_position.setter
    def sensor_position(self, position):
        self._sensor_position = (int(position[0]), int(position[1]))

    def index_to_local_position(self, column, row):
        column = np.array(column, dtype=np.float64)
        row = np.array(row, dtype=np.float64)
        x, y, z = super(DiamondPseudo3DpCVD, self).index_to_local_position(column=column, row=row)
        # select all pixels, move positions where the bump bonds are
        hit_selection = np.mod(column, 2) == 1
        x[hit_selection] -= 100
        hit_selection = np.mod(column, 2) == 0
        x[hit_selection] += 100
        # select square pixels (125um x 100um)
        hit_selection = (column >= (self.sensor_position[0])) & (column < (self.sensor_position[0] + 12)) & (row >= (self.sensor_position[1])) & (row < (self.sensor_position[1] + 30))
        hit_selection_0 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 0)
        hit_selection_0_0 = hit_selection_0 & (np.mod(row - self.sensor_position[1], 2) == 0)
        x[hit_selection_0_0] += 100
        y[hit_selection_0_0] += 25
        hit_selection_0_1 = hit_selection_0 & (np.mod(row - self.sensor_position[1], 2) == 1)
        x[hit_selection_0_1] += 225
        y[hit_selection_0_1] -= 25
        hit_selection_1 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 1)
        # hit_selection_1_0 = hit_selection_1 & (np.mod(row - self.sensor_position[1], 2) == 0
        # x[hit_selection_1_0] -= 100
        # y[hit_selection_1_0] += 25
        hit_selection_1_1 = hit_selection_1 & (np.mod(row - self.sensor_position[1], 2) == 1)
        x[hit_selection_1_1] -= 100
        y[hit_selection_1_1] -= 25
        hit_selection_2 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 2)
        hit_selection_2_0 = hit_selection_2 & (np.mod(row - self.sensor_position[1], 2) == 0)
        x[hit_selection_2_0] += 100
        y[hit_selection_2_0] += 25
        # hit_selection_2_1 = hit_selection_2 & (np.mod(row - self.sensor_position[1], 2) == 1)
        # x[hit_selection_2_1] += 100
        # y[hit_selection_2_1] -= 25
        hit_selection_3 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 3)
        hit_selection_3_0 = hit_selection_3 & (np.mod(row - self.sensor_position[1], 2) == 0)
        x[hit_selection_3_0] -= 100
        y[hit_selection_3_0] += 25
        hit_selection_3_1 = hit_selection_3 & (np.mod(row - self.sensor_position[1], 2) == 1)
        x[hit_selection_3_1] -= 225
        y[hit_selection_3_1] -= 25
        # select hexagonal pixels (115.5um x 133,3um)
        hit_selection = (column >= (self.sensor_position[0])) & (column < (self.sensor_position[0] + 12)) & (row >= (self.sensor_position[1] + 30)) & (row < (self.sensor_position[1] + 62))
        hit_selection_0 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 0)
        hit_selection_0_0 = hit_selection_0 & (np.mod(row - self.sensor_position[1] - 30, 4) == 0)
        x[hit_selection_0_0] += 167.265
        y[hit_selection_0_0] += 75
        hit_selection_0_1 = hit_selection_0 & (np.mod(row - self.sensor_position[1] - 30, 4) == 1)
        x[hit_selection_0_1] += 51.785
        y[hit_selection_0_1] += 25
        hit_selection_0_2 = hit_selection_0 & (np.mod(row - self.sensor_position[1] - 30, 4) == 2)
        x[hit_selection_0_2] += 109.525
        y[hit_selection_0_2] += 75
        # hit_selection_0_3 = hit_selection_0 & (np.mod(row - self.sensor_position[1] - 30, 4) == 3)
        # x[hit_selection_0_3] += 225
        # y[hit_selection_0_3] += 25
        hit_selection_1 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 1)
        hit_selection_1_0 = hit_selection_1 & (np.mod(row - self.sensor_position[1] - 30, 4) == 0)
        x[hit_selection_1_0] -= 167.265
        y[hit_selection_1_0] += 75
        hit_selection_1_1 = hit_selection_1 & (np.mod(row - self.sensor_position[1] - 30, 4) == 1)
        x[hit_selection_1_1] -= 51.785
        y[hit_selection_1_1] += 25
        hit_selection_1_2 = hit_selection_1 & (np.mod(row - self.sensor_position[1] - 30, 4) == 2)
        x[hit_selection_1_2] -= 109.525
        y[hit_selection_1_2] += 75
        hit_selection_1_3 = hit_selection_1 & (np.mod(row - self.sensor_position[1] - 30, 4) == 3)
        x[hit_selection_1_3] -= 225
        y[hit_selection_1_3] += 25
        hit_selection_2 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 2)
        hit_selection_2_0 = hit_selection_2 & (np.mod(row - self.sensor_position[1] - 30, 4) == 0)
        x[hit_selection_2_0] += 167.265
        y[hit_selection_2_0] += 75
        hit_selection_2_1 = hit_selection_2 & (np.mod(row - self.sensor_position[1] - 30, 4) == 1)
        x[hit_selection_2_1] += 51.785
        y[hit_selection_2_1] += 25
        hit_selection_2_2 = hit_selection_2 & (np.mod(row - self.sensor_position[1] - 30, 4) == 2)
        x[hit_selection_2_2] += 109.525
        y[hit_selection_2_2] += 75
        hit_selection_2_3 = hit_selection_2 & (np.mod(row - self.sensor_position[1] - 30, 4) == 3)
        x[hit_selection_2_3] += 225
        y[hit_selection_2_3] += 25
        hit_selection_3 = hit_selection & (np.mod(column - self.sensor_position[0], 4) == 3)
        hit_selection_3_0 = hit_selection_3 & (np.mod(row - self.sensor_position[1] - 30, 4) == 0)
        x[hit_selection_3_0] -= 167.265
        y[hit_selection_3_0] += 75
        hit_selection_3_1 = hit_selection_3 & (np.mod(row - self.sensor_position[1] - 30, 4) == 1)
        x[hit_selection_3_1] -= 51.785
        y[hit_selection_3_1] += 25
        hit_selection_3_2 = hit_selection_3 & (np.mod(row - self.sensor_position[1] - 30, 4) == 2)
        x[hit_selection_3_2] -= 109.525
        y[hit_selection_3_2] += 75
        # hit_selection_3_3 = hit_selection_3 & (np.mod(row - self.sensor_position[1] - 30, 4) == 3)
        # x[hit_selection_3_3] -= 225
        # y[hit_selection_3_3] += 25
        return x, y, z
