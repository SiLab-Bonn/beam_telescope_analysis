import logging
import os
from collections import OrderedDict
from inspect import isclass
import importlib

from yaml import safe_load, safe_dump

from beam_telescope_analysis.telescope.dut import Dut


def open_configuation(configuation):
    configuration_dict = {}
    if not configuation:
        pass
    elif isinstance(configuation, basestring):  # parse the first YAML document in a stream
        if os.path.isfile(os.path.abspath(configuation)):
            logging.info('Loading configuration from file %s', os.path.abspath(configuation))
            with open(os.path.abspath(configuation), mode='r') as f:
                configuration_dict.update(safe_load(f))
        else:  # YAML string
            configuration_dict.update(safe_load(configuation))
    elif isinstance(configuation, file):  # parse the first YAML document in a stream
        logging.info('Loading configuration from file %s', os.path.abspath(configuation.name))
        configuration_dict.update(safe_load(configuation))
    elif isinstance(configuation, (dict, OrderedDict)):  # conf is already a dict
        configuration_dict.update(configuation)
    else:
        raise ValueError("Configuration cannot be parsed.")
    return configuration_dict


class Telescope(object):
    telescope_attributes = ["translation_x", "translation_y", "translation_z", "rotation_alpha", "rotation_beta", "rotation_gamma"]

    def __init__(self, configuration_file=None, translation_x=0.0, translation_y=0.0, translation_z=0.0, rotation_alpha=0.0, rotation_beta=0.0, rotation_gamma=0.0):
        self.dut = {}
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.translation_z = translation_z
        self.rotation_alpha = rotation_alpha
        self.rotation_beta = rotation_beta
        self.rotation_gamma = rotation_gamma
        if configuration_file is not None:
            self.load_configuration(configuration_file)

    def __len__(self):
        return len(self.dut)

    def __getitem__(self, key):
        return self.dut[key]

    def __setitem__(self, key, value):
        if not isinstance(value, Dut):
            raise ValueError("Must be DUT.")
        self.dut[key] = value

    def __iter__(self):
        for sorted_key in sorted(self.dut.iterkeys()):
            yield self.dut[sorted_key]

    def __str__(self):
        string = ""
        for item in self:
            if string:
                string += '\n'
            string += str(item)
        string += "\nTelescope: %s:" % ", ".join([(name + ": " + str(getattr(self, name))) for name in self.telescope_attributes])
        return string

    @property
    def dut_names(self):
        return [item.name for item in self]

    @property
    def pixel_sizes(self):
        return [(item.column_size, item.row_size) for item in self]

    @property
    def n_pixels(self):
        return [(item.n_columns, item.n_rows) for item in self]

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

    def load_configuration(self, configuration_file=None):
        if configuration_file:
            self.configuration_file = configuration_file
        else:
            configuration_file = self.configuration_file

        configuration = None
        if configuration_file and os.path.isfile(os.path.abspath(configuration_file)):
            with open(os.path.abspath(configuration_file), mode='r') as f:
                configuration = safe_load(f)
        else:
            raise ValueError("No valid configuration file given.")

        if configuration and "TELESCOPE" in configuration:
            if "DUT" in configuration["TELESCOPE"]:
                for dut_id, dut_configuration in configuration["TELESCOPE"]["DUT"].items():
                    dut_type = dut_configuration.pop("dut_type", "RectangularPixelDut")
                    self.add_dut(dut_type=dut_type, dut_id=dut_id, **dut_configuration)
            for key, value in configuration["TELESCOPE"].items():
                if key in self.telescope_attributes:
                    setattr(self, key, value)

    def save_configuration(self, configuration_file=None, keep_others=False):
        if configuration_file:
            self.configuration_file = configuration_file
        else:
            configuration_file = self.configuration_file

        if configuration_file:
            if keep_others and os.path.isfile(os.path.abspath(configuration_file)):
                with open(os.path.abspath(configuration_file), mode='r') as f:
                    configuration = safe_load(f)
            else:
                configuration = {}
            # create Telescope configuration
            if 'TELESCOPE' not in configuration:
                configuration["TELESCOPE"] = {}
            for name in self.telescope_attributes:
                configuration["TELESCOPE"][name] = getattr(self, name)
            # overwrite all existing DUTs
            configuration["TELESCOPE"]["DUT"] = {}
            for dut_id, dut in self.dut.items():
                dut_configuration = {name: getattr(dut, name) for name in dut.dut_attributes}
                dut_configuration["dut_type"] = dut.__class__.__name__
                configuration["TELESCOPE"]["DUT"][dut_id] = dut_configuration
            if not configuration["TELESCOPE"]["DUT"]:
                configuration["TELESCOPE"]["DUT"] = {}
            with open(configuration_file, mode='w') as f:
                safe_dump(configuration, f, default_flow_style=False)
        else:
            raise ValueError("No valid configuration file given.")

    def add_dut(self, dut_type, dut_id, **kwargs):
        if not isinstance(dut_id, (long, int)):
            raise ValueError("DUT ID has to be an integer.")
        if "name" not in kwargs:
            kwargs["name"] = "DUT%d" % dut_id
        if isinstance(dut_type, basestring):
            m = importlib.import_module("beam_telescope_analysis.telescope.dut")
            # get the class, will raise AttributeError if class cannot be found
            c = getattr(m, dut_type)
            self.dut[dut_id] = c(**kwargs)
        elif isclass(dut_type):
            # instantiate the class
            self.dut[dut_id] = dut_type(**kwargs)
        else:
            raise ValueError("Unknown DUT type.")
