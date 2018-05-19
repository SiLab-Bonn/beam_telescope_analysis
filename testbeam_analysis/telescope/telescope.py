import logging
import os
from collections import OrderedDict
from inspect import isclass
import importlib

from yaml import safe_load, safe_dump

from testbeam_analysis.telescope.dut import Dut


def open_configuation(configuation):
    configuration_dict = {}
    if not configuation:
        pass
    elif isinstance(configuation, basestring):  # parse the first YAML document in a stream
        if os.path.isfile(os.path.abspath(configuation)):
            logging.info('Loading configuration from file %s', os.path.abspath(configuation))
            with open(os.path.abspath(configuation), 'r') as f:
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
    def __init__(self, configuration_file=None):
        self.dut = {}
        if configuration_file is not None:
            self.load_configuration(configuration_file)

    def __len__(self):
        return len(self.dut)

    def __getitem__(self, key):
        return self.dut[key]

    def __iter__(self):
        for sorted_key in sorted(self.dut.iterkeys()):
            yield self.dut[sorted_key]

    def __str__(self):
        for item in self:
            print item

    def load_configuration(self, configuration_file=None):
        if configuration_file:
            self.configuration_file = configuration_file
        else:
            configuration_file = self.configuration_file

        if configuration_file is not None:
            if os.path.isfile(os.path.abspath(configuration_file)):
                with open(os.path.abspath(configuration_file), 'r') as f:
                    configuration = safe_load(f)
                if not configuration:
                    configuration = {}
                if not configuration["TELESCOPE"]:
                    configuration["TELESCOPE"] = {}
                if not configuration["TELESCOPE"]["DUT"]:
                    configuration["TELESCOPE"]["DUT"] = {}
            else:
                configuration = {}
        else:
            raise ValueError("No configuration file given.")

        if "TELESCOPE" in configuration:
            if "DUT" in configuration["TELESCOPE"]:
                for dut_id, dut_configuration in configuration["TELESCOPE"]["DUT"].items():
                    dut_type = dut_configuration.pop("dut_type", "RectangularPixelDut")
                    self.add_dut(dut_type=dut_type, dut_id=dut_id, **dut_configuration)

    def save_configuration(self, configuration_file=None):
        if configuration_file:
            self.configuration_file = configuration_file
        else:
            configuration_file = self.configuration_file

        if configuration_file is not None:
            if os.path.isfile(os.path.abspath(configuration_file)):
                with open(os.path.abspath(configuration_file), 'r') as f:
                    configuration = safe_load(f)
            else:
                configuration = {}
            if not configuration:
                configuration = {}
            if not configuration["TELESCOPE"]:
                configuration["TELESCOPE"] = {}
            if not configuration["TELESCOPE"]["DUT"]:
                configuration["TELESCOPE"]["DUT"] = {}
            for dut_id, dut in self.dut.items():
                dut_configuration = {name: getattr(dut, name) for name in dut.dut_attributes}
                dut_configuration["dut_type"] = dut.__class__.__name__
                configuration["TELESCOPE"]["DUT"][dut_id] = dut_configuration
            if not configuration["TELESCOPE"]["DUT"]:
                configuration["TELESCOPE"]["DUT"] = {}
            with open(configuration_file, 'w') as f:
                safe_dump(configuration, f, default_flow_style=False)
        else:
            raise ValueError("No configuration file given.")

    def add_dut(self, dut_type, dut_id, **kwargs):
        if not isinstance(dut_id, (long, int)):
            raise ValueError("DUT ID has to be an integer.")
        if "name" not in kwargs:
            kwargs["name"] = "DUT%d" % dut_id
        if isinstance(dut_type, basestring):
            m = importlib.import_module("testbeam_analysis.telescope.dut")
            # get the class, will raise AttributeError if class cannot be found
            c = getattr(m, dut_type)
            self.dut[dut_id] = c(**kwargs)
        elif isclass(dut_type):
            # instantiate the class
            self.dut[dut_id] = dut_type(**kwargs)
        else:
            raise ValueError("Unknown DUT type.")
