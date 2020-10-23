#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from pathlib import Path


class Configuration:
    """
    Reads all the simulation configuration constants to be read
    by other classes
    """
    def __init__(self, config_path, model_path, output_path=None):
        """
        :param config_path: path to the config file
        :param run_mode: Default is 'impact' - this runs the full MC simulation
                         If option is 'analysis' then new output folders are
                         not created
        """
        # cater for 3 different types of config files .conf, .json , .ini
        # file_ext = Path(config_path).suffix
        # if file_ext != '.json':
        #     config_path = converter.convert_to_json(config_path)

        self.root_dir = Path(config_path).resolve().parent.parent

        self._VALID_HAZARD_INPUT_METHODS = [
            'hazard_array', 'calculated_array'
            'scenario_file', 'hazard_file'
        ]

        with open(config_path, 'r') as f:
            config = json.load(f)

        self.SCENARIO_NAME = str(config['SCENARIO_NAME'])
        self.MODEL_NAME = str(config['MODEL_NAME'])
        self.CONFIGURATION_ID = str(config['CONFIGURATION_ID'])

        # reading in simulation scenario parameters
        self.HAZARD_INTENSITY_MEASURE_PARAM = \
            str(config['HAZARD_INTENSITY_MEASURE_PARAM'])
        self.HAZARD_INTENSITY_MEASURE_UNIT = \
            str(config['HAZARD_INTENSITY_MEASURE_UNIT'])
        # Set of scenario(s) to investigate in detail
        #   - List of strings
        #   - Used in post processing stage
        self.FOCAL_HAZARD_SCENARIO_NAMES = \
            config['SCENARIO_FOCAL_HAZARD_SCENARIO_NAMES']
        self.FOCAL_HAZARD_SCENARIOS = config['SCENARIO_FOCAL_HAZARD_SCENARIOS']

        self.HAZARD_INPUT_METHOD = str(config['HAZARD_INPUT_METHOD'])
        self.HAZARD_TYPE = str(config['HAZARD_TYPE'])
        self.NUM_SAMPLES = int(config['HAZARD_NUM_SAMPLES'])

        if config['HAZARD_INPUT_METHOD'] in ['hazard_array', 'calculated_array']:
            self.HAZARD_INPUT_METHOD = 'Calculated_Array'
        elif config['HAZARD_INPUT_METHOD'] in ['scenario_file', 'hazard_file']:
            self.HAZARD_INPUT_METHOD = 'Hazard_File'
        else:
            raise ValueError(
                "Unrecognised HAZARD_INPUT_METHOD. Valid values are: " +
                "{}".format(self._VALID_HAZARD_INPUT_METHODS)
            )

        if self.HAZARD_INPUT_METHOD == 'Hazard_File':
            self.HAZARD_INPUT_FILE = str(
                Path(self.root_dir, 'input', config['HAZARD_INPUT_FILE']))
        elif self.HAZARD_INPUT_METHOD == 'Calculated_Array':
            self.HAZARD_INPUT_FILE = None

        self.INTENSITY_MEASURE_MIN = \
            float(config['HAZARD_INTENSITY_MEASURE_MIN'])
        self.INTENSITY_MEASURE_MAX = \
            float(config['HAZARD_INTENSITY_MEASURE_MAX'])
        self.INTENSITY_MEASURE_STEP = \
            float(config['HAZARD_INTENSITY_MEASURE_STEP'])

        self.TIME_UNIT = str(config['RESTORATION_TIME_UNIT'])
        self.RESTORE_PCT_CHECKPOINTS = int(config['RESTORATION_PCT_CHECKPOINTS'])
        self.RESTORE_TIME_STEP = int(config['RESTORATION_TIME_STEP'])
        self.RESTORE_TIME_MAX = int(config['RESTORATION_TIME_MAX'])
        self.RESTORATION_STREAMS = config['RESTORATION_STREAMS']

        self.INFRASTRUCTURE_LEVEL = str(config['SYSTEM_INFRASTRUCTURE_LEVEL'])
        self.SYSTEM_CLASSES = config['SYSTEM_CLASSES']
        self.SYSTEM_CLASS = config['SYSTEM_CLASS']
        self.SYSTEM_SUBCLASS = config['SYSTEM_SUBCLASS']
        self.COMMODITY_FLOW_TYPES = int(config['SYSTEM_COMMODITY_FLOW_TYPES'])
        self.COMPONENT_LOCATION_CONF = config['SYSTEM_COMPONENT_LOCATION_CONF']

        self.MULTIPROCESS = int(config['SWITCH_MULTIPROCESS'])
        self.RUN_CONTEXT = int(config['SWITCH_RUN_CONTEXT'])

        self.SWITCH_FIT_RESTORATION_DATA = \
            bool(config['SWITCH_FIT_RESTORATION_DATA'])
        self.SWITCH_SAVE_VARS_NPY = bool(config['SWITCH_SAVE_VARS_NPY'])

        self.INPUT_MODEL_PATH = Path(model_path)
        if output_path is not None:
            self.OUTPUT_PATH = Path(output_path)
        else:
            input_dir = Path(config_path).resolve().parent
            parent_dir = input_dir.parent
            self.OUTPUT_PATH = Path(parent_dir, 'output')

        # create output dir: root/SCENARIO_NAME+self.timestamp
        if not self.OUTPUT_PATH.exists():
            self.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        self.RAW_OUTPUT_DIR = Path(self.OUTPUT_PATH, 'RAW_OUTPUT')
        # create output dir: root/SCENARIO_NAME+self.timestamp/RAW_OUTPUT
        if not self.RAW_OUTPUT_DIR.exists():
            self.RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.SYS_CONF_FILE_NAME = self.INPUT_MODEL_PATH.stem
