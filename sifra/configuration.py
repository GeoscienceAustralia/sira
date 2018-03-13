import os
import time
import json


class Configuration:
    """
    Reads all the simulation configuration constants to be read by other classes
    """
    def __init__(self, configuration_file_path):
        with open(configuration_file_path, 'r') as f:
            config = json.load(f)

        # reading in simulation scenario parameters
        self.SCENARIO_NAME = config['Scenario']['SCENARIO_NAME']
        self.INTENSITY_MEASURE_PARAM = config['Scenario']['INTENSITY_MEASURE_PARAM']
        self.INTENSITY_MEASURE_UNIT = config['Scenario']['INTENSITY_MEASURE_UNIT']
        self.SCENARIO_HAZARD_VALUES = config['Scenario']['SCENARIO_HAZARD_VALUES']

        self.PGA_MIN = config['Hazard']['PGA_MIN']
        self.PGA_MAX = config['Hazard']['PGA_MAX']
        self.PGA_STEP = config['Hazard']['PGA_STEP']
        self.NUM_SAMPLES = config['Hazard']['NUM_SAMPLES']
        self.HAZARD_TYPE = config['Hazard']['HAZARD_TYPE']
        self.HAZARD_RASTER = config['Hazard']['HAZARD_RASTER']

        self.TIME_UNIT = config['Restoration']['TIME_UNIT']
        self.RESTORE_PCT_CHECKPOINTS = config['Restoration']['RESTORE_PCT_CHECKPOINTS']
        self.RESTORE_TIME_STEP = config['Restoration']['RESTORE_TIME_STEP']
        self.RESTORE_TIME_MAX = config['Restoration']['RESTORE_TIME_MAX']
        self.RESTORATION_STREAMS = config['Restoration']['RESTORATION_STREAMS']

        self.SYSTEM_CLASSES = config['System']['SYSTEM_CLASSES']
        self.SYSTEM_CLASS = config['System']['SYSTEM_CLASS']
        self.SYSTEM_SUBCLASS = config['System']['SYSTEM_SUBCLASS']
        self.COMMODITY_FLOW_TYPES = config['System']['COMMODITY_FLOW_TYPES']
        self.SYS_CONF_FILE_NAME = config['System']['SYS_CONF_FILE_NAME']

        self.INPUT_DIR_NAME = config['Input']['INPUT_DIR_NAME']

        root = os.getcwd()
        if 'sifra' in root:
            root = os.path.dirname(root)

        self.SYS_CONF_FILE = os.path.join(root,
                                          self.INPUT_DIR_NAME,
                                          self.SYS_CONF_FILE_NAME)


        self.OUTPUT_DIR_NAME = config['Output']['OUTPUT_DIR_NAME'] + self.SCENARIO_NAME

        self.FIT_PE_DATA = config['Test']['FIT_PE_DATA']
        self.FIT_RESTORATION_DATA = config['Test']['FIT_RESTORATION_DATA']
        self.SAVE_VARS_NPY = config['Test']['SAVE_VARS_NPY']

        self.MULTIPROCESS = config['Switches']['MULTIPROCESS']
        self.RUN_CONTEXT = config['Switches']['RUN_CONTEXT']

        # reading in setup information

        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.INPUT_PATH = os.path.join(self.ROOT_DIR, self.INPUT_DIR_NAME)

        timestamp = time.strftime('_%Y%m%d_%H%M%S')
        output_dir_timestamped = self.OUTPUT_DIR_NAME + timestamp

        self.OUTPUT_PATH = os.path.join(self.ROOT_DIR, output_dir_timestamped)

        # create output dir: root/SCENARIO_NAME+_timestamp
        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)

        self.RAW_OUTPUT_DIR = os.path.join(self.OUTPUT_PATH, 'RAW_OUTPUT')

        # create output dir: root/SCENARIO_NAME+_timestamp/RAW_OUTPUT
        if not os.path.exists(self.RAW_OUTPUT_DIR):
            os.makedirs(self.RAW_OUTPUT_DIR)
