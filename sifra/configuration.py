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

        self.INTENSITY_MEASURE_MIN = config['Hazard']['INTENSITY_MEASURE_MIN']
        self.INTENSITY_MEASURE_MAX = config['Hazard']['INTENSITY_MEASURE_MAX']
        self.INTENSITY_MEASURE_STEP = config['Hazard']['INTENSITY_MEASURE_STEP']
        self.NUM_SAMPLES = config['Hazard']['NUM_SAMPLES']
        self.HAZARD_TYPE = config['Hazard']['HAZARD_TYPE']

        self.HAZARD_INPUT_METHOD = config['Hazard']['HAZARD_INPUT_METHOD']
        self.SCENARIO_FILE = config['Hazard']['SCENARIO_FILE']

        self.TIME_UNIT = config['Restoration']['TIME_UNIT']
        self.RESTORE_PCT_CHECKPOINTS = config['Restoration']['RESTORE_PCT_CHECKPOINTS']
        self.RESTORE_TIME_STEP = config['Restoration']['RESTORE_TIME_STEP']
        self.RESTORE_TIME_MAX = config['Restoration']['RESTORE_TIME_MAX']
        self.RESTORATION_STREAMS = config['Restoration']['RESTORATION_STREAMS']

        self.INFRASTRUCTURE_LEVEL = config['System']['INFRASTRUCTURE_LEVEL'] = config['System']['INFRASTRUCTURE_LEVEL']
        self.SYSTEM_CLASSES = config['System']['SYSTEM_CLASSES']
        self.SYSTEM_CLASS = config['System']['SYSTEM_CLASS']
        self.SYSTEM_SUBCLASS = config['System']['SYSTEM_SUBCLASS']
        self.COMMODITY_FLOW_TYPES = config['System']['COMMODITY_FLOW_TYPES']
        self.SYS_CONF_FILE_NAME = config['System']['SYS_CONF_FILE_NAME']

        self.INPUT_DIR_NAME = config['Input']['INPUT_DIR_NAME']

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

    def __str__(self):
        return "SCENARIO_NAME = " + str(self.SCENARIO_NAME) + '\n' + "INTENSITY_MEASURE_PARAM = " + \
               str(self.INTENSITY_MEASURE_PARAM) + '\n' + "INTENSITY_MEASURE_UNIT = " + \
               str(self.INTENSITY_MEASURE_UNIT) + '\n' + "SCENARIO_HAZARD_VALUES = " + \
               str(self.SCENARIO_HAZARD_VALUES) + '\n' + "INTENSITY_MEASURE_MIN = " + \
               str(self.INTENSITY_MEASURE_MIN) + '\n' + "INTENSITY_MEASURE_MAX = " + \
               str(self.INTENSITY_MEASURE_MAX) + '\n' + "INTENSITY_MEASURE_STEP = " + \
               str(self.INTENSITY_MEASURE_STEP) + '\n' + "NUM_SAMPLES = " + \
               str(self.NUM_SAMPLES) + '\n' + "HAZARD_TYPE = " + str(self.HAZARD_TYPE) + '\n' + \
               "HAZARD_INPUT_METHOD = " + str(self.HAZARD_INPUT_METHOD) + '\n' + "SCENARIO_FILE = " + \
               str(self.SCENARIO_FILE) + '\n' + "TIME_UNIT = " + str(self.TIME_UNIT) + '\n' + \
               "RESTORE_PCT_CHECKPOINTS = " + str(self.RESTORE_PCT_CHECKPOINTS) + '\n' + "RESTORE_TIME_STEP = " + \
               str(self.RESTORE_TIME_STEP) + '\n' + "RESTORE_TIME_MAX = " + str(self.RESTORE_TIME_MAX) + '\n' + \
               "RESTORATION_STREAMS = " + str(self.RESTORATION_STREAMS) + '\n' + "SYSTEM_CLASSES = " + \
               str(self.SYSTEM_CLASSES) + '\n' + "SYSTEM_CLASS = " + str(self.SYSTEM_CLASS) + '\n' + \
               "SYSTEM_SUBCLASS = " + str(self.SYSTEM_SUBCLASS) + '\n' + "COMMODITY_FLOW_TYPES = " + \
               str(self.COMMODITY_FLOW_TYPES) + '\n' + "SYS_CONF_FILE_NAME = " + str(self.SYS_CONF_FILE_NAME) + \
               '\n' + "INPUT_DIR_NAME = " + str(self.INPUT_DIR_NAME) + '\n' + "SYS_CONF_FILE = " + \
               str(self.SYS_CONF_FILE) + '\n' + "OUTPUT_DIR_NAME = " + str(self.OUTPUT_DIR_NAME) + '\n' + \
               "FIT_PE_DATA = " + str(self.FIT_PE_DATA) + '\n' + "FIT_RESTORATION_DATA = " + \
               str(self.FIT_RESTORATION_DATA) + '\n' + "SAVE_VARS_NPY = " + str(self.SAVE_VARS_NPY) + \
               '\n' + "MULTIPROCESS = " + str(self.MULTIPROCESS) + '\n' + "RUN_CONTEXT = " + \
               str(self.RUN_CONTEXT) + '\n' + "ROOT_DIR = " + str(self.ROOT_DIR) + '\n' + "INPUT_PATH = " + \
               str(self.INPUT_PATH) + '\n' + "OUTPUT_PATH = " + str(self.OUTPUT_PATH) + '\n' + \
               "RAW_OUTPUT_DIR = " + str(self.RAW_OUTPUT_DIR)  + '\n' + \
               "INFRASTRUCTURE_LEVEL = " + str(self.INFRASTRUCTURE_LEVEL)


if __name__ == '__main__':
    configuration = Configuration("C:/Users/u12089/Desktop/sifra-dev/test_config.json")
    print(configuration)
