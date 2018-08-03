import os
import json
from sifra.logger import rootLogger
import scripts.convert_setup_files_to_json as converter


class Configuration:
    """
    Reads all the simulation configuration constants to be read
    by other classes
    """
    def __init__(self, configuration_file_path,
                 run_mode='impact', output_path=''):
        """
        :param configuration_file_path: path to the config file
        :param run_mode: Default is 'impact' - this runs the full MC simulation
                         If option is 'analysis' then new output folders are
                         not created
        """
        file_ext = \
            os.path.splitext(os.path.basename(configuration_file_path))[1]
        if file_ext != '.json':
            configuration_file_path = converter.convert_to_json(
                configuration_file_path)

        with open(configuration_file_path, 'r') as f:
            config = json.load(f)

        # reading in simulation scenario parameters
        self.SCENARIO_NAME \
            = config['Scenario']['SCENARIO_NAME']
        # print(config['Scenario']['SCENARIO_NAME'])
        self.INTENSITY_MEASURE_PARAM \
            = config['Scenario']['INTENSITY_MEASURE_PARAM']
        self.INTENSITY_MEASURE_UNIT \
            = config['Scenario']['INTENSITY_MEASURE_UNIT']

        # Set of scenario(s) to investigate in detail
        #   - List of strings
        #   - Used in post processing stage
        self.FOCAL_HAZARD_SCENARIOS \
            = config['Scenario']['FOCAL_HAZARD_SCENARIOS']

        self.NUM_SAMPLES = config['Hazard']['NUM_SAMPLES']
        self.HAZARD_TYPE = config['Hazard']['HAZARD_TYPE']

        self.HAZARD_INPUT_METHOD = config['Hazard']['HAZARD_INPUT_METHOD']

        if config['Hazard']['HAZARD_INPUT_METHOD'] is "scenario_file":
            self.INTENSITY_MEASURE_MIN = None
            self.INTENSITY_MEASURE_MAX = None
            self.INTENSITY_MEASURE_STEP = None
            self.SCENARIO_FILE = config['Hazard']['SCENARIO_FILE']
        else:
            self.SCENARIO_FILE = None
            self.INTENSITY_MEASURE_MIN \
                = config['Hazard']['INTENSITY_MEASURE_MIN']
            self.INTENSITY_MEASURE_MAX \
                = config['Hazard']['INTENSITY_MEASURE_MAX']
            self.INTENSITY_MEASURE_STEP \
                = config['Hazard']['INTENSITY_MEASURE_STEP']

        self.TIME_UNIT \
            = config['Restoration']['TIME_UNIT']
        self.RESTORE_PCT_CHECKPOINTS \
            = config['Restoration']['RESTORE_PCT_CHECKPOINTS']
        self.RESTORE_TIME_STEP \
            = config['Restoration']['RESTORE_TIME_STEP']
        self.RESTORE_TIME_MAX \
            = config['Restoration']['RESTORE_TIME_MAX']
        self.RESTORATION_STREAMS \
            = config['Restoration']['RESTORATION_STREAMS']

        self.INFRASTRUCTURE_LEVEL \
            = config['System']['INFRASTRUCTURE_LEVEL']
        self.SYSTEM_CLASSES = config['System']['SYSTEM_CLASSES']
        self.SYSTEM_CLASS = config['System']['SYSTEM_CLASS']
        self.SYSTEM_SUBCLASS = config['System']['SYSTEM_SUBCLASS']
        self.COMMODITY_FLOW_TYPES = config['System']['COMMODITY_FLOW_TYPES']
        self.SYS_CONF_FILE_NAME = config['System']['SYS_CONF_FILE_NAME']
        self.COMPONENT_LOCATION_CONF \
            = config['System']['COMPONENT_LOCATION_CONF']

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.INPUT_DIR_NAME \
            = config['Input']['INPUT_DIR_NAME']
        self.OUTPUT_DIR_NAME \
            = config['Output']['OUTPUT_DIR_NAME'] + self.SCENARIO_NAME

        self.SYS_CONF_FILE = os.path.join(root,
                                          self.INPUT_DIR_NAME,
                                          self.SYS_CONF_FILE_NAME)

        self.FIT_PE_DATA = config['Test']['FIT_PE_DATA']
        self.FIT_RESTORATION_DATA = config['Test']['FIT_RESTORATION_DATA']
        self.SAVE_VARS_NPY = config['Test']['SAVE_VARS_NPY']

        self.MULTIPROCESS = config['Switches']['MULTIPROCESS']
        self.RUN_CONTEXT = config['Switches']['RUN_CONTEXT']

        # reading in setup information

        self.ROOT_DIR \
            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.INPUT_PATH \
            = os.path.join(self.ROOT_DIR, self.INPUT_DIR_NAME)

        self.timestamp = rootLogger.timestamp

        if run_mode=='impact':
            output_dir_timestamped = self.OUTPUT_DIR_NAME + "_" + self.timestamp
            self.OUTPUT_PATH = os.path.join(self.ROOT_DIR, output_dir_timestamped)

            # create output dir: root/SCENARIO_NAME+self.timestamp
            if not os.path.exists(self.OUTPUT_PATH):
                os.makedirs(self.OUTPUT_PATH)

            self.RAW_OUTPUT_DIR = os.path.join(self.OUTPUT_PATH, 'RAW_OUTPUT')
            # create output dir: root/SCENARIO_NAME+self.timestamp/RAW_OUTPUT
            if not os.path.exists(self.RAW_OUTPUT_DIR):
                os.makedirs(self.RAW_OUTPUT_DIR)
        elif run_mode=='analysis':
            self.OUTPUT_PATH = output_path
            self.RAW_OUTPUT_DIR = os.path.join(self.OUTPUT_PATH, 'RAW_OUTPUT')

        self.record_dirs()


    def __str__(self):
        line = "\n----------------------------------------\n"
        excluded = ['SCENARIO_NAME', 'SYSTEM_CLASSES']
        attr = '\n'.join(('{} = {}'.format(item, self.__dict__[item])
                          for item in self.__dict__
                          if item not in excluded))
        note = line + str(self.SCENARIO_NAME) + line + attr + line
        return note


    def record_dirs(self):
        rootLogger.info("System model   : " + self.SYS_CONF_FILE_NAME)
        rootLogger.info("Input dir      : " + self.INPUT_PATH)
        rootLogger.info("Output dir     : " + self.OUTPUT_PATH)
        rootLogger.info("Raw output dir : " + self.RAW_OUTPUT_DIR + "\n")

        self.dir_dict = {}
        self.dir_dict["SYS_CONF_FILE_NAME"] = self.SYS_CONF_FILE_NAME
        self.dir_dict["INPUT_PATH"] = self.INPUT_PATH
        self.dir_dict["OUTPUT_PATH"] = self.OUTPUT_PATH
        self.dir_dict["RAW_OUTPUT_DIR"] = self.RAW_OUTPUT_DIR

        self.file_with_dirs \
            = os.path.splitext(rootLogger.logfile)[0]+"_dirs.json"
        with open(self.file_with_dirs, 'w') as dirfile:
            json.dump(self.dir_dict, dirfile)


if __name__ == '__main__':
    configuration = Configuration("tests/simulation_setup/test_setup.json")
    print(configuration)
