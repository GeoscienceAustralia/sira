import os
import json
from sifra.logger import rootLogger
import scripts.convert_setup_files_to_json as converter
from utilities import get_file_name, get_dir_path, get_file_extension


class Configuration:
    """
    Reads all the simulation configuration constants to be read
    by other classes
    """
    def __init__(self, config_path, model_path, output_path):
        """
        :param configuration_file_path: path to the config file
        :param run_mode: Default is 'impact' - this runs the full MC simulation
                         If option is 'analysis' then new output folders are
                         not created
        """
        # cater for 3 different types of config files .conf, .json , .ini
        # file_ext = os.path.splitext(os.path.basename(configuration_file_path))[1]
        # if file_ext != '.json':
        #     configuration_file_path = converter.convert_to_json(configuration_file_path)



        with open(config_path, 'r') as f:
            config = json.load(f)

        # reading in simulation scenario parameters
        self.INTENSITY_MEASURE_PARAM = config['SCENARIO_INTENSITY_MEASURE_PARAM']
        self.INTENSITY_MEASURE_UNIT = config['SCENARIO_INTENSITY_MEASURE_UNIT']
        # Set of scenario(s) to investigate in detail
        #   - List of strings
        #   - Used in post processing stage
        self.FOCAL_HAZARD_SCENARIO_NAMES = config['SCENARIO_FOCAL_HAZARD_SCENARIO_NAMES']
        self.FOCAL_HAZARD_SCENARIOS = config['SCENARIO_FOCAL_HAZARD_SCENARIOS']

        self.HAZARD_INPUT_METHOD = config['HAZARD_INPUT_METHOD']
        self.HAZARD_TYPE = config['HAZARD_TYPE']
        self.NUM_SAMPLES = config['HAZARD_NUM_SAMPLES']
        if config['HAZARD_INPUT_METHOD'] is "scenario_file":
            self.INTENSITY_MEASURE_MIN = None
            self.INTENSITY_MEASURE_MAX = None
            self.INTENSITY_MEASURE_STEP = None
            self.SCENARIO_FILE = config['SCENARIO_FILE']
        else:
            self.SCENARIO_FILE = None
            self.INTENSITY_MEASURE_MIN = config['HAZARD_INTENSITY_MEASURE_MIN']
            self.INTENSITY_MEASURE_MAX = config['HAZARD_INTENSITY_MEASURE_MAX']
            self.INTENSITY_MEASURE_STEP = config['HAZARD_INTENSITY_MEASURE_STEP']

        self.TIME_UNIT = config['RESTORATION_TIME_UNIT']
        self.RESTORE_PCT_CHECKPOINTS = config['RESTORATION_PCT_CHECKPOINTS']
        self.RESTORE_TIME_STEP = config['RESTORATION_TIME_STEP']
        self.RESTORE_TIME_MAX = config['RESTORATION_TIME_MAX']
        self.RESTORATION_STREAMS = config['RESTORATION_STREAMS']

        self.INFRASTRUCTURE_LEVEL = config['SYSTEM_INFRASTRUCTURE_LEVEL']
        self.SYSTEM_CLASSES = config['SYSTEM_CLASSES']
        self.SYSTEM_CLASS = config['SYSTEM_CLASS']
        self.SYSTEM_SUBCLASS = config['SYSTEM_SUBCLASS']
        self.COMMODITY_FLOW_TYPES = config['SYSTEM_COMMODITY_FLOW_TYPES']
        self.COMPONENT_LOCATION_CONF = config['SYSTEM_COMPONENT_LOCATION_CONF']

        self.FIT_PE_DATA = config['TEST_FIT_PE_DATA']
        self.FIT_RESTORATION_DATA = config['TEST_FIT_RESTORATION_DATA']
        self.SAVE_VARS_NPY = config['TEST_SAVE_VARS_NPY']

        self.MULTIPROCESS = config['SWITCH_MULTIPROCESS']
        self.RUN_CONTEXT = config['SWITCH_RUN_CONTEXT']

        self.SCENARIO_NAME = config['SCENARIO_NAME']

        # self.SYS_CONF_FILE_NAME = get_file_name(model_path)
        # self.INPUT_DIR_NAME = get_dir_path(model_path)
        # self.SYS_CONF_FILE = model_path

        # reading in setup information

        # self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # self.INPUT_PATH = os.path.join(self.ROOT_DIR, self.INPUT_DIR_NAME)
        self.INPUT_MODEL_PATH = model_path
        # self.timestamp = rootLogger.timestamp

        # if run_mode == 'impact':
        # output_dir_timestamped = self.OUTPUT_DIR_NAME + "_" + self.timestamp
        self.OUTPUT_PATH = output_path

        # create output dir: root/SCENARIO_NAME+self.timestamp
        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH)

        self.RAW_OUTPUT_DIR = os.path.join(self.OUTPUT_PATH, 'RAW_OUTPUT')
        # create output dir: root/SCENARIO_NAME+self.timestamp/RAW_OUTPUT
        if not os.path.exists(self.RAW_OUTPUT_DIR):
            os.makedirs(self.RAW_OUTPUT_DIR)

        # elif run_mode == 'analysis':
        #     self.OUTPUT_PATH = output_path
        #     self.RAW_OUTPUT_DIR = os.path.join(self.OUTPUT_PATH, 'RAW_OUTPUT')

        self.record_dirs()

    # def __str__(self):
    #     line = "\n----------------------------------------\n"
    #     excluded = ['SCENARIO_NAME', 'SYSTEM_CLASSES']
    #     attr = '\n'.join(('{} = {}'.format(item, self.__dict__[item])
    #                       for item in self.__dict__
    #                       if item not in excluded))
    #     note = line + str(self.SCENARIO_NAME) + line + attr + line
    #     return note

    def record_dirs(self):
        rootLogger.info("System model   : " + get_file_name(self.INPUT_MODEL_PATH))
        rootLogger.info("Input dir      : " + get_dir_path(self.INPUT_MODEL_PATH))
        rootLogger.info("Output dir     : " + self.OUTPUT_PATH)
        rootLogger.info("Raw output dir : " + self.RAW_OUTPUT_DIR + "\n")

        self.dir_dict = {}
        self.dir_dict["SYS_CONF_FILE_NAME"] = get_file_name(self.INPUT_MODEL_PATH)
        self.dir_dict["INPUT_PATH"] = get_dir_path(self.INPUT_MODEL_PATH)
        self.dir_dict["OUTPUT_PATH"] = self.OUTPUT_PATH
        self.dir_dict["RAW_OUTPUT_DIR"] = self.RAW_OUTPUT_DIR

        # self.file_with_dirs = os.path.splitext(rootLogger.logfile)[0]+"_dirs.json"

        self.file_with_dirs = "output/file_paths.json"
        with open(self.file_with_dirs, 'w') as dirfile:
            json.dump(self.dir_dict, dirfile)


# if __name__ == '__main__':
#     configuration = Configuration("tests/simulation_setup/test_setup.json")
#     print(configuration)
