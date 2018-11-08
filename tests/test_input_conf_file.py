import unittest
import os
from sifra.logger import rootLogger
from os.path import exists
from sifra.configuration import Configuration
import logging
rootLogger.set_log_level(logging.INFO)


class TestInputConfFile(unittest. TestCase):

    def setUp(self):

        self.conf_file_paths = []

        root = os.path.join(os.getcwd(), 'simulation_setup')
        for root, dir_names, file_names in os.walk(root):
            for file_name in file_names:
                if file_name.endswith('.json'):
                    if 'simulation_setup' in root:
                        conf_file_path = os.path.join(root, file_name)
                        self.conf_file_paths.append(conf_file_path)

        self.confs = []
        for conf_file_path in self.conf_file_paths:
            conf = Configuration(conf_file_path)
            self.confs.append(conf)

    def test_dose_file_exist(self):
        for conf_file_path in self.conf_file_paths:
            self.assertEqual(exists(conf_file_path), True)

    def test_datatype_of_SCENARIO_NAME(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SCENARIO_NAME, unicode or str))

    def test_datatype_of_INTENSITY_MEASURE_MIN(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.INTENSITY_MEASURE_MIN, float))

    def test_datatype_of_INTENSITY_MEASURE_MAX(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.INTENSITY_MEASURE_MAX, float))

    def test_datatype_of_INTENSITY_MEASURE_STEP(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.INTENSITY_MEASURE_STEP, float))

    def test_datatype_of_NUM_SAMPLES(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.NUM_SAMPLES, int))


    def test_datatype_of_INTENSITY_MEASURE_PARAM(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.INTENSITY_MEASURE_PARAM, unicode or str))


    def test_datatype_of_INTENSITY_MEASURE_UNIT(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.INTENSITY_MEASURE_UNIT, unicode or str))


    def test_datatype_of_FOCAL_HAZARD_SCENARIOS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.FOCAL_HAZARD_SCENARIOS, list))

    def test_datatype_of_FOCAL_HAZARD_SCENARIO_NAMES(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.FOCAL_HAZARD_SCENARIO_NAMES, list))

    def test_datatype_of_TIME_UNIT(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.TIME_UNIT, unicode or str))

    def test_datatype_of_RESTORE_PCT_CHKPOINTS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.RESTORE_PCT_CHECKPOINTS, int))

    def test_datatype_of_RESTORE_TIME_STEP(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.RESTORE_TIME_STEP, int))

    def test_datatype_of_RESTORE_TIME_MAX(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.RESTORE_TIME_MAX, float))

    def test_datatype_of_RESTORATION_STREAMS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.RESTORATION_STREAMS, list))

    def test_datatype_of_INFRASTRUCTURE_LEVEL(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.INFRASTRUCTURE_LEVEL,
                                       unicode or str))

    def test_datatype_of_SYSTEM_CLASSES(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SYSTEM_CLASSES, list))

    def test_datatype_of_SYSTEM_CLASS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SYSTEM_CLASS, unicode or str))

    def test_datatype_of_SYSTEM_SUBCLASS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SYSTEM_SUBCLASS, unicode or str))

    def test_datatype_of_COMMODITY_FLOW_TYPES(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.COMMODITY_FLOW_TYPES, int))

    def test_datatype_of_COMPONENT_LOCATION_CONF(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.COMPONENT_LOCATION_CONF,
                                       unicode or str))

    def test_datatype_of_SYS_CONF_FILE_NAME(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SYS_CONF_FILE_NAME, unicode or str))

    def test_datatype_of_INPUT_DIR_NAME(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.INPUT_DIR_NAME, unicode or str))

    def test_datatype_of_OUTPUT_DIR_NAME(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.OUTPUT_DIR_NAME, unicode or str))

    def test_datatype_of_FIT_PE_DATA(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.FIT_PE_DATA, bool))

    def test_datatype_of_FIT_RESTORATION_DATA(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.FIT_RESTORATION_DATA, bool))

    def test_datatype_of_SAVE_VARS_NPY(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SAVE_VARS_NPY, bool))

    def test_datatype_of_MULTIPROCESS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.MULTIPROCESS, int))

    def test_datatype_of_RUN_CONTEXT(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.RUN_CONTEXT, int))


if __name__ == "__main__":
    unittest.main()

