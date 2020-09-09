import unittest
from pathlib import Path
from sira.configuration import Configuration
import logging
rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.INFO)


class TestInputConfFile(unittest. TestCase):

    def setUp(self):

        self.sim_dir_name = 'models'
        root_dir = Path(__file__).resolve().parent
        models_dir = Path(root_dir, self.sim_dir_name)
        # ------------------------------------------------------------
        self.conf_file_paths = [
            x for x in models_dir.rglob(f'input/*config*.json')]
        self.model_file_paths = [
            x for x in models_dir.rglob(f'input/*model*.json')]        
        # ------------------------------------------------------------
        self.confs = []
        for conf_file_path, model_file_path in \
            zip(self.conf_file_paths, self.model_file_paths):
            conf = Configuration(conf_file_path, model_file_path)
            self.confs.append(conf)


    def test_does_file_exist(self):
        for conf_file_path in self.conf_file_paths:
            self.assertEqual(conf_file_path.exists(), True)

    def test_datatype_of_SCENARIO_NAME(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SCENARIO_NAME, str or bytes))

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
            self.assertTrue(isinstance(conf.HAZARD_INTENSITY_MEASURE_PARAM, str or bytes))

    def test_datatype_of_INTENSITY_MEASURE_UNIT(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.HAZARD_INTENSITY_MEASURE_UNIT, str or bytes))

    def test_datatype_of_FOCAL_HAZARD_SCENARIOS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.FOCAL_HAZARD_SCENARIOS, list))

    def test_datatype_of_FOCAL_HAZARD_SCENARIO_NAMES(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.FOCAL_HAZARD_SCENARIO_NAMES, list))

    def test_datatype_of_TIME_UNIT(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.TIME_UNIT, str or bytes))

    def test_datatype_of_RESTORE_PCT_CHKPOINTS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.RESTORE_PCT_CHECKPOINTS, int))

    def test_datatype_of_RESTORE_TIME_STEP(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.RESTORE_TIME_STEP, int))

    def test_datatype_of_RESTORE_TIME_MAX(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.RESTORE_TIME_MAX, int))

    def test_datatype_of_RESTORATION_STREAMS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.RESTORATION_STREAMS, list))

    def test_datatype_of_INFRASTRUCTURE_LEVEL(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.INFRASTRUCTURE_LEVEL,
                                       str or bytes))

    def test_datatype_of_SYSTEM_CLASSES(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SYSTEM_CLASSES, list))

    def test_datatype_of_SYSTEM_CLASS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SYSTEM_CLASS, str or bytes))

    def test_datatype_of_SYSTEM_SUBCLASS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SYSTEM_SUBCLASS, str or bytes))

    def test_datatype_of_COMMODITY_FLOW_TYPES(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.COMMODITY_FLOW_TYPES, int))

    def test_datatype_of_COMPONENT_LOCATION_CONF(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.COMPONENT_LOCATION_CONF,
                                       str or bytes))

    def test_datatype_of_SYS_CONF_FILE_NAME(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SYS_CONF_FILE_NAME, str or bytes))

    # def test_datatype_of_FIT_PE_DATA(self):
    #     for conf in self.confs:
    #         self.assertTrue(isinstance(conf.SWITCH_FIT_PE_DATA, bool))

    def test_datatype_of_FIT_RESTORATION_DATA(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SWITCH_FIT_RESTORATION_DATA, bool))

    def test_datatype_of_SWITCH_SAVE_VARS_NPY(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.SWITCH_SAVE_VARS_NPY, bool))

    def test_datatype_of_MULTIPROCESS(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.MULTIPROCESS, int))

    def test_datatype_of_RUN_CONTEXT(self):
        for conf in self.confs:
            self.assertTrue(isinstance(conf.RUN_CONTEXT, int))


if __name__ == "__main__":
    unittest.main()
