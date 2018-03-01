import unittest
import os

from os.path import exists


from sifra.sifraclasses import _readfile

CONF_FILENAME = os.path.join(os.path.dirname(__file__),
                             'C:\\Users\\u12089\\Desktop\\sifra-dev\\tests\\test_simple_series_struct_dep.conf')


class TestInputConfFile(unittest. TestCase):

    def setUp(self):
        self.conf = _readfile(CONF_FILENAME)

    def test_dose_file_exist(self):
        self.assertEqual(exists(CONF_FILENAME), True)

    def test_dose_file_have_all_required_properties(self):
        self.assertEqual(len(self.conf ), 26)

    def test_dose_file_have_all_required_properties(self):

        list_of_required_setting=['SYSTEM_CLASSES','RESTORE_TIME_STEP','INPUT_DIR_NAME',
                                                        'OUTPUT_DIR_NAME','FIT_PE_DATA','NUM_SAMPLES',
                                                        'SYS_CONF_FILE_NAME','PGA_STEP','SYSTEM_SUBCLASS',
                                                        'SYSTEM_CLASS','FIT_RESTORATION_DATA','TIME_UNIT',
                                                        'PGA_MIN','PS_GEN_TECH','INTENSITY_MEASURE_UNIT',
                                                        'RESTORE_TIME_MAX','MULTIPROCESS','SCENARIO_HAZARD_VALUES',
                                                        'RESTORATION_STREAMS','RESTORE_PCT_CHKPOINTS','PGA_MAX',
                                                        'INTENSITY_MEASURE_PARAM','RUN_CONTEXT','COMMODITY_FLOW_TYPES',
                                                        'SCENARIO_NAME','SAVE_VARS_NPY]'].sort()

        list_of_setting_read_from_file = [setting for setting in self.conf].sort()

        self.assertEqual(list_of_required_setting, list_of_setting_read_from_file)

    def test_datatype_of_SCENARIO_NAME(self):
        self.assertEqual(type(self.conf['SCENARIO_NAME']), str)

    def test_datatype_of_PGA_MIN(self):
        self.assertEqual(type(self.conf['PGA_MIN']), float)

    def test_datatype_of_PGA_MAX(self):
        self.assertEqual(type(self.conf['PGA_MAX']), float)

    def test_datatype_of_PGA_STEP(self):
        self.assertEqual(type(self.conf['PGA_STEP']), float)

    def test_datatype_of_NUM_SAMPLES(self):
        self.assertEqual(type(self.conf['NUM_SAMPLES']), int)

    def test_datatype_of_INTENSITY_MEASURE_PARAM(self):
        self.assertEqual(type(self.conf['INTENSITY_MEASURE_PARAM']), str)

    def test_datatype_of_INTENSITY_MEASURE_UNIT(self):
        self.assertEqual(type(self.conf['INTENSITY_MEASURE_UNIT']), str)

    def test_datatype_of_SCENARIO_HAZARD_VALUES(self):
        self.assertEqual(type(self.conf['SCENARIO_HAZARD_VALUES']), list)

    def test_datatype_of_TIME_UNIT(self):
        self.assertEqual(type(self.conf['TIME_UNIT']), str)

    def test_datatype_of_RESTORE_PCT_CHKPOINTS(self):
        self.assertEqual(type(self.conf['RESTORE_PCT_CHKPOINTS']), int)

    def test_datatype_of_RESTORE_TIME_STEP(self):
        self.assertEqual(type(self.conf['RESTORE_TIME_STEP']), int)

    def test_datatype_of_RESTORE_TIME_MAX(self):
        self.assertEqual(type(self.conf['RESTORE_TIME_MAX']), float)

    def test_datatype_of_RESTORATION_STREAMS(self):
        self.assertEqual(type(self.conf['RESTORATION_STREAMS']), list)

    def test_datatype_of_SYSTEM_CLASSES(self):
        self.assertEqual(type(self.conf['SYSTEM_CLASSES']), list)

    def test_datatype_of_SYSTEM_CLASS(self):
        self.assertEqual(type(self.conf['SYSTEM_CLASS']), str)

    def test_datatype_of_SYSTEM_SUBCLASS(self):
        self.assertEqual(type(self.conf['SYSTEM_SUBCLASS']), str)

    def test_datatype_of_PS_GEN_TECH(self):
        self.assertEqual(type(self.conf['PS_GEN_TECH']), str)

    def test_datatype_of_COMMODITY_FLOW_TYPES(self):
        self.assertEqual(type(self.conf['COMMODITY_FLOW_TYPES']), int)

    def test_datatype_of_SYS_CONF_FILE_NAME(self):
        self.assertEqual(type(self.conf['SYS_CONF_FILE_NAME']), str)

    def test_datatype_of_INPUT_DIR_NAME(self):
        self.assertEqual(type(self.conf['INPUT_DIR_NAME']), str)

    def test_datatype_of_OUTPUT_DIR_NAME(self):
        self.assertEqual(type(self.conf['OUTPUT_DIR_NAME']), str)

    def test_datatype_of_FIT_PE_DATA(self):
        self.assertEqual(type(self.conf['FIT_PE_DATA']), bool)

    def test_datatype_of_FIT_RESTORATION_DATA(self):
        self.assertEqual(type(self.conf['FIT_RESTORATION_DATA']), bool)

    def test_datatype_of_SAVE_VARS_NPY(self):
        self.assertEqual(type(self.conf['SAVE_VARS_NPY']), bool)

    def test_datatype_of_MULTIPROCESS(self):
        self.assertEqual(type(self.conf['MULTIPROCESS']), int)

    def test_datatype_of_RUN_CONTEXT(self):
        self.assertEqual(type(self.conf['RUN_CONTEXT']), int)



if __name__ == "__main__":
    unittest.main()

