# import unittest
# from sifra.configuration import Configuration
#
#
# class TestStringMethods(unittest.TestCase):
#
#     # test input file ends with json
#     # test if there are correct number of variable provided
#     # test type of each variable
#     # test the range of values of each value
#     # there has to be tralling slash at end of each file name
#
#     def test_upper(self):
#         self.assertEqual('foo'.upper(), 'FOO')
#
#     def test_isupper(self):
#         self.assertTrue('FOO'.isupper())
#         self.assertFalse('Foo'.isupper())
#
#     def test_split(self):
#         s = 'hello world'
#         self.assertEqual(s.split(), ['hello', 'world'])
#         # check that s.split fails when the separator is not a string
#         with self.assertRaises(TypeError):
#             s.split(2)
#
#
# if __name__ == '__main__':
#     unittest.main()
#
#     config_file_path = 'C:/Users/u12089/Desktop/sifra-dev/simulation_setup/config.json'
#
#     config = Configuration(config_file_path)
#
#     print("config.SCENARIO_NAME", config.SCENARIO_NAME, type(config.SCENARIO_NAME))
#     print("config.INTENSITY_MEASURE_PARAM", config.INTENSITY_MEASURE_PARAM, type(config.INTENSITY_MEASURE_PARAM))
#     print("config.INTENSITY_MEASURE_UNIT", config.INTENSITY_MEASURE_UNIT, type(config.INTENSITY_MEASURE_UNIT))
#     print("config.SCENARIO_HAZARD_VALUES", config.SCENARIO_HAZARD_VALUES, type(config.SCENARIO_HAZARD_VALUES))
#     print("config.PGA_MIN", config.PGA_MIN, type(config.PGA_MIN))
#     print("config.PGA_MAX", config.PGA_MAX, type(config.PGA_MAX))
#     print("config.PGA_STEP", config.PGA_STEP, type(config.PGA_STEP))
#     print("config.NUM_SAMPLES", config.NUM_SAMPLES, type(config.NUM_SAMPLES))
#     print("config.TIME_UNIT", config.TIME_UNIT, type(config.TIME_UNIT))
#     print("config.RESTORE_PCT_CHECKPOINTS", config.RESTORE_PCT_CHECKPOINTS, type(config.RESTORE_PCT_CHECKPOINTS))
#     print("config.RESTORE_TIME_STEP", config.RESTORE_TIME_STEP, type(config.RESTORE_TIME_STEP))
#     print("config.RESTORE_TIME_MAX", config.RESTORE_TIME_MAX, type(config.RESTORE_TIME_MAX))
#     print("config.RESTORATION_STREAMS", config.RESTORATION_STREAMS, type(config.RESTORATION_STREAMS))
#     print("config.SYSTEM_CLASSES", config.SYSTEM_CLASSES, type(config.SYSTEM_CLASSES))
#     print("config.SYSTEM_CLASS", config.SYSTEM_CLASS, type(config.SYSTEM_CLASS))
#     print("config.SYSTEM_SUBCLASS", config.SYSTEM_SUBCLASS, type(config.SYSTEM_SUBCLASS))
#     print("config.PS_GEN_TECH", config.PS_GEN_TECH, type(config.PS_GEN_TECH))
#     print("config.COMMODITY_FLOW_TYPES", config.COMMODITY_FLOW_TYPES, type(config.COMMODITY_FLOW_TYPES))
#     print("config.SYS_CONF_FILE_NAME", config.SYS_CONF_FILE_NAME, type(config.SYS_CONF_FILE_NAME))
#     print("config.INPUT_DIR_NAME", config.INPUT_DIR_NAME, type(config.INPUT_DIR_NAME))
#     print("config.OUTPUT_DIR_NAME", config.OUTPUT_DIR_NAME, type(config.OUTPUT_DIR_NAME))
#     print("config.FIT_PE_DATA", config.FIT_PE_DATA, type(config.FIT_PE_DATA))
#     print("config.FIT_RESTORATION_DATA", config.FIT_RESTORATION_DATA, type(config.FIT_RESTORATION_DATA))
#     print("config.SAVE_VARS_NPY", config.SAVE_VARS_NPY, type(config.SAVE_VARS_NPY))
#     print("config.MULTIPROCESS", config.MULTIPROCESS, type(config.MULTIPROCESS))
#     print("config.RUN_CONTEXT", config.RUN_CONTEXT, type(config.RUN_CONTEXT))
#     print("config.ROOT_DIR", config.ROOT_DIR, type(config.ROOT_DIR))
#     print("config.INPUT_PATH", config.INPUT_PATH, type(config.INPUT_PATH))
#     print("config.OUTPUT_PATH", config.OUTPUT_PATH, type(config.OUTPUT_PATH))
#     print("config.RAW_OUTPUT_DIR", config.RAW_OUTPUT_DIR, type(config.RAW_OUTPUT_DIR))