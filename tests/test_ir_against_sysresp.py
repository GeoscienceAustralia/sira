# from __future__ import print_function
#
# import unittest
# import cPickle
# import os
# import numpy as np
# import logging
#
# # from sifra.sysresponse import calc_loss_arrays, calc_sys_output, compute_output_given_ds
# from sifra.sifraclasses import FacilitySystem, Scenario
# from sifra.infrastructure_response import calculate_response, read_model_from_xlxs
#
# # logging.basicConfig(level=logging.INFO)
# config_file = './tests/test_scenario_ps_coal.conf'
# ident_config_file = './tests/test_scenario_ps_coal.conf'
#
#
# class TestNewModel(unittest.TestCase):
#     def test_infraresp_vs_sysresponse(self):
#         scenario = Scenario(config_file)
#         facility = FacilitySystem(config_file)
#         component_resp_df = calc_sys_output(facility, scenario)
#         sr_result_list = calc_loss_arrays(facility, scenario,
#                                           component_resp_df,
#                                           parallel_proc=scenario.run_parallel_proc)
#
#         infrastructure = read_model_from_xlxs(config_file)  # `IFSystem` object
#         scenario.algorithm_factory = infrastructure[1]
#         if_result_list = calculate_response(scenario, infrastructure[0])
#
#         self.dump_results(if_result_list, sr_result_list)
#
#     def test_if_vs_sys_ident(self):
#         scenario = Scenario(ident_config_file)
#         facility = FacilitySystem(ident_config_file)
#         component_resp_df = calc_sys_output(facility, scenario)
#         sr_result_list = calc_loss_arrays(facility,
#                                           scenario,
#                                           component_resp_df,
#                                           parallel_proc=scenario.run_parallel_proc)
#
#         infrastructure = read_model_from_xlxs(ident_config_file)  # `IFSystem` object
#         scenario.algorithm_factory = infrastructure[1]
#         if_result_list = calculate_response(scenario, infrastructure[0])
#
#         self.dump_results(if_result_list, sr_result_list)
#
#     def test_if_vs_sys_simple(self):
#         simple_config = './tests/test_simple_series_struct.conf'
#         scenario = Scenario(simple_config)
#         facility = FacilitySystem(simple_config)
#         component_resp_df = calc_sys_output(facility, scenario)
#         sr_result_list = calc_loss_arrays(facility,
#                                           scenario,
#                                           component_resp_df,
#                                           parallel_proc=scenario.run_parallel_proc)
#
#         infrastructure = read_model_from_xlxs(simple_config)  # `IFSystem` object
#         scenario.algorithm_factory = infrastructure[1]
#         if_result_list = calculate_response(scenario, infrastructure[0])
#
#         self.dump_results(if_result_list, sr_result_list)
#
#     def dump_results(self, if_result_list, sr_result_list):
#         # check the differences between the two results
#         result_names = ['comp_damage_state', 'sys_output_dict',
#                         'component_resp_dict', 'calculated_output_array']
#         for name, if_result, sys_result in zip(result_names,
#                                                if_result_list,
#                                                sr_result_list):
#             # Dump the means (or means of means)
#             if isinstance(if_result, dict):
#                 logging.info("{} dict".format(name))
#                 # check the keys are the same
#                 self.assertTrue(set(if_result.keys()) == set(sys_result.keys()))
#                 # compare the differences between the values
#                 for key in sorted(sys_result.keys()):
#                     level_sys_result = sys_result[key]
#                     level_if_result = if_result[key]
#                     if not isinstance(level_sys_result, dict):
#                         if np.sum(level_sys_result) != 0:
#                             diff = (np.mean(level_sys_result - level_if_result))/float(np.mean(level_sys_result))
#                         else:
#                             diff = 0
#                     else:
#                         level_diff = 0
#                         level_base = 0
#                         for level_key in sorted(level_sys_result.keys()):
#                             level_diff += \
#                                 level_sys_result[level_key] - level_if_result[level_key]
#                             level_base += level_sys_result[level_key]
#                         if level_base != 0:
#                             diff = np.mean(level_diff)/np.mean(level_base)
#                         else:
#                             diff = 0
#
#                     logging.info("{}".format(diff))
#                     diff = 0
#             elif isinstance(if_result, np.ndarray):
#                 logging.info("{} {}".format(name, sys_result.shape))
#                 if name != 'output_array_given_recovery':
#                     array_mean = (np.mean(sys_result-if_result, axis=0)) / np.mean(sys_result, axis=0)
#                 else:
#                     array_mean = (np.mean(sys_result - if_result, axis=(0, 2))) / np.mean(sys_result, axis=(0, 2))
#
#                 logging.info("{}".format(np.array2string(array_mean,
#                                                          precision=5,
#                                                          separator='\n',
#                                                          suppress_small=True)))
#             else:
#                 logging.info('wtf {0}???'.format(name))
#
#             # check the length of the data are the same
#             self.assertTrue(len(if_result) == len(sys_result))
#
#
# if __name__ == '__main__':
#     unittest.main()
