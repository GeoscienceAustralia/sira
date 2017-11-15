from __future__ import print_function

import unittest
import cPickle
import os
import numpy as np
import logging

from sifra.sysresponse import calc_loss_arrays, calc_sys_output, compute_output_given_ds
from sifra.sifraclasses import FacilitySystem, Scenario
from infrastructure_response import calculate_response, ingest_spreadsheet

logging.basicConfig(level=logging.INFO)
ident_config_file = '/opt/project/tests/test_scenario_ps_coal.conf'
config_file = '/opt/project/tests/test_scenario_ps_coal.conf'


class TestNewModel(unittest.TestCase):
    def test_infraresp_vs_sysresponse(self):
        scenario = Scenario(config_file)
        facility = FacilitySystem(config_file)
        component_resp_df = calc_sys_output(facility, scenario)
        sr_result_list = calc_loss_arrays(facility, scenario,
                                          component_resp_df,
                                          parallel_proc=scenario.run_parallel_proc)

        infrastructure = ingest_spreadsheet(config_file)  # `IFSystem` object

        if_result_list = calculate_response(scenario, infrastructure)

        self.dump_results(if_result_list, sr_result_list)

    def test_if_vs_sys_ident(self):
        scenario = Scenario(ident_config_file)
        facility = FacilitySystem(ident_config_file)
        component_resp_df = calc_sys_output(facility, scenario)
        sr_result_list = calc_loss_arrays(facility,
                                          scenario,
                                          component_resp_df,
                                          parallel_proc=scenario.run_parallel_proc)

        infrastructure = ingest_spreadsheet(ident_config_file)  # `IFSystem` object

        if_result_list = calculate_response(scenario, infrastructure)

        self.dump_results(if_result_list, sr_result_list)

    def test_if_vs_sys_simple(self):
        simple_config = 'test_simple_series_struct.conf'
        scenario = Scenario(simple_config)
        facility = FacilitySystem(simple_config)
        component_resp_df = calc_sys_output(facility, scenario)
        sr_result_list = calc_loss_arrays(facility,
                                          scenario,
                                          component_resp_df,
                                          parallel_proc=scenario.run_parallel_proc)

        infrastructure = ingest_spreadsheet(simple_config)  # `IFSystem` object

        if_result_list = calculate_response(scenario, infrastructure)

        self.dump_results(if_result_list, sr_result_list)

    def dump_results(self, if_result_list, sr_result_list):
        # check the differences between the two results
        result_names = ['comp_damage_state', 'sys_output_dict',
                        'component_resp_dict', 'calculated_output_array',
                        'economic_loss_array', 'output_array_given_recovery']
        for name, if_result, sys_result in zip(result_names,
                                               if_result_list,
                                               sr_result_list):
            if isinstance(if_result, dict):
                logging.info("{} dict".format(name))
                # check the keys are the same
                self.assertTrue(set(if_result.keys()) == set(sys_result.keys()))
                # compare the differences between the values
                for key in sorted(sys_result.keys()):
                    level_sys_result = sys_result[key]
                    level_if_result = if_result[key]
                    if not isinstance(level_sys_result, dict):
                        if np.sum(level_sys_result) != 0:
                            diff = (np.mean(level_sys_result - level_if_result))/float(np.mean(level_sys_result))
                        else:
                            diff = 0
                    else:
                        level_diff = 0
                        level_base = 0
                        for level_key in sorted(level_sys_result.keys()):
                            level_diff += \
                                level_sys_result[level_key] - level_if_result[level_key]
                            level_base += level_sys_result[level_key]
                        if level_base != 0:
                            diff = np.mean(level_diff)/np.mean(level_base)
                        else:
                            diff = 0

                    logging.info("{}".format(diff))
                    diff = 0
            elif isinstance(if_result, np.ndarray):
                logging.info("{} {}".format(name, sys_result.shape))
                if name != 'output_array_given_recovery':
                    array_mean = (np.mean(sys_result-if_result, axis=0)) / np.mean(sys_result, axis=0)
                else:
                    array_mean = (np.mean(sys_result - if_result, axis=(0, 2))) / np.mean(sys_result, axis=(0, 2))

                logging.info("{}".format(np.array2string(array_mean,
                                                         precision=5,
                                                         separator='\n',
                                                         suppress_small=True)))
            else:
                logging.info('wtf {0}???'.format(name))

            # check the length of the data are the same
            self.assertTrue(len(if_result) == len(sys_result))

    def test_calc_loss_arrays_parallel(self):
        """
        :return: tests the calc_loss_arrays function, on a parallel run.
        """
        SETUPFILE = 'tests/config_ps_X_test.conf'
        SETUPFILE = os.path.join(os.getcwd(), SETUPFILE)
        logging.info ('using default test setupfile')
        scenario = Scenario(SETUPFILE)
        facility = FacilitySystem('tests/config_ps_X_test.conf')
        logging.info('\n========================= Testing parallel run =========================')
        component_resp_df = calc_sys_output(facility, scenario)
        ids_comp_vs_haz, sys_output_dict, component_resp_dict, calculated_output_array, \
            economic_loss_array, output_array_given_recovery \
            = calc_loss_arrays(facility, scenario, component_resp_df, parallel_proc=1)
        test_ids_comp_vs_haz = cPickle.load(open('tests/ids_comp_vs_haz.pickle', 'rb'))
        test_sys_output_dict = cPickle.load(open('tests/sys_output_dict.pickle', 'rb'))
        for k, v in ids_comp_vs_haz.iteritems():
            self.assertEqual(v.shape, (scenario.num_samples, facility.num_elements), msg='size mismatch')

        for k in ids_comp_vs_haz:
            np.testing.assert_array_equal(ids_comp_vs_haz[k], test_ids_comp_vs_haz[k], 'arrays not equal', verbose=True)

        for k in sys_output_dict:
            np.testing.assert_array_equal(sys_output_dict[k], test_sys_output_dict[k], 'arrays not equal', verbose=True)

    def test_extreme_values(self):
        # sys_output_dict # should be full when 0, and 0 when hazard level 10

        scenario = Scenario('tests/config_ps_X_test_extremes.conf')
        facility = FacilitySystem('tests/config_ps_X_test_extremes.conf')
        component_resp_df = calc_sys_output(facility, scenario)

        ids_comp_vs_haz, sys_output_dict, component_resp_dict, calculated_output_array, \
            economic_loss_array, output_array_given_recovery \
            = calc_loss_arrays(facility, scenario, component_resp_df, parallel_proc=1)

        # logging.info facility.comp_df['cost_fraction']
        for k, v in component_resp_dict.iteritems():
            for kk, vv in v.iteritems():
                component_cost_fraction = facility.comp_df['cost_fraction']['component_id'==kk[0]]
                if k == scenario.hazard_intensity_str[0] and kk[1] == 'func_mean':
                    self.assertEqual(vv, 1.0)
                if k == scenario.hazard_intensity_str[0] and kk[1] == 'loss_mean':
                    self.assertEqual(vv, 0.0)
                if k == scenario.hazard_intensity_str[1] and kk[1] == 'func_mean':
                    if component_cost_fraction > 1e-3:
                        self.assertEqual(vv, 0.0, 'test for {} failed for PGA Level: {}'.format(kk[0], k))
                if k == scenario.hazard_intensity_str[1] and kk[1] == 'loss_mean':
                    if component_cost_fraction > 1e-3:
                        self.assertEqual(vv, 1.0, 'test for {} failed for PGA Level: {}'.format(kk[0], k))


if __name__ == '__main__':
    unittest.main()
