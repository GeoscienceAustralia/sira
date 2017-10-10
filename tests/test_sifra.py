from __future__ import print_function

import unittest
import cPickle
import os
import numpy as np

from sifra.sysresponse import calc_loss_arrays
from sifra.sifraclasses import FacilitySystem, Scenario
from sifra.sysresponse import calc_sys_output
from infrastructure_response import calculate_response, ingest_spreadsheet


class TestSifra(unittest.TestCase):
    def test_calc_loss_arrays(self):
        """
        :return: tests the calc_loss_arrays function, on a non-parallel run.
        """
        # SETUPFILE = 'tests/config_ps_X_test.conf'
        SETUPFILE = 'test_scenario_ps_coal.conf'
        SETUPFILE = os.path.join(os.getcwd(), SETUPFILE)
        print('\nUsing default test setup file')
        scenario = Scenario('/opt/project/tests/test_scenario_ps_coal.conf')
        facility = FacilitySystem('/opt/project/tests/test_scenario_ps_coal.conf')
        print('\n========================= Testing serial run =========================')
        component_resp_df = calc_sys_output(facility, scenario)
        ids_comp_vs_haz, sys_output_dict, component_resp_dict, calculated_output_array, \
            economic_loss_array, output_array_given_recovery \
            = calc_loss_arrays(facility, scenario, component_resp_df, parallel_proc=0)
        test_ids_comp_vs_haz = cPickle.load(open('tests/ids_comp_vs_haz.pickle', 'rb'))
        test_sys_output_dict = cPickle.load(open('tests/sys_output_dict.pickle', 'rb'))
        for k, v in ids_comp_vs_haz.iteritems():
            self.assertEqual(v.shape, (scenario.num_samples, facility.num_elements), msg='size mismatch')

        for k in ids_comp_vs_haz:
            np.testing.assert_array_equal(ids_comp_vs_haz[k], test_ids_comp_vs_haz[k], 'arrays not equal', verbose=True)
        #
        for k in sys_output_dict:
            np.testing.assert_array_equal(sys_output_dict[k], test_sys_output_dict[k], 'arrays not equal', verbose=True)

    def test_if_vs_sysresponse(self):
        config_file = '/opt/project/tests/test_scenario_ps_coal.conf'
        scenario = Scenario(config_file)
        facility = FacilitySystem(config_file)
        component_resp_df = calc_sys_output(facility, scenario)
        sr_result_list = calc_loss_arrays(facility, scenario, component_resp_df, parallel_proc=1)

        scenario = Scenario(config_file)
        infrastructure = ingest_spreadsheet(config_file)  # `IFSystem` object

        if_result_list = calculate_response(scenario, infrastructure)

        # check the differences between the two results
        result_names = ['ds_comp_vs_haz', 'sys_output_dict',
                        'component_resp_dict', 'calculated_output_array',
                        'economic_loss_array', 'output_array_given_recovery']
        for name, if_result, sys_result in zip(result_names,
                                               if_result_list,
                                               sr_result_list):
            print(name)
            if isinstance(if_result, dict):
                # check the keys are the same
                self.assertTrue(set(if_result.keys()) == set(sys_result.keys()))
                # compare the differences between the values
                for key in sorted(sys_result.keys()):
                    level_sys_result = sys_result[key]
                    level_if_result = if_result[key]
                    if not isinstance(level_sys_result, dict):
                        diff = (np.average(level_sys_result - level_if_result))/np.average(level_sys_result)
                    else:
                        level_diff = 0
                        level_base = 0
                        for level_key in sorted(level_sys_result.keys()):
                            level_diff += \
                                np.average(level_sys_result[level_key] - level_if_result[level_key])
                            level_base += np.average(level_sys_result[level_key])
                        diff = level_diff/level_base
                    if diff != 0:
                        print(key, diff)
                        diff=0
            elif isinstance(if_result, np.ndarray):
                if name != 'output_array_given_recovery':
                    print((np.average(if_result, axis=0)-np.average(sys_result, axis=0))/np.average(if_result, axis=0))
                else:
                    print((np.average(if_result, axis=(0, 2)) - np.average(sys_result, axis=(0, 2)))/np.average(if_result, axis=(0, 2)))
            else:
                print('wtf {0}'.format(name))

            # check the length of the data are the same
            self.assertTrue(len(if_result) == len(sys_result))

    def test_if_vs_sys_ident(self):
        config_file = '/opt/project/tests/test_identical_comps.conf'
        scenario = Scenario(config_file)
        facility = FacilitySystem(config_file)
        component_resp_df = calc_sys_output(facility, scenario)
        sr_result_list = calc_loss_arrays(facility, scenario, component_resp_df, parallel_proc=1)

        scenario = Scenario(config_file)
        infrastructure = ingest_spreadsheet(config_file)  # `IFSystem` object

        if_result_list = calculate_response(scenario, infrastructure)

        # check the differences between the two results
        result_names = ['ds_comp_vs_haz', 'sys_output_dict',
                        'component_resp_dict', 'calculated_output_array',
                        'economic_loss_array', 'output_array_given_recovery']
        for name, if_result, sys_result in zip(result_names,
                                               if_result_list,
                                               sr_result_list):
            print(name)
            if isinstance(if_result, dict):
                # check the keys are the same
                self.assertTrue(set(if_result.keys()) == set(sys_result.keys()))
                # compare the differences between the values
                for key in sorted(sys_result.keys()):
                    level_sys_result = sys_result[key]
                    level_if_result = if_result[key]
                    if not isinstance(level_sys_result, dict) and np.average(level_sys_result) > 0:
                        diff = (np.average(level_sys_result - level_if_result))/np.average(level_sys_result)
                    elif isinstance(level_sys_result, dict):
                        level_diff = 0
                        level_base = 0
                        for level_key in sorted(level_sys_result.keys()):
                            level_diff += \
                                np.average(level_sys_result[level_key] - level_if_result[level_key])
                            level_base += np.average(level_sys_result[level_key])
                        diff = level_diff/level_base
                    else:
                        diff = 0
                    if not np.isnan(diff) and diff != 0:
                        print(key, diff)
                        diff=0
            elif isinstance(if_result, np.ndarray):
                if name != 'output_array_given_recovery':
                    if np.all(np.average(if_result, axis=0)> 0):
                        print((np.average(if_result, axis=0)-np.average(sys_result, axis=0))/np.average(if_result, axis=0))
                elif np.all(np.average(if_result, axis=(0, 2)) > 0):
                    print((np.average(if_result, axis=(0, 2)) - np.average(sys_result, axis=(0, 2)))/np.average(if_result, axis=(0, 2)))
            else:
                print('wtf {0}'.format(name))

            # check the length of the data are the same
            self.assertTrue(len(if_result) == len(sys_result))

    def test_calc_loss_arrays_parallel(self):
        """
        :return: tests the calc_loss_arrays function, on a parallel run.
        """
        SETUPFILE = 'tests/config_ps_X_test.conf'
        SETUPFILE = os.path.join(os.getcwd(), SETUPFILE)
        print ('using default test setupfile')
        scenario = Scenario(SETUPFILE)
        facility = FacilitySystem('tests/config_ps_X_test.conf')
        print('\n========================= Testing parallel run =========================')
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

        # print facility.comp_df['cost_fraction']
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
