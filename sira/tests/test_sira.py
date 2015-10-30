__author__ = 'sudipta'

import unittest
import cPickle
import os
import numpy as np

from sira.sira_bk import calc_loss_arrays as calc_loss_arrays_bk
from sira.sira import calc_loss_arrays
from sira.siraclasses import ScenarioDataGetter, Facility, Scenario
from sira.sira import convert_df_to_dict
from sira.sira import simulation_parameters, power_calc


class TestSiraBk(unittest.TestCase):

    def test_calc_loss_arrays(self):
        sc = ScenarioDataGetter('tests/config_ps_X_test.conf')
        facility = Facility('tests/config_ps_X_test.conf')
        """
        :return: tests the calc_loss_arrays function, which is the main.
        """
        print '======================Testing serial run ============================='
        ids_comp_vs_haz, sys_output_dict, component_resp_dict = calc_loss_arrays_bk(parallel_or_serial=0)
        test_ids_comp_vs_haz = cPickle.load(open('tests/ids_comp_vs_haz.pick', 'rb'))
        test_sys_output_dict = cPickle.load(open('tests/sys_output_dict.pick', 'rb'))
        for k, v in ids_comp_vs_haz.iteritems():
            self.assertEqual(v.shape, (sc.num_samples, facility.num_elements))

        for k in ids_comp_vs_haz:
            np.testing.assert_array_equal(ids_comp_vs_haz[k], test_ids_comp_vs_haz[k], 'arrays not equal', verbose=True)

        for k in sys_output_dict:
            np.testing.assert_array_equal(sys_output_dict[k], test_sys_output_dict[k], 'arrays not equal', verbose=True)

    def test_calc_loss_arrays_parallel(self):
        """
        :return: tests the calc_loss_arrays function, which is the main.
        """
        sc = ScenarioDataGetter('tests/config_ps_X_test.conf')
        facility = Facility('tests/config_ps_X_test.conf')
        print '\n\n======================Testing parallel run ============================='
        ids_comp_vs_haz, sys_output_dict, component_resp_dict = calc_loss_arrays_bk(parallel_or_serial=1)
        test_ids_comp_vs_haz = cPickle.load(open('tests/ids_comp_vs_haz.pick', 'rb'))
        test_sys_output_dict = cPickle.load(open('tests/sys_output_dict.pick', 'rb'))
        for k, v in ids_comp_vs_haz.iteritems():
            self.assertEqual(v.shape, (sc.num_samples, facility.num_elements))

        for k in ids_comp_vs_haz:
            np.testing.assert_array_equal(ids_comp_vs_haz[k], test_ids_comp_vs_haz[k], 'arrays not equal', verbose=True)

        for k in sys_output_dict:
            np.testing.assert_array_equal(sys_output_dict[k], test_sys_output_dict[k], 'arrays not equal', verbose=True)


class TestSira(unittest.TestCase):
    SETUPFILE = 'tests/config_ps_X_test.conf'

    sc = Scenario(SETUPFILE)
    fc = Facility(SETUPFILE)

    # Define input files, output location, scenario inputs
    INPUT_PATH = os.path.join(os.getcwd(), sc.input_dir_name)
    SYS_CONFIG_FILE = os.path.join(INPUT_PATH, fc.sys_config_file_name)

    if not os.path.exists(sc.output_dir_name):
        os.makedirs(sc.output_dir_name)

    # cp_types_in_system, cp_types_in_db = check_types_with_db(fc)
    # costed_comptypes, comps_costed = list_of_components_for_cost_calculation(fc.comp_df, fc.cp_types_in_system)

    nominal_production = fc.sysout_setup['Capacity'].sum()
    hazard_transfer_label = sc.hazard_transfer_param + ' (' + sc.hazard_transfer_unit+ ')'

    cpdict, output_dict, input_dict, nodes_by_commoditytype = convert_df_to_dict(fc)

    restoration_time_range, dmg_states, restoration_chkpoints, restoration_pct_steps, hazard_intensity_vals, \
           num_hazard_pts, num_time_steps = simulation_parameters(fc, sc)

    nodes_all = sorted(fc.comp_df.index)
    no_elements = len(nodes_all)

    component_resp_df = power_calc(fc, sc)

    ids_comp_vs_haz, sys_output_dict, component_resp_dict = calc_loss_arrays(fc, sc,
                                                    component_resp_df, parallel_or_serial=sc.parallel_or_serial)

    # def test_calc_loss_arrays(self):
    #     sc = ScenarioDataGetter('tests/config_ps_X_test.conf')
    #     facility = Facility('tests/config_ps_X_test.conf')
    #     """
    #     :return: tests the calc_loss_arrays function, which is the main.
    #     """
    #     print '======================Testing serial run ============================='
    #
    #     ids_comp_vs_haz, sys_output_dict, component_resp_dict = calc_loss_arrays(parallel_or_serial=0)
    #     test_ids_comp_vs_haz = cPickle.load(open('tests/ids_comp_vs_haz.pick', 'rb'))
    #     test_sys_output_dict = cPickle.load(open('tests/sys_output_dict.pick', 'rb'))
    #     for k, v in ids_comp_vs_haz.iteritems():
    #         self.assertEqual(v.shape, (sc.num_samples, facility.num_elements))
    #
    #     for k in ids_comp_vs_haz:
    #         np.testing.assert_array_equal(ids_comp_vs_haz[k], test_ids_comp_vs_haz[k], 'arrays not equal', verbose=True)
    #
    #     for k in sys_output_dict:
    #         np.testing.assert_array_equal(sys_output_dict[k], test_sys_output_dict[k], 'arrays not equal', verbose=True)

if __name__ == '__main__':
    unittest.main()
