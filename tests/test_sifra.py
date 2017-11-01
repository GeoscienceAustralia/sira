from __future__ import print_function

import unittest
import cPickle
import os
import numpy as np
import logging

from sifra.sysresponse import calc_loss_arrays, calc_sys_output
from sifra.sifraclasses import FacilitySystem, Scenario


class TestSifra(unittest.TestCase):
    def test_calc_loss_arrays(self):
        """
        :return: tests the calc_loss_arrays function, on a non-parallel run.
        """
        # SETUPFILE = 'tests/config_ps_X_test.conf'
        SETUPFILE = 'test_scenario_ps_coal.conf'
        SETUPFILE = os.path.join(os.getcwd(), SETUPFILE)
        logging.info('\nUsing default test setup file')
        scenario = Scenario('/opt/project/tests/test_scenario_ps_coal.conf')
        facility = FacilitySystem('/opt/project/tests/test_scenario_ps_coal.conf')
        logging.info('\n========================= Testing serial run =========================')
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
