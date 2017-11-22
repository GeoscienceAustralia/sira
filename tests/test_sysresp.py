from __future__ import print_function

import unittest
import cPickle
import numpy as np
import logging

from sifra.sysresponse import calc_loss_arrays, calc_sys_output
from sifra.sifraclasses import FacilitySystem, Scenario

logging.basicConfig(level=logging.INFO)


class TestNewModel(unittest.TestCase):

    def test_calc_loss_arrays_parallel(self):
        """
        :return: tests the calc_loss_arrays function, on a parallel run.
        """
        SETUPFILE = 'config_ps_X_test.conf'
        logging.info ('using default test setupfile')
        scenario = Scenario(SETUPFILE)
        facility = FacilitySystem('config_ps_X_test.conf')
        logging.info('\n========================= Testing parallel run =========================')
        component_resp_df = calc_sys_output(facility, scenario)
        ids_comp_vs_haz, sys_output_dict, component_resp_dict, calculated_output_array, \
            economic_loss_array, output_array_given_recovery \
            = calc_loss_arrays(facility, scenario, component_resp_df, parallel_proc=1)
        test_ids_comp_vs_haz = cPickle.load(open('ids_comp_vs_haz.pickle', 'rb'))
        test_sys_output_dict = cPickle.load(open('sys_output_dict.pickle', 'rb'))
        for k, v in ids_comp_vs_haz.iteritems():
            self.assertEqual(v.shape, (scenario.num_samples, facility.num_elements), msg='size mismatch')

        for k in ids_comp_vs_haz:
            np.testing.assert_array_equal(ids_comp_vs_haz[k], test_ids_comp_vs_haz[k], 'arrays not equal', verbose=True)

        for k in sys_output_dict:
            np.testing.assert_array_equal(sys_output_dict[k], test_sys_output_dict[k], 'arrays not equal', verbose=True)

    def test_extreme_values(self):
        # sys_output_dict # should be full when 0, and 0 when hazard level 10

        scenario = Scenario('config_ps_X_test_extremes.conf')
        facility = FacilitySystem('config_ps_X_test_extremes.conf')
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
