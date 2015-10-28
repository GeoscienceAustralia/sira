__author__ = 'sudipta'

import unittest
from sira.sira_bk import calc_loss_arrays
import cPickle
import numpy as np

class MyTestCase(unittest.TestCase):

    def test_calc_loss_arrays(self):
        """
        :return: tests the calc_loss_arrays function, which is the main.
        """
        ids_comp_vs_haz, sys_output_dict, component_resp_dict = calc_loss_arrays()
        test_ids_comp_vs_haz = cPickle.load(open('tests/ids_comp_vs_haz.pick', 'rb'))
        test_sys_output_dict = cPickle.load(open('tests/sys_output_dict.pick', 'rb'))
        for k, v in ids_comp_vs_haz.iteritems():
            self.assertEqual(v.shape, (250, 33))

        for k in ids_comp_vs_haz:
            np.testing.assert_array_equal(ids_comp_vs_haz[k], test_ids_comp_vs_haz[k], 'arrays not equal', verbose=True)

        for k in sys_output_dict:
            np.testing.assert_array_equal(sys_output_dict[k], test_sys_output_dict[k], 'arrays not equal', verbose=True)

if __name__ == '__main__':
    unittest.main()
