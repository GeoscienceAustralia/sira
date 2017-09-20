import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import unittest as ut

from model_ingest import ingest_spreadsheet

class TestIngestResponseModel(ut.TestCase):
    def test_damage_states(self):

        test_conf = '../tests/test_scenario_ps_coal.conf'

        if_system = ingest_spreadsheet(test_conf)

        # now jsonify
        json_components = if_system.jsonify_with_metadata()

        self.assertTrue(len(json_components) > 0)


if __name__ == '__main__':
    ut.main()
