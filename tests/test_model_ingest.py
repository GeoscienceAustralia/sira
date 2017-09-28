import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import unittest as ut
from model_ingest import ingest_spreadsheet
from modelling.structural import jsonify


class TestIngestResponseModel(ut.TestCase):
    def test_ingest_1(self):

        test_conf = '../tests/test_scenario_ps_coal.conf'

        if_system = ingest_spreadsheet(test_conf)

        # now jsonify
        json_components = jsonify(if_system)

        self.assertTrue(len(json_components) > 0)

    def test_ingest_2(self):

        test_conf = '../tests/test_identical_comps.conf'

        if_system = ingest_spreadsheet(test_conf)

        # now jsonify
        json_components = if_system.jsonify()

        self.assertTrue(len(json_components) > 0)


if __name__ == '__main__':
    ut.main()
