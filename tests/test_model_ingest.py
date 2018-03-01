import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import unittest as ut
from sifra.model_ingest import ingest_spreadsheet
from sifra.modelling.structural import jsonify
from sifra.sifraclasses import Scenario
from sifra.modelling.responsemodels import DamageAlgorithm


class TestIngestResponseModel(ut.TestCase):
    def test_ingest_1(self):

        test_conf = './tests/test_scenario_ps_coal.conf'

        if_system, _ = ingest_spreadsheet(test_conf)

        # now jsonify
        json_components = jsonify(if_system)

        self.assertTrue(len(json_components) > 0)

    def test_ingest_2(self):

        test_conf = './tests/test_identical_comps.conf'

        if_system, _ = ingest_spreadsheet(test_conf)

        # now jsonify
        json_components = jsonify(if_system)

        self.assertTrue(len(json_components) > 0)

    def test_algorithm_factory_population(self):
        # Damage and recovery algorithms are now separated
        # so check the ingest is populating the Class

        test_conf = './tests/test_scenario_ps_coal.conf'

        if_system, algorithm_factory = ingest_spreadsheet(test_conf)

        self.assertTrue(len(algorithm_factory.response_algorithms) > 0)
        self.assertTrue(len(algorithm_factory.recovery_algorithms) > 0)

        # we always initialize with earthquakes
        self.assertTrue(len(algorithm_factory.response_algorithms['earthquake']) > 0)
        self.assertTrue(len(algorithm_factory.recovery_algorithms['earthquake']) > 0)

    def test_algorithm_factory(self):
        test_conf = './tests/test_scenario_ps_coal.conf'
        scenario = Scenario(test_conf)
        if_system, algorithm_factory = ingest_spreadsheet(test_conf)

        resp_alg = algorithm_factory.get_response_algorithm('Ash Disposal System', 'earthquake')

        self.assertTrue(resp_alg.__class__.__name__ ==  DamageAlgorithm.__name__)

if __name__ == '__main__':
    ut.main()
