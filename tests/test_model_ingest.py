import matplotlib
matplotlib.use('Agg')

import unittest as ut
from sifra.model_ingest import read_model_from_xlxs
from sifra.modelling.responsemodels import DamageAlgorithm
from sifra.configuration import Configuration


class TestIngestResponseModel(ut.TestCase):



    def test_algorithm_factory_population(self):
        # Damage and recovery algorithms are now separated
        # so check the ingest is populating the Class

        test_conf_file_path = './simulation_setup/test_scenario_ps_coal.json'

        test_conf = Configuration(test_conf_file_path)

        if_system, algorithm_factory = read_model_from_xlxs(test_conf)

        self.assertTrue(len(algorithm_factory.response_algorithms) > 0)
        self.assertTrue(len(algorithm_factory.recovery_algorithms) > 0)

        # we always initialize with earthquakes
        self.assertTrue(len(algorithm_factory.response_algorithms['earthquake']) > 0)
        self.assertTrue(len(algorithm_factory.recovery_algorithms['earthquake']) > 0)

    def test_algorithm_factory(self):

        test_conf_file_path = './simulation_setup/test_scenario_ps_coal.json'

        test_conf = Configuration(test_conf_file_path)

        if_system, algorithm_factory = read_model_from_xlxs(test_conf)

        resp_alg = algorithm_factory.get_response_algorithm('Ash Disposal System', 'earthquake')

        self.assertTrue(resp_alg.__class__.__name__ ==  DamageAlgorithm.__name__)


if __name__ == '__main__':
    ut.main()
