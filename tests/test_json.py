from __future__ import print_function

import unittest

# import logging

from sifra.modelling.utils import jsonify, pythonify
from sifra.modelling.infrastructure_system import IFSystem
from sifra.model_ingest import ingest_spreadsheet

# logging.basicConfig(level=logging.INFO)
ident_config_file = './tests/test_scenario_ps_coal.conf'


class TestNewModel(unittest.TestCase):
    def test_pythonify(self):
        config_file = './tests/test_scenario_ps_coal.conf'
        infrastructure, alg_fact = ingest_spreadsheet(config_file)  # `IFSystem` object
        # serialise to json
        json_if = jsonify(infrastructure)
        self.assertTrue(len(json_if) > 0)

        # deserialise to Python
        infra_deserial = pythonify(json_if)
        self.assertTrue(isinstance(infra_deserial, IFSystem))




