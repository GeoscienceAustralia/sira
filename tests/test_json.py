# from __future__ import print_function
#
# import unittest
#
# from sifra.modelling.utils import jsonify, pythonify
# from sifra.modelling.infrastructure_system import IFSystem
# from sifra.model_ingest import read_model_from_xlxs
# from sifra.configuration import Configuration
# # logging.basicConfig(level=logging.INFO)
# ident_config_file = './simulation_setup/test_scenario_ps_coal.conf'
#
#
# class TestNewModel(unittest.TestCase):
#     def test_pythonify(self):
#         config_file = './simulation_setup/test_scenario_ps_coal.conf'
#         configuration = Configuration(config_file)
#         infrastructure, alg_fact = read_model_from_xlxs(configuration)  # `IFSystem` object
#         # serialise to json
#         json_if = jsonify(infrastructure)
#         self.assertTrue(len(json_if) > 0)
#
#         # deserialise to Python
#         infra_deserial = pythonify(json_if)
#         self.assertTrue(isinstance(infra_deserial, IFSystem))
#
#
#
#
