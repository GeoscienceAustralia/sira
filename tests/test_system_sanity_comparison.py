from __future__ import print_function

import unittest
import os
import pickle
import numpy as np
import json
from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.modelling.hazard import HazardsContainer
from sifra.model_ingest import ingest_model
from sifra.simulation import calculate_response
from sifra.logger import rootLogger, logging
rootLogger.set_log_level(logging.CRITICAL)


class TestSystemSanity(unittest.TestCase):

    def test_economic_loss_comparison_for_system_sanity(self):

        root_dir = os.path.dirname(os.path.abspath(__file__))
        conf_file_path = os.path.join(root_dir, "simulation_setup", "test_scenario_sysconfig_pscoal_600MW.json")

        config = Configuration(conf_file_path)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)

        response_list = calculate_response(scenario, infrastructure, hazards)
        economic_loss_array = response_list[4]

        input_pickle_filename = os.path.join(root_dir, 'historical_data', "economic_loss_for_system_sanity_testing.p")
        historical_economic_loss_array = pickle.load(open(input_pickle_filename, 'rb'))

        self.assertTrue(np.array_equal(economic_loss_array, historical_economic_loss_array))

    def test_run_scenario_lower_limit(self):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        conf_file_path = os.path.join(root_dir, "simulation_setup", "test_scenario_test_run_scenario_lower_limit.json")

        config = Configuration(conf_file_path)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)
        response_list = calculate_response(scenario, infrastructure, hazards)

        output_node_capacity = 0
        with open(config.SYS_CONF_FILE, 'r') as f:
            json_infra_model = json.load(f)
            output_node_capacity = json_infra_model["sysout_setup"]["output_node"]["output_node_capacity"]

        self.assertTrue(int(response_list[3][0][0]) == int(output_node_capacity))

    def test_run_scenario_upper_limit(self):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        conf_file_path = os.path.join(root_dir, "simulation_setup", "test_scenario_test_run_scenario_upper_limit.json")
        config = Configuration(conf_file_path)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)
        response_list = calculate_response(scenario, infrastructure, hazards)

        self.assertTrue(int(response_list[3][0][0]) == int(0))


if __name__ == '__main__':
    unittest.main()
