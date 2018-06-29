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
        conf_file_path = os.path.join(root_dir, "simulation_setup",
                                      "test_scenario_pscoal_600MW.json")

        config = Configuration(conf_file_path)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)

        response_list = calculate_response(hazards, scenario, infrastructure)
        economic_loss_array = response_list[5]

        input_pickle_filename \
            = os.path.join(root_dir, "historical_data",
                           "economic_loss_for_system_sanity_testing.p")

        historical_economic_loss_array \
            = pickle.load(open(input_pickle_filename, 'rb'))

        self.assertTrue(
            np.array_equal(economic_loss_array,
                           historical_economic_loss_array),
            str(len(economic_loss_array))+'\n'+
            str(len(historical_economic_loss_array))
        )

    def test_run_scenario_lower_limit(self):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        conf_file_path = os.path.join(root_dir,
                                      "simulation_setup",
                                      "test_scenario_lower_limit.json")

        config = Configuration(conf_file_path)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)
        response_list = calculate_response(hazards, scenario, infrastructure)

        output_node_capacity = 0
        with open(config.SYS_CONF_FILE, 'r') as f:
            json_infra_model = json.load(f)
            output_node_capacity \
                = json_infra_model\
                  ["sysout_setup"]["output_node"]["output_node_capacity"]

        self.assertTrue(
            int(response_list[4][0][0]) == int(output_node_capacity)
        )

    def test_run_scenario_upper_limit(self):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        conf_file_path = os.path.join(root_dir,
                                      "simulation_setup",
                                      "test_scenario_upper_limit.json")
        config = Configuration(conf_file_path)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)
        response_list = calculate_response(hazards, scenario, infrastructure)

        self.assertTrue(int(response_list[4][0][0]) == int(0))

    def test_compare_economic_loss_for_existing_models(self):

        conf_file_paths = []
        root_dir = os.path.dirname(os.path.abspath(__file__))
        for root, dir_names, file_names in os.walk(root_dir):
            for file_name in file_names:
                if "tests" in root:
                    if "simulation_setup" in root:
                        # adjusted to keep track of old values in old branches
                        if 'test_scenario_pscoal_600MW' not in file_name:
                            conf_file_paths.append(os.path.join(root,
                                                                file_name))

        for conf_file_path in conf_file_paths:
            if os.path.isfile(conf_file_path):
                config = Configuration(conf_file_path)
                scenario = Scenario(config)
                hazards = HazardsContainer(config)
                infrastructure = ingest_model(config)
                response_list = calculate_response(hazards,
                                                   scenario,
                                                   infrastructure)
                economic_loss_of_model = response_list[5]
                pickel_flename = os.path.join(
                    root_dir,
                    'historical_data',
                    "economic_loss_for_"+config.SCENARIO_NAME + '.p')

                history_economic_loss_for_model \
                    = pickle.load(open(pickel_flename, 'rb'))

                self.assertTrue(
                    np.array_equal(economic_loss_of_model,
                                   history_economic_loss_for_model),
                    conf_file_path
                )


if __name__ == '__main__':
    unittest.main()
