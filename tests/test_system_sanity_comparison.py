import unittest
import os
import ntpath
import pickle
import numpy as np
import json

from sira.configuration import Configuration
from sira.scenario import Scenario
from sira.modelling.hazard import HazardsContainer
from sira.model_ingest import ingest_model
from sira.simulation import calculate_response
from sira.utilities import get_config_file_path, get_model_file_path

import logging
rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.CRITICAL)


class TestSystemSanity(unittest.TestCase):

    def test_economic_loss_comparison_for_system_sanity(self):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(
            root_dir, "models", "powerstation_coal_A", "input")
        conf_file_path = get_config_file_path(input_dir)
        model_file_path = get_model_file_path(input_dir)

        config = Configuration(conf_file_path, model_file_path)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)

        response_list = calculate_response(hazards, scenario, infrastructure)
        economic_loss_array = response_list[5]

        input_pickle_filename \
            = os.path.join(root_dir, "historical_data",
                           "economic_loss_for_system_sanity_testing.p")

        with open(input_pickle_filename, 'rb') as p:
            historical_economic_loss_array = pickle.load(p, encoding='bytes')

        self.assertTrue(
            np.array_equal(economic_loss_array,
                           historical_economic_loss_array),
            str(len(economic_loss_array))+'\n'+
            str(len(historical_economic_loss_array))
        )

    # # -------------------------------------------------------------------------
    # def test_run_scenario_lower_limit(self):
    #     root_dir = os.path.dirname(os.path.abspath(__file__))
    #     input_dir = os.path.join(
    #         root_dir, "models", "test_structure__limit_lower", "input")
    #     conf_file_path = get_config_file_path(input_dir)
    #     model_file_path = get_model_file_path(input_dir)

    #     config = Configuration(conf_file_path, model_file_path)
    #     scenario = Scenario(config)
    #     hazards = HazardsContainer(config)
    #     infrastructure = ingest_model(config)
    #     response_list = calculate_response(hazards, scenario, infrastructure)

    #     output_node_capacity = 0
    #     with open(config.SYS_CONF_FILE, 'r') as f:
    #         json_infra_model = json.load(f)
    #         output_node_capacity \
    #             = json_infra_model\
    #               ["sysout_setup"]["output_node"]["output_node_capacity"]

    #     self.assertTrue(
    #         int(response_list[4][0][0]) == int(output_node_capacity)
    #     )

    # # -------------------------------------------------------------------------
    # def test_run_scenario_upper_limit(self):
    #     root_dir = os.path.dirname(os.path.abspath(__file__))
    #     input_dir = os.path.join(
    #         root_dir, "models", "test_structure__limit_upper", "input")
    #     conf_file_path = get_config_file_path(input_dir)
    #     model_file_path = get_model_file_path(input_dir)

    #     config = Configuration(conf_file_path, model_file_path)
    #     scenario = Scenario(config)
    #     hazards = HazardsContainer(config)
    #     infrastructure = ingest_model(config)
    #     response_list = calculate_response(hazards, scenario, infrastructure)

    #     self.assertTrue(int(response_list[4][0][0]) == int(0))

    # -------------------------------------------------------------------------
    def test_compare_economic_loss_for_existing_models(self):

        print("\n{}\n>>> Initiating sanity check aganist pre-run models...".\
            format('-'*70))

        root_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(root_dir, 'models')
        conf_file_paths = []
        model_file_paths = []

        for root, dir_names, file_names in os.walk(models_dir):
            for dir_name in dir_names:
                if "tests" in root:
                    if "input" in dir_name:
                        input_dir = os.path.join(root, 'input')
                        conf_file = get_config_file_path(input_dir)
                        model_file = get_model_file_path(input_dir)
                        conf_file_paths.append(conf_file)
                        model_file_paths.append(model_file)
    
        for conf_file_path, model_file_path in zip(conf_file_paths, model_file_paths):
            if os.path.isfile(conf_file_path):
                print("\nMatching results for: " +
                      ntpath.basename(conf_file_path))

                config = Configuration(conf_file_path, model_file_path)
                scenario = Scenario(config)
                hazards = HazardsContainer(config)
                infrastructure = ingest_model(config)

                response_list = calculate_response(
                    hazards, scenario, infrastructure)
                econ_loss_calculated = response_list[5]
                
                stored_data_file = os.path.join(
                    root_dir,
                    'historical_data',
                    "economic_loss_for_"+config.SCENARIO_NAME+'.npy')
                econ_loss_historic = np.load(stored_data_file)

                self.assertTrue(
                    np.array_equal(econ_loss_calculated, econ_loss_historic),
                    conf_file_path
                )
                print("OK")


if __name__ == '__main__':
    unittest.main()
