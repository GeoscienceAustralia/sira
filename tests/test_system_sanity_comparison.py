import unittest
from pathlib import Path
import pickle
import json
import logging
import numpy as np

from sira.configuration import Configuration
from sira.scenario import Scenario
from sira.modelling.hazard import HazardsContainer
from sira.model_ingest import ingest_model
from sira.simulation import calculate_response

rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.CRITICAL)


class TestSystemSanity(unittest.TestCase):
    """
    Sets up and runs tests to compare against results from pre-run and checked
    simulations to check that code is producing the expected results.
    """

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent
        self.models_dir = Path(self.root_dir, 'models')
        self.comparison_data_dir = Path(self.root_dir, 'historical_data')

    # -------------------------------------------------------------------------
    def test_economic_loss_comparison_for_system_sanity(self):
        input_dir = Path(
            self.models_dir, "powerstation_coal_A", "input")
        conf_file_path = [d for d in input_dir.glob('*config*.json')].pop()
        model_file_path = [d for d in input_dir.glob('*model*.json')].pop()

        config = Configuration(conf_file_path, model_file_path)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)

        response_list = calculate_response(hazards, scenario, infrastructure)
        economic_loss_array = response_list[5]

        test_file_path = Path(
            self.comparison_data_dir,
            "economic_loss_for_system_sanity_testing.npy")

        historical_economic_loss_array = np.load(test_file_path)
        self.assertTrue(
            np.array_equal(economic_loss_array,
                           historical_economic_loss_array),
            str(len(economic_loss_array))+'\n'+
            str(len(historical_economic_loss_array))
        )

    # -------------------------------------------------------------------------
    def test_run_scenario_lower_limit(self):
        input_dir = Path(
            self.models_dir, "test_structure__limit_lower", "input"
        )
        conf_file_path = [d for d in input_dir.glob('*config*.json')].pop()
        model_file_path = [d for d in input_dir.glob('*model*.json')].pop()

        config = Configuration(conf_file_path, model_file_path)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)
        response_list = calculate_response(hazards, scenario, infrastructure)

        output_node_capacity = 0
        with open(model_file_path, 'r') as mdl:
            json_infra_model = json.load(mdl)
            output_node_capacity \
                = json_infra_model\
                  ["sysout_setup"]["output_node"]["output_node_capacity"]

        self.assertTrue(
            int(response_list[4][0][0]) == int(output_node_capacity)
        )

    # -------------------------------------------------------------------------
    def test_run_scenario_upper_limit(self):
        input_dir = Path(
            self.models_dir, "test_structure__limit_upper", "input"
        )
        conf_file_path = [d for d in input_dir.glob('*config*.json')].pop()
        model_file_path = [d for d in input_dir.glob('*model*.json')].pop()

        config = Configuration(conf_file_path, model_file_path)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)
        response_list = calculate_response(hazards, scenario, infrastructure)

        self.assertTrue(int(response_list[4][0][0]) == int(0))

    # -------------------------------------------------------------------------
    def test_compare_economic_loss_for_existing_models(self):

        print("\n{}\n>>> Initiating sanity check aganist pre-run models...".\
            format('-'*70))

        conf_file_paths = [
            d for d in self.models_dir.rglob('input/*config*.json')]
        model_file_paths = [
            d for d in self.models_dir.rglob('input/*model*.json')]

        for conf_file_path, model_file_path in \
            zip(conf_file_paths, model_file_paths):

            if conf_file_path.is_file():
                print("\nMatching results for: "+Path(conf_file_path).name)

                config = Configuration(conf_file_path, model_file_path)
                scenario = Scenario(config)
                hazards = HazardsContainer(config)
                infrastructure = ingest_model(config)

                response_list = calculate_response(
                    hazards, scenario, infrastructure)
                econ_loss_calculated = response_list[5]

                stored_data_file = Path(
                    self.comparison_data_dir,
                    "economic_loss_for_"+config.SCENARIO_NAME+'.npy')
                econ_loss_historic = np.load(stored_data_file)

                self.assertTrue(
                    np.array_equal(econ_loss_calculated, econ_loss_historic),
                    conf_file_path
                )
                print("OK")


if __name__ == '__main__':
    unittest.main()
