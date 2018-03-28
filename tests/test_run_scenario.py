from __future__ import print_function

import unittest
import json
import pandas as pd
from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.modelling.hazard import HazardsContainer
from sifra.model_ingest import ingest_model
from sifra.simulation import calculate_response
from scripts.convert_excel_files_to_json import standardize_json_string


class TestSifra(unittest.TestCase):

    def test_run_scenario_lower_limit(self):

        conf_file_path = "C:\\Users\\u12089\\Desktop\\sifra-dev\\test_config.json"

        config = Configuration(conf_file_path)
        config.SYS_CONF_FILE = "C:\\Users\\u12089\\Desktop\\sifra-dev\models\\test_structures\\test_run_scenario_lower_limit.json"
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)
        response_list = calculate_response(scenario, infrastructure, hazards)

        model_file_path = "C:\\Users\\u12089\\Desktop\\sifra-dev\models\\test_structures\\test_run_scenario_lower_limit.json"
        output_node_capacity = 0
        with open(model_file_path, 'r') as f:
            json_infra_model = json.load(f)
            output_node_capacity = json_infra_model["sysout_setup"]["output_node"]["output_node_capacity"]

        self.assertTrue(int(response_list[3][0][0]) == int(output_node_capacity))

    def test_run_scenario_upper_limit(self):

        conf_file_path = "C:\\Users\\u12089\\Desktop\\sifra-dev\\test_config.json"
        config = Configuration(conf_file_path)
        config.SYS_CONF_FILE =  "C:\\Users\\u12089\\Desktop\\sifra-dev\models\\test_structures\\test_run_scenario_upper_limit.json"
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)
        response_list = calculate_response(scenario, infrastructure, hazards)

        self.assertTrue(int(response_list[3][0][0]) == int(0))

    def test_economic_losses(self):
        return None
