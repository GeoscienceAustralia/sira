from __future__ import print_function
import os
import unittest
import logging
from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.model_ingest import ingest_model
from sifra.modelling.system_topology import SystemTopology
from sifra.infrastructure_response import calculate_response, post_processing
from sifra.logger import rootLogger


class TestSifra(unittest.TestCase):

    def setUp(self):
        rootLogger.set_log_level(logging.INFO)
        rootLogger.info('Start')

    def test_run_scenario(self):
        configuration_file_path = os.path.abspath(
            os.path.join("simulation_setup/test_ps.json"))
        config = Configuration(configuration_file_path)
        scenario = Scenario(config)

        infrastructure, algorithm_factory = ingest_model(config)
        scenario.algorithm_factory = algorithm_factory
        sys_topology_view = SystemTopology(infrastructure, scenario)
        sys_topology_view.draw_sys_topology(viewcontext="as-built")
        post_processing_list = calculate_response(scenario, infrastructure)
        post_processing(infrastructure, scenario, post_processing_list)

        rootLogger.info('End')
