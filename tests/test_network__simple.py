import subprocess
import sys
import unittest
from pathlib import Path

import pytest
from sira.configuration import Configuration
from sira.model_ingest import ingest_model
from sira.modelling.hazard import HazardsContainer
from sira.scenario import Scenario

# from sira.simulation import calculate_response

# Add the source dir to system path
root_dir = Path(__file__).resolve().parent.parent
code_dir = Path(root_dir, 'sira')
sys.path.insert(0, str(code_dir))


class TestModuleRunProcess(unittest.TestCase):
    """
    Sets up expected paths, and runs tests to simulate how an user would
    typically run the application from the terminal.
    """

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.test_dir = Path(self.root_dir, 'tests')
        self.code_dir = Path(self.root_dir, 'sira')
        self.mdls_dir = Path(self.test_dir, 'models')

    @pytest.mark.skip()
    def test_network_config_load(self):
        """
        This module tests:
        loading config file for network models
        """
        model_name = 'basic_distributed_network'
        input_dir = Path(self.mdls_dir, model_name, 'input')

        conf_file_path = [d for d in input_dir.glob('*config*.json')].pop()
        model_file_path = [d for d in input_dir.glob('*model*.json')].pop()

        config = Configuration(conf_file_path, model_file_path)
        scenario = Scenario(config)
        hazards = HazardsContainer(config, model_file_path)
        infrastructure = ingest_model(config)
        # response_list = calculate_response(hazards, scenario, infrastructure)

        process = subprocess.run(
            ['python', str(self.code_dir), '-d', str(input_dir), '-sl'],
            stdout=subprocess.PIPE,
            universal_newlines=True)
        exitstatus = process.returncode
        # An exit status of 0 typically indicates process ran successfully:
        self.assertEqual(exitstatus, 0)


if __name__ == '__main__':
    unittest.main()
