from __future__ import print_function

import unittest
import os
fromrun_scenario sifra.simulation import
from sifra.logger import rootLogger

class TestSifra(unittest.TestCase):

    def setUp(self):

        self.conf_file_paths = []
        parent_folder_name = os.path.dirname(os.getcwd())

        for root, dir_names, file_names in os.walk(parent_folder_name):
            for file_name in file_names:
                if file_name.endswith('.json'):
                    if 'simulation_setup' in root:
                        conf_file_path = os.path.join(root, file_name)
                        self.conf_file_paths.append(conf_file_path)

    def test_run_scenario(self):

        for conf_file_path in self.conf_file_paths:
            run_scenario(conf_file_path)
