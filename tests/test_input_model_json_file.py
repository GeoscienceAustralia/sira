import os
import unittest
import json
from collections import OrderedDict
import logging
rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.CRITICAL)
from sira.utilities import get_config_file_path, get_model_file_path

class TestReadingInfrastructureModelJsonFile(unittest.TestCase):

    def setUp(self):

        self.required_headers = [
            "component_list",
            "node_conn_df",
            "sysinp_setup",
            "sysout_setup"]

        self.required_component_headers = [
            "component_type",
            "component_class",
            "cost_fraction",
            "node_cluster",
            "node_type",
            "operating_capacity",
            "pos_x",
            "pos_y",
            "damages_states_constructor"]

        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_model_dir = os.path.join(root_dir, 'models')
        self.model_json_files = []

        for root, dir_names, file_names in os.walk(self.test_model_dir):
            for dir_name in dir_names:
                if "tests" in root:
                    if "input" in dir_name:
                        input_dir = os.path.join(root, 'input')
                        model_file = get_model_file_path(input_dir)
                        self.model_json_files.append(model_file)


    def test_folder_structure(self):
        mdl_dir_list = [d for d in os.listdir(self.test_model_dir) 
                if (not d.startswith('.') and os.path.isdir(d))]
        self.assertTrue(os.path.isdir(self.test_model_dir), 
            "Directory for test models not found:\n"+self.test_model_dir)
        for d in mdl_dir_list:
            fullp = os.path.abspath(os.path.join(d, 'input'))
            self.assertTrue(
                os.path.exists(fullp),
                "No `input` directory found in simulation model dir:\n"+fullp
            )

    def test_model_files_existence(self):
        print(f"\n{'-'*70}\n>>> Initiating check on JSON model files...")
        print("Test model directory: {}".format(self.test_model_dir))
        for model_file in self.model_json_files:
            test_model_relpath = os.path.relpath(
                model_file, start=os.path.abspath(__file__))
            print("\nRunning check on model file (json): \n{}".\
                format(test_model_relpath))
            self.assertTrue(
                os.path.isfile(model_file),
                "Model json file not found on path at" + model_file + " !")

    def test_json_structure(self):
        for model_file in self.model_json_files:
            try:
                with open(model_file, 'r') as f:
                    model_json_object = \
                        json.load(f, object_pairs_hook=OrderedDict)
                self.assertTrue(
                    model_json_object is not None,
                    "None value object.")
            except ValueError:
                self.assertTrue(
                    False,
                    "Invalid Json format.")

    def test_required_headers_exist(self):
        for model_file in self.model_json_files:
            with open(model_file, 'r') as f:
                model_json_object = json.load(f, object_pairs_hook=OrderedDict)
            self.assertTrue(
                set(self.required_headers) <= set(model_json_object .keys()),
                "Required header name not found in \n" + model_file)

    def test_reading_data_from_component_list(self):
        for model_file in self.model_json_files:

            with open(model_file, 'r') as f:
                model_json_object = json.load(f, object_pairs_hook=OrderedDict)

            component_list = model_json_object['component_list']
            for component in component_list.keys():
                self.assertTrue(
                    set(self.required_component_headers) <= set(component_list[component].keys()),
                    "Required header names in component_list not found: \n" +
                    model_file + '\n' + str(set(component_list[component].keys()))
                    )

    # TODO extend test case over rest of the structure.


if __name__ == '__main__':
    unittest.main()
