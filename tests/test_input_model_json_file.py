import os
import unittest
import json
from collections import OrderedDict
from sifra.logger import rootLogger, logging
rootLogger.set_log_level(logging.CRITICAL)


class TestReadingInfrastructureModelJsonFile(unittest.TestCase):

    def setUp(self):

        self.required_headers = ["component_list", "node_conn_df", "sysinp_setup", "sysout_setup"]
        self.required_component_headers = ["component_class", "component_type", "cost_fraction", "node_cluster","node_type","operating_capacity","longitude","latitude","damages_states_constructor"]

        self.project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_json_files = []

        for root, dir_names, file_names in os.walk(self.project_root_dir):
            for file_name in file_names:
                if "models" in root:
                    if ".json" in file_name:
                        self.model_json_files.append(os.path.join(root, file_name))

    def test_folder_structure(self):

        self.assertTrue(os.path.isdir(os.path.join(self.project_root_dir, "models")), "core models folder not found!"+self.project_root_dir +"!")
        self.assertTrue(os.path.isdir(os.path.join(self.project_root_dir, "tests", "models")), "test models folder not found!"+self.project_root_dir +"!")

        self.assertTrue(os.path.isdir(os.path.join(self.project_root_dir, "simulation_setup")), "core simulation setup folder not found!"+self.project_root_dir +"!")
        self.assertTrue(os.path.isdir(os.path.join(self.project_root_dir, "tests",  "simulation_setup")),"test simulation setup folder not found at "+self.project_root_dir +"!")

    def test_model_files_existence(self):
        for model_file in self.model_json_files:
            self.assertTrue(os.path.isfile(model_file), "Model json file not found on path at" + model_file + " !")

    def test_json_structure(self):

        for model_file in self.model_json_files:
            try:
                with open(model_file, 'r') as f:
                    model_json_object = json.load(f, object_pairs_hook=OrderedDict)

                self.assertTrue(model_json_object is not None,"None value object.")
            except ValueError:
                self.assertTrue(False, "Invalid Json format.")

    def test_required_headers_exist(self):
        for model_file in self.model_json_files:
            with open(model_file, 'r') as f:
                model_json_object = json.load(f, object_pairs_hook=OrderedDict)

            self.assertTrue(set(self.required_headers) <= set(model_json_object .keys()), "Required header name not found in " +model_file + " !")

    def test_reading_data_from_component_list(self):
        for model_file in self.model_json_files:

            with open(model_file, 'r') as f:
                model_json_object = json.load(f, object_pairs_hook=OrderedDict)

            component_list = model_json_object['component_list']

            for component in component_list.keys():
                self.assertTrue(set(self.required_component_headers) <= set(component_list[component].keys()), "Required header name in components not found in " + model_file+str(set(component_list[component].keys())) + " !")

    # TODO extend test case over rest of the structure.


if __name__ == '__main__':
    unittest.main()
