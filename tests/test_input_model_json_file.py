import unittest
from pathlib import Path
from sira.tools.utils import relpath
import json
from collections import OrderedDict
import logging
rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.CRITICAL)


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

        self.sim_dir_name = 'models'
        self.test_root_dir = Path(__file__).resolve().parent
        self.test_model_dir = Path(self.test_root_dir, self.sim_dir_name)
        self.model_json_files = [
            x for x in self.test_model_dir.rglob('input/*model*.json')]

    def test_folder_structure(self):
        # mdl_dir_list = [d for d in Path(self.test_model_dir).glob('*')
        #                 if not str(d.name).startswith('.')]
        self.assertTrue(
            Path(self.test_model_dir).is_dir(),
            "Directory for test models not found:\n" + str(self.test_model_dir))

    def test_model_files_existence(self):
        print(f"\n{'-' * 70}\n>>> Initiating check on JSON model files...")
        print("Test model directory: {}".format(self.test_model_dir))
        for model_file in self.model_json_files:
            test_model_relpath = relpath(
                model_file, start=Path(__file__))
            print(f"\nRunning check on model file (json): \n{str(test_model_relpath)}")
            self.assertTrue(
                Path(model_file).exists(),
                "Model json file not found on path at" + str(model_file) + " !")

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
                self.assertEqual(False, "Invalid Json format.")

    def test_required_headers_exist(self):
        for model_file in self.model_json_files:
            with open(model_file, 'r') as fobj:
                model_json_object = json.load(fobj, object_pairs_hook=OrderedDict)
            self.assertTrue(
                set(self.required_headers) <= set(model_json_object .keys()),
                "Required header name not found in \n" + str(model_file)
            )

    def test_reading_data_from_component_list(self):
        for model_file in self.model_json_files:

            with open(model_file, 'r') as f:
                model_json_object = json.load(f, object_pairs_hook=OrderedDict)

            component_list = model_json_object['component_list']
            for component in component_list.keys():
                required_components = set(self.required_component_headers)
                component_keys = set(component_list[component].keys())
                self.assertTrue(
                    required_components <= component_keys,
                    "Required header names in component_list not found: "
                    f"\n  {str(model_file)}\n  str(component_keys))"
                )


if __name__ == '__main__':
    unittest.main()
