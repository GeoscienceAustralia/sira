import logging
import unittest
import warnings
from pathlib import Path

import pandas as pd
from sira.tools.utils import relpath

rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.CRITICAL)


def read_excel_file(filepath, sheet_name=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        df = pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            header=0,
            skiprows=0,
            index_col=None,
            engine='openpyxl'
        )
    return df


class TestReadingExcelFile(unittest.TestCase):

    def setUp(self):

        self.sim_dir_name = 'models'
        self.test_root_dir = Path(__file__).resolve().parent
        self.test_model_dir = Path(self.test_root_dir, self.sim_dir_name)

        self.xl_engine_args = dict(
            engine='openpyxl', read_only='True', data_only='True')

        self.required_sheets = [
            'system_meta',
            'component_list',
            'component_connections',
            'supply_setup',
            'output_setup',
            'comp_type_dmg_algo',
            'damage_state_def']

        self.model_xlsx_files = []
        self.model_df = {}
        self.component_names_dict = {}
        self.model_xlsx_files = [
            x for x in self.test_model_dir.rglob('input/*model*.xlsx')]

        for model_file in self.model_xlsx_files:
            print(model_file)
            df = read_excel_file(model_file, sheet_name=None)
            self.model_df[model_file] = df
            component_list = df['component_list']
            self.component_names_dict[model_file] = \
                component_list['component_id'].values.tolist()

    def test01_folder_structure(self):
        self.assertTrue(
            self.test_model_dir.exists(),
            f"test models folder not found at {str(self.test_root_dir)}!"
        )

    def test02_model_files_exists(self):
        print("\n" + '-' * 70)
        print("Initiating check on model files in MS Excel format...")
        print("Test model directory: {}".format(self.test_model_dir))
        for model_file in self.model_xlsx_files:
            self.assertTrue(
                model_file.is_file(),
                "Model excel file not found on path  at: " + str(model_file)
            )

    def test03_required_sheets_exist(self):
        for model_file in self.model_xlsx_files:
            rootLogger.info(model_file)
            test_model_relpath = relpath(
                model_file, start=Path(__file__)
            )
            print("\nTest loading model file (xlsx): \n{}".
                  format(test_model_relpath))
            # df = pd.read_excel(model_file, **self.xl_engine_args)
            df = self.model_df[model_file]

            # check if the required worksheets exist in the loaded file
            self.assertTrue(
                set(self.required_sheets) <= set(df.keys()),
                "Required worksheet(s) not found in file:\n  {}\n  {}".format(
                    model_file, list(df.keys()))
            )

    def test04_data_component_list_connections(self):
        print(f"\n{'-'*70}\nChecking component definitions and connections...")
        required_col_names_clist = [
            'component_id',
            'component_type',
            'component_class',
            'cost_fraction',
            'node_type',
            'node_cluster',
            'operating_capacity',
            'pos_x',
            'pos_y'
        ]
        required_col_names_conn = [
            'origin',
            'destination',
            'link_capacity',
            'weight'
        ]

        for model_file in self.model_xlsx_files:
            model = self.model_df[model_file]

            # Test component listing (with attributes)
            component_list = model['component_list']
            component_list = component_list.dropna(how='all')
            component_names = \
                component_list['component_id'].values.tolist()

            self.assertTrue(
                isinstance(len(component_list.index.tolist()), int)
            )
            self.assertTrue(
                set(required_col_names_clist) <= set(component_list.columns.values.tolist()),
                "Required column name(s) not found!"
            )

            # Test component connections
            component_connections = model['component_connections']
            component_connections = component_connections.dropna(how='all')
            col_names_in_file = component_connections.columns.values.tolist()
            self.assertTrue(
                set(required_col_names_conn) == set(col_names_in_file),
                f"Required column name(s) not found in {model_file}\n"
                f"Columns names read: {col_names_in_file}")
            origin_nodes = set(component_connections['origin'].values)
            destin_nodes = set(component_connections['destination'].values)
            all_nodes = set(component_names)

            # Test consistency between component definitions and connections
            print("\nCheck origin nodes are subset of "
                  "primary component list in system...")
            self.assertTrue(origin_nodes.issubset(all_nodes), model_file)
            print("OK")

            print("\nCheck destination nodes are subset of "
                  "primary component list in system...")
            self.assertTrue(destin_nodes.issubset(all_nodes), model_file)
            print("OK")

            for connection_values in component_connections.itertuples():
                self.assertTrue(
                    isinstance(float(connection_values.link_capacity), float)
                )
                self.assertTrue(
                    isinstance(float(connection_values.weight), float)
                )

    def test05_reading_data_from_comp_type_dmg_algo(self):
        required_col_names = [
            'is_piecewise',
            'damage_function',
            'damage_ratio',
            'functionality',
            'recovery_function',
            'recovery_param1',
            'recovery_param2'
        ]
        for model_file in self.model_xlsx_files:
            model = self.model_df[model_file]
            dfa = model['comp_type_dmg_algo']
            comp_type_dmg_algo = \
                dfa.set_index(dfa.columns[0:3].to_list())

            self.assertTrue(
                set(required_col_names) <= set(comp_type_dmg_algo.columns.tolist()),
                "Required column name not found!\n"
                f"col expected: {str(required_col_names)}\n"
                f"col supplied: {str(comp_type_dmg_algo.columns.values.tolist())}\n"
                f"file name : {str(model_file)}"
            )
            # current implemented function
            valid_function_name_list = [
                "stepfunc", "step_func",
                "lognormal", "lognormalcdf", "lognormal_cdf",
                "rayleigh", "rayleighcdf", "rayleigh"
                "normal", "normalcdf", "normal_cdf",
                "constantfunction", "constant_function",
                "level0response",
                "piecewisefunction", "piecewise_function",
                # "level0recovery", "recoveryfunction", "recovery_function"
            ]

            for algo_row in comp_type_dmg_algo.itertuples():
                # component_type
                self.assertTrue(isinstance(algo_row.Index[1], str))
                # damage_state
                self.assertTrue(isinstance(algo_row.Index[2], str))

                fn_err_msg = \
                    "Required damage_function name not found!\n"\
                    f"  values expected: {valid_function_name_list}\n"\
                    f"  values supplied: {algo_row.damage_function}\n"\
                    f"  FILE NAME: {model_file}"

                self.assertTrue(
                    str(algo_row.damage_function).lower() in
                    valid_function_name_list, fn_err_msg
                )

                chk_list = ["yes", "no", "true", "false"]
                self.assertTrue(
                    str(algo_row.is_piecewise).lower() in chk_list)
                self.assertTrue(
                    isinstance(float(algo_row.damage_ratio), float))
                self.assertTrue(
                    isinstance(float(algo_row.functionality), float))
                self.assertTrue(
                    isinstance(float(algo_row.recovery_param1), float))
                self.assertTrue(
                    isinstance(float(algo_row.recovery_param2), float))

    def test_reading_data_from_supply_setup(self):
        required_col_names = [
            'input_capacity',
            'capacity_fraction',
            'commodity_type'
        ]
        for model_file in self.model_xlsx_files:
            model = self.model_df[model_file]
            supply_setup = model['supply_setup']

            # set `input_node` as index
            supply_setup = supply_setup.set_index(supply_setup.columns[0])
            supply_setup = supply_setup.dropna(axis=0, how='all')
            supply_setup = supply_setup.dropna(axis=1, how='all')

            self.assertTrue(
                set(required_col_names) <= set(supply_setup.columns.tolist()),
                "Required column name not found!\n"
                f"col expected: {str(required_col_names)}\n"
                f"col supplied: {str(supply_setup.columns.values.tolist())}\n"
                f"file name   : {str(model_file)}"
            )

            for sv in supply_setup.itertuples():
                self.assertTrue(
                    isinstance(sv.Index, str), sv.Index)
                self.assertTrue(
                    isinstance(float(sv.input_capacity), float))
                self.assertTrue(
                    isinstance(float(sv.capacity_fraction), float))
                self.assertTrue(
                    type(sv.commodity_type) == str)

    def test_reading_data_from_output_setup(self):
        required_col_names = [
            'output_node',
            'production_node',
            'output_node_capacity',
            'capacity_fraction',
            'priority'
        ]
        for model_file in self.model_xlsx_files:
            model = self.model_df[model_file]
            output_setup = model['output_setup']

            output_setup = output_setup.dropna(axis=0, how='all')
            output_setup = output_setup.dropna(axis=1, how='all')

            errmsg = "Required column name not found!\n"\
                f"headers expected: {required_col_names}\n"\
                f"headers supplied: {list(output_setup.columns.values)}\n"\
                f"file name: {model_file}"
            self.assertTrue(
                set(required_col_names) <= set(output_setup.columns.tolist()),
                errmsg
            )

            for row_tuple in output_setup.itertuples():
                self.assertTrue(
                    float(row_tuple.output_node_capacity),
                    'Expected float, got {}.'.
                    format(type(row_tuple.output_node_capacity))
                )
                self.assertTrue(
                    float(row_tuple.capacity_fraction),
                    'Expected float, got {}.'.
                    format(type(row_tuple.capacity_fraction))
                )
                self.assertTrue(
                    int(row_tuple.priority),
                    'Expected int, got {}.'.
                    format(type(row_tuple.priority))
                )

            self.assertTrue(output_setup['output_node_capacity'].sum() > 0)
            production_nodes = output_setup['production_node'].values.tolist()
            self.assertTrue(
                set(production_nodes).issubset(
                    set(self.component_names_dict[model_file])),
                f"Error in output_setup in model file: {model_file}"
            )

    def test_reading_data_from_damage_state_def(self):
        for model_file in self.model_xlsx_files:
            model = self.model_df[model_file]
            damage_state_def = model['damage_state_def']
            damage_state_def = damage_state_def.dropna(axis=1, how='all')

            for row_tuple in damage_state_def.itertuples():
                self.assertTrue(
                    isinstance(row_tuple.component_type, str),
                    "Error in index `{}` file:\n {}".format(
                        row_tuple.component_type, model_file)
                )
                self.assertTrue(
                    isinstance(row_tuple.damage_state, str),
                    "Error in index `{}` file:\n {}".format(
                        row_tuple.damage_state, model_file)
                )


if __name__ == "__main__":
    unittest.main()
