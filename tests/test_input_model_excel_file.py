import os
import unittest as ut
import pandas as pd
import logging
rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.CRITICAL)

class TestReadingExcelFile(ut.TestCase):

    def setUp(self):

        self.project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.model_xlsx_files = []

        for root, dir_names, file_names in os.walk(self.project_root_dir):
            for file_name in file_names:
                if "models" in root:
                    if ".xlsx" in file_name:
                        self.model_xlsx_files.append(os.path.join(root, file_name))

        self.required_sheets = ['component_list',
                                'component_connections',
                                'supply_setup',
                                'output_setup',
                                'comp_type_dmg_algo',
                                'damage_state_def']

    def test_folder_structure(self):

        # self.assertTrue(
        #     os.path.isdir(os.path.join(self.project_root_dir, "models")),
        #     "core models folder not found at " + self.project_root_dir + "!"
        # )

        self.assertTrue(
            os.path.isdir(os.path.join(self.project_root_dir, "tests", "models")),
            "test models folder not found at " + self.project_root_dir + "!"
        )

        # self.assertTrue(
        #     os.path.isdir(os.path.join(self.project_root_dir, "simulation_setup")),
        #     "core simulation setup folder not found at " + self.project_root_dir + "!"
        # )

        self.assertTrue(
            os.path.isdir(os.path.join(self.project_root_dir, "tests",  "simulation_setup")),
            "test simulation setup folder not found at " + self.project_root_dir + "!"
        )


    def test_model_files_exists(self):

        for model_file in self.model_xlsx_files:
            self.assertTrue(
                os.path.isfile(model_file),
                "Model excel file not found on path  at" + model_file + " !"
            )

    def test_required_sheets_exist(self):

        for model_file in self.model_xlsx_files:
            
            rootLogger.info(model_file)
            df = pd.read_excel(model_file, None)
            # check if the sheets is a subset of the required sheets
            self.assertTrue(set(self.required_sheets) <= set(df.keys()), "Required sheet name not found!")

    def test_reading_data_from_component_list(self):

        for model_file in self.model_xlsx_files:
            component_list = pd.read_excel(model_file,
                                           sheet_name='component_list',
                                           header=0,
                                           skiprows=0,
                                           index_col=None,
                                           skipinitialspace=True)

            self.assertTrue(isinstance(len(component_list.index.tolist()), int))

    def test_reading_data_from_component_connections(self):

        required_col_names = ['origin', 'destination', 'link_capacity', 'weight']

        for model_file in self.model_xlsx_files:
            component_connections = pd.read_excel(model_file,
                                                  sheet_name='component_connections',
                                                  header=0,
                                                  skiprows=0,
                                                  index_col=None,
                                                  skipinitialspace=True)

            self.assertTrue(set(required_col_names) <= set(component_connections.columns.values.tolist()),
                            "Required column name not found!")

            for index, connection_values in component_connections.iterrows():

                self.assertTrue(isinstance(connection_values['origin'], unicode or str))
                self.assertTrue(isinstance(float(connection_values['link_capacity']), float))
                self.assertTrue(isinstance(float(connection_values['weight']), float))
                self.assertTrue(isinstance((connection_values['destination']), unicode or str))

    def test_reading_data_from_supply_setup(self):

        # index coloum ingnored : 'input_node'
        required_col_names = ['input_capacity', 'capacity_fraction', 'commodity_type']

        for model_file in self.model_xlsx_files:
            supply_setup = pd.read_excel(model_file,
                                         sheet_name='supply_setup',
                                         index_col=0,
                                         header=0,
                                         skiprows=0,
                                         skipinitialspace=True)

            self.assertTrue(set(required_col_names) <= set(supply_setup.columns.tolist()),
                            "Required column name not found!" +
                            "col expected: "+str(required_col_names) +
                            "col supplied: "+str(supply_setup.columns.values.tolist()) + '\n' +
                            "file name : " + model_file)

            for index, supply_values in supply_setup.iterrows():
                self.assertTrue(isinstance(float(supply_values['input_capacity']), float))
                self.assertTrue(isinstance((float(supply_values['capacity_fraction'])), float))
                self.assertTrue(isinstance((supply_values['commodity_type']), unicode or str))
                self.assertTrue(isinstance((index[0]), unicode or str))

    def test_reading_data_from_output_setup(self):
        # index column ignored : 'output_node'
        required_col_names = ['production_node',
                              'output_node_capacity',
                              'capacity_fraction',
                              'priority']

        for model_file in self.model_xlsx_files:
            output_setup = pd.read_excel(model_file,
                                         sheet_name='output_setup',
                                         header=0,
                                         skiprows=0,
                                         skipinitialspace=True)

            self.assertTrue(set(required_col_names) <= set(output_setup.columns.tolist()),
                            "Required column name not found!" + '\n' +
                            "col expected: " + str(required_col_names) + '\n' +
                            "col supplied: " + str(output_setup.columns.values.tolist()) + '\n' +
                            "file name : " + model_file)

            self.assertTrue(output_setup['output_node_capacity'].sum() > 0)

            for index, output_values in output_setup.iterrows():
                self.assertTrue(isinstance((output_values['production_node']), unicode or str))
                self.assertTrue(isinstance((float(output_values['output_node_capacity'])), float))
                self.assertTrue(isinstance((float(output_values['capacity_fraction'])), float))
                self.assertTrue(isinstance(int(output_values['priority']), int))


    def test_reading_data_from_comp_type_dmg_algo(self):

        # there can be arbitrary number of coloums to supply parameters for specific functions
        required_col_names = ['is_piecewise',
                              'damage_function',
                              'damage_ratio',
                              'functionality',
                              'recovery_function',
                              'recovery_mean',
                              'recovery_std']

        for model_file in self.model_xlsx_files:
            comp_type_dmg_algo = pd.read_excel(model_file,
                                               sheet_name='comp_type_dmg_algo',
                                               index_col=[0, 1, 2],
                                               header=0,
                                               skiprows=0,
                                               skipinitialspace=True)

            self.assertTrue(set(required_col_names) <= set(comp_type_dmg_algo.columns.tolist()),
                            "Required column name not found!" + '\n' +
                            "col expected: " + str(required_col_names) + '\n' +
                            "col supplied: " + str(comp_type_dmg_algo.columns.values.tolist()) + '\n' +
                            "file name : " + model_file)

            # current implemented function
            possible_values_of_damage_function = ["StepFunc",
                                                  "LogNormalCDF",
                                                  "Lognormal",
                                                  "NormalCDF",
                                                  "ConstantFunction",
                                                  "Level0Response",
                                                  "Level0Recovery",
                                                  "PiecewiseFunction",
                                                  "RecoveryFunction"]

            for index, damage_state in comp_type_dmg_algo.iterrows():
                # id
                # self.assertTrue(isinstance((index[0]), int))
                # component_type
                self.assertTrue(isinstance((index[1]), unicode or str))
                # damage_state
                self.assertTrue(isinstance((index[2]), unicode or str))

                self.assertTrue(
                    str(damage_state['damage_function']) in
                        set(possible_values_of_damage_function),
                    "Required damage_function name not found!" + '\n' +
                    "damage_function expected names: " +
                    str(possible_values_of_damage_function) + '\n' +
                    "damage_function name supplied: " +
                    str(damage_state['damage_function']) + '\n' +
                    "file name : " + model_file
                )

                self.assertTrue(isinstance(damage_state['is_piecewise'],  unicode or str))
                self.assertTrue(isinstance(damage_state['damage_function'], unicode or str))
                self.assertTrue(isinstance(float(damage_state['damage_ratio']), float))
                self.assertTrue(isinstance(float(damage_state['functionality']), float))
                self.assertTrue(isinstance(float(damage_state['recovery_mean']), float))
                self.assertTrue(isinstance(float(damage_state['recovery_std']), float))
                # self.assertTrue(isinstance(float(damage_state['recovery_95percentile']), float))

                # TODO damage_state['fragility_source'] not used in code
                # self.assertTrue(isinstance(str(damage_state['fragility_source']), unicode or str), type(damage_state['fragility_source']))

    def test_reading_data_from_damage_state_def(self):
        for model_file in self.model_xlsx_files:
            damage_state_def = pd.read_excel(model_file,
                                             sheet_name='damage_state_def',
                                             index_col=[0, 1],
                                             header=0,
                                             skiprows=0,
                                             skipinitialspace=True)

            for index, damage_def in damage_state_def.iterrows():
                self.assertTrue(isinstance(index[0], unicode or str), type(index[0]))
                self.assertTrue(isinstance(index[1], unicode or str), str(index[1]))

                # TODO excel files not in standard form -- need to standardise
                # self.assertTrue(isinstance(damage_def['damage_state_definition'], unicode or str or numpy.float64), type(damage_def['damage_state_definition'])+model_file)
                # self.assertTrue(isinstance(str(damage_def['fragility_source']), unicode or str), model_file)

if __name__ == "__main__":
    ut.main()