import pandas as pd


class Serialise:

    def __init__(self, file_path):


        if file_path.split('.')[1] == 'xlsx':

            self.component_list = pd.read_excel(file_path, sheet_name='component_list', index_col='component_id',
                                                header=0, skiprows=3, skipinitialspace=True)
            self.component_connections = pd.read_excel(file_path, sheet_name='component_connections', index_col=None,
                                                       header=0, skiprows=3, skipinitialspace=True)
            self.output_setup = pd.read_excel(file_path, sheet_name='output_setup', index_col='output_node',
                                              header=0, skiprows=3, skipinitialspace=True)
            self.supply_setup =pd.read_excel(file_path, sheet_name='supply_setup', index_col='input_node',
                                             header=0, skiprows=3, skipinitialspace=True)
            self.comp_type_dmg_algo = pd.read_excel(file_path, sheet_name='comp_type_dmg_algo', index_col=[0, 1],
                                                    header=0, skiprows=3, skipinitialspace=True)
            self.damage_state_def = pd.read_excel(file_path, sheet_name='damage_state_def', index_col=[0, 1],
                                                  header=0, skiprows=3, skipinitialspace=True)
        elif file_path.split('.')[1] == 'json':
            self.component_list = None
            self.component_connections = None
            self.output_setup = None
            self.supply_setup = None
            self.comp_type_dmg_algo = None
            self.damage_state_def = None
        else:
            raise ValueError('Provide Facility definition in xlsx or json format. File provided: ', file_path)

        def get_component_list():
            return self.component_list

        def get_component_connections():
            return self.component_connections

        def get_output_setup():
            return self.output_setup

        def get_supply_setup():
            return self.supply_setup

        def get_comp_type_dmg_algo():
            return self.component_list

        def get_comp_type_dmg_algo():
            return self.component_list
        def get_damage_state_def():

            return self.damage_state_def
def main():

    file_path = "C:\Users\u12089\Desktop\sifra-dev\models\potable_water_treatment_plant\sysconfig_pwtp_400ML.xlsx"
    facility = Serialise(file_path)


if __name__ == '__main__':
    main()
