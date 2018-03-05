import os
import pandas as pd

test_folder = os.getcwd()
project_folder = os.path.dirname(test_folder)
models_folder = os.path.join(project_folder, 'models')

model_file=''
#find all the files enidning with xlsx in models folder
for path, subdirs, files in os.walk(models_folder):
    for name in files:
        if name.endswith(".xlsx"):
            model_file = os.path.join(path, name)

required_sheets = ['component_list', 'component_connections', 'supply_setup', 'output_setup',
                        'comp_type_dmg_algo']

output_setup = pd.read_excel(model_file, sheet_name='comp_type_dmg_algo', index_col=[0, 1],
                                   header=0, skiprows=3, skipinitialspace=True)

print([c for c in output_setup])

for index, output_values in output_setup.iterrows():
    print(output_values['component_id'])

