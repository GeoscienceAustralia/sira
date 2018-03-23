import os
import json
from collections import OrderedDict

python_file_paths = []
for root, dir_names, file_names in os.walk(os.path.dirname(os.getcwd())):
    for file_name in file_names:
        if file_name.endswith('.json'):
            if 'models' in root:
                if 'sysconfig_simple_parallel' not in file_name:
                    if 'test_changes' in file_name:
                        python_file_paths.append(os.path.join(root, file_name))



for python_file_path in python_file_paths:
    print(python_file_path )
    with open(python_file_path, 'r') as input_file:
        data = json.load(input_file, object_pairs_hook=OrderedDict)
        new_structure = {}

        sysinp_setup = data["sysinp_setup"]
        damage_state_df = data["damage_state_df"]
        fragility_data = data["fragility_data"]
        sysout_setup = data["sysout_setup"]
        node_conn_df = data["node_conn_df"]
        component_list = data["component_list"]

        new_structure["sysout_setup"] = sysout_setup
        new_structure["sysinp_setup"] = sysinp_setup
        new_structure["node_conn_df"] = node_conn_df
        new_structure["component_list"] = {}

        for component in component_list:

            new_structure["component_list"][component ] = {}
            new_structure["component_list"][component]["component_class"] = component_list[component]["component_class"]
            new_structure["component_list"][component]["component_type"] = component_list[component]["component_type"]
            new_structure["component_list"][component]["cost_fraction"] = component_list[component]["cost_fraction"]
            new_structure["component_list"][component]["node_cluster"] = component_list[component]["node_cluster"]
            new_structure["component_list"][component]["node_type"] = component_list[component]["node_type"]
            new_structure["component_list"][component]["operating_capacity"] = component_list[component]["op_capacity"]
            new_structure["component_list"][component]["longitude"] = 0
            new_structure["component_list"][component]["latitude"] = 0
            new_structure["component_list"][component]["damages_states_constructor"] = {}

            parsed = json.dumps(new_structure["component_list"][component], indent=4, sort_keys=True)

            counter = -1
            for key in fragility_data.keys():
                if eval(key)[0] == component_list[component]["component_type"]:
                    counter = counter + 1

                    new_structure["component_list"][component]["damages_states_constructor"][counter] = {}
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["damage_state_name"] = eval(key)[1]
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["functionality"] = fragility_data[key]["functionality"]
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["damage_ratio"] = fragility_data[key]["damage_ratio"]

                    new_structure["component_list"][component]["damages_states_constructor"][counter]["response_function_constructor"] = {}
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["response_function_constructor"]["function_name"] = fragility_data[key]["damage_function"]
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["response_function_constructor"]["beta"] = fragility_data[key]["beta"]
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["response_function_constructor"]["median"] = fragility_data[key]["median"]
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["response_function_constructor"]["fragility_source"] = fragility_data[key]["fragility_source"]
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["response_function_constructor"]["minimum"] = fragility_data[key]["minimum"]

                    if key in damage_state_df.keys():
                        new_structure["component_list"][component]["damages_states_constructor"][counter]["response_function_constructor"]["damage_state_definition"] = damage_state_df[key]
                    else:
                        new_structure["component_list"][component]["damages_states_constructor"][counter]["response_function_constructor"]["damage_state_definition"] = "Not Available."

                    new_structure["component_list"][component]["damages_states_constructor"][counter]["recovery_function_constructor"] = {}

                    new_structure["component_list"][component]["damages_states_constructor"][counter]["recovery_function_constructor"]["recovery_95percentile"] = fragility_data[key]["recovery_95percentile"]
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["recovery_function_constructor"]["recovery_mean"] = fragility_data[key]["recovery_mean"]
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["recovery_function_constructor"]["recovery_std"] = fragility_data[key]["recovery_std"]
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["recovery_function_constructor"]["function_name"] = "RecoveryFunction"
                    new_structure["component_list"][component]["damages_states_constructor"][counter]["recovery_function_constructor"]["recovery_state_definition"] = "Not Available."

            parsed = json.dumps(new_structure["component_list"], indent=4, sort_keys=True)
            print(parsed)


            break
            #     with open(python_file_path, "w") as output_file:
#         output_file.write(data)


# sysout_setup
# sysinp_setup
# node_conn_df

# component_list