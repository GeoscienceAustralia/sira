import os
import pandas as pd
import json


def write_json_to_file(json_data, jsonFilePath):

    print(json_data)
    parsed = json.loads(json_data)
    parsed = json.dumps(parsed, indent=4, sort_keys=True)
    obj = open(jsonFilePath, 'w')
    obj.write(parsed)
    obj.close()


# replace " with ' if the occour within brackets
# eg {"key":"["Key":"value"]"} => {"key":"['Key':'value']"}
def standardize_json_string(json_string):

    inside_brackets_flag = False

    standard_json_string = ""
    for i in range(0,len(json_string)):
        if json_string[i] == '[':
            inside_brackets_flag = True
        if json_string[i] == ']':
            inside_brackets_flag = False

        if inside_brackets_flag:
            if json_string[i] == '\"':
                standard_json_string += "\'"
            else:
                standard_json_string += json_string[i]
        else:
            standard_json_string += json_string[i]
    return standard_json_string


def convert_to_json(excel_file_path, parent_folder_name, file_name):

    component_list = pd.read_excel(
        excel_file_path, sheet_name='component_list',
        index_col=0, header=0,
        skiprows=0, skipinitialspace=True)
    component_list = component_list.to_json(orient='index')
    component_list = standardize_json_string(component_list)


    node_conn_df = pd.read_excel(
        excel_file_path, sheet_name='component_connections',
        index_col=None, header=0,
        skiprows=0, skipinitialspace=True)
    node_conn_df = node_conn_df.to_json(orient='index')
    node_conn_df = standardize_json_string(node_conn_df)



    sysinp_setup = pd.read_excel(
        excel_file_path, sheet_name='supply_setup',
        index_col='input_node', header=0,
        skiprows=0, skipinitialspace=True)
    sysinp_setup = sysinp_setup.to_json(orient='index')
    sysinp_setup = standardize_json_string(sysinp_setup)



    sysout_setup = pd.read_excel(
        excel_file_path, sheet_name='output_setup',
        index_col='output_node', header=0,
        skiprows=0, skipinitialspace=True).sort_values(by='priority', ascending=True)
    sysout_setup = sysout_setup.to_json(orient='index')
    sysout_setup = standardize_json_string(sysout_setup)


    fragility_data = pd.read_excel(
        excel_file_path, sheet_name='comp_type_dmg_algo',
        index_col=[0, 1], header=0,
        skiprows=0, skipinitialspace=True)
    fragility_data = fragility_data.to_json(orient='index')
    fragility_data = standardize_json_string(fragility_data)

    damage_state_df = pd.read_excel(
        excel_file_path, sheet_name='damage_state_def',
        index_col=[0, 1], header=0,
        skiprows=0, skipinitialspace=True)
    damage_state_df = damage_state_df.to_json(orient='index')
    damage_state_df = standardize_json_string(damage_state_df)

    main_json_obj='{ "component_list": '+component_list+',"node_conn_df": '+node_conn_df+', "sysinp_setup": '+sysinp_setup+', "sysout_setup": '+sysout_setup+', "fragility_data": '+fragility_data+', "damage_state_df": '+damage_state_df+'}'


    json_file_path = os.path.join(parent_folder_name, file_name + '.json')
    print(json_file_path )
    write_json_to_file(main_json_obj, json_file_path)


def main():

    model_file_paths = []
    for root, dir_names, file_names in os.walk(os.getcwd()):

        for file_name in file_names:
            if file_name.endswith('.xlsx'):
                if 'models' in root:
                    excel_file_path = os.path.join(root, file_name)
                    model_file_paths.append(excel_file_path)
                    print(excel_file_path)
    for excel_file_path in model_file_paths:

        parent_folder_name = os.path.dirname(excel_file_path)
        file_name = os.path.splitext(os.path.basename(excel_file_path))[0]

        if not os.path.exists(os.path.join(parent_folder_name, file_name)):
            os.makedirs(os.path.join(parent_folder_name, file_name))

        convert_to_json(excel_file_path, parent_folder_name, file_name)


if __name__ == "__main__":
    main()
