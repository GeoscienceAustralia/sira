import os
import json
from collections import OrderedDict
import pandas as pd


def write_json_to_file(json_data, json_file_path):

    parsed = json.loads(json_data)
    parsed = json.dumps(parsed, indent=4, sort_keys=True)
    obj = open(json_file_path, 'w')
    obj.write(parsed)
    obj.close()


# replace " with ' if the occur within brackets
# eg {"key":"["Key":"value"]"} => {"key":"['Key':'value']"}
def standardize_json_string(json_string):

    inside_brackets_flag = False

    standard_json_string = ""
    for i in range(0, len(json_string)):
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

        # Note: json object cant have python lists as keys
        # standard_json_string \
        #     = standard_json_string.replace("\"[","[").replace("]\"","]")
    return standard_json_string


def update_json_structure(main_json_obj):

    sysout_setup = main_json_obj["sysout_setup"]
    sysinp_setup = main_json_obj["sysinp_setup"]
    node_conn_df = main_json_obj["node_conn_df"]
    component_list = main_json_obj["component_list"]

    damage_state_df = main_json_obj["damage_state_df"]
    fragility_data = main_json_obj["fragility_data"]

    new_json_structure = OrderedDict()
    new_json_structure["sysout_setup"] = sysout_setup
    new_json_structure["sysinp_setup"] = sysinp_setup
    new_json_structure["node_conn_df"] = node_conn_df
    new_json_structure["component_list"] = OrderedDict()

    for component in component_list:

        new_json_structure["component_list"][component] = OrderedDict()

        new_json_structure["component_list"][component]\
            ["component_class"] \
            = component_list[component]["component_class"]

        new_json_structure["component_list"][component]\
            ["component_type"] \
            = component_list[component]["component_type"]

        new_json_structure["component_list"][component]\
            ["cost_fraction"] \
            = component_list[component]["cost_fraction"]

        new_json_structure["component_list"][component]\
            ["node_cluster"] \
            = component_list[component]["node_cluster"]

        new_json_structure["component_list"][component]\
            ["node_type"] \
            = component_list[component]["node_type"]

        new_json_structure["component_list"][component]\
            ["operating_capacity"] \
            = component_list[component]["op_capacity"]

        new_json_structure["component_list"][component]\
            ["longitude"] = 0

        new_json_structure["component_list"][component]\
            ["latitude"] = 0

        new_json_structure["component_list"][component]\
            ["damages_states_constructor"] = OrderedDict()

        new_json_structure["component_list"][component]\
            ["damages_states_constructor"]["0"] = OrderedDict()

        new_json_structure["component_list"][component]\
            ["damages_states_constructor"]["0"]["damage_state_name"] \
            = "DS0 None"

        new_json_structure["component_list"][component]\
            ["damages_states_constructor"]["0"]["functionality"]\
            = 1.0

        new_json_structure["component_list"][component]\
            ["damages_states_constructor"]["0"]["damage_ratio"]\
            = 0.0

        new_json_structure["component_list"][component]\
            ["damages_states_constructor"]["0"]\
            ["response_function_constructor"]\
            = OrderedDict()

        new_json_structure["component_list"][component]\
            ["damages_states_constructor"]["0"]\
            ["response_function_constructor"]\
            ["function_name"] \
            = "Level0Response"

        new_json_structure["component_list"][component]\
            ["damages_states_constructor"]["0"]\
            ["response_function_constructor"]\
            ["damage_state_definition"]\
            = "Not Available."

        new_json_structure["component_list"][component]\
            ["damages_states_constructor"]["0"]\
            ["recovery_function_constructor"]\
            = OrderedDict()

        new_json_structure["component_list"][component]\
            ["damages_states_constructor"]["0"]\
            ["recovery_function_constructor"]["function_name"]\
            = "Level0Response"

        new_json_structure["component_list"][component]\
            ["damages_states_constructor"]["0"]\
            ["recovery_function_constructor"]["recovery_state_definition"]\
            = "Not Available."

        counter = 0

        for key in fragility_data.keys():
            component_type = eval(key)[1]
            damage_state = eval(key)[2]

            if component_type == component_list[component]["component_type"]:
                damage_states_in_component = [
                    new_json_structure["component_list"][component]\
                    ["damages_states_constructor"][ds]["damage_state_name"]
                    for ds in
                    new_json_structure["component_list"][component]\
                    ["damages_states_constructor"]
                    ]
                if damage_state not in damage_states_in_component:

                    counter = counter + 1

                    new_json_structure["component_list"][component]\
                        ["damages_states_constructor"][counter]\
                        = OrderedDict()
                    new_json_structure["component_list"][component]\
                        ["damages_states_constructor"][counter]\
                        ["damage_state_name"]\
                        = damage_state
                    new_json_structure["component_list"][component]\
                        ["damages_states_constructor"][counter]\
                        ["functionality"]\
                        = fragility_data[key]["functionality"]
                    new_json_structure["component_list"][component]\
                        ["damages_states_constructor"][counter]\
                        ["damage_ratio"]\
                        = fragility_data[key]["damage_ratio"]

                    new_json_structure["component_list"][component]\
                        ["damages_states_constructor"][counter]\
                        ["response_function_constructor"]\
                        = OrderedDict()

                    if fragility_data[key]["is_piecewise"] == "no":
                        # -----------------------------------------------------
                        # <BEGIN> Non-piecewise damage function
                        new_json_structure["component_list"][component]\
                            ["damages_states_constructor"][counter]\
                            ["response_function_constructor"]\
                            ["function_name"]\
                            = fragility_data[key]["damage_function"]

                        new_json_structure["component_list"][component]\
                            ["damages_states_constructor"][counter]\
                            ["response_function_constructor"]\
                            ["median"]\
                            = fragility_data[key]["median"]
                        new_json_structure["component_list"][component]\
                            ["damages_states_constructor"][counter]\
                            ["response_function_constructor"]\
                            ["beta"]\
                            = fragility_data[key]["beta"]
                        new_json_structure["component_list"][component]\
                            ["damages_states_constructor"][counter]\
                            ["response_function_constructor"]\
                            ["location"]\
                            = fragility_data[key]["location"]
                        new_json_structure["component_list"][component]\
                            ["damages_states_constructor"][counter]\
                            ["response_function_constructor"]\
                            ["fragility_source"]\
                            = fragility_data[key]["fragility_source"]
                        new_json_structure["component_list"][component]\
                            ["damages_states_constructor"][counter]\
                            ["response_function_constructor"]\
                            ["minimum"]\
                            = fragility_data[key]["minimum"]

                        if key in damage_state_df.keys():
                            new_json_structure["component_list"][component]\
                                ["damages_states_constructor"][counter]\
                                ["response_function_constructor"]\
                                ["damage_state_definition"]\
                                = damage_state_df[str(eval(key).pop(0))]
                        else:
                            new_json_structure["component_list"][component]\
                                ["damages_states_constructor"][counter]\
                                ["response_function_constructor"]\
                                ["damage_state_definition"]\
                                = "Not Available."
                    # <END> Non-piecewise damage function
                    # ---------------------------------------------------------
                    # <BEGIN> Piecewise defined damage function
                    else:
                        new_json_structure["component_list"][component]\
                            ["damages_states_constructor"][counter]\
                            ["response_function_constructor"]\
                            ["function_name"] = "PiecewiseFunction"
                        new_json_structure["component_list"][component]\
                            ["damages_states_constructor"][counter]\
                            ["response_function_constructor"]\
                            ["piecewise_function_constructor"] = []

                        tempDic = OrderedDict()
                        tempDic["function_name"]\
                            = fragility_data[key]["damage_function"]
                        tempDic["median"]\
                            = fragility_data[key]["median"]
                        tempDic["beta"]\
                            = fragility_data[key]["beta"]
                        tempDic["location"]\
                            = fragility_data[key]["location"]
                        tempDic["fragility_source"]\
                            = fragility_data[key]["fragility_source"]
                        tempDic["minimum"]\
                            = fragility_data[key]["minimum"]

                        if key in damage_state_df.keys():
                            tempDic["damage_state_definition"]\
                                = damage_state_df[str(eval(key).pop(0))]
                        else:
                            tempDic["damage_state_definition"]\
                                = "Not Available."

                        new_json_structure["component_list"][component]\
                            ["damages_states_constructor"][counter]\
                            ["response_function_constructor"]\
                            ["piecewise_function_constructor"].append(tempDic)

                    # <END> Piecewise defined damage function
                    # ---------------------------------------------------------

                    new_json_structure["component_list"][component]\
                        ["damages_states_constructor"][counter]\
                        ["recovery_function_constructor"]\
                        = OrderedDict()
                    new_json_structure["component_list"][component]\
                        ["damages_states_constructor"][counter]\
                        ["recovery_function_constructor"]\
                        ["function_name"]\
                        = fragility_data[key]["recovery_function"]
                    new_json_structure["component_list"][component]\
                        ["damages_states_constructor"][counter]\
                        ["recovery_function_constructor"]\
                        ["norm_mean"]\
                        = fragility_data[key]["recovery_mean"]
                    new_json_structure["component_list"][component]\
                        ["damages_states_constructor"][counter]\
                        ["recovery_function_constructor"]\
                        ["norm_stddev"]\
                        = fragility_data[key]["recovery_std"]
                    new_json_structure["component_list"][component]\
                        ["damages_states_constructor"][counter]\
                        ["recovery_function_constructor"]\
                        ["recovery_state_definition"]\
                        = "Not Available."

                else:
                    tempDic = OrderedDict()

                    tempDic["function_name"]\
                        = fragility_data[key]["damage_function"]
                    tempDic["median"]\
                        = fragility_data[key]["median"]
                    tempDic["beta"]\
                        = fragility_data[key]["beta"]
                    tempDic["location"]\
                        = fragility_data[key]["location"]
                    tempDic["fragility_source"]\
                        = fragility_data[key]["fragility_source"]
                    tempDic["minimum"]\
                        = fragility_data[key]["minimum"]

                    if key in damage_state_df.keys():
                        tempDic["damage_state_definition"]\
                            = damage_state_df[str(eval(key).pop(0))]
                    else:
                        tempDic["damage_state_definition"]\
                            = "Not Available."

                    new_json_structure["component_list"][component]\
                        ["damages_states_constructor"][counter]\
                        ["response_function_constructor"]\
                        ["piecewise_function_constructor"].append(tempDic)

    return new_json_structure


def read_excel_to_json(excel_file_path):

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
        index_col=0, header=0,
        skiprows=0, skipinitialspace=True)
    sysinp_setup = sysinp_setup.to_json(orient='index')
    sysinp_setup = standardize_json_string(sysinp_setup)
    sysout_setup = pd.read_excel(
        excel_file_path, sheet_name='output_setup',
        index_col=0, header=0,
        skiprows=0, skipinitialspace=True).\
        sort_values(by='priority', ascending=True)
    sysout_setup = sysout_setup.to_json(orient='index')
    sysout_setup = standardize_json_string(sysout_setup)

    fragility_data = pd.read_excel(
        excel_file_path, sheet_name='comp_type_dmg_algo',
        index_col=[0, 1, 2], header=0,
        skiprows=0, skipinitialspace=True)
    fragility_data = fragility_data.to_json(orient='index')
    fragility_data = standardize_json_string(fragility_data)

    damage_state_df = pd.read_excel(
        excel_file_path, sheet_name='damage_state_def',
        index_col=[0, 1], header=0,
        skiprows=0, skipinitialspace=True)
    damage_state_df = damage_state_df.to_json(orient='index')
    damage_state_df = standardize_json_string(damage_state_df)

    sys_model_json = '{ "component_list": ' + component_list + ',' \
                    '"node_conn_df": ' + node_conn_df + ',' \
                    '"sysinp_setup": ' + sysinp_setup + ',' \
                    '"sysout_setup": ' + sysout_setup + ',' \
                    '"fragility_data": ' + fragility_data + ',' \
                    '"damage_state_df": ' + damage_state_df + '}'

    return sys_model_json


def main():

    # get list off all the excel files
    model_file_paths = []
    for root, dir_names, file_names in os.walk(os.path.dirname(os.getcwd())):
        for file_name in file_names:
            if file_name.endswith('.xlsx'):
                if 'models' in root:
                    excel_file_path = os.path.join(root, file_name)
                    model_file_paths.append(excel_file_path)

    for excel_file_path in model_file_paths:

        parent_folder_name = os.path.dirname(excel_file_path)
        file_name = os.path.splitext(os.path.basename(excel_file_path))[0]

        json_obj = json.loads(read_excel_to_json(excel_file_path),
                              object_pairs_hook=OrderedDict)
        new_json_structure_obj = update_json_structure(json_obj)

        parsed = json.dumps(new_json_structure_obj, indent=4, sort_keys=True)

        json_file_path = os.path.join(parent_folder_name, file_name + '.json')
        write_json_to_file(parsed, json_file_path)


if __name__ == "__main__":
    main()
