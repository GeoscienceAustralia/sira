import os
import pandas as pd
import json


def write_json_to_file(pandaObject, jsonFilePath):
    print(pandaObject.keys(), jsonFilePath)
    json_data = pandaObject.to_json(orient='index')
    json_data = standaise_json_string(json_data)
    parsed = json.loads(json_data)
    parsed = json.dumps(parsed, indent=4, sort_keys=True)
    obj = open(jsonFilePath, 'w')
    obj.write(parsed)
    obj.close()

# replace " with ' if the occour within brackets
# eg {"key":"["Key":"value"]"} => {"key":"['Key':'value']"}
def standaise_json_string(json_string):

    insideBracketsFlag=False

    standJsonString=""
    for i in range(0,len(json_string)):
        if json_string[i] == '[':
            insideBracketsFlag = True
        if json_string[i] == ']':
            insideBracketsFlag = False

        if insideBracketsFlag :
            if json_string[i] == '\"':
                standJsonString += "\'"
            else:
                standJsonString += json_string[i]
        else:
            standJsonString += json_string[i]
    return standJsonString

def convert_to_json(excelFilePath, parentFolderName, fileName):
    print(excelFilePath, parentFolderName, fileName)
    component_list = pd.read_excel(
        excelFilePath, sheet_name='component_list',
        index_col=0, header=0,
        skiprows=3, skipinitialspace=True)
    jsonFilePath = os.path.join(parentFolderName, fileName, 'component_list'+'.json')
    write_json_to_file(component_list, jsonFilePath)

    node_conn_df = pd.read_excel(
        excelFilePath, sheet_name='component_connections',
        index_col=None, header=0,
        skiprows=3, skipinitialspace=True)

    jsonFilePath = os.path.join(parentFolderName, fileName, 'component_connections'+'.json')
    write_json_to_file(node_conn_df, jsonFilePath)

    sysinp_setup = pd.read_excel(
        excelFilePath, sheet_name='supply_setup',
        index_col='input_node', header=0,
        skiprows=3, skipinitialspace=True)

    jsonFilePath = os.path.join(parentFolderName, fileName, 'supply_setup'+'.json')
    write_json_to_file(sysinp_setup, jsonFilePath)

    sysout_setup = pd.read_excel(
        excelFilePath, sheet_name='output_setup',
        index_col='output_node', header=0,
        skiprows=3, skipinitialspace=True).sort_values(by='priority', ascending=True)

    jsonFilePath = os.path.join(parentFolderName, fileName, 'output_setup'+'.json')
    write_json_to_file(sysout_setup, jsonFilePath)

    fragility_data = pd.read_excel(
        excelFilePath, sheet_name='comp_type_dmg_algo',
        index_col=[0, 1], header=0,
        skiprows=3, skipinitialspace=True)
    jsonFilePath = os.path.join(parentFolderName, fileName, 'comp_type_dmg_algo'+'.json')
    write_json_to_file(fragility_data, jsonFilePath)

    damage_state_df = pd.read_excel(
        excelFilePath, sheet_name='damage_state_def',
        index_col=[0, 1], header=0,
        skiprows=3, skipinitialspace=True)

    jsonFilePath = os.path.join(parentFolderName, fileName, 'damage_state_def'+'.json')
    write_json_to_file(damage_state_df, jsonFilePath)

def main():

    model_file_paths = []
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file.endswith('.xlsx'):
                if 'models' in root:
                    excelFilePath = os.path.join(root, file)
                    model_file_paths.append(excelFilePath)

    for excelFilePath in model_file_paths:

        parentFolderName = os.path.dirname(excelFilePath)
        fileName = os.path.splitext(os.path.basename(excelFilePath))[0]

        if not os.path.exists(os.path.join(parentFolderName, fileName)):
            os.makedirs(os.path.join(parentFolderName, fileName))

        convert_to_json(excelFilePath, parentFolderName, fileName)


if __name__ == "__main__":
    main()
