import os
import json
# given model file validate it based on rules


def config_file_valid(config_file):
    return True


def model_file_valid(model_file):
    return True


SIFRA_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path_to_test_model_file = os.path.join(SIFRA_ROOT_DIR, "models", "test_structures", "sysconfig_simple_linear.json")

with open(path_to_test_model_file) as json_data:
    parsed = json.load(json_data)
    component_list = parsed["component_list"].keys()
    print("component_list ", component_list)

    destination = []
    origin = []
    for key in parsed["node_conn_df"].keys():
        destination.append(parsed["node_conn_df"][key]["destination"])
        origin.append(parsed["node_conn_df"][key]["origin"])

    print("destination", destination)
    print("origin", origin)

# no new nodes
# component list should appear in connections
# no new nodes that are orphans
# component type -> dmg algorithm and component list
