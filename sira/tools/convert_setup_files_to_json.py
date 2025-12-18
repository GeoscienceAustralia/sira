import argparse
import ast
import collections
import configparser
import json
import os


def _load_conf_literals(conf_path):
    """
    Safely load a simple Python-style .conf file containing only literal
    assignments (e.g., FOO = 1, BAR = [1, 2], BAZ = {"k": "v"}).

    Returns a plain dict of names to values using ast.literal_eval on the RHS.
    Any non-literal or unsafe statements are ignored.
    """
    with open(conf_path, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src, filename=conf_path, mode="exec")
    result = {}

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            # Only accept literal-safe value nodes
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                # Skip non-literal expressions to avoid code execution
                continue
            result[key] = value
        # Ignore all other node types (imports, exec, calls, etc.)

    return result


def _read_file(setup_file):
    """
    Module for reading in scenario data file
    """
    if not os.path.isfile(setup_file):
        print("[ERROR] could not read file: {}".format(setup_file))
        raise SystemExit()

    file_name, file_ext = os.path.splitext(os.path.basename(setup_file))

    if file_ext == ".conf":
        # Load as a dict of literal assignments (no code execution)
        setup = _load_conf_literals(setup_file)
    elif file_ext == ".ini":
        setup = configparser.ConfigParser()
        setup.optionxform = str
        setup.read(setup_file, encoding="utf-8")
    return setup, file_ext


def convert_ini_object_to_json(setup):
    data = collections.OrderedDict()
    for section in setup.sections():
        data[section] = collections.OrderedDict()
        for param in setup[section].keys():
            try:
                data[section][param] = ast.literal_eval(setup.get(section, param))
            except KeyError:
                data[section][param] = None
    json_data = json.dumps(data, indent=4, sort_keys=True)
    return json_data


def convert_conf_object_to_json(setup):
    data = collections.OrderedDict()

    data["Scenario"] = collections.OrderedDict()

    try:
        data["Scenario"]["SCENARIO_NAME"] = setup["SCENARIO_NAME"]
    except KeyError:
        data["Scenario"]["SCENARIO_NAME"] = None

    try:
        data["Scenario"]["INTENSITY_MEASURE_PARAM"] = setup["INTENSITY_MEASURE_PARAM"]
    except KeyError:
        data["Scenario"]["INTENSITY_MEASURE_PARAM"] = None
    try:
        data["Scenario"]["INTENSITY_MEASURE_UNIT"] = setup["INTENSITY_MEASURE_UNIT"]
    except KeyError:
        data["Scenario"]["INTENSITY_MEASURE_UNIT"] = None
    try:
        data["Scenario"]["FOCAL_HAZARD_SCENARIOS"] = setup["FOCAL_HAZARD_SCENARIOS"]
    except KeyError:
        data["Scenario"]["FOCAL_HAZARD_SCENARIOS"] = None

    data["Hazard"] = collections.OrderedDict()

    try:
        data["Hazard"]["HAZARD_INPUT_METHOD"] = setup["HAZARD_INPUT_METHOD"]
    except KeyError:
        data["Hazard"]["HAZARD_INPUT_METHOD"] = None
    try:
        data["Hazard"]["INTENSITY_MEASURE_MIN"] = setup["INTENSITY_MEASURE_MIN"]
    except KeyError:
        data["Hazard"]["INTENSITY_MEASURE_MIN"] = None
    try:
        data["Hazard"]["INTENSITY_MEASURE_MAX"] = setup["INTENSITY_MEASURE_MAX"]
    except KeyError:
        data["Hazard"]["INTENSITY_MEASURE_MAX"] = None
    try:
        data["Hazard"]["INTENSITY_MEASURE_STEP"] = setup["INTENSITY_MEASURE_STEP"]
    except KeyError:
        data["Hazard"]["INTENSITY_MEASURE_STEP"] = None
    try:
        data["Hazard"]["NUM_SAMPLES"] = setup["NUM_SAMPLES"]
    except KeyError:
        data["Hazard"]["NUM_SAMPLES"] = None

    # Future improvement: read this value from the configuration file.
    try:
        data["Hazard"]["HAZARD_TYPE"] = setup.get("HAZARD_TYPE", "earthquake")
    except Exception:
        data["Hazard"]["HAZARD_TYPE"] = "earthquake"

    # Future improvement: read this value from the configuration file.
    try:
        data["Hazard"]["HAZARD_RASTER"] = None
    except KeyError:
        data["Hazard"]["HAZARD_RASTER"] = None

    data["Restoration"] = collections.OrderedDict()

    try:
        data["Restoration"]["TIME_UNIT"] = setup["TIME_UNIT"]
    except KeyError:
        data["Restoration"]["TIME_UNIT"] = None

    try:
        data["Restoration"]["RESTORATION_PCT_CHECKPOINTS"] = setup["RESTORE_PCT_CHKPOINTS"]
    except KeyError:
        data["Restoration"]["RESTORATION_PCT_CHECKPOINTS"] = None

    try:
        data["Restoration"]["RESTORATION_TIME_STEP"] = setup["RESTORATION_TIME_STEP"]
    except KeyError:
        data["Restoration"]["RESTORATION_TIME_STEP"] = None

    try:
        data["Restoration"]["RESTORATION_TIME_MAX"] = setup["RESTORATION_TIME_MAX"]
    except KeyError:
        data["Restoration"]["RESTORATION_TIME_MAX"] = None

    try:
        data["Restoration"]["RESTORATION_STREAMS"] = setup["RESTORATION_STREAMS"]
    except KeyError:
        data["Restoration"]["RESTORATION_STREAMS"] = None

    data["System"] = collections.OrderedDict()

    try:
        data["System"]["INFRASTRUCTURE_LEVEL"] = setup["INFRASTRUCTURE_LEVEL"]
    except KeyError:
        data["System"]["INFRASTRUCTURE_LEVEL"] = None

    try:
        data["System"]["SYSTEM_CLASSES"] = setup["SYSTEM_CLASSES"]
    except KeyError:
        data["System"]["SYSTEM_CLASSES"] = None

    try:
        data["System"]["SYSTEM_CLASS"] = setup["SYSTEM_CLASS"]
    except KeyError:
        data["System"]["SYSTEM_CLASS"] = None

    try:
        data["System"]["SYSTEM_SUBCLASS"] = setup["SYSTEM_SUBCLASS"]
    except KeyError:
        data["System"]["SYSTEM_SUBCLASS"] = None

    try:
        data["System"]["COMMODITY_FLOW_TYPES"] = setup["COMMODITY_FLOW_TYPES"]
    except KeyError:
        data["System"]["COMMODITY_FLOW_TYPES"] = None

    try:
        data["System"]["SYS_CONF_FILE_NAME"] = setup["SYS_CONF_FILE_NAME"]
    except KeyError:
        data["System"]["SYS_CONF_FILE_NAME"] = None

    data["SWITCHES"] = collections.OrderedDict()

    try:
        data["SWITCHES"]["FIT_PE_DATA"] = setup["FIT_PE_DATA"]
    except KeyError:
        data["SWITCHES"]["FIT_PE_DATA"] = None

    try:
        data["SWITCHES"]["SWITCH_FIT_RESTORATION_DATA"] = setup["SWITCH_FIT_RESTORATION_DATA"]
    except KeyError:
        data["SWITCHES"]["SWITCH_FIT_RESTORATION_DATA"] = None

    try:
        data["SWITCHES"]["SWITCH_SAVE_VARS_NPY"] = setup["SWITCH_SAVE_VARS_NPY"]
    except KeyError:
        data["SWITCHES"]["SWITCH_SAVE_VARS_NPY"] = None

    try:
        data["SWITCHES"]["MULTIPROCESS"] = setup["MULTIPROCESS"]
    except KeyError:
        data["SWITCHES"]["MULTIPROCESS"] = None

    try:
        data["SWITCHES"]["RUN_CONTEXT"] = setup["RUN_CONTEXT"]
    except KeyError:
        data["SWITCHES"]["RUN_CONTEXT"] = None

    json_data = json.dumps(data, indent=4, sort_keys=True)

    return json_data


def convert_to_json(conf_file_path):
    parent_folder_name = os.path.dirname(conf_file_path)
    file_name = os.path.splitext(os.path.basename(conf_file_path))[0]
    json_filename = os.path.join(parent_folder_name, file_name + ".json")
    setup, file_type = _read_file(conf_file_path)
    if file_type == ".conf":
        # When loaded from .conf, `setup` is a dict of literal values
        json_data = convert_conf_object_to_json(setup)
    elif file_type == ".ini":
        json_data = convert_ini_object_to_json(setup)
    else:
        print("\n[ERROR] Incompatible file type for setup file.\n")
    obj = open(json_filename, "w")
    obj.write(json_data)
    obj.close()
    json_filepath = os.path.abspath(json_filename)
    return os.path.abspath(json_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file", type=str, help="Convert specified setup file from `conf` to json"
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Covert all files under specified directory to json.",
    )
    args = parser.parse_args()

    if args.file:
        conf_file_path = args.file
        convert_to_json(conf_file_path)

    # ***********************************************
    # The default location of simulation setup files:
    par_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sim_setup_dir = os.path.join(par_dir, "simulation_setup")
    # ***********************************************

    if args.all:
        conf_file_paths = []
        for root, dir_names, file_names in os.walk(sim_setup_dir):
            for file_name in file_names:
                if file_name.endswith(".conf"):
                    if "simulation_setup" in root:
                        conf_file_path = os.path.join(root, file_name)
                        conf_file_paths.append(conf_file_path)

        for conf_file_path in conf_file_paths:
            convert_to_json(conf_file_path)


if __name__ == "__main__":
    main()
