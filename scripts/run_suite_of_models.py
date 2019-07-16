import sys
import os
import ntpath
import shutil
import re
import subprocess


project_root_dir = os.path.abspath(
    "./simulation_setup/Multiple_Model_Test__WTP/")

if not os.path.isdir(project_root_dir):
    print("Invalid path supplied:\n {}".format(project_root_dir))
    sys.exit(1)

# ------------------------------------------------------------------------------
# Find and check that `config` and `model` files are present in target dir.
#
# Required inputs for each batch model run:
#     - one single config file in json
#     - one or more model files in json

config_file_path = None        # This must be an absolute path
config_file_name = None
model_file_path_list = []

for fname in os.listdir(project_root_dir):

    confmatch = re.search(r"(?i)^config.*\.json$", fname)
    if confmatch is not None:
        config_file_name = confmatch.string
        config_file_path = os.path.join(project_root_dir, config_file_name)

    modelmatch = re.search(r"(?i)^model.*\.json$", fname)
    if modelmatch is not None:
        model_file_path_list.append(
            os.path.join(project_root_dir, modelmatch.string))


if not os.path.isfile(config_file_path):
    print("config file dose not exist.")
    sys.exit(1)

for model_file_path in model_file_path_list:
    if not os.path.isfile(model_file_path):
        print(model_file_path +
              " does not match model file naming requirement.")
        sys.exit(1)

# ------------------------------------------------------------------------------
# Organise all models and config files in its separate directories as
# required by SIRA

for model_file_path in model_file_path_list:

    src_model_file = model_file_path
    model_file_name = ntpath.basename(model_file_path)
    model_file_no_ext = ntpath.basename(os.path.splitext(model_file_path)[0])

    # Check for dir with same name as MODEL_FILE, without the extension.
    #   Create the dir if it does not exist.
    new_model_dir = os.path.join(project_root_dir, model_file_no_ext)
    if not os.path.exists(new_model_dir):
        os.makedirs(new_model_dir)

    # Check for input dir ./MODEL_FILE/input
    #   Create the dir if it does not exist.
    new_input_dir = os.path.join(new_model_dir, "input")
    if not os.path.exists(new_input_dir):
        os.makedirs(new_input_dir)

    des_model_file = os.path.join(new_model_dir, "input", model_file_name)

    # Copy MODEL_FILE in new location.
    shutil.copyfile(src_model_file, des_model_file)

    # Create symlink to CONFIG_FILE, rather than copying the same file across
    #   multiple model directories. All models refer to same config file
    new_config_file_path = os.path.join(
        new_model_dir, "input", config_file_name)
    if not os.path.exists(new_config_file_path):
        os.symlink(config_file_path, new_config_file_path)

# ------------------------------------------------------------------------------
# Run each distinct model with the same scenario config setup

for model_dir in os.listdir(project_root_dir):
    model_dir_path = os.path.join(project_root_dir, model_dir)
    if os.path.isdir(model_dir_path):
        subprocess.call(
            ['python', "sira", "-d", model_dir_path, "-sfl"])

# ------------------------------------------------------------------------------
