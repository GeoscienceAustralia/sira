import re
import shutil
import subprocess
import sys
from pathlib import Path

project_root_dir_str = "./simulation_setup/batch_model_run_test/"

project_root_dir = Path(project_root_dir_str).resolve()
if not project_root_dir.is_dir():
    print(f"Invalid path supplied:\n {str(project_root_dir)}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Find and check that `config` and `model` files are present in target dir.
#
# Required inputs for each batch model run:
#     - one single config file in json
#     - one or more model files in json

config_file_path = [x for x in project_root_dir.glob("config*.json")][0]
config_file_name = config_file_path.name
model_file_path_list = [x for x in project_root_dir.rglob("model*.json")]

if not config_file_path.is_file():
    print("config file dose not exist.")
    sys.exit(1)

if not model_file_path_list:
    print("No model files found with required naming structure.")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Organise all models and config files in its separate directories as
# required by SIRA

for model_file_path in model_file_path_list:
    src_model_file = model_file_path
    model_file_name = model_file_path.name
    model_file_no_ext = model_file_path.stem

    # Check for dir with same name as MODEL_FILE, without the extension.
    #   Create the dir if it does not exist.
    new_model_dir = Path(project_root_dir, model_file_no_ext)
    if not new_model_dir.exists():
        new_model_dir.mkdir(parents=True, exist_ok=True)

    # Check for input dir ./MODEL_FILE/input
    #   Create the dir if it does not exist.
    new_input_dir = Path(new_model_dir, "input")
    if not new_input_dir.exists():
        new_input_dir.mkdir(parents=True, exist_ok=True)

    des_model_file = Path(new_model_dir, "input", model_file_name)

    # Copy MODEL_FILE in new location.
    shutil.copyfile(src_model_file, des_model_file)

    # Create symlink to CONFIG_FILE, rather than copying the same file across
    #   multiple model directories. All models refer to same config file
    new_config_file_path = Path(new_model_dir, "input", config_file_name)
    if not new_config_file_path.exists():
        new_config_file_path.symlink_to(config_file_path)

# ------------------------------------------------------------------------------
# Run each distinct model with the same scenario config setup

model_dir_path_list = [
    x for x in Path(project_root_dir).glob("*") if not re.search(r"^[_.]", x.name)
]
for model_dir_path in model_dir_path_list:
    if model_dir_path.is_dir():
        subprocess.call(["python", "sira", "-d", model_dir_path, "-sfl"])

# ------------------------------------------------------------------------------
