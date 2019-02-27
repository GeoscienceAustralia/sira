import sys
import os
import shutil
import subprocess


path_to_multiple_models = "~/code/sifra/simulation_setup/" \
                          "Multiple_Model_Test__WTP"
path_to_python = "~/anaconda/envs/sifra_py3"

if os.path.isdir(path_to_multiple_models):

    config_file_path = ""
    list_of_model_file_path = []

    for some_file in os.listdir(path_to_multiple_models):
        if "config.json" in some_file:
            config_file_path = os.path.join(path_to_multiple_models, some_file)
        if "model.json" in some_file:
            list_of_model_file_path.append(os.path.join(path_to_multiple_models, some_file))

    if not os.path.isfile(config_file_path):
        print("config file dose not exist.")
        sys.exit(1)

    for model_file_path in list_of_model_file_path:
        if not os.path.isfile(model_file_path):
            print(model_file_path + " not a file model file path.")
            sys.exit(1)

    for model_file_path in list_of_model_file_path:
        src_model_file = model_file_path

        new_model_dir = os.path.join(path_to_multiple_models,model_file_path.split('/')[-1].split('.')[0].split("_")[0])
        des_model_file = os.path.join(new_model_dir,"input", "model.json")
        des_config_file = os.path.join(new_model_dir,"input", "config.json")


        if not os.path.exists(new_model_dir ):
            os.makedirs(new_model_dir)

        new_input_dir =os.path.join(new_model_dir,"input")
        if not os.path.exists(new_input_dir):
            os.makedirs(new_input_dir)

        shutil.copyfile(src_model_file, des_model_file)
        shutil.copyfile(config_file_path, des_config_file)

    for model_dir in os.listdir(path_to_multiple_models):
        model_dir_path = os.path.join(path_to_multiple_models,model_dir)
        if os.path.isdir(model_dir_path):
            subprocess.call([path_to_python, "sifra", "-d",model_dir_path,"-s"])

        
