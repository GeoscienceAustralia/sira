import os
import re


def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_dir_path(path):
    return os.path.dirname(path)


def get_file_extension(path):
    return os.path.splitext(os.path.basename(path))[1][1:]


def get_config_file_path(input_dir):
    config_file_name = None
    for fname in os.listdir(input_dir):
        confmatch = re.search(r"(?i)^config.*\.json$", fname)
        if confmatch is not None:
            config_file_name = confmatch.string
    config_file = os.path.join(input_dir, config_file_name)
    return config_file


def get_model_file_path(input_dir):
    model_file_name = None
    for fname in os.listdir(input_dir):
        modelmatch = re.search(r"(?i)^model.*\.json$", fname)
        if modelmatch is not None:
            model_file_name = modelmatch.string
    model_file = os.path.join(input_dir, model_file_name)
    return model_file
