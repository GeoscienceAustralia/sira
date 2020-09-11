"""
sira/utilities.py
This module provides a collection of helper functions
"""
from pathlib import Path
import os
import re

def relpath(path_to, start):
    """
    `pathlib` only implementation of `os.path.relpath`
    Taken from answer by Brett Ryland
    https://stackoverflow.com/a/60671745
    """
    path_to = Path(path_to).resolve()
    path_from = Path(start).resolve()
    try:
        for p in (*reversed(path_from.parents), path_from):
            head, tail = p, path_to.relative_to(p)
    except ValueError:  # Stop when the paths diverge.
        pass
    return Path('./' * (len(path_from.parents) - len(head.parents))).joinpath(tail)


def get_config_file_path(input_dir):
    """Returns path to the scneario config file, given the input dir path"""
    config_file_name = None
    for fname in os.listdir(input_dir):
        confmatch = re.search(r"(?i)^config.*\.json$", fname)
        if confmatch is not None:
            config_file_name = confmatch.string
    config_file = os.path.join(input_dir, config_file_name)
    return config_file


def get_model_file_path(input_dir):
    """Returns path to the model file, given the input dir path"""
    model_file_name = None
    for fname in os.listdir(input_dir):
        modelmatch = re.search(r"(?i)^model.*\.json$", fname)
        if modelmatch is not None:
            model_file_name = modelmatch.string
    model_file = os.path.join(input_dir, model_file_name)
    return model_file
