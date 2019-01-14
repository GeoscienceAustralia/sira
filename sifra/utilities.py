import os


def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_dir_path(path):
    return os.path.dirname(path)


def get_file_extension(path):
    return os.path.splitext(os.path.basename(path))[1][1:]
