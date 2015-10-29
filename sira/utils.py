import pandas as pd

__author__ = 'sudipta'
import ConfigParser


def gc(config_file, a, b):
    '''
    This reads data from ConfigParser.
    '''
    Config = ConfigParser.ConfigParser()
    Config.read(config_file)
    return Config.get(a, b)


def read_input_data(config_file):
    NODE_CONN_DF = pd.read_excel(
        config_file, 'component_connections',
        index_col=None, skiprows=3, skipinitialspace=True)

    COMP_DF = pd.read_excel(
        config_file, 'comp_list',
        index_col='component_id',
        skiprows=3, skipinitialspace=True)

    SYSOUT_SETUP = pd.read_excel(
        config_file, 'output_setup',
        index_col='OutputNode',
        skiprows=3, skipinitialspace=True)
    SYSOUT_SETUP = SYSOUT_SETUP.sort('Priority', ascending=True)

    SYSINP_SETUP = pd.read_excel(
        config_file, 'supply_setup',
        index_col='InputNode',
        skiprows=3, skipinitialspace=True)

    FRAGILITIES = pd.read_excel(
        config_file, 'comp_type_dmg_algo',
        index_col=['component_type', 'damage_state'],
        skiprows=3, skipinitialspace=True)

    return NODE_CONN_DF, COMP_DF, SYSOUT_SETUP, SYSINP_SETUP, FRAGILITIES