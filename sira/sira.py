#!/usr/bin/env python

"""
sira.py

A tool for seismic performace analysis of electrical power infrastructure
----------------------------------------------------------------------------

    INPUTS
[1] List of system components, each must be assigned to a specific 'type'
[2] Component configuration within system
    This is represented as a list of pairwise connections between nodes
    Order is important. Directionality defined as node1 -> node2
    Bidirectional edges are supported
[3] Fragility functions for each component type, for each damage state
[4] Functionality of each component type, associated with each damage state
[5] Recovery functions for each component type, for each damage state

    OUTPUTS
[1] Mean economic loss vs. shaking intensity
[2] Mean system output vs. shaking intensity
[3] Mean required time to restore full capacity vs. PGA
[4] Simulated system fragility
[5] Loss of functionality based on components type

----------------------------------------------------------------------------
"""

# from __future__ import print_function
import sys
import getopt
import os
import scipy.stats as stats
from colorama import Fore
from siraclasses import ScenarioDataGetter

import numpy as np


# init()
from utils import read_input_data


# import brewer2mpl
# import operator
# import functools

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def pe2pb(pe):
    """
    Convert probablity of exceedence of damage states, to
    probability of being in each discrete damage state
    """
    # sorted array: from max to min
    pex = np.sort(pe)[::-1]
    tmp = -1.0 * np.diff(pex)
    pb = np.append(tmp, pex[-1])
    pb = np.insert(pb, 0, 1 - pex[0])
    return pb


def check_types_with_db():
    # check to ensure component types match with DB
    cp_types_in_system = list(np.unique(COMP_DF['component_type'].tolist()))
    cp_types_in_db = list(FRAGILITIES.index.levels[0])
    assert set(cp_types_in_system).issubset(cp_types_in_db) == True
    return cp_types_in_system, cp_types_in_db

def list_of_components_for_cost_calculation(cp_types_in_system, uncosted_comptypes):
    # get list of only those components that are included in cost calculations
    cp_types_costed = [x for x in cp_types_in_system
                       if x not in uncosted_comptypes]
    costed_comptypes = sorted(list(set(cp_types_in_system) -
                                   set(uncosted_comptypes)))

    cpmap = {c: sorted(COMP_DF[COMP_DF['component_type'] == c].index.tolist())
             for c in cp_types_in_system}
    comps_costed = [v for x in cp_types_costed for v in cpmap[x]]

    return costed_comptypes, comps_costed

def convert_df_to_dict():
    # -----------------------------------------------------------------------------
    # Convert Dataframes to Dicts for lookup efficiency
    # -----------------------------------------------------------------------------

    cpdict = {}
    for i in list(COMP_DF.index):
        cpdict[i] = COMP_DF.ix[i].to_dict()

    output_dict = {}
    for k1 in list(np.unique(SYSOUT_SETUP.index.get_level_values('OutputNode'))):
        output_dict[k1] = {}
        output_dict[k1] = SYSOUT_SETUP.ix[k1].to_dict()

    input_dict = {}
    for k1 in list(np.unique(SYSINP_SETUP.index.get_level_values('InputNode'))):
        input_dict[k1] = {}
        input_dict[k1] = SYSINP_SETUP.ix[k1].to_dict()
        # input_dict[k1]['AvlCapacity'] = input_dict[k1]['Capacity']

    nodes_by_commoditytype = {}
    for i in np.unique(SYSINP_SETUP['CommodityType']):
        nodes_by_commoditytype[i] \
            = [x for x in SYSINP_SETUP.index
               if SYSINP_SETUP.ix[x]['CommodityType'] == i]

    return cpdict, output_dict, input_dict, nodes_by_commoditytype


def simulation_parameters():
    # -----------------------------------------------------------------------------
    # Simulation Parameters
    # -----------------------------------------------------------------------------

    dmg_states = sorted([str(d) for d in FRAGILITIES.index.levels[1]])

    # max_recoverytimes_dict = {}
    # for x in cp_types_in_system:
    #     max_recoverytimes_dict[x] =\
    #         FRAGILITIES.ix[x, dmg_states[len(dmg_states) - 1]]['recovery_mean']

    restoration_time_range, time_step =\
        np.linspace(0, RESTORE_TIME_UPPER, num=RESTORE_TIME_UPPER+1,
                    endpoint=USE_ENDPOINT, retstep=True)

    restoration_chkpoints, restoration_pct_steps =\
        np.linspace(0.0, 1.0, RESTORE_PCT_CHKPOINTS, retstep=True)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    hazard_data_points = int(round((PGA_MAX - PGA_MIN) / float(PGA_STEP) + 1))

    hazard_intensity_vals = np.linspace(PGA_MIN, PGA_MAX,
                             num=hazard_data_points)

    num_hazard_pts = len(hazard_intensity_vals)
    num_time_steps = len(restoration_time_range)

    return dmg_states, restoration_chkpoints, restoration_pct_steps, num_hazard_pts, num_time_steps

def fragility_dict():
    # --- Fragility data ---

    # add 'DS0 None' damage state
    for comp in FRAGILITIES.index.levels[0]:
        FRAGILITIES.loc[(comp, 'DS0 None'), 'damage_function'] = 'Lognormal'
        FRAGILITIES.loc[(comp, 'DS0 None'), 'damage_median'] = np.inf
        FRAGILITIES.loc[(comp, 'DS0 None'), 'damage_logstd'] = 1.0
        FRAGILITIES.loc[(comp, 'DS0 None'), 'damage_lambda'] = 0.01
        FRAGILITIES.loc[(comp, 'DS0 None'), 'damage_ratio'] = 0.0
        FRAGILITIES.loc[(comp, 'DS0 None'), 'recovery_mean'] = -np.inf
        FRAGILITIES.loc[(comp, 'DS0 None'), 'recovery_std'] = 1.0
        FRAGILITIES.loc[(comp, 'DS0 None'), 'functionality'] = 1.0
        FRAGILITIES.loc[(comp, 'DS0 None'), 'mode'] = 1
        FRAGILITIES.loc[(comp, 'DS0 None'), 'minimum'] = -np.inf
        FRAGILITIES.loc[(comp, 'DS0 None'), 'sigma_1'] = 'NA'
        FRAGILITIES.loc[(comp, 'DS0 None'), 'sigma_2'] = 'NA'

    fgdt = FRAGILITIES.to_dict()
    fragdict = {}
    for key, val in fgdt.iteritems():
        elemdict = {}
        for t, v in val.iteritems():
            elem = t[0]
            ds = t[1]
            if elem not in elemdict.keys():
                elemdict[elem] = {}
                elemdict[elem][ds] = v
            elif ds not in elemdict[elem].keys():
                elemdict[elem][ds] = v
        fragdict[key] = elemdict

    return fragdict

if __name__ == "__main__":
    SETUPFILE = sys.argv[1]
    discard = {}
    config = {}
    execfile(SETUPFILE, discard, config)
    from siraclasses import FacilityDataGetter

    scenario_data = ScenarioDataGetter(SETUPFILE)
    facility_data = FacilityDataGetter(SETUPFILE)

    SYSTEM_CLASSES = facility_data.system_classes  # config["SYSTEM_CLASSES"]
    SYSTEM_CLASS = facility_data.system_class  # config["SYSTEM_CLASS"]
    COMMODITY_FLOW_TYPES = config["COMMODITY_FLOW_TYPES"]

    PGA_MIN = config["PGA_MIN"]
    PGA_MAX = config["PGA_MAX"]
    PGA_STEP = config["PGA_STEP"]
    NUM_SAMPLES = config["NUM_SAMPLES"]

    HAZARD_TRANSFER_PARAM = config["HAZARD_TRANSFER_PARAM"]
    HAZARD_TRANSFER_UNIT = config["HAZARD_TRANSFER_UNIT"]

    TIME_UNIT = config["TIME_UNIT"]
    RESTORE_TIME_STEP = config["RESTORE_TIME_STEP"]
    RESTORE_PCT_CHKPOINTS = config["RESTORE_PCT_CHKPOINTS"]
    RESTORE_TIME_UPPER = config["RESTORE_TIME_UPPER"]
    RESTORE_TIME_MAX = config["RESTORE_TIME_MAX"]
    INPUT_DIR_NAME = config["INPUT_DIR_NAME"]
    OUTPUT_DIR_NAME = config["OUTPUT_DIR_NAME"]

    SYS_CONF_FILE_NAME = config["SYS_CONF_FILE_NAME"]

    USE_ENDPOINT = config["USE_ENDPOINT"]
    FIT_PE_DATA = config["FIT_PE_DATA"]
    FIT_RESTORATION_DATA = config["FIT_RESTORATION_DATA"]
    SAVE_VARS_NPY = config["SAVE_VARS_NPY"]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Define input files, output location, scenario inputs
    INPUT_PATH = os.path.join(os.getcwd(), INPUT_DIR_NAME)
    SYS_CONFIG_FILE = os.path.join(INPUT_PATH, SYS_CONF_FILE_NAME)

    if not os.path.exists(OUTPUT_DIR_NAME):
        os.makedirs(OUTPUT_DIR_NAME)
    OUTPUT_PATH = os.path.join(os.getcwd(), OUTPUT_DIR_NAME)

    RAW_OUTPUT_DIR = os.path.join(os.getcwd(), OUTPUT_DIR_NAME, 'raw_output')
    if not os.path.exists(RAW_OUTPUT_DIR):
        os.makedirs(RAW_OUTPUT_DIR)

    # Read in INPUT data files
    NODE_CONN_DF, COMP_DF, SYSOUT_SETUP, SYSINP_SETUP, FRAGILITIES = read_input_data(config_file=SYS_CONFIG_FILE)

    cp_types_in_system, cp_types_in_db = check_types_with_db()
    uncosted_comptypes = ['CONN_NODE', 'SYSTEM_INPUT', 'SYSTEM_OUTPUT']
    costed_comptypes, comps_costed = list_of_components_for_cost_calculation(cp_types_in_system, uncosted_comptypes)
    nominal_production = SYSOUT_SETUP['Capacity'].sum()
    hazard_transfer_label = HAZARD_TRANSFER_PARAM+' ('+HAZARD_TRANSFER_UNIT+')'

    comp_dict = COMP_DF.to_dict()
    cpdict, output_dict, input_dict, nodes_by_commoditytype = convert_df_to_dict()

    fragdict = fragility_dict()

    dmg_states, restoration_chkpoints, restoration_pct_steps, num_hazard_pts, num_time_steps = simulation_parameters()





