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

import numpy as np


# init()
from utils import read_input_data


# import brewer2mpl
# import operator
# import functools

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def cal_pe_ds(comp_name, pga_val, comp_dict, frag_dict):
    """
    Computes prob. of exceedence of component given PGA
    """
    cp_type = comp_dict['component_type'][comp_name]
    ds_list = sorted(frag_dict['damage_median'][cp_type].keys())
    ds_list.remove('DS0 None')
    pe_ds = np.zeros(len(ds_list))
    for i, ds in enumerate(ds_list):
        m = frag_dict['damage_median'][cp_type][ds]
        b = frag_dict['damage_logstd'][cp_type][ds]
        algo = frag_dict['damage_function'][cp_type][ds].lower()
        mode = int(frag_dict['mode'][cp_type][ds])

        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # For single mode functions
        if algo == 'lognormal' and mode == 1:
            pe_ds[i] = stats.lognorm.cdf(pga_val, b, scale=m)
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
        # For functions with two modes
        elif algo == 'lognormal' and mode == 2:
            lower_lim = frag_dict['minimum'][cp_type][ds]
            minpos = min(range(len(hazard_intensity_vals)),
                         key=lambda i: abs(hazard_intensity_vals[i] - lower_lim))
            zl = [0.0] * (minpos + 1)
            ol = [1] * (len(hazard_intensity_vals) - (minpos + 1))
            stepfn = zl + ol
            stepv = stepfn[minpos]

            m = 0.25
            s1 = np.exp(frag_dict['sigma_1'][cp_type][ds])
            s2 = np.exp(frag_dict['sigma_2'][cp_type][ds])
            w1 = 0.5
            w2 = 0.5

            pe_ds[i] = (
                w1 * stats.lognorm.cdf(pga_val, s1, loc=0.0, scale=m) +
                w2 * stats.lognorm.cdf(pga_val, s2, loc=0.0, scale=m)) * stepv
        # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    return np.sort(pe_ds)


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


def calc_recov_time_given_comp_ds(comp, ids):
    """
    Calculates the recovery time of a component, given damage state index
    """
    dmg_state = dmg_states[ids]
    cp_type = comp_dict['component_type'][comp]
    mu = fragdict['recovery_mean'][cp_type][dmg_state]
    sdev = fragdict['recovery_std'][cp_type][dmg_state]
    func = fragdict['functionality'][cp_type][dmg_state]
    cdf = stats.norm.cdf(restoration_time_range, loc=mu, scale=sdev)
    return cdf + (1.0 - cdf) * func


def compute_output_given_ds(cp_func):
    """
    Computes system output given list of component functional status
    
    INPUTS:
    [1] cp_func:
        list of 'functionality' values of components in the system

    OUTPUTS:
    [1] sys_out_capacity_list:
        list of output capacities for each output (production) line
    """
    for tpl in G.get_edgelist():
        eid = G.get_eid(*t)
        origin = G.vs[tpl[0]]['name']
        destin = G.vs[tpl[1]]['name']
        if cpdict[origin]['node_type'] == 'dependency':
            cp_func[nodes.index(destin)] *= cp_func[nodes.index(origin)]
        cap = cp_func[nodes.index(origin)]
        G.es[eid]["capacity"] = cap

    sys_out_capacity_list = []  # normalised capacity: [0.0, 1.0]

    for onode in out_node_list:
        for sup_node_list in nodes_by_commoditytype.values():
            total_available_flow_list = []
            avl_sys_flow_by_src = []
            for inode in sup_node_list:
                avl_sys_flow_by_src.append(
                    G.maxflow_value(G.vs.find(inode).index,
                                    G.vs.find(onode).index,
                                    G.es["capacity"])
                    * input_dict[inode]['CapFraction']
                )

            total_available_flow_list.append(sum(avl_sys_flow_by_src))

        total_available_flow = min(total_available_flow_list)
        sys_out_capacity_list.append(
            min(total_available_flow, output_dict[onode]['CapFraction'])
            * nominal_production
        )

    return sys_out_capacity_list

# -----------------------------------------------------------------------------
# READ in SETUP data
# -----------------------------------------------------------------------------

def check_types_with_db():
    # check to ensure component types match with DB
    cp_types_in_system = list(np.unique(COMP_DF['component_type'].tolist()))
    cp_types_in_db = list(FRAGILITIES.index.levels[0])

    # assert if set(cp_types_in_system).issubset(cp_types_in_db) is True
    assert set(cp_types_in_system).issubset(cp_types_in_db) == True

    return cp_types_in_system, cp_types_in_db

def list_for_calcs(cp_types_in_system, uncosted_comptypes):
    # get list of only those components that are included in cost calculations
    cp_types_costed = [x for x in cp_types_in_system
                       if x not in uncosted_comptypes]
    costed_comptypes = sorted(list(set(cp_types_in_system) -
                                   set(uncosted_comptypes)))

    cpmap = {c: sorted(COMP_DF[COMP_DF['component_type'] == c].index.tolist())
             for c in cp_types_in_system}
    comps_costed = [v for x in cp_types_costed for v in cpmap[x]]

    return

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def something_else():
    nominal_production = SYSOUT_SETUP['Capacity'].sum()
    hazard_transfer_label = HAZARD_TRANSFER_PARAM+' ('+HAZARD_TRANSFER_UNIT+')'

    # -----------------------------------------------------------------------------
    # Convert Dataframes to Dicts for lookup efficiency
    # -----------------------------------------------------------------------------

    comp_dict = COMP_DF.to_dict()
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


if __name__ == "__main__":
    SETUPFILE = sys.argv[1]
    discard = {}
    config = {}
    execfile(SETUPFILE, discard, config)
    SYSTEM_CLASSES = config["SYSTEM_CLASSES"]
    SYSTEM_CLASS = config["SYSTEM_CLASS"]
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

    print read_input_data(config_file=SYS_CONFIG_FILE)
    cp_types_in_system, cp_types_in_db = check_types_with_db()
    uncosted_comptypes = ['CONN_NODE', 'SYSTEM_INPUT', 'SYSTEM_OUTPUT']


