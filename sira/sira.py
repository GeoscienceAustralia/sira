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
from __future__ import print_function
import sys
import getopt
import os
import operator
import functools
import csv
import copy
import re
import numpy as np
import scipy.stats as stats
import pickle

import networkx as nx
import igraph
import pandas as pd

import matplotlib.pyplot as plt
import prettyplotlib as ppl
import brewer2mpl
from colorama import Fore, Back, Style, init
import parmap
import cPickle


# custom imports
import systemlayout
import siraplot as spl
SETUPFILE = None
from siraclasses import ScenarioDataGetter
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

    return dmg_states, restoration_chkpoints, restoration_pct_steps, hazard_intensity_vals, num_hazard_pts, num_time_steps

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

def network():
    # -----------------------------------------------------------------------------
    # Define the system as a network, with components as nodes
    # -----------------------------------------------------------------------------

    nodes_all = sorted(COMP_DF.index)
    num_elements = len(nodes_all)

    #                    ------
    # Network setup with igraph (for analysis)
    #                    ------
    G = igraph.Graph(directed=True)
    nodes = COMP_DF.index.tolist()

    G.add_vertices(len(nodes))
    G.vs["name"] = nodes
    G.vs["component_type"] = list(COMP_DF['component_type'].values)
    G.vs["cost_fraction"] = list(COMP_DF['cost_fraction'].values)
    G.vs["node_type"] = list(COMP_DF['node_type'].values)
    G.vs["node_cluster"] = list(COMP_DF['node_cluster'].values)
    G.vs["capacity"] = 1.0
    G.vs["functionality"] = 1.0

    for index, row in NODE_CONN_DF.iterrows():
        G.add_edge(row['Orig'], row['Dest'],
                   capacity=G.vs.find(row['Orig'])["capacity"],
                   weight=row['Weight'],
                   distance=row['Distance'])
    return num_elements, G

def network_setup():
    #                    --------
    # Network setup with NetworkX (for drawing graph)
    #                    --------
    X = nx.DiGraph()
    for index, row in NODE_CONN_DF.iterrows():
        X.add_edge(row['Orig'], row['Dest'],
                   capacity=row['Capacity'],
                   weight=row['Weight'],
                   distance=row['Distance'])

    systemlayout.draw_sys_layout(X, COMP_DF, out_dir=OUTPUT_PATH,
                                 graph_label="System Component Layout")
    # -----------------------------------------------------------------------------
    # List of tagged nodes with special roles:
    sup_node_list = [str(k) for k in
                     list(COMP_DF.ix[COMP_DF['node_type'] == 'supply'].index)]
    dep_node_list = [str(k) for k in
                     list(COMP_DF.ix[COMP_DF['node_type'] == 'dependency'].index)]
    src_node_list = [k for (k, v)in X.in_degree().iteritems() if v == 0]
    out_node_list = list(SYSOUT_SETUP.index.get_level_values('OutputNode'))

    return sup_node_list, dep_node_list, src_node_list, out_node_list


def power_calc():
    # -----------------------------------------------------------------------------
    # Power output and economic loss calculations
    # -----------------------------------------------------------------------------

    PGA_str = [('%0.3f' % np.float(x)) for x in hazard_intensity_vals]

    cptype = {}
    cptype_ds_edges = {}
    for comp in nodes_all:
        cptype[comp] = comp_dict['component_type'][comp]
        cptype_ds_edges[cptype[comp]] =\
            sorted(fragdict['damage_ratio'][cptype[comp]].values())

    ###############################################################################
    # component_resp_dict = {k:{c:{} for c in nodes_all} for k in PGA_str}
    tp_cp = []
    for x in comps_costed:
        tp_cp.extend(
            ((x, 'loss_mean'), (x, 'loss_std'), (x, 'func_mean'), (x, 'func_std'))
        )
    mindex = pd.MultiIndex.from_tuples(tp_cp, names=['component_id', 'response'])
    component_resp_df = pd.DataFrame(index=mindex, columns=[PGA_str])
    component_resp_dict = component_resp_df.to_dict()

    ###############################################################################
    # simulation of damage of each component

    calculated_output_array = np.zeros((NUM_SAMPLES, num_hazard_pts))
    economic_loss_array = np.zeros_like(calculated_output_array)

    comp_loss_array = np.zeros((NUM_SAMPLES, num_hazard_pts))
    comp_loss_dict = {c: np.zeros((NUM_SAMPLES, num_hazard_pts)) for c in nodes_all}

    # Record output for:
    # <samples> vs <hazard parameter index> vs <time step index>
    output_array_given_recovery = np.zeros((NUM_SAMPLES, num_hazard_pts, num_time_steps))

    # rnd = stats.uniform.rvs(loc=0, scale=1, size=(NUM_SAMPLES, num_elements))
    # np.save(os.path.join(RAW_OUTPUT_DIR, 'rnd_samples_x_elements.npy'), rnd)

    sys_output_dict = {k: {o: 0 for o in out_node_list} for k in PGA_str}

    # List of output values at output_nodes:
    sys_output_list_given_pga = {k: np.zeros((NUM_SAMPLES, len(out_node_list)))
                                 for k in PGA_str}

    comp_dsix_given_pga = {k: np.zeros((NUM_SAMPLES, len(nodes_all)))
                           for k in PGA_str}

    ids_comp_vs_haz = {p: np.zeros((NUM_SAMPLES, num_elements)) for p in PGA_str}

    return component_resp_df, economic_loss_array, comp_loss_array, comp_loss_dict, \
           output_array_given_recovery, sys_output_dict, sys_output_list_given_pga, comp_dsix_given_pga, ids_comp_vs_haz


def multiprocess_enabling_loop(idxPGA, _PGA_dummy, nPGA):

    if isinstance(_PGA_dummy, list):
        _PGA = _PGA_dummy[idxPGA]
    else:
        _PGA = _PGA_dummy
    print(" {0:3d}  out of {1:3d}".format(idxPGA+1, nPGA))

    # compute pe and determine ds for each component
    ids_comp = np.zeros((NUM_SAMPLES, no_elements), dtype=int)

    rnd = stats.uniform.rvs(loc=0, scale=1, size=(NUM_SAMPLES, num_elements))

    # index of damage state of components: from 0 to nds+1
    for j, comp in enumerate(nodes_all):
        ids_comp[:, j] = np.sum(cal_pe_ds(comp, float(_PGA), compdict, fragdict) > rnd[:, j][:, np.newaxis], axis=1)
        # comp_loss_dict[comp] = np.zeros((num_samples,nPGA))

    component_loss_tmp = {c: [] for c in nodes_all}
    component_func_tmp = {c: [] for c in nodes_all}

    # system output and economic loss
    for i in range(NUM_SAMPLES):
        loss_list_all_comp = []
        cp_func = []
        cp_func_given_time = []

        for j, comp_name in enumerate(nodes_all):
            # ........................................................
            comp_type = compdict['component_type'][comp_name]
            ids = ids_comp[i, j]     # index for component damage state
            ds = dmg_states[ids]   # damage state name
            cf = compdict['cost_fraction'][comp_name]
            dr = fragdict['damage_ratio'][comp_type][ds]
            fn = fragdict['functionality'][comp_type][ds]
            loss = dr * cf
            loss_list_all_comp.append(loss)

            # ........................................................
            # component functionality for calculated damage state:
            cp_func.append(fn)
            cp_func_given_time.append(
                calc_recov_time_given_comp_ds(comp_name, ids))

            comp_loss_dict[comp_name][i, idxPGA] = loss
            component_loss_tmp[comp_name].append(loss)
            component_func_tmp[comp_name].append(fn)
            # ........................................................

        economic_loss_array[i, idxPGA] = sum(loss_list_all_comp)

        outputlist = compute_output_given_ds(cp_func)
        calculated_output_array[i, idxPGA] = sum(outputlist)

        sys_output_list_given_pga[_PGA][i, :] = outputlist

        # restoration status of components over the range of time
        # (num elements X num specified time units)
        cp_func_given_time = np.array(cp_func_given_time)
        for t in range(num_time_steps):
            output_array_given_recovery[i, idxPGA, t]\
                = sum(compute_output_given_ds(cp_func_given_time[:, t]))

    comp_resp_dict = dict()

    for j, comp_name in enumerate(nodes_all):
        comp_resp_dict[(comp_name, 'loss_mean')]\
            = np.mean(component_loss_tmp[comp_name])

        comp_resp_dict[(comp_name, 'loss_std')]\
            = np.std(component_loss_tmp[comp_name])

        comp_resp_dict[(comp_name, 'func_mean')]\
            = np.mean(component_func_tmp[comp_name])

        comp_resp_dict[(comp_name, 'func_std')]\
            = np.std(component_func_tmp[comp_name])

        comp_resp_dict[(comp_name, 'num_failures')]\
            = np.mean(ids_comp[:, j] >= (len(dmg_states) - 1))

    sys_out_dict = dict()
    for onx, onode in enumerate(out_node_list):
        sys_out_dict[onode]\
            = np.mean(sys_output_list_given_pga[_PGA][:, onx])
    return ids_comp, sys_out_dict, comp_resp_dict


def calc_loss_arrays(parallel_or_serial):

    print("\nCalculating system response to hazard transfer parameters...")
    component_resp_dict = component_resp_df.to_dict()
    sys_output_dict = {k: {o: 0 for o in out_node_list} for k in PGA_str}
    ids_comp_vs_haz = {p: np.zeros((num_samples, no_elements)) for p in PGA_str}

    if parallel_or_serial:
        parallel_return \
            = parmap.map(multiprocess_enabling_loop, range(len(PGA_str)), PGA_str, nPGA)

        for idxPGA, _PGA in enumerate(PGA_str):
            ids_comp_vs_haz[_PGA] = parallel_return[idxPGA][0]
            sys_output_dict[_PGA] = parallel_return[idxPGA][1]
            component_resp_dict[_PGA] = parallel_return[idxPGA][2]
    else:
        for idxPGA, _PGA in enumerate(PGA_str):
            ids_comp_vs_haz[_PGA], sys_output_dict[_PGA], component_resp_dict[_PGA] = \
                multiprocess_enabling_loop(idxPGA=idxPGA, _PGA_dummy=_PGA, nPGA=nPGA)

    # saving for test cases
    # cPickle.dump(ids_comp_vs_haz, open('tests/ids_comp_vs_haz.pick', 'wb'))
    # cPickle.dump(sys_output_dict, open('tests/sys_output_dict.pick', 'wb'))

    return ids_comp_vs_haz, sys_output_dict, component_resp_dict

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

    dmg_states, restoration_chkpoints, restoration_pct_steps, hazard_intensity_vals, \
        num_hazard_pts, num_time_steps = simulation_parameters()
    num_elements, G = network()
    sup_node_list, dep_node_list, src_node_list, out_node_list = network_setup()

    nodes_all = sorted(COMP_DF.index)
    no_elements = len(nodes_all)

    component_resp_df, economic_loss_array, comp_loss_array, comp_loss_dict, \
           output_array_given_recovery, sys_output_dict, sys_output_list_given_pga, \
            comp_dsix_given_pga, ids_comp_vs_haz = power_calc()

    rnd = stats.uniform.rvs(loc=0, scale=1, size=(NUM_SAMPLES, num_elements))
    print (rnd)
    print (rnd.shape)



