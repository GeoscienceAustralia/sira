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
from utils import read_input_data
from siraclasses import Scenario, Facility


# import brewer2mpl
# import operator
# import functools

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def cal_pe_ds(comp, PGA, compdict, fragdict):
    '''
    Computes prob. of exceedence of component given PGA
    '''
    ct = compdict['component_type'][comp]
    ds_list = sorted(fragdict['damage_median'][ct].keys())
    ds_list.remove('DS0 None')
    pe_ds = np.zeros(len(ds_list))
    for i, ds in enumerate(ds_list):
        m = fragdict['damage_median'][ct][ds]
        b = fragdict['damage_logstd'][ct][ds]
        algo = fragdict['damage_function'][ct][ds].lower()
        mode = int(fragdict['mode'][ct][ds])
        # pe_ds[i] = stats.lognorm.cdf(PGA,b,scale=m)
        if algo == 'lognormal' and mode == 1:
            pe_ds[i] = stats.lognorm.cdf(PGA, b, scale=m)
        elif algo == 'lognormal' and mode == 2:
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            lower_lim = fragdict['minimum'][ct][ds]
            minpos = min(range(len(hazard_intensity_vals)),
                         key=lambda i: abs(hazard_intensity_vals[i] - lower_lim))
            zl = [0.0] * (minpos + 1)
            ol = [1] * (len(hazard_intensity_vals) - (minpos + 1))
            stepfn = zl + ol
            stepv = stepfn[minpos]
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            m = 0.25
            s1 = np.exp(fragdict['sigma_1'][ct][ds])
            s2 = np.exp(fragdict['sigma_2'][ct][ds])
            w1 = 0.5
            w2 = 0.5
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            pe_ds[i] = (w1 * stats.lognorm.cdf(PGA, s1, loc=0.0, scale=m) +
                        w2 * stats.lognorm.cdf(PGA, s2, loc=0.0, scale=m)) * stepv
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return np.sort(pe_ds)


def calc_recov_time_given_comp_ds(comp, ids):
    '''
    Calculates the recovery time of a component, given damage state index
    '''
    ds = dmg_states[ids]
    ct = comp_dict['component_type'][comp]
    m = fragdict['recovery_mean'][ct][ds]
    s = fragdict['recovery_std'][ct][ds]
    fn = fragdict['functionality'][ct][ds]
    cdf = stats.norm.cdf(restoration_time_range, loc=m, scale=s)
    return cdf + (1.0 - cdf) * fn


def compute_output_given_ds(cp_func):
    '''
    Computes system output given list of component functional status
    '''
    for t in G.get_edgelist():
        eid = G.get_eid(*t)
        origin = G.vs[t[0]]['name']
        destin = G.vs[t[1]]['name']
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
    cp_types_in_system = list(np.unique(comp_df['component_type'].tolist()))
    cp_types_in_db = list(fragilities.index.levels[0])
    assert set(cp_types_in_system).issubset(cp_types_in_db) == True
    return cp_types_in_system, cp_types_in_db


def list_of_components_for_cost_calculation(cp_types_in_system, uncosted_comptypes):
    # get list of only those components that are included in cost calculations
    cp_types_costed = [x for x in cp_types_in_system
                       if x not in uncosted_comptypes]
    costed_comptypes = sorted(list(set(cp_types_in_system) -
                                   set(uncosted_comptypes)))

    cpmap = {c: sorted(comp_df[comp_df['component_type'] == c].index.tolist())
             for c in cp_types_in_system}
    comps_costed = [v for x in cp_types_costed for v in cpmap[x]]

    return costed_comptypes, comps_costed


def convert_df_to_dict():
    # -----------------------------------------------------------------------------
    # Convert Dataframes to Dicts for lookup efficiency
    # -----------------------------------------------------------------------------

    cpdict = {}
    for i in list(comp_df.index):
        cpdict[i] = comp_df.ix[i].to_dict()

    output_dict = {}
    for k1 in list(np.unique(sysout_setup.index.get_level_values('OutputNode'))):
        output_dict[k1] = {}
        output_dict[k1] = sysout_setup.ix[k1].to_dict()

    input_dict = {}
    for k1 in list(np.unique(sysinp_setup.index.get_level_values('InputNode'))):
        input_dict[k1] = {}
        input_dict[k1] = sysinp_setup.ix[k1].to_dict()
        # input_dict[k1]['AvlCapacity'] = input_dict[k1]['Capacity']

    nodes_by_commoditytype = {}
    for i in np.unique(sysinp_setup['CommodityType']):
        nodes_by_commoditytype[i] \
            = [x for x in sysinp_setup.index
               if sysinp_setup.ix[x]['CommodityType'] == i]

    return cpdict, output_dict, input_dict, nodes_by_commoditytype


def simulation_parameters():
    # -----------------------------------------------------------------------------
    # Simulation Parameters
    # -----------------------------------------------------------------------------
    dmg_states = sorted([str(d) for d in fragilities.index.levels[1]])
    restoration_time_range, time_step =\
        np.linspace(0, sc.restore_time_upper, num= sc.restore_time_upper + 1,
                    endpoint=sc.use_end_point, retstep=True)
    restoration_chkpoints, restoration_pct_steps =\
        np.linspace(0.0, 1.0, sc.restore_pct_chkpoints, retstep=True)
    hazard_data_points = int(round((sc.haz_param_max - sc.haz_param_min) / float(sc.haz_param_step) + 1))
    hazard_intensity_vals = np.linspace(sc.haz_param_min, sc.haz_param_max, num=hazard_data_points)
    num_hazard_pts = len(hazard_intensity_vals)
    num_time_steps = len(restoration_time_range)

    return restoration_time_range, dmg_states, restoration_chkpoints, restoration_pct_steps, hazard_intensity_vals, \
           num_hazard_pts, num_time_steps


def fragility_dict():
    # --- Fragility data ---

    # add 'DS0 None' damage state
    for comp in fragilities.index.levels[0]:
        fragilities.loc[(comp, 'DS0 None'), 'damage_function'] = 'Lognormal'
        fragilities.loc[(comp, 'DS0 None'), 'damage_median'] = np.inf
        fragilities.loc[(comp, 'DS0 None'), 'damage_logstd'] = 1.0
        fragilities.loc[(comp, 'DS0 None'), 'damage_lambda'] = 0.01
        fragilities.loc[(comp, 'DS0 None'), 'damage_ratio'] = 0.0
        fragilities.loc[(comp, 'DS0 None'), 'recovery_mean'] = -np.inf
        fragilities.loc[(comp, 'DS0 None'), 'recovery_std'] = 1.0
        fragilities.loc[(comp, 'DS0 None'), 'functionality'] = 1.0
        fragilities.loc[(comp, 'DS0 None'), 'mode'] = 1
        fragilities.loc[(comp, 'DS0 None'), 'minimum'] = -np.inf
        fragilities.loc[(comp, 'DS0 None'), 'sigma_1'] = 'NA'
        fragilities.loc[(comp, 'DS0 None'), 'sigma_2'] = 'NA'

    fgdt = fragilities.to_dict()
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

    nodes_all = sorted(comp_df.index)
    num_elements = len(nodes_all)

    #                    ------
    # Network setup with igraph (for analysis)
    #                    ------
    G = igraph.Graph(directed=True)
    nodes = comp_df.index.tolist()

    G.add_vertices(len(nodes))
    G.vs["name"] = nodes
    G.vs["component_type"] = list(comp_df['component_type'].values)
    G.vs["cost_fraction"] = list(comp_df['cost_fraction'].values)
    G.vs["node_type"] = list(comp_df['node_type'].values)
    G.vs["node_cluster"] = list(comp_df['node_cluster'].values)
    G.vs["capacity"] = 1.0
    G.vs["functionality"] = 1.0

    for index, row in node_conn_df.iterrows():
        G.add_edge(row['Orig'], row['Dest'],
                   capacity=G.vs.find(row['Orig'])["capacity"],
                   weight=row['Weight'],
                   distance=row['Distance'])
    return nodes, num_elements, G


def network_setup():
    #                    --------
    # Network setup with NetworkX (for drawing graph)
    #                    --------
    X = nx.DiGraph()
    for index, row in node_conn_df.iterrows():
        X.add_edge(row['Orig'], row['Dest'],
                   capacity=row['Capacity'],
                   weight=row['Weight'],
                   distance=row['Distance'])
    systemlayout.draw_sys_layout(X, comp_df, out_dir=OUTPUT_PATH,
                                 graph_label="System Component Layout")
    # -----------------------------------------------------------------------------
    # List of tagged nodes with special roles:
    sup_node_list = [str(k) for k in
                     list(comp_df.ix[comp_df['node_type'] == 'supply'].index)]
    dep_node_list = [str(k) for k in
                     list(comp_df.ix[comp_df['node_type'] == 'dependency'].index)]
    src_node_list = [k for (k, v)in X.in_degree().iteritems() if v == 0]
    out_node_list = list(sysout_setup.index.get_level_values('OutputNode'))

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

    calculated_output_array = np.zeros((sc.num_samples, num_hazard_pts))
    economic_loss_array = np.zeros_like(calculated_output_array)

    comp_loss_array = np.zeros((sc.num_samples, num_hazard_pts))
    comp_loss_dict = {c: np.zeros((sc.num_samples, num_hazard_pts)) for c in nodes_all}

    # Record output for:
    # <samples> vs <hazard parameter index> vs <time step index>
    output_array_given_recovery = np.zeros((sc.num_samples, num_hazard_pts, num_time_steps))

    # rnd = stats.uniform.rvs(loc=0, scale=1, size=(NUM_SAMPLES, num_elements))
    # np.save(os.path.join(RAW_OUTPUT_DIR, 'rnd_samples_x_elements.npy'), rnd)

    sys_output_dict = {k: {o: 0 for o in out_node_list} for k in PGA_str}

    # List of output values at output_nodes:
    sys_output_list_given_pga = {k: np.zeros((sc.num_samples, len(out_node_list)))
                                 for k in PGA_str}

    comp_dsix_given_pga = {k: np.zeros((sc.num_samples, len(nodes_all)))
                           for k in PGA_str}

    ids_comp_vs_haz = {p: np.zeros((sc.num_samples, num_elements)) for p in PGA_str}

    return PGA_str, calculated_output_array, component_resp_df, economic_loss_array, comp_loss_array, comp_loss_dict, \
           output_array_given_recovery, sys_output_dict, sys_output_list_given_pga, comp_dsix_given_pga, ids_comp_vs_haz


def multiprocess_enabling_loop(idxPGA, _PGA_dummy, nPGA):

    if isinstance(_PGA_dummy, list):
        _PGA = _PGA_dummy[idxPGA]
    else:
        _PGA = _PGA_dummy
    print(" {0:3d}  out of {1:3d}".format(idxPGA+1, nPGA))

    # compute pe and determine ds for each component
    ids_comp = np.zeros((sc.num_samples, no_elements), dtype=int)

    rnd = stats.uniform.rvs(loc=0, scale=1, size=(sc.num_samples, num_elements))

    # index of damage state of components: from 0 to nds+1
    for j, comp in enumerate(nodes_all):
        ids_comp[:, j] = np.sum(cal_pe_ds(comp, float(_PGA), comp_dict, fragdict) > rnd[:, j][:, np.newaxis], axis=1)
        # comp_loss_dict[comp] = np.zeros((num_samples,nPGA))

    component_loss_tmp = {c: [] for c in nodes_all}
    component_func_tmp = {c: [] for c in nodes_all}

    # system output and economic loss
    for i in range(sc.num_samples):
        loss_list_all_comp = []
        cp_func = []
        cp_func_given_time = []

        for j, comp_name in enumerate(nodes_all):
            # ........................................................
            comp_type = comp_dict['component_type'][comp_name]
            ids = ids_comp[i, j]     # index for component damage state
            ds = dmg_states[ids]   # damage state name
            cf = comp_dict['cost_fraction'][comp_name]
            dr = fragdict['damage_ratio'][comp_type][ds]
            fn = fragdict['functionality'][comp_type][ds]
            loss = dr * cf
            loss_list_all_comp.append(loss)

            # ........................................................
            # component functionality for calculated damage state:
            cp_func.append(fn)
            cp_func_given_time.append(calc_recov_time_given_comp_ds(comp_name, ids))

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
    ids_comp_vs_haz = {p: np.zeros((sc.num_samples, no_elements)) for p in PGA_str}

    if parallel_or_serial:
        print('\n===================>>>>>multiprocessor computation on <<<<========================')
        parallel_return = parmap.map(multiprocess_enabling_loop, range(len(PGA_str)), PGA_str, num_hazard_pts)

        for idxPGA, _PGA in enumerate(PGA_str):
            ids_comp_vs_haz[_PGA] = parallel_return[idxPGA][0]
            sys_output_dict[_PGA] = parallel_return[idxPGA][1]
            component_resp_dict[_PGA] = parallel_return[idxPGA][2]
    else:
        for idxPGA, _PGA in enumerate(PGA_str):
            ids_comp_vs_haz[_PGA], sys_output_dict[_PGA], component_resp_dict[_PGA] = \
                multiprocess_enabling_loop(idxPGA=idxPGA, _PGA_dummy=_PGA, nPGA=num_hazard_pts)

    # saving for test cases
    # cPickle.dump(ids_comp_vs_haz, open('tests/ids_comp_vs_haz.pick', 'wb'))
    # cPickle.dump(sys_output_dict, open('tests/sys_output_dict.pick', 'wb'))
    return ids_comp_vs_haz, sys_output_dict, component_resp_dict

if __name__ == "__main__":
    SETUPFILE = sys.argv[1]
    discard = {}
    config = {}
    execfile(SETUPFILE, discard, config)
    sc = Scenario(SETUPFILE)
    fc = Facility(SETUPFILE)

    # Define input files, output location, scenario inputs
    INPUT_PATH = os.path.join(os.getcwd(), sc.input_dir_name)
    SYS_CONFIG_FILE = os.path.join(INPUT_PATH, fc.sys_config_file_name)

    if not os.path.exists(sc.output_dir_name):
        os.makedirs(sc.output_dir_name)

    OUTPUT_PATH = os.path.join(os.getcwd(), sc.output_dir_name)
    RAW_OUTPUT_DIR = os.path.join(os.getcwd(), sc.output_dir_name, 'raw_output')

    if not os.path.exists(RAW_OUTPUT_DIR):
        os.makedirs(RAW_OUTPUT_DIR)

    # Read in INPUT data files
    comp_df, fragilities, sysinp_setup, sysout_setup, node_conn_df = fc.assign_infrastructure_data()

    cp_types_in_system, cp_types_in_db = check_types_with_db()
    uncosted_comptypes = ['CONN_NODE', 'SYSTEM_INPUT', 'SYSTEM_OUTPUT']
    costed_comptypes, comps_costed = list_of_components_for_cost_calculation(cp_types_in_system, uncosted_comptypes)
    nominal_production = sysout_setup['Capacity'].sum()
    hazard_transfer_label = sc.hazard_transfer_param + ' (' + sc.hazard_transfer_unit+ ')'

    comp_dict = comp_df.to_dict()
    cpdict, output_dict, input_dict, nodes_by_commoditytype = convert_df_to_dict()

    fragdict = fragility_dict()

    restoration_time_range, dmg_states, restoration_chkpoints, restoration_pct_steps, hazard_intensity_vals, \
           num_hazard_pts, num_time_steps = simulation_parameters()
    nodes, num_elements, G = network()
    sup_node_list, dep_node_list, src_node_list, out_node_list = network_setup()

    nodes_all = sorted(comp_df.index)
    no_elements = len(nodes_all)

    PGA_str, calculated_output_array, component_resp_df, economic_loss_array, comp_loss_array, comp_loss_dict, \
           output_array_given_recovery, sys_output_dict, sys_output_list_given_pga, \
            comp_dsix_given_pga, ids_comp_vs_haz = power_calc()

    ids_comp_vs_haz, sys_output_dict, component_resp_dict = calc_loss_arrays(parallel_or_serial=1)
