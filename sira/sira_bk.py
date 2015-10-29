'''
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
'''

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

import systemlayout
import siraplot as spl
SETUPFILE = None


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

np.random.seed(100)  # add seed for reproducibility

def plot_comp_frag(fragilities, costed_comptypes, hazard_levels, output_path):
    '''
    Generates fragility plots of given component types, and
    saves them in <output_path>
    '''
    dsclrs = ["#2ecc71", "#3498db", "#feb24c", "#de2d26"]
    grid_colr = '#B6B6B6'

    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_palette(dsclrs)

    fragpath = os.path.join(output_path, "component_fragilities")
    if not os.path.exists(fragpath):
        os.makedirs(fragpath)
    fragfile = os.path.join(fragpath, 'component_fragility_algorithms.csv')
    frdf = copy.deepcopy(fragilities)
    frdf = frdf.query("damage_state!='DS0 None'")

    costed_comptypes = sorted(costed_comptypes)

    frdf.rename(columns={'damage_function': 'Function',
                         'damage_median': 'Fragility Median',
                         'damage_logstd': 'Fragility Beta',
                         'functionality': 'Functionality',
                         'recovery_mean': 'Restoration Mean',
                         'recovery_std': 'Restoration StdDev',
                         'ds_definition': 'Damage State Def',
                         'fragility_source': 'Source'}, inplace=True)

    req_cols = ['Fragility Median', 'Fragility Beta', 'Functionality',
                'Restoration Mean', 'Restoration StdDev', 'Damage State Def']

    print("\nSaving component fragility data for ... ")
    for i, x in enumerate(costed_comptypes):
        ds_list = list(fragilities.index.levels[1].values)
        ds_list.remove('DS0 None')

        print("{0:>7s} {1:<s}".format('['+str(i+1)+']', x))
        with open(fragfile, 'ab') as f:
            ds = ds_list[0]
            fwriter = csv.writer(f, delimiter=',',
                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
            fwriter.writerow(["Component Type", x])
            fwriter.writerow(
                ["Source", frdf.loc[(x, ds), 'Source']
                 if type(frdf.loc[(x, ds), 'Source']) == str
                 else '<Unavailable>'])
            fwriter.writerow(["Function", frdf.loc[(x, ds), 'Function']])
            fwriter.writerow(' ')
            df = frdf.loc[(x,), req_cols].T
            df.to_csv(f, mode='a')
            fwriter.writerow(' ')

        fig = plt.figure(figsize=(6.0, 3.2))
        ax = fig.add_subplot(111)
        for ds in ds_list:
            m = fragilities.loc[(x, ds), 'damage_median']
            b = fragilities.loc[(x, ds), 'damage_logstd']
            plt.plot(
                hazard_levels, stats.lognorm.cdf(hazard_levels, b, scale=m),
                clip_on=False)

        plt.grid(True, which='major', axis='both', linestyle='-',
                 linewidth=0.5, color=grid_colr)
        plt.title(x, loc='center', y=1.02, size=8)

        plt.xlabel('PGA (g)', size=7, labelpad=8)
        plt.ylabel('Probability of Exceedence', size=7, labelpad=8)
        plt.yticks(np.linspace(0.0, 1.0, 11))

        ax.tick_params(
            axis='both',
            which='both',
            left='off',
            right='off',
            labelleft='on',
            labelright='off',
            direction='out',
            labelsize=6,
            pad=5)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax.legend(ds_list, loc='upper left', ncol=1,
                  bbox_to_anchor=(1.02, 1.0),
                  frameon=0, prop={'size': 6})

        sns.set_context('paper')
        s = re.sub(r"/|\\|\*", " ", x)
        plt.savefig(os.path.join(fragpath, s+'.png'),
                    format='png', bbox_inches='tight', dpi=300)
        plt.close()


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
            minpos = min(range(len(PGA_levels)),
                         key=lambda i: abs(PGA_levels[i] - lower_lim))
            zl = [0.0] * (minpos + 1)
            ol = [1] * (len(PGA_levels) - (minpos + 1))
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
                        w2 * stats.lognorm.cdf(PGA, s2, loc=0.0, scale=m)
                       ) * stepv
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return np.sort(pe_ds)


def pe2pb(pe):
    '''
    Convert probablity of exceedence of damage states, to
    probability of being in each discrete damage state
    '''
    # sorted array
    pex = np.sort(pe)[::-1]
    tmp = -1.0*np.diff(pex)
    pb = np.append(tmp, pex[-1])
    pb = np.insert(pb, 0, 1 - pex[0])
    return pb


def calc_recov_time_given_comp_ds(comp, ids):
    '''
    Calculates the recovery time of a component, given damage state index
    '''
    ds = dmg_states[ids]
    ct = compdict['component_type'][comp]
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


def multiprocess_enabling_loop(idxPGA, _PGA_dummy, nPGA):

    if isinstance(_PGA_dummy, list):
        _PGA = _PGA_dummy[idxPGA]
    else:
        _PGA = _PGA_dummy
    print(" {0:3d}  out of {1:3d}".format(idxPGA+1, nPGA))

    # compute pe and determine ds for each component
    ids_comp = np.zeros((num_samples, no_elements), dtype=int)

    # index of damage state of components: from 0 to nds+1
    for j, comp in enumerate(nodes_all):
        ids_comp[:, j] = np.sum(
            cal_pe_ds(comp, float(_PGA), compdict, fragdict)
            > rnd[:, j][:, np.newaxis], axis=1
            )

        # comp_loss_dict[comp] = np.zeros((num_samples,nPGA))

    component_loss_tmp = {c: [] for c in nodes_all}
    component_func_tmp = {c: [] for c in nodes_all}

    # system output and economic loss
    for i in range(num_samples):
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
        print('\n===================>>>>>multiprocessor computation on <<<<========================')
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


if not SETUPFILE:  # used for running test case
    SETUPFILE = 'tests/config_ps_X_test.conf'
    print ('using default test setupfile')

discard = {}
config = {}
execfile(SETUPFILE, discard, config)

SYSTEM_CLASSES = config["SYSTEM_CLASSES"]
SYSTEM_CLASS = config["SYSTEM_CLASS"]
COMMODITY_FLOW_TYPES = config["COMMODITY_FLOW_TYPES"]

pga_min = config["PGA_MIN"]
pga_max = config["PGA_MAX"]
pga_step = config["PGA_STEP"]
num_samples = config["NUM_SAMPLES"]

hazard_transfer_param = config["HAZARD_TRANSFER_PARAM"]
hazard_transfer_unit = config["HAZARD_TRANSFER_UNIT"]

timeunit = config["TIME_UNIT"]
restore_time_step = config["RESTORE_TIME_UPPER"]
restore_pct_chkpoints = config["RESTORE_PCT_CHKPOINTS"]
restore_time_upper = config["RESTORE_TIME_STEP"]
restore_time_max = config["RESTORE_TIME_MAX"]

input_dir_name = config["INPUT_DIR_NAME"]
output_dir_name = config["OUTPUT_DIR_NAME"]

ifile_name_sys_conf = config["SYS_CONF_FILE_NAME"]

USE_ENDPOINT = config["USE_ENDPOINT"]
FIT_PE_DATA = config["FIT_PE_DATA"]
FIT_RESTORATION_DATA = config["FIT_RESTORATION_DATA"]
SAVE_VARS_NPY = config["SAVE_VARS_NPY"]

# Multiprocess or not
PARALLEL = config['MULTIPROCESS']
print(PARALLEL)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Define input files, output location, scenario inputs

input_path = os.path.join(os.getcwd(), input_dir_name)
ifile_sys_config = os.path.join(input_path, ifile_name_sys_conf)

if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)
output_path = os.path.join(os.getcwd(), output_dir_name)

raw_output_dir = os.path.join(os.getcwd(), output_dir_name, 'raw_output')
if not os.path.exists(raw_output_dir):
    os.makedirs(raw_output_dir)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Read in INPUT data files

ndf = pd.read_excel(ifile_sys_config, 'component_connections',
                    index_col=None,
                    skiprows=3, skipinitialspace=True)

comp_df = pd.read_excel(
    ifile_sys_config, 'comp_list',
    index_col='component_id',
    skiprows=3, skipinitialspace=True)

sysout_setup = pd.read_excel(
    ifile_sys_config, 'output_setup',
    index_col='OutputNode',
    skiprows=3, skipinitialspace=True)
sysout_setup = sysout_setup.sort('Priority', ascending=True)
                    # ^ Prioritised list of output nodes

sysinp_setup = pd.read_excel(
    ifile_sys_config, 'supply_setup',
    index_col='InputNode',
    skiprows=3, skipinitialspace=True)

fragilities = pd.read_excel(
    ifile_sys_config, 'comp_type_dmg_algo',
    index_col=['component_type', 'damage_state'],
    skiprows=3, skipinitialspace=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# check to ensure component types match with DB
cp_types_in_system = list(np.unique(comp_df['component_type'].tolist()))
cp_types_in_db = list(fragilities.index.levels[0])
assert set(cp_types_in_system).issubset(cp_types_in_db) == True
# assert if set(cp_types_in_system).issubset(cp_types_in_db) is True

# get list of only those components that are included in cost calculations
uncosted_comptypes = ['CONN_NODE', 'SYSTEM_INPUT', 'SYSTEM_OUTPUT']
cp_types_costed = [x for x in cp_types_in_system
                   if x not in uncosted_comptypes]
costed_comptypes = list(set(cp_types_in_system) - set(uncosted_comptypes))

cpmap = {c: sorted(comp_df[comp_df['component_type'] == c].index.tolist())
         for c in cp_types_in_system}
comps_costed = [v for x in cp_types_costed for v in cpmap[x]]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

nominal_production = sysout_setup['Capacity'].sum()
hazard_transfer_label = hazard_transfer_param+' ('+hazard_transfer_unit+')'

# -----------------------------------------------------------------------------
# Convert Dataframes to Dicts for lookup efficiency
# -----------------------------------------------------------------------------

compdict = comp_df.to_dict()
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

# -----------------------------------------------------------------------------
# Simulation Parameters
# -----------------------------------------------------------------------------

dmg_states = sorted([str(d) for d in fragilities.index.levels[1]])

max_recoverytimes_dict = {}
for x in cp_types_in_system:
    max_recoverytimes_dict[x] =\
        fragilities.ix[x, dmg_states[len(dmg_states) - 1]]['recovery_mean']

restoration_time_range, time_step =\
    np.linspace(0, restore_time_upper, num=restore_time_upper+1,
                endpoint=USE_ENDPOINT, retstep=True)

restoration_chkpoints, restoration_pct_steps =\
    np.linspace(0.0, 1.0, restore_pct_chkpoints, retstep=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

hazard_data_points = int(round((pga_max - pga_min) / float(pga_step) + 1))

PGA_levels = np.linspace(pga_min, pga_max,
                         num=hazard_data_points)

nPGA = len(PGA_levels)
num_time_steps = len(restoration_time_range)

# -----------------------------------------------------------------------------
# Define the system as a network, with components as nodes
# -----------------------------------------------------------------------------

nodes_all = sorted(comp_df.index)
no_elements = len(nodes_all)

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

for index, row in ndf.iterrows():
    G.add_edge(row['Orig'], row['Dest'],
               capacity=G.vs.find(row['Orig'])["capacity"],
               weight=row['Weight'],
               distance=row['Distance'])

#                    --------
# Network setup with NetworkX (for drawing graph)
#                    --------
X = nx.DiGraph()
for index, row in ndf.iterrows():
    X.add_edge(row['Orig'], row['Dest'],
               capacity=row['Capacity'],
               weight=row['Weight'],
               distance=row['Distance'])

systemlayout.draw_sys_layout(X, comp_df, out_dir=output_path,
                             graph_label="System Component Layout")

# -----------------------------------------------------------------------------
# List of tagged nodes with special roles:
sup_node_list = [str(k) for k in
                 list(comp_df.ix[comp_df['node_type'] == 'supply'].index)]
dep_node_list = [str(k) for k in
                 list(comp_df.ix[comp_df['node_type'] == 'dependency'].index)]
src_node_list = [k for (k, v)in X.in_degree().iteritems() if v == 0]
out_node_list = list(sysout_setup.index.get_level_values('OutputNode'))

# -----------------------------------------------------------------------------
# Power output and economic loss calculations
# -----------------------------------------------------------------------------

PGA_str = [('%0.3f' % np.float(x)) for x in PGA_levels]

cptype = {}
cptype_ds_edges = {}
for comp in nodes_all:
    cptype[comp] = compdict['component_type'][comp]
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


###############################################################################
# simulation of damage of each component

calculated_output_array = np.zeros((num_samples, nPGA))
economic_loss_array = np.zeros_like(calculated_output_array)

comp_loss_array = np.zeros((num_samples, nPGA))
comp_loss_dict = {c: np.zeros((num_samples, nPGA)) for c in nodes_all}

# Record output for:
# <samples> vs <hazard parameter index> vs <time step index>
output_array_given_recovery = np.zeros((num_samples, nPGA, num_time_steps))

rnd = stats.uniform.rvs(loc=0, scale=1, size=(num_samples, no_elements))
np.save(os.path.join(raw_output_dir, 'rnd_samples_x_elements.npy'), rnd)



# List of output values at output_nodes:
sys_output_list_given_pga = {k: np.zeros((num_samples, len(out_node_list)))
                             for k in PGA_str}

comp_dsix_given_pga = {k: np.zeros((num_samples, len(nodes_all)))
                       for k in PGA_str}

#################################################################################################
######################## monte carlo compuration ################################################
#################################################################################################

ids_comp_vs_haz, sys_output_dict, component_resp_dict = calc_loss_arrays(parallel_or_serial=PARALLEL)

idshaz = os.path.join(raw_output_dir, 'ids_comp_vs_haz.pickle')
with open(idshaz, 'w') as handle:
    pickle.dump(ids_comp_vs_haz, handle)

# -----------------------------------------------------------------------------
# System output file (for given hazard transfer parameter value)
# -----------------------------------------------------------------------------

sys_output_df = pd.DataFrame(sys_output_dict)
sys_output_df.index.name = 'Output Nodes'

outfile_sysoutput = os.path.join(output_path, 'system_output_vs_hazparam.csv')
sys_output_df.to_csv(outfile_sysoutput, sep=',', index_label=['Output Nodes'])

# -----------------------------------------------------------------------------
# Loss calculations by Component Type
# -----------------------------------------------------------------------------
tp_ct = []
for x in cp_types_costed:
    tp_ct.extend(
        ((x, 'loss_mean'), (x, 'loss_std'), (x, 'loss_tot'),
         (x, 'func_mean'), (x, 'func_std'))
    )

mindex = pd.MultiIndex.from_tuples(tp_ct, names=['component_type', 'response'])
comptype_resp_df = pd.DataFrame(index=mindex, columns=[PGA_str])
comptype_resp_dict = comptype_resp_df.to_dict()

for p in PGA_str:
    for ct in cp_types_costed:

        comptype_resp_dict[p][(ct, 'loss_mean')] =\
            np.mean([component_resp_dict[p][(cn, 'loss_mean')]
                     for cn in cpmap[ct]])

        comptype_resp_dict[p][(ct, 'loss_tot')] =\
            np.sum([component_resp_dict[p][(cn, 'loss_mean')]
                    for cn in cpmap[ct]])

        comptype_resp_dict[p][(ct, 'loss_std')] =\
            np.mean([component_resp_dict[p][(cn, 'loss_std')]
                     for cn in cpmap[ct]])

        comptype_resp_dict[p][(ct, 'func_mean')] =\
            np.mean([component_resp_dict[p][(cn, 'func_mean')]
                     for cn in cpmap[ct]])

        comptype_resp_dict[p][(ct, 'func_std')] =\
            np.mean([component_resp_dict[p][(cn, 'func_std')]
                     for cn in cpmap[ct]])

        comptype_resp_dict[p][(ct, 'num_failures')] =\
            np.mean([component_resp_dict[p][(cn, 'num_failures')]
                     for cn in cpmap[ct]])


# -----------------------------------------------------------------------------
# System Fragility & Probability of Exceedence
# -----------------------------------------------------------------------------
#
#   Damage state boundaries for Economic Loss calculations
#
#   None: < 0.01,
#   Slight: 0.01 to 0.15,
#   Moderate: 0.15 to 0.4,
#   Extensive: 0.4 to 0.8,
#   Complete: 0.8 to 1.0
#
# -----------------------------------------------------------------------------

sys_dmg_states = ['DS0 None',
                  'DS1 Slight',
                  'DS2 Moderate',
                  'DS3 Extensive',
                  'DS4 Complete']

ds_bounds = [0.01, 0.15, 0.4, 0.8, 1.0]
# ds_bounds = [0.04, 0.30, 0.75, 0.99, 1.10]

# --- System fragility ---
sys_frag = np.zeros_like(economic_loss_array, dtype=int)
for j in range(nPGA):
    for i in range(num_samples):
        # system output and economic loss
        sys_frag[i, j] = np.sum(economic_loss_array[i, j] > ds_bounds)

# --- Probability of Exceedence ---
pe_sys_econloss = np.zeros((len(sys_dmg_states), nPGA))
for j in range(nPGA):
    for i in range(len(sys_dmg_states)):
        pe_sys_econloss[i, j] = np.sum(sys_frag[:, j] >= i)/float(num_samples)

# -----------------------------------------------------------------------------
# For Probability of Exceedence calculations based on component failures
# -----------------------------------------------------------------------------
#
#   Damage state boundaries for Component Type Failures (Substations) are
#   based on HAZUS MH MR3, p 8-66 to 8-68
#
# -----------------------------------------------------------------------------

cp_classes_in_system = list(np.unique(comp_df['component_class'].tolist()))
cp_class_map = {k: [] for k in cp_classes_in_system}
for k, v in compdict['component_class'].iteritems():
    cp_class_map[v].append(k)

if SYSTEM_CLASS == 'Substation':
    uncosted_classes = ['JUNCTION POINT', 'SYSTEM INPUT', 'SYSTEM OUTPUT',
                        'Generator', 'Bus', 'Lightning Arrester']
    ds_lims_compclasses = {
        'Disconnect Switch':   [0.05, 0.40, 0.70, 0.99, 1.00],
        'Circuit Breaker':     [0.05, 0.40, 0.70, 0.99, 1.00],
        'Current Transformer': [0.05, 0.40, 0.70, 0.99, 1.00],
        'Voltage Transformer': [0.05, 0.40, 0.70, 0.99, 1.00],
        'Power Transformer':   [0.05, 0.40, 0.70, 0.99, 1.00],
        'Control Building':    [0.06, 0.30, 0.75, 0.99, 1.00]
    }

    cp_classes_costed = \
        [x for x in cp_classes_in_system if x not in uncosted_classes]

    # --- System fragility - Based on Failure of Component Classes ---
    comp_class_failures = {cc: np.zeros((num_samples, nPGA))
                           for cc in cp_classes_costed}
    comp_class_frag = {cc: np.zeros((num_samples, nPGA))
                       for cc in cp_classes_costed}
    for j, PGA in enumerate(PGA_str):
        for i in range(num_samples):
            for compclass in cp_classes_costed:
                for c in cp_class_map[compclass]:
                    comp_class_failures[compclass][i, j] += \
                        ids_comp_vs_haz[PGA][i, nodes_all.index(c)]
                comp_class_failures[compclass][i, j] /= \
                    len(cp_class_map[compclass])
                comp_class_frag[compclass][i, j] = \
                    np.sum(comp_class_failures[compclass][i, j]
                           > ds_lims_compclasses[compclass])

    # --- Probability of Exceedence - Based on Failure of Component Classes ---
    pe_sys_cpfailrate = np.zeros((len(sys_dmg_states), nPGA))
    for p in range(nPGA):
        for d in range(len(sys_dmg_states)):
            ds_ss_ix = []
            for compclass in cp_classes_costed:
                ds_ss_ix.append(np.sum(comp_class_frag[compclass][:, p] >= d)
                                / float(num_samples))
            pe_sys_cpfailrate[d, p] = np.median(ds_ss_ix)

    # --- Save prob exceedance data as npy ---
    np.save(os.path.join(raw_output_dir, 'pe_sys_cpfailrate.npy'),
            pe_sys_cpfailrate)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if SYSTEM_CLASS == 'Power Station (Coal Fired)':
    uncosted_classes = ['JUNCTION POINT', 'SYSTEM INPUT', 'SYSTEM OUTPUT']
    ds_lims_compclasses = {
        'Boiler':                       [0.0, 0.05, 0.40, 0.70, 1.00],
        'Control Building':             [0.0, 0.05, 0.40, 0.70, 1.00],
        'Emission Management':          [0.0, 0.05, 0.40, 0.70, 1.00],
        'Fuel Delivery and Storage':    [0.0, 0.05, 0.40, 0.70, 1.00],
        'Fuel Movement':                [0.0, 0.05, 0.40, 0.70, 1.00],
        'Generator':                    [0.0, 0.05, 0.40, 0.70, 1.00],
        'SYSTEM OUTPUT':                [0.0, 0.05, 0.40, 0.70, 1.00],
        'Stepup Transformer':           [0.0, 0.05, 0.40, 0.70, 1.00],
        'Turbine':                      [0.0, 0.05, 0.40, 0.70, 1.00],
        'Water System':                 [0.0, 0.05, 0.40, 0.70, 1.00]
    }

    cp_classes_costed = \
        [x for x in cp_classes_in_system if x not in uncosted_classes]

# -----------------------------------------------------------------------------
# Validate damage ratio of the system

exp_damage_ratio = np.zeros((len(nodes_all), nPGA))
for l, PGA in enumerate(PGA_levels):
    # compute expected damage ratio
    for j, comp_name in enumerate(nodes_all):
        pb = pe2pb(cal_pe_ds(comp_name, PGA, compdict, fragdict))
        comp_type = compdict['component_type'][comp_name]
        dr = np.array([fragdict['damage_ratio'][comp_type][ds]
                       for ds in dmg_states])
        cf = compdict['cost_fraction'][comp_name]
        loss_list = dr * cf
        exp_damage_ratio[j, l] = np.sum(pb * loss_list)

# -----------------------------------------------------------------------------
# Time to Restoration of Full Capacity
# -----------------------------------------------------------------------------

threshold = 0.99
required_time = []

for j in range(nPGA):
    cpower = \
        np.mean(output_array_given_recovery[:, j, :], axis=0) /\
        nominal_production
    temp = cpower > threshold
    if sum(temp) > 0:
        required_time.append(np.min(restoration_time_range[temp]))
    else:
        required_time.append(restore_time_max)

# -----------------------------------------------------------------------------
# *** Saving vars ***
# -----------------------------------------------------------------------------

if SAVE_VARS_NPY:
    np.save(os.path.join(raw_output_dir, 'economic_loss_array.npy'),
            economic_loss_array)
    np.save(os.path.join(raw_output_dir, 'calculated_output_array.npy'),
            calculated_output_array)
    np.save(os.path.join(raw_output_dir, 'output_array_given_recovery.npy'),
            output_array_given_recovery)
    np.save(os.path.join(raw_output_dir, 'exp_damage_ratio.npy'),
            exp_damage_ratio)
    np.save(os.path.join(raw_output_dir, 'sys_frag.npy'),
            sys_frag)
    np.save(os.path.join(raw_output_dir, 'required_time.npy'),
            required_time)
    np.save(os.path.join(raw_output_dir, 'pe_sys_econloss.npy'),
            pe_sys_econloss)

print("\nOutputs saved in: " + Fore.GREEN + output_path)