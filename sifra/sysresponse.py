#!/usr/bin/env python

"""
sifra.py

System for Infrastructure Facility Resilience Analysis
A tool for seismic performance analysis of infrastructure facilities

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

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sifraclasses import *

import os
import sys
import cPickle
import zipfile

import numpy as np
import scipy.stats as stats
import pandas as pd
import parmap

import seaborn as sns
from colorama import Fore

SETUPFILE = None

# ============================================================================


def cal_pe_ds(comp, PGA, compdict, fragdict, sc):
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
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            lower_lim = fragdict['minimum'][ct][ds]
            minpos = min(
                range(len(sc.hazard_intensity_vals)),
                key=lambda i: abs(sc.hazard_intensity_vals[i] - lower_lim)
            )
            zl = [0.0] * (minpos + 1)
            ol = [1] * (len(sc.hazard_intensity_vals) - (minpos + 1))
            stepfn = zl + ol
            stepv = stepfn[minpos]
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            m = 0.25
            s1 = np.exp(fragdict['sigma_1'][ct][ds])
            s2 = np.exp(fragdict['sigma_2'][ct][ds])
            w1 = 0.5
            w2 = 0.5
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            pe_ds[i] = (
                w1 * stats.lognorm.cdf(PGA, s1, loc=0.0, scale=m) +
                w2 * stats.lognorm.cdf(PGA, s2, loc=0.0, scale=m)
            ) * stepv
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return np.sort(pe_ds)

# ============================================================================


def calc_recov_time_given_comp_ds(comp, ids, comp_dict, fragdict, fc, sc):
    '''
    Calculates the recovery time of a component, given damage state index
    '''
    ds = fc.sys_dmg_states[ids]
    ct = comp_dict['component_type'][comp]
    m = fragdict['recovery_mean'][ct][ds]
    s = fragdict['recovery_std'][ct][ds]
    fn = fragdict['functionality'][ct][ds]
    cdf = stats.norm.cdf(sc.restoration_time_range, loc=m, scale=s)
    return cdf + (1.0 - cdf) * fn

# ============================================================================


def compute_output_given_ds(cp_func, fc):
    '''
    Computes system output given list of component functional status
    '''
    G = fc.network.G
    nodes = fc.network.nodes_all

    for t in G.get_edgelist():
        eid = G.get_eid(*t)
        origin = G.vs[t[0]]['name']
        destin = G.vs[t[1]]['name']
        if fc.cpdict[origin]['node_type'] == 'dependency':
            cp_func[nodes.index(destin)] *= cp_func[nodes.index(origin)]
        cap = cp_func[nodes.index(origin)]
        G.es[eid]["capacity"] = cap

    sys_out_capacity_list = []  # normalised capacity: [0.0, 1.0]

    for onode in fc.network.out_node_list:
        for sup_node_list in fc.nodes_by_commoditytype.values():
            total_available_flow_list = []
            avl_sys_flow_by_src = []
            for inode in sup_node_list:
                avl_sys_flow_by_src.append(
                    G.maxflow_value(G.vs.find(inode).index,
                                    G.vs.find(onode).index, G.es["capacity"])
                    * fc.input_dict[inode]['capacity_fraction']
                )

            total_available_flow_list.append(sum(avl_sys_flow_by_src))

        total_available_flow = min(total_available_flow_list)
        sys_out_capacity_list.append(
            min(total_available_flow,
                fc.output_dict[onode]['capacity_fraction'])
            * fc.nominal_production
        )

    return sys_out_capacity_list

# ============================================================================


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

# ============================================================================


def calc_sys_output(fc, sc):
    """
    Power output and economic loss calculations for each component
    :param fc: Facility object, from sifraclasses
    :param sc: Scenario object, from sifraclasses
    :return:
        component_resp_df
        pandas dataframe with following values for each system component
        - mean of simulated loss
        - standard deviation of simulated loss
        - mean of functionality
        - standard deviation of functionality
    """
    cptype = {}
    cptype_ds_edges = {}
    comp_dict = fc.comp_df.to_dict()
    fragdict = fc.fragdict
    for comp in sorted(fc.comp_df.index):
        cptype[comp] = comp_dict['component_type'][comp]
        cptype_ds_edges[cptype[comp]] =\
            sorted(fragdict['damage_ratio'][cptype[comp]].values())

    tp_cp = []
    for x in fc.comps_costed:
        tp_cp.extend((
            (x, 'loss_mean'),
            (x, 'loss_std'),
            (x, 'func_mean'),
            (x, 'func_std')))
    mindex = pd.MultiIndex.from_tuples(tp_cp,
                                       names=['component_id', 'response'])
    component_resp_df = pd.DataFrame(
        index=mindex, columns=[sc.hazard_intensity_str]
    )

    return component_resp_df

# ============================================================================


def multiprocess_enabling_loop(idxPGA, _PGA_dummy, nPGA, fc, sc):
    if isinstance(_PGA_dummy, list):
        _PGA = _PGA_dummy[idxPGA]
    else:
        _PGA = _PGA_dummy
    print(" {0:3d}  out of {1:3d}".format(idxPGA+1, nPGA))

    comp_dict = fc.compdict
    fragdict = fc.fragdict

    ##########################################################################
    # simulation of damage of each component

    calculated_output_array_single = np.zeros(sc.num_samples)
    economic_loss_array_single = np.zeros_like(calculated_output_array_single)

    nodes_sorted = sorted(fc.network.nodes_all)
    comp_loss_dict = {c: np.zeros((sc.num_samples, sc.num_hazard_pts))
                      for c in nodes_sorted}

    # Record output for:
    # <samples> vs <hazard parameter index> vs <time step index>
    output_array_given_recovery = np.zeros(
        (sc.num_samples, sc.num_time_steps)
    )

    # rnd = stats.uniform.rvs(
    #     loc=0, scale=1, size=(NUM_SAMPLES, num_elements))
    # np.save(os.path.join(RAW_OUTPUT_DIR, 'rnd_samples_x_elements.npy'), rnd)

    # List of output values at output_nodes:
    sys_output_list_given_pga = {
        k: np.zeros((sc.num_samples, len(fc.network.out_node_list)))
        for k in sc.hazard_intensity_str
        }

    # compute pe and determine ds for each component
    ids_comp = np.zeros((sc.num_samples, fc.num_elements), dtype=int)

    # index of damage state of components: from 0 to nds+1
    if sc.run_context:  # test run
        prng = np.random.RandomState(idxPGA)
    else:
        prng = np.random.RandomState()

    rnd = prng.uniform(size=(sc.num_samples, fc.num_elements))
    # index of damage state of components: from 0 to nds+1
    for j, comp in enumerate(nodes_sorted):
        ids_comp[:, j] = np.sum(
            cal_pe_ds(comp, float(_PGA), comp_dict, fragdict, sc) >
            rnd[:, j][:, np.newaxis], axis=1
        )
        # comp_loss_dict[comp] = np.zeros((num_samples,nPGA))

    component_loss_tmp = {c: [] for c in nodes_sorted}
    component_func_tmp = {c: [] for c in nodes_sorted}

    # system output and economic loss
    for i in range(sc.num_samples):
        loss_list_all_comp = []
        cp_func = []
        cp_func_given_time = []

        for j, comp_name in enumerate(nodes_sorted):
            # ................................................................
            comp_type = comp_dict['component_type'][comp_name]
            ids = ids_comp[i, j]   # index for component damage state
            ds = fc.sys_dmg_states[ids]   # damage state name
            cf = comp_dict['cost_fraction'][comp_name]
            dr = fragdict['damage_ratio'][comp_type][ds]
            fn = fragdict['functionality'][comp_type][ds]
            loss = dr * cf
            loss_list_all_comp.append(loss)

            # ................................................................
            # component functionality for calculated damage state:
            cp_func.append(fn)
            cp_func_given_time.append(
                calc_recov_time_given_comp_ds(
                    comp_name, ids, comp_dict, fragdict, fc, sc
                )
            )

            comp_loss_dict[comp_name][i, idxPGA] = loss
            component_loss_tmp[comp_name].append(loss)
            component_func_tmp[comp_name].append(fn)
            # ................................................................

        economic_loss_array_single[i] = sum(loss_list_all_comp)

        outputlist = compute_output_given_ds(cp_func, fc)
        calculated_output_array_single[i] = sum(outputlist)

        sys_output_list_given_pga[_PGA][i, :] = outputlist

        # restoration status of components over the range of time
        # (num elements X num specified time units)
        cp_func_given_time = np.array(cp_func_given_time)
        for t in range(sc.num_time_steps):
            output_array_given_recovery[i, t] = \
                sum(compute_output_given_ds(cp_func_given_time[:, t], fc))

    comp_resp_dict = dict()

    for j, comp_name in enumerate(nodes_sorted):
        comp_resp_dict[(comp_name, 'loss_mean')]\
            = np.mean(component_loss_tmp[comp_name])

        comp_resp_dict[(comp_name, 'loss_std')]\
            = np.std(component_loss_tmp[comp_name])

        comp_resp_dict[(comp_name, 'func_mean')]\
            = np.mean(component_func_tmp[comp_name])

        comp_resp_dict[(comp_name, 'func_std')]\
            = np.std(component_func_tmp[comp_name])

        comp_resp_dict[(comp_name, 'num_failures')]\
            = np.mean(ids_comp[:, j] >= (len(fc.sys_dmg_states) - 1))

    sys_out_dict = dict()
    for onx, onode in enumerate(fc.network.out_node_list):
        sys_out_dict[onode]\
            = np.mean(sys_output_list_given_pga[_PGA][:, onx])
    return ids_comp, \
           sys_out_dict, \
           comp_resp_dict, \
           calculated_output_array_single, \
           economic_loss_array_single, \
           output_array_given_recovery

# ============================================================================


def calc_loss_arrays(fc, sc, component_resp_df, parallel_proc):

    # print("\nCalculating system response to hazard transfer parameters...")
    component_resp_dict = component_resp_df.to_dict()
    sys_output_dict = {k: {o: 0 for o in fc.network.out_node_list}
                       for k in sc.hazard_intensity_str}
    ids_comp_vs_haz = {p: np.zeros((sc.num_samples, fc.num_elements))
                       for p in sc.hazard_intensity_str}

    calculated_output_array = np.zeros((sc.num_samples, sc.num_hazard_pts))
    economic_loss_array = np.zeros_like(calculated_output_array)
    output_array_given_recovery = np.zeros(
        (sc.num_samples, sc.num_hazard_pts, sc.num_time_steps)
    )

    if parallel_proc:
        print('\nInitiating computation of loss arrays...')
        print(Fore.YELLOW + 'using parallel processing\n' + Fore.RESET)
        parallel_return = parmap.map(
            multiprocess_enabling_loop, range(len(sc.hazard_intensity_str)),
            sc.hazard_intensity_str, sc.num_hazard_pts, fc, sc
        )

        for idxPGA, _PGA in enumerate(sc.hazard_intensity_str):
            ids_comp_vs_haz[_PGA] = parallel_return[idxPGA][0]
            sys_output_dict[_PGA] = parallel_return[idxPGA][1]
            component_resp_dict[_PGA] = parallel_return[idxPGA][2]
            calculated_output_array[:, idxPGA] = parallel_return[idxPGA][3]
            economic_loss_array[:, idxPGA] = parallel_return[idxPGA][4]
            output_array_given_recovery[:, idxPGA, :] = \
                parallel_return[idxPGA][5]
    else:
        print('\nInitiating computation of loss arrays...')
        print(Fore.RED + 'not using parallel processing\n' + Fore.RESET)
        for idxPGA, _PGA in enumerate(sc.hazard_intensity_str):
            ids_comp_vs_haz[_PGA], \
            sys_output_dict[_PGA], \
            component_resp_dict[_PGA], \
            calculated_output_array[:, idxPGA], \
            economic_loss_array[:, idxPGA], \
            output_array_given_recovery[:, idxPGA, :] = \
                multiprocess_enabling_loop(
                    idxPGA=idxPGA, _PGA_dummy=_PGA,
                    nPGA=sc.num_hazard_pts, fc=fc, sc=sc)

    return ids_comp_vs_haz, \
           sys_output_dict, \
           component_resp_dict, \
           calculated_output_array, \
           economic_loss_array, \
           output_array_given_recovery


# ****************************************************************************
# BEGIN POST-PROCESSING ...
# ****************************************************************************


def plot_mean_econ_loss(fc, sc, economic_loss_array):
    """Draws and saves a boxplot of mean economic loss"""

    fig = plt.figure(figsize=(9, 5), facecolor='white')
    sns.set(style='ticks', palette='Set3')
    # ax = sns.boxplot(economic_loss_array*100, showmeans=True,
    #                  widths=0.3, linewidth=0.7, color='lightgrey',
    #                  meanprops=dict(marker='s',
    #                                 markeredgecolor='salmon',
    #                                 markerfacecolor='salmon')
    #                 )
    ax = sns.boxplot(economic_loss_array * 100, showmeans=True,
                     linewidth=0.7, color='lightgrey',
                     meanprops=dict(marker='s',
                                    markeredgecolor='salmon',
                                    markerfacecolor='salmon')
                     )
    sns.despine(top=True, left=True, right=True)
    ax.tick_params(axis='y', left='off', right='off')
    ax.yaxis.grid(True)

    intensity_label = sc.intensity_measure_param+' ('\
                      +sc.intensity_measure_unit+')'
    ax.set_xlabel(intensity_label)
    ax.set_ylabel('Loss Fraction (%)')
    ax.set_xticklabels(sc.hazard_intensity_vals);
    ax.set_title('Loss Ratio', loc='center', y=1.04);
    ax.title.set_fontsize(12)

    figfile = os.path.join(sc.output_path, 'fig_lossratio_boxplot.png')
    plt.savefig(figfile, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def post_processing(fc, sc, ids_comp_vs_haz, sys_output_dict,
                    component_resp_dict, calculated_output_array,
                    economic_loss_array, output_array_given_recovery):

    # ------------------------------------------------------------------------
    # 'ids_comp_vs_haz' is a dict of numpy arrays
    # We pickle it for archival. But the file size can get very large.
    # So we zip it for archival and delete the original
    idshaz = os.path.join(sc.raw_output_dir, 'ids_comp_vs_haz.pickle')
    with open(idshaz, 'w') as handle:
        cPickle.dump(ids_comp_vs_haz, handle)

    idshaz_zip = os.path.join(sc.raw_output_dir, 'ids_comp_vs_haz.zip')
    zipmode = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(idshaz_zip, 'w', zipmode) as zip:
        zip.write(idshaz)
    os.remove(idshaz)

    # ------------------------------------------------------------------------
    # System output file (for given hazard transfer parameter value)
    # ------------------------------------------------------------------------

    sys_output_df = pd.DataFrame(sys_output_dict)
    sys_output_df.index.name = 'Output Nodes'

    outfile_sysoutput = os.path.join(sc.output_path,
                                     'system_output_given_haz_param.csv')
    sys_output_df.to_csv(outfile_sysoutput,
                         sep=',', index_label=['Output Nodes'])

    # ------------------------------------------------------------------------
    # Loss calculations by Component Type
    # ------------------------------------------------------------------------
    tp_ct = []
    for x in fc.cp_types_costed:
        tp_ct.extend(
            ((x, 'loss_mean'), (x, 'loss_std'), (x, 'loss_tot'),
             (x, 'func_mean'), (x, 'func_std'))
        )

    mindex = pd.MultiIndex.from_tuples(tp_ct,
                                       names=['component_type', 'response'])
    comptype_resp_df = pd.DataFrame(index=mindex,
                                    columns=[sc.hazard_intensity_str])
    comptype_resp_dict = comptype_resp_df.to_dict()

    for p in sc.hazard_intensity_str:
        for ct in fc.cp_types_costed:

            comptype_resp_dict[p][(ct, 'loss_mean')] =\
                np.mean([component_resp_dict[p][(cn, 'loss_mean')]
                         for cn in fc.cpmap[ct]])

            comptype_resp_dict[p][(ct, 'loss_tot')] =\
                np.sum([component_resp_dict[p][(cn, 'loss_mean')]
                        for cn in fc.cpmap[ct]])

            comptype_resp_dict[p][(ct, 'loss_std')] =\
                np.mean([component_resp_dict[p][(cn, 'loss_std')]
                         for cn in fc.cpmap[ct]])

            comptype_resp_dict[p][(ct, 'func_mean')] =\
                np.mean([component_resp_dict[p][(cn, 'func_mean')]
                         for cn in fc.cpmap[ct]])

            comptype_resp_dict[p][(ct, 'func_std')] =\
                np.mean([component_resp_dict[p][(cn, 'func_std')]
                         for cn in fc.cpmap[ct]])

            comptype_resp_dict[p][(ct, 'num_failures')] =\
                np.mean([component_resp_dict[p][(cn, 'num_failures')]
                         for cn in fc.cpmap[ct]])

    # ------------------------------------------------------------------------
    # Calculating system fragility:
    sys_frag = np.zeros_like(economic_loss_array, dtype=int)
    for j in range(sc.num_hazard_pts):
        for i in range(sc.num_samples):
            # system output and economic loss
            sys_frag[i, j] = np.sum(
                economic_loss_array[i, j] > fc.dmg_scale_bounds
            )

    # Calculating Probability of Exceedence:
    pe_sys_econloss = np.zeros((len(fc.sys_dmg_states), sc.num_hazard_pts))
    for j in range(sc.num_hazard_pts):
        for i in range(len(fc.sys_dmg_states)):
            pe_sys_econloss[i, j] = \
                np.sum(sys_frag[:, j] >= i)/float(sc.num_samples)

    # ------------------------------------------------------------------------
    # For Probability of Exceedence calculations based on component failures
    # ------------------------------------------------------------------------
    #
    #   Damage state boundaries for Component Type Failures (Substations) are
    #   based on HAZUS MH MR3, p 8-66 to 8-68
    #
    # ------------------------------------------------------------------------

    cp_classes_in_system = list(
        np.unique(fc.comp_df['component_class'].tolist())
    )
    cp_class_map = {k: [] for k in cp_classes_in_system}
    for k, v in fc.compdict['component_class'].iteritems():
        cp_class_map[v].append(k)

    # ------------------------------------------------------------------------

    if fc.system_class == 'Substation':
        uncosted_classes = ['JUNCTION POINT',
                            'SYSTEM INPUT', 'SYSTEM OUTPUT',
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
        comp_class_failures = \
            {cc: np.zeros((sc.num_samples, sc.num_hazard_pts))
             for cc in cp_classes_costed}

        comp_class_frag = {cc: np.zeros((sc.num_samples, sc.num_hazard_pts))
                           for cc in cp_classes_costed}
        for j, PGA in enumerate(sc.hazard_intensity_str):
            for i in range(sc.num_samples):
                for compclass in cp_classes_costed:
                    for c in cp_class_map[compclass]:
                        comp_class_failures[compclass][i, j] += \
                            ids_comp_vs_haz[PGA][
                                i, fc.network.nodes_all.index(c)
                            ]
                    comp_class_failures[compclass][i, j] /= \
                        len(cp_class_map[compclass])

                    comp_class_frag[compclass][i, j] = \
                        np.sum(comp_class_failures[compclass][i, j]
                               > ds_lims_compclasses[compclass])

        # Probability of Exceedence -- Based on Failure of Component Classes
        pe_sys_cpfailrate = np.zeros(
            (len(fc.sys_dmg_states), sc.num_hazard_pts)
        )
        for p in range(sc.num_hazard_pts):
            for d in range(len(fc.sys_dmg_states)):
                ds_ss_ix = []
                for compclass in cp_classes_costed:
                    ds_ss_ix.append(
                        np.sum(comp_class_frag[compclass][:, p] >= d) /
                        float(sc.num_samples)
                    )
                pe_sys_cpfailrate[d, p] = np.median(ds_ss_ix)

        # --- Save prob exceedance data as npy ---
        np.save(os.path.join(sc.raw_output_dir, 'pe_sys_cpfailrate.npy'),
                pe_sys_cpfailrate)

    # ------------------------------------------------------------------------

    if fc.system_class == 'PowerStation':
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

    # ------------------------------------------------------------------------
    # Validate damage ratio of the system
    # ------------------------------------------------------------------------

    exp_damage_ratio = np.zeros((len(fc.network.nodes_all),
                                 sc.num_hazard_pts))
    for l, PGA in enumerate(sc.hazard_intensity_vals):
        # compute expected damage ratio
        for j, comp_name in enumerate(fc.network.nodes_all):
            pb = pe2pb(
                cal_pe_ds(comp_name, PGA, fc.compdict, fc.fragdict, sc)
            )
            comp_type = fc.compdict['component_type'][comp_name]
            dr = np.array([fc.fragdict['damage_ratio'][comp_type][ds]
                           for ds in fc.sys_dmg_states])
            cf = fc.compdict['cost_fraction'][comp_name]
            loss_list = dr * cf
            exp_damage_ratio[j, l] = np.sum(pb * loss_list)

    # ------------------------------------------------------------------------
    # Time to Restoration of Full Capacity
    # ------------------------------------------------------------------------

    threshold = 0.99
    required_time = []

    for j in range(sc.num_hazard_pts):
        cpower = \
            np.mean(output_array_given_recovery[:, j, :], axis=0)/\
            fc.nominal_production
        temp = cpower > threshold
        if sum(temp) > 0:
            required_time.append(np.min(sc.restoration_time_range[temp]))
        else:
            required_time.append(sc.restore_time_max)

    # ------------------------------------------------------------------------
    # Write analytical outputs to file
    # ------------------------------------------------------------------------

    # --- Output File --- summary output ---
    outfile_sys_response = os.path.join(
        sc.output_path, 'system_response.csv')
    out_cols = ['PGA',
                'Economic Loss',
                'Mean Output',
                'Days to Full Recovery']
    outdat = {out_cols[0]: sc.hazard_intensity_vals,
              out_cols[1]: np.mean(economic_loss_array, axis=0),
              out_cols[2]: np.mean(calculated_output_array, axis=0),
              out_cols[3]: required_time}
    df = pd.DataFrame(outdat)
    df.to_csv(
        outfile_sys_response, sep=',',
        index=False, columns=out_cols
    )

    # --- Output File --- response of each COMPONENT to hazard ---
    outfile_comp_resp = os.path.join(
        sc.output_path, 'component_response.csv')
    component_resp_df = pd.DataFrame(component_resp_dict)
    component_resp_df.index.names = ['component_id', 'response']
    component_resp_df.to_csv(
        outfile_comp_resp, sep=',',
        index_label=['component_id', 'response']
    )

    # --- Output File --- mean loss of component ---
    outfile_comp_loss = os.path.join(
        sc.output_path, 'component_meanloss.csv')
    component_loss_df = component_resp_df.iloc[
        component_resp_df.index.get_level_values(1) == 'loss_mean']
    component_loss_df.reset_index(level='response', inplace=True)
    component_loss_df = component_loss_df.drop('response', axis=1)
    component_loss_df.to_csv(
        outfile_comp_loss, sep=',',
        index_label=['component_id']
    )

    # --- Output File --- response of each COMPONENT TYPE to hazard ---
    outfile_comptype_resp = os.path.join(
        sc.output_path, 'comp_type_response.csv')
    comptype_resp_df = pd.DataFrame(comptype_resp_dict)
    comptype_resp_df.index.names = ['component_type', 'response']
    comptype_resp_df.to_csv(
        outfile_comptype_resp, sep=',',
        index_label=['component_type', 'response']
    )

    # --- Output File --- mean loss of component type ---
    outfile_comptype_loss = os.path.join(
        sc.output_path, 'comp_type_meanloss.csv')
    comptype_loss_df = comptype_resp_df.iloc[
        comptype_resp_df.index.get_level_values(1) == 'loss_mean']
    comptype_loss_df.reset_index(level='response', inplace=True)
    comptype_loss_df = comptype_loss_df.drop('response', axis=1)
    comptype_loss_df.to_csv(
        outfile_comptype_loss, sep=',',
        index_label=['component_type']
    )

    # --- Output File --- mean failures for component types ---
    outfile_comptype_failures = os.path.join(
        sc.output_path, 'comp_type_meanfailures.csv')
    comptype_failure_df = comptype_resp_df.iloc[
        comptype_resp_df.index.get_level_values(1) == 'num_failures']
    comptype_failure_df.reset_index(level='response', inplace=True)
    comptype_failure_df = comptype_failure_df.drop('response', axis=1)
    comptype_failure_df.to_csv(
        outfile_comptype_failures, sep=',',
        index_label=['component_type']
    )

    # # --- Output File --- DataFrame of mean failures per component CLASS ---
    # outfile_compclass_failures = os.path.join(
    #     output_path, 'comp_class_meanfailures.csv')
    # compclass_failure_df.to_csv(outfile_compclass_failures, sep=',',
    #                         index_label=['component_class'])

    # ------------------------------------------------------------------------
    # *** Saving vars ***
    # ------------------------------------------------------------------------

    if sc.save_vars_npy:

        np.save(
            os.path.join(sc.raw_output_dir, 'economic_loss_array.npy'),
            economic_loss_array
        )

        np.save(
            os.path.join(sc.raw_output_dir, 'calculated_output_array.npy'),
            calculated_output_array
        )

        np.save(
            os.path.join(sc.raw_output_dir,
                         'output_array_given_recovery.npy'),
            output_array_given_recovery
        )

        np.save(
            os.path.join(sc.raw_output_dir, 'exp_damage_ratio.npy'),
            exp_damage_ratio
        )

        np.save(
            os.path.join(sc.raw_output_dir, 'sys_frag.npy'),
            sys_frag
        )

        np.save(
            os.path.join(sc.raw_output_dir, 'required_time.npy'),
            required_time
        )

        np.save(
            os.path.join(sc.raw_output_dir, 'pe_sys_econloss.npy'),
            pe_sys_econloss
        )
    # ------------------------------------------------------------------------
    print("\nOutputs saved in: " +
          Fore.GREEN + sc.output_path + Fore.RESET + '\n')

    plot_mean_econ_loss(fc, sc, economic_loss_array)

# ... END POST-PROCESSING
# ****************************************************************************

def main():

    SETUPFILE = sys.argv[1]
    discard = {}
    config = {}

    exec (open(SETUPFILE).read(), discard, config)
    FacilityObj = eval(config["SYSTEM_CLASS"])
    sc = Scenario(SETUPFILE)
    fc = FacilityObj(SETUPFILE)

    # Define input files, output location, scenario inputs
    SYS_CONFIG_FILE = os.path.join(sc.input_path, fc.sys_config_file_name)

    if not os.path.exists(sc.output_path):
        os.makedirs(sc.output_path)

    # Component Response DataFrame
    component_resp_df = calc_sys_output(fc, sc)

    ids_comp_vs_haz, sys_output_dict, \
    component_resp_dict, calculated_output_array, \
    economic_loss_array, output_array_given_recovery \
        = calc_loss_arrays(fc, sc,
                           component_resp_df,
                           parallel_proc=sc.run_parallel_proc)

    post_processing(fc, sc,
                    ids_comp_vs_haz,
                    sys_output_dict,
                    component_resp_dict,
                    calculated_output_array,
                    economic_loss_array,
                    output_array_given_recovery)


if __name__ == '__main__':
    main()
