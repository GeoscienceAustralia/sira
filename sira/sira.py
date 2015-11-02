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
import numpy as np
import scipy.stats as stats
import pandas as pd
import parmap

SETUPFILE = None


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

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
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            lower_lim = fragdict['minimum'][ct][ds]
            minpos = min(range(len(sc.hazard_intensity_vals)),
                         key=lambda i: abs(sc.hazard_intensity_vals[i] - lower_lim))
            zl = [0.0] * (minpos + 1)
            ol = [1] * (len(sc.hazard_intensity_vals) - (minpos + 1))
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


def compute_output_given_ds(cp_func, fc):
    '''
    Computes system output given list of component functional status
    '''
    G = fc.network.G
    nodes = fc.network.nodes

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
                    G.maxflow_value(G.vs.find(inode).index, G.vs.find(onode).index, G.es["capacity"])
                    * fc.input_dict[inode]['CapFraction']
                )

            total_available_flow_list.append(sum(avl_sys_flow_by_src))

        total_available_flow = min(total_available_flow_list)
        sys_out_capacity_list.append(
            min(total_available_flow, fc.output_dict[onode]['CapFraction'])
            * fc.nominal_production
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


def power_calc(fc, sc):
    # -----------------------------------------------------------------------------
    # Power output and economic loss calculations
    # -----------------------------------------------------------------------------
    cptype = {}
    cptype_ds_edges = {}
    comp_dict = fc.comp_df.to_dict()
    fragdict = fc.fragdict
    for comp in sorted(fc.comp_df.index):
        cptype[comp] = comp_dict['component_type'][comp]
        cptype_ds_edges[cptype[comp]] =\
            sorted(fragdict['damage_ratio'][cptype[comp]].values())

    ###############################################################################
    tp_cp = []
    for x in fc.comps_costed:
        tp_cp.extend(((x, 'loss_mean'), (x, 'loss_std'), (x, 'func_mean'), (x, 'func_std')))
    mindex = pd.MultiIndex.from_tuples(tp_cp, names=['component_id', 'response'])
    component_resp_df = pd.DataFrame(index=mindex, columns=[sc.PGA_str])

    return component_resp_df


def multiprocess_enabling_loop(idxPGA, _PGA_dummy, nPGA, fc, sc):
    if isinstance(_PGA_dummy, list):
        _PGA = _PGA_dummy[idxPGA]
    else:
        _PGA = _PGA_dummy
    print(" {0:3d}  out of {1:3d}".format(idxPGA+1, nPGA))

    comp_dict = fc.compdict
    fragdict = fc.fragdict

    ###############################################################################
    # simulation of damage of each component

    calculated_output_array = np.zeros((sc.num_samples, sc.num_hazard_pts))
    economic_loss_array = np.zeros_like(calculated_output_array)

    nodes_all = sorted(fc.comp_df.index)
    comp_loss_dict = {c: np.zeros((sc.num_samples, sc.num_hazard_pts)) for c in nodes_all}

    # Record output for:
    # <samples> vs <hazard parameter index> vs <time step index>
    output_array_given_recovery = np.zeros((sc.num_samples, sc.num_hazard_pts, sc.num_time_steps))

    # rnd = stats.uniform.rvs(loc=0, scale=1, size=(NUM_SAMPLES, num_elements))
    # np.save(os.path.join(RAW_OUTPUT_DIR, 'rnd_samples_x_elements.npy'), rnd)

    # List of output values at output_nodes:
    sys_output_list_given_pga = {k: np.zeros((sc.num_samples, len(fc.network.out_node_list))) for k in sc.PGA_str}

    # compute pe and determine ds for each component
    ids_comp = np.zeros((sc.num_samples, fc.num_elements), dtype=int)

    # index of damage state of components: from 0 to nds+1
    if sc.env:  # test run
        prng = np.random.RandomState(idxPGA)
    else:
        prng = np.random.RandomState()

    rnd = prng.uniform(size=(sc.num_samples, fc.num_elements))
    # index of damage state of components: from 0 to nds+1
    for j, comp in enumerate(nodes_all):
        ids_comp[:, j] = np.sum(cal_pe_ds(comp, float(_PGA), comp_dict, fragdict, sc) >
                                rnd[:, j][:, np.newaxis], axis=1)
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
            ids = ids_comp[i, j]   # index for component damage state
            ds = fc.sys_dmg_states[ids]   # damage state name
            cf = comp_dict['cost_fraction'][comp_name]
            dr = fragdict['damage_ratio'][comp_type][ds]
            fn = fragdict['functionality'][comp_type][ds]
            loss = dr * cf
            loss_list_all_comp.append(loss)

            # ........................................................
            # component functionality for calculated damage state:
            cp_func.append(fn)
            cp_func_given_time.append(calc_recov_time_given_comp_ds(comp_name, ids, comp_dict, fragdict, fc, sc))

            comp_loss_dict[comp_name][i, idxPGA] = loss
            component_loss_tmp[comp_name].append(loss)
            component_func_tmp[comp_name].append(fn)
            # ........................................................

        economic_loss_array[i, idxPGA] = sum(loss_list_all_comp)

        outputlist = compute_output_given_ds(cp_func, fc)
        calculated_output_array[i, idxPGA] = sum(outputlist)

        sys_output_list_given_pga[_PGA][i, :] = outputlist

        # restoration status of components over the range of time
        # (num elements X num specified time units)
        cp_func_given_time = np.array(cp_func_given_time)
        for t in range(sc.num_time_steps):
            output_array_given_recovery[i, idxPGA, t]\
                = sum(compute_output_given_ds(cp_func_given_time[:, t], fc))

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
            = np.mean(ids_comp[:, j] >= (len(fc.sys_dmg_states) - 1))

    sys_out_dict = dict()
    for onx, onode in enumerate(fc.network.out_node_list):
        sys_out_dict[onode]\
            = np.mean(sys_output_list_given_pga[_PGA][:, onx])
    return ids_comp, sys_out_dict, comp_resp_dict


def calc_loss_arrays(fc, sc, component_resp_df, parallel_or_serial):

    print("\nCalculating system response to hazard transfer parameters...")
    component_resp_dict = component_resp_df.to_dict()
    sys_output_dict = {k: {o: 0 for o in fc.network.out_node_list} for k in sc.PGA_str}
    ids_comp_vs_haz = {p: np.zeros((sc.num_samples, fc.num_elements)) for p in sc.PGA_str}

    if parallel_or_serial:
        print('\n===================>>>>>multiprocessor computation on <<<<========================')
        parallel_return = parmap.map(multiprocess_enabling_loop, range(len(sc.PGA_str)),
                                     sc.PGA_str, sc.num_hazard_pts, fc, sc)

        for idxPGA, _PGA in enumerate(sc.PGA_str):
            ids_comp_vs_haz[_PGA] = parallel_return[idxPGA][0]
            sys_output_dict[_PGA] = parallel_return[idxPGA][1]
            component_resp_dict[_PGA] = parallel_return[idxPGA][2]
    else:
        print('\n==================>>>>>single processor computation on <<<<=======================')
        for idxPGA, _PGA in enumerate(sc.PGA_str):
            ids_comp_vs_haz[_PGA], sys_output_dict[_PGA], component_resp_dict[_PGA] = \
                multiprocess_enabling_loop(idxPGA=idxPGA, _PGA_dummy=_PGA, nPGA=sc.num_hazard_pts, fc=fc, sc=sc)

    return ids_comp_vs_haz, sys_output_dict, component_resp_dict
