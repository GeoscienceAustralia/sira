from __future__ import print_function
from sifraclasses import *

import numpy as np
import scipy.stats as stats
import pandas as pd
import igraph

import sys, os
import copy
import itertools
from colorama import Fore, init
init()

import sifraplot as spl

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
sns.set(style='whitegrid', palette='coolwarm')

# ============================================================================


def fill_between_steps(ax, x, y1, y2=0, step_where='pre', **kwargs):
    ''' 
    Fills between a step plot and x-axis

    ********************************************************************
    Source:        https://github.com/matplotlib/matplotlib/issues/643
    From post by:  tacaswell
    Post date:     Nov 20, 2014
    ********************************************************************

    Parameters
    ----------
    ax : Axes
       The axes to draw to

    x : array-like
        Array/vector of index values.

    y1 : array-like or float
        Array/vector of values to be filled under.
    y2 : array-Like or float, optional
        Array/vector or bottom values for filled area. Default is 0.

    step_where : {'pre', 'post', 'mid'}
        where the step happens, same meanings as for `step`

    **kwargs will be passed to the matplotlib fill_between() function.

    Returns
    -------
    ret : PolyCollection
       The added artist

    '''
    if step_where not in {'pre', 'post', 'mid'}:
        raise ValueError("where must be one of {{'pre', 'post', 'mid'}} "
                         "You passed in {wh}".format(wh=step_where))

    # make sure y values are up-converted to arrays 
    if np.isscalar(y1):
        y1 = np.ones_like(x) * y1

    if np.isscalar(y2):
        y2 = np.ones_like(x) * y2

    # temporary array for up-converting the values to step corners
    # 3 x 2N - 1 array 

    vertices = np.vstack((x, y1, y2))

    # this logic is lifted from lines.py
    # this should probably be centralized someplace
    if step_where == 'pre':
        steps = np.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, 0::2], steps[0, 1::2] = vertices[0, :], vertices[0, :-1]
        steps[1:, 0::2], steps[1:, 1:-1:2] = vertices[1:, :], vertices[1:, 1:]

    elif step_where == 'post':
        steps = np.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, ::2], steps[0, 1:-1:2] = vertices[0, :], vertices[0, 1:]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :-1]

    elif step_where == 'mid':
        steps = np.zeros((3, 2 * len(x)), np.float)
        steps[0, 1:-1:2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 2::2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 0] = vertices[0, 0]
        steps[0, -1] = vertices[0, -1]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :]
    else:
        raise RuntimeError(
            "should never hit end of if-elif block for validated input"
        )

    # un-pack
    xx, yy1, yy2 = steps

    # now to the plotting part:
    return ax.fill_between(xx, yy1, y2=yy2, **kwargs)

# ============================================================================


def comp_recovery_given_haz(facility, compname, hazval, t,
                            comptype_dmg_states, component_response,
                            comps_avl_for_int_replacement, 
                            threshold = 0.99):
    """
    Calculates level of recovery of component at time t after impact

    Currently implemented for earthquake only.
    Hazard transfer parameter is PGA.

    TEST PARAMETERS:
    Example from HAZUS MH MR3, Technical Manual, Ch.8, p8-73
    t = 3
    comptype_dmg_states = ['DS0 None', 'DS1 Slight', 'DS2 Moderate',
        'DS3 Extensive', 'DS4 Complete']
    m   = [np.inf, 0.15, 0.25, 0.35, 0.70]
    b   = [   1.0, 0.60, 0.50, 0.40, 0.40]
    rmu = [-np.inf, 1.0, 3.0, 7.0, 30.0]
    rsd = [    1.0, 0.5, 1.5, 3.5, 15.0]
    --------------------------------------------------------------------------
    """

    compdict = facility.compdict
    fragdict = facility.fragdict

    ct = compdict['component_type'][compname]
    
    m = [fragdict['damage_median'][ct][ds] for ds in comptype_dmg_states]
    b = [fragdict['damage_logstd'][ct][ds] for ds in comptype_dmg_states]
    # fn  = sorted(fragdict['functionality'][ct].values(), reverse=True)
    comp_fn = component_response.loc[(compname, 'func_mean'), ('%0.3f'% hazval)]
    
    haz_val_str = ("%0.3f" % np.float(hazval))
    if ct not in uncosted_comptypes \
        and comps_avl_for_int_replacement >= 1:
        # Parameters for Temporary Restoration:
        rmu = [fragdict['tmp_rst_mean'][ct][ds] for ds in comptype_dmg_states]
        rsd = [fragdict['tmp_rst_std'][ct][ds] for ds in comptype_dmg_states]
    else:
        # Parameters for Full Restoration: 
        rmu = [fragdict['recovery_mean'][ct][ds] for ds in comptype_dmg_states]
        rsd = [fragdict['recovery_std'][ct][ds] for ds in comptype_dmg_states]

    nd = len(comptype_dmg_states)
    ptmp = []
    pe = np.array(np.zeros(nd))
    pb = np.array(np.zeros(nd))
    recov = np.array(np.zeros(nd))
    reqtime = np.array(np.zeros(nd))

    for d in range(0,nd,1)[::-1]:
        pe[d] = stats.lognorm.cdf(hazval, b[d], scale=m[d])
        ptmp.append(pe[d])
        
    for d in range(0, nd, 1):
        if d == 0:
            pb[d] = 1.0 - pe[d+1]
        elif d >= 1 and d < nd-1:
            pb[d] = pe[d] - pe[d+1]
        elif d == nd-1:
            pb[d] = pe[d]
    
    for d, ds in enumerate(comptype_dmg_states):
        if ds == 'DS0 None':
            recov[d] = 1.0
            reqtime[d] = 0.00
        else:
            recov[d] = stats.norm.cdf(t, rmu[d], scale=rsd[d])
            reqtime[d] = stats.norm.ppf(threshold, loc=rmu[d], scale=rsd[d])\
                         - stats.norm.ppf(comp_fn, loc=rmu[d], scale=rsd[d])

    comp_status_agg = sum(pb*recov)
    # rst_time_agg = sum(pb*reqtime)
    rst_time_agg = reqtime[nd-1]
    
    return comp_status_agg, rst_time_agg
    
# ============================================================================


def prep_repair_list(facility_obj,
                     weight_criteria, sc_haz_val_str,
                     component_meanloss, comp_fullrst_time):
    """
    Identify the shortest component repair list to restore supply to output

    This is done based on:
       [1] the priority assigned to the output line
       [2] a weighting criterion applied to each node in the system
    """
    G = facility_obj.network.G
    input_dict = facility_obj.input_dict
    output_dict = facility_obj.output_dict
    nodes_by_commoditytype = facility_obj.nodes_by_commoditytype
    out_node_list = facility_obj.network.out_node_list

    w = 'weight'
    for tp in G.get_edgelist():
        eid = G.get_eid(*tp)
        origin = G.vs[tp[0]]['name']
        destin = G.vs[tp[1]]['name']
        if weight_criteria == None:
            wt = 1.0
        elif weight_criteria == 'MIN_TIME':
            wt = 1.0/comp_fullrst_time.ix[origin]['Full Restoration Time']
        elif weight_criteria == 'MIN_COST':
            wt = 1.0/component_meanloss.loc[origin, sc_haz_val_str]
        G.es[eid][w] = wt
    
    repair_list = {outnode:{sn:0 for sn in nodes_by_commoditytype.keys()}
                   for outnode in out_node_list}
    repair_list_combined = {}
    
    for o,onode in enumerate(out_node_list):
        for CK, sup_nodes_by_commtype in nodes_by_commoditytype.iteritems():
            arr_row = []
            for i,inode in enumerate(sup_nodes_by_commtype):
                arr_row.append(input_dict[inode]['capacity_fraction'])
        
            for i,inode in enumerate(sup_nodes_by_commtype):
                thresh = output_dict[onode]['capacity_fraction']
            
                vx = []
                vlist = []
                for L in range(0, len(arr_row)+1):
                    for subset in \
                            itertools.combinations(range(0, len(arr_row)), L):
                        vx.append(subset)
                    for subset in itertools.combinations(arr_row, L):
                        vlist.append(subset)
                vx = vx[1:]
                vlist = [sum(x) for x in vlist[1:]]
                vcrit = np.array(vlist)>=thresh
            
                sp_len = np.zeros(len(vx))
                LEN_CHK = np.inf
                
                sp_dep = []
                for dnode in facility_obj.network.dep_node_list:
                    sp_dep.extend(G.get_shortest_paths(
                        G.vs.find(dnode), to=G.vs.find(onode),
                        weights=w, mode='OUT')[0]
                    )
                for cix, criteria in enumerate(vcrit):
                    sp_list = []
                    if not criteria:
                        sp_len[cix] = np.inf
                    else:
                        for inx in vx[cix]:
                            icnode = sup_nodes_by_commtype[inx]
                            sp_list.extend(G.get_shortest_paths(
                                G.vs.find(icnode),
                                to=G.vs.find(onode),
                                weights=w, mode='OUT')[0])
                        sp_list = np.unique(sp_list)
                        RL = [G.vs[x]['name']
                              for x in set([]).union(sp_dep, sp_list)]
                        sp_len[cix] = len(RL)
                    if sp_len[cix] < LEN_CHK:
                        LEN_CHK = sp_len[cix]
                        repair_list[onode][CK] = sorted(RL)
    
        repair_list_combined[onode] = sorted(
            list(set([]).union(*repair_list[onode].values()))
        )

    return repair_list_combined

# ============================================================================


def calc_restoration_setup(out_node_list, repair_list_combined, 
                           rst_stream, rst_offset, sc_haz_val_str, 
                           comps_costed, comp_fullrst_time, output_path,
                           haztag,
                           buffer_time_to_commission=0.00):
    """
    Calculates the timeline for full repair of all output lines of the system

    Depends on the given the hazard/restoration scenario specified through
    the parameters
    --------------------------------------------------------------------------
    :param out_node_list:
            list of output nodes
    :param repair_list_combined:
            dict with output nodes as keys, with list of nodes needing repair
            for each output node as values
    :param rst_stream:
            maximum number of components that can be repaired concurrently
    :param rst_offset:
            time delay from hazard impact to start of repair
    :param sc_haz_val_str:
            hazard intensity value for scenario, given as a string
    :param comps_costed:
            list of all system components for which costs are modelled
    :param comp_fullrst_time:
            dataframe with components names as indices, and time required to
            restore those components
    :param output_path:
            directory path for saving output
    :param haztag:
            tag to add to the outputs produced
    :param buffer_time_to_commission:
            buffer time between completion of one repair task and
            commencement of the next repair task
    --------------------------------------------------------------------------
    :return: rst_setup_df: PANDAS DataFrame with:
                - The components to repaired as indices
                - Time required to repair each component
                - Repair start time for each component
                - Repair end time for each component
    --------------------------------------------------------------------------
    """
    cols = ['NodesToRepair', 'OutputNode', 'RestorationTimes', 
            'RstStart', 'RstEnd', 'DeltaTC', 'RstSeq', 'Fin', 'EconLoss']


    fixed_asset_list = []
    restore_time_each_node = {}
    # restore_time_aggregate = {}
    # rst_setup_dict = {col:{n:[] for n in out_node_list} for col in cols}
    
    rst_setup_df = pd.DataFrame(columns=cols)
    # df = pd.DataFrame(columns=cols)
    
    for onode in out_node_list:
    
        repair_list_combined[onode] = \
            list(set(repair_list_combined[onode]).difference(fixed_asset_list))
        fixed_asset_list.extend(repair_list_combined[onode])
    
        restore_time_each_node[onode] = \
            [comp_fullrst_time.ix[i]['Full Restoration Time'] 
             for i in repair_list_combined[onode]]
        # restore_time_aggregate[onode] = \
        #     max(restore_time_each_node[onode]) + \
        #     sum(np.array(restore_time_each_node[onode]) * 0.01)
    
        df = pd.DataFrame(
            {'NodesToRepair': repair_list_combined[onode],
             'OutputNode': [onode]*len(repair_list_combined[onode]),
             'RestorationTimes': restore_time_each_node[onode],
             'Fin': 0
             })
        df = df.sort_values(by=['RestorationTimes'], ascending=[0])
        rst_setup_df = rst_setup_df.append(df)

    uncosted_comps = set(facility.network.nodes_all).difference(comps_costed)
    comps_to_drop = set(rst_setup_df.index.values.tolist()).\
                        intersection(uncosted_comps)
    
    rst_setup_df = rst_setup_df.drop(comps_to_drop, axis=0)
    rst_setup_df = rst_setup_df[rst_setup_df['RestorationTimes']!=0]
    rst_setup_df = rst_setup_df.set_index('NodesToRepair')[cols[1:]]
    rst_setup_df['DeltaTC'] = pd.Series(
        rst_setup_df['RestorationTimes'].values * buffer_time_to_commission,
        index=rst_setup_df.index
    )

    for k in repair_path.keys():
        oldlist = repair_path[k]
        repair_path[k] = [v for v in oldlist if v not in uncosted_comps]

    rst_seq = []
    num = len(rst_setup_df.index)
    for i in range(1, 1+int(np.ceil(num/float(rst_stream)))):
        rst_seq.extend([i]*rst_stream)
    rst_seq = rst_seq[:num]
    rst_setup_df['RstSeq'] = pd.Series(rst_seq, index=rst_setup_df.index)

    t_init = 0
    t0 = t_init+rst_offset
    for inx in rst_setup_df.index[0:rst_stream]:
        if inx!=rst_setup_df.index[0]: t0 += rst_setup_df.ix[inx]['DeltaTC']
        rst_setup_df.loc[inx, 'RstStart'] = t0
        rst_setup_df.loc[inx, 'RstEnd'] = \
            rst_setup_df.ix[inx]['RstStart'] + \
            rst_setup_df.ix[inx]['RestorationTimes']

    dfx = copy.deepcopy(rst_setup_df)
    for inx in rst_setup_df.index[rst_stream:]:
        t0 = min(dfx['RstEnd'])   #rst_setup_df.ix[inx]['DeltaTC']

        finx = rst_setup_df[rst_setup_df['RstEnd']==min(dfx['RstEnd'])]

        for x in finx.index:
            if rst_setup_df.loc[x, 'Fin'] == 0:
                rst_setup_df.loc[x, 'Fin'] = 1
                break
        dfx = rst_setup_df[rst_setup_df['Fin']!=1]
        rst_setup_df.loc[inx, 'RstStart'] = t0
        rst_setup_df.loc[inx, 'RstEnd'] = \
            rst_setup_df.ix[inx]['RstStart'] + \
            rst_setup_df.ix[inx]['RestorationTimes']

    cp_losses = [component_meanloss.loc[c, sc_haz_val_str] 
                for c in rst_setup_df.index]
    rst_setup_df['EconLoss'] = cp_losses

    # add a column for 'component_meanloss'
    rst_setup_df.to_csv(
        os.path.join(output_path, 'restoration_setup_'+haztag+'.csv'),
        index_label=['NodesToRepair'], sep=','
    )

    return rst_setup_df

# ============================================================================


def vis_restoration_process(scenario,
                            facility,
                            rst_setup_df,
                            rst_stream,
                            repair_path,
                            haztag):
    """
    Plots restoration timeline of components and supply to output nodes

    --------------------------------------------------------------------------
    Outputs:
    [1] Plot of restored capacity, as step functions
        - Restoration displayed as percentage of pre-disasater
          system output capacity
        - Restoration work is conducted to recover output based on
          'output streams' or 'production lines'
        - Restoration is prioritised in accordance with line priorities
          defined in input file
    [2] A simple Gantt chart of component restoration
    [3] rst_time_line:
        Array of restored line capacity for each time step simulated
    [2] line_rst_times:
        Dict with LINES as keys, and TIME to full restoration as values
    --------------------------------------------------------------------------
    """
    sns.set(style='white')
    
    gainsboro = "#DCDCDC"
    whitesmoke = "#F5F5F5"
    lineht = 10

    out_node_list = facility.network.out_node_list

    comps = rst_setup_df.index.values.tolist()
    y = range(1, len(comps)+1)
    xstep = 10
    # xmax  = int(xstep * np.ceil(max(rst_setup_df['RstEnd'])/np.float(xstep)))
    xmax = \
        int(xstep * np.ceil(1.05*max(rst_setup_df['RstEnd'])/np.float(xstep)))
    if xmax < xstep:
        xstep = 1
    elif xmax == 0:
        xmax = 2
        xstep = 1
    xtiks = range(0, xmax+1, xstep)

    hw_ratio_ax2 = 1.0/2.8
    fig_w_cm  = 9.0
    fig_h_cm  = (fig_w_cm*hw_ratio_ax2) * (1 + len(y)/7.5 + 0.5)
    num_grids = 7 + 3 + len(y)

    # hw_ratio_ax2 = 1.0/3.3
    # fig_w_cm  = 9
    # fig_h_cm  = (fig_w_cm*hw_ratio_ax2) * (1 + len(y)/8.0 + 0.4)
    # num_grids = 7 + 3 + len(y)
    
    fig = plt.figure(facecolor='white', figsize=(fig_w_cm, fig_h_cm))
    gs  = gridspec.GridSpec(num_grids, 1)
    ax1 = plt.subplot(gs[:-11])
    ax2 = plt.subplot(gs[-8:])

    ax1.hlines(y, rst_setup_df['RstStart'], rst_setup_df['RstEnd'],
               linewidth=lineht, color=spl.COLR_SET2[2])
    ax1.set_title('Component Restoration Timeline: '+str(rst_stream)+
                  ' Simultaneous Repairs', loc='left', y=1.01, size=18)
    ax1.set_xlim([0, xmax-1])
    ax1.set_ylim([0.5,max(y)+0.5])
    ax1.set_yticks(y)
    ax1.set_yticklabels(comps, size=14);
    ax1.set_xticks(xtiks)
    ax1.set_xticklabels([]);
    for i in range(0, xmax+1, xstep):
        ax1.axvline(i, color='w', linewidth=0.5)
    ax1.yaxis.grid(True, which="major", linestyle='-',
                   linewidth=lineht, color=whitesmoke)
    
    spines_to_remove = ['left', 'top', 'right', 'bottom']
    for spine in spines_to_remove:
        ax1.spines[spine].set_visible(False)
    
    sns.axes_style(style='ticks')
    sns.despine(ax=ax2, left=True)
    ax2.set_xlim([0, xmax-1])
    ax2.set_ylim([0,100])
    ax2.set_yticks(range(0,101,20))
    ax2.set_yticklabels(range(0,101,20), size=14)
    ax2.yaxis.grid(True, which="major", color=gainsboro)
    ax2.tick_params(axis='x', which="major", bottom='on', length=4) 
    ax2.set_xticks(xtiks);
    ax2.set_xticklabels(range(0, xmax+1, xstep), size=14)
    ax2.set_xlabel('Restoration Time ('+scenario.time_unit+')', size=16)
    ax2.set_ylabel('System Capacity (%)', size=16)
    
    rst_time_line = np.zeros((len(out_node_list), xmax))
    line_rst_times = {}
    ypos = 0
    for x, onode in enumerate(out_node_list): 
        ypos += 100.0 * facility.output_dict[onode]['capacity_fraction']
        
        # line_rst_times[onode] = \
        #     max(rst_setup_df[rst_setup_df['OutputNode']==onode]['RstEnd'])
        line_rst_times[onode] = \
        max(rst_setup_df.loc[repair_path[onode]]['RstEnd'])
        
        ax1.axvline(line_rst_times[onode], linestyle=':', 
                    color=spl.COLR_SET1[2], alpha=0.8)
        ax2.axvline(line_rst_times[onode], linestyle=':', 
                    color=spl.COLR_SET1[2], alpha=0.8)
        ax2.annotate(onode, xy=(line_rst_times[onode], 105),
                     ha='center', va='bottom', rotation=90, 
                     fontname='Open Sans', size=12, color='k', 
                     annotation_clip=False)    
        rst_time_line[x, :] = \
            100. * facility.output_dict[onode]['capacity_fraction'] * \
            np.array(list(np.zeros(int(line_rst_times[onode]))) +
                     list(np.ones(xmax - int(line_rst_times[onode])))
                     )

    xrst = np.array(range(0, xmax, 1))
    yrst = np.sum(rst_time_line, axis=0)
    ax2.step(xrst, yrst, where='post', color=spl.COLR_SET1[2], clip_on=False)
    fill_between_steps(ax2, xrst, yrst, 0, step_where='post',
                       alpha=0.25, color=spl.COLR_SET1[2])
    
    fig.savefig(
        os.path.join(scenario.output_path, 'fig_' + haztag + '_str' +
                     str(rst_stream) + '_restoration.png'),
        format='png', bbox_inches='tight', dpi=300
    )

    plt.close(fig)

    return rst_time_line, line_rst_times

# ============================================================================


def component_criticality(facility, scenario,
                          ctype_scenario_outcomes,
                          sc_haz_val_str,
                          haztag):
    """
    Plots criticality of components based on cost & time of reparation

    **************************************************************************
    REQUIRED IMPROVEMENTS:
     1. implement a criticality ranking
     2. use the criticality ranking as the label
     3. remove label overlap
    **************************************************************************
    """

    # sns.set(style="ticks",
    #         rc={"axes.facecolor": '#EAEAF2',
    #             "axes.linewidth": 0.75,
    #             "grid.color": 'white',
    #             "grid.linestyle": '-',
    #             "grid.linewidth": 1.5,
    #             "xtick.major.size": 8,
    #             "ytick.major.size": 8})

    sns.set(style="darkgrid",
            rc={"axes.edgecolor": '0.15',
                "axes.linewidth": 0.75,
                "grid.color": 'white',
                "grid.linestyle": '-',
                "grid.linewidth": 1.5,
                "xtick.major.size": 8,
                "ytick.major.size": 8})

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    rt = ctype_scenario_outcomes['restoration_time']
    pctloss_sys = ctype_scenario_outcomes['loss_tot']
    pctloss_ntype = ctype_scenario_outcomes['loss_per_type']*15

    nt_names = ctype_scenario_outcomes.index.tolist()
    nt_ids = range(1, len(nt_names)+1)

    clrmap = [plt.cm.autumn(1.2 * x/float(len(ctype_scenario_outcomes.index)))
              for x in range(len(ctype_scenario_outcomes.index))]

    ax.scatter(rt, pctloss_sys, s=pctloss_ntype, 
            c=clrmap, label=nt_ids,
            marker='o', edgecolor='bisque', lw=1.5,
            clip_on=False)

    for cid, name, i, j in zip(nt_ids, nt_names, rt, pctloss_sys):
        plt.annotate(
            cid, 
            xy=(i, j), xycoords='data',
            xytext=(-20, 20), textcoords='offset points',
            ha='center', va='bottom', rotation=0,
            size=13, fontweight='bold', color='dodgerblue',
            annotation_clip=False,
            bbox=dict(boxstyle='round, pad=0.2', fc='yellow', alpha=0.0),
            arrowprops=dict(arrowstyle = '-|>',
                            shrinkA=5.0,
                            shrinkB=5.0,
                            connectionstyle = 'arc3,rad=0.0',
                            color='dodgerblue',
                            alpha=0.8,
                            linewidth=0.5),)
        
        plt.annotate(
            "{0:>2.0f}   {1:<s}".format(cid, name), 
            xy = (1.05, 0.95-0.035*cid),
            xycoords=('axes fraction', 'axes fraction'),
            ha = 'left', va = 'top', rotation=0,
            size=9)
    
    ax.text(1.05, 0.995, 
            'Facility:  '+facility.system_class+'\n'+
                'Hazard:  '+'Earthquake '+
                sc_haz_val_str+scenario.intensity_measure_unit+' '+
                scenario.intensity_measure_param,
            ha = 'left', va = 'top', rotation=0,
            fontsize=11, clip_on=False, transform=ax.transAxes)
    
    ylim = [0, int(max(pctloss_sys)+1)]
    ax.set_ylim(ylim)
    ax.set_yticks([0, max(ylim)*0.5, max(ylim)])
    ax.set_yticklabels(['%0.2f' %y for y in [0, max(ylim)*0.5, max(ylim)]],
                       size=12)
    
    xlim = [0, np.ceil(max(rt)/10.0)*10]
    ax.set_xlim(xlim)
    ax.set_xticks([0, max(xlim)*0.5, max(xlim)])
    ax.set_xticklabels([int(x) for x in [0, max(xlim)*0.5, max(xlim)]],
                       size=12)
    
    # plt.grid(linewidth=3.0)
    ax.set_title('Component Criticality', size=13, y=1.04)
    ax.set_xlabel('Time to Restoration (' +scenario.time_unit+')',
                  size=13, labelpad=14)
    ax.set_ylabel('System Loss (%)', size=13, labelpad=14)

    sns.despine(left=False, bottom=False, right=True, top=True,
                offset=12, trim=True)

    figfile = os.path.join(scenario.output_path,
                           'fig_' + haztag + '_component_criticality.png')
    fig.savefig(figfile, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)

# ============================================================================


def draw_component_loss_barchart_v1(scenario,
                                    ctype_loss_vals_tot,
                                    ctype_loss_by_type,
                                    ctype_lossbytype_rank,
                                    sc_haz_val_str,
                                    fig_name):
    """ Plots bar charts of direct economic losses for components types """

    # Set color maps:
    clrmap1 = [plt.cm.autumn(1.2*x/float(len(ctype_loss_vals_tot)))
               for x in range(len(ctype_loss_vals_tot))]
    clrmap2 = [clrmap1[int(i)] for i in ctype_lossbytype_rank]

    a = 1.0     # transparency
    bar_width = 0.7
    yadjust = bar_width/2.0
    subplot_spacing = 0.6

    cpt = [spl.split_long_label(x,delims=[' ', '_'], max_chars_per_line=22)
           for x in ctype_resp_sorted.index.tolist()]
    pos = np.arange(0,len(cpt))

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True,
                                   facecolor='white',
                                   figsize=(12,len(pos)*0.6))
    fig.subplots_adjust(wspace=subplot_spacing)

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # Economic loss contributed by all components of a specific type,
    # as a percentage of the value of the system

    ax1.barh(pos, ctype_loss_vals_tot, bar_width,
             color=clrmap1, edgecolor='bisque', alpha=a)
    ax1.set_xlim(0, max(ctype_loss_by_type))
    ax1.set_ylim(pos.max()+bar_width, pos.min()-bar_width)
    # ax1.grid(False)
    ax1.tick_params(top='off', bottom='off', left='off', right='on')
    ax1.set_title('Economic Loss \nPercent of System Value',
                     fontname='Open Sans', fontsize=12,
                     fontweight='bold', ha='right', x=1.00, y=0.99)

    # add the numbers to the side of each bar
    for p, c, cv in zip(pos, cpt, ctype_loss_vals_tot):
        ax1.annotate(('%0.1f' % np.float(cv))+'%',
                     xy=(cv+0.5, p+yadjust),
                     xycoords=('data', 'data'),
                     ha='right', va='center', size=11,
                     annotation_clip=False)

    ax1.xaxis.set_ticks_position('none')
    ax1.set_axis_off()

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # Aggregated economic loss for all components of a specific type

    ax2.barh(pos, ctype_loss_by_type, bar_width,
             color=clrmap2, edgecolor='bisque', alpha=a)
    ax2.set_xlim(0, max(ctype_loss_by_type))
    ax2.set_ylim(pos.max()+bar_width, pos.min()-bar_width)
    ax2.tick_params(top='off', bottom='off', left='on', right='off')
    ax2.set_title('Economic Loss \nPercent of Component Type Value',
                  fontname='Open Sans', fontsize=12,
                  fontweight='bold', ha='left', x=0,  y=0.99)

    for p, c, cv in zip(pos, cpt, ctype_loss_by_type):
        ax2.annotate(('%0.1f' % np.float(cv))+'%', xy=(cv+0.5, p+yadjust),
                     xycoords=('data', 'data'),
                     ha='left', va='center', size=11,
                     annotation_clip=False)

    ax2.xaxis.set_ticks_position('none')
    ax2.set_axis_off()

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    ax1.invert_xaxis()

    for yloc, ylab in zip(pos, cpt):
        ax1.annotate(ylab,
                     xy=(1.0+subplot_spacing/2.0, yloc+yadjust),
                     xycoords=('axes fraction', 'data'),
                     ha='center', va='center', size=11, color='k',
                     annotation_clip=False)

    ax1.annotate('HAZARD EVENT\n' +
                 'Earthquake\n' +
                 sc_haz_val_str + ' ' + scenario.intensity_measure_param,
                 xy=(1.0 + subplot_spacing/2.0, -1.25),
                 xycoords=('axes fraction', 'data'),
                 ha='center', va='center', size=12,
                 fontname='Open Sans', color='darkgrey', weight='bold',
                 annotation_clip=False)

    fig.savefig(os.path.join(scenario.output_path, fig_name),
                format='png', bbox_inches='tight', dpi=300)

    plt.close(fig)

# ============================================================================


def draw_component_loss_barchart_v2(scenario,
                                    ctype_loss_vals_tot,
                                    ctype_loss_by_type,
                                    sc_haz_val_str,
                                    fig_name):
    """ Plots bar charts of direct economic losses for components types """

    bar_width = 0.3
    bar_offset = 0.03
    bar_clr_1 = spl.COLR_SET1[0]
    bar_clr_2 = spl.COLR_SET1[1]

    cpt = [spl.split_long_label(x,delims=[' ', '_'], max_chars_per_line=22)
           for x in ctype_resp_sorted.index.tolist()]
    pos = np.arange(0, len(cpt))

    fig = plt.figure(figsize=(4.5, len(pos)*0.6), facecolor='white')
    axes = fig.add_subplot(111, axisbg='white')

    # Economic loss:
    #   - Contribution to % loss of total system, by components type
    #   - Percentage econ loss for all components of a specific type

    axes.barh(
        pos-bar_width-bar_offset, ctype_loss_vals_tot, bar_width,
        color=bar_clr_1, alpha=0.85, edgecolor='bisque',
        label="Percentage loss of total system value (by component type)"
    )

    axes.barh(
        pos+bar_offset*2, ctype_loss_by_type, bar_width,
        color=bar_clr_2, alpha=0.85, edgecolor='bisque',
        label="Percentage loss for component type"
    )

    axes.tick_params(top='off', bottom='off', left='on', right='off')

    axes.set_xlim(0, max(ctype_loss_by_type))
    axes.set_xticklabels([''])
    axes.set_ylim([pos.max()+bar_width+0.4, pos.min()-bar_width-0.4])
    axes.set_yticks(pos)
    axes.set_yticklabels(cpt, size=11, color='k')
    axes.tick_params(top='off', bottom='off', left='on', right='off')

    for p, c, cv in zip(pos, cpt, ctype_loss_vals_tot):
        axes.annotate(('%0.1f' % np.float(cv))+'%',
                      xy=(cv+0.5, p-bar_offset-bar_width/2.0),
                      xycoords=('data', 'data'),
                      ha='left', va='center', size=11, color=bar_clr_1,
                      annotation_clip=False)

    for p, c, cv in zip(pos, cpt, ctype_loss_by_type):
        axes.annotate(('%0.1f' % np.float(cv))+'%',
                      xy=(cv+0.5, p+bar_offset*2+bar_width/2.0),
                      xycoords=('data', 'data'),
                      ha='left', va='center', size=11, color=bar_clr_2,
                      annotation_clip=False)

    axes.xaxis.set_ticks_position('none')
    spines_to_remove = ['left', 'top', 'right']
    for spine in spines_to_remove:
        axes.spines[spine].set_visible(False)

    axes.yaxis.grid(False)
    axes.xaxis.grid(False)

    axes.annotate(
        'Percentage Economic Loss by Component Type',
        xy=(0.0, -1.6), xycoords=('axes fraction', 'data'),
        ha='left', va='top',
        fontname='Open Sans', size=12, color='k', weight='bold',
        annotation_clip=False);
    axes.annotate(
        'Hazard Event: Earthquake ' + sc_haz_val_str + ' ' +
            scenario.intensity_measure_param,
        xy=(0.0, -1.1), xycoords=('axes fraction', 'data'),
        ha='left', va='top',
        fontname='Open Sans', size=12, color='grey', weight='bold',
        annotation_clip=False);

    axes.legend(loc=9, ncol=1, bbox_to_anchor=(0.46, -0.01),
                frameon=0, prop={'size':10})

    fig.savefig(
        os.path.join(scenario.output_path, fig_name),
        format='png', bbox_inches='tight', dpi=300
    )

    plt.close(fig)

# ============================================================================


def draw_component_failure_barchart(uncosted_comptypes,
                                    ctype_failure_mean,
                                    sc_haz_val_str,
                                    output_path,
                                    figname):

    comp_type_fail_sorted = \
        ctype_failure_mean.sort_values(by=[sc_haz_val_str], ascending=[0])

    for x in uncosted_comptypes:
        if x in comp_type_fail_sorted.index.tolist():
            comp_type_fail_sorted = comp_type_fail_sorted.drop(x, axis=0)

    cptypes = comp_type_fail_sorted.index.tolist()
    cpt = [spl.split_long_label(x,delims=[' ', '_'],max_chars_per_line=22)
           for x in cptypes]
    pos = np.arange(len(cptypes))
    cpt_failure_vals = comp_type_fail_sorted[sc_haz_val_str].values * 100

    fig = plt.figure(figsize=(10, len(pos)*0.6), facecolor='white')
    ax = fig.add_subplot(111, axisbg='white')
    bar_width = 0.6
    bar_clr = spl.COLR_SET3[4]    # dodgerblue

    ax.barh(pos, cpt_failure_vals, bar_width,
            color=bar_clr, edgecolor="bisque")

    # add the numbers to the side of each bar
    for p, c, cv in zip(pos, cptypes, cpt_failure_vals):
        plt.annotate(('%0.1f' % cv)+'%', xy=(cv+0.3, p+bar_width/2.0),
                     va='center', size=11, color='k')

    spines_to_remove = ['left', 'top', 'right', 'bottom']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

    #cutomize ticks
    plt.gca().yaxis.tick_left()
    plt.yticks(pos + bar_width/2.0, cpt, size=11, color='k')
    xt = list(plt.xticks()[0])
    xt.append(max(xt)+10.0)
    plt.xticks(xt, [' '] * len(xt))
    ax.grid(False)
    ax.tick_params(top='off', bottom='off', left='off', right='off')

    #set plot limits
    plt.xlim(0, max(cpt_failure_vals)+5.0)
    plt.ylim(pos.max() + 1.0, pos.min() - 1.0)

    ax.annotate('Percentage Loss: Number of Components by Type',
                xy=(0.0, -1.3), xycoords=('axes fraction', 'data'),
                ha='left', va='top', annotation_clip=False,
                fontname='Open Sans', size=12, weight='bold', color='k');
    ax.annotate('Hazard Event: Earthquake ' + sc_haz_val_str + ' ' +
                    scenario.intensity_measure_param,
                xy=(0.0, -0.8), xycoords=('axes fraction', 'data'),
                ha='left', va='top', annotation_clip=False,
                fontname='Open Sans', size=12, weight='bold',
                color='darkgrey');

    fig.savefig(
        os.path.join(output_path, figname),
        format='png', bbox_inches='tight', dpi=300
    )

    plt.close(fig)

# ============================================================================


def calc_comptype_damage_scenario_given_hazard(facility,
                                               cp_types_costed,
                                               comptype_resp_df):
    comptype_num = {
        x:facility.comp_df[facility.comp_df['component_type']==x]\
            ['component_type'].count()
        for x in facility.cp_types_in_system
        }
    comptype_used = {x:0 for x in facility.cp_types_in_system}

    # ------------------------------------------------------------------------
    # Logic for internal component replacement / scavenging

    comptype_for_internal_replacement = {}
    for x in facility.cp_types_in_system:
        if x in cp_types_costed:
            comptype_for_internal_replacement[x] = \
                int(np.floor(
                    (1.0 - comptype_resp_df.loc[
                        (x, 'num_failures'), sc_haz_val_str]
                     ) * comptype_num[x]
                ))
        else:
            comptype_for_internal_replacement[x] = 0

    # ------------------------------------------------------------------------
    # Using the HAZUS method:

    comp_rst = {t: {n: 0 for n in nodes_all}
                for t in scenario.restoration_time_range}
    for c in nodes_all:

        ct = facility.compdict['component_type'][c]
        comptype_used[ct] += 1
        comps_avl_for_int_replacement = \
            comptype_for_internal_replacement[ct] - comptype_used[ct]

        for t in scenario.restoration_time_range:
            comp_rst[t][c] = comp_recovery_given_haz(
                facility, c, sc_haz_val, t,
                COMP_TYPE_DS,
                component_response,
                comps_avl_for_int_replacement=0,
                threshold=RST_THRESHOLD)[0]
    comp_rst_df = pd.DataFrame(comp_rst,
                               index=nodes_all,
                               columns=scenario.restoration_time_range)

    comp_rst_time_given_haz = \
        [np.round(comp_rst_df.columns[comp_rst_df.loc[c]
                                      >= RST_THRESHOLD][0], 0)
         for c in nodes_all]

    # ------------------------------------------------------------------------

    component_fullrst_time = pd.DataFrame(
                        {'Full Restoration Time': comp_rst_time_given_haz},
                        index=nodes_all)
    component_fullrst_time.index.names=['component_id']

    # ------------------------------------------------------------------------

    ctype_scenario_outcomes = copy.deepcopy(
        100 * ctype_resp_sorted.drop(['func_mean', 'func_std'], axis=1)
    )

    rtimes = []
    for x in ctype_scenario_outcomes.index:
        rtimes.append(np.mean(component_fullrst_time.loc[cpmap[x]].values))
    ctype_scenario_outcomes['restoration_time'] = rtimes

    return component_fullrst_time, ctype_scenario_outcomes

# ============================================================================
# READ in SETUP data
# The first argument is the full path to the config file
# ============================================================================

# --- BEGIN SETUP ---

SETUPFILE = sys.argv[1]
discard = {}
config = {}

# Read in config file and define Scenario & Facility objects
exec (open(SETUPFILE).read(), discard, config)
FacilityObj = eval(config["SYSTEM_CLASS"])
scenario = Scenario(SETUPFILE)
facility = FacilityObj(SETUPFILE)

# Define input files, output location, scenario inputs
INPUT_PATH = scenario.input_path
OUTPUT_PATH = scenario.output_path
RAW_OUTPUT_DIR = scenario.raw_output_dir
SYS_CONFIG_FILE = os.path.join(INPUT_PATH, facility.sys_config_file_name)

RESTORATION_TIME_RANGE = scenario.restoration_time_range
RESTORATION_STREAMS = scenario.restoration_streams
SCENARIO_HAZARD_VALUES = scenario.scenario_hazard_values
SYS_DS = facility.sys_dmg_states
COMP_TYPE_DS = sorted(list(facility.fragility_data.index.\
                           get_level_values('damage_state').unique()))

haz_vals_str = [('%0.3f' % np.float(x))
                for x in scenario.hazard_intensity_vals]

# ------------------------------------------------------------------------
# READ in raw output files from prior analysis of system fragility

economic_loss_array = \
    np.load(os.path.join(
        RAW_OUTPUT_DIR, 'economic_loss_array.npy'))

calculated_output_array = \
    np.load(os.path.join(
        RAW_OUTPUT_DIR, 'calculated_output_array.npy'))

output_array_given_recovery = \
    np.load(os.path.join(
        RAW_OUTPUT_DIR, 'output_array_given_recovery.npy'))

exp_damage_ratio = \
    np.load(os.path.join(
        RAW_OUTPUT_DIR, 'exp_damage_ratio.npy'))

sys_frag = \
    np.load(os.path.join(
        RAW_OUTPUT_DIR, 'sys_frag.npy'))

required_time = \
    np.load(os.path.join(RAW_OUTPUT_DIR, 'required_time.npy'))

if facility.system_class == 'PowerStation':
    pe_sys = \
        np.load(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_econloss.npy'))
elif facility.system_class == 'Substation':
    pe_sys = \
        np.load(os.path.join(RAW_OUTPUT_DIR, 'pe_sys_cpfailrate.npy'))

# ------------------------------------------------------------------------
# Read in SIMULATED HAZARD RESPONSE for <COMPONENT TYPES>

comptype_resp_df = \
    pd.read_csv(os.path.join(scenario.output_path,
                             'comp_type_response.csv'),
                index_col=['component_type', 'response'],
                skipinitialspace=True)
comptype_resp_df.columns = [haz_vals_str]

ct_resp_flat = comptype_resp_df.reset_index('response')

ctype_loss_mean = \
    ct_resp_flat[ct_resp_flat['response'] == 'loss_mean'].\
    drop('response', axis=1)

ctype_loss_tot = \
    ct_resp_flat[ct_resp_flat['response'] == 'loss_tot'].\
    drop('response', axis=1)

ctype_loss_std = \
    ct_resp_flat[ct_resp_flat['response'] == 'loss_std'].\
    drop('response', axis=1)

ctype_func_mean = \
    ct_resp_flat[ct_resp_flat['response'] == 'func_mean'].\
    drop('response', axis=1)

ctype_func_std = \
    ct_resp_flat[ct_resp_flat['response'] == 'func_std'].\
    drop('response', axis=1)

ctype_failure_mean = \
    ct_resp_flat[ct_resp_flat['response'] == 'num_failures'].\
    drop('response', axis=1)

# ****************************************************************************
# Nodes not considered in the loss calculations
# NEED TO MOVE THESE TO A MORE LOGICAL PLACE

uncosted_comptypes = ['CONN_NODE', 'SYSTEM_INPUT', 'SYSTEM_OUTPUT',
                      'Bus', 'Bus 230kV', 'Bus 69kV',
                      'Generation Source', 'Grounding']

cp_types_costed = [x for x in facility.cp_types_in_system
                   if x not in uncosted_comptypes]

# Get list of only those components that are included in cost calculations:

cpmap = {c:sorted(facility.comp_df[facility.comp_df['component_type'] == c]
                  .index.tolist())
        for c in facility.cp_types_in_system}
comps_costed = [v for x in cp_types_costed for v in cpmap[x]]

# ****************************************************************************

# Value of component types relative to system value
comptype_value = []
for k in sorted(cp_types_costed):
    v = facility.comp_df[facility.comp_df['component_type']==k]\
        ['cost_fraction'].sum(axis=0)
    n = facility.comp_df[facility.comp_df['component_type']==k].index.size
    comptype_value.append(v/n)

# Read in SIMULATED HAZARD RESPONSE - for <COMPONENT INSTANCES>
component_response = \
    pd.read_csv(os.path.join(OUTPUT_PATH, 'component_response.csv'),
                index_col=['component_id', 'response'],
                skiprows=0, skipinitialspace=True)

component_meanloss = \
    component_response.query('response == "loss_mean"').\
        reset_index('response').drop('response', axis=1)

# Read in the <SYSTEM FRAGILITY MODEL> fitted to simulated data
system_fragility_mdl = \
    pd.read_csv(os.path.join(OUTPUT_PATH, 'Model_System_Fragility.csv'),
                index_col=0)

system_fragility_mdl.index.name = "Damage States"

# Define the system as a network, with components as nodes
# Network setup with igraph

G = facility.network.G
nodes_all = facility.network.nodes_all

# -- END SETUP ---

# ****************************************************************************
# Configuration values that can be adjusted for specific scenarios:

RST_THRESHOLD = 0.98

# Restoration time starts x time units after hazard impact:
# This represents lead up time for damage and safety assessments
rst_offset = 1

# Set weighting criteria for edges: 
# This influences the path chosen for restoration
# Options are: 
#   [1] None
#   [2] 'MIN_COST'
#   [3] 'MIN_TIME'
weight_criteria = 'MIN_COST'

# ****************************************************************************

col_tp = []
for h in SCENARIO_HAZARD_VALUES:
    col_tp.extend(zip([h]*len(RESTORATION_STREAMS), RESTORATION_STREAMS))
mcols = pd.MultiIndex.from_tuples(
            col_tp, names=['Hazard', 'Restoration Streams'])
line_rst_times_df = \
    pd.DataFrame(index=facility.network.out_node_list, columns=mcols)
line_rst_times_df.index.name = 'Output Lines'


for h in SCENARIO_HAZARD_VALUES:
    sc_haz_val = h
    sc_haz_val_str = '%0.3f' % np.float(sc_haz_val)
    haztag = 'SC_'+('%0.2f' % np.float(sc_haz_val))+'g'

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # Extract scenario-specific values from the 'hazard response' dataframe
    # Scenario response: by component type

    ctype_resp_scenario = comptype_resp_df[sc_haz_val_str].unstack(level=-1)
    ctype_resp_scenario = ctype_resp_scenario.sort_index()
    ctype_resp_scenario['loss_per_type'] = \
        ctype_resp_scenario['loss_mean']/comptype_value
    ctype_resp_scenario = \
        ctype_resp_scenario.loc[ctype_resp_scenario['loss_per_type']>=0]
    ctype_resp_sorted = ctype_resp_scenario.sort_values(by=['loss_tot'],
                                                        ascending=[0])

    ctype_loss_vals_tot = ctype_resp_sorted['loss_tot'].values * 100
    ctype_loss_by_type = ctype_resp_sorted['loss_per_type'].values * 100
    ctype_lossbytype_rank = \
        len(ctype_loss_by_type) - \
        stats.rankdata(ctype_loss_by_type, method='dense').astype(int)

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # Economic loss percentage for component types

    fig_name = 'fig_'+haztag+'_loss_sys_vs_comptype_v1.png'
    draw_component_loss_barchart_v1(scenario,
                                    ctype_loss_vals_tot,
                                    ctype_loss_by_type,
                                    ctype_lossbytype_rank,
                                    sc_haz_val_str,
                                    fig_name)

    fig_name = 'fig_'+haztag+'_loss_sys_vs_comptype_v2.png'
    draw_component_loss_barchart_v2(scenario,
                                    ctype_loss_vals_tot,
                                    ctype_loss_by_type,
                                    sc_haz_val_str,
                                    fig_name)

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # Failure percentage of component types

    figname = 'fig_'+haztag+'_comptype_failures.png'
    draw_component_failure_barchart(uncosted_comptypes,
                                    ctype_failure_mean,
                                    sc_haz_val_str,
                                    OUTPUT_PATH,
                                    figname)

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    # RESTORATION PROGNOSIS for specified scenarios

    component_fullrst_time, ctype_scenario_outcomes = \
        calc_comptype_damage_scenario_given_hazard(facility,
                                                   cp_types_costed,
                                                   comptype_resp_df)

    # All the nodes that need to be fixed for each output node:
    repair_list_combined = prep_repair_list(facility,
                                            weight_criteria,
                                            sc_haz_val_str,
                                            component_meanloss,
                                            component_fullrst_time)

    repair_path = copy.deepcopy(repair_list_combined)

    for RS in RESTORATION_STREAMS:
        rst_setup_df = calc_restoration_setup(
            facility.network.out_node_list, repair_list_combined,
            RS, rst_offset, sc_haz_val_str,
            comps_costed, component_fullrst_time,
            scenario.output_path,
            haztag
        )

        rst_time_line, line_rst_times = vis_restoration_process(
            scenario,
            facility,
            rst_setup_df,
            RS,
            repair_path,
            haztag
        )

        line_rst_times_df[(h, RS)] = \
            [line_rst_times[x] for x in facility.network.out_node_list]
        
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    component_criticality(facility, scenario,
                          ctype_scenario_outcomes,
                          sc_haz_val_str,
                          haztag)

    ##########################################################################
    # --- END FOR LOOP ---

line_rst_times_csv = os.path.join(OUTPUT_PATH,
                                  'line_restoration_prognosis.csv')
line_rst_times_df.to_csv(line_rst_times_csv, sep=',')

##############################################################################

print(Fore.YELLOW + "\nScenario loss analysis complete." + Fore.RESET)
print("Outputs saved in: \n" +
      Fore.GREEN + scenario.output_path + Fore.RESET + '\n')