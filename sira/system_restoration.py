from __future__ import print_function
import os
import itertools
import operator, functools
import copy

import numpy as np
import scipy.stats as stats
import pandas as pd

import networkx as nx
import igraph

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
sns.set_style('whitegrid')

import siraplot as spl

################################################################################

def fill_between_steps(ax, x, y1, y2=0, step_where='pre', **kwargs):
    ''' 
    ********************************************************************
    Source:        https://github.com/matplotlib/matplotlib/issues/643
    From post by:  tacaswell
    Post date:     Nov 20, 2014
    ********************************************************************

    fill between a step plot and 

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
        steps = ma.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, 0::2], steps[0, 1::2] = vertices[0, :], vertices[0, :-1]
        steps[1:, 0::2], steps[1:, 1:-1:2] = vertices[1:, :], vertices[1:, 1:]

    elif step_where == 'post':
        steps = ma.zeros((3, 2 * len(x) - 1), np.float)
        steps[0, ::2], steps[0, 1:-1:2] = vertices[0, :], vertices[0, 1:]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :-1]

    elif step_where == 'mid':
        steps = ma.zeros((3, 2 * len(x)), np.float)
        steps[0, 1:-1:2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 2::2] = 0.5 * (vertices[0, :-1] + vertices[0, 1:])
        steps[0, 0] = vertices[0, 0]
        steps[0, -1] = vertices[0, -1]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :]
    else:
        raise RuntimeError("should never hit end of if-elif block for validated input")

    # un-pack
    xx, yy1, yy2 = steps

    # now to the plotting part:
    return ax.fill_between(xx, yy1, y2=yy2, **kwargs)

# ==============================================================================

def comp_recovery_given_haz(compname, hazval, t, compdict, fragdict, 
                            dmg_states, threshold = 0.98):
    ''' 
    Calculates level of recovery of component, given time t after impact 
    of hazard with intensity 'hazval'. 
    
    Currently implemented for earthquake only. 
    Hazard transfer parameter is PGA.
    
    '''
    
    ct  = compdict['component_type'][compname]
    m   = [fragdict['damage_median'][ct][ds] for ds in dmg_states]
    b   = [fragdict['damage_logstd'][ct][ds] for ds in dmg_states]

    rmu = [fragdict['recovery_mean'][ct][ds] for ds in dmg_states]
    rsd = [fragdict['recovery_std'][ct][ds] for ds in dmg_states]
    fn  = sorted(fragdict['functionality'][ct].values(), reverse=True)

    ptmp  = []
    pe    = np.array(np.zeros(len(dmg_states)))
    recov = np.array(np.zeros(len(dmg_states)))
    reqtime = np.array(np.zeros(len(dmg_states)))

    for d, ds in enumerate(dmg_states):
        if d < len(dmg_states)-1:
            pe[d] = 1.0 - stats.lognorm.cdf(hazval, b[d+1], scale=m[d+1]) - sum(ptmp)
        else:
            pe[d] = 1.0 - sum(ptmp)
        ptmp.append(pe[d])
        if ds=='DS0 None':
            recov[d] = 1.00
            reqtime[d] = 0.00
        else:
            recov[d] = stats.norm.cdf(t, rmu[d], scale=rsd[d])

    return sum(pe*recov)

# ==============================================================================

def prep_repair_list(G, weight_criteria, sc_haz_val_str, 
                     out_node_list, nodes_by_commoditytype, 
                     component_meanloss, comp_fullrst_time):
    '''
    ***************************************************************************
    Identify the shortest paths that need to be repaired in order to supply to
    each output node. 
    This is done based on:
       [1] the priority assigned to the output line
       [2] a weighting criterion applied to each node in the system    
    ***************************************************************************
    '''
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
    
    repair_list = {outnode:{sn:0 for sn in nodes_by_commoditytype.keys()} for outnode in out_node_list}
    repair_list_combined = {}
    
    for o,onode in enumerate(out_node_list):
        for CK, sup_nodes_by_commtype in nodes_by_commoditytype.iteritems():
            arr_row = []
            for i,inode in enumerate(sup_nodes_by_commtype):
                arr_row.append(input_dict[inode]['CapFraction'])
        
            for i,inode in enumerate(sup_nodes_by_commtype):
                thresh = output_dict[onode]['CapFraction']
            
                vx = []
                vlist = []
                for L in range(0, len(arr_row)+1):
                    for subset in itertools.combinations(range(0, len(arr_row)), L):
                        vx.append(subset)
                    for subset in itertools.combinations(arr_row, L):
                        vlist.append(subset)
                vx = vx[1:]
                vlist = [sum(x) for x in vlist[1:]]
                vcrit = np.array(vlist)>=thresh
            
                sp_len = np.zeros(len(vx))
                LEN_CHK = np.inf
                
                sp_dep = []
                for dnode in dep_node_list:
                    sp_dep.extend(G.get_shortest_paths(G.vs.find(dnode), to=G.vs.find(onode), 
                                  weights=w, mode='OUT')[0])
                for cix, criteria in enumerate(vcrit):
                    sp_list = []
                    if not criteria:
                        sp_len[cix] = np.inf
                    else:
                        for inx in vx[cix]:
                            icnode = sup_nodes_by_commtype[inx]
                            sp_list.extend(G.get_shortest_paths(G.vs.find(icnode), to=G.vs.find(onode), 
                                                            weights=w, mode='OUT')[0])
                        sp_list = np.unique(sp_list)
                        RL = [G.vs[x]['name'] for x in set([]).union(sp_dep, sp_list)]
                        sp_len[cix] = len(RL)
                    if sp_len[cix] < LEN_CHK:
                        LEN_CHK = sp_len[cix]
                        repair_list[onode][CK] = sorted(RL)
    
        repair_list_combined[onode] = sorted(list(set([]).union(*repair_list[onode].values())))

    return repair_list_combined

# ==============================================================================

def calc_restoration_setup(out_node_list, repair_list_combined, 
                           rst_stream, rst_offset, sc_haz_val_str):
    
    cols = ['NodesToRepair', 'OutputNode', 'RestorationTimes', 
            'RstStart', 'RstEnd', 'DeltaTC', 'RstSeq', 'Fin', 'EconLoss']
    
    fixed_asset_list = []
    restore_time_each_node = {}
    restore_time_aggregate = {}
    rst_setup_dict = {col:{n:[] for n in out_node_list} for col in cols}
    
    rst_setup_df = pd.DataFrame(columns=cols)
    df = pd.DataFrame(columns=cols)
    
    for onode in out_node_list:
    
        repair_list_combined[onode] = list(set(repair_list_combined[onode]).difference(fixed_asset_list))
        fixed_asset_list.extend(repair_list_combined[onode])
    
        restore_time_each_node[onode] = [comp_fullrst_time.ix[i]['Full Restoration Time'] 
                                         for i in repair_list_combined[onode]]
        restore_time_aggregate[onode] = max(restore_time_each_node[onode])\
                                      + sum(np.array(restore_time_each_node[onode]) * 0.01)
    
        df = pd.DataFrame({'NodesToRepair': repair_list_combined[onode], 
                           'OutputNode': [onode]*len(repair_list_combined[onode]),
                           'RestorationTimes': restore_time_each_node[onode],
                           'Fin': 0
                           })
        df = df.sort(['RestorationTimes'], ascending=[0])
        rst_setup_df = rst_setup_df.append(df)
    
    # rst_setup_df = rst_setup_df[rst_setup_df['RestorationTimes']!=0]
    
    # ************************************************************************************
    # get list of only those components that are included in cost calculations
    uncosted_comptypes  = ['CONN_NODE', 'SYSTEM_INPUT','SYSTEM_OUTPUT',
                           'Bus 230kV', 'Generator', 'Grounding']
    cp_types_costed     = [x for x in cp_types_in_system 
                              if x not in uncosted_comptypes]
    
    cpmap = {c:sorted(comp_df[comp_df['component_type']==c].index.tolist())
            for c in cp_types_in_system}
    comps_costed = [v for x in cp_types_costed for v in cpmap[x]]
    # ************************************************************************************
    
    uncosted_comps = set(nodes_all).difference(comps_costed)
    comps_to_drop = set(rst_setup_df.index.values.tolist()).intersection(uncosted_comps)
    
    rst_setup_df = rst_setup_df.drop(comps_to_drop, axis=0)
    rst_setup_df = rst_setup_df[rst_setup_df['RestorationTimes']!=0]
    rst_setup_df = rst_setup_df.set_index('NodesToRepair')[cols[1:]]
    rst_setup_df['DeltaTC'] = pd.Series(rst_setup_df['RestorationTimes'].values*0.00, \
                                        index=rst_setup_df.index) 
    for k in repair_path.keys():
        oldlist = repair_path[k]
        repair_path[k] = [v for v in oldlist if v not in uncosted_comps]
    
    rst_seq    = []
    num = len(rst_setup_df.index)
    for i in range(1, 1+int(ceil(num/float(rst_stream)))):
        rst_seq.extend([i]*rst_stream)
    rst_seq = rst_seq[:num]
    rst_setup_df['RstSeq'] = pd.Series(rst_seq, index=rst_setup_df.index)
    
    t_init = 0
    t0 = t_init+rst_offset
    for inx in rst_setup_df.index[0:rst_stream]:
        if inx!=rst_setup_df.index[0]: t0 += rst_setup_df.ix[inx]['DeltaTC']
        rst_setup_df.loc[inx, 'RstStart'] = t0
        rst_setup_df.loc[inx, 'RstEnd']   = rst_setup_df.ix[inx]['RstStart'] + rst_setup_df.ix[inx]['RestorationTimes']
    
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
        rst_setup_df.loc[inx, 'RstEnd']   = rst_setup_df.ix[inx]['RstStart'] + rst_setup_df.ix[inx]['RestorationTimes']
    
    cp_losses = [component_meanloss.loc[c, sc_haz_val_str] for c in rst_setup_df.index]
    rst_setup_df['EconLoss'] = cp_losses
    # add a column for 'component_meanloss'
    rst_setup_df.to_csv(os.path.join(output_path, 'restoration_setup'+haztag+'.csv'),
                        index_label=['NodesToRepair'], sep=',')

    return rst_setup_df
    
# ==============================================================================

def vis_restoration_process(rst_setup_df, rst_stream, out_node_list, repair_path):

    import seaborn as sns
    sns.set(style='white')
    
    gainsboro  = "#DCDCDC"
    whitesmoke = "#F5F5F5"
    lineht = 14
    
    comps = rst_setup_df.index.values.tolist()
    y     = range(1, len(comps)+1)
    xstep = 20
    xmax  = int(xstep * ceil(max(rst_setup_df['RstEnd'])/np.float(xstep)))
    xtiks = range(0, xmax+1, xstep)
    
    fig = plt.figure(facecolor='white', figsize=(17, len(y)*0.09))
    # gs  = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    # ax1 = plt.subplot(gs[0])
    # ax2 = plt.subplot(gs[1])
    ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    ax1 = fig.add_axes([0.1, 0.7, 0.8, len(y)*0.04])
    
    ax1.hlines(y, rst_setup_df['RstStart'], rst_setup_df['RstEnd'], linewidth=lineht, color=spl.colr_set2[2])
    ax1.set_title('Component Restoration Timeline: '+str(rst_stream)+' Simultaneous Repairs', loc='left', y=1.01, size=18)
    ax1.set_xlim([0, xmax])
    ax1.set_ylim([0.5,max(y)+0.5])
    ax1.set_yticks(y)
    ax1.set_yticklabels(comps, size=14);
    ax1.set_xticks(xtiks)
    ax1.set_xticklabels([]);
    for i in range(0, xmax+1, xstep): ax1.axvline(i, color='w', linewidth=0.5)
    ax1.yaxis.grid(True, which="major", linestyle='-', 
                linewidth=lineht, color=whitesmoke)
    
    spines_to_remove = ['left', 'top', 'right', 'bottom']
    for spine in spines_to_remove:
        ax1.spines[spine].set_visible(False)
    
    sns.axes_style(style='ticks')
    sns.despine(ax=ax2, left=True)
    ax2.set_xlim([0, xmax])
    ax2.set_ylim([0,100])
    ax2.set_yticks(range(0,101,20))
    ax2.set_yticklabels(range(0,101,20), size=14)
    ax2.yaxis.grid(True, which="major", color=gainsboro)
    ax2.tick_params(axis='x', which="major", bottom='on', length=4) 
    ax2.set_xticks(xtiks);
    ax2.set_xticklabels(range(0, xmax+1, xstep), size=14)
    ax2.set_xlabel('Restoration Time ('+timeunit+')', size=16)
    ax2.set_ylabel('System Capacity (%)', size=16)
    
    rst_time_line = np.zeros((len(out_node_list), xmax))
    line_rst_times = {}
    ypos = 0
    for x, onode in enumerate(out_node_list): 
        ypos += 100.0*output_dict[onode]['CapFraction']
        
        # line_rst_times[onode] = max(rst_setup_df[rst_setup_df['OutputNode']==onode]['RstEnd'])
        line_rst_times[onode] = max(rst_setup_df.loc[repair_path[onode]]['RstEnd'])
        
        ax1.axvline(line_rst_times[onode], linestyle=':', color=spl.colr_set1[2], alpha=0.8)
        ax2.axvline(line_rst_times[onode], linestyle=':', color=spl.colr_set1[2], alpha=0.8)
        ax2.annotate(onode, xy=(line_rst_times[onode], 105),
                     rotation=90, ha='center', va='bottom',
                     fontname='Open Sans', size=12, color='k', annotation_clip=False)    
        rst_time_line[x,:] = 100. * output_dict[onode]['CapFraction'] * \
                             np.array(list(np.zeros(int(line_rst_times[onode]))) +\
                                      list(np.ones(xmax - int(line_rst_times[onode]))))
    
    xrst = np.array(range(0, xmax, 1))
    yrst = np.sum(rst_time_line, axis=0)
    ax2.step(xrst, yrst, where='post', color=spl.colr_set1[2], clip_on=False)
    fill_between_steps(ax2, xrst, yrst, 0, step_where='post', alpha=0.25, color=spl.colr_set1[2])
    
    fig.savefig(os.path.join(output_path, 'fig'+haztag+'str'+str(rst_stream)+'_restoration.png'), 
                format='png', bbox_inches='tight', dpi=300)

    return rst_time_line, line_rst_times

# ==============================================================================

def component_criticality(ctype_scenario_outcomes, output_path, haztag):
    '''
    ****************************************************************
    REQUIRED IMPROVEMENTS:
     1. implement a criticality ranking
     2. use the criticality ranking as the label
     3. remove label overlap
    ****************************************************************
    '''
    
    import seaborn as sns
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    
    rt  = ctype_scenario_outcomes['restoration_time']
    pctloss_sys = ctype_scenario_outcomes['loss_tot']
    pctloss_ntype = ctype_scenario_outcomes['loss_per_type']*15
    
    nt_names  = ctype_scenario_outcomes.index.tolist()
    nt_ids    = range(1, len(nt_names)+1)
    # nt_labels = [str(m)+'  '+n for m,n in zip(nt_ids, nt_names)]
    
    clrmap = [plt.cm.autumn(1.2*x/float(len(ctype_scenario_outcomes.index)))
               for x in range(len(ctype_scenario_outcomes.index))]
    
    ax.scatter(rt, pctloss_sys, s=pctloss_ntype, 
            c=clrmap, label=nt_ids,
            marker='o', edgecolor='bisque', lw=1.5,
            clip_on=False)
    
    for cid, name, i, j in zip(nt_ids, nt_names, rt, pctloss_sys):
        plt.annotate(
            cid, 
            xy = (i, j), xycoords='data', 
            xytext = (-20, 20), textcoords='offset points', 
            ha = 'center', va = 'bottom', rotation=0,
            size=13, fontweight='bold', color='dodgerblue', annotation_clip=False, 
            bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.0),
            arrowprops = dict(arrowstyle = '-|>', 
                              shrinkA=5.0,
                              shrinkB=5.0,
                              connectionstyle = 'arc3,rad=0.0',
                              color='dodgerblue', 
                              alpha=0.8,
                              linewidth=0.5),)
        
        plt.annotate(
            "{0:>2.0f}   {1:<s}".format(cid, name), 
            xy = (1.05, 0.95-0.035*cid), xycoords=('axes fraction', 'axes fraction'),
            ha = 'left', va = 'top', rotation=0,
            size=9)
    
    ax.text(1.05, 0.995, 
        'Facility:  '+ SYSTEM_CLASS + '\n'
        'Hazard:  '+'Earthquake '+sc_haz_val_str+hazard_transfer_unit+' '+hazard_transfer_param,
        ha = 'left', va = 'top', rotation=0,
        fontsize=11, clip_on=False, transform=ax.transAxes)
    
    ylim = [0, int(max(pctloss_sys)+1)]
    ax.set_ylim(ylim)
    ax.set_yticks([0, max(ylim)*0.5, max(ylim)])
    ax.set_yticklabels(['%0.2f' %y for y in [0, max(ylim)*0.5, max(ylim)]], size=12)
    
    xlim = [0, np.ceil(max(rt)/10.0)*10]
    ax.set_xlim(xlim)
    ax.set_xticks([0, max(xlim)*0.5, max(xlim)])
    ax.set_xticklabels([int(x) for x in [0, max(xlim)*0.5, max(xlim)]], size=12)
    
    plt.grid(linewidth=3.0)
    ax.set_title('Component Criticality', size=13, y=1.04)
    ax.set_xlabel('Time to Restoration ('+timeunit+')', size=13, labelpad=14)
    ax.set_ylabel('System Loss (%)', size=13, labelpad=14)
    
    fig.savefig(os.path.join(output_path, 'fig'+haztag+'component_criticality.png'), 
                format='png', bbox_inches='tight', dpi=300)

################################################################################