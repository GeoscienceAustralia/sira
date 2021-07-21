import copy
import itertools
import logging
import os

import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore, init
from matplotlib import gridspec

import sira.tools.siraplot as spl

init()
rootLogger = logging.getLogger(__name__)
np.seterr(divide='print', invalid='raise')
plt.switch_backend('agg')
sns.set(style='whitegrid', palette='coolwarm')

# **************************************************************************
# Configuration values that can be adjusted for specific scenarios:

RESTORATION_THRESHOLD = 0.98

# *****************************************************************************


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


# *************************************************************************************

def comp_recovery_given_hazard_and_time(
    component,
    component_response,
    scenario_header,
    hazval,
    time_after_impact,
    comps_avl_for_int_replacement,  # noqa:W0613
    threshold_recovery=0.98,
    threshold_exceedance=0.001,
):

    """
    Calculates level of recovery of component at time t after impact
    Hazard transfer parameter is INTENSITY_MEASURE.

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

    comptype_dmg_states = component.damage_states
    # if len(comptype_dmg_states) != len(system_damage_states):
    #     rootLogger.warning("*** Damage state levels are "+
    #     "not matched between system and component.\n"
    # )

    damage_functions = \
        [component.damage_states[i].response_function
         for i in range(len(comptype_dmg_states))]
    recovery_functions = \
        [component.damage_states[i].recovery_function
         for i in range(len(comptype_dmg_states))]

    comp_fn = component_response.loc[(component.component_id, 'func_mean'),
                                     scenario_header]

    num_dmg_states = len(comptype_dmg_states)
    pe = np.array(np.zeros(num_dmg_states))
    pb = np.array(np.zeros(num_dmg_states))
    recov = np.array(np.zeros(num_dmg_states))
    reqtime = np.array(np.zeros(num_dmg_states))

    for d in range(0, num_dmg_states, 1):
        pe[d] = damage_functions[d](hazval)

    for dmg_index in range(0, num_dmg_states, 1):
        if dmg_index == 0:
            pb[dmg_index] = 1.0 - pe[dmg_index + 1]
        elif dmg_index >= 1 and dmg_index < num_dmg_states - 1:
            pb[dmg_index] = pe[dmg_index] - pe[dmg_index + 1]
        elif dmg_index == num_dmg_states - 1:
            pb[dmg_index] = pe[dmg_index]

    # *********************************************************************************
    # Component 'cannibalisation' for temporary restoration
    # TODO: implement alternate functions for temporary restoration  # noqa:W0511
    # reqtime_X = np.array(np.zeros(num_dmg_states))
    # if comps_avl_for_int_replacement >= 1:
    #     # Parameters for Temporary Restoration:
    #     rmu = [fragdict['tmp_rst_mean'][ct][ds] for ds in comptype_dmg_states]
    #     rsd = [fragdict['tmp_rst_std'][ct][ds] for ds in comptype_dmg_states]
    # else:
    #     # Parameters for Full Restoration:
    #     rmu = [fragdict['recovery_param1'][ct][ds] for ds in comptype_dmg_states]
    #     rsd = [fragdict['recovery_param2'][ct][ds] for ds in comptype_dmg_states]
    # *********************************************************************************
    for dmg_index, ds in enumerate(comptype_dmg_states):
        if ds == 'DS0 None' or dmg_index == 0:
            recov[dmg_index] = 1.0
            reqtime[dmg_index] = 0.00
        elif pb[dmg_index] < threshold_exceedance:
            recov[dmg_index] = 1.0
            reqtime[dmg_index] = 0.00
        else:
            recov[dmg_index] = recovery_functions[dmg_index](time_after_impact)
            reqtime[dmg_index] = \
                recovery_functions[dmg_index](threshold_recovery, inverse=True)\
                - recovery_functions[dmg_index](comp_fn, inverse=True)

    comp_status_agg = sum(pb * recov)
    restoration_time_agg = sum(pb * reqtime)

    return comp_status_agg, restoration_time_agg

# *****************************************************************************


def prep_repair_list(infrastructure_obj,
                     component_meanloss,
                     component_fullrst_time,
                     uncosted_comps,
                     weight_criteria,
                     scenario_header):
    """
    Identify the shortest component repair list to restore supply to output

    This is done based on:
       [1] the priority assigned to the output line
       [2] a weighting criterion applied to each node in the system
    """
    G = infrastructure_obj.get_component_graph()
    input_dict = infrastructure_obj.supply_nodes
    output_dict = infrastructure_obj.output_nodes
    commodity_types = list(
        set([input_dict[i]['commodity_type'] for i in list(input_dict.keys())])
    )
    nodes_by_commoditytype = {}
    for comm_type in commodity_types:
        nodes_by_commoditytype[comm_type] = \
            [x for x in list(input_dict.keys())
             if input_dict[x]['commodity_type'] == comm_type]

    out_node_list = output_dict.keys()
    dependency_node_list = \
        [node_id for node_id, infodict
         in list(infrastructure_obj.components.items())
         if infrastructure_obj.components[node_id].node_type == 'dependency']

    w = 'weight'
    for tp in G.get_edgelist():
        eid = G.get_eid(*tp)
        origin = G.vs[tp[0]]['name']
        # destin = G.vs[tp[1]]['name']
        if weight_criteria is None:
            wt = 1.0
        elif weight_criteria == 'MIN_TIME':
            wt = component_fullrst_time.loc[origin, 'Full Restoration Time']
        elif weight_criteria == 'MIN_COST':
            wt = component_meanloss.loc[origin, scenario_header]
        G.es[eid][w] = wt

    repair_list = {
        outnode: {sn: 0 for sn in list(nodes_by_commoditytype.keys())}
        for outnode in out_node_list
    }
    repair_list_combined = {}

    for onode in out_node_list:
        for CK, sup_nodes_by_commtype in list(nodes_by_commoditytype.items()):
            arr_row = []
            for inode in sup_nodes_by_commtype:
                arr_row.append(input_dict[inode]['capacity_fraction'])

            thresh = output_dict[onode]['capacity_fraction']
            vx = []
            vlist = []
            for L in range(0, len(arr_row) + 1):
                for subset in itertools.combinations(
                        list(range(0, len(arr_row))), L):
                    vx.append(subset)
                for subset in itertools.combinations(arr_row, L):
                    vlist.append(subset)
            vx = vx[1:]
            vlist = [sum(x) for x in vlist[1:]]
            vcrit = np.array(vlist) >= thresh

            sp_len = np.zeros(len(vx))
            LEN_CHK = np.inf

            sp_dep = []
            for dnode in dependency_node_list:
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

        c_list = []
        for k in list(repair_list[onode].values()):
            c_list.extend(k)
        temp_repair_list = list(set(c_list))

        repair_list_combined[onode] = sorted(
            [x for x in temp_repair_list if x not in uncosted_comps]
        )

    return repair_list_combined

# *****************************************************************************


def calc_restoration_setup(component_meanloss,
                           out_node_list, comps_uncosted,
                           repair_list_combined,
                           rst_stream, RESTORATION_OFFSET,
                           comp_fullrst_time, output_path,
                           scenario_header, filename,
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
    :param RESTORATION_OFFSET:
            time delay from hazard impact to start of repair
    :param scenario_header:
            hazard intensity value for scenario, given as a string
    :param comps_costed:
            list of all system components for which costs are modelled
    :param comp_fullrst_time:
            dataframe with components names as indices, and time required to
            restore those components
    :param output_path:
            directory path for saving output
    :param scenario_tag:
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

    repair_path = copy.deepcopy(repair_list_combined)
    fixed_asset_list = []
    restore_time_each_node = {}

    rst_setup_df = pd.DataFrame(columns=cols)

    for onode in out_node_list:
        repair_list_combined[onode] = \
            list(set(repair_list_combined[onode]).difference(fixed_asset_list))
        fixed_asset_list.extend(repair_list_combined[onode])

        restore_time_each_node[onode] = \
            [comp_fullrst_time.loc[c, 'Full Restoration Time']
             for c in repair_list_combined[onode]]
        df = pd.DataFrame(
            {'NodesToRepair': repair_list_combined[onode],
             'OutputNode': [onode] * len(repair_list_combined[onode]),
             'RestorationTimes': restore_time_each_node[onode],
             'Fin': 0
             })
        df = df.sort_values(by=['RestorationTimes'], ascending=[0])
        rst_setup_df = rst_setup_df.append(df, sort=True)

    comps_to_drop = set(rst_setup_df.index.values.tolist()). \
        intersection(comps_uncosted)

    rst_setup_df = rst_setup_df.drop(comps_to_drop, axis=0)
    rst_setup_df = rst_setup_df[rst_setup_df['RestorationTimes'] != 0]
    rst_setup_df = rst_setup_df.set_index('NodesToRepair')[cols[1:]]
    rst_setup_df['DeltaTC'] = pd.Series(
        rst_setup_df['RestorationTimes'].values * buffer_time_to_commission,
        index=rst_setup_df.index
    )

    for k in list(repair_path.keys()):
        oldlist = repair_path[k]
        repair_path[k] = [v for v in oldlist if v not in comps_uncosted]

    rst_seq = []
    num = len(rst_setup_df.index)
    for i in range(1, 1 + int(np.ceil(num / float(rst_stream)))):
        rst_seq.extend([i] * rst_stream)
    rst_seq = rst_seq[:num]
    rst_setup_df['RstSeq'] = pd.Series(rst_seq, index=rst_setup_df.index)

    t_init = 0
    t0 = t_init + RESTORATION_OFFSET
    for inx in rst_setup_df.index[0:rst_stream]:
        if inx != rst_setup_df.index[0]:
            t0 += rst_setup_df.loc[inx, 'DeltaTC']
        rst_setup_df.loc[inx, 'RstStart'] = t0
        rst_setup_df.loc[inx, 'RstEnd'] = \
            rst_setup_df.loc[inx, 'RstStart'] + \
            rst_setup_df.loc[inx, 'RestorationTimes']

    dfx = copy.deepcopy(rst_setup_df)
    for inx in rst_setup_df.index[rst_stream:]:
        t0 = min(dfx['RstEnd'])

        finx = rst_setup_df[rst_setup_df['RstEnd'] == min(dfx['RstEnd'])]

        for x in finx.index:
            if rst_setup_df.loc[x, 'Fin'] == 0:
                rst_setup_df.loc[x, 'Fin'] = 1
                break
        dfx = rst_setup_df[rst_setup_df['Fin'] != 1]
        rst_setup_df.loc[inx, 'RstStart'] = t0
        rst_setup_df.loc[inx, 'RstEnd'] = \
            rst_setup_df.loc[inx, 'RstStart'] + \
            rst_setup_df.loc[inx, 'RestorationTimes']

    cp_losses = [component_meanloss.loc[c, scenario_header]
                 for c in rst_setup_df.index]
    rst_setup_df['EconLoss'] = cp_losses

    # add a column for 'component_meanloss'
    rst_setup_df.to_csv(
        os.path.join(output_path, filename),
        index_label=['NodesToRepair'], sep=','
    )

    return rst_setup_df

# *****************************************************************************


def vis_restoration_process(scenario,
                            infrastructure,
                            rst_setup_df,
                            num_rst_steams,
                            repair_path,
                            fig_name,
                            scenario_tag,
                            hazard_type):
    """
    Plots:
    - restoration timeline of components, and
    - restoration of supply to output nodes

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
    [2] time_to_full_restoration_for_lines:
        Dict with LINES as keys, and TIME to full restoration as values
    --------------------------------------------------------------------------
    """
    sns.set(style='white')

    gainsboro = "#DCDCDC"
    whitesmoke = "#F5F5F5"
    lineht = 11

    output_dict = infrastructure.output_nodes
    out_node_list = list(infrastructure.output_nodes.keys())

    comps = rst_setup_df.index.values.tolist()

    # Check if nothing to repair, i.e. if repair list is empty:
    if not comps:
        print("Nothing to repair. Time to repair is zero.")
        print("Skipping repair visualisation for scenario : {}".format(
            scenario_tag))
        return

    y = list(range(1, len(comps) + 1))
    xstep = 10
    xbase = 5.0
    xmax = \
        int(xstep * np.ceil(
            1.01 * max(rst_setup_df['RstEnd']) / np.float(xstep)))

    # Limit number of x-steps for labelling
    if xmax / xstep > 15:
        xstep = int(xbase * round((xmax / 10.0) / xbase))
    if xmax <= xstep:
        xstep = 1
    elif xmax == 0:
        xmax = 2
        xstep = 1

    xtiks = list(range(0, xmax + 1, xstep))
    if xmax > xtiks[-1]:
        xtiks.append(xmax)

    hw_ratio_ax2 = 1.0 / 3.3
    fig_w_cm = 9.0
    fig_h_cm = (fig_w_cm * hw_ratio_ax2) * (1.5 + len(y) / 7.5)
    num_grids = 10 + len(y)

    fig = plt.figure(facecolor='white', figsize=(fig_w_cm, fig_h_cm))
    gs = gridspec.GridSpec(num_grids, 1)

    gx = len(y) + 2
    ax1 = plt.subplot(gs[:gx])
    ax2 = plt.subplot(gs[-8:])
    ticklabelsize = 12

    colours = spl.ColourPalettes()
    linecolour1 = colours.BrewerSet2[2]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Primary Figure Title

    ax1.annotate(
        f"Restoration Prognosis for: {hazard_type} {scenario_tag}",
        xy=(0, len(comps) + 3.1), xycoords=('axes fraction', 'data'),
        ha='left', va='top', annotation_clip=False,
        color='slategrey', size=18, weight='bold'
    )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Component restoration plot
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ax1.annotate(
        f"Component Restoration Timeline for {str(num_rst_steams)} Repair Streams\n",
        xy=(0, len(comps) + 1.8), xycoords=('axes fraction', 'data'),
        ha='left', va='top', annotation_clip=False,
        color='k', size=16, weight='normal')

    ax1.hlines(y, rst_setup_df['RstStart'], rst_setup_df['RstEnd'],
               linewidth=lineht, color=linecolour1)
    ax1.set_ylim([0.5, max(y) + 0.5])
    ax1.set_yticks(y)
    ax1.set_yticklabels(comps, size=ticklabelsize)

    ax1.set_xlim([0, max(xtiks)])
    ax1.set_xticks(xtiks)
    ax1.set_xticklabels([])

    for i in range(0, xmax + 1, xstep):
        ax1.axvline(i, color='w', linewidth=0.5)
    ax1.yaxis.grid(True, which="major", linestyle='-',
                   linewidth=lineht, color=whitesmoke)

    spines_to_remove = ['left', 'top', 'right', 'bottom']
    for spine in spines_to_remove:
        ax1.spines[spine].set_visible(False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Supply restoration plot
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    sns.axes_style(style='ticks')
    sns.despine(ax=ax2, left=True)

    ax2.set_ylim([0, 100])
    ax2.set_yticks(list(range(0, 101, 20)))
    ax2.set_yticklabels(list(range(0, 101, 20)), size=ticklabelsize)
    ax2.yaxis.grid(True, which="major", color=gainsboro)

    ax2.set_xlim([0, max(xtiks)])
    ax2.set_xticks(xtiks)
    ax2.set_xticklabels(xtiks, size=ticklabelsize, rotation=0)
    ax2.tick_params(axis='x', which="major", bottom=True, length=4)

    ax2.set_xlabel('Restoration Time (' + scenario.time_unit + ')', size=14)
    ax2.set_ylabel('System Capacity (%)', size=14)

    supply_rst_clr = "#377EB8"

    restoration_timeline_array = np.zeros((len(out_node_list), max(xtiks) + 1))
    time_to_full_restoration_for_lines = {}

    for x, onode in enumerate(out_node_list):
        nodes_to_repair_for_line = \
            list(rst_setup_df.index.intersection(repair_path[onode]))
        time_to_full_restoration_for_lines[onode] = \
            max(rst_setup_df.loc[nodes_to_repair_for_line]['RstEnd'])

        ax1.axvline(time_to_full_restoration_for_lines[onode], linestyle=':',
                    color=supply_rst_clr, alpha=0.8)
        ax2.axvline(time_to_full_restoration_for_lines[onode], linestyle=':',
                    color=supply_rst_clr, alpha=0.8)
        ax2.annotate(onode, xy=(time_to_full_restoration_for_lines[onode], 105),
                     ha='center', va='bottom', rotation=90,
                     size=ticklabelsize, color='k',
                     annotation_clip=False)

        if not nodes_to_repair_for_line:
            repair_time_for_line = 0
        else:
            repair_time_for_line = time_to_full_restoration_for_lines[onode]

        restoration_arr_template = np.array(
            list(np.zeros(int(repair_time_for_line)))
            + list(np.ones(max(xtiks) + 1 - int(repair_time_for_line)))
        )
        restoration_timeline_array[x, :] = \
            100 * output_dict[onode]['capacity_fraction'] * restoration_arr_template

    xrst = np.array(list(range(0, max(xtiks) + 1, 1)))
    yrst = np.sum(restoration_timeline_array, axis=0)
    ax2.step(xrst, yrst, where='post', color=supply_rst_clr, clip_on=False)
    fill_between_steps(ax2, xrst, yrst, 0, step_where='post',
                       alpha=0.25, color=supply_rst_clr)

    if fig.get_figwidth() == 0 or fig.get_figheight() == 0:
        return restoration_timeline_array, time_to_full_restoration_for_lines

    fig.savefig(
        os.path.join(scenario.output_path, fig_name),
        format='png', dpi=400, bbox_inches='tight'
    )
    plt.close(fig)

    return restoration_timeline_array, time_to_full_restoration_for_lines

# *****************************************************************************


def component_criticality(infrastructure,
                          scenario,
                          ctype_scenario_outcomes,
                          hazard_type,
                          scenario_tag,
                          fig_name):
    """
    Plots criticality of components based on cost & time of reparation

    **************************************************************************
    REQUIRED IMPROVEMENTS:
     1. implement a criticality ranking
     2. use the criticality ranking as the label
     3. remove label overlap
    **************************************************************************
    """
    axes_lw = 0.75

    sns.set(
        style="darkgrid",
        rc={
            # --- Edges & Spine ---
            "axes.edgecolor": '0.15',
            "axes.linewidth": axes_lw,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.spines.right": False,
            "axes.spines.top": False,
            # --- Line style ---
            "grid.color": 'white',
            "grid.linestyle": '-',
            "grid.linewidth": 2.0,
            # --- Y-axis customisation ---
            "ytick.left": True,
            "ytick.labelsize": 13,
            "ytick.major.size": 7,
            "ytick.major.width": axes_lw,
            "ytick.major.pad": 4,
            "ytick.minor.left": False,
            # --- X-axis customisation ---
            "xtick.bottom": True,
            "xtick.labelsize": 13,
            "xtick.major.size": 7,
            "xtick.major.width": axes_lw,
            "xtick.major.pad": 4,
            "xtick.minor.bottom": False,
        }
    )

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    rt = ctype_scenario_outcomes['restoration_time']
    pctloss_sys = ctype_scenario_outcomes['loss_tot']
    pctloss_ntype = ctype_scenario_outcomes['loss_per_type'] * 15

    nt_names = ctype_scenario_outcomes.index.tolist()
    nt_ids = list(range(1, len(nt_names) + 1))

    autumn = mpl.cm.get_cmap("autumn")
    clrmap = [autumn(1.2 * x / float(len(ctype_scenario_outcomes.index)))
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
            arrowprops=dict(arrowstyle='-|>',
                            shrinkA=5.0,
                            shrinkB=5.0,
                            connectionstyle='arc3,rad=0.0',
                            color='dodgerblue',
                            alpha=0.8,
                            linewidth=0.5),
            path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")]
        )

        plt.annotate(
            "{0:>2.0f}   {1:<s}".format(cid, name),
            xy=(1.05, 0.90 - 0.035 * cid),
            xycoords=('axes fraction', 'axes fraction'),
            ha='left', va='top', rotation=0, size=9)

    txt_infra = "Infrastructure: " + infrastructure.system_class + "\n"
    txt_haz = "Hazard: " + hazard_type + "\n"
    txt_scenario = "Scenario: " + scenario_tag
    ax.text(1.05, 0.995,
            txt_infra + txt_haz + txt_scenario,
            ha='left', va='top', rotation=0, linespacing=1.5,
            fontsize=11, clip_on=False, transform=ax.transAxes)

    ylim = [0, int(max(pctloss_sys) + 1)]
    ax.set_ylim(ylim)
    ax.set_yticks([0, max(ylim) * 0.5, max(ylim)])
    ax.set_yticklabels(['%0.2f' % y for y in [0, max(ylim) * 0.5, max(ylim)]],
                       size=12)

    xlim = [0, np.ceil(max(rt) / 10.0) * 10]
    ax.set_xlim(xlim)
    ax.set_xticks([0, max(xlim) * 0.5, max(xlim)])
    ax.set_xticklabels([int(x) for x in [0, max(xlim) * 0.5, max(xlim)]],
                       size=12)

    ax.set_title('COMPONENT CRITICALITY GRID',
                 size=14, y=1.04, weight='bold')
    ax.set_xlabel('Time to Restoration (' + scenario.time_unit + ')',
                  size=13, labelpad=10)
    ax.set_ylabel('System Loss (%)',
                  size=13, labelpad=10)

    sns.despine(left=False, bottom=False, right=True, top=True,
                offset=15, trim=True)

    figfile = os.path.join(scenario.output_path, fig_name)
    fig.savefig(figfile, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)

# *****************************************************************************


def draw_component_loss_barchart_s1(ctype_resp_sorted,
                                    scenario_tag,
                                    hazard_type,
                                    output_path,
                                    fig_name):
    """ Plots bar charts of direct economic losses for components types """

    ctype_loss_tot_mean = ctype_resp_sorted['loss_tot'].values * 100
    ctype_loss_by_type = ctype_resp_sorted['loss_per_type'].values * 100

    bar_width = 0.36
    bar_offset = 0.02
    bar_clr_1 = spl.ColourPalettes().BrewerSet1[0]  # '#E41A1C'
    bar_clr_2 = spl.ColourPalettes().BrewerSet1[1]  # '#377EB8'
    grid_clr = "#BBBBBB"

    cpt = [spl.split_long_label(x, delims=[' ', '_'], max_chars_per_line=22)
           for x in ctype_resp_sorted.index.tolist()]
    pos = np.arange(0, len(cpt))

    fig = plt.figure(figsize=(4.5, len(pos) * 0.6), facecolor='white')
    axes = fig.add_subplot(111, facecolor='white')

    # ------------------------------------------------------------------------
    # Economic loss:
    #   - Contribution to % loss of total system, by components type
    #   - Percentage econ loss for all components of a specific type
    # ------------------------------------------------------------------------

    axes.barh(
        pos - bar_width / 2.0 - bar_offset, ctype_loss_tot_mean, bar_width,
        align='center',
        color=bar_clr_1, alpha=0.7, edgecolor=None,
        label="% loss of total system value (for a component type)"
    )

    axes.barh(
        pos + bar_width / 2.0 + bar_offset, ctype_loss_by_type, bar_width,
        align='center',
        color=bar_clr_2, alpha=0.7, edgecolor=None,
        label="% loss for component type"
    )

    for p, cv in zip(pos, ctype_loss_tot_mean):
        axes.annotate(('%0.1f' % np.float(cv)) + '%',
                      xy=(cv + 0.7, p - bar_width / 2.0 - bar_offset),
                      xycoords=('data', 'data'),
                      ha='left', va='center', size=8, color=bar_clr_1,
                      annotation_clip=False)

    for p, cv in zip(pos, ctype_loss_by_type):
        axes.annotate(('%0.1f' % np.float(cv)) + '%',
                      xy=(cv + 0.7, p + bar_width / 2.0 + bar_offset * 2),
                      xycoords=('data', 'data'),
                      ha='left', va='center', size=8, color=bar_clr_2,
                      annotation_clip=False)

    axes.annotate(
        "ECONOMIC LOSS % by COMPONENT TYPE",
        xy=(0.0, -1.65), xycoords=('axes fraction', 'data'),
        ha='left', va='top',
        size=10, color='k', weight='bold',
        annotation_clip=False)
    axes.annotate(
        "Hazard: " + hazard_type + " " + scenario_tag,
        xy=(0.0, -1.2), xycoords=('axes fraction', 'data'),
        ha='left', va='top',
        size=10, color='slategrey', weight='bold',
        annotation_clip=False)

    lgnd = axes.legend(loc="upper left", ncol=1, bbox_to_anchor=(-0.1, -0.04),
                       borderpad=0, frameon=0,
                       prop={'size': 8, 'weight': 'medium'})
    for text in lgnd.get_texts():
        text.set_color('#555555')
    axes.axhline(y=pos.max() + bar_width * 2.4, xmin=0, xmax=0.4,
                 lw=0.6, ls='-', color='#444444', clip_on=False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    spines_to_remove = ['top', 'bottom', 'right']
    for spine in spines_to_remove:
        axes.spines[spine].set_visible(False)
    axes.spines['left'].set_color(grid_clr)
    axes.spines['left'].set_linewidth(0.5)

    axes.set_xlim(0, 100)
    axes.set_xticks(np.linspace(0, 100, 5, endpoint=True))
    axes.set_xticklabels([' '] * 5)
    axes.xaxis.grid(False)

    axes.set_ylim([pos.max() + bar_width * 1.5, pos.min() - bar_width * 1.5])
    axes.set_yticks(pos)
    axes.set_yticklabels(cpt, size=8, color='k')
    axes.yaxis.grid(False)

    axes.tick_params(top=False, bottom=False, left=False, right=False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fig.savefig(
        os.path.join(output_path, fig_name),
        format='png', bbox_inches='tight', dpi=400
    )

    plt.close(fig)

# *****************************************************************************


def draw_component_loss_barchart_s2(ctype_resp_sorted,
                                    scenario_tag,
                                    hazard_type,
                                    output_path,
                                    fig_name):
    """ Plots bar charts of direct economic losses for components types """

    ctype_loss_tot_mean = ctype_resp_sorted['loss_tot'].values * 100
    ctype_loss_mean_by_type = ctype_resp_sorted['loss_per_type'].values * 100
    ctype_loss_std_by_type = ctype_resp_sorted['loss_per_type_std'].values * 100

    bar_width = 0.35
    bar_space_index = 2.0
    bar_clr_1 = '#3288BD'
    grid_clr = 'dimgrey'
    header_size = 9

    comptypes_sorted = \
        [spl.split_long_label(x, delims=[' ', '_'], max_chars_per_line=18)
         for x in ctype_resp_sorted.index.tolist()]

    barpos = np.linspace(0, len(comptypes_sorted) * bar_width * bar_space_index,
                         len(comptypes_sorted))

    fig = plt.figure(figsize=(5.0, len(barpos) * 0.6), facecolor='white')
    num_grids = 7 + len(comptypes_sorted)
    gs = gridspec.GridSpec(num_grids, 1)
    ax1 = plt.subplot(gs[:-3])
    ax2 = plt.subplot(gs[-2:])
    ax1.set_facecolor('w')
    ax2.set_facecolor('w')

    # ==========================================================================
    # Percentage of lost value for each `component type`
    # ==========================================================================
    # if bar_width <1:
    #     bar_width=1

    ax1.barh(
        barpos, 100, bar_width,
        align='center',
        color='gainsboro', alpha=0.7, edgecolor=None,
        label='',
    )
    ax1.barh(
        barpos, ctype_loss_mean_by_type, bar_width,
        align='center',
        color=bar_clr_1, alpha=0.75, edgecolor=None,
        label="% loss for component type",
        xerr=ctype_loss_std_by_type,
        error_kw={
            'ecolor': 'lightcoral',
            'capsize': 2,
            'elinewidth': 0.7,
            'markeredgewidth': 0.7}
    )

    for p, cv in zip(barpos, ctype_loss_mean_by_type):
        ax1.annotate(('%0.1f' % np.float(cv)) + '%',
                     xy=(cv + 0.7, p - bar_width * 0.8),
                     xycoords=('data', 'data'),
                     ha='left', va='center',
                     size=8, color=bar_clr_1,
                     annotation_clip=False)

    ax1.annotate(
        "HAZARD: " + hazard_type + " " + scenario_tag,
        xy=(0.0, -1.2), xycoords=('axes fraction', 'data'),
        ha='left', va='top',
        size=header_size, color='slategrey', weight='bold',
        annotation_clip=False)

    ax1.annotate(
        "LOSS METRIC: % loss for each Component Type",
        xy=(0.0, -0.8), xycoords=('axes fraction', 'data'),
        ha='left', va='top',
        size=header_size, color='k', weight='bold',
        annotation_clip=False)

    ax1.axhline(y=barpos.max() + bar_width * (bar_space_index - 0.5),
                xmin=0, xmax=0.25, lw=0.6, ls='-',
                color='grey', clip_on=False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    spines_to_remove = ['top', 'bottom', 'right', 'left']
    for spine in spines_to_remove:
        ax1.spines[spine].set_visible(False)

    ax1.xaxis.grid(False)
    ax1.set_xlim(0, 100)
    ax1.set_xticks(np.linspace(0, 100, 5, endpoint=True))
    ax1.set_xticklabels([' '] * 5)

    ax1.yaxis.grid(False)
    ax1.set_ylim([barpos.max() + bar_width, barpos.min() - bar_width])
    ax1.set_yticks(barpos)
    ax1.set_yticklabels(comptypes_sorted, size=7, color='k')

    ax1.tick_params(top=False, bottom=False, left=False, right=False)
    ax1.yaxis.set_tick_params(
        which='major',
        labelleft=True,
        labelright=False,
        labelsize=8,
        pad=18)

    # ==========================================================================
    # Stacked Bar Chart: demonstrates relative contribution of `component types`
    # to the aggregated direct economic loss of the system.
    # ==========================================================================

    stacked_bar_width = 0.5

    colours = spl.ColourPalettes()
    if len(comptypes_sorted) <= 11:
        COLR_SET = colours.BrewerSpectral
    elif len(comptypes_sorted) <= 20:
        COLR_SET = colours.Trubetskoy
    else:
        COLR_SET = colours.Tartarize269

    scaled_tot_loss = [x * 100 / sum(ctype_loss_tot_mean)
                       if sum(ctype_loss_tot_mean) > 0 else 0
                       for x in ctype_loss_tot_mean]

    leftpos = 0.0
    for ind, loss, pos in zip(list(range(len(comptypes_sorted))),
                              scaled_tot_loss,
                              barpos):
        ax2.barh(0.0, loss, stacked_bar_width,
                 left=leftpos, color=COLR_SET[ind])
        leftpos += loss
        ax1.annotate(u"$\u25FC$", xy=(-2.0, pos),
                     xycoords=('data', 'data'),
                     color=COLR_SET[ind], size=10,
                     ha='right', va='center',
                     annotation_clip=False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ax2.xaxis.grid(False)
    ax2.set_xticklabels([])
    ax2.set_xlim(0, 100)
    agg_sys_loss = sum(ctype_resp_sorted['loss_tot'].values)
    ax2.set_xlabel(
        "Aggregated loss as fraction of System Value: {:.1f}% "
        .format(round(agg_sys_loss * 100)), size=8, labelpad=12)

    ax2.yaxis.grid(False)
    ax2.set_yticklabels([])
    ax2.set_ylim([-stacked_bar_width * 0.5, stacked_bar_width * 1.1])

    ax2.tick_params(top=False, bottom=False, left=False, right=False)

    ax2.annotate(
        "LOSS METRIC: relative contributions to system loss",
        xy=(0.0, 0.7), xycoords=('axes fraction', 'data'),
        ha='left', va='top',
        size=header_size, color='k', weight='bold',
        annotation_clip=False)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax2.spines[spine].set_visible(False)

    arrowprops = dict(arrowstyle="|-|, widthA=0.4, widthB=0.4",
                      linewidth=0.7,
                      facecolor=grid_clr, edgecolor=grid_clr,
                      shrinkA=0, shrinkB=0,
                      clip_on=False)
    ax2.annotate('', xy=(0.0, -stacked_bar_width),
                 xytext=(1.0, -stacked_bar_width),
                 xycoords=('axes fraction', 'data'),
                 arrowprops=arrowprops)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fig.savefig(
        os.path.join(output_path, fig_name),
        format='png', bbox_inches='tight', dpi=400
    )

    plt.close(fig)

# *****************************************************************************


def draw_component_loss_barchart_s3(ctype_resp_sorted,
                                    scenario_tag,
                                    hazard_type,
                                    output_path,
                                    fig_name):
    """ Plots bar charts of direct economic losses for components types """

    ctype_loss_tot_mean = ctype_resp_sorted['loss_tot'].values * 100
    ctype_loss_mean_by_type = ctype_resp_sorted['loss_per_type'].values * 100
    ctype_loss_std_by_type = ctype_resp_sorted['loss_per_type_std'].values * 100

    bar_width = 0.35
    bar_clr_1 = spl.ColourPalettes().BrewerSet1[0]  # '#E41A1C'
    grid_clr = "dimgrey"
    bg_box_clr = 'gainsboro'

    comptypes_sorted = ctype_resp_sorted.index.tolist()
    barpos = np.arange(0, len(comptypes_sorted))

    fig = plt.figure(figsize=(5.0, len(barpos) * 0.6), facecolor='white')
    num_grids = 8 + len(comptypes_sorted)
    gs = gridspec.GridSpec(num_grids, 1)
    ax1 = plt.subplot(gs[:-4], facecolor='white')
    ax2 = plt.subplot(gs[-3:], facecolor='white')

    colours = spl.ColourPalettes()
    if len(comptypes_sorted) <= 11:
        COLR_SET = colours.BrewerSpectral
    elif len(comptypes_sorted) <= 20:
        COLR_SET = colours.Trubetskoy
    else:
        COLR_SET = colours.Tartarize269

    # ==========================================================================
    # Percentage of lost value for each `component type`
    # ==========================================================================

    ax1.barh(
        barpos, 100, bar_width,
        align='center',
        color=bg_box_clr, alpha=0.85, edgecolor=None,
        label='',
    )
    ax1.barh(
        barpos, ctype_loss_mean_by_type, bar_width,
        align='center',
        color=bar_clr_1, alpha=0.75, edgecolor=None,
        label="% loss for component type",
        xerr=ctype_loss_std_by_type,
        error_kw={
            'ecolor': 'cornflowerblue',
            'capsize': 0,
            'elinewidth': 1.2,
            'markeredgewidth': 0.0}
    )

    def selective_rounding(fltval):
        return "{:>4.0f}%".format(round(fltval)) if fltval > 1.0 \
            else "{:>4.1f}%".format(fltval)

    for p, ct, loss_ct in zip(barpos,
                              comptypes_sorted,
                              ctype_loss_mean_by_type):
        ax1.annotate(
            ct,
            xy=(3, p - bar_width), xycoords=('data', 'data'),
            ha='left', va='center', size=7, color='k',
            annotation_clip=False)

        ax1.annotate(
            selective_rounding(loss_ct),
            xy=(100, p - bar_width), xycoords=('data', 'data'),
            ha='right', va='center', size=7, color=bar_clr_1,
            annotation_clip=False)

    ax1.annotate(
        "HAZARD: " + hazard_type + " " + scenario_tag,
        xy=(0.0, -1.8), xycoords=('axes fraction', 'data'),
        ha='left', va='top',
        size=9, color='slategrey', weight='bold',
        annotation_clip=False)

    ax1.annotate(
        "LOSS METRIC: % loss for each Component Type",
        xy=(0.0, -1.25), xycoords=('axes fraction', 'data'),
        ha='left', va='top',
        size=9, color='k', weight='bold',
        annotation_clip=False)

    ax1.axhline(y=barpos.max() + bar_width * 2.0,
                xmin=0, xmax=0.20,
                lw=0.6, ls='-', color='grey', clip_on=False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    spines_to_remove = ['top', 'bottom', 'right', 'left']
    for spine in spines_to_remove:
        ax1.spines[spine].set_visible(False)

    ax1.xaxis.grid(False)
    ax1.set_xlim(0, 100)
    ax1.set_xticks(np.linspace(0, 100, 5, endpoint=True))
    ax1.set_xticklabels([' '] * 5)

    ax1.yaxis.grid(False)
    ax1.set_ylim([barpos.max() + bar_width, barpos.min() - bar_width])
    ax1.set_yticks(barpos)
    ax1.set_yticklabels([])

    ax1.tick_params(top=False, bottom=False, left=False, right=False)
    ax1.yaxis.set_tick_params(
        which='major',
        labelleft=True,
        labelright=False,
        labelsize=8,
        pad=20)

    # ==========================================================================
    # Stacked Bar Chart: demonstrates relative contribution of `component types`
    # to the aggregated direct economic loss of the system.
    # ==========================================================================

    stacked_bar_width = 1.0

    scaled_tot_loss = [x * 100 / sum(ctype_loss_tot_mean)
                       if sum(ctype_loss_tot_mean) > 0 else 0
                       for x in ctype_loss_tot_mean]

    leftpos = 0.0
    rank = 1
    for ind, loss, sysloss, pos in zip(list(range(len(comptypes_sorted))),
                                       scaled_tot_loss,
                                       ctype_loss_tot_mean,
                                       barpos):
        ax2.barh(0.0, loss, stacked_bar_width,
                 left=leftpos, color=COLR_SET[ind])

        # Annotate the values for the top five contributors
        if rank <= 5:
            xt = (2 * leftpos + loss) / 2.0
            yt = stacked_bar_width * 0.6
            ax2.text(
                xt, yt, selective_rounding(sysloss),
                ha='center', va='bottom', size=7, color='k', weight='bold',
                path_effects=[PathEffects.withStroke(linewidth=1,
                                                     foreground="w")])
            rank += 1

        leftpos += loss
        ax1.annotate(u"$\u25FC$",
                     xy=(-0.4, pos - bar_width),
                     xycoords=('data', 'data'),
                     color=COLR_SET[ind], size=8,
                     ha='left', va='center',
                     annotation_clip=False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for spine in ['top', 'right', 'left', 'bottom']:
        ax2.spines[spine].set_visible(False)

    ax2.xaxis.grid(False)
    ax2.set_xticklabels([])
    ax2.set_xlim(0, 100)
    agg_sys_loss = sum(ctype_resp_sorted['loss_tot'].values)
    ax2.set_xlabel(
        "Aggregated loss as fraction of System Value: {:.0f}% "
        .format(round(agg_sys_loss * 100)), size=8, labelpad=8)

    ax2.yaxis.grid(False)
    ax2.set_yticklabels([])
    ax2.set_ylim([-stacked_bar_width * 0.7, stacked_bar_width * 1.8])

    ax2.tick_params(top=False, bottom=False, left=False, right=False)

    ax2.annotate(
        "LOSS METRIC: % system loss attributed to Component Types",
        xy=(0.0, stacked_bar_width * 1.5), xycoords=('axes fraction', 'data'),
        ha='left', va='bottom',
        size=9, color='k', weight='bold',
        annotation_clip=False)

    arrow_ypos = -stacked_bar_width * 1.2
    arrowprops = dict(arrowstyle="|-|, widthA=0.4, widthB=0.4",
                      linewidth=0.7,
                      facecolor=grid_clr, edgecolor=grid_clr,
                      shrinkA=0, shrinkB=0,
                      clip_on=False)
    ax2.annotate('', xy=(0.0, arrow_ypos * 0.9),
                 xytext=(1.0, arrow_ypos * 0.9),
                 xycoords=('axes fraction', 'data'),
                 arrowprops=arrowprops)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fig.savefig(
        os.path.join(output_path, fig_name),
        format='png', bbox_inches='tight', dpi=400
    )

    plt.close(fig)

# *****************************************************************************


def draw_component_failure_barchart(uncosted_comptypes,
                                    ctype_failure_mean,
                                    scenario_name,
                                    scenario_tag,
                                    hazard_type,
                                    output_path,
                                    figname):
    comp_type_fail_sorted = \
        ctype_failure_mean.sort_values(by=(scenario_name), ascending=False)
    cpt_failure_vals = comp_type_fail_sorted[scenario_name].values * 100

    for x in uncosted_comptypes:
        if x in comp_type_fail_sorted.index.tolist():
            comp_type_fail_sorted = comp_type_fail_sorted.drop(x, axis=0)

    cptypes = comp_type_fail_sorted.index.tolist()
    cpt = [spl.split_long_label(x, delims=[' ', '_'], max_chars_per_line=22)
           for x in cptypes]
    pos = np.arange(len(cptypes))

    fig = plt.figure(figsize=(4.5, len(pos) * 0.5),
                     facecolor='white')
    ax = fig.add_subplot(111)
    bar_width = 0.4
    bar_clr = "#D62F20"
    grid_clr = "#BBBBBB"

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    spines_to_remove = ['top', 'bottom', 'right']
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(grid_clr)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_color(grid_clr)
    ax.spines['right'].set_linewidth(0.5)

    ax.set_xlim(0, 100)
    ax.set_xticks(np.linspace(0, 100, 5, endpoint=True))
    ax.set_xticklabels([' '] * 5)
    ax.xaxis.grid(True, color=grid_clr, linewidth=0.5, linestyle='-')

    ax.set_ylim([pos.max() + bar_width, pos.min() - bar_width])
    ax.set_yticks(pos)
    ax.set_yticklabels(cpt, size=8, color='k')
    ax.yaxis.grid(False)

    ax.tick_params(top=False, bottom=False, left=False, right=False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ax.barh(pos, cpt_failure_vals, bar_width,
            color=bar_clr, alpha=0.8, edgecolor=None)

    # add the numbers to the side of each bar
    for p, cv in zip(pos, cpt_failure_vals):
        plt.annotate(('%0.1f' % cv) + '%', xy=(cv + 0.6, p),
                     va='center', size=8, color="#CA1C1D",
                     annotation_clip=False)

    ax.annotate('FAILURE RATE: % FAILED COMPONENTS by TYPE',
                xy=(0.0, -1.45), xycoords=('axes fraction', 'data'),
                ha='left', va='top', annotation_clip=False,
                size=10, weight='bold', color='k')
    ax.annotate('Hazard: ' + hazard_type + " " + scenario_tag,
                xy=(0.0, -1.0), xycoords=('axes fraction', 'data'),
                ha='left', va='top', annotation_clip=False,
                size=10, weight='bold',
                color='slategrey')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    fig.savefig(
        os.path.join(output_path, figname),
        format='png', bbox_inches='tight', dpi=400
    )

    plt.close(fig)


# *****************************************************************************

def calc_comptype_damage_scenario_given_hazard(infrastructure,
                                               scenario,
                                               hazards,
                                               ctype_resp_sorted,
                                               component_response,
                                               cp_types_costed,
                                               scenario_header):
    nodes_all = list(infrastructure.components.keys())
    nodes_costed = [x for x in nodes_all
                    if infrastructure.components[x].component_type in
                    cp_types_costed]
    nodes_costed.sort()

    comptype_num = {
        x: len(list(infrastructure.get_components_for_type(x)))
        for x in cp_types_costed
    }
    comptype_used = {x: 0 for x in cp_types_costed}

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Logic for internal component replacement / scavenging

    comptype_for_internal_replacement = {}

    for x in cp_types_costed:
        comptype_for_internal_replacement[x] = int(
            np.floor(
                (1.0 - ctype_resp_sorted.loc[x, 'num_failures']) * comptype_num[x])
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Using the HAZUS method:

    comp_rst = {t: {n: 0 for n in nodes_costed}
                for t in scenario.restoration_time_range}
    restoration_time_agg = {nc: 0 for nc in nodes_costed}
    for comp_name in nodes_costed:
        comp_obj = infrastructure.components[comp_name]
        ct = infrastructure.components[comp_name].component_type
        haz_ix = hazards.hazard_scenario_name.index(scenario_header)
        x_loc, y_loc = comp_obj.get_location()
        sc_haz_val = \
            hazards.listOfhazards[haz_ix].get_hazard_intensity(
                x_loc, y_loc)

        comptype_used[ct] += 1
        comps_avl_for_int_replacement = \
            comptype_for_internal_replacement[ct] - comptype_used[ct]

        system_damage_states = infrastructure.get_system_damage_states()
        comptype_dmg_states = comp_obj.damage_states

        if len(comptype_dmg_states) != len(system_damage_states):
            print("\n")
            rootLogger.warning(
                "%s *** Damage state levels are not matched"
                " between system and component.%s", Fore.MAGENTA, Fore.RESET)

        rootLogger.info(
            "Calculating Recovery Level for component: %s for hazard [%s]",
            comp_obj.component_id, sc_haz_val)

        for t in scenario.restoration_time_range:
            comp_rst[t][comp_name], restoration_time_agg[comp_name] = \
                comp_recovery_given_hazard_and_time(
                    comp_obj,
                    component_response,
                    scenario_header,
                    sc_haz_val,
                    t,
                    comps_avl_for_int_replacement,
                    threshold_recovery=RESTORATION_THRESHOLD,
                    threshold_exceedance=0.001)

    comp_rst_df = pd.DataFrame(comp_rst,
                               index=nodes_costed,
                               columns=scenario.restoration_time_range)

    comp_rst_time_given_haz = []
    for c in nodes_costed:
        post_rst_times = comp_rst_df.loc[c] >= RESTORATION_THRESHOLD
        if True in list(post_rst_times.array):
            time_restored = np.round(comp_rst_df.columns[post_rst_times][0], 0)
        else:
            time_restored = max(scenario.restoration_time_range) + 1
        comp_rst_time_given_haz.append(time_restored)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    component_fullrst_time = pd.DataFrame(
        {'Full Restoration Time': comp_rst_time_given_haz},
        index=nodes_costed)
    component_fullrst_time.index.names = ['component_id']

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ctype_scenario_outcomes = copy.deepcopy(
        100 * ctype_resp_sorted.drop(['func_mean', 'func_std'], axis=1)
    )

    cpmap = {c: sorted(list(infrastructure.get_components_for_type(c)))
             for c in cp_types_costed}

    rtimes = []
    for x in ctype_scenario_outcomes.index:
        rtimes.append(np.mean(component_fullrst_time.loc[cpmap[x]].values))
    ctype_scenario_outcomes['restoration_time'] = rtimes

    return component_fullrst_time, ctype_scenario_outcomes


def run_scenario_loss_analysis(scenario,
                               hazards,
                               infrastructure,
                               config,
                               input_comptype_response_file,
                               input_component_response_file):
    rootLogger.info('Start: SCENARIO LOSS ANALYSIS')

    RESTORATION_STREAMS = scenario.restoration_streams
    FOCAL_HAZARD_SCENARIOS = hazards.focal_hazard_scenarios

    # Restoration time starts x time units after hazard impact:
    # This represents lead up time for damage and safety assessments
    RESTORATION_OFFSET = 1

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Read in SIMULATED HAZARD RESPONSE for <COMPONENT TYPES>

    comptype_resp_df = pd.read_csv(
        input_comptype_response_file,
        index_col=['component_type', 'response'],
        skipinitialspace=True)
    comptype_resp_df.columns = hazards.hazard_scenario_name

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Nodes not considered in the loss calculations
    # NEED TO MOVE THESE TO A MORE LOGICAL PLACE

    uncosted_comptypes = [
        'CONN_NODE', 'JUNCTION_NODE', 'JUNCTION',
        'SYSTEM_INPUT', 'SYSTEM_OUTPUT',
        'Generation Source', 'Grounding'
    ]

    cp_types_in_system = infrastructure.get_component_types()
    cp_types_costed = [x for x in cp_types_in_system
                       if x not in uncosted_comptypes]

    comptype_resp_df.drop(
        uncosted_comptypes,
        level='component_type', axis=0, inplace=True, errors='ignore')
    comptype_resp_df = comptype_resp_df.sort_index()

    # Get list of only those components that are included in cost calculations:
    cpmap = {c: sorted(list(infrastructure.get_components_for_type(c)))
             for c in cp_types_in_system}
    comps_costed = [v for x in cp_types_costed for v in cpmap[x]]

    nodes_all = list(infrastructure.components.keys())
    nodes_all.sort()
    comps_uncosted = list(set(nodes_all).difference(comps_costed))

    ctype_failure_mean = comptype_resp_df.xs('num_failures', level='response')
    ctype_failure_mean.columns.names = ['Scenario Name']

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Value of component types relative to system value

    comptype_value_dict = {}
    for ct in sorted(cp_types_costed):
        comp_val = [infrastructure.components[comp_id].cost_fraction
                    for comp_id in cpmap[ct]]
        comptype_value_dict[ct] = sum(comp_val)

    comptype_value_list = [comptype_value_dict[ct]
                           for ct in sorted(comptype_value_dict.keys())]

    component_response = pd.read_csv(input_component_response_file,
                                     index_col=['component_id', 'response'],
                                     skiprows=0,
                                     skipinitialspace=True)

    component_meanloss = component_response.query('response == "loss_mean"'). \
        reset_index('response').drop('response', axis=1)

    weight_criteria = 'MIN_COST'

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    col_tp = []
    for h in FOCAL_HAZARD_SCENARIOS:
        col_tp.extend(
            list(zip([h] * len(RESTORATION_STREAMS), RESTORATION_STREAMS))
        )

    mcols = pd.MultiIndex.from_tuples(
        col_tp, names=['Hazard', 'Restoration Streams'])
    time_to_full_restoration_for_lines_df = pd.DataFrame(
        index=list(infrastructure.output_nodes.keys()), columns=mcols)
    time_to_full_restoration_for_lines_df.index.name = 'Output Lines'

    # --------------------------------------------------------------------------
    # *** BEGIN : FOCAL_HAZARD_SCENARIOS FOR LOOP ***
    for sc_haz_str in FOCAL_HAZARD_SCENARIOS:

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Differentiated setup based on hazard input type - scenario vs array
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        sc_haz_str = "{:.3f}".format(float(sc_haz_str))
        if str(config.HAZARD_INPUT_METHOD).lower() == "calculated_array":
            scenario_header = hazards.hazard_scenario_name[
                hazards.hazard_scenario_list.index(sc_haz_str)]
        elif str(config.HAZARD_INPUT_METHOD).lower() == "hazard_file":
            scenario_header = sc_haz_str
        else:
            raise ValueError("Unrecognised value for HAZARD_INPUT_METHOD.")

        scenario_tag = str(sc_haz_str) + " " + \
            hazards.intensity_measure_unit + " " + \
            hazards.intensity_measure_param

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Extract scenario-specific values from the 'hazard response' dataframe
        # Scenario response: by component type
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        ctype_resp_scenario = comptype_resp_df[scenario_header].unstack(
            level=-1)
        ctype_resp_scenario = ctype_resp_scenario.sort_index()

        ctype_resp_scenario['loss_per_type'] \
            = ctype_resp_scenario['loss_mean'] / comptype_value_list

        ctype_resp_scenario['loss_per_type_std'] =\
            ctype_resp_scenario['loss_std'] * \
            [len(list(infrastructure.get_components_for_type(ct)))
             for ct in ctype_resp_scenario.index.values.tolist()]

        ctype_resp_sorted = ctype_resp_scenario.sort_values(by='loss_tot',
                                                            ascending=False)

        fig_name = 'fig_SC_' + sc_haz_str + '_loss_sys_vs_comptype_s1.png'
        draw_component_loss_barchart_s1(
            ctype_resp_sorted,
            scenario_tag,
            hazards.hazard_type,
            config.OUTPUT_PATH,
            fig_name)

        fig_name = 'fig_SC_' + sc_haz_str + '_loss_sys_vs_comptype_s2.png'
        draw_component_loss_barchart_s2(
            ctype_resp_sorted,
            scenario_tag,
            hazards.hazard_type,
            config.OUTPUT_PATH,
            fig_name)

        fig_name = 'fig_SC_' + sc_haz_str + '_loss_sys_vs_comptype_s3.png'
        draw_component_loss_barchart_s3(
            ctype_resp_sorted,
            scenario_tag,
            hazards.hazard_type,
            config.OUTPUT_PATH,
            fig_name)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # FAILURE RATE -- PERCENTAGE of component types

        fig_name = 'fig_SC_' + sc_haz_str + '_comptype_failures.png'
        draw_component_failure_barchart(
            uncosted_comptypes,
            ctype_failure_mean,
            scenario_header,
            scenario_tag,
            hazards.hazard_type,
            config.OUTPUT_PATH,
            fig_name)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # RESTORATION PROGNOSIS for specified scenarios

        component_fullrst_time, ctype_scenario_outcomes = \
            calc_comptype_damage_scenario_given_hazard(
                infrastructure,
                scenario,
                hazards,
                ctype_resp_sorted,
                component_response,
                cp_types_costed,
                scenario_header)

        # All the nodes that need to be fixed for each output node:
        repair_list_combined = prep_repair_list(infrastructure,
                                                component_meanloss,
                                                component_fullrst_time,
                                                comps_uncosted,
                                                weight_criteria,
                                                scenario_header)

        repair_path = copy.deepcopy(repair_list_combined)
        output_node_list = list(infrastructure.output_nodes.keys())

        for num_rst_steams in RESTORATION_STREAMS:
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # SYSTEM RESTORATION for given scenario & restoration setup
            rst_setup_filename = 'restoration_setup_' + sc_haz_str + '_' + \
                                 hazards.intensity_measure_unit + '_' + \
                                 hazards.intensity_measure_param + '.csv'

            rst_setup_df = calc_restoration_setup(component_meanloss,
                                                  output_node_list,
                                                  comps_uncosted,
                                                  repair_list_combined,
                                                  num_rst_steams,
                                                  RESTORATION_OFFSET,
                                                  component_fullrst_time,
                                                  scenario.output_path,
                                                  scenario_header,
                                                  rst_setup_filename)

            # Check if nothing to repair, i.e. if repair list is empty:
            comps_to_repair = rst_setup_df.index.values.tolist()
            if not comps_to_repair:
                rootLogger.warning("*** For scenario: %s", scenario_tag)
                rootLogger.warning("Nothing to repair. Time to repair is zero.")
                rootLogger.warning("Skipping repair visualisation for this scenario.")
                break

            fig_rst_gantt_name = 'fig_SC_' + sc_haz_str + '_str' + \
                                 str(num_rst_steams) + '_restoration.png'
            _, time_to_full_restoration_for_lines = \
                vis_restoration_process(
                    scenario,
                    infrastructure,
                    rst_setup_df,
                    num_rst_steams,
                    repair_path,
                    fig_rst_gantt_name,
                    scenario_tag,
                    hazards.hazard_type)

            time_to_full_restoration_for_lines_df[(sc_haz_str, num_rst_steams)]\
                = [time_to_full_restoration_for_lines[x]
                   for x in output_node_list]

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # COMPONENT CRITICALITY for given scenario & restoration setup
            fig_name = 'fig_SC_' + sc_haz_str + '_str' + \
                       str(num_rst_steams) + '_component_criticality.png'
            component_criticality(
                infrastructure,
                scenario,
                ctype_scenario_outcomes,
                hazards.hazard_type,
                scenario_tag,
                fig_name)

    # --------------------------------------------------------------------------
    # *** END : FOCAL_HAZARD_SCENARIOS FOR LOOP ***

    time_to_full_restoration_for_lines_csv = \
        os.path.join(config.OUTPUT_PATH, 'line_restoration_prognosis.csv')
    time_to_full_restoration_for_lines_df.to_csv(
        time_to_full_restoration_for_lines_csv, sep=',')

    rootLogger.info('End: SCENARIO LOSS ANALYSIS')
    # --------------------------------------------------------------------------
