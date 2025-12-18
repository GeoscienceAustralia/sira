import copy
import itertools
import logging
import os
import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore, init
from matplotlib import cm, gridspec, transforms

import sira.tools.siraplot as spl

init()
rootLogger = logging.getLogger(__name__)
np.seterr(divide="print", invalid="raise")
plt.switch_backend("agg")
sns.set_theme(style="whitegrid", palette="coolwarm")

#######################################################################################
RESTORATION_THRESHOLD = 0.95
#######################################################################################


def calc_component_recovery_time(
    config,
    component,
    event_id: str,
    hazval,
    threshold_recovery=RESTORATION_THRESHOLD,
    threshold_damage=0.01,
    comps_avl_for_int_replacement=0,
) -> int:
    """
    Calculates the time needed for a component to reach the recovery threshold
    """
    damage_functions = [ds.response_function for ds in component.damage_states.values()]
    recovery_functions = [ds.recovery_function for ds in component.damage_states.values()]

    # comp_functionality = component_response.loc[
    #     event_id, (component.component_id, "func_mean")
    # ]

    comptype_dmg_states = component.damage_states
    num_dmg_states = len(component.damage_states)

    # -----------------------------------------------------------------------------------
    # Calculate damage exceedance probabilities
    pe = np.array([damage_functions[d](hazval) for d in range(num_dmg_states)])

    # Calculate probability of being in each damage state
    pb = np.zeros(num_dmg_states)
    pb[0] = 1.0 - pe[1]  # Probability of no damage
    for d in range(1, num_dmg_states - 1):
        pb[d] = pe[d] - pe[d + 1]
    pb[-1] = pe[-1]  # Probability of being in worst damage state

    # -----------------------------------------------------------------------------------
    # Build discretised recovery functions

    # t_max = config.RESTORATION_TIME_MAX
    # t_step = config.RESTORATION_TIME_STEP
    if config is not None:
        if hasattr(config, "get"):
            t_max = float(config.get("RESTORATION_TIME_MAX", 365))
            t_step = float(config.get("RESTORATION_TIME_STEP", 1))
            t_min = float(config.get("RESTORATION_TIME_MIN", 1))
        elif hasattr(config, "RESTORATION_TIME_MAX"):
            t_max = float(getattr(config, "RESTORATION_TIME_MAX", 365))
            t_step = float(getattr(config, "RESTORATION_TIME_STEP", 1))
            t_min = float(getattr(config, "RESTORATION_TIME_MIN", 1))

    # t_min = 0.0

    # Build time array according to config: 1, 1+step, ..., up to max
    if t_step <= 0:
        t_step = 1.0
    if t_max <= 1.0:
        t_max = 365.0

    n_steps = int(np.floor((t_max - t_min) / t_step)) + 1  # type: ignore
    time_array = np.linspace(t_min, t_max, n_steps)
    time_array = time_array[time_array <= t_max]

    functionality_array = np.empty((len(pb), len(time_array)), dtype=np.float64)

    for d, ds in enumerate(comptype_dmg_states):
        if d == 0:
            functionality_array[d] = np.ones(len(time_array))
        else:
            functionality_array[d] = np.round(recovery_functions[d](time_array), 2)

    # -----------------------------------------------------------------------------------
    # Calculate expected recovery time - discretised interpolation method

    # Try to discretise the provided parametric recovery function directly
    # baseline_func = float(getattr(self, "functionality", 0.0) or 0.0)

    pb_T = np.array(pb).reshape(-1, 1)
    functionality_agg = np.sum(pb_T * functionality_array, axis=0)
    restoration_time_agg = np.interp(threshold_recovery, functionality_agg, time_array)

    if (
        np.isnan(restoration_time_agg)
        or restoration_time_agg < 0
        or pb[0] >= (1.0 - threshold_damage)  # No damage
    ):
        restoration_time_agg = 0

    # print(f"--> restoration_time_agg {comp_id} at {hazval} : {restoration_time_agg}")
    return int(round(restoration_time_agg))


def analyse_system_recovery(
    infrastructure,
    config,
    hazard_obj,
    event_id,
    comps_costed,
    verbosity=True,
):
    """
    Analyses system recovery for specified hazard scenario
    """

    # Calculate recovery time for each component
    recovery_times = OrderedDict()
    nodes_with_recovery = set()
    event_id = str(event_id)

    for comp_id in comps_costed:
        # hazard_obj = hazards.listOfhazards[hazards.hazard_scenario_list.index(event_id)]
        comp_obj = infrastructure.components[comp_id]
        loc_params = comp_obj.get_location()
        sc_haz_val = hazard_obj.get_hazard_intensity(*loc_params)

        # # Test prints
        # print("========================================")
        # print(f"comp_id    : {comp_id}")
        # print(f"loc_params : {loc_params}")
        # print(f"event_id   : {event_id}\n")

        try:
            comp_restoration_time = calc_component_recovery_time(
                config,
                comp_obj,
                event_id,
                sc_haz_val,
                threshold_recovery=RESTORATION_THRESHOLD,
            )
            if (
                isinstance(comp_restoration_time, (int, float))
                and comp_restoration_time is not None
            ):
                recovery_times[comp_id] = comp_restoration_time
            else:
                recovery_times[comp_id] = 0

            if comp_restoration_time > 0:
                nodes_with_recovery.add(comp_id)

        except Exception as e:
            rootLogger.error(f"*** Recovery time calculation failed for component {comp_id}: {e}")
            recovery_times[comp_id] = 0

    # Create recovery timeline dataframe with set structure
    # component_fullrst_times = pd.DataFrame(
    #     {"Full Restoration Time": list(recovery_times.values())},
    #     index=pd.Index(list(recovery_times.keys()), name="component_id"),
    # )

    if recovery_times:
        component_fullrst_times = pd.DataFrame(
            {"Full Restoration Time": list(recovery_times.values())},
            index=list(recovery_times.keys()),
        )
    else:
        # If no components need repair, create empty DataFrame with same structure
        component_fullrst_times = pd.DataFrame(
            columns=["Full Restoration Time"], index=pd.Index([], name="component_id")
        )

    component_fullrst_times.index.names = ["component_id"]

    return component_fullrst_times


# =====================================================================================


def calc_restoration_schedule(
    component_recovery_times, repair_dict_combined, num_rst_streams, restoration_offset=1
):
    """
    Creates an optimised restoration schedule based on
    expected calculated recovery times
    """
    schedule_cols = [
        "OutputNode",
        "RestorationTimes",
        "RstStart",
        "RstEnd",
        "Priority",
        "RepairStream",
        "AssociatedLines",
    ]

    schedule = []  # Initialise schedule list for DataFrame
    component_assignments = {}  # Track which line first repairs each component
    component_lines = {}  # Track all output lines associated with components

    # First pass - collect all lines associated with each component
    for output_node, repair_data in repair_dict_combined.items():
        if repair_data["undamaged"]:
            continue
        # Identify shared components across lines and first repair time
        components = repair_data["residual_repair_list"]

        ordered_components = sorted(
            components,
            key=lambda x: component_recovery_times.loc[x, "Full Restoration Time"],
            reverse=True,
        )

        for comp in ordered_components:
            if comp not in component_lines:
                component_lines[comp] = []
            component_lines[comp].append(output_node)

            # for comp in ordered_components:
            if comp not in component_assignments:
                component_assignments[comp] = output_node
                recovery_time = component_recovery_times.loc[comp, "Full Restoration Time"]
                if recovery_time > 0:
                    schedule.append(
                        {
                            "NodesToRepair": comp,
                            "OutputNode": output_node,
                            "RestorationTimes": recovery_time,
                            "FirstRepair": component_assignments[comp] == output_node,  # True
                            "RstStart": None,
                            "RstEnd": None,
                            "Priority": None,
                            "RepairStream": None,
                            "AssociatedLines": None,  # To be added later
                        }
                    )

    # Add associated lines information
    for entry in schedule:
        comp = entry["NodesToRepair"]
        entry["AssociatedLines"] = ", ".join(sorted(component_lines[comp]))

    schedule_df = pd.DataFrame(schedule)
    if schedule_df.empty:
        return pd.DataFrame(columns=schedule_cols)

    # Assign priorities based on original order - by restoration time
    schedule_df["Priority"] = range(1, len(schedule_df) + 1)

    # Set index before assignment for easier .at usage
    schedule_df.set_index("NodesToRepair", inplace=True)

    current_time = restoration_offset
    stream_end_times = [current_time] * num_rst_streams
    component_end_times = {}

    for comp, row in schedule_df.iterrows():
        if row["FirstRepair"]:
            stream_idx = stream_end_times.index(min(stream_end_times))
            start_time = stream_end_times[stream_idx]

            schedule_df.at[comp, "RstStart"] = start_time  # pyright: ignore[reportArgumentType]
            schedule_df.at[comp, "RstEnd"] = start_time + row["RestorationTimes"]  # type: ignore[reportArgumentType]
            schedule_df.at[comp, "RepairStream"] = stream_idx  # pyright: ignore[reportArgumentType]

            component_end_times[comp] = schedule_df.at[comp, "RstEnd"]  # pyright: ignore[reportArgumentType]
            stream_end_times[stream_idx] = schedule_df.at[comp, "RstEnd"]  # pyright: ignore[reportArgumentType]

        else:
            # This component is repaired as part of another line
            schedule_df.at[comp, "RstStart"] = 0  # pyright: ignore[reportArgumentType]
            schedule_df.at[comp, "RstEnd"] = component_end_times[comp]  # pyright: ignore[reportArgumentType]
            schedule_df.at[comp, "RepairStream"] = 0  # pyright: ignore[reportArgumentType]

    # Ensure correct dtypes for columns
    schedule_df = schedule_df.astype({"RstStart": float, "RstEnd": float, "RepairStream": int})

    return schedule_df


#######################################################################################


def calculate_system_recovery_curve(schedule_df, infrastructure, output_dict, xmax):
    """
    Calculates the system recovery curve based on component restoration schedule
    """
    timeline = np.arange(0, xmax + 1)
    recovery_curve = np.zeros_like(timeline, dtype=float)

    for t in timeline:
        restored_components = schedule_df[schedule_df["RstEnd"] <= t].index
        system_capacity = calculate_system_capacity(
            infrastructure, output_dict, restored_components
        )
        recovery_curve[t] = system_capacity

    return recovery_curve


#######################################################################################


def make_recovery_process_diagrams(
    scenario,
    config,
    infrastructure,
    schedule_df,
    num_rst_streams,
    repair_dict_init,
    repair_dict_combined,
    hazard_type,
    scenario_tag,
    fig_file,
):
    """
    --------------------------------------------------------------------------
    Produces a recovery gantt chart based on dynamic scheduling

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
    sns.set_theme(style="white")

    # Setup colors and styling
    gainsboro = "#DCDCDC"
    whitesmoke = "#F5F5F5"
    lineht = 11

    # Get component and output information
    comps = schedule_df.index.values.tolist()
    output_dict = infrastructure.output_nodes
    out_node_list = sorted(list(infrastructure.output_nodes.keys()))

    # Check if nothing to repair, i.e. if repair list is empty:
    if not comps:
        print("Nothing to repair. Skipping visualisations.")
        return None, {}

    # ---------------------------------------------------------------------------------
    # Calculate figure dimensions

    max_time = schedule_df["RstEnd"].max()

    xstep = 10
    xbase = 5.0
    xmax = int(xstep * np.ceil(1.01 * max_time / float(xstep))) + xstep

    # Adjust step size for readability
    if (xmax / xstep) > 15:
        xstep = int(xbase * round((xmax / 10.0) / xbase))

    if xmax <= xstep:
        xstep = 1
    elif xmax == 0:
        xmax = 2
        xstep = 1

    xtiks = list(range(0, xmax + 1, xstep))
    if xmax > xtiks[-1]:
        xtiks.append(xmax)

    # --------------------------------------------------------------------------
    # Set up figure with two subplots

    y_comps = list(range(1, len(comps) + 1))
    priorities = schedule_df["Priority"].values.tolist()

    hw_ratio_ax2 = 1.0 / 3.3
    fig_w_cm = 9.0
    fig_h_cm = (fig_w_cm * hw_ratio_ax2) * (1.55 + len(y_comps) / 7.5)
    num_grids = 11 + len(y_comps)

    fig = plt.figure(facecolor="white", figsize=(fig_w_cm, fig_h_cm))
    gs = gridspec.GridSpec(num_grids, 1)

    gx = len(y_comps) + 2
    ticklabelsize = 12

    linecolour1 = "xkcd:dark sky blue"

    # --------------------------------------------------------------------------
    # Subplot 1: Gantt chart of repairs
    # --------------------------------------------------------------------------
    ax1 = plt.subplot(gs[:gx])

    ax1.hlines(
        priorities,
        schedule_df["RstStart"],
        schedule_df["RstEnd"],
        linewidth=lineht,
        color=linecolour1,
    )

    ax1.set_ylim((0.5, max(y_comps) + 0.5))
    ax1.set_yticks(y_comps)
    ax1.set_yticklabels(comps, size=ticklabelsize)

    ax1.set_xlim((0, max(xtiks)))
    ax1.set_xticks(xtiks)
    ax1.set_xticklabels([])

    for i in range(0, xmax + 1, xstep):
        ax1.axvline(i, color="white", linewidth=0.5)
    ax1.yaxis.grid(True, which="major", linestyle="-", linewidth=lineht, color=whitesmoke)

    spines_to_remove = ["left", "top", "right", "bottom"]
    for spine in spines_to_remove:
        ax1.spines[spine].set_visible(False)

    ax1.annotate(
        f"Restoration Timelines for: {hazard_type} {scenario_tag}\n"
        f"{str(num_rst_streams)} Repair Streams",
        xy=(0, len(comps) + 1.8),
        # xycoords=('axes fraction', 'data'),
        xycoords="data",
        ha="left",
        va="top",
        annotation_clip=False,
        color="k",
        size=14,
        weight="normal",
    )

    # --------------------------------------------------------------------------
    # Subplot 2: System recovery curve
    # --------------------------------------------------------------------------
    ax2 = plt.subplot(gs[-8:])

    sns.axes_style(style="ticks")
    sns.despine(ax=ax2, left=True)

    ax2.set_ylim((0, 100))
    ax2.set_yticks(list(range(0, 101, 20)))
    ax2.set_yticklabels([str(x) for x in range(0, 101, 20)], size=ticklabelsize)
    ax2.yaxis.grid(True, which="major", color=gainsboro)

    ax2.set_xlim((0, max(xtiks)))
    ax2.set_xticks(xtiks)

    xtik_labels = [str(x) for x in xtiks]
    if (xtiks[-2] - xtiks[-1]) < xstep:
        xtik_labels[-1] = ""
    ax2.set_xticklabels(xtik_labels, size=ticklabelsize, rotation=0)
    ax2.tick_params(axis="x", which="major", bottom=True, length=4)

    ax2.set_xlabel("Restoration Time (" + scenario.time_unit + ")", size=14)
    ax2.set_ylabel("System Capacity (%)", size=14)

    # ---------------------------------------------------------------------------------
    # Calculate system recovery curve
    # ---------------------------------------------------------------------------------

    supply_rst_clr = "#377EB8"
    restoration_timeline_array = np.zeros((len(out_node_list), max(xtiks) + 1))

    for x, onode in enumerate(out_node_list):
        restoration_arr_template = np.zeros(max(xtiks) + 1)
        init_damage_list = repair_dict_combined[onode]["initial_damage_list"]
        nodes_to_repair_for_line = list(schedule_df.index.intersection(init_damage_list))

        if not nodes_to_repair_for_line or repair_dict_combined[onode]["undamaged"]:
            # Line is fully operational
            repair_time_for_line = 0
            restoration_arr_template = np.ones(max(xtiks) + 1)
        else:
            # Find the latest repair time for any component needed by this line
            # repair_start = schedule_df.loc[nodes_to_repair_for_line]["RstStart"].min()
            repair_end = schedule_df.loc[nodes_to_repair_for_line]["RstEnd"].max()
            repair_time_for_line = repair_end
            restoration_arr_template[int(repair_time_for_line) :] = 1.0

        restoration_timeline_array[x, :] = (
            100 * output_dict[onode]["capacity_fraction"] * restoration_arr_template
        )

        # -----------------------------------------------------------------------------
        if repair_time_for_line != 0:
            ax1.axvline(repair_time_for_line, linestyle=":", color=supply_rst_clr, alpha=0.8)
            ax2.axvline(repair_time_for_line, linestyle=":", color=supply_rst_clr, alpha=0.8)

            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            # Future improvement: restore these labels after fixing their positioning.
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            # ax2.annotate(onode, xy=(repair_time_for_line, 105),
            #              ha='center', va='bottom', rotation=90,
            #              size=ticklabelsize, color='k',
            #              annotation_clip=False)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # -----------------------------------------------------------------------------

    recovery_curve = np.sum(restoration_timeline_array, axis=0)

    recovery_x = np.arange(0, max(xtiks) + 1, step=1)
    ax2.step(recovery_x, recovery_curve, where="post", color=supply_rst_clr, clip_on=False)
    fill_between_steps(
        ax2, recovery_x, recovery_curve, 0, step_where="post", alpha=0.25, color=supply_rst_clr
    )

    if not (fig.get_figwidth() == 0 or fig.get_figheight() == 0):
        fig.savefig(fig_file, format="png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------------------------
    output_line_restoration_times = calculate_restoration_times(
        schedule_df, out_node_list, repair_dict_combined
    )

    return recovery_curve, output_line_restoration_times


# =====================================================================================


def calculate_restoration_times(
    schedule_df: pd.DataFrame, out_node_list: List[str], repair_dict_combined: OrderedDict
) -> Dict[str, float]:
    """
    Calculate the restoration time for each output node
    """
    restoration_times = {}
    for node in out_node_list:
        node_components = schedule_df[schedule_df["OutputNode"] == node]
        if not node_components.empty:
            restoration_times[node] = node_components["RstEnd"].max()
        else:
            restoration_times[node] = 0
        # Update restoration times based on `fixed_with_prev` information
        prev_line = repair_dict_combined[node].get("fixed_with_prev")
        if prev_line:
            restoration_times[node] = restoration_times[prev_line]

    return restoration_times


def calc_outcomes_for_component_types(
    infrastructure, component_fullrst_times, ctype_resp_sorted, cp_types_costed
):
    ctype_scenario_outcomes = copy.deepcopy(
        100 * ctype_resp_sorted.drop(["func_mean", "func_std"], axis=1)
    )
    cpmap = {c: sorted(list(infrastructure.get_components_for_type(c))) for c in cp_types_costed}
    rtimes = []
    for x in ctype_scenario_outcomes.index:
        rtimes.append(np.mean(component_fullrst_times.loc[cpmap[x]].values))
    ctype_scenario_outcomes["restoration_time"] = rtimes
    return ctype_scenario_outcomes


def perform_recovery_analysis(
    infrastructure,
    scenario,
    config,
    hazards,
    event_id,
    component_response,
    components_costed,
    components_uncosted,
    output_path,
    scenario_tag,
    scenario_num=None,
    weight_criteria="MIN_TIME",
    verbosity=True,
):
    """
    Main function to perform the complete recovery analysis
    """
    results = {}

    # Calculate component recovery times
    hazard_obj = hazards.listOfhazards[hazards.hazard_scenario_list.index(event_id)]
    component_recovery_times = analyse_system_recovery(
        infrastructure,
        config,
        hazard_obj,
        event_id,
        components_costed,
        verbosity=verbosity,
    )

    # Calculate repair paths and priorities
    repair_dict_init, repair_dict_combined = prep_repair_list(
        infrastructure,
        component_response,
        component_recovery_times,
        components_uncosted,
        event_id,
        weight_criteria=weight_criteria,
        verbose=verbosity,
    )

    if isinstance(scenario_num, int):
        sc_ix = str(scenario_num) + "_"
    else:
        sc_ix = ""

    # For each number of restoration streams
    for num_streams in scenario.restoration_streams:
        # Create restoration schedule
        schedule_df = calc_restoration_schedule(
            component_recovery_times, repair_dict_combined, num_streams, restoration_offset=1
        )

        # Generate visualisation
        fig_file = Path(output_path, f"sc_{sc_ix}{event_id}_RECOVERY__streams_{num_streams}.png")

        recovery_curve, system_restoration_times = make_recovery_process_diagrams(
            scenario,
            config,
            infrastructure,
            schedule_df,
            num_streams,
            repair_dict_init,
            repair_dict_combined,
            hazards.hazard_type,
            scenario_tag,
            fig_file,
        )

        # Store results
        results[(event_id, num_streams)] = {
            "schedule": schedule_df,
            "recovery_curve": recovery_curve,
            "system_restoration_times": system_restoration_times,
        }

    return results, component_recovery_times


# =====================================================================================
# Helper function for FUTURE IMPLEMENTATION


def calculate_system_capacity(
    infrastructure, output_dict: Dict, restored_components: List[str]
) -> float:
    """
    Calculate the system capacity given a set of restored components
    This is a placeholder - implement based on your system's specific logic
    """
    # This needs to be implemented according to your system's specific rules
    # for calculating capacity based on component states
    return 0.0


# =====================================================================================


def fill_between_steps(ax, x, y1, y2=0, step_where="pre", **kwargs):
    """
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
    ret : PolyCollection, The added artist

    """
    if step_where not in {"pre", "post", "mid"}:
        raise ValueError(
            "Step position must be one of {{'pre', 'post', 'mid'}} You passed in {wh}".format(
                wh=step_where
            )
        )

    # make sure y-values are up-converted to arrays
    if np.isscalar(y1):
        y1 = np.ones_like(x) * y1

    if np.isscalar(y2):
        y2 = np.ones_like(x) * y2

    # temporary array for up-converting the values to step corners
    # 3 x 2N - 1 array

    vertices = np.vstack((x, y1, y2))

    # this logic is lifted from lines.py
    # this should probably be centralised someplace
    if step_where == "pre":
        steps = np.zeros((3, 2 * len(x) - 1), float)
        steps[0, 0::2], steps[0, 1::2] = vertices[0, :], vertices[0, :-1]
        steps[1:, 0::2], steps[1:, 1:-1:2] = vertices[1:, :], vertices[1:, 1:]

    elif step_where == "post":
        steps = np.zeros((3, 2 * len(x) - 1), float)
        steps[0, ::2], steps[0, 1:-1:2] = vertices[0, :], vertices[0, 1:]
        steps[1:, 0::2], steps[1:, 1::2] = vertices[1:, :], vertices[1:, :-1]

    elif step_where == "mid":
        steps = np.zeros((3, 2 * len(x)), float)
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


# =====================================================================================


def prep_repair_list(
    infrastructure_obj,
    component_meanloss,
    component_fullrst_times,
    uncosted_comps,
    scenario_header,
    recovery_time_threshold=0.5,
    weight_criteria="MIN_TIME",
    verbose=True,
):
    """
    Identify the shortest component repair list to restore supply to output.
    Enensure components are not duplicated across repair lists.

    This is done based on:
    [1] the priority assigned to the output line
    [2] a weighting criterion applied to each node in the system

    Parameters:
    -----------
    infrastructure_obj : Object
        Infrastructure system object containing component connectivity info
    component_meanloss : DataFrame
        Component loss values
    component_fullrst_times : DataFrame
        Component restoration times
    uncosted_comps : List
        List of components not included in cost calculations
    weight_criteria : str
        Criterion for prioritising repairs ('MIN_TIME' or 'MIN_COST')
    scenario_header : str
        Identifier for the scenario being analysed
    """

    if verbose:
        print("Initiating analysis for prioritised repair schedule...")
    G = infrastructure_obj.get_component_graph()
    input_dict = infrastructure_obj.supply_nodes
    output_dict = infrastructure_obj.output_nodes

    # ---------------------------------------------------------------------------------
    df_FRT = component_fullrst_times.copy()
    all_graph_nodes = set(df_FRT.index.values)  # set(G.vs['name'])
    uncosted_set = set(uncosted_comps)

    # recovery_nodes = set(df_FRT.index)
    recovery_nodes = set(
        df_FRT.loc[df_FRT["Full Restoration Time"] >= recovery_time_threshold].index
    )

    # identify if nodes do not have recovery data
    nodes_without_recovery = all_graph_nodes - recovery_nodes - uncosted_set
    if nodes_without_recovery:
        rootLogger.warning(
            Fore.RED
            + "\nFound nodes without recovery times that are not in uncosted list:"
            + f"\n{nodes_without_recovery}\n"
            + Fore.RESET
        )
        uncosted_comps = list(uncosted_set.union(nodes_without_recovery))

    # ---------------------------------------------------------------------------------
    commodity_types = list(set([input_dict[i]["commodity_type"] for i in list(input_dict.keys())]))
    nodes_by_commoditytype = {}
    for comm_type in commodity_types:
        nodes_by_commoditytype[comm_type] = [
            x for x in list(input_dict.keys()) if input_dict[x]["commodity_type"] == comm_type
        ]

    # ---------------------------------------------------------------------------------
    # Get initial list of output nodes and dependency nodes

    out_node_list = sorted(list(output_dict.keys()))
    dependency_node_list = [
        node_id
        for node_id, infodict in list(infrastructure_obj.components.items())
        if infrastructure_obj.components[node_id].node_type == "dependency"
    ]

    # ---------------------------------------------------------------------------------
    # First, identify fully operational lines by checking their required components

    operational_lines = {}
    all_paths = {}  # Store paths for each line to avoid recalculating

    if verbose:
        print("\nChecking operational status of output lines...")
    for onode in out_node_list:
        onode_vertex = G.vs.find(onode)
        paths = []

        # Get dependency paths
        for dnode in dependency_node_list:
            try:
                dnode_paths = G.get_shortest_paths(G.vs.find(dnode), to=onode_vertex, mode="OUT")
                if dnode_paths and dnode_paths[0]:
                    paths.extend(dnode_paths[0])
            except ValueError:
                continue

        # Get supply paths
        for commodity_type in nodes_by_commoditytype:
            for inode in nodes_by_commoditytype[commodity_type]:
                try:
                    supply_paths = G.get_shortest_paths(
                        G.vs.find(inode), to=onode_vertex, mode="OUT"
                    )
                    if supply_paths and supply_paths[0]:
                        paths.extend(supply_paths[0])
                except ValueError:
                    continue

        # Convert vertex indices to node names and get unique nodes
        nodes_for_line = list(set(G.vs[x]["name"] for x in paths))
        all_paths[onode] = nodes_for_line

        # Check if any required nodes need repair
        damaged_nodes = [
            n for n in nodes_for_line if n in recovery_nodes and n not in uncosted_comps
        ]
        operational_lines[onode] = len(damaged_nodes) == 0

        if verbose:
            print(f"  {onode}: {'Operational' if operational_lines[onode] else 'Needs repair'}")

    # Sort output nodes - operational ones first
    out_node_list = sorted(out_node_list, key=lambda x: (not operational_lines[x], x))

    # ---------------------------------------------------------------------------------
    # Set the edge weight to reflect damage to its adjacent nodes
    if verbose:
        print("\n  Assigning repair priority weights to nodes...")
    PRIORITY_ATTR = "weight"
    for tp in G.get_edgelist():
        edge_id = G.get_eid(*tp)
        origin = G.vs[tp[0]]["name"]
        destin = G.vs[tp[1]]["name"]
        if weight_criteria == "MIN_TIME":
            if origin not in uncosted_comps:
                w1 = df_FRT.loc[origin, "Full Restoration Time"]
            else:
                w1 = 0
            if destin not in uncosted_comps:
                w2 = df_FRT.loc[destin, "Full Restoration Time"]
            else:
                w2 = 0
            wt = (w1 + w2) / 2
        elif weight_criteria == "MIN_COST":
            if origin not in uncosted_comps:
                w1 = component_meanloss.loc[origin, scenario_header]
            else:
                w1 = 0
            if destin not in uncosted_comps:
                w2 = component_meanloss.loc[destin, scenario_header]
            else:
                w2 = 0
            wt = (w1 + w2) / 2
        else:
            wt = 1.0
        G.es[edge_id][PRIORITY_ATTR] = wt

    if verbose:
        print("    Done.\n")

    # ---------------------------------------------------------------------------------
    # Set up repair list

    # repair_dict = {
    #     outnode: {sn: [] for sn in list(nodes_by_commoditytype.keys())}
    #     for outnode in out_node_list
    # }
    # REPAIR_META = ["initial_damage_list", "residual_repair_list", "undamaged", "fixed_with_prev"]

    repair_dict_init = OrderedDict()
    repair_dict_combined = OrderedDict()
    already_scheduled = set()

    if verbose:
        print("\nBuilding repair list for non-operational lines...\n")
    for out_idx, onode in enumerate(out_node_list):
        repair_dict_combined[onode] = {}
        if operational_lines[onode]:
            if verbose:
                print(f"  {onode}: No repairs needed")
            repair_dict_init[onode] = []
            repair_dict_combined[onode]["initial_damage_list"] = []
            repair_dict_combined[onode]["residual_repair_list"] = []
            repair_dict_combined[onode]["undamaged"] = True
            repair_dict_combined[onode]["fixed_with_prev"] = None
            continue

        onode_vertex = G.vs.find(onode)
        if verbose:
            print(f"\n  Processing repair list for output node: {onode}")

        # -----------------------------------------------------------------------------
        damaged_components = set()
        for CIDX, sup_nodes_by_commtype in list(nodes_by_commoditytype.items()):
            input_capacity_fractions = []
            for inode in sup_nodes_by_commtype:
                input_capacity_fractions.append(input_dict[inode]["capacity_fraction"])

            thresh = output_dict[onode]["capacity_fraction"]
            vx = []
            vlist = []

            # Skip empty supply nodes list
            if not input_capacity_fractions:
                continue

            # Generate combinations
            for L in range(0, len(input_capacity_fractions) + 1):
                for subset in itertools.combinations(
                    list(range(0, len(input_capacity_fractions))), L
                ):
                    vx.append(subset)
                for subset in itertools.combinations(input_capacity_fractions, L):
                    vlist.append(subset)

            # Skip if no valid combinations
            if len(vx) <= 1:
                continue

            vx = vx[1:]  # Remove empty combination
            vlist = [sum(x) for x in vlist[1:]]
            vcrit = np.array(vlist) >= thresh

            # Get dependency paths first
            sp_len = np.zeros(len(vx))
            sp_dep = []
            # LEN_CHK = np.inf
            for dnode in dependency_node_list:
                try:
                    paths = G.get_shortest_paths(
                        G.vs.find(dnode), to=onode_vertex, weights=PRIORITY_ATTR, mode="OUT"
                    )
                    if paths and paths[0]:  # Check if path exists and is not empty
                        sp_dep.extend(paths[0])
                except ValueError as verr:
                    rootLogger.warning(
                        f"Could not find path from dependency node "
                        f"{dnode} to output {onode}: {verr}"
                    )
                    continue
            sp_dep = list(set(sp_dep)) if sp_dep else []  # Ensure unique nodes

            # Process supply paths
            for cix, criteria in enumerate(vcrit):
                if not criteria:
                    sp_len[cix] = np.inf
                    continue

                sp_list = []
                for inx in vx[cix]:
                    try:
                        icnode = sup_nodes_by_commtype[inx]
                        paths = G.get_shortest_paths(
                            G.vs.find(icnode), to=onode_vertex, weights=PRIORITY_ATTR, mode="OUT"
                        )
                        if paths and paths[0]:
                            sp_list.extend(paths[0])
                    except (ValueError, IndexError) as e:
                        rootLogger.warning(f"Could not find path from supply node to {onode}: {e}")
                        continue

                if sp_list:  # Only process if paths(s) were found
                    sp_list = list(set(sp_list))  # Get unique nodes
                    RL = [G.vs[x]["name"] for x in set(sp_dep + sp_list)]
                    damaged_for_line = [x for x in RL if x not in uncosted_comps]
                    damaged_components.update(damaged_for_line)

                    # Filter out uncosted components and already scheduled ones
                    # RL = [
                    #     x for x in RL
                    #     if x not in uncosted_comps and x not in already_scheduled]
                    # sp_len[cix] = len(RL)
                    # if sp_len[cix] < LEN_CHK:
                    #     LEN_CHK = sp_len[cix]
                    #     repair_dict[onode][CIDX] = sorted(damaged_for_line)

        repair_dict_init[onode] = sorted(damaged_components)

        # -----------------------------------------------------------------------------
        # Combine repair lists for the current output node

        new_repairs = damaged_components - already_scheduled
        already_scheduled.update(new_repairs)

        repair_dict_combined[onode]["initial_damage_list"] = sorted(damaged_components)
        repair_dict_combined[onode]["residual_repair_list"] = sorted(new_repairs)
        repair_dict_combined[onode]["undamaged"] = False

        txtwrapper = textwrap.TextWrapper(
            width=80, initial_indent=" " * 6, subsequent_indent=" " * 6, break_long_words=False
        )
        repair_items_init = txtwrapper.fill(str(repair_dict_init[onode]))
        repair_items_combined = txtwrapper.fill(
            str(repair_dict_combined[onode]["residual_repair_list"])
        )

        if verbose:
            print(f"    Initial set of damaged components:\n{repair_items_init}")
        if repair_dict_combined[onode]["residual_repair_list"]:
            if verbose:
                print(f"    Components to repair:\n{repair_items_combined}")
            repair_dict_combined[onode]["fixed_with_prev"] = None
        else:
            if verbose:
                print("    No additional components need repair.")
            prev_idx = out_idx - 1
            if prev_idx < 0:
                repair_dict_combined[onode]["fixed_with_prev"] = None
            else:
                repair_dict_combined[onode]["fixed_with_prev"] = out_node_list[out_idx - 1]

    if verbose:
        print("\nCompleted preparing component repair list.\n")

    return repair_dict_init, repair_dict_combined


# =====================================================================================


def component_criticality(
    infrastructure,
    ctype_scenario_outcomes,
    hazard_type,
    scenario_tag,
    fig_path,
    fig_name="component_criticality_analysis.png",
):
    """
    This function:
    - checks criticality of 'component types' based on cost & time of reparation
    - assigns a criticality ranking to each component type
    - plots the component types in a grid to indicate their criticality

    """

    axes_lw = 0.75

    customised_canvas_dict = {
        # ---------------------------
        # Edges & Spine
        "axes.edgecolor": "0.15",
        "axes.linewidth": axes_lw,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
        # ---------------------------
        # Line style
        "grid.color": "white",
        "grid.linestyle": "-",
        "grid.linewidth": 2.0,
        # ---------------------------
        # Y-axis customisation
        "ytick.left": True,
        "ytick.labelsize": 13,
        "ytick.major.size": 7,
        "ytick.major.width": axes_lw,
        "ytick.major.pad": 4,
        "ytick.minor.left": False,
        # ---------------------------
        # X-axis customisation
        "xtick.bottom": True,
        "xtick.labelsize": 13,
        "xtick.major.size": 7,
        "xtick.major.width": axes_lw,
        "xtick.major.pad": 4,
        "xtick.minor.bottom": False,
        # ---------------------------
    }

    sns.set_theme(style="darkgrid", rc=customised_canvas_dict)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)

    restore_times = ctype_scenario_outcomes["restoration_time"]
    if np.max(restore_times) <= 0:
        print(f"No damage data for scenario {scenario_tag}. Skipping criticality analysis.\n")
        return
    rt_norm = restore_times / np.max(restore_times)

    pctloss_sys = ctype_scenario_outcomes["loss_tot"]
    pctloss_ntype = ctype_scenario_outcomes["loss_per_type"] * 15

    nt_names = ctype_scenario_outcomes.index.tolist()
    nt_ids = list(range(1, len(nt_names) + 1))
    autumn = cm.get_cmap("autumn")

    clrmap = [
        autumn(1.2 * x / float(len(ctype_scenario_outcomes.index)))
        for x in range(len(ctype_scenario_outcomes.index))
    ]

    ax.scatter(
        rt_norm,
        pctloss_sys,
        s=pctloss_ntype,
        c=clrmap,
        label=nt_ids,
        marker="o",
        edgecolor="bisque",
        lw=1.5,
        clip_on=False,
    )

    for cid, name, i, j in zip(nt_ids, nt_names, rt_norm, pctloss_sys):
        plt.annotate(
            str(cid),
            xy=(i, j),
            xycoords="data",
            xytext=(-20, 20),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=0,
            size=13,
            fontweight="bold",
            color="dodgerblue",
            annotation_clip=False,
            bbox=dict(boxstyle="round, pad=0.2", fc="yellow", alpha=0.0),
            arrowprops=dict(
                arrowstyle="-|>",
                shrinkA=5.0,
                shrinkB=5.0,
                connectionstyle="arc3,rad=0.0",
                color="dodgerblue",
                alpha=0.8,
                linewidth=0.5,
            ),
            path_effects=[PathEffects.withStroke(linewidth=2, foreground="white")],
        )

        plt.annotate(
            "{0:>2.0f}   {1:<s}".format(cid, name),
            xy=(1.05, 0.90 - 0.035 * cid),
            xycoords="axes fraction",
            ha="left",
            va="top",
            rotation=0,
            size=9,
        )

    txt_infra = "Infrastructure: " + infrastructure.system_class + "\n"
    txt_haz = "Hazard: " + hazard_type + "\n"
    txt_scenario = "Scenario: " + scenario_tag
    ax.text(
        1.05,
        0.995,
        txt_infra + txt_haz + txt_scenario,
        ha="left",
        va="top",
        rotation=0,
        linespacing=1.5,
        fontsize=11,
        clip_on=False,
        transform=ax.transAxes,
    )

    ylim = (0, int(max(pctloss_sys) + 1))
    ax.set_ylim(ylim)
    ax.set_yticks([0, max(ylim) * 0.5, max(ylim)])
    ax.set_yticklabels(["%0.2f" % y for y in [0, max(ylim) * 0.5, max(ylim)]], size=12)

    # xlim = (0, np.ceil(max(restore_times) / 10.0) * 10.0)
    xlim = (0, max(rt_norm))
    ax.set_xlim(xlim)
    ax.set_xticks([0, max(xlim) * 0.5, max(xlim)])
    ax.set_xticklabels([f"{x:.1f}" for x in [0, max(xlim) * 0.5, max(xlim)]], size=12)

    ax.set_title("COMPONENT CRITICALITY GRID", size=14, y=1.04, weight="bold")
    ax.set_xlabel("Time to Restoration (normalised)", size=13, labelpad=10)
    ax.set_ylabel("System Loss (%)", size=13, labelpad=10)

    sns.despine(left=False, bottom=False, right=True, top=True, offset=15, trim=True)

    figfile = Path(fig_path, fig_name)
    fig.savefig(figfile, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# =====================================================================================


def draw_component_loss_barchart_s1(
    ctype_resp_sorted, scenario_tag, hazard_type, output_path, fig_name
):
    """Plots bar charts of direct economic losses for components types"""

    ctype_loss_tot_mean = ctype_resp_sorted["loss_tot"].values * 100
    ctype_loss_by_type = ctype_resp_sorted["loss_per_type"].values * 100

    bar_width = 0.36
    bar_offset = 0.02
    bar_clr_1 = spl.ColourPalettes().BrewerSet1[0]  # '#E41A1C'
    bar_clr_2 = spl.ColourPalettes().BrewerSet1[1]  # '#377EB8'
    grid_clr = "#999999"

    cpt = [
        spl.split_long_label(x, delims=[" ", "_"], max_chars_per_line=23)
        for x in ctype_resp_sorted.index.tolist()
    ]
    pos = np.arange(0, len(cpt))

    fig = plt.figure(figsize=(4.5, len(pos) * 0.6), facecolor="white")
    axes = fig.add_subplot(111, facecolor="white")

    # ------------------------------------------------------------------------
    # Economic loss:
    #   - Contribution to % loss of total system, by components type
    #   - Percentage econ loss for all components of a specific type
    # ------------------------------------------------------------------------

    axes.barh(
        pos - bar_width / 2.0 - bar_offset,
        ctype_loss_tot_mean,
        bar_width,
        align="center",
        color=bar_clr_1,
        alpha=0.7,
        edgecolor=None,
        label="% loss of total system value (for a given component type)",
    )

    axes.barh(
        pos + bar_width / 2.0 + bar_offset,
        ctype_loss_by_type,
        bar_width,
        align="center",
        color=bar_clr_2,
        alpha=0.7,
        edgecolor=None,
        label="% loss for all components of specific type",
    )

    for p, cv in zip(pos, ctype_loss_tot_mean):
        axes.annotate(
            ("%0.1f" % float(cv)) + "%",
            xy=(cv + 0.7, p - bar_width / 2.0 - bar_offset),
            xycoords="data",
            ha="left",
            va="center",
            size=8,
            color=bar_clr_1,
            annotation_clip=False,
        )

    for p, cv in zip(pos, ctype_loss_by_type):
        axes.annotate(
            ("%0.1f" % float(cv)) + "%",
            xy=(cv + 0.7, p + bar_width / 2.0 + bar_offset * 2),
            xycoords="data",
            ha="left",
            va="center",
            size=8,
            color=bar_clr_2,
            annotation_clip=False,
        )

    # ------------------------------------------------------------------------
    # Title at the top
    axes.annotate(
        "ECONOMIC LOSS % by COMPONENT TYPE",
        xy=(0.0, -1.65),
        xycoords="data",
        ha="left",
        va="top",
        size=10,
        color="k",
        weight="bold",
        annotation_clip=False,
    )

    # Hazard label above the axes
    axes.annotate(
        "Hazard: " + hazard_type + " " + scenario_tag,
        xy=(0.0, -1.15),
        xycoords="data",
        ha="left",
        va="top",
        size=10,
        color="slategrey",
        weight="bold",
        annotation_clip=False,
    )

    # ------------------------------------------------------------------------
    # Legend
    legend_offset_points = 20  # consistent 20 points below axes

    # Create a blended transform
    # x: use axes coordinates (0 = left edge of axes)
    # y: use axes coordinates with fixed offset in points
    trans = transforms.blended_transform_factory(
        axes.transAxes,
        transforms.offset_copy(
            axes.transAxes,
            y=-legend_offset_points,
            units="points",
            fig=axes.figure,  # type: ignore
        ),
    )

    lgnd = axes.legend(
        loc="upper left",
        ncol=1,
        bbox_to_anchor=(-0.1, 0),
        bbox_transform=trans,
        borderpad=0,
        frameon=0,
        prop={"size": 8, "weight": "medium"},
    )

    for text in lgnd.get_texts():
        text.set_color("#555555")

    # Draw a separator line with legend below and hline above it
    axes.axhline(
        y=pos.max() + bar_width * 2.4,
        xmin=0,
        xmax=0.3,
        lw=0.5,
        ls="-",
        color=grid_clr,
        clip_on=False,
    )

    # ------------------------------------------------------------------------
    # Axes formatting

    spines_to_remove = ["top", "bottom", "right"]
    for spine in spines_to_remove:
        axes.spines[spine].set_visible(False)
    axes.spines["left"].set_color(grid_clr)
    axes.spines["left"].set_linewidth(0.5)

    axes.set_xlim(0, 100)
    axes.set_xticks(np.linspace(0, 100, 5, endpoint=True))
    axes.set_xticklabels([" "] * 5)
    axes.xaxis.grid(False)

    axes.set_ylim((pos.max() + bar_width * 1.5, pos.min() - bar_width * 1.5))
    axes.set_yticks(pos)
    axes.set_yticklabels(cpt, size=8, color="k")
    axes.yaxis.grid(False)

    axes.tick_params(top=False, bottom=False, left=False, right=False)

    fig.savefig(os.path.join(output_path, fig_name), format="png", bbox_inches="tight", dpi=400)

    plt.close(fig)


# =====================================================================================


def draw_component_loss_barchart_s2(
    ctype_resp_sorted, scenario_tag, hazard_type, output_path, fig_name
):
    """Plots bar charts of direct economic losses for components types"""

    ctype_loss_tot_mean = ctype_resp_sorted["loss_tot"].values * 100
    ctype_loss_mean_by_type = ctype_resp_sorted["loss_per_type"].values * 100
    ctype_loss_std_by_type = ctype_resp_sorted["loss_per_type_std"].values * 100

    bar_width = 0.4
    bar_space_index = 2.0
    bar_clr_1 = "#3288BD"
    grid_clr = "dimgrey"
    header_size = 9

    comptypes_sorted = [
        spl.split_long_label(x, delims=[" ", "_"], max_chars_per_line=23)
        for x in ctype_resp_sorted.index.tolist()
    ]

    barpos = np.linspace(
        0, len(comptypes_sorted) * bar_width * bar_space_index, len(comptypes_sorted)
    )

    fig = plt.figure(figsize=(5.0, len(barpos) * 0.6), facecolor="white")
    num_grids = 10 + len(comptypes_sorted)
    gs = gridspec.GridSpec(num_grids, 1)
    ax1 = plt.subplot(gs[:-5])
    ax2 = plt.subplot(gs[-3:])
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")

    # ==========================================================================
    # Percentage of lost value for each `component type`
    # ==========================================================================

    ax1.barh(
        barpos,
        100,
        bar_width,
        align="center",
        color="gainsboro",
        alpha=0.7,
        edgecolor=None,
        label="",
    )
    ax1.barh(
        barpos,
        ctype_loss_mean_by_type,
        bar_width,
        align="center",
        color=bar_clr_1,
        alpha=0.75,
        edgecolor=None,
        label="% loss for component type",
        xerr=ctype_loss_std_by_type,
        error_kw={"ecolor": "lightcoral", "capsize": 2, "elinewidth": 0.7, "markeredgewidth": 0.7},
    )

    for p, cv in zip(barpos, ctype_loss_mean_by_type):
        ax1.annotate(
            ("%0.1f" % float(cv)) + "%",
            xy=(cv + 0.7, p - bar_width * 0.8),
            xycoords="data",
            ha="left",
            va="center",
            size=8,
            color=bar_clr_1,
            annotation_clip=False,
        )

    ax1.annotate(
        "HAZARD: " + hazard_type + " " + scenario_tag,
        xy=(0.0, -1.65),
        xycoords="data",
        ha="left",
        va="top",
        size=header_size,
        color="slategrey",
        weight="bold",
        annotation_clip=False,
    )

    ax1.annotate(
        "LOSS METRIC: % loss for each Component Type",
        xy=(0.0, -1.1),
        xycoords="data",
        ha="left",
        va="top",
        size=header_size,
        color="k",
        weight="bold",
        annotation_clip=False,
    )

    ax1.axhline(
        y=barpos.max() + bar_width * 2.2,
        xmin=0,
        xmax=0.3,
        lw=0.5,
        ls="-",
        color=grid_clr,
        clip_on=False,
    )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    spines_to_remove = ["top", "bottom", "right", "left"]
    for spine in spines_to_remove:
        ax1.spines[spine].set_visible(False)

    ax1.xaxis.grid(False)
    ax1.set_xlim(0, 100)
    ax1.set_xticks(np.linspace(0, 100, 5, endpoint=True))
    ax1.set_xticklabels([" "] * 5)

    ax1.yaxis.grid(False)
    ax1.set_ylim((barpos.max() + bar_width, barpos.min() - bar_width))
    ax1.set_yticks(barpos)
    ax1.set_yticklabels(comptypes_sorted, size=7, color="k")

    ax1.tick_params(top=False, bottom=False, left=False, right=False)
    ax1.yaxis.set_tick_params(which="major", labelleft=True, labelright=False, labelsize=8, pad=18)

    # ==========================================================================
    # Stacked Bar Chart: demonstrates relative contribution of `component types`
    # to the aggregated direct economic loss of the system.
    # ==========================================================================

    desired_height_points = 15
    fig2 = ax2.figure
    height_pixels = fig2.dpi * desired_height_points / 72.0  # 1 point = 1/72 inch

    # Get the height in data units
    inv = ax2.transData.inverted()
    y0 = ax2.transAxes.transform((0, 0))[1]
    y1 = y0 + height_pixels
    stacked_bar_width = inv.transform((0, y1))[1] - inv.transform((0, y0))[1]

    colours = spl.ColourPalettes()
    if len(comptypes_sorted) <= 11:
        COLR_SET = colours.BrewerSpectral
    elif len(comptypes_sorted) <= 20:
        COLR_SET = colours.Trubetskoy
    else:
        COLR_SET = colours.Tartarize269

    scaled_tot_loss = [
        x * 100 / sum(ctype_loss_tot_mean) if sum(ctype_loss_tot_mean) > 0 else 0
        for x in ctype_loss_tot_mean
    ]

    leftpos = 0.0
    for ind, loss, pos in zip(list(range(len(comptypes_sorted))), scaled_tot_loss, barpos):
        ax2.barh(0.0, loss, stacked_bar_width, left=leftpos, color=COLR_SET[ind])
        leftpos += loss
        ax1.annotate(
            "$\u25fc$",
            xy=(-2.0, pos),
            xycoords="data",
            color=COLR_SET[ind],
            size=10,
            ha="right",
            va="center",
            annotation_clip=False,
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ax2.xaxis.grid(False)
    ax2.set_xticklabels([])
    ax2.set_xlim(0, 100)
    agg_sys_loss = sum(ctype_resp_sorted["loss_tot"].values)
    # ax2.set_xlabel(
    #     "Aggregated loss as fraction of System Value: {:.1f}% ".format(round(agg_sys_loss * 100)),
    #     size=8,
    #     labelpad=2,
    # )

    ax2.yaxis.grid(False)
    ax2.set_yticklabels([])
    ax2.set_ylim((-stacked_bar_width, stacked_bar_width))

    ax2.tick_params(top=False, bottom=False, left=False, right=False)

    ax2.annotate(
        "LOSS METRIC: component loss as % of total system loss",
        xy=(0.0, stacked_bar_width * 1.3),
        xycoords="data",
        ha="left",
        va="top",
        size=header_size,
        color="k",
        weight="bold",
        annotation_clip=False,
    )

    for spine in ["top", "right", "left", "bottom"]:
        ax2.spines[spine].set_visible(False)

    arrowprops = dict(
        arrowstyle="|-|, widthA=0.4, widthB=0.4",
        linewidth=0.7,
        facecolor=grid_clr,
        edgecolor=grid_clr,
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
    )
    ax2.annotate(
        "",
        # xy=(0.0, -0.025),
        # xytext=(1.0, -0.025),
        # xycoords="axes fraction",
        xy=(0.0, -stacked_bar_width),
        xytext=(100, -stacked_bar_width),
        xycoords="data",
        arrowprops=arrowprops,
    )

    ax2.annotate(
        "Combined loss as fraction of System value: {:.1f}% ".format(round(agg_sys_loss * 100)),
        xy=(50, stacked_bar_width * -1.35),
        xycoords="data",
        ha="center",
        va="top",
        size=9,
        color="k",
        weight="normal",
        annotation_clip=False,
    )

    fig.savefig(os.path.join(output_path, fig_name), format="png", bbox_inches="tight", dpi=400)

    plt.close(fig)


# =====================================================================================


def draw_component_loss_barchart_s3(
    ctype_resp_sorted, scenario_tag, hazard_type, output_path, fig_name
):
    """Plots bar charts of direct economic losses for components types"""

    ctype_loss_tot_mean = ctype_resp_sorted["loss_tot"].values * 100
    ctype_loss_mean_by_type = ctype_resp_sorted["loss_per_type"].values * 100
    ctype_loss_std_by_type = ctype_resp_sorted["loss_per_type_std"].values * 100

    bar_width = 0.35
    bar_clr_1 = spl.ColourPalettes().BrewerSet1[0]  # '#E41A1C'
    grid_clr = "dimgrey"
    bg_box_clr = "gainsboro"

    comptypes_sorted = ctype_resp_sorted.index.tolist()
    barpos = np.arange(0, len(comptypes_sorted))

    fig = plt.figure(figsize=(5.0, len(barpos) * 0.6), facecolor="white")
    num_grids = 8 + len(comptypes_sorted)
    gs = gridspec.GridSpec(num_grids, 1)
    ax1 = plt.subplot(gs[:-4], facecolor="white")
    ax2 = plt.subplot(gs[-3:], facecolor="white")

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
        barpos,
        100,
        bar_width,
        align="center",
        color=bg_box_clr,
        alpha=0.85,
        edgecolor=None,
        label="",
    )
    ax1.barh(
        barpos,
        ctype_loss_mean_by_type,
        bar_width,
        align="center",
        color=bar_clr_1,
        alpha=0.75,
        edgecolor=None,
        label="% loss for component type",
        xerr=ctype_loss_std_by_type,
        error_kw={
            "ecolor": "cornflowerblue",
            "capsize": 0,
            "elinewidth": 1.2,
            "markeredgewidth": 0.0,
        },
    )

    def selective_rounding(fltval):
        if fltval > 1.0:
            fmt = "{:>4.0f}%".format(round(fltval))
        else:
            fmt = "{:>4.1f}%".format(fltval)
        return fmt

    for p, ct, loss_ct in zip(barpos, comptypes_sorted, ctype_loss_mean_by_type):
        ax1.annotate(
            ct,
            xy=(3, p - bar_width),
            xycoords="data",
            ha="left",
            va="center",
            size=7,
            color="k",
            annotation_clip=False,
        )

        ax1.annotate(
            selective_rounding(loss_ct),
            xy=(100, p - bar_width),
            xycoords="data",
            ha="right",
            va="center",
            size=7,
            color=bar_clr_1,
            annotation_clip=False,
        )

    ax1.annotate(
        "HAZARD: " + hazard_type + " " + scenario_tag,
        xy=(0.0, -1.8),
        xycoords="data",
        ha="left",
        va="top",
        size=9,
        color="slategrey",
        weight="bold",
        annotation_clip=False,
    )

    ax1.annotate(
        "LOSS METRIC: % loss for each Component Type",
        xy=(0.0, -1.25),
        xycoords="data",
        ha="left",
        va="top",
        size=9,
        color="k",
        weight="bold",
        annotation_clip=False,
    )

    ax1.axhline(
        y=barpos.max() + bar_width * 2.0,
        xmin=0,
        xmax=0.20,
        lw=0.6,
        ls="-",
        color="grey",
        clip_on=False,
    )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    spines_to_remove = ["top", "bottom", "right", "left"]
    for spine in spines_to_remove:
        ax1.spines[spine].set_visible(False)

    ax1.xaxis.grid(False)
    ax1.set_xlim(0, 100)
    ax1.set_xticks(np.linspace(0, 100, 5, endpoint=True))
    ax1.set_xticklabels([" "] * 5)

    ax1.yaxis.grid(False)
    ax1.set_ylim((barpos.max() + bar_width, barpos.min() - bar_width))
    ax1.set_yticks(barpos)
    ax1.set_yticklabels([])

    ax1.tick_params(top=False, bottom=False, left=False, right=False)
    ax1.yaxis.set_tick_params(which="major", labelleft=True, labelright=False, labelsize=8, pad=20)

    # ===========================================================================
    # Stacked Bar Chart: demonstrates relative contribution of `component types`
    # to the aggregated direct economic loss of the system.
    # ===========================================================================

    stacked_bar_width = 1.0

    scaled_tot_loss = [
        x * 100 / sum(ctype_loss_tot_mean) if sum(ctype_loss_tot_mean) > 0 else 0
        for x in ctype_loss_tot_mean
    ]

    leftpos = 0.0
    rank = 1
    for ind, loss, sysloss, pos in zip(
        list(range(len(comptypes_sorted))), scaled_tot_loss, ctype_loss_tot_mean, barpos
    ):
        ax2.barh(0.0, loss, stacked_bar_width, left=leftpos, color=COLR_SET[ind])

        # Annotate the values for the top five contributors
        if rank <= 5:
            xt = (2 * leftpos + loss) / 2.0
            yt = stacked_bar_width * 0.6
            ax2.text(
                xt,
                yt,
                selective_rounding(sysloss),
                ha="center",
                va="bottom",
                size=7,
                color="k",
                weight="bold",
                path_effects=[PathEffects.withStroke(linewidth=1, foreground="w")],
            )
            rank += 1

        leftpos += loss
        ax1.annotate(
            "$\u25fc$",
            xy=(-0.4, pos - bar_width),
            xycoords="data",
            color=COLR_SET[ind],
            size=8,
            ha="left",
            va="center",
            annotation_clip=False,
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    for spine in ["top", "right", "left", "bottom"]:
        ax2.spines[spine].set_visible(False)

    ax2.xaxis.grid(False)
    ax2.set_xticklabels([])
    ax2.set_xlim(0, 100)
    agg_sys_loss = sum(ctype_resp_sorted["loss_tot"].values)
    ax2.set_xlabel(
        "Aggregated loss as fraction of System Value: {:.0f}% ".format(round(agg_sys_loss * 100)),
        size=8,
        labelpad=8,
    )

    ax2.yaxis.grid(False)
    ax2.set_yticklabels([])
    ax2.set_ylim((-stacked_bar_width * 0.7, stacked_bar_width * 1.8))

    ax2.tick_params(top=False, bottom=False, left=False, right=False)

    ax2.annotate(
        "LOSS METRIC: % system loss attributed to Component Types",
        xy=(0.0, stacked_bar_width * 1.5),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        size=9,
        color="k",
        weight="bold",
        annotation_clip=False,
    )

    arrow_ypos = -stacked_bar_width * 1.2
    ax2.annotate(
        "",
        xy=(0.0, arrow_ypos * 0.9),
        xytext=(1.0, arrow_ypos * 0.9),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="|-|, widthA=0.4, widthB=0.4",
            linewidth=0.7,
            facecolor=grid_clr,
            edgecolor=grid_clr,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
        ),
    )

    fig.savefig(os.path.join(output_path, fig_name), format="png", bbox_inches="tight", dpi=400)

    plt.close(fig)


# =====================================================================================


def draw_component_failure_barchart(
    uncosted_comptypes,
    ctype_failure_mean,
    scenario_name,
    scenario_tag,
    hazard_type,
    output_path,
    figname,
):
    comp_type_fail_sorted = ctype_failure_mean.loc[scenario_name].sort_values(ascending=False)
    cpt_failure_vals = comp_type_fail_sorted.values * 100

    for x in uncosted_comptypes:
        if x in comp_type_fail_sorted.index.tolist():
            comp_type_fail_sorted = comp_type_fail_sorted.drop(x, axis=0)

    cptypes = comp_type_fail_sorted.index.tolist()
    cpt = [spl.split_long_label(x, delims=[" ", "_"], max_chars_per_line=22) for x in cptypes]
    pos = np.arange(len(cptypes))

    fig = plt.figure(figsize=(4.5, len(pos) * 0.5), facecolor="white")
    ax = fig.add_subplot(111)
    bar_width = 0.4
    bar_clr = "#D62F20"
    grid_clr = "#BBBBBB"

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    spines_to_remove = ["top", "bottom", "right"]
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(grid_clr)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["right"].set_color(grid_clr)
    ax.spines["right"].set_linewidth(0.5)

    ax.set_xlim(0, 100)
    ax.set_xticks(np.linspace(0, 100, 5, endpoint=True))
    ax.set_xticklabels([" "] * 5)
    ax.xaxis.grid(True, color=grid_clr, linewidth=0.5, linestyle="-")

    ax.set_ylim((pos.max() + bar_width, pos.min() - bar_width))
    ax.set_yticks(pos)
    ax.set_yticklabels(cpt, size=8, color="k")
    ax.yaxis.grid(False)

    ax.tick_params(top=False, bottom=False, left=False, right=False)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ax.barh(pos, cpt_failure_vals, bar_width, color=bar_clr, alpha=0.8, edgecolor=None)

    # add the numbers to the side of each bar
    for p, cv in zip(pos, cpt_failure_vals):
        plt.annotate(
            "{:0.1f}".format(cv) + "%",
            xy=(cv + 0.6, float(p)),
            va="center",
            size=8,
            color="#CA1C1D",
            annotation_clip=False,
        )

    ax.annotate(
        "FAILURE RATE: % FAILED COMPONENTS by TYPE",
        xy=(0.0, -1.45),
        xycoords="data",
        ha="left",
        va="top",
        annotation_clip=False,
        size=10,
        weight="bold",
        color="k",
    )
    ax.annotate(
        "Hazard: " + hazard_type + " " + scenario_tag,
        xy=(0.0, -1.0),
        xycoords="data",
        ha="left",
        va="top",
        annotation_clip=False,
        size=10,
        weight="bold",
        color="slategrey",
    )

    fig.savefig(os.path.join(output_path, figname), format="png", bbox_inches="tight", dpi=400)
    plt.close(fig)


# =====================================================================================


def run_scenario_loss_analysis(
    scenario,
    hazards,
    infrastructure,
    config,
    input_comptype_response_file,
    input_component_response_file,
    scenario_dir_name="scenario_analysis",
    verbosity=True,
):
    rootLogger.info(Fore.MAGENTA + "Initiating : SCENARIO LOSS ANALYSIS\n" + Fore.RESET)

    RESTORATION_STREAMS = scenario.restoration_streams
    FOCAL_HAZARD_SCENARIOS = hazards.focal_hazard_scenarios
    FOCAL_HAZARD_SCENARIO_IDS = config.FOCAL_HAZARD_SCENARIO_NAMES

    SCENARIO_OUTPUT_PATH = Path(config.OUTPUT_DIR, scenario_dir_name)
    SCENARIO_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # FOR FUTURE IMPLEMENTATION:
    # Restoration time starts x time units after hazard impact:
    # This represents lead up time for damage and safety assessments
    # RESTORATION_OFFSET = 1

    # ------------------------------------------------------------------------------
    # Read in SIMULATED HAZARD RESPONSE for <COMPONENT TYPES>
    # ------------------------------------------------------------------------------

    comptype_resp_df = pd.read_csv(
        input_comptype_response_file, index_col=0, header=[0, 1]
    ).sort_index(axis=1)

    if scenario.hazard_input_method.lower() in ["calculated_array"]:
        hazard_events = [f"{i:.3f}" for i in comptype_resp_df.index.values.tolist()]
    else:
        hazard_events = [str(i) for i in comptype_resp_df.index.values.tolist()]
    comptype_resp_df.index = pd.Index(hazard_events)

    component_response = pd.read_csv(input_component_response_file, index_col=0, header=[0, 1])
    component_response.index.astype(str)
    component_response.index = pd.Index(hazard_events)

    # hazard_events = [str(i) for i in component_response.index.values.tolist()]
    # component_meanloss = component_response.xs("loss_mean", axis=1, level=1, drop_level=True)

    # ------------------------------------------------------------------------------
    # Nodes not considered in the loss calculations
    # NEED TO MOVE THESE TO A MORE LOGICAL PLACE
    # ------------------------------------------------------------------------------

    uncosted_comptypes = [
        "CONN_NODE",
        "JUNCTION_NODE",
        "JUNCTION",
        "SYSTEM_INPUT",
        "SYSTEM_OUTPUT",
        "Generation Source",
        "Grounding",
    ]

    cp_types_in_system = infrastructure.get_component_types()
    cp_types_costed = [x for x in cp_types_in_system if x not in uncosted_comptypes]

    comptype_resp_df.drop(uncosted_comptypes, level=0, axis=1, inplace=True, errors="ignore")

    # Get list of only those components that are included in cost calculations:
    cpmap = {c: sorted(list(infrastructure.get_components_for_type(c))) for c in cp_types_in_system}
    comps_costed = [v for x in cp_types_costed for v in cpmap[x]]

    nodes_all = list(infrastructure.components.keys())
    nodes_all.sort()
    comps_uncosted = list(set(nodes_all).difference(comps_costed))

    ctype_failure_mean = comptype_resp_df.xs("failure_rate", level=1, axis=1)

    # ------------------------------------------------------------------------------
    # Value of component types relative to system value
    # ------------------------------------------------------------------------------

    comptype_value_dict = {}
    for ct in sorted(cp_types_costed):
        comp_val = [infrastructure.components[comp_id].cost_fraction for comp_id in cpmap[ct]]
        comptype_value_dict[ct] = sum(comp_val)

    comptype_value_list = [comptype_value_dict[ct] for ct in sorted(comptype_value_dict.keys())]

    # weight_criteria = "MIN_COST"

    # ------------------------------------------------------------------------------

    col_tp = []
    for scn in FOCAL_HAZARD_SCENARIO_IDS:
        col_tp.extend(zip([scn] * len(RESTORATION_STREAMS), RESTORATION_STREAMS))

    mcols = pd.MultiIndex.from_tuples(col_tp, names=["Hazard", "Restoration Streams"])
    time_to_full_restoration_for_lines_df = pd.DataFrame(
        index=sorted(list(infrastructure.output_nodes.keys())), columns=mcols
    )
    time_to_full_restoration_for_lines_df.index.name = "Output Lines"

    # ------------------------------------------------------------------------------

    def get_nearest_index(thelist, flt):
        thelist = [float(x) for x in thelist]
        arr = np.asarray(thelist)
        flt = float(flt)
        idx = (np.abs(arr - flt)).argmin()
        return idx

    def get_scenarioname_from_val(sc_haz_str):
        nearest_val_idx = get_nearest_index(hazards.hazard_scenario_list, sc_haz_str)
        scn_name = hazards.hazard_scenario_list[nearest_val_idx]
        return scn_name

    # ------------------------------------------------------------------------------
    # >>> BEGIN : FOCAL_HAZARD_SCENARIOS for LOOP

    for sc_idx, sc_haz_str in enumerate(FOCAL_HAZARD_SCENARIOS):
        # --------------------------------------------------------------------------
        # Differentiated setup based on hazard input type - scenario vs array
        # --------------------------------------------------------------------------
        sc_haz_str = "{:.3f}".format(float(sc_haz_str))
        if str(config.HAZARD_INPUT_METHOD).lower() == "calculated_array":
            scenario_header = get_scenarioname_from_val(sc_haz_str)
        elif str(config.HAZARD_INPUT_METHOD).lower() in ["hazard_file", "scenario_file"]:
            scenario_header = sc_haz_str
        else:
            raise ValueError("Unrecognised value for HAZARD_INPUT_METHOD.")

        scenario_tag = (
            hazards.intensity_measure_param
            + " "
            + str(sc_haz_str)
            + " "
            + hazards.intensity_measure_unit
        )

        # --------------------------------------------------------------------------
        # Extract scenario-specific values from the 'hazard response' dataframe
        # Scenario response: by component type
        # --------------------------------------------------------------------------
        print("-" * 80)
        rootLogger.info(
            f"{Fore.MAGENTA}Running analysis for scenario_header: {scenario_header}{Fore.RESET}\n"
        )
        ctype_resp_scenario = comptype_resp_df.loc[scenario_header].unstack(level=-1)
        ctype_resp_scenario = ctype_resp_scenario.sort_index()

        ctype_resp_scenario["loss_per_type"] = (
            ctype_resp_scenario["loss_mean"] / comptype_value_list
        )

        ctype_resp_scenario["loss_per_type_std"] = ctype_resp_scenario["loss_std"] * [
            len(list(infrastructure.get_components_for_type(ct)))
            for ct in ctype_resp_scenario.index.values.tolist()
        ]

        ctype_resp_sorted = ctype_resp_scenario.sort_values(
            by=["loss_tot"], axis=0, ascending=[False]
        )  # type: ignore

        fig_name = f"sc_{sc_idx + 1}_{sc_haz_str}_LOSS_vs_comptype_s1.png"
        draw_component_loss_barchart_s1(
            ctype_resp_sorted, scenario_tag, hazards.hazard_type, SCENARIO_OUTPUT_PATH, fig_name
        )

        fig_name = f"sc_{sc_idx + 1}_{sc_haz_str}_LOSS_vs_comptype_s2.png"
        draw_component_loss_barchart_s2(
            ctype_resp_sorted, scenario_tag, hazards.hazard_type, SCENARIO_OUTPUT_PATH, fig_name
        )

        # fig_name = f'sc_{sc_idx + 1}_{sc_haz_str}_LOSS_vs_comptype_s3.png'
        # draw_component_loss_barchart_s3(
        #     ctype_resp_sorted,
        #     scenario_tag,
        #     hazards.hazard_type,
        #     SCENARIO_OUTPUT_PATH,
        #     fig_name)

        # ==========================================================================
        # FAILURE RATE -- PERCENTAGE of component types
        # --------------------------------------------------------------------------
        fig_name = f"sc_{sc_idx + 1}_{sc_haz_str}_FAILURES_component_type.png"
        draw_component_failure_barchart(
            uncosted_comptypes,
            ctype_failure_mean,
            scenario_header,
            scenario_tag,
            hazards.hazard_type,
            SCENARIO_OUTPUT_PATH,
            fig_name,
        )

        # ==========================================================================
        # RESTORATION PROGNOSIS -- for specified scenarios
        # --------------------------------------------------------------------------
        scn_name = FOCAL_HAZARD_SCENARIO_IDS[sc_idx]
        event_id = sc_haz_str

        recovery_results, component_recovery_times = perform_recovery_analysis(
            infrastructure,
            scenario,
            config,
            hazards,
            event_id,
            component_response,
            comps_costed,
            comps_uncosted,
            SCENARIO_OUTPUT_PATH,
            scenario_tag,
            scenario_num=sc_idx + 1,
            verbosity=verbosity,
        )

        output_node_list = sorted(list(infrastructure.output_nodes.keys()))
        for num_streams in RESTORATION_STREAMS:
            recovery_info = recovery_results[(scenario_header, num_streams)]
            time_to_full_restoration_for_lines_df[(scn_name, num_streams)] = [
                recovery_info["system_restoration_times"].get(x, 0) for x in output_node_list
            ]

        # ==========================================================================
        # COMPONENT CRITICALITY -- for given scenario & restoration setup
        # --------------------------------------------------------------------------
        ctype_scenario_outcomes = calc_outcomes_for_component_types(
            infrastructure, component_recovery_times, ctype_resp_sorted, cp_types_costed
        )

        fig_name = f"sc_{sc_idx + 1}_{sc_haz_str}_CRITICALITY_of_components.png"
        component_criticality(
            infrastructure,
            ctype_scenario_outcomes,
            hazards.hazard_type,
            scenario_tag,
            fig_path=SCENARIO_OUTPUT_PATH,
            fig_name=fig_name,
        )

    # <<< END : FOCAL_HAZARD_SCENARIOS FOR LOOP
    # ---------------------------------------------------------------------------------

    line_rst_csv = os.path.join(SCENARIO_OUTPUT_PATH, "line_restoration_prognosis.csv")
    time_to_full_restoration_for_lines_df = time_to_full_restoration_for_lines_df.apply(
        pd.to_numeric
    ).round(0)
    time_to_full_restoration_for_lines_df.to_csv(line_rst_csv, sep=",")
    print("-" * 80)
    rootLogger.info("End: SCENARIO LOSS ANALYSIS")

    # ---------------------------------------------------------------------------------
