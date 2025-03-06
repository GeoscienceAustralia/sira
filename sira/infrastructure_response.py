import logging
import os
from pathlib import Path
import math
from tqdm import tqdm
import dask.dataframe as dd                         # type: ignore
from dask.diagnostics.progress import ProgressBar   # type: ignore
from contextlib import nullcontext

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Union, Tuple, Optional

from numba import njit
from concurrent.futures import ProcessPoolExecutor
import pickle
import pyarrow as pa           # type: ignore
import pyarrow.parquet as pq   # type: ignore
from colorama import Fore, init
init()

from sira.tools import utils
import sira.loss_analysis as loss_analysis

rootLogger = logging.getLogger(__name__)
matplotlib.use('Agg')
plt.switch_backend('agg')

mpl_looger = logging.getLogger('matplotlib')
mpl_looger.setLevel(logging.WARNING)


CALC_SYSTEM_RECOVERY = True

# ****************************************************************************
# BEGIN POST-PROCESSING ...
# ****************************************************************************

def calc_tick_vals(val_list, xstep=0.1):
    num_ticks = int(round(len(val_list) / xstep)) + 1
    if (num_ticks > 12) and (num_ticks <= 20):
        xstep = 0.2
        num_ticks = int(round(len(val_list) / xstep)) + 1
    elif num_ticks > 20:
        num_ticks = 11
    tick_labels = val_list[::(num_ticks - 1)]
    if isinstance(tick_labels[0], float):
        tick_labels = ['{:.3f}'.format(val) for val in tick_labels]
    return tick_labels


def plot_mean_econ_loss(
        hazard_intensity_list: Union[List[float], np.ndarray],
        loss_array: Union[List[float], np.ndarray],
        x_label: str = "Hazard Intensity",
        y_label: str = "Direct Loss Fraction",
        fig_title: str = "Loss Ratio",
        fig_name: str = "fig_lossratio_boxplot",
        output_path: Union[str, Path] = ".") -> None:

    """Draws and saves a boxplot of mean economic loss"""

    # --------------------------------------------------------------------------

    econ_loss = np.array(loss_array)
    # econ_loss = econ_loss.transpose()

    x_values = list(hazard_intensity_list)
    y_values_list = econ_loss

    x_max = max(x_values)
    x_min = min(x_values)
    x_diff = x_max - x_min

    if x_diff <= 0.1:
        bin_width = 0.02
    elif x_diff <= 0.6:
        bin_width = 0.05
    elif x_diff <= 1.1:
        bin_width = 0.1
    elif x_diff <= 2.1:
        bin_width = 0.2
    elif x_diff <= 3.1:
        bin_width = 0.25
    elif x_diff <= 5.1:
        bin_width = 0.5
    elif x_diff <= 10:
        bin_width = 1
    elif x_diff <= 20:
        bin_width = 2
    elif x_diff <= 50:
        bin_width = 5
    elif x_diff <= 100:
        bin_width = 10
    else:
        bin_width = int(x_diff / 10)

    if x_diff <= 0.25:
        precision_digits = 3
    elif x_diff <= 0.5:
        precision_digits = 2
    elif x_diff <= 1.0:
        precision_digits = 2
    else:
        precision_digits = 1

    bin_edges = np.arange(0, x_max + bin_width, bin_width)
    if bin_edges[-1] > x_max:
        bin_edges[-1] = x_max
    binned_x = np.digitize(x_values, bin_edges, right=True)
    binned_x = binned_x * bin_width
    binned_x = binned_x[1:]

    all_x = []
    all_y = []

    for x, y_array in zip(binned_x, y_values_list):
        all_x.extend([x] * len(y_array))
        all_y.extend(y_array)

    pts = str(precision_digits)
    format_string = "{:." + pts + "f}-{:." + pts + "f}"

    bin_labels = [
        format_string.format(bin_edges[i], bin_edges[i + 1])
        for i in range(len(bin_edges) - 1)]

    # --------------------------------------------------------------------------

    fig, ax = plt.subplots(1, figsize=(12, 7), facecolor='white')
    sns.set_theme(style='ticks', palette='Set2')
    sns.boxplot(
        ax=ax,
        x=all_x,
        y=all_y,
        linewidth=0.8,
        width=0.35,
        color='whitesmoke',
        showmeans=True,
        showfliers=True,
        meanprops=dict(
            marker='o',
            markeredgecolor='coral',
            markerfacecolor='coral'),
        flierprops=dict(
            marker='x',
            markerfacecolor='#BBB',
            markersize=6,
            linestyle='none'),
    )

    # --------------------------------------------------------------------------

    sns.despine(
        bottom=False, top=True, left=True, right=True,
        offset=None, trim=True)

    ax.spines['bottom'].set(
        linewidth=1.0, color='#444444', position=('axes', -0.02))

    ax.yaxis.grid(
        True, which="major", linestyle='-',
        linewidth=0.5, color='#B6B6B6')

    ax.tick_params(
        axis='x', bottom=True, top=False,
        width=1.0, labelsize=10, color='#444444')

    ax.tick_params(
        axis='y', left=False, right=False,
        width=1.0, labelsize=10, color='#444444')

    ax.set_xticks(range(len(bin_labels)), bin_labels, rotation=45, ha='right')

    _, y_max = ax.get_ylim()
    y_max = np.round(y_max, 1)
    y_ticks = np.arange(0.0, y_max + 0.1, 0.2)
    ax.set_yticks(y_ticks)

    # --------------------------------------------------------------------------

    ax.set_xlabel(x_label, labelpad=9, size=11)
    ax.set_ylabel(y_label, labelpad=9, size=11)
    ax.set_title(
        fig_title, loc='center', y=1.04,
        fontsize=12, weight='bold')

    # --------------------------------------------------------------------------
    fig_name = fig_name + ".png"
    figfile = Path(output_path, fig_name)
    plt.margins(0.05)
    plt.savefig(figfile, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    # --------------------------------------------------------------------------


def analyze_single_event(
        event_id, infrastructure, scenario, hazards, component_resp_df,
        components_costed, components_uncosted):
    """
    Wrapper function to analyze a single hazard event
    """
    return loss_analysis.analyse_system_recovery(
        infrastructure,
        scenario,
        hazards,
        str(event_id),
        component_resp_df,
        components_costed,
        components_uncosted,
        verbosity=False
    )

def process_chunk(
        chunk_events,
        infrastructure,
        scenario,
        hazards,
        component_resp_df,
        components_costed,
        components_uncosted):
    """
    Process a chunk of hazard events in parallel
    """
    with ProcessPoolExecutor() as executor:
        futures = []
        for event_id in chunk_events:
            future = executor.submit(
                analyze_single_event,
                event_id,
                infrastructure,
                scenario,
                hazards,
                component_resp_df,
                components_costed,
                components_uncosted
            )
            futures.append(future)

        # Get results as they complete
        chunk_recovery_times = [
            f.result()['Full Restoration Time'].max()
            for f in futures]

    return chunk_recovery_times


def parallel_recovery_analysis(
        hazard_event_list,
        infrastructure,
        scenario,
        hazards,
        component_resp_df,
        components_costed,
        components_uncosted,
        chunk_size: int = 10000):
    """
    Parallel processing of recovery analysis across hazard events in chunks

    Parameters
    ----------
    hazard_event_list : list
        List of hazard events to process
    infrastructure : Infrastructure object
        The infrastructure system being analyzed
    scenario : Scenario object
        The scenario being analyzed
    hazards : Hazard object
        The hazards being analyzed
    component_resp_df : DataFrame
        Component response data
    components_costed : list
        List of costed component IDs
    components_uncosted : list
        List of uncosted component IDs
    chunk_size : int, optional
        Size of chunks to process at once, by default 20000

    Returns
    -------
    list
        List of recovery times for all hazard events
    """
    total_events = len(hazard_event_list)
    if total_events <= chunk_size:
        chunk_size = total_events
    num_chunks = math.ceil(total_events / chunk_size)
    recovery_times = []
    rootLogger.info(f"Processing {total_events} events in {num_chunks} chunks of {chunk_size}\n")

    # Chunk the hazard events
    pbar = tqdm(total=total_events, desc="Processing recovery analysis")
    for i in range(0, total_events, chunk_size):
        chunk = hazard_event_list[i:i + chunk_size]
        chunk_results = process_chunk(
            chunk,
            infrastructure,
            scenario,
            hazards,
            component_resp_df,
            components_costed,
            components_uncosted
        )
        recovery_times.extend(chunk_results)
        pbar.update(len(chunk))

    pbar.close()
    print()
    return recovery_times


def calculate_loss_stats(df, progress_bar=True):
    """Calculate summary statistics for loss -- using dash dataframe"""
    print()
    rootLogger.info(
        f"\n{Fore.CYAN}Calculating summary stats for system loss...{Fore.RESET}")
    with ProgressBar() if progress_bar else nullcontext():
        return {
            'Mean': df.loss_mean.mean().compute(),
            'Std': df.loss_mean.std().compute(),
            'Min': df.loss_mean.min().compute(),
            'Max': df.loss_mean.max().compute(),
            'Median': df.loss_mean.quantile(0.5).compute(),
            'Q1': df.loss_mean.quantile(0.25).compute(),
            'Q3': df.loss_mean.quantile(0.75).compute()
        }

def calculate_output_stats(df, progress_bar=True):
    """Calculate summary statistics for output -- using dash dataframe"""
    print()
    rootLogger.info(
        f"\n{Fore.CYAN}Calculating summary stats for system output...{Fore.RESET}")
    with ProgressBar() if progress_bar else nullcontext():
        return {
            'Mean': df.output_mean.mean().compute(),
            'Std': df.output_mean.std().compute(),
            'Min': df.output_mean.min().compute(),
            'Max': df.output_mean.max().compute(),
            'Median': df.output_mean.quantile(0.5).compute(),
            'Q1': df.output_mean.quantile(0.25).compute(),
            'Q3': df.output_mean.quantile(0.75).compute()
        }

def calculate_recovery_stats(df, progress_bar=True):
    """Calculate summary statistics for recovery time -- using dash dataframe"""
    print()
    rootLogger.info(
        f"\n{Fore.CYAN}Calculating summary stats for system recovery...{Fore.RESET}")
    with ProgressBar() if progress_bar else nullcontext():
        return {
            'Mean': df.recovery_time_100pct.mean().compute(),
            'Std': df.recovery_time_100pct.std().compute(),
            'Min': df.recovery_time_100pct.min().compute(),
            'Max': df.recovery_time_100pct.max().compute(),
            'Median': df.recovery_time_100pct.quantile(0.5).compute(),
            'Q1': df.recovery_time_100pct.quantile(0.25).compute(),
            'Q3': df.recovery_time_100pct.quantile(0.75).compute()
        }

def calculate_summary_statistics(df, calc_recovery=False):
    """Combine all summary statistics"""

    summary_stats = {
        'Loss': calculate_loss_stats(df),
        'Output': calculate_output_stats(df)
    }
    if calc_recovery:
        summary_stats['Recovery Time'] = calculate_recovery_stats(df)

    return summary_stats


def write_system_response(response_list, infrastructure, scenario, hazards):

    # ------------------------------------------------------------------------
    # haz_vs_ds_index_of_comp = response_list[0]
    # haz_vs_ds_df = pd.DataFrame.from_dict(haz_vs_ds_index_of_comp)
    # haz_vs_ds_table = pa.Table.from_pandas(haz_vs_ds_df)
    # haz_vs_ds_path = Path(scenario.raw_output_dir, 'ids_comp_vs_haz.parquet')
    # pq.write_table(haz_vs_ds_table, haz_vs_ds_path)

    # haz_vs_ds_index_of_comp = response_list[0]
    # idshaz = os.path.join(scenario.raw_output_dir, 'ids_comp_vs_haz.pickle')
    # with open(idshaz, 'wb') as handle:
    #     for response_key in sorted(haz_vs_ds_index_of_comp.keys()):
    #         pickle.dump(
    #             {response_key: haz_vs_ds_index_of_comp[response_key]},
    #             handle, pickle.HIGHEST_PROTOCOL
    #         )

    # ---------------------------------------------------------------------------------
    # Hazard response for component types
    # ---------------------------------------------------------------------------------
    comptype_resp_dict = response_list[3]

    comptype_resp_df = pd.DataFrame(comptype_resp_dict)
    comptype_resp_df.index.names = ['component_type', 'response']
    comptype_resp_df = comptype_resp_df.transpose()
    comptype_resp_df.index.name = 'hazard_event'

    outfile_comptype_resp = Path(scenario.output_path, 'comptype_response.csv')
    print("-" * 81)
    outpath_wrapped = utils.wrap_file_path(str(outfile_comptype_resp))
    rootLogger.info(f"Writing {Fore.CYAN}component type response{Fore.RESET} to: \n"
                    f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}")
    comptype_resp_df.to_csv(outfile_comptype_resp, sep=',')
    rootLogger.info("Done.\n")

    # ---------------------------------------------------------------------------------
    # Output File -- response of each COMPONENT to hazard
    # ---------------------------------------------------------------------------------

    costed_component_ids = set()
    for comp_id, component in infrastructure.components.items():
        if component.component_class not in infrastructure.uncosted_classes:
            costed_component_ids.add(comp_id)

    comp_response_list = response_list[2]
    component_resp_df = pd.DataFrame(comp_response_list)
    component_resp_df.columns = hazards.hazard_scenario_list
    component_resp_df.index.names = ['component_id', 'response']

    # Filter for costed components
    component_resp_df = component_resp_df[
        component_resp_df.index.get_level_values('component_id')
        .isin(costed_component_ids)
    ]

    component_resp_df = component_resp_df.transpose()
    component_resp_df.index.names = ['hazard_event']

    if scenario.hazard_input_method.lower() in ["calculated_array"]:
        outfile_comp_resp = Path(scenario.output_path, 'component_response.csv')
        outpath_wrapped = utils.wrap_file_path(str(outfile_comp_resp))
        rootLogger.info(f"\nWriting component hazard response data to: \n"
                        f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}")
        component_resp_df.to_csv(outfile_comp_resp, sep=',')
        rootLogger.info("Done.\n")

    # =================================================================================
    # Option to save as parquet file - if space if space becomes an issue
    # ---------------------------------------------------------------------------------
    # component_response_dict = response_list[2]
    # crd_df = pd.DataFrame.from_dict(component_response_dict)
    # crd_table = pa.Table.from_pandas(crd_df)
    # crd_path = Path(scenario.raw_output_dir, 'component_response_dict.parquet')
    # pq.write_table(crd_table, crd_path)

    # =================================================================================
    # System output file (for given hazard transfer parameter value)
    # ---------------------------------------------------------------------------------
    sys_output_dict = response_list[1]

    rootLogger.info("Collating data output line capacities of system ...")
    sys_output_df = pd.DataFrame(sys_output_dict)
    sys_output_df = sys_output_df.transpose()
    sys_output_df.index.name = 'event_id'

    # Get individual line capacities from output_nodes
    output_nodes_dict = infrastructure.output_nodes
    output_nodes_df = pd.DataFrame(output_nodes_dict)
    output_nodes_df = output_nodes_df.transpose()
    output_nodes_df.index.name = "output_node"
    line_capacities = output_nodes_df['output_node_capacity']

    # Calculate percentage values (0-100) by dividing by each line's capacity
    for line in sys_output_df.columns:
        # Get the capacity for the specific line
        line_capacity = line_capacities[line]
        # Convert to percentage of line capacity
        sys_output_df[line] = (sys_output_df[line] / line_capacity) * 100
        # Ensure values are between 0 and 100
        sys_output_df[line] = sys_output_df[line].clip(0, 100)

    outfile_sysoutput = Path(
        scenario.output_path, 'system_output_vs_hazard_intensity.csv')
    outpath_wrapped = utils.wrap_file_path(str(outfile_sysoutput))
    rootLogger.info(f"Writing {Fore.CYAN}system line capacity data{Fore.RESET} to: \n"
                    f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}")
    sys_output_df.to_csv(
        outfile_sysoutput, sep=',', index_label=[sys_output_df.index.name])
    rootLogger.info("Done.\n")

    # =================================================================================
    # Output File -- system response summary outputs
    # ---------------------------------------------------------------------------------

    hazard_col = hazards.HAZARD_INPUT_HEADER
    rootLogger.info("Collating data on system loss and output ...")

    out_cols = [
        'event_id',                 # [0] Formerly 'INTENSITY_MEASURE'
        'loss_mean',                # [1]
        'loss_std',                 # [2]
        'output_mean',              # [3]
        'output_std',               # [4]
        'recovery_time_100pct',     # [5]
    ]

    sys_economic_loss_array = response_list[5]
    sys_output_array = response_list[4]

    output_array = np.divide(sys_output_array, infrastructure.system_output_capacity)
    output_array_mean = np.mean(output_array, axis=0)
    output_array_std = np.std(output_array, axis=0)

    df_sys_response = pd.DataFrame(columns=out_cols)
    hazard_event_list = hazards.hazard_data_df.index.tolist()
    rootLogger.info("Done.\n")

    # -----------------------------------------------
    # Calculate recovery times for each hazard event

    if CALC_SYSTEM_RECOVERY:
        recovery_time_100pct = []
        components_uncosted = [
            comp_id for comp_id, component in infrastructure.components.items()
            if component.component_class in infrastructure.uncosted_classes]
        components_costed = [
            comp_id for comp_id in infrastructure.components.keys()
            if comp_id not in components_uncosted]

        rootLogger.info("Calculating system recovery information ...")
        recovery_time_100pct = parallel_recovery_analysis(
            hazard_event_list,
            infrastructure,
            scenario,
            hazards,
            component_resp_df,
            components_costed,
            components_uncosted)
    else:
        recovery_time_100pct = None

    # =================================================================================

    df_sys_response[out_cols[0]] = hazard_event_list
    df_sys_response[out_cols[1]] = np.mean(sys_economic_loss_array, axis=0)
    df_sys_response[out_cols[2]] = np.std(sys_economic_loss_array, axis=0)
    df_sys_response[out_cols[3]] = output_array_mean
    df_sys_response[out_cols[4]] = output_array_std
    df_sys_response[out_cols[5]] = recovery_time_100pct

    if (scenario.infrastructure_level).lower() == 'facility':
        if scenario.hazard_input_method in ['calculated_array', 'hazard_array']:
            site_id = '0'
        else:
            component1 = list(infrastructure.components.values())[0]
            site_id = str(component1.site_id)
        haz_vals = hazards.hazard_data_df[site_id].values
        df_sys_response.insert(1, hazard_col, haz_vals)

    df_sys_response = df_sys_response.sort_values('loss_mean', ascending=True)

    outfile_sys_response = Path(scenario.output_path, 'system_response.csv')
    outpath_wrapped = utils.wrap_file_path(str(outfile_sys_response))
    rootLogger.info(f"Writing {Fore.CYAN}system hazard response data{Fore.RESET} to:\n"
                    f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}")
    df_sys_response.to_csv(outfile_sys_response, sep=',', index=False)
    rootLogger.info("Done.\n")

    # =================================================================================
    # Risk calculations
    # ---------------------------------------------------------------------------------
    # Calculating summary statistics using Dask - for speed on large datasets

    df = dd.from_pandas(df_sys_response, npartitions=12)

    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(df, calc_recovery=CALC_SYSTEM_RECOVERY)

    # Convert to DataFrame for better display
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(Path(scenario.output_path, "risk_summary_statistics.csv"))
    summary_df.to_json(Path(scenario.output_path, "risk_summary_statistics.json"))

    # Print summary statistics
    print(f"\n{Fore.CYAN}Summary Statistics:{Fore.RESET}")
    print(summary_df.round(4))

    # --------------------------------------------------------------------------------
    # Calculate correlations

    print()
    rootLogger.info(
        f"\n{Fore.CYAN}Calculating correlations between loss & output...{Fore.RESET}")
    with ProgressBar():
        # Calculate appropriate fraction based on total rows
        total_rows = len(df.compute())  # Be careful with this on very large datasets
        desired_sample_size = min(1_000_000, total_rows)
        sample_fraction = desired_sample_size / total_rows
        # Sample the data using frac
        sample_df = df.sample(frac=sample_fraction).compute()

    # Create separate plots for each distribution/scatter
    plot_params = [
        {
            'x': 'loss_mean',
            'title': 'Distribution of Loss',
            'xlabel': 'Loss Ratio',
            'plot_type': 'hist'
        },
        {
            'x': 'output_mean',
            'title': 'Distribution of Output',
            'xlabel': 'Output Mean',
            'plot_type': 'hist'
        },
        {
            'x': 'loss_mean',
            'y': 'output_mean',
            'title': 'Loss Ratio vs System Output',
            'xlabel': 'Loss Ratio',
            'ylabel': 'Output Fraction',
            'plot_type': 'scatter'
        }
    ]

    # Create separate plots - count values on Y-axis
    plt.style.use("seaborn-v0_8-ticks")
    for params in plot_params:
        fig, ax = plt.subplots(figsize=(10, 8))

        if params.get('plot_type') == 'hist':
            sns.histplot(data=sample_df, x=params['x'], kde=True, ax=ax)
        else:  # scatter plot
            sns.scatterplot(data=sample_df, x=params['x'], y=params['y'], alpha=0.5, ax=ax)

        ax.set_title(params['title'])
        ax.set_xlabel(params['xlabel'])
        if params.get('ylabel'):
            ax.set_ylabel(params['ylabel'])

        plt.tight_layout()
        plot_name = params['title'].lower().replace(' ', '_')
        fig.savefig(
            Path(scenario.output_path, f"sys_{plot_name}__counts.png"),
            dpi=300, format='png')
        plt.close(fig)

    # Create separate plots - normalised Y-axis
    for params in plot_params:
        fig, ax = plt.subplots(figsize=(10, 8))

        if params.get('plot_type') == 'hist':
            sns.histplot(data=sample_df, x=params['x'], kde=True, ax=ax, stat='probability')
            ax.set_ylim(0, 1)
            title = f"{params['title']} (Normalised Frequency)"
        else:  # scatter plot
            sns.scatterplot(data=sample_df, x=params['x'], y=params['y'], alpha=0.5, ax=ax)
            title = f"{params['title']} (Sampled Data)"

        ax.set_title(title)
        ax.set_xlabel(params['xlabel'])
        if params.get('ylabel'):
            ax.set_ylabel(params['ylabel'])

        plt.tight_layout()
        # Save with descriptive filename
        plot_name = params['title'].lower().replace(' ', '_')
        fig.savefig(
            Path(scenario.output_path, f"sys_{plot_name}__normalised.png"),
            dpi=300, format='png')
        plt.close(fig)

    # =================================================================================
    # Calculate system fragility & exceedance probabilities
    # ---------------------------------------------------------------------------------

    # Infrastructure econ loss for sample
    sys_economic_loss_array = response_list[5]
    sys_ds_bounds = np.array(infrastructure.get_system_damage_state_bounds())

    # Vectorised fragility calculation
    comparisons = sys_economic_loss_array[:, :, np.newaxis] >= sys_ds_bounds
    sys_fragility = np.sum(comparisons, axis=2)

    # Adjust highest state in one operation
    sys_fragility[sys_economic_loss_array >= sys_ds_bounds[-1]] = len(sys_ds_bounds)

    # Prob of exceedance
    num_ds = len(infrastructure.get_system_damage_states())
    pe_sys_econloss = np.array([
        np.mean(sys_fragility >= ds, axis=0) for ds in range(num_ds)
    ], dtype=np.float32)

    # *************************************************************************
    # Suppressing saving of this file - due to the disc space and time required
    # -------------------------------------------------------------------------
    # path_sys_frag = Path(scenario.raw_output_dir, 'sys_fragility.npy')
    # outpath_wrapped = utils.wrap_file_path(str(path_sys_frag))
    # rootLogger.info(f"Writing system fragility data to: \n{outpath_wrapped}")
    # np.save(path_sys_frag, sys_fragility)
    # -------------------------------------------------------------------------

    if scenario.save_vars_npy:
        np.save(
            Path(scenario.raw_output_dir, 'sys_output_array.npy'),
            sys_output_array
        )

        np.save(
            Path(scenario.raw_output_dir, 'economic_loss_array.npy'),
            sys_economic_loss_array
        )

    if not str(scenario.infrastructure_level).lower() == "network":
        path_pe_sys_econloss = Path(scenario.raw_output_dir, 'pe_sys_econloss.npy')
        outpath_wrapped = utils.wrap_file_path(str(path_pe_sys_econloss))
        print()
        rootLogger.info(f"Writing prob of exceedance data to: \n"
                        f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}")
        np.save(path_pe_sys_econloss, pe_sys_econloss)
        rootLogger.info("Done.\n")

    return pe_sys_econloss

# -------------------------------------------------------------------------------------

# @njit
def _pe2pb(pe):
    """Numba-optimized version of pe2pb"""
    pex = np.sort(pe)[::-1]
    tmp = -1.0 * np.diff(pex)
    pb = np.zeros(len(pe) + 1)
    pb[1:-1] = tmp
    pb[-1] = pex[-1]
    pb[0] = 1 - pex[0]
    return pb

# -------------------------------------------------------------------------------------

def pe_by_component_class(response_list, infrastructure, scenario, hazards):
    """
    Calculates probability of exceedance based on failure of component classes.
    Damage state boundaries for Component Type Failures (Substations) are
    based on HAZUS MH MR3, p 8-66 to 8-68.

    Parameters:
    -----------
    response_list : list
    infrastructure : Infastructure object
    scenario : Scenario object
    hazards : HazardContainer object

    Returns:
    --------
    pe_sys_cpfailrate : numpy array
        array with exceedance probabilities of for component failures
    """
    if not str(infrastructure.system_class).lower() == 'substation':
        return None

    # Pre-calculate all indices and mappings once
    component_keys = list(infrastructure.components.keys())
    cp_classes_in_system = np.unique(list(infrastructure.get_component_class_list()))

    # Create mapping of class -> array of component indices
    cp_class_indices = {k: np.array([
        component_keys.index(comp_id)
        for comp_id, comp in infrastructure.components.items()
        if comp.component_class == k
    ]) for k in cp_classes_in_system}

    cp_classes_costed = [
        x for x in cp_classes_in_system
        if x not in infrastructure.uncosted_classes
    ]

    # Convert response data to single numpy array upfront
    num_samples = scenario.num_samples
    num_events = len(hazards.hazard_data_df)
    num_components = len(infrastructure.components)

    # Pre-allocate the full response array
    response_array = np.zeros((num_samples, num_events, num_components))

    # Fill response array (do this once instead of repeatedly accessing dict)
    for j, scenario_index in enumerate(hazards.hazard_data_df.index):
        event_id = scenario_index[0] if isinstance(scenario_index, (tuple, list)) else scenario_index
        response_array[:, j, :] = response_list[0][event_id]

    # --- System fragility - Based on Failure of Component Classes ---
    comp_class_failures = {}
    comp_class_frag = {}

    for compclass in cp_classes_costed:
        indices = cp_class_indices[compclass]
        if len(indices) == 0:
            continue

        # Calculate failures for entire class at once
        failures = (response_array[:, :, indices] >= 2).sum(axis=2) / len(indices)
        comp_class_failures[compclass] = failures

        # Calculate fragility using vectorised operations
        ds_lims = np.array(infrastructure.get_ds_lims_for_compclass(compclass))
        comp_class_frag[compclass] = (failures[:, :, np.newaxis] > ds_lims).sum(axis=2)

    # Probability of Exceedance -- Based on Failure of Component Classes
    pe_sys_cpfailrate = np.zeros(
        (len(infrastructure.system_dmg_states), hazards.num_hazard_pts))

    for d in range(len(infrastructure.system_dmg_states)):
        exceedance_probs = []
        for compclass in cp_classes_costed:
            if compclass in comp_class_frag:
                class_exceed = (comp_class_frag[compclass] >= d).mean(axis=0)
                exceedance_probs.append(class_exceed)

        if exceedance_probs:
            pe_sys_cpfailrate[d, :] = np.median(exceedance_probs, axis=0)

    # Vectorised damage ratio calculations
    exp_damage_ratio = np.zeros((len(infrastructure.components), hazards.num_hazard_pts))
    hazard_data = hazards.hazard_data_df.values

    # Process in large batches for better vectorization
    for comp_class in cp_classes_costed:
        indices = cp_class_indices[comp_class]
        if len(indices) == 0:
            continue

        batch_components = [infrastructure.components[component_keys[i]] for i in indices]

        for i, component in enumerate(batch_components):
            comp_idx = indices[i]
            try:
                loc_params = component.get_location()
                site_id = str(loc_params[0]) if isinstance(loc_params, tuple) else '0'

                if site_id in hazards.hazard_data_df.columns:
                    site_col_idx = hazards.hazard_data_df.columns.get_loc(site_id)
                    hazard_intensities = hazard_data[:, site_col_idx]

                    # Vectorised response calculation
                    pe_ds = np.zeros((hazards.num_hazard_pts, len(component.damage_states)))
                    valid_mask = ~np.isnan(hazard_intensities)

                    if np.any(valid_mask):
                        for ds_idx in component.damage_states.keys():
                            pe_ds[valid_mask, ds_idx] = \
                                component.damage_states[ds_idx].response_function(
                                    hazard_intensities[valid_mask])

                        # Vectorised damage ratio calculation
                        pe_ds = pe_ds[:, 1:]  # Remove first state
                        pb = _pe2pb(pe_ds[0])  # Calculate probability bins
                        dr = np.array([
                            component.damage_states[int(ds)].damage_ratio
                            for ds in range(len(component.damage_states))
                        ])
                        exp_damage_ratio[comp_idx, valid_mask] = \
                            np.sum(pb * dr * component.cost_fraction)

            except Exception as e:
                rootLogger.warning(f"Error calculating damage ratio for component {comp_idx}: {e}")
                continue

    # Save results
    if scenario.save_vars_npy:
        np.save(
            Path(scenario.raw_output_dir, 'exp_damage_ratio.npy'),
            exp_damage_ratio
        )

    if scenario.hazard_input_method.lower() in ["calculated_array"]:
        path_sys_cpfailrate = Path(scenario.raw_output_dir, 'pe_sys_cpfailrate.npy')
        np.save(path_sys_cpfailrate, pe_sys_cpfailrate)

    return pe_sys_cpfailrate
