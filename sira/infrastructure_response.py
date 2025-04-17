import logging
import pickle
import math
import os
import time
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
from typing import List, Union, Tuple, Optional, Dict, Any

import dask  # type: ignore
import dask.dataframe as dd   # type: ignore
from dask.diagnostics.progress import ProgressBar  # type: ignore

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import threading
import traceback
from numba import njit
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory, cpu_count

from colorama import Fore, init
init()

import sira.loss_analysis as loss_analysis
from sira.tools import utils
from sira.tools.parallelisation import get_available_cores
from sira.modelling.responsemodels import Algorithm
import psutil  # For memory monitoring

rootLogger = logging.getLogger(__name__)
matplotlib.use('Agg')
plt.switch_backend('agg')

mpl_looger = logging.getLogger('matplotlib')
mpl_looger.setLevel(logging.WARNING)

CALC_SYSTEM_RECOVERY = True

# Configure Dask settings for better performance:
dask.config.set({"dataframe.shuffle.method": "tasks"})
# Maximum rows to process in each worker chunk:
MAX_ROWS_PER_CHUNK = 5000

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
        event_id,
        infrastructure,
        scenario,
        hazards,
        component_resp_df,
        components_costed,
        components_uncosted
):
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

# =====================================================================================

# Function to extract minimal infrastructure data
def extract_essential_infrastructure_data(infrastructure):
    """
    Extract only the essential infrastructure data needed for recovery calculation

    This function minimizes memory usage by extracting only the necessary
    attributes for recovery time calculation.

    Parameters
    ----------
    infrastructure : Infrastructure object
        Full infrastructure model

    Returns
    -------
    dict
        Minimal infrastructure data needed for recovery analysis
    """
    # Only extract the minimal data needed for recovery calculation
    minimal_data = {
        'system_class': infrastructure.system_class,
        'system_output_capacity': infrastructure.system_output_capacity,
        'uncosted_classes': infrastructure.uncosted_classes,
        'components': {}
    }

    # Only include necessary component attributes for recovery calculation
    for comp_id, component in infrastructure.components.items():
        comp_data = {
            'component_class': component.component_class,
            'damage_states': {}
        }

        # Extract only damage state data needed for recovery
        for ds_id, ds in component.damage_states.items():
            comp_data['damage_states'][ds_id] = {
                'damage_ratio': ds.damage_ratio,
                'functionality': ds.functionality,
                'recovery_function_constructor': ds.recovery_function_constructor \
                if hasattr(ds, 'recovery_function_constructor') else None
            }

        minimal_data['components'][comp_id] = comp_data

    return minimal_data

# Function to extract minimal scenario data
def extract_essential_scenario_data(scenario):
    """
    Extract only the essential scenario data needed for recovery calculation

    Parameters
    ----------
    scenario : Scenario object
        Full scenario model

    Returns
    -------
    dict
        Minimal scenario data needed for recovery analysis
    """
    # Only extract the minimal data needed for recovery calculation
    return {
        'time_step': scenario.time_step,
        'restoration_pct_steps': scenario.restoration_pct_steps,
        'output_path': str(scenario.output_path)
    }


def to_dask_dataframe(df, num_partitions=None):
    """
    Convert pandas DataFrame to optimally partitioned Dask DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to convert
    num_partitions : int, optional
        Number of partitions to use. If None, calculates based on CPU count and dataframe size.

    Returns
    -------
    dask.dataframe.DataFrame
        Dask DataFrame with optimal partitioning
    """
    if df is None or df.empty:
        return None

    # Determine appropriate number of partitions based on dataframe size and available cores
    if num_partitions is None:
        # Estimate memory per row (in bytes)
        sample_size = min(1000, len(df))
        if sample_size > 0:
            memory_per_row = df.memory_usage(deep=True).sum() / len(df)

            # Target partition size: aim for ~100MB chunks (adjust as needed)
            target_partition_size = 100 * 1024 * 1024  # 100MB in bytes
            rows_per_partition = max(1, int(target_partition_size / memory_per_row))

            # Calculate partitions needed
            num_partitions = max(1, min(
                cpu_count() * 2,  # Cap at 2x CPU count
                math.ceil(len(df) / rows_per_partition)
            ))
        else:
            num_partitions = 1

    # Create Dask DataFrame with calculated partitions
    return dd.from_pandas(df, npartitions=num_partitions)


def process_event_batch(
        batch_events,
        component_resp_df_path,
        minimal_infrastructure,
        minimal_scenario,
        components_costed,
        batch_id
):
    """
    Process a small batch of events using Dask dataframes, minimizing memory usage

    Parameters
    ----------
    batch_events : list
        List of event IDs to process
    component_resp_df_path : str
        Path to the parquet file containing component response data
    minimal_infrastructure : dict
        Minimal infrastructure data needed for recovery analysis
    minimal_scenario : dict
        Minimal scenario data needed for recovery analysis
    components_costed : list
        List of costed component IDs
    batch_id : int
        Identifier for this batch

    Returns
    -------
    tuple
        (batch_id, recovery_times, num_processed)
    """
    start_time = time.time()
    batch_size = len(batch_events)

    # Monitor memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    try:
        # Load only the required portion of the dataframe for these events
        # Reading from parquet with filters
        filters = [[('event_id', 'in', batch_events)]]
        # Assumes a column titled "event_id" is present in the parquet file
        batch_df = pd.read_parquet(
            component_resp_df_path,
            filters=filters
        )

        mid_memory = process.memory_info().rss / 1024 / 1024  # MB

        recovery_times = []
        num_processed = 0

        # Process each event in the batch
        for event_id in batch_events:
            try:
                # Check if this event exists in our filtered dataframe
                if event_id in batch_df.index:
                    # Process the event using minimal data
                    event_data = batch_df.loc[event_id]

                    # Calculate recovery time
                    recovery_time = calculate_recovery_time(
                        event_id,
                        event_data,
                        minimal_infrastructure,
                        components_costed
                    )
                    recovery_times.append(recovery_time)
                    num_processed += 1
                else:
                    # Event not found - add default
                    recovery_times.append(0)
            except Exception as e:
                # Log error but continue with next event
                print(f"Error processing event {event_id} in batch {batch_id}: {e}")
                recovery_times.append(0)

        # Calculate processing statistics
        end_time = time.time()
        processing_time = end_time - start_time
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_change = final_memory - initial_memory

        print(
            f"Batch {batch_id}: Processed {num_processed}/{batch_size} "
            f"events in {processing_time:.2f}s. "
            f"Memory: {initial_memory:.1f}MB → "
            f"{final_memory:.1f}MB ({memory_change:+.1f}MB)"
        )

        return batch_id, recovery_times, num_processed

    except Exception as e:
        print(f"Error processing batch {batch_id}: {e}")
        print(traceback.format_exc())
        # Return empty results with batch ID to maintain order
        return batch_id, [0] * len(batch_events), 0


def calculate_recovery_time(
        event_id,
        event_data,
        minimal_infrastructure,
        components_costed):
    """
    Calculate recovery time for a single event using minimal component data

    Parameters
    ----------
    event_id : str
        Event identifier
    event_data : pandas.Series or pandas.DataFrame
        Component response data for this event
    minimal_infrastructure : dict
        Minimal infrastructure data for recovery calculation
    components_costed : list
        List of costed component IDs

    Returns
    -------
    float
        Recovery time for this event
    """
    try:
        # Extract component states for this event
        event_component_states = {}

        # Process component data for this event
        # Handle different DataFrame formats
        if isinstance(event_data, pd.Series):
            # Series format - with MultiIndex columns
            for comp_id in components_costed:
                try:
                    ds_key = (comp_id, 'damage_state')
                    if ds_key in event_data:
                        damage_state = int(event_data[ds_key])
                        functionality = float(
                            event_data.get((comp_id, 'func_mean'), 1.0)
                        )
                        event_component_states[comp_id] = {
                            'damage_state': damage_state,
                            'functionality': functionality
                        }
                except Exception:
                    # Skip components with missing or invalid data
                    pass
        else:
            # Try to handle as DataFrame or other format
            try:
                for comp_id in components_costed:
                    try:
                        damage_state = int(
                            event_data.get((comp_id, 'damage_state'), 0))
                        functionality = float(
                            event_data.get((comp_id, 'func_mean'), 1.0))
                        event_component_states[comp_id] = {
                            'damage_state': damage_state,
                            'functionality': functionality
                        }
                    except (KeyError, ValueError, TypeError):
                        # Skip this component
                        pass
            except Exception:
                # Handle case where event_data is in a different format
                pass

        # Calculate recovery times for damaged components
        max_recovery_time = 0
        recovery_times = []

        for comp_id, state in event_component_states.items():
            if comp_id in minimal_infrastructure['components']:
                damage_state = state['damage_state']
                comp_info = minimal_infrastructure['components'][comp_id]

                if damage_state > 0 and damage_state in comp_info['damage_states']:
                    # Calculate recovery time based on damage state
                    ds_info = comp_info['damage_states'][damage_state]

                    # Different methods to get recovery time based on available data
                    if 'recovery_function_constructor' in \
                            ds_info and ds_info['recovery_function_constructor']:
                        # Use recovery function if available
                        try:
                            recovery_func = Algorithm.factory(ds_info['recovery_function_constructor'])
                            recovery_curve = recovery_func(1.0)  # Get full recovery time
                            recovery_time = max(recovery_curve)
                            recovery_times.append(recovery_time)
                        except Exception:
                            # Fallback if recovery function fails
                            recovery_time = damage_state * 100  # Simple estimate
                            recovery_times.append(recovery_time)
                    else:
                        # Simple recovery time estimation based on damage state
                        recovery_time = damage_state * 100  # Simple estimate
                        recovery_times.append(recovery_time)

        # Get maximum recovery time
        if recovery_times:
            max_recovery_time = max(recovery_times)

        return max_recovery_time

    except Exception as e:
        # Log error but continue processing
        print(f"Error calculating recovery time for event {event_id}: {e}")
        return 0  # Default recovery time

# =====================================================================================

def dask_parallel_recovery_analysis(
        hazard_event_list,
        infrastructure,
        scenario,
        hazards,
        component_resp_df,
        components_costed,
        components_uncosted):
    """
    Parallel recovery analysis using Dask for processing large datasets
    with minimal memory usage

    This implementation uses Dask dataframes to efficiently process data in
    partitions and manages memory carefully to avoid out-of-memory errors.

    Parameters
    ----------
    hazard_event_list : list
        List of hazard events to process
    infrastructure : Infrastructure object
        Infrastructure system being analyzed
    scenario : Scenario object
        Scenario being analyzed
    hazards : Hazard object
        Hazards being analyzed
    component_resp_df : pandas.DataFrame
        Component response data
    components_costed : list
        List of costed component IDs
    components_uncosted : list
        List of uncosted component IDs

    Returns
    -------
    list
        List of recovery times for all events
    """
    rootLogger.info("Starting Dask-based recovery analysis with memory tracking...")

    # Monitor available memory
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)

    rootLogger.info(f"Total physical memory: {total_gb:.2f} GB")
    rootLogger.info(f"Available memory: {available_gb:.2f} GB")
    rootLogger.info(f"Current memory usage: {(total_gb - available_gb):.2f} GB")

    total_events = len(hazard_event_list)

    # Create temp directory for intermediate files
    temp_dir = Path(scenario.output_path, "temp_dask_recovery")
    temp_dir.mkdir(exist_ok=True)

    # Convert component_resp_df to Dask dataframe and save to parquet
    rootLogger.info("Converting component response data to Dask format...")

    try:
        # Save component_resp_df to parquet with event_id column for filtering
        comp_resp_path = Path(temp_dir, "component_resp.parquet")

        # Make sure we have an explicit event_id column for filtering
        temp_df = component_resp_df.copy()
        if 'event_id' not in temp_df.columns and isinstance(temp_df.index, pd.Index):
            # Add event_id as explicit column from index
            temp_df['event_id'] = temp_df.index

        # Save to parquet with index
        temp_df.to_parquet(comp_resp_path, engine='pyarrow', index=True)

        # Free memory
        del temp_df

        # Extract minimal infrastructure and scenario data
        rootLogger.info("Extracting essential infrastructure and scenario data...")
        minimal_infrastructure = extract_essential_infrastructure_data(infrastructure)
        minimal_scenario = extract_essential_scenario_data(scenario)

        # Determine optimal batch and worker configuration
        available_cores = min(get_available_cores(), 48)  # Cap at 48 cores

        # Calculate memory-based limits
        # Estimate 2GB base per worker + overhead
        memory_per_worker_gb = 2.0
        max_workers_by_memory = max(1, int(available_gb / memory_per_worker_gb))
        num_workers = min(available_cores, max_workers_by_memory)

        # Calculate optimal batch size
        # Smaller batches for very large datasets to avoid memory issues
        if total_events > 1_000_000:
            batch_size = 1000
        elif total_events > 100_000:
            batch_size = 2000
        else:
            batch_size = 5000

        # Cap batch size at MAX_ROWS_PER_CHUNK
        batch_size = min(batch_size, MAX_ROWS_PER_CHUNK)

        # Calculate number of batches
        num_batches = math.ceil(total_events / batch_size)

        rootLogger.info(
            f"Processing {total_events} events in {num_batches} batches with "
            f"{batch_size} events per batch using {num_workers} workers"
        )

        # Create checkpoint file for recovery
        checkpoint_file = Path(scenario.output_path, "recovery_checkpoint.pkl")
        recovery_times = []

        # Load checkpoint if it exists
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    recovery_times = pickle.load(f)
                    rootLogger.info(f"Loaded checkpoint with {len(recovery_times)} events already processed")

                    # If we have all results, return them
                    if len(recovery_times) == total_events:
                        rootLogger.info("All events already processed in checkpoint. Returning results.")
                        return recovery_times

                    # If we have partial results, adjust hazard_event_list
                    hazard_event_list = hazard_event_list[len(recovery_times):]
                    total_events = len(hazard_event_list)
                    num_batches = math.ceil(total_events / batch_size)

                    rootLogger.info(f"Continuing with remaining {total_events} events")
            except Exception as e:
                rootLogger.warning(f"Failed to load checkpoint: {e}")
                recovery_times = []

        # Progress tracking
        pbar = tqdm(total=total_events, desc="Processing recovery analysis")
        progress_lock = threading.Lock()

        # Create batches of events
        batches = []
        for i in range(0, total_events, batch_size):
            end_idx = min(i + batch_size, total_events)
            batch_id = i // batch_size
            current_batch = hazard_event_list[i:end_idx]
            batches.append((batch_id, current_batch))

        # Results collection
        all_results = {}
        checkpoint_interval = 5000  # Save checkpoint every 5000 events

        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            # Submit all batches for processing
            for batch_id, batch_events in batches:
                future = executor.submit(
                    process_event_batch,
                    batch_events,
                    str(comp_resp_path),
                    minimal_infrastructure,
                    minimal_scenario,
                    components_costed,
                    batch_id
                )
                futures.append(future)

            # Process results as they complete
            for future in as_completed(futures):
                try:
                    batch_id, batch_results, num_processed = future.result()
                    all_results[batch_id] = batch_results

                    # Update progress
                    with progress_lock:
                        pbar.update(len(batches[batch_id][1]))  # Update with actual batch size

                    # Save checkpoint at regular intervals
                    current_processed = sum(len(batches[i][1]) for i in all_results.keys())
                    if current_processed % checkpoint_interval == 0:
                        # Sort and combine results from all processed batches
                        temp_recovery_times = recovery_times.copy()
                        for i in sorted(all_results.keys()):
                            temp_recovery_times.extend(all_results[i])

                        # Save checkpoint
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump(temp_recovery_times, f)
                        rootLogger.info(f"Checkpoint saved: {len(temp_recovery_times)} events processed")

                except Exception as e:
                    rootLogger.error(f"Error processing batch: {str(e)}")
                    # Add default values for this batch to maintain result order
                    batch_id = futures.index(future)
                    if 0 <= batch_id < len(batches):
                        all_results[batch_id] = [0] * len(batches[batch_id][1])

        # Close progress bar
        pbar.close()

        # Combine results in correct order
        batch_results = []
        for i in range(len(batches)):
            if i in all_results:
                batch_results.extend(all_results[i])
            else:
                # Fill with zeros for missing batches
                batch_results.extend([0] * len(batches[i][1]))

        # Combine with any previously checkpointed results
        recovery_times.extend(batch_results)

        # Save final checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(recovery_times, f)

        # Verify we have the correct number of results
        if len(recovery_times) < total_events + len(hazard_event_list) - len(batches[0][1]):
            rootLogger.warning(
                f"Expected {total_events} results but got {len(recovery_times)}. "
                "Adding default values to match."
            )
            recovery_times = recovery_times + [0] * (
                total_events + len(hazard_event_list) - len(batches[0][1]) - len(recovery_times)
            )

        return recovery_times

    except Exception as e:
        rootLogger.error(f"Dask-based processing failed: {str(e)}")
        rootLogger.error(traceback.format_exc())
        rootLogger.info("Falling back to sequential processing...")

        return sequential_recovery_analysis(
            hazard_event_list,
            infrastructure,
            scenario,
            hazards,
            component_resp_df,
            components_costed,
            components_uncosted
        )

    finally:
        # Clean up temporary files
        try:
            if 'comp_resp_path' in locals() and Path(comp_resp_path).exists():
                # Optionally keep files for debugging
                if not scenario.save_vars_npy:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    rootLogger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            rootLogger.warning(f"Error cleaning up temp files: {e}")

# =====================================================================================

def sequential_recovery_analysis(
        hazard_event_list,
        infrastructure,
        scenario,
        hazards,
        component_resp_df,
        components_costed,
        components_uncosted):
    """
    Sequential processing fallback for recovery analysis
    """
    rootLogger.info("Processing events sequentially...")
    recovery_times = []

    # Use smaller chunks even in sequential mode to minimize memory usage
    chunk_size = 1000
    num_chunks = math.ceil(len(hazard_event_list) / chunk_size)

    # Extract minimal data once
    minimal_infrastructure = extract_essential_infrastructure_data(infrastructure)

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(hazard_event_list))
        chunk_events = hazard_event_list[start_idx:end_idx]

        chunk_pbar = tqdm(
            chunk_events,
            desc=f"Processing chunk {chunk_idx + 1}/{num_chunks}",
            leave=False
        )

        chunk_results = []
        for event_id in chunk_pbar:
            try:
                # Use the original analyze_single_event function
                result = analyze_single_event(
                    event_id,
                    infrastructure,
                    scenario,
                    hazards,
                    component_resp_df,
                    components_costed,
                    components_uncosted
                )

                # Extract only the recovery time
                if result and 'Full Restoration Time' in result:
                    recovery_time = result['Full Restoration Time'].max()
                else:
                    recovery_time = 0

                chunk_results.append(recovery_time)
            except (KeyError, ValueError, AttributeError, RuntimeError, IndexError) as e:
                rootLogger.warning(f"Error processing event {event_id}: {str(e)}")
                chunk_results.append(0)  # Default recovery time

        # Extend recovery times with chunk results
        recovery_times.extend(chunk_results)

        # Save checkpoint after each chunk
        checkpoint_file = Path(scenario.output_path, "recovery_checkpoint_sequential.pkl")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(recovery_times, f)
        rootLogger.info(f"Checkpoint saved: {len(recovery_times)}/{len(hazard_event_list)} events processed")

    return recovery_times


def calculate_loss_stats(df, progress_bar=True):
    """Calculate summary statistics for loss -- using dask dataframe"""
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
    """Calculate summary statistics for output -- using dask dataframe"""
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
    """Calculate summary statistics for recovery time -- using dask dataframe"""
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
        rootLogger.info(f"Writing component hazard response data to: \n"
                        f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}")
        component_resp_df.to_csv(outfile_comp_resp, sep=',')
        rootLogger.info("Done.\n")

    # =================================================================================
    # System output file (for given hazard transfer parameter value)
    # ---------------------------------------------------------------------------------
    sys_output_dict = response_list[1]

    rootLogger.info("Collating data on output line capacities of system ...")
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

    # Create dataframe without recovery times first
    df_sys_response = pd.DataFrame(columns=out_cols[:5])
    hazard_event_list = hazards.hazard_data_df.index.tolist()
    rootLogger.info("Done.\n")

    # -----------------------------------------------
    # Calculate recovery times for each hazard event
    df_sys_response[out_cols[0]] = hazard_event_list
    df_sys_response[out_cols[1]] = np.mean(sys_economic_loss_array, axis=0)
    df_sys_response[out_cols[2]] = np.std(sys_economic_loss_array, axis=0)
    df_sys_response[out_cols[3]] = output_array_mean
    df_sys_response[out_cols[4]] = output_array_std

    if CALC_SYSTEM_RECOVERY:
        recovery_time_100pct = []
        components_uncosted = [
            comp_id for comp_id, component in infrastructure.components.items()
            if component.component_class in infrastructure.uncosted_classes]
        components_costed = [
            comp_id for comp_id in infrastructure.components.keys()
            if comp_id not in components_uncosted]

        rootLogger.info("Calculating system recovery information ...")

        # Use our new dask-based recovery analysis
        recovery_time_100pct = dask_parallel_recovery_analysis(
            hazard_event_list,
            infrastructure,
            scenario,
            hazards,
            component_resp_df,
            components_costed,
            components_uncosted)

        # IMPORTANT: Ensure recovery_time_100pct has the correct length
        if recovery_time_100pct is not None:
            if len(recovery_time_100pct) != len(hazard_event_list):
                rootLogger.warning(
                    f"Recovery time length ({len(recovery_time_100pct)}) does not match "
                    f"hazard event list length ({len(hazard_event_list)}). Adjusting..."
                )

                if len(recovery_time_100pct) < len(hazard_event_list):
                    # Add zeros to match length
                    recovery_time_100pct.extend([0] * (len(hazard_event_list) - len(recovery_time_100pct)))
                else:
                    # Truncate to match length
                    recovery_time_100pct = recovery_time_100pct[:len(hazard_event_list)]

            # Now add to dataframe
            df_sys_response[out_cols[5]] = recovery_time_100pct
    else:
        # Add zeros for recovery time if not calculated
        df_sys_response[out_cols[5]] = [0] * len(hazard_event_list)

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

    dask_df = to_dask_dataframe(df_sys_response)

    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(dask_df, calc_recovery=CALC_SYSTEM_RECOVERY)

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
        total_rows = len(dask_df.compute())  # Be careful with this on very large datasets
        desired_sample_size = min(1_000_000, total_rows)
        sample_fraction = desired_sample_size / total_rows
        # Sample the data using frac
        sample_df = dask_df.sample(frac=sample_fraction).compute()

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
        outpath_wrapped = utils.wrap_file_path(str(path_pe_sys_econloss), max_width=120)
        print()
        rootLogger.info(f"Writing prob of exceedance data to: \n"
                        f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}")
        np.save(path_pe_sys_econloss, pe_sys_econloss)
        rootLogger.info("Done.\n")

    return pe_sys_econloss

# -------------------------------------------------------------------------------------

@njit
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

def exceedance_prob_by_component_class(response_list, infrastructure, scenario, hazards):
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
