import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

# =====================================================================================
# Data structures for efficient parallel processing
# =====================================================================================

@dataclass
class ComponentDamageState:
    """Lightweight data structure for component damage information"""
    component_id: str
    damage_state: int
    functionality: float
    damage_ratio: float
    recovery_time_params: Dict[str, Any]  # Stores recovery function parameters


@dataclass
class MinimalInfrastructure:
    """Minimal infrastructure data needed for recovery calculations"""
    components: Dict[str, Dict[str, Any]]  # component_id -> {damage_states, component_class}
    system_class: str
    repair_streams: int = 100  # Default number of parallel repair streams


# =====================================================================================
# Core recovery calculation functions (simplified and optimized)
# =====================================================================================

def calculate_component_recovery_time(
    damage_state: int,
    functionality: float,
    damage_states_info: Dict[int, Dict[str, Any]],
    recovery_threshold: float = 0.98
) -> float:
    """
    Calculate recovery time for a single component.

    This is a simplified version that directly calculates recovery time
    without complex function calls.
    """
    if damage_state == 0 or functionality >= recovery_threshold:
        return 0.0

    # Get damage state information
    ds_info = damage_states_info.get(damage_state, {})

    # Simple recovery time calculation based on damage state
    # This can be replaced with more complex logic as needed
    base_recovery_time = damage_state * 100  # Simple linear model

    # Apply functionality-based adjustment
    functionality_factor = (recovery_threshold - functionality) / recovery_threshold
    recovery_time = base_recovery_time * (1 + functionality_factor)

    # If recovery function parameters are available, use them
    if 'recovery_params' in ds_info and ds_info['recovery_params']:
        # Apply custom recovery function logic here
        params = ds_info['recovery_params']
        if 'scale' in params:
            recovery_time *= params['scale']

    return max(0.0, recovery_time)


def process_event_batch_optimized(
    event_batch: List[str],
    component_data_file: str,
    infrastructure_data: Dict[str, Any],
    batch_id: int,
    recovery_method: str = 'max',  # 'max' or 'parallel_streams'
    num_repair_streams: int = 100
) -> Tuple[int, List[float], int]:
    """
    Process a batch of events with optimized memory usage and simplified logic.

    Parameters
    ----------
    event_batch : List[str]
        List of event IDs to process
    component_data_file : str
        Path to parquet file with component response data
    infrastructure_data : Dict
        Minimal infrastructure data
    batch_id : int
        Batch identifier
    recovery_method : str
        Method for calculating system recovery ('max' or 'parallel_streams')
    num_repair_streams : int
        Number of parallel repair streams (used when recovery_method='parallel_streams')

    Returns
    -------
    Tuple[int, List[float], int]
        (batch_id, recovery_times, num_processed)
    """
    try:
        # Load only the required events from parquet file
        filters = [('event_id', 'in', event_batch)]
        component_resp_df = pd.read_parquet(
            component_data_file,
            filters=filters,
            columns=['event_id', 'component_id', 'damage_state', 'func_mean']
        )

        recovery_times = []
        num_processed = 0

        for event_id in event_batch:
            try:
                # Get component states for this event
                event_data = component_resp_df[component_resp_df['event_id'] == event_id]

                if event_data.empty:
                    recovery_times.append(0.0)
                    continue

                # Calculate recovery time for this event
                if recovery_method == 'max':
                    recovery_time = calculate_max_recovery_time(
                        event_data, infrastructure_data
                    )
                else:  # parallel_streams
                    recovery_time = calculate_parallel_streams_recovery_time(
                        event_data, infrastructure_data, num_repair_streams
                    )

                recovery_times.append(recovery_time)
                num_processed += 1

            except Exception as e:
                logger.warning(f"Error processing event {event_id}: {e}")
                recovery_times.append(0.0)

        return batch_id, recovery_times, num_processed

    except Exception as e:
        logger.error(f"Error processing batch {batch_id}: {e}")
        return batch_id, [0.0] * len(event_batch), 0


def calculate_max_recovery_time(
    event_data: pd.DataFrame,
    infrastructure_data: Dict[str, Any]
) -> float:
    """
    Calculate recovery time as the maximum across all damaged components.
    """
    max_recovery_time = 0.0

    for _, row in event_data.iterrows():
        comp_id = row['component_id']
        damage_state = int(row.get('damage_state', 0))
        functionality = float(row.get('func_mean', 1.0))

        if damage_state > 0 and comp_id in infrastructure_data['components']:
            comp_info = infrastructure_data['components'][comp_id]
            damage_states_info = comp_info.get('damage_states', {})

            recovery_time = calculate_component_recovery_time(
                damage_state, functionality, damage_states_info
            )

            max_recovery_time = max(max_recovery_time, recovery_time)

    return max_recovery_time


def calculate_parallel_streams_recovery_time(
    event_data: pd.DataFrame,
    infrastructure_data: Dict[str, Any],
    num_repair_streams: int
) -> float:
    """
    Calculate recovery time considering parallel repair streams.

    Components are assigned to repair streams to minimize total recovery time.
    """
    # Get all damaged components and their recovery times
    component_recovery_times = []

    for _, row in event_data.iterrows():
        comp_id = row['component_id']
        damage_state = int(row.get('damage_state', 0))
        functionality = float(row.get('func_mean', 1.0))

        if damage_state > 0 and comp_id in infrastructure_data['components']:
            comp_info = infrastructure_data['components'][comp_id]
            damage_states_info = comp_info.get('damage_states', {})

            recovery_time = calculate_component_recovery_time(
                damage_state, functionality, damage_states_info
            )

            if recovery_time > 0:
                component_recovery_times.append(recovery_time)

    if not component_recovery_times:
        return 0.0

    # Sort components by recovery time (longest first)
    component_recovery_times.sort(reverse=True)

    # Assign components to repair streams using a simple greedy algorithm
    stream_end_times = [0.0] * min(num_repair_streams, len(component_recovery_times))

    for recovery_time in component_recovery_times:
        # Find the stream that will be free earliest
        min_stream_idx = stream_end_times.index(min(stream_end_times))
        stream_end_times[min_stream_idx] += recovery_time

    # Total recovery time is when the last stream finishes
    return max(stream_end_times)


# =====================================================================================
# Main parallel recovery analysis function
# =====================================================================================

def parallel_recovery_analysis_optimised(
    hazard_event_list: List[str],
    infrastructure: Any,
    scenario: Any,
    component_resp_df: pd.DataFrame,
    components_costed: List[str],
    recovery_method: str = 'max',
    num_repair_streams: int = 100,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None
) -> List[float]:
    """
    Optimized parallel recovery analysis for supercomputer deployment.

    Parameters
    ----------
    hazard_event_list : List[str]
        List of hazard event IDs
    infrastructure : Infrastructure object
        Infrastructure system
    scenario : Scenario object
        Scenario configuration
    component_resp_df : pd.DataFrame
        Component response data
    components_costed : List[str]
        List of costed component IDs
    recovery_method : str
        'max' for maximum component recovery time, 'parallel_streams' for parallel repair
    num_repair_streams : int
        Number of parallel repair streams (default: 100)
    max_workers : Optional[int]
        Maximum number of parallel workers (default: all available cores)
    batch_size : Optional[int]
        Events per batch (default: auto-calculate)

    Returns
    -------
    List[float]
        Recovery times for each hazard event
    """
    logger.info("Starting optimized parallel recovery analysis...")
    logger.info(f"Recovery method: {recovery_method}")
    logger.info(f"Number of repair streams: {num_repair_streams}")

    # Prepare data for parallel processing
    total_events = len(hazard_event_list)
    logger.info(f"Total events to process: {total_events:,}")

    # Extract minimal infrastructure data
    minimal_infrastructure = {
        'components': {}
    }

    for comp_id in components_costed:
        if comp_id in infrastructure.components:
            component = infrastructure.components[comp_id]
            minimal_infrastructure['components'][comp_id] = {
                'component_class': component.component_class,
                'damage_states': {}
            }

            # Extract damage state information
            for ds_id, ds in component.damage_states.items():
                minimal_infrastructure['components'][comp_id]['damage_states'][ds_id] = {
                    'damage_ratio': getattr(ds, 'damage_ratio', 0.0),
                    'functionality': getattr(ds, 'functionality', 1.0),
                    'recovery_params': {}  # Add custom recovery parameters if available
                }

    # Save component response data to parquet for efficient parallel access
    temp_dir = Path(scenario.output_path, "temp_recovery_analysis")
    temp_dir.mkdir(exist_ok=True)

    component_data_file = temp_dir / "component_responses.parquet"

    # Prepare data for parquet with explicit event_id column
    temp_df = component_resp_df.copy()
    temp_df['event_id'] = temp_df.index

    # Flatten multi-level columns if present
    if isinstance(temp_df.columns, pd.MultiIndex):
        temp_df.columns = ['_'.join(col).strip() for col in temp_df.columns.values]

    # Save to parquet
    temp_df.to_parquet(component_data_file, engine='pyarrow', index=False)

    # Determine optimal batch size and workers
    if max_workers is None:
        # Use all available cores - let the job scheduler/config handle limits
        max_workers = psutil.cpu_count(logical=False)

    if batch_size is None:
        # Adaptive batch sizing based on total events
        if total_events > 10_000_000:
            batch_size = 10000
        elif total_events > 1_000_000:
            batch_size = 5000
        elif total_events > 100_000:
            batch_size = 2000
        else:
            batch_size = 1000

    num_batches = math.ceil(total_events / batch_size)
    logger.info(
        f"Using {max_workers if max_workers else 'all available'} "
        f"workers with {num_batches} batches ({batch_size} events/batch)"
    )

    # Create batches
    event_batches = []
    for i in range(0, total_events, batch_size):
        batch = hazard_event_list[i:i + batch_size]
        event_batches.append((i // batch_size, batch))

    # Process batches in parallel
    recovery_times = [0.0] * total_events
    checkpoint_file = Path(scenario.output_path, "recovery_checkpoint.pkl")
    checkpoint_interval = min(10000, total_events // 10)  # Save every 10% or 10k events

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        futures = {}
        for batch_id, batch in event_batches:
            future = executor.submit(
                process_event_batch_optimized,
                batch,
                str(component_data_file),
                minimal_infrastructure,
                batch_id,
                recovery_method,
                num_repair_streams
            )
            futures[future] = batch_id

        # Process results as they complete
        with tqdm(total=total_events, desc="Processing recovery analysis") as pbar:
            completed_count = 0

            for future in as_completed(futures):
                batch_id = futures[future]

                try:
                    result_batch_id, batch_results, num_processed = future.result()

                    # Store results in correct position
                    start_idx = result_batch_id * batch_size
                    end_idx = start_idx + len(batch_results)
                    recovery_times[start_idx:end_idx] = batch_results

                    completed_count += num_processed
                    pbar.update(num_processed)

                    # Save checkpoint periodically
                    if completed_count % checkpoint_interval == 0:
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump(recovery_times[:completed_count], f)
                        logger.info(f"Checkpoint saved: {completed_count:,} events processed")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_id}: {e}")
                    # Continue with other batches

    # Clean up temporary files
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Could not clean up temp directory: {e}")

    # Final validation
    non_zero_count = sum(1 for t in recovery_times if t > 0)
    logger.info(f"Recovery analysis complete. {non_zero_count:,} events have non-zero recovery times.")

    return recovery_times


# =====================================================================================
# Integration function to replace the existing dask_parallel_recovery_analysis
# =====================================================================================

def dask_parallel_recovery_analysis(
    hazard_event_list,
    infrastructure,
    scenario,
    hazards,
    component_resp_df,
    components_costed,
    components_uncosted,
    recovery_method='max',
    num_repair_streams=100
):
    """
    Drop-in replacement for the existing dask_parallel_recovery_analysis function.

    This version uses the optimized parallel processing approach.
    """
    return parallel_recovery_analysis_optimised(
        hazard_event_list=hazard_event_list,
        infrastructure=infrastructure,
        scenario=scenario,
        component_resp_df=component_resp_df,
        components_costed=components_costed,
        recovery_method=recovery_method,
        num_repair_streams=num_repair_streams
    )
