import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Any, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

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
    Streamlined recovery analysis that robustly handles event ID lookups and DataFrame structures.
    """
    logger.info("Starting streamlined recovery analysis...")
    logger.info(f"Recovery method: {recovery_method}")
    logger.info(f"Number of repair streams: {num_repair_streams}")
    total_events = len(hazard_event_list)
    logger.info(f"Total events to process: {total_events:,}")
    logger.info(f"Component response DataFrame shape: {component_resp_df.shape}")
    logger.info(f"DataFrame has MultiIndex columns: {isinstance(component_resp_df.columns, pd.MultiIndex)}")
    logger.info(f"DataFrame index dtype: {component_resp_df.index.dtype}")

    # Validate DataFrame structure
    validation_results = validate_dataframe_structure(component_resp_df, components_costed)
    logger.info(f"Validation results: {validation_results}")

    recovery_times = []
    successful_events = 0
    events_with_damage = 0

    for i, event_id in enumerate(tqdm(hazard_event_list, desc="Calculating recovery times")):
        try:
            event_recovery_time = process_single_event(
                event_id, component_resp_df, components_costed, infrastructure,
                recovery_method, num_repair_streams
            )
            recovery_times.append(event_recovery_time)
            if event_recovery_time > 0:
                events_with_damage += 1
            successful_events += 1
            # Log progress for first few events and periodically
            if i < 5 or (i + 1) % max(1, total_events // 10) == 0:
                logger.info(f"Event {i+1}/{total_events} (ID: {event_id}): recovery_time = {event_recovery_time}")
        except Exception as e:
            logger.warning(f"Error processing event {event_id}: {e}")
            recovery_times.append(0.0)

    print("*" * 80)
    print("In Module: recovery_analysis_optimised")
    print(f"events_with_damage:\n    {events_with_damage}")
    print(f"recovery_times:\n    {recovery_times}")
    print("*" * 80)

    logger.info(f"Successfully processed: {successful_events}/{total_events} events")
    logger.info(f"Events with damage (recovery_time > 0): {events_with_damage}/{total_events}")
    if events_with_damage > 0:
        non_zero_times = [t for t in recovery_times if t > 0]
        max_time = max(non_zero_times)
        avg_time = np.mean(non_zero_times)
        logger.info(f"Max recovery time: {max_time:.1f}, Average (non-zero): {avg_time:.1f}")
    else:
        logger.warning("No events had any recovery time > 0. This suggests a data processing issue.")
    return recovery_times

def validate_dataframe_structure(component_resp_df: pd.DataFrame, components_costed: List[str]) -> dict:
    """
    Validate the structure of the component response DataFrame.
    """
    results = {
        'has_multiindex_columns': isinstance(component_resp_df.columns, pd.MultiIndex),
        'components_found': 0,
        'damage_state_columns': 0,
        'func_mean_columns': 0,
        'sample_columns': []
    }
    if isinstance(component_resp_df.columns, pd.MultiIndex):
        level_0_values = component_resp_df.columns.get_level_values(0).unique()
        level_1_values = component_resp_df.columns.get_level_values(1).unique()
        results['level_0_values'] = list(level_0_values)[:10]
        results['level_1_values'] = list(level_1_values)
        for comp_id in components_costed:
            if comp_id in level_0_values:
                results['components_found'] += 1
        results['damage_state_columns'] = sum(1 for val in level_1_values if 'damage_state' in str(val))
        results['func_mean_columns'] = sum(1 for val in level_1_values if 'func_mean' in str(val))
    else:
        results['sample_columns'] = list(component_resp_df.columns[:10])
        for comp_id in components_costed:
            if any(
                    f"{comp_id}_damage_state" in str(col) or f"{comp_id}.damage_state" in str(col)
                    for col in component_resp_df.columns):
                results['components_found'] += 1
        results['damage_state_columns'] = sum(
            1 for col in component_resp_df.columns if 'damage_state' in str(col))
        results['func_mean_columns'] = sum(
            1 for col in component_resp_df.columns if 'func_mean' in str(col))
    return results


def process_single_event(
    event_id: str,
    component_resp_df: pd.DataFrame,
    components_costed: List[str],
    infrastructure: Any,
    recovery_method: str,
    num_repair_streams: int
) -> float:
    """
    Process a single event to calculate recovery time, with robust event ID handling.
    """
    try:
        event_data = None
        # Try direct lookup
        if event_id in component_resp_df.index:
            event_data = component_resp_df.loc[event_id]
        else:
            # Try as integer
            try:
                event_id_int = int(event_id)
                if event_id_int in component_resp_df.index:
                    event_data = component_resp_df.loc[event_id_int]
            except (ValueError, TypeError):
                pass
        if event_data is None:
            # Try as string
            event_id_str = str(event_id)
            if event_id_str in component_resp_df.index:
                event_data = component_resp_df.loc[event_id_str]
        if event_data is None:
            # Try by position if event_id is a digit
            try:
                if isinstance(event_id, (int, str)) and str(event_id).isdigit():
                    idx = int(event_id)
                    if 0 <= idx < len(component_resp_df):
                        event_data = component_resp_df.iloc[idx]
            except (ValueError, IndexError):
                pass
        if event_data is None:
            logger.warning(f"Event ID {event_id} not found in component_resp_df index.")
            return 0.0

        if isinstance(event_data, pd.DataFrame):
            event_data = event_data.iloc[0]

        component_recovery_times = []
        for comp_id in components_costed:
            try:
                damage_state, functionality = extract_component_state(
                    event_data, comp_id, component_resp_df.columns
                )
                if damage_state > 0:
                    recovery_time = calculate_component_recovery_time(
                        damage_state, functionality, comp_id, infrastructure
                    )
                    if recovery_time > 0:
                        component_recovery_times.append(recovery_time)
            except Exception as e:
                logger.debug(f"Error processing component {comp_id} for event {event_id}: {e}")
                continue

        if not component_recovery_times:
            return 0.0

        if recovery_method == 'parallel_streams':
            return calculate_parallel_streams_recovery(component_recovery_times, num_repair_streams)
        else:
            return max(component_recovery_times)
    except Exception as e:
        logger.debug(f"Error in process_single_event for event {event_id}: {e}")
        return 0.0

def extract_component_state(
    event_data: pd.Series,
    comp_id: str,
    columns: pd.Index
) -> Tuple[int, float]:
    """
    Extract damage state and functionality for a component from event data.
    """
    damage_state = 0
    functionality = 1.0
    if isinstance(columns, pd.MultiIndex):
        damage_state_key = (comp_id, 'damage_state')
        func_key = (comp_id, 'func_mean')
        if damage_state_key in columns:
            try:
                value = event_data[damage_state_key]
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                if not pd.isna(value):
                    damage_state = int(float(value))
            except (ValueError, TypeError, KeyError):
                pass
        if func_key in columns:
            try:
                value = event_data[func_key]
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                if not pd.isna(value):
                    functionality = float(value)
            except (ValueError, TypeError, KeyError):
                pass
        # Fallback: try alternative column names
        if damage_state == 0:
            for level1_val in columns.get_level_values(1).unique():
                if 'damage' in str(level1_val).lower() and 'state' in str(level1_val).lower():
                    alt_key = (comp_id, level1_val)
                    if alt_key in columns:
                        try:
                            value = event_data[alt_key]
                            if isinstance(value, pd.Series):
                                value = value.iloc[0]
                            if not pd.isna(value):
                                damage_state = int(float(value))
                                break
                        except (ValueError, TypeError, KeyError):
                            continue
        if functionality == 1.0:
            for level1_val in columns.get_level_values(1).unique():
                if 'func' in str(level1_val).lower() and 'mean' in str(level1_val).lower():
                    alt_key = (comp_id, level1_val)
                    if alt_key in columns:
                        try:
                            value = event_data[alt_key]
                            if isinstance(value, pd.Series):
                                value = value.iloc[0]
                            if not pd.isna(value):
                                functionality = float(value)
                                break
                        except (ValueError, TypeError, KeyError):
                            continue
    else:
        # Handle flat columns
        damage_state_patterns = [
            f"{comp_id}_damage_state", f"{comp_id}.damage_state",
            f"damage_state_{comp_id}", f"{comp_id}_dmg_state", f"{comp_id}.dmg_state"
        ]
        func_patterns = [
            f"{comp_id}_func_mean", f"{comp_id}.func_mean",
            f"func_mean_{comp_id}", f"{comp_id}_functionality", f"{comp_id}.functionality"
        ]
        for pattern in damage_state_patterns:
            if pattern in columns:
                try:
                    value = event_data[pattern]
                    if not pd.isna(value):
                        damage_state = int(float(value))
                        break
                except (ValueError, TypeError, KeyError):
                    continue
        for pattern in func_patterns:
            if pattern in columns:
                try:
                    value = event_data[pattern]
                    if not pd.isna(value):
                        functionality = float(value)
                        break
                except (ValueError, TypeError, KeyError):
                    continue
    return damage_state, functionality

def calculate_component_recovery_time(
    damage_state: int,
    functionality: float,
    comp_id: str,
    infrastructure: Any,
    recovery_threshold: float = 0.98
) -> float:
    """
    Calculate recovery time for a component based on its damage state and functionality.
    """
    if damage_state == 0 or functionality >= recovery_threshold:
        return 0.0
    try:
        if hasattr(infrastructure, 'components') and comp_id in infrastructure.components:
            component = infrastructure.components[comp_id]
            if hasattr(component, 'damage_states') and damage_state in component.damage_states:
                ds = component.damage_states[damage_state]
                # If recovery function is available, try to use it
                if hasattr(ds, 'recovery_function') and callable(ds.recovery_function):
                    try:
                        recovery_progress_needed = max(0.0, (recovery_threshold - functionality) / recovery_threshold)
                        base_time = damage_state * 100  # Base time proportional to damage state
                        return base_time * (1 + recovery_progress_needed)
                    except Exception:
                        pass
                if hasattr(ds, 'recovery_time'):
                    try:
                        recovery_time = float(ds.recovery_time)
                        if recovery_time > 0:
                            functionality_factor = max(0.0, (recovery_threshold - functionality) / recovery_threshold)
                            return recovery_time * (1 + functionality_factor)
                    except (ValueError, TypeError):
                        pass
                # Fallback calculation
                base_recovery_time = damage_state * 168  # 168 hours = 1 week per damage state
                functionality_deficit = max(0.0, recovery_threshold - functionality)
                functionality_factor = functionality_deficit / recovery_threshold
                recovery_time = base_recovery_time * (1 + functionality_factor)
                return max(0.0, recovery_time)
        # Final fallback
        return damage_state * 100.0
    except Exception:
        return damage_state * 100.0

def calculate_parallel_streams_recovery(
    component_recovery_times: List[float],
    num_repair_streams: int
) -> float:
    """
    Calculate system recovery time when components can be repaired in parallel streams.
    """
    if not component_recovery_times:
        return 0.0
    component_recovery_times_sorted = sorted(component_recovery_times, reverse=True)
    num_streams = min(num_repair_streams, len(component_recovery_times_sorted))
    streams = [0.0] * num_streams
    for recovery_time in component_recovery_times_sorted:
        min_stream_idx = streams.index(min(streams))
        streams[min_stream_idx] += recovery_time
    return max(streams)
