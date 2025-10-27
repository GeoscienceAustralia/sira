import logging
import math
import os
import pickle
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from colorama import Fore
from tqdm import tqdm

from sira import loss_analysis
from sira.configuration import Configuration
from sira.modelling.component import Component
from sira.modelling.hazard import Hazard, HazardsContainer

# Configure logging
rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.INFO)

# Constants for recovery calculation
RESTORATION_THRESHOLD = 0.95
MAX_FUNCTIONALITY_THRESHOLD = 0.95  # Below this, component is considered damaged
MIN_FUNCTIONALITY_THRESHOLD = 0.03  # Lowest functionality threshold considered
MIN_RECOVERY_TIME = 1  # Set the minimum recovery time for damaged components

# Performance monitoring
ENABLE_PROFILING = os.environ.get("SIRA_ENABLE_PROFILING", "0") == "1"
PROFILE_OUTPUT_DIR = os.environ.get("SIRA_PROFILE_DIR", ".")


def profile_performance(func):
    """Decorator to profile function performance when enabled."""
    if not ENABLE_PROFILING:
        return func

    def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        from pathlib import Path

        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            elapsed_time = time.time() - start_time

            # Save profile data
            profile_dir = Path(PROFILE_OUTPUT_DIR)
            profile_dir.mkdir(exist_ok=True)
            profile_file = profile_dir / f"{func.__name__}_{int(time.time())}.prof"
            profiler.dump_stats(str(profile_file))

            # Generate readable stats
            stats_file = profile_dir / f"{func.__name__}_{int(time.time())}.txt"
            with open(stats_file, "w") as f:
                stats = pstats.Stats(profiler, stream=f)
                stats.sort_stats("cumulative")
                stats.print_stats(50)  # Top 50 functions

            rootLogger.info(f"Profile saved: {profile_file}")
            rootLogger.info(f"Function {func.__name__} completed in {elapsed_time:.2f}s")

    return wrapper


def monitor_memory_usage():
    """Monitor current memory usage."""
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return memory_mb
    except ImportError:
        return None


def log_system_resources():
    """Log current system resource usage."""
    try:
        import psutil

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent

        rootLogger.info(
            f"System Resources - CPU: {cpu_percent}% ({cpu_count} cores), "
            f"Memory: {memory_percent}% ({memory_available_gb:.1f}GB available), "
            f"Disk: {disk_percent}%"
        )

    except ImportError:
        rootLogger.warning("psutil not available for resource monitoring")


def is_mpi_environment():
    """
    Detect if we're in a proper MPI environment.

    Returns
    -------
    bool
        True if in proper MPI environment, False otherwise
    """
    # Check for explicit disabling
    if os.environ.get("SIRA_FORCE_NO_MPI", "0") == "1":
        return False

    # Check for SLURM (most common HPC scheduler)
    slurm_vars = ["SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_PROCID", "SLURM_NODELIST"]
    if any(var in os.environ for var in slurm_vars):
        return True

    # Check for PBS/Torque (another common HPC scheduler)
    pbs_vars = ["PBS_JOBID", "PBS_NCPUS", "PBS_NODEFILE"]
    if any(var in os.environ for var in pbs_vars):
        return True

    # Check for explicit MPI runtime variables
    mpi_runtime_vars = [
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_RANK",  # OpenMPI
        "PMI_SIZE",
        "PMI_RANK",  # MPICH
        "MPI_LOCALRANKID",
        "MPI_LOCALNRANKS",  # Intel MPI
    ]
    if any(var in os.environ for var in mpi_runtime_vars):
        return True

    # Check if we're being launched with mpirun/mpiexec
    parent_process = os.environ.get("_", "")
    if any(launcher in parent_process for launcher in ["mpirun", "mpiexec", "srun"]):
        return True

    # Check for HPC-specific hostnames or environments
    hostname = os.environ.get("HOSTNAME", "")
    if any(pattern in hostname.lower() for pattern in ["hpc", "cluster", "node", "compute"]):
        return True

    return False


def safe_mpi_import():
    """
    Safely attempt to import and initialise MPI only if in proper environment.

    Returns
    -------
    tuple
        (mpi_module, comm, rank, size) or (None, None, 0, 1) if not available
    """
    if not is_mpi_environment():
        return None, None, 0, 1

    try:
        # Set environment variable to prevent automatic initialisation
        os.environ["MPI4PY_RC_INITIALISE"] = "False"

        from mpi4py import MPI

        # Check if already initialised
        if not MPI.Is_initialized():
            MPI.Init()

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        rootLogger.info(f"MPI successfully initialised: rank {rank} of {size}")
        return MPI, comm, rank, size

    except ImportError:
        rootLogger.warning("MPI4Py not available even in MPI environment")
        return None, None, 0, 1
    except Exception as e:
        rootLogger.warning(f"MPI initialisation failed: {e}")
        return None, None, 0, 1


class RecoveryAnalysisEngine:
    """
    Main engine for parallel recovery analysis supporting multiple backends:
    - MPI4Py for HPC environments
    - Multiprocessing for single-node systems
    """

    def __init__(self, backend="auto"):
        """
        Initialise recovery analysis engine.

        Parameters
        ----------
        backend : str
            'mpi', 'multiprocessing', or 'auto' (detects best available)
        """
        self.backend = self._select_backend(backend)
        self.comm = None
        self.rank = 0
        self.size = 1
        self.MPI = None

        if self.backend == "mpi":
            self.MPI, self.comm, self.rank, self.size = safe_mpi_import()
            if self.MPI is None:
                rootLogger.warning(
                    "MPI backend requested but not available. Falling back to multiprocessing."
                )
                self.backend = "multiprocessing"

    def _select_backend(self, requested_backend):
        """Select the best available backend."""
        if requested_backend == "auto":
            # Check for MPI environment first
            if is_mpi_environment():
                return "mpi"
            else:
                return "multiprocessing"
        return requested_backend

    @profile_performance
    def analyse(
        self,
        config: Configuration,
        hazards: HazardsContainer,
        infrastructure: Any,
        scenario: Any,
        components_costed: List[str],
        recovery_method: str = "max",
        num_repair_streams: int = 100,
    ) -> List[float]:
        """
        Main analysis function that routes to appropriate backend.
        """
        hazard_event_list = hazards.hazard_scenario_list

        # Log system resources at start
        log_system_resources()

        rootLogger.info(f"Starting recovery analysis with backend: {self.backend}")
        rootLogger.info(f"Processing {len(hazard_event_list)} events")
        rootLogger.info(f"Recovery method: {recovery_method}, Streams: {num_repair_streams}")

        if self.backend == "mpi":
            return self._analyse_mpi(
                config,
                hazards,
                hazard_event_list,
                infrastructure,
                scenario,
                components_costed,
                recovery_method,
                num_repair_streams,
            )
        else:
            return self._analyse_multiprocessing(
                config,
                hazards,
                hazard_event_list,
                infrastructure,
                components_costed,
                recovery_method,
                num_repair_streams,
            )

    def _analyse_multiprocessing(
        self,
        config: Configuration,
        hazards: HazardsContainer,
        hazard_event_list: List[str],
        infrastructure: Any,
        components_costed: List[str],
        recovery_method: str,
        num_repair_streams: int,
    ) -> List[float]:
        """Multiprocessing implementation for single-node systems."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        hazard_event_list = hazards.hazard_scenario_list

        # Extract infrastructure data once
        # infra_data = extract_infrastructure_data(infrastructure)

        # Determine number of workers
        n_workers = min(os.cpu_count() or 1, len(hazard_event_list))

        rootLogger.info(f"Using multiprocessing with {n_workers} workers")

        # Create chunks of events
        chunk_size = max(1, len(hazard_event_list) // (n_workers * 4))
        event_chunks = [
            hazard_event_list[i : i + chunk_size]
            for i in range(0, len(hazard_event_list), chunk_size)
        ]

        # Process chunks in parallel
        recovery_times = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(
                    process_event_chunk,
                    chunk,
                    config,
                    hazards,
                    components_costed,
                    infrastructure,
                    recovery_method,
                    num_repair_streams,
                ): chunk
                for chunk in event_chunks
            }

            # Collect results maintaining order
            chunk_results = {}
            with tqdm(total=len(event_chunks), desc="Processing chunks") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    chunk_idx = event_chunks.index(chunk)
                    try:
                        result = future.result()
                        chunk_results[chunk_idx] = result
                        pbar.update(1)
                    except Exception as e:
                        rootLogger.error(f"Error processing chunk: {e}")
                        chunk_results[chunk_idx] = [0.0] * len(chunk)
                        pbar.update(1)

            # Combine results in order
            for i in range(len(event_chunks)):
                recovery_times.extend(chunk_results[i])

        return recovery_times

    def _analyse_mpi(
        self,
        config: Configuration,
        hazards: HazardsContainer,
        hazard_event_list: List[str],
        infrastructure: Any,
        scenario: Any,
        components_costed: List[str],
        recovery_method: str,
        num_repair_streams: int,
    ) -> List[float]:
        """Optimised MPI4Py implementation for HPC environments like NCI Gadi."""
        if self.MPI is None or self.comm is None:
            rootLogger.error("MPI not available. Falling back to multiprocessing.")
            self.backend = "multiprocessing"
            return self._analyse_multiprocessing(
                config,
                hazards,
                hazard_event_list,
                infrastructure,
                components_costed,
                recovery_method,
                num_repair_streams,
            )

        start_time = time.time()

        # Only rank 0 prepares data for broadcasting
        if self.rank == 0:
            rootLogger.info(
                f"Starting MPI analysis with {self.size} ranks for {len(hazard_event_list)} events"
            )

            # Extract minimal infrastructure data for better broadcast performance
            infra_data = extract_infrastructure_data(infrastructure)

            # Prepare hazard data for efficient access
            hazard_lookup = {event_id: i for i, event_id in enumerate(hazards.hazard_scenario_list)}

            # Create optimised work distribution using cyclic assignment for better load balancing
            work_assignment = [[] for _ in range(self.size)]
            for i, event_id in enumerate(hazard_event_list):
                rank_id = i % self.size
                work_assignment[rank_id].append(event_id)

            broadcast_time = time.time()
        else:
            infra_data = None
            hazard_lookup = None
            work_assignment = None
            broadcast_time = time.time()

        # Efficient broadcast of essential data only
        infra_data = self.comm.bcast(infra_data, root=0)
        hazard_lookup = self.comm.bcast(hazard_lookup, root=0)

        if self.rank == 0:
            rootLogger.info(f"Broadcast completed in {time.time() - broadcast_time:.2f}s")

        # Scatter work assignments
        local_events = self.comm.scatter(work_assignment if self.rank == 0 else None, root=0)

        # Process local events with progress tracking
        local_results = []
        processed_count = 0
        local_start = time.time()

        for event_id in local_events:
            try:
                if event_id not in hazard_lookup:
                    rootLogger.warning(
                        f"Rank {self.rank}: Event {event_id} not found in hazard lookup"
                    )
                    local_results.append(0.0)
                    continue

                hazard_idx = hazard_lookup[event_id]
                hazard_obj = hazards.listOfhazards[hazard_idx]

                recovery_time = calculate_event_recovery(
                    config,
                    event_id,
                    hazard_obj,
                    components_costed,
                    infrastructure,
                    recovery_method,
                    num_repair_streams,
                )
                local_results.append(recovery_time)
                processed_count += 1

                # Progress reporting every 100 events
                if processed_count % 100 == 0:
                    elapsed = time.time() - local_start
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    rootLogger.info(
                        f"Rank {self.rank}: Processed {processed_count}/"
                        f"{len(local_events)} events, rate: {rate:.1f} events/s"
                    )

            except Exception as e:
                rootLogger.error(f"Rank {self.rank}: Error processing event {event_id}: {e}")
                local_results.append(0.0)

        # Gather results with timing
        gather_start = time.time()
        all_results = self.comm.gather(local_results, root=0)

        if self.rank == 0:
            gather_time = time.time() - gather_start
            total_time = time.time() - start_time

            rootLogger.info(f"Gather completed in {gather_time:.2f}s")
            rootLogger.info(f"Total MPI analysis time: {total_time:.2f}s")

            # Reconstruct results maintaining original event order
            recovery_times = [0.0] * len(hazard_event_list)
            event_to_idx = {event_id: i for i, event_id in enumerate(hazard_event_list)}

            if all_results is not None:
                for rank_idx, rank_results in enumerate(all_results):
                    rank_events = work_assignment[rank_idx]  # type: ignore
                    for event_id, result in zip(rank_events, rank_results):
                        original_idx = event_to_idx[event_id]
                        recovery_times[original_idx] = result

            return recovery_times
        else:
            return []

    def cleanup(self):
        """Cleanup resources."""
        # MPI finalisation is handled automatically by mpi4py.
        # No-op for multiprocessing cleanup.
        return None


def extract_infrastructure_data(infrastructure: Any) -> Dict:
    """
    Extract essential infrastructure data for recovery calculation.
    Includes proper recovery function data.
    """
    data = {
        "components": {},
        "system_class": getattr(infrastructure, "system_class", "unknown"),
        "output_capacity": getattr(infrastructure, "system_output_capacity", 1.0),
    }

    for comp_id, component_obj in infrastructure.components.items():
        comp_data = {
            "component_type": component_obj.component_type,
            "component_class": component_obj.component_class,
            "cost_fraction": getattr(component_obj, "cost_fraction"),
            "damage_states": {},
        }

        # Extract damage state information
        for ds_idx, damage_state in component_obj.damage_states.items():
            ds_recovery_config = {
                "damage_ratio": getattr(damage_state, "damage_ratio"),
                "functionality": getattr(damage_state, "functionality"),
                "recovery_function": getattr(damage_state, "recovery_function"),
            }

            # Check for recovery function
            if hasattr(damage_state, "recovery_function_constructor"):
                ds_recovery_config["recovery_function_constructor"] = (
                    damage_state.recovery_function_constructor
                )

            comp_data["damage_states"][ds_idx] = ds_recovery_config

        data["components"][comp_id] = comp_data

    return data


def calculate_event_recovery(
    config: Configuration,
    event_id: str,
    hazard_obj: Hazard,
    components_costed: List[str],
    infrastructure: Any,
    recovery_method: str = "max",
    num_repair_streams: int = 100,
) -> int:
    """
    Calculate recovery time for a single event with robust error handling.
    """
    # component_recovery_times = []
    # damaged_components = 0

    recovery_times = OrderedDict()
    nodes_with_recovery = set()
    event_id = str(event_id)

    # hazard_obj = hazards.listOfhazards[hazards.hazard_scenario_list.index(event_id)]
    for comp_id in components_costed:
        # if comp_id not in infra_data["components"]:
        #     print(f"Component {comp_id} not found in infrastructure data")
        #     continue
        comp_obj = infrastructure.components[comp_id]
        loc_params = comp_obj.get_location()
        hazval = hazard_obj.get_hazard_intensity(*loc_params)
        # print(f"\nComponent: {comp_id} | hazard intensity: {hazval} | event: {event_id}")
        try:
            comp_restoration_time = loss_analysis.calc_component_recovery_time(
                config,
                comp_obj,
                event_id,
                hazval,
                threshold_recovery=RESTORATION_THRESHOLD,
            )
            # print(f"  --> Restoration time A : {comp_restoration_time}")
            if (
                isinstance(comp_restoration_time, (int, float))
                and comp_restoration_time is not None
            ):
                comp_restoration_time = int(round(comp_restoration_time))
            else:
                comp_restoration_time = 0

            # print(f"  --> Restoration time B : {comp_restoration_time}")
            if comp_restoration_time > 0:
                nodes_with_recovery.add(comp_id)

        except Exception as e:
            rootLogger.error(f"*** Recovery time calculation failed for component {comp_id}: {e}")
            comp_restoration_time = 0

        recovery_times[comp_id] = comp_restoration_time
        # print(f"  --> Restoration time C : {recovery_times[comp_id]}")

    # Future: Apply specific recovery method
    # if recovery_method == "parallel_streams":
    #     sys_recovery_time = calculate_constrained_recovery(recovery_times, num_repair_streams)
    # else:  # max method
    #     sys_recovery_time = max(recovery_times.values())

    sys_recovery_time = max(recovery_times.values())  # Default to max restoration time

    if not sys_recovery_time:
        sys_recovery_time = 0

    # Log if we found damaged components but no recovery times
    if nodes_with_recovery is not None:
        if len(nodes_with_recovery) > 0 and sys_recovery_time == 0:
            rootLogger.warning(
                f"Found {len(nodes_with_recovery)}, but zero recovery time. Error in recovery."
            )

    return sys_recovery_time


def extract_component_state(
    event_response_components: pd.Series, comp_id: str
) -> Tuple[int, float]:
    """
    Extract damage state and functionality from event data.
    """
    damage_state = 0
    functionality = 1.0

    # Check if columns are MultiIndex
    if isinstance(event_response_components.index, pd.MultiIndex):
        # Try standard MultiIndex access
        try:
            dmg_val = event_response_components.get((comp_id, "damage_index"))
            func_val = event_response_components.get((comp_id, "func_mean"))
            if dmg_val is not None:
                damage_state = int(dmg_val)
            if func_val is not None:
                functionality = float(func_val)
        except (KeyError, ValueError):
            # Try alternative names
            for idx in event_response_components.index:
                if idx[0] == comp_id:
                    if "damage" in str(idx[1]).lower():
                        try:
                            val = event_response_components[idx]
                            if isinstance(val, pd.Series):
                                val = val.iloc[0]
                            damage_state = int(val)
                        except (ValueError, TypeError):
                            pass
                    elif "func" in str(idx[1]).lower():
                        try:
                            val = event_response_components[idx]
                            if isinstance(val, pd.Series):
                                val = val.iloc[0]
                            functionality = float(val)
                        except (ValueError, TypeError):
                            pass
    else:
        # Handle flat index with various naming patterns
        patterns = {
            "damage": [
                f"{comp_id}_damage_state",
                f"{comp_id}.damage_state",
                f"{comp_id}_dmg_state",
                f"damage_state_{comp_id}",
                f"{comp_id}_ds",
            ],
            "func": [
                f"{comp_id}_func_mean",
                f"{comp_id}.func_mean",
                f"{comp_id}_functionality",
                f"func_mean_{comp_id}",
                f"{comp_id}_func",
            ],
        }

        # Try damage state patterns
        for pattern in patterns["damage"]:
            if pattern in event_response_components.index:
                try:
                    val = event_response_components[pattern]
                    if not pd.isna(val):
                        damage_state = int(float(val))
                        break
                except Exception:
                    continue

        # Try functionality patterns
        for pattern in patterns["func"]:
            if pattern in event_response_components.index:
                try:
                    val = event_response_components[pattern]
                    if not pd.isna(val):
                        functionality = float(val)
                        break
                except Exception:
                    continue

    return damage_state, functionality


# def _calculate_probabilistic_recovery_fast(
#     comp_id: str,
#     event_id: str,
#     comp_obj: Component,
#     hazards: HazardsContainer,
#     functionality: float,
#     threshold_recovery: float,
#     config: Optional[Any] = None,
# ) -> float:
#     """
#     HAZUS-compliant probabilistic recovery using fast discrete restoration functions.

#     This implements the HAZUS formula:
#     Recovery Time = Sum(Recovery_Time_i x P[damage_state_i])

#     But uses fast discrete lookups instead of inverse CDF calculations.
#     """

#     # Get hazard intensity for probability calculations
#     hazard_obj = hazards.listOfhazards[hazards.hazard_scenario_list.index(event_id)]
#     loc_params = comp_obj.get_location()
#     hazard_intensity = hazard_obj.get_hazard_intensity(*loc_params)

#     # Calculate damage state probabilities (same as original HAZUS method)
#     damage_functions = [ds.response_function for ds in comp_obj.damage_states.values()]
#     num_dmg_states = len(comp_obj.damage_states.keys())

#     # Calculate exceedance probabilities
#     pe = np.array([damage_functions[d](hazard_intensity) for d in range(num_dmg_states)])
#     pe = np.clip(pe, 0.0, 1.0)
#     for i in range(1, len(pe)):
#         pe[i] = min(pe[i], pe[i - 1])  # Enforce monotonicity

#     # Calculate discrete probabilities
#     pb = np.zeros(num_dmg_states)
#     pb[0] = 1.0 - pe[1] if len(pe) > 1 else 1.0
#     for d in range(1, num_dmg_states - 1):
#         pb[d] = max(0.0, pe[d] - pe[d + 1])
#     if num_dmg_states > 1:
#         pb[-1] = max(0.0, pe[-1])

#     # Normalise probabilities
#     pb_sum = np.sum(pb)
#     if pb_sum > 0:
#         pb = pb / pb_sum
#     else:
#         pb[0] = 1.0

#     # Calculate recovery times for each damage state using fast lookups
#     recovery_times = np.zeros(num_dmg_states)

#     for ds_index in range(num_dmg_states):
#         if ds_index == 0:
#             recovery_times[ds_index] = 0.0  # No recovery needed
#         elif (
#             ds_index in comp_obj.damage_states.keys()
#             and getattr(comp_obj.damage_states[ds_index], "recovery_function_discrete", None)
#             is not None
#         ):
#             try:
#                 # Fast discrete lookup
#                 restoration_func = comp_obj.damage_states[ds_index].recovery_function_discrete

#                 # Get baseline functionality for this damage state
#                 damage_state_obj = comp_obj.damage_states[ds_index]
#                 baseline_functionality = getattr(damage_state_obj, "functionality", 0.0)

#                 # Use minimum of current and expected functionality as baseline
#                 effective_baseline = min(functionality, baseline_functionality)
#                 effective_baseline = max(effective_baseline, MIN_FUNCTIONALITY_THRESHOLD)

#                 # Calculate recovery time from baseline to threshold
#                 target_time = restoration_func.get_recovery_time(threshold_recovery)
#                 baseline_time = restoration_func.get_recovery_time(effective_baseline)

#                 recovery_time = max(0.0, target_time - baseline_time)

#                 # Enforce minimum for damaged states
#                 if recovery_time < MIN_RECOVERY_TIME:
#                     recovery_time = MIN_RECOVERY_TIME

#                 recovery_times[ds_index] = recovery_time

#             except Exception as e:
#                 rootLogger.debug(f"Fast lookup failed for {comp_id} DS{ds_index}: {e}")
#                 # Minimal fallback: small positive time for damaged states
#                 recovery_times[ds_index] = max(MIN_RECOVERY_TIME, 0.0)
#         else:
#             # No restoration function available
#             recovery_times[ds_index] = max(MIN_RECOVERY_TIME, 0.0)

#     # HAZUS weighted combination
#     total_recovery_time = np.sum(pb * recovery_times)

#     # Validate result
#     if np.isnan(total_recovery_time) or np.isinf(total_recovery_time) or total_recovery_time < 0:
#         total_recovery_time = 0.0

#     rootLogger.debug(
#         f"[FAST PROBABILISTIC] {comp_id} func={functionality:.3f} "
#         f"-> {total_recovery_time:.1f} days (prob weights: {pb})"
#     )

#     return round(total_recovery_time, 1)


# def calculate_component_recovery_time_rand(
#     comp_id: str,
#     event_id: str,
#     comp_obj: Component,
#     hazards: HazardsContainer,
#     functionality: float,
#     threshold_recovery: float = RESTORATION_THRESHOLD,
#     noise_scale: float = 0.5,
# ) -> int:
#     """
#     Alternate (randomised) component recovery time calculation.

#     Logic:
#     1) Compute the most likely damage state (argmax P[DS=i|H]) and its typical functionality,
#        using exceedance fragilities.
#     2) Select the recovery algorithm for that damage state.
#     3) Compute recovery time from baseline functionality to threshold, then add random noise
#        (which can be positive or negative) to represent uncertainty.
#     4) Enforce a minimum of 1 day if damage state > 0 or functionality < 1.0.
#     5) Return the recovery time as an integer (days).
#     """
#     # Hazard intensity at this component for this event
#     hazard_obj = hazards.listOfhazards[hazards.hazard_scenario_list.index(event_id)]
#     loc_params = comp_obj.get_location()
#     H = hazard_obj.get_hazard_intensity(*loc_params)

#     # Fragility (exceedance) and recovery functions
#     damage_functions = [ds.response_function for ds in comp_obj.damage_states.values()]
#     recovery_functions = [ds.recovery_function for ds in comp_obj.damage_states.values()]
#     dmg_states = list(comp_obj.damage_states.keys())
#     num_dmg_states = len(dmg_states)

#     # Exceedance pe[i] = P(DS >= i | H), enforce bounds and monotonicity
#     pe = np.array([damage_functions[d](H) for d in range(num_dmg_states)], dtype=float)
#     pe = np.clip(pe, 0.0, 1.0)
#     for d in range(1, len(pe)):
#         pe[d] = min(pe[d], pe[d - 1])

#     # Disjoint probabilities pb[i] = P(DS = i | H)
#     pb = np.zeros(num_dmg_states, dtype=float)
#     if num_dmg_states == 1:
#         pb[0] = 1.0
#     else:
#         pb[0] = max(0.0, 1.0 - float(pe[1]))
#         for d in range(1, num_dmg_states - 1):
#             pb[d] = max(0.0, float(pe[d]) - float(pe[d + 1]))
#         pb[-1] = max(0.0, float(pe[-1]))

#     s = float(pb.sum())
#     if s > 0.0:
#         pb /= s
#     else:
#         pb[:] = 0.0
#         pb[0] = 1.0

#     # 1) Most likely damage state and its typical functionality
#     ds_mle = int(np.argmax(pb))
#     ds_func = float(getattr(dmg_states[ds_mle], "functionality", 1.0))

#     # Baseline functionality: be conservative by using the lower of observed and typical
#     baseline_func = min(float(functionality), ds_func)

#     # 2) Select the recovery function for this damage state
#     rf = recovery_functions[ds_mle]

#     # 3) Compute recovery time from baseline to threshold, add random noise
#     def _safe_inv(f, x: float) -> Optional[float]:
#         try:
#             val = f(x, inverse=True)
#             if val is None or not np.isfinite(val):
#                 return None
#             return float(val)
#         except Exception as e:
#             rootLogger.debug(f"[RECOVERY DEBUG] Inverse failed for f({x}): {e}")
#             return None

#     t_thresh = _safe_inv(rf, threshold_recovery)
#     t_base = _safe_inv(rf, max(0.0, min(1.0, baseline_func)))

#     rootLogger.debug(
#         f"[RECOVERY DEBUG] comp_id={comp_id} event_id={event_id} "
#         f"DS={ds_mle} func={functionality:.3f} "
#         f"baseline_func={baseline_func:.3f} t_thresh={t_thresh} t_base={t_base} "
#         f"rf={getattr(rf, '__name__', str(rf))}"
#     )

#     if t_thresh is not None and t_base is not None:
#         rec_time = max(0.0, t_thresh - t_base)
#         rootLogger.debug(f"[RECOVERY DEBUG] Using recovery function: rec_time={rec_time}")
#     else:
#         # Fallback engineering estimate if inverse not available
#         rec_time = max(1.0 if (ds_mle > 0 or baseline_func < 1.0) else 0.0, ds_mle * 7.0)
#         rootLogger.debug(f"[RECOVERY DEBUG] Fallback used: rec_time={rec_time}")

#     # Add random noise (can be positive or negative)
#     noise = float(np.random.normal(0.0, noise_scale))
#     rec_time += noise
#     rootLogger.debug(f"[RECOVERY DEBUG] Added noise: noise={noise:.3f} rec_time+noise={rec_time}")

#     # 4) Enforce minimum of 1 day if damaged or not fully functional
#     needs_repair = (ds_mle > 0) or (baseline_func < 1.0)
#     rec_time = max(1.0 if needs_repair else 0.0, rec_time)
#     rootLogger.debug(f"[RECOVERY DEBUG] Final rec_time after min check: {rec_time}")

#     # 5) Return integer days
#     return int(round(rec_time))


def calculate_constrained_recovery(recovery_times: List[float], num_streams: int) -> float:
    """
    Calculate recovery time with simulated resource constraints.
    Uses load balancing algorithm across parallel repair streams.
    """
    if not recovery_times:
        return 0.0

    # Sort components by recovery time (longest first)
    sorted_times = sorted(recovery_times, reverse=True)

    # Initialise streams
    n_streams = min(num_streams, len(sorted_times))
    streams = [0.0] * n_streams

    # Assign components to streams (load balancing)
    for recovery_time in sorted_times:
        # Find stream with minimum total time
        min_idx = streams.index(min(streams))
        streams[min_idx] += recovery_time

    # Total time is when last stream finishes
    return max(streams)


def process_event_chunk(
    event_ids: List[str],
    config: Configuration,
    hazards: HazardsContainer,
    components_costed: List[str],
    infrastructure: Any,
    recovery_method: str,
    num_repair_streams: int,
) -> List[float]:
    """
    Process a chunk of events for multiprocessing.
    """
    results = []
    for event_id in event_ids:
        try:
            # # Get event data
            # if event_id in component_response_dict:
            #     event_response_components = pd.Series(component_response_dict[event_id])
            # elif str(event_id) in component_response_dict:
            #     event_response_components = pd.Series(component_response_dict[str(event_id)])
            # else:
            #     rootLogger.warning(f"Event {event_id} not found in component response data")
            #     results.append(0.0)
            #     continue

            hazard_obj = hazards.listOfhazards[hazards.hazard_scenario_list.index(event_id)]
            recovery_time = calculate_event_recovery(
                config,
                event_id,
                hazard_obj,
                components_costed,
                infrastructure,
                recovery_method,
                num_repair_streams,
            )
            results.append(recovery_time)

        except Exception as e:
            rootLogger.error(f"Error processing event {event_id}: {e}")
            results.append(0.0)

    return results


def parallel_recovery_analysis(
    config: Configuration,
    hazards: HazardsContainer,
    components: List[Component],
    infrastructure: Any,
    scenario: Any,
    components_costed: List[str],
    recovery_method: str = "max",
    num_repair_streams: int = 1000,
    max_workers: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> List[float]:
    """
    Optimised parallel recovery analysis with automatic backend selection.

    This function automatically selects the best available backend:
    - MPI4Py for HPC environments
    - Multiprocessing for single nodes
    """
    # Check if parallel config exists in scenario
    backend = "auto"
    if hasattr(scenario, "parallel_config"):
        backend = scenario.parallel_config.config.get("backend", "auto")
        # Coerce legacy 'dask' to 'multiprocessing'
        if backend == "dask":
            backend = "multiprocessing"

        # Override max_workers if specified in config
        if max_workers is None:
            if backend == "multiprocessing":
                max_workers = scenario.parallel_config.config.get("mp_n_processes")

    # Create analysis engine
    engine = RecoveryAnalysisEngine(backend=backend)

    try:
        # Run analysis
        recovery_times = engine.analyse(
            config,
            hazards,
            infrastructure,
            scenario,
            components_costed,
            recovery_method,
            num_repair_streams,
        )

        # Validate results
        if recovery_times:
            non_zero = sum(1 for t in recovery_times if t > 0)
            rootLogger.info(
                f"Analysis complete: {non_zero}/{len(recovery_times)} "
                f"events have non-zero recovery times"
            )

            if non_zero == 0:
                rootLogger.warning(
                    f"{Fore.RED}*** All recovery times are zero. "
                    f"This may indicate a data issue. ***{Fore.RESET}\n"
                )

        return recovery_times

    finally:
        engine.cleanup()


# =======================================================================================


def sequential_recovery_analysis(
    hazard_event_list,
    infrastructure,
    config,
    hazards,
    components_costed,
):
    """
    Sequential processing fallback for recovery analysis
    """
    rootLogger.info("Processing events sequentially...")
    recovery_times = []

    # Use smaller chunks even in sequential mode to minimize memory usage
    chunk_size = 1000
    num_chunks = math.ceil(len(hazard_event_list) / chunk_size)

    restoration_df = pd.DataFrame(index=hazard_event_list, columns=components_costed)
    restoration_df.index.name = "event_id"

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(hazard_event_list))
        chunk_events = hazard_event_list[start_idx:end_idx]

        chunk_pbar = tqdm(
            chunk_events,
            desc=f"Processing chunk {chunk_idx + 1}/{num_chunks}",
            leave=False,
        )

        chunk_results = []
        for event_id in chunk_pbar:
            try:
                hazard_obj = hazards.listOfhazards[hazards.hazard_scenario_list.index(event_id)]
                result = calculate_event_recovery(
                    config,
                    str(event_id),
                    hazard_obj,
                    components_costed,
                    infrastructure,
                )
                # Extract only the recovery time
                if result is not None:
                    recovery_time = result
                else:
                    recovery_time = 0

                chunk_results.append(recovery_time)

            except (
                KeyError,
                ValueError,
                AttributeError,
                RuntimeError,
                IndexError,
            ) as e:
                rootLogger.warning(f"Error processing event {event_id}: {str(e)}")
                chunk_results.append(0)  # Default recovery time

        # chunk_results = []
        # for event_id in chunk_pbar:
        #     try:
        #         hazard_obj = hazards.listOfhazards[hazards.hazard_scenario_list.index(event_id)]
        #         result = loss_analysis.analyse_system_recovery(
        #             infrastructure,
        #             config,
        #             hazard_obj,
        #             str(event_id),
        #             components_costed,
        #             verbosity=False,
        #         )

        #         # Extract only the recovery time
        #         if result is not None and (
        #             (isinstance(result, pd.DataFrame) and
        #              "Full Restoration Time" in result.columns)
        #             or ("Full Restoration Time" in result)
        #         ):
        #             recovery_time = result["Full Restoration Time"].max()
        #         else:
        #             recovery_time = 0

        #         chunk_results.append(recovery_time)
        #         # print(result["Full Restoration Time"].values.tolist())
        #         restoration_df.loc[event_id] = result["Full Restoration Time"].values.tolist()
        #     except (
        #         KeyError,
        #         ValueError,
        #         AttributeError,
        #         RuntimeError,
        #         IndexError,
        #     ) as e:
        #         rootLogger.warning(f"Error processing event {event_id}: {str(e)}")
        #         chunk_results.append(0)  # Default recovery time

        # # DEBUG PRINTS
        # print("############################################################")
        # bad_rst_comps = sorted(check_non_monotonic_cols(restoration_df))
        # bad_rst_comptypes = sorted(
        #     list(set([infrastructure.components[n].component_type for n in bad_rst_comps]))
        # )
        # print(f"Problematic components:\n{bad_rst_comps}")
        # print(f"Problematic component types:\n{bad_rst_comptypes}")
        # print("############################################################")
        # restoration_df.to_csv(Path(config.OUTPUT_DIR, "component_restoration_times.csv"))

        # Extend recovery times with chunk results
        recovery_times.extend(chunk_results)

        # Save checkpoint after each chunk
        checkpoint_file = Path(config.OUTPUT_DIR, "recovery_checkpoint_sequential.pkl")
        with open(checkpoint_file, "wb") as f:
            pickle.dump(recovery_times, f)
        rootLogger.info(
            f"Checkpoint saved: {len(recovery_times)}/{len(hazard_event_list)} events processed"
        )

    return recovery_times


def check_non_monotonic_cols(df: pd.DataFrame) -> List[str]:
    """
    Check for non-monotonic columns in a DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check

    Returns
    -------
    list
        List of column names that are non-monotonic
    """
    non_monotonic_cols = [col for col in df.columns if not df[col].is_monotonic_increasing]
    return non_monotonic_cols
