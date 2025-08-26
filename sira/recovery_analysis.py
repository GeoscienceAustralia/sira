"""
Optimised recovery analysis module using MPI4Py for HPC scaling and Dask for data processing.
Fixes zero recovery time issues and provides significant performance improvements.
"""

import numpy as np
import scipy.stats as stats
import pandas as pd
import logging
from typing import List, Optional, Any, Tuple, Dict, Union
import os
from pathlib import Path
from colorama import Fore

# Progress tracking
from sympy import Li
from torch import inverse
from tqdm import tqdm

from sira.modelling.component import Component
from sira.modelling.hazard import HazardsContainer

# Configure logging
logger = logging.getLogger(__name__)

# Constants for recovery calculation
DEFAULT_RECOVERY_TIME_PER_DS = 168  # hours (1 week) per damage state
RESTORATION_THRESHOLD = 0.95
MIN_FUNCTIONALITY_FOR_DAMAGE = 0.98  # Below this, component is considered damaged
MIN_FUNCTIONALITY_THRESHOLD = 0.03   # Lowest functionality threshold considered
MIN_RECOVERY_TIME = 24               # Set the minimum recovery time for damaged components

def is_mpi_environment():
    """
    Detect if we're in a proper MPI environment.

    Returns
    -------
    bool
        True if in proper MPI environment, False otherwise
    """
    # Check for explicit disabling
    if os.environ.get('SIRA_FORCE_NO_MPI', '0') == '1':
        return False

    # Check for SLURM (most common HPC scheduler)
    slurm_vars = ['SLURM_JOB_ID', 'SLURM_NTASKS', 'SLURM_PROCID', 'SLURM_NODELIST']
    if any(var in os.environ for var in slurm_vars):
        return True

    # Check for PBS/Torque (another common HPC scheduler)
    pbs_vars = ['PBS_JOBID', 'PBS_NCPUS', 'PBS_NODEFILE']
    if any(var in os.environ for var in pbs_vars):
        return True

    # Check for explicit MPI runtime variables
    mpi_runtime_vars = [
        'OMPI_COMM_WORLD_SIZE', 'OMPI_COMM_WORLD_RANK',  # OpenMPI
        'PMI_SIZE', 'PMI_RANK',                          # MPICH
        'MPI_LOCALRANKID', 'MPI_LOCALNRANKS'             # Intel MPI
    ]
    if any(var in os.environ for var in mpi_runtime_vars):
        return True

    # Check if we're being launched with mpirun/mpiexec
    parent_process = os.environ.get('_', '')
    if any(launcher in parent_process for launcher in ['mpirun', 'mpiexec', 'srun']):
        return True

    # Check for HPC-specific hostnames or environments
    hostname = os.environ.get('HOSTNAME', '')
    if any(pattern in hostname.lower() for pattern in ['hpc', 'cluster', 'node', 'compute']):
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
        os.environ['MPI4PY_RC_INITIALISE'] = 'False'

        from mpi4py import MPI

        # Check if already initialised
        if not MPI.Is_initialized():
            MPI.Init()

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        logger.info(f"MPI successfully initialised: rank {rank} of {size}")
        return MPI, comm, rank, size

    except ImportError:
        logger.warning("MPI4Py not available even in MPI environment")
        return None, None, 0, 1
    except Exception as e:
        logger.warning(f"MPI initialisation failed: {e}")
        return None, None, 0, 1


class RecoveryAnalysisEngine:
    """
    Main engine for parallel recovery analysis supporting multiple backends:
    - MPI4Py for HPC environments
    - Dask for distributed computing
    - Multiprocessing for single-node systems
    """

    def __init__(self, backend='auto'):
        """
        Initialise recovery analysis engine.

        Parameters
        ----------
        backend : str
            'mpi', 'dask', 'multiprocessing', or 'auto' (detects best available)
        """
        self.backend = self._select_backend(backend)
        self.comm = None
        self.rank = 0
        self.size = 1
        self.dask_client = None
        self.MPI = None

        if self.backend == 'mpi':
            self.MPI, self.comm, self.rank, self.size = safe_mpi_import()
            if self.MPI is None:
                logger.warning("MPI backend requested but not available. Falling back to Dask.")
                self.backend = 'dask'

    def _select_backend(self, requested_backend):
        """Select the best available backend."""
        if requested_backend == 'auto':
            # Check for MPI environment first
            if is_mpi_environment():
                return 'mpi'
            # Check for Dask scheduler
            elif 'DASK_SCHEDULER_ADDRESS' in os.environ:
                return 'dask'
            else:
                return 'multiprocessing'
        return requested_backend

    def setup_dask_client(self, n_workers=None, threads_per_worker=1, memory_limit='4GB'):
        """Setup Dask client for distributed computing."""
        if self.backend == 'dask' and self.dask_client is None:
            try:
                # Try to connect to existing scheduler
                scheduler_address = os.environ.get('DASK_SCHEDULER_ADDRESS')
                if scheduler_address:
                    from dask.distributed import Client
                    self.dask_client = Client(scheduler_address)
                else:
                    # Create local cluster
                    from dask.distributed import LocalCluster, Client
                    if n_workers is None:
                        cpu_count = os.cpu_count() if os.cpu_count() is not None else 1
                        n_workers = min(cpu_count, 8)  # type: ignore
                    cluster = LocalCluster(
                        n_workers=n_workers,
                        threads_per_worker=threads_per_worker,
                        memory_limit=memory_limit
                    )
                    self.dask_client = Client(cluster)
                logger.info(f"Dask client initialised: {self.dask_client}")
            except Exception as e:
                logger.warning(f"Failed to setup Dask client: {e}. Falling back to multiprocessing.")
                self.backend = 'multiprocessing'

    def analyze(
        self,
        hazards: HazardsContainer,
        infrastructure: Any,
        scenario: Any,
        component_resp_df: pd.DataFrame,
        components_costed: List[str],
        recovery_method: str = 'max',
        num_repair_streams: int = 100
    ) -> List[float]:
        """
        Main analysis function that routes to appropriate backend.
        """
        hazard_event_list = hazards.hazard_scenario_list
        logger.info(f"Starting recovery analysis with backend: {self.backend}")
        logger.info(f"Processing {len(hazard_event_list)} events")

        if self.backend == 'mpi':
            return self._analyze_mpi(
                hazards, hazard_event_list, infrastructure, scenario,
                component_resp_df, components_costed,
                recovery_method, num_repair_streams
            )
        elif self.backend == 'dask':
            return self._analyze_dask(
                hazards, hazard_event_list, infrastructure, scenario,
                component_resp_df, components_costed,
                recovery_method, num_repair_streams
            )
        else:
            return self._analyze_multiprocessing(
                hazards, hazard_event_list, infrastructure, scenario,
                component_resp_df, components_costed,
                recovery_method, num_repair_streams
            )

    def _analyze_mpi(
        self,
        hazards: HazardsContainer,
        hazard_event_list: List[str],
        infrastructure: Any,
        scenario: Any,
        component_resp_df: pd.DataFrame,
        components_costed: List[str],
        recovery_method: str,
        num_repair_streams: int
    ) -> List[float]:
        """MPI4Py implementation for HPC environments."""
        if self.MPI is None or self.comm is None:
            logger.error("MPI not available. Falling back to multiprocessing.")
            self.backend = 'multiprocessing'
            return self._analyze_multiprocessing(
                hazards, hazard_event_list, infrastructure, scenario,
                component_resp_df, components_costed,
                recovery_method, num_repair_streams
            )

        # Broadcast infrastructure data to all ranks
        if self.rank == 0:
            infra_data = extract_infrastructure_data(infrastructure)
            # Convert DataFrame to dict for MPI broadcast
            resp_data = component_resp_df.to_dict('index')
        else:
            infra_data = None
            resp_data = None

        infra_data = self.comm.bcast(infra_data, root=0)
        resp_data = self.comm.bcast(resp_data, root=0)

        # Scatter events across ranks
        if self.rank == 0:
            # Divide events into chunks for each rank
            chunks = np.array_split(hazard_event_list, self.size)
            chunks = [chunk.tolist() for chunk in chunks]
        else:
            chunks = None

        local_events = self.comm.scatter(chunks, root=0)

        # Process local events
        local_results = []
        for event_id in local_events:
            try:
                # Get event data from broadcasted dict
                if event_id in resp_data:
                    event_response_components = pd.Series(resp_data[event_id])
                else:
                    # Try string conversion
                    event_id_str = str(event_id)
                    if event_id_str in resp_data:
                        event_response_components = pd.Series(resp_data[event_id_str])
                    else:
                        logger.warning(f"Event {event_id} not found in data")
                        local_results.append(0.0)
                        continue

                recovery_time = calculate_event_recovery_robust(
                    event_id, hazards, event_response_components, components_costed,
                    infrastructure, infra_data,
                    recovery_method, num_repair_streams
                )
                local_results.append(recovery_time)

            except Exception as e:
                logger.error(f"Error processing event {event_id}: {e}")
                local_results.append(0.0)

        # Gather results at rank 0
        all_results = self.comm.gather(local_results, root=0)

        if self.rank == 0:
            # Flatten results maintaining order
            if chunks is None:
                return []
            recovery_times = []
            for i, chunk in enumerate(chunks):
                for j, event_id in enumerate(chunk):
                    recovery_times.append(all_results[i][j])  # type: ignore
            return recovery_times
        else:
            return []

    def _analyze_dask(
        self,
        hazards: HazardsContainer,
        hazard_event_list: List[str],
        infrastructure: Any,
        scenario: Any,
        component_resp_df: pd.DataFrame,
        components_costed: List[str],
        recovery_method: str,
        num_repair_streams: int
    ) -> List[float]:
        """Dask implementation for distributed computing."""
        # Setup Dask client if not already done
        if self.dask_client is None:
            self.setup_dask_client()

        if self.dask_client is None:
            logger.warning("Dask client not available. Falling back to multiprocessing.")
            self.backend = 'multiprocessing'
            return self._analyze_multiprocessing(
                hazards, hazard_event_list, infrastructure, scenario,
                component_resp_df, components_costed,
                recovery_method, num_repair_streams
            )

        # Convert to Dask DataFrame for efficient processing
        logger.info("Converting to Dask DataFrame...")

        # Ensure index is string type for consistent lookups
        component_resp_df.index = component_resp_df.index.astype(str)

        # Calculate optimal partitions based on data size
        max_numcores = max(self.dask_client.ncores().values())
        n_partitions = max(1, min(
            len(component_resp_df) // 1000,  # ~1000 rows per partition
            max_numcores * 4    # 4 partitions per core
        ))

        import dask.dataframe as dd
        from dask.delayed import delayed
        from dask.distributed import as_completed

        ddf = dd.from_pandas(component_resp_df, npartitions=n_partitions)

        # Extract infrastructure data once
        infra_data = extract_infrastructure_data(infrastructure)

        # Create delayed tasks for each event
        logger.info("Creating Dask computation graph...")

        @delayed
        def process_event_delayed(event_id, infra_data):
            try:
                # Get event data from Dask DataFrame
                event_df = ddf.loc[str(event_id)].compute()
                if isinstance(event_df, pd.DataFrame):
                    event_data = event_df.iloc[0]
                else:
                    event_data = event_df

                return calculate_event_recovery_robust(
                    event_id, hazards, event_data, components_costed,
                    infrastructure, infra_data,
                    recovery_method, num_repair_streams
                )
            except Exception as e:
                logger.error(f"Error processing event {event_id}: {e}")
                return 0.0

        # Create delayed tasks
        tasks = [
            process_event_delayed(event_id, infra_data)
            for event_id in hazard_event_list]

        # Compute with progress bar
        logger.info("Computing recovery times with Dask...")
        futures = self.dask_client.compute(tasks)

        # Create mapping for efficient lookup (ensuring futures is a list)
        future_to_idx = {future: i for i, future in enumerate(futures)}  # type: ignore

        # Process results as they complete
        recovery_times = [None] * len(hazard_event_list)

        with tqdm(total=len(futures), desc="Processing events") as pbar:  # type: ignore
            for future, result in as_completed(futures, with_results=True):
                idx = future_to_idx[future]  # O(1) lookup instead of O(n)
                recovery_times[idx] = result
                pbar.update(1)

        return recovery_times   # type: ignore

    def _analyze_multiprocessing(
        self,
        hazards: HazardsContainer,
        hazard_event_list: List[str],
        infrastructure: Any,
        scenario: Any,
        component_resp_df: pd.DataFrame,
        components_costed: List[str],
        recovery_method: str,
        num_repair_streams: int
    ) -> List[float]:

        """Multiprocessing implementation for single-node systems."""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        hazard_event_list = hazards.hazard_scenario_list

        # Extract infrastructure data once
        infra_data = extract_infrastructure_data(infrastructure)
        components = infrastructure.components

        # Ensure consistent index type
        component_resp_df.index = component_resp_df.index.astype(str)

        # Determine number of workers
        n_workers = min(os.cpu_count() or 1, len(hazard_event_list))

        logger.info(f"Using multiprocessing with {n_workers} workers")

        # Create chunks of events
        chunk_size = max(1, len(hazard_event_list) // (n_workers * 4))
        event_chunks = [
            hazard_event_list[i:i + chunk_size]
            for i in range(0, len(hazard_event_list), chunk_size)
        ]

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # # DEBUG PRINT
        # print("-" * 50, "\n")
        # print(f"Event chunks created: {event_chunks}")
        # print(f"component_resp_df:\n{component_resp_df}")
        # print("x" * 50, "\n")
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Process chunks in parallel
        recovery_times = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(
                    process_event_chunk,
                    chunk,
                    hazards,
                    component_resp_df.to_dict('index'),
                    components_costed,
                    infrastructure,
                    infra_data,
                    recovery_method,
                    num_repair_streams
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
                        logger.error(f"Error processing chunk: {e}")
                        chunk_results[chunk_idx] = [0.0] * len(chunk)
                        pbar.update(1)

            # Combine results in order
            for i in range(len(event_chunks)):
                recovery_times.extend(chunk_results[i])

        return recovery_times

    def cleanup(self):
        """Cleanup resources."""
        if self.dask_client:
            self.dask_client.close()
        # MPI finalisation is handled automatically by mpi4py


def extract_infrastructure_data(infrastructure: Any) -> Dict:
    """
    Extract essential infrastructure data for recovery calculation.
    Includes proper recovery function data.
    """
    data = {
        'components': {},
        'system_class': getattr(infrastructure, 'system_class', 'unknown'),
        'output_capacity': getattr(infrastructure, 'system_output_capacity', 1.0)
    }

    for comp_id, component_obj in infrastructure.components.items():
        comp_data = {
            'component_type': component_obj.component_type,
            'component_class': component_obj.component_class,
            'cost_fraction': getattr(component_obj, 'cost_fraction'),
            'damage_states': {}
        }

        # Extract damage state information
        for ds_idx, damage_state in component_obj.damage_states.items():
            ds_recovery_config = {
                'damage_ratio': getattr(damage_state, 'damage_ratio'),
                'functionality': getattr(damage_state, 'functionality'),
                'recovery_function': getattr(damage_state, 'recovery_function')  # yields RecoveryFunction object
            }

            # Check for recovery function
            if hasattr(damage_state, 'recovery_function_constructor'):
                ds_recovery_config['recovery_function_constructor'] = \
                    damage_state.recovery_function_constructor
                # ds_recovery_config['recovery_function_name'] = getattr(
                #     damage_state, 'recovery_function_constructor').get('function_name')
                # ds_recovery_config['recovery_mean'] = getattr(
                #     damage_state, 'recovery_function_constructor').get('mean')
                # ds_recovery_config['recovery_std'] = getattr(
                #     damage_state, 'recovery_function_constructor').get('stddev')

            comp_data['damage_states'][ds_idx] = ds_recovery_config

        data['components'][comp_id] = comp_data

    return data


def calculate_event_recovery_robust(
    event_id: str,
    hazards: HazardsContainer,
    event_response_components: pd.Series,
    components_costed: List[str],
    infrastructure: Any,
    infra_data: Dict,
    recovery_method: str,
    num_repair_streams: int
) -> float:
    """
    Calculate recovery time for a single event with robust error handling.
    Fixes the zero recovery time issue.
    """
    component_recovery_times = []
    damaged_components = 0

    # comp_obj = infrastructure.components[comp_name]
    # hazard_obj = hazards.listOfhazards[
    #     hazards.hazard_scenario_list.index(event_id)]
    # loc_params = comp_obj.get_location()
    # sc_haz_val = hazard_obj.get_hazard_intensity(*loc_params)

    for comp_id in components_costed:
        if comp_id not in infra_data['components']:
            continue

        # Extract component state
        damage_state, functionality = extract_component_state_robust(
            event_response_components, comp_id
        )

        if damage_state > 0 or functionality < MIN_FUNCTIONALITY_FOR_DAMAGE:
            damaged_components += 1

            # Calculate recovery time
            # recovery_time = calculate_component_recovery_robust(
            #     comp_id, damage_state, functionality, infra_data
            # )

            comp_obj = infrastructure.components[comp_id]

            recovery_time = calculate_component_recovery_time(
                comp_id, event_id, comp_obj, hazards, functionality
            )

            if recovery_time > 0:
                component_recovery_times.append(recovery_time)

    # Log if we found damaged components but no recovery times
    if damaged_components > 0 and len(component_recovery_times) == 0:
        logger.warning(
            f"Found {damaged_components} damaged components but no recovery times calculated"
        )

    if not component_recovery_times:
        return 0.0

    # Apply recovery method
    if recovery_method == 'parallel_streams':
        return calculate_parallel_recovery(component_recovery_times, num_repair_streams)
    else:  # max method
        return max(component_recovery_times)


def extract_component_state_robust(
    event_response_components: pd.Series,
    comp_id: str
) -> Tuple[int, float]:
    """
    Robustly extract damage state and functionality from event data.
    Handles various column naming conventions.
    """
    damage_state = 0
    functionality = 1.0

    # Check if columns are MultiIndex
    if isinstance(event_response_components.index, pd.MultiIndex):
        # Try standard MultiIndex access
        try:
            damage_state = int(event_response_components.get((comp_id, 'damage_index')))  # type: ignore
            functionality = float(event_response_components.get((comp_id, 'func_mean')))  # type: ignore
        except (KeyError, ValueError):
            # Try alternative names
            for idx in event_response_components.index:
                if idx[0] == comp_id:
                    if 'damage' in str(idx[1]).lower():
                        try:
                            damage_state = int(event_response_components[idx])
                        except (ValueError, TypeError):
                            pass
                    elif 'func' in str(idx[1]).lower():
                        try:
                            functionality = float(event_response_components[idx])
                        except (ValueError, TypeError):
                            pass
    else:
        # Handle flat index with various naming patterns
        patterns = {
            'damage': [
                f"{comp_id}_damage_state",
                f"{comp_id}.damage_state",
                f"{comp_id}_dmg_state",
                f"damage_state_{comp_id}",
                f"{comp_id}_ds"
            ],
            'func': [
                f"{comp_id}_func_mean",
                f"{comp_id}.func_mean",
                f"{comp_id}_functionality",
                f"func_mean_{comp_id}",
                f"{comp_id}_func"
            ]
        }

        # Try damage state patterns
        for pattern in patterns['damage']:
            if pattern in event_response_components.index:
                try:
                    val = event_response_components[pattern]
                    if not pd.isna(val):
                        damage_state = int(float(val))
                        break
                except Exception:
                    continue

        # Try functionality patterns
        for pattern in patterns['func']:
            if pattern in event_response_components.index:
                try:
                    val = event_response_components[pattern]
                    if not pd.isna(val):
                        functionality = float(val)
                        break
                except Exception:
                    continue

    return damage_state, functionality


def calculate_component_recovery_time(
    comp_id: str,
    event_id: str,
    comp_obj: Component,
    hazards: HazardsContainer,
    functionality: float,
    threshold_recovery=RESTORATION_THRESHOLD,  # recovery_threshold: float = RESTORATION_THRESHOLD,
    min_functionality_threshold: float = MIN_FUNCTIONALITY_THRESHOLD,
) -> float:
    """
    Calculates the time needed for a component to reach the recovery threshold
    """

    if functionality >= threshold_recovery:
        return 0.0

    # Component.damage_states is a dict of `DamageState` objects
    damage_functions = [
        ds.response_function for ds in comp_obj.damage_states.values()]  # type: ignore
    recovery_functions = [
        ds.recovery_function for ds in comp_obj.damage_states.values()]  # type: ignore

    comp_functionality = functionality
    comptype_dmg_states = comp_obj.damage_states
    num_dmg_states = len(comp_obj.damage_states)  # type: ignore

    hazard_obj = hazards.listOfhazards[hazards.hazard_scenario_list.index(event_id)]
    loc_params = comp_obj.get_location()
    sc_haz_val = hazard_obj.get_hazard_intensity(*loc_params)

    # Calculate damage exceedance probabilities
    pe = np.array([damage_functions[d](sc_haz_val) for d in range(num_dmg_states)])

    # Calculate probability of being in each damage state
    pb = np.zeros(num_dmg_states)
    pb[0] = 1.0 - pe[1]  # Probability of no damage
    for d in range(1, num_dmg_states - 1):
        pb[d] = pe[d] - pe[d + 1]
    pb[-1] = pe[-1]  # Probability of being in worst damage state

    reqtime = np.zeros(num_dmg_states)

    for d, ds in enumerate(comptype_dmg_states):
        if (
            ds == 'DS0 None'
            or d == 0
            or pb[d] < min_functionality_threshold
        ):
            reqtime[d] = 0.00
        else:
            try:
                # Calculate time difference for recovery
                recovery_time = recovery_functions[d](threshold_recovery, inverse=True)\
                    - recovery_functions[d](comp_functionality, inverse=True)
                # Ensure we don't get negative or infinite times
                reqtime[d] = max(0.0, recovery_time) if not np.isinf(recovery_time) else 0.0
            except (ValueError, TypeError, ZeroDivisionError, RuntimeError) as e:
                logger.warning(
                    f"Error calculating recovery time for component {comp_obj.component_id} "
                    f"in damage state {ds}: {str(e)}")
                reqtime[d] = 0.00

    restoration_time_agg = round(sum(pb * reqtime), 1)

    if (
        np.isnan(restoration_time_agg)
        or np.isinf(restoration_time_agg)
        or restoration_time_agg < 0
    ):
        restoration_time_agg = 0.0

    return restoration_time_agg


def calculate_component_recovery_robust(
    comp_id: str,
    damage_state: int,
    functionality: float,
    infra_data: Dict,
    recovery_threshold: float = RESTORATION_THRESHOLD,
    min_functionality: float = MIN_FUNCTIONALITY_THRESHOLD,
    min_recovery_time: float = 1.0  # Minimum recovery time in days
) -> float:
    """
    Calculate recovery time with proper handling of damage states.
    Fixes zero recovery time issue.
    """

    if damage_state == 0 or functionality >= MIN_FUNCTIONALITY_FOR_DAMAGE:
        return 0.0

    # if functionality < min_functionality, set to min threshold
    fn_lo = max(min_functionality, functionality)

    comp_data = infra_data['components'].get(comp_id, {})
    damage_states = comp_data.get('damage_states', {})

    # Get recovery time from damage state data
    recovery_time = 0.0
    ds_data = damage_states[damage_state]

    recovery_func = ds_data['recovery_function']
    # ---------------------------------------------------------------
    # Or explicitly construct the recovery function using the params
    # ---------------------------------------------------------------
    # from sira.modelling.responsemodels import Algorithm
    # response_params = dict(
    #     function_name=ds_data['recovery_function'],
    #     mean=ds_data.get('recovery_mean'),
    #     stddev=ds_data.get('recovery_std')
    # )
    # # recovery_time = mean + 2 * std
    # recovery_func = Algorithm.factory(response_params)
    # ---------------------------------------------------------------

    recovery_time_th = recovery_func(recovery_threshold, inverse=True)
    recovery_time_t0 = recovery_func(fn_lo, inverse=True)  # type: ignore
    recovery_time = recovery_time_th - recovery_time_t0
    logger.debug(f"Recovery time for {comp_id} (DS {damage_state}): {recovery_time} days")

    # Checkpoint to allow minimum recovery time for damaged components
    if damage_state > 0 and recovery_time < min_recovery_time:  # Minimum 24 hours
        recovery_time = min_recovery_time

    recovery_time = np.round(recovery_time, 1)

    return max(recovery_time, 0.0)  # Ensures non-negative recovery time


def calculate_parallel_recovery(
    recovery_times: List[float],
    num_streams: int
) -> float:
    """
    Calculate recovery time with parallel repair streams.
    Uses load balancing algorithm.
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
    hazards: HazardsContainer,
    component_response_dict: Dict,
    components_costed: List[str],
    infrastructure: Any,
    infra_data: Dict,
    recovery_method: str,
    num_repair_streams: int
) -> List[float]:
    """
    Process a chunk of events for multiprocessing.
    """
    results = []
    for event_id in event_ids:
        try:
            # Get event data
            if event_id in component_response_dict:
                event_response_components = pd.Series(component_response_dict[event_id])
            elif str(event_id) in component_response_dict:
                event_response_components = pd.Series(component_response_dict[str(event_id)])

            recovery_time = calculate_event_recovery_robust(
                event_id, hazards, event_response_components, components_costed,
                infrastructure, infra_data,
                recovery_method, num_repair_streams
            )
            results.append(recovery_time)

        except Exception as e:
            logger.error(f"Error processing event {event_id}: {e}")
            results.append(0.0)

    return results


# Main entry point function for backward compatibility
def parallel_recovery_analysis(
    hazards: HazardsContainer,
    components: List[Component],
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
    Optimised parallel recovery analysis with automatic backend selection.

    This function automatically selects the best available backend:
    - MPI4Py for HPC environments
    - Dask for distributed computing
    - Multiprocessing for single nodes
    """
    # Check if parallel config exists in scenario
    backend = 'auto'
    if hasattr(scenario, 'parallel_config'):
        backend = scenario.parallel_config.config.get('backend', 'auto')

        # Override max_workers if specified in config
        if max_workers is None:
            if backend == 'dask':
                max_workers = scenario.parallel_config.config.get('dask_n_workers')
            elif backend == 'multiprocessing':
                max_workers = scenario.parallel_config.config.get('mp_n_processes')

    # Create analysis engine
    engine = RecoveryAnalysisEngine(backend=backend)

    # Use existing Dask client if available
    if (
        backend in ['auto', 'dask'] and\
        hasattr(scenario, 'parallel_backend_data') and\
        'dask_client' in scenario.parallel_backend_data
    ):
        engine.dask_client = scenario.parallel_backend_data['dask_client']
        logger.info("Using existing Dask client from scenario")

    try:
        # Run analysis
        recovery_times = engine.analyze(
            hazards,
            infrastructure,
            scenario,
            component_resp_df,
            components_costed,
            recovery_method,
            num_repair_streams
        )

        # Validate results
        if recovery_times:
            non_zero = sum(1 for t in recovery_times if t > 0)
            logger.info(
                f"Analysis complete: {non_zero}/{len(recovery_times)} "
                f"events have non-zero recovery times"
            )

            if non_zero == 0:
                logger.warning(
                    f"{Fore.RED}*** All recovery times are zero. This may indicate a data issue. ***{Fore.RESET}\n"
                )

        return recovery_times

    finally:
        # Only cleanup if we created our own client
        if not (
            hasattr(scenario, 'parallel_backend_data') and\
            'dask_client' in scenario.parallel_backend_data
        ):
            engine.cleanup()


# Additional utility functions for debugging
def validate_component_response_data(
    component_resp_df: pd.DataFrame,
    components_costed: List[str],
    sample_size: int = 5
) -> Dict[str, Any]:
    """
    Validate component response DataFrame structure and content.
    Useful for debugging zero recovery times.
    """
    validation_report = {
        'shape': component_resp_df.shape,
        'index_type': type(component_resp_df.index).__name__,
        'columns_type': type(component_resp_df.columns).__name__,
        'components_found': 0,
        'damage_states_found': 0,
        'functionality_found': 0,
        'sample_damage_values': [],
        'issues': []
    }

    # Check column structure
    if isinstance(component_resp_df.columns, pd.MultiIndex):
        level0_values = component_resp_df.columns.get_level_values(0).unique()
        level1_values = component_resp_df.columns.get_level_values(1).unique()

        validation_report['multiindex_levels'] = {
            'level0_sample': list(level0_values)[:10],
            'level1_values': list(level1_values)
        }

        # Check for components
        for comp_id in components_costed[:sample_size]:
            if comp_id in level0_values:
                validation_report['components_found'] += 1

                # Check for damage state values
                if (comp_id, 'damage_state') in component_resp_df.columns:
                    sample_values = component_resp_df[(comp_id, 'damage_state')].dropna().head(5).tolist()
                    validation_report['sample_damage_values'].extend(sample_values)

                    # Check for non-zero damage states
                    non_zero = (component_resp_df[(comp_id, 'damage_state')] > 0).sum()
                    if non_zero > 0:
                        validation_report['damage_states_found'] += non_zero
    else:
        # Flat columns
        validation_report['sample_columns'] = list(component_resp_df.columns)[:20]

        # Look for damage state columns
        damage_cols = [col for col in component_resp_df.columns if 'damage' in str(col).lower()]
        func_cols = [col for col in component_resp_df.columns if 'func' in str(col).lower()]

        validation_report['damage_columns_found'] = len(damage_cols)
        validation_report['functionality_columns_found'] = len(func_cols)

        if damage_cols:
            for col in damage_cols[:sample_size]:
                sample_values = component_resp_df[col].dropna().head(5).tolist()
                validation_report['sample_damage_values'].extend(sample_values)

    # Check for potential issues
    if validation_report['components_found'] == 0:
        validation_report['issues'].append("No costed components found in DataFrame columns")

    if validation_report['damage_states_found'] == 0:
        validation_report['issues'].append("No non-zero damage states found")

    if not validation_report['sample_damage_values']:
        validation_report['issues'].append("No damage values could be extracted")

    return validation_report
