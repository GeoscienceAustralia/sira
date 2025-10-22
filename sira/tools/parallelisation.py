import logging
import math
import multiprocessing
import os
import time
from typing import Any, Dict

import dask.dataframe as dd  # type: ignore
import pandas as pd

logger = logging.getLogger(__name__)


def get_available_cores():
    """
    Determine the number of available CPU cores, accounting for HPC environment limitations.
    """
    # Get the number of CPU cores using multiprocessing
    cpu_count = multiprocessing.cpu_count()

    # Check for environment variables that might limit cores (common in HPC environments)
    env_cores = None

    # Check PBS/Torque environment variables (used by NCI)
    if "PBS_NCPUS" in os.environ:
        env_cores = int(os.environ["PBS_NCPUS"])
    elif "PBS_NP" in os.environ:
        env_cores = int(os.environ["PBS_NP"])

    # Check SLURM environment variables
    elif "SLURM_CPUS_PER_TASK" in os.environ:
        env_cores = int(os.environ["SLURM_CPUS_PER_TASK"])
    elif "SLURM_NTASKS" in os.environ:
        env_cores = int(os.environ["SLURM_NTASKS"])

    # Check SGE environment variables
    elif "NSLOTS" in os.environ:
        env_cores = int(os.environ["NSLOTS"])

    # Return the smaller of the detected values (if env_cores is set)
    if env_cores is not None:
        return min(cpu_count, env_cores)

    return cpu_count


def recommend_partitions(df, task_type="balanced", partition_size_mb=50, override_cores=None):
    """
    Recommend the optimal number of partitions for converting a pandas DataFrame to Dask.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be converted to Dask
    task_type : str, optional
        Type of workload: 'cpu_bound', 'io_bound', or 'balanced' (default)
    partition_size_mb : int, optional
        Target size of each partition in MB (default: 50)
    override_cores : int, optional
        Manually specify number of cores, otherwise auto-detected

    Returns:
    --------
    int
        Recommended number of partitions
    """
    # Get available cores
    cores = override_cores if override_cores is not None else get_available_cores()

    # Calculate DataFrame size in bytes
    df_size_bytes = df.memory_usage(deep=True).sum()
    df_size_mb = df_size_bytes / (1024 * 1024)

    # Calculate partitions based on data size
    size_based_partitions = max(1, math.ceil(df_size_mb / partition_size_mb))

    # Apply multiplier based on task type
    if task_type == "cpu_bound":
        # For CPU-bound tasks, use 1-2x cores
        core_multiplier = 1.5
    elif task_type == "io_bound":
        # For I/O-bound tasks, use 3-4x cores
        core_multiplier = 3.5
    else:  # balanced
        # For mixed workloads, use 2-3x cores
        core_multiplier = 2.5

    core_based_partitions = max(1, math.ceil(cores * core_multiplier))

    # Use the larger of the two recommendations
    recommended_partitions = max(size_based_partitions, core_based_partitions)

    # Ensure we have at least as many partitions as cores
    recommended_partitions = max(recommended_partitions, cores)

    return recommended_partitions


def pandas_to_dask_optimal(df, task_type="balanced", partition_size_mb=50, override_cores=None):
    """
    Convert a pandas DataFrame to a Dask DataFrame with optimal partitioning.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be converted to Dask
    task_type : str, optional
        Type of workload: 'cpu_bound', 'io_bound', or 'balanced' (default)
    partition_size_mb : int, optional
        Target size of each partition in MB (default: 50)
    override_cores : int, optional
        Manually specify number of cores, otherwise auto-detected

    Returns:
    --------
    dask.dataframe.DataFrame
        Dask DataFrame with optimal partitioning
    """
    npartitions = recommend_partitions(df, task_type, partition_size_mb, override_cores)
    print(f"Converting pandas DataFrame to Dask using {npartitions} partitions")
    print(f"Available cores detected: {get_available_cores()}")

    return dd.from_pandas(df, npartitions=npartitions)


class DaskClientManager:
    """
    Manages Dask client lifecycle for parallel processing.

    This class handles the creation, configuration, and cleanup of Dask clients
    for distributed computing tasks in SIRA.
    """

    def __init__(
        self,
        scheduler_address=None,
        n_workers=None,
        threads_per_worker=None,
        memory_limit=None,
    ):
        """
        Initialise Dask client manager.

        Parameters
        ----------
        scheduler_address : str, optional
            Address of Dask scheduler. If None, creates local cluster.
        n_workers : int, optional
            Number of workers for local cluster.
        threads_per_worker : int, optional
            Number of threads per worker.
        memory_limit : str, optional
            Memory limit per worker (e.g., '2GB').
        """
        self.client = None
        self.cluster = None

        try:
            from dask.distributed import Client, LocalCluster

            # Environment-driven overrides for multi-node friendliness
            env_scheduler = (
                scheduler_address
                or os.environ.get("SIRA_DASK_SCHEDULER")
                or os.environ.get("DASK_SCHEDULER_ADDRESS")
                or os.environ.get("DASK_SCHEDULER")
            )
            if env_scheduler:
                scheduler_address = env_scheduler

            if n_workers is None:
                env_workers = os.environ.get("SIRA_DASK_WORKERS")
                if env_workers:
                    try:
                        n_workers = int(env_workers)
                    except Exception:
                        pass

            if threads_per_worker is None:
                env_threads = os.environ.get("SIRA_DASK_THREADS")
                if env_threads:
                    try:
                        threads_per_worker = int(env_threads)
                    except Exception:
                        pass

            if memory_limit is None:
                env_mem = os.environ.get("SIRA_DASK_MEMORY_LIMIT")
                if env_mem is not None:
                    if env_mem.strip().lower() in ("", "auto", "default"):
                        memory_limit = None
                    else:
                        memory_limit = env_mem

            if scheduler_address:
                # Connect to existing scheduler
                self.client = Client(scheduler_address)
                logger.info(f"Connected to Dask scheduler at: {scheduler_address}")
            else:
                # Create local cluster
                cluster_kwargs: Dict[str, Any] = {}

                if n_workers:
                    cluster_kwargs["n_workers"] = n_workers
                if threads_per_worker:
                    cluster_kwargs["threads_per_worker"] = threads_per_worker
                if memory_limit:
                    cluster_kwargs["memory_limit"] = memory_limit

                # Set reasonable defaults
                if not cluster_kwargs:
                    n_cores = multiprocessing.cpu_count()
                    # Conservative defaults; no hard memory cap
                    cluster_kwargs = {
                        "n_workers": min(n_cores, 8),
                        "threads_per_worker": max(1, n_cores // min(n_cores, 8)),
                    }

                self.cluster = LocalCluster(**cluster_kwargs)
                self.client = Client(self.cluster)
                logger.info(f"Created local Dask cluster with {cluster_kwargs}")

        except ImportError:
            logger.warning("Dask not available. Falling back to sequential processing.")
            self.client = None
            self.cluster = None
        except Exception as e:
            logger.error(f"Failed to initialise Dask client: {e}")
            self.client = None
            self.cluster = None

    def close(self, futures=None, quiet: bool = True):
        """Close Dask client and cluster gracefully.

        Parameters
        ----------
        futures : Iterable, optional
            Futures to cancel before closing (if the caller tracks them).
        quiet : bool
            Temporarily suppress distributed heartbeat/comm errors during teardown.
        """
        noisy_logger_names = [
            "distributed",
            "distributed.worker",
            "distributed.core",
            "distributed.comm",
            "distributed.comm.tcp",
            "distributed.scheduler",
            "distributed.nanny",
            "distributed.client",
        ]
        previous_levels = {}

        try:
            if quiet:
                for name in noisy_logger_names:
                    lg = logging.getLogger(name)
                    previous_levels[name] = lg.level
                    lg.setLevel(logging.CRITICAL)

            # Best effort: cancel any outstanding futures if provided
            try:
                if self.client and futures is not None:
                    try:
                        self.client.cancel(futures, force=True)
                    except Exception:
                        pass
            except Exception:
                pass

            # Close the client first, then the cluster (if we own it)
            if self.client:
                try:
                    self.client.close()
                except Exception:
                    pass
                logger.info("Dask client closed")

            # Give workers a brief moment to wind down before closing cluster
            try:
                time.sleep(0.2)
            except Exception:
                pass

            if self.cluster:
                try:
                    # Only close the cluster if it was created by us
                    self.cluster.close()
                except Exception:
                    pass
                logger.info("Dask cluster closed")

        finally:
            # Restore previous logger levels
            if quiet:
                for name, lvl in previous_levels.items():
                    try:
                        logging.getLogger(name).setLevel(lvl)
                    except Exception:
                        pass


# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    df = pd.DataFrame({"a": range(1000000), "b": range(1000000)})

    # Get recommended partitions
    rec_parts = recommend_partitions(df)
    print(f"Recommended partitions: {rec_parts}")

    # Convert to Dask with optimal partitioning
    ddf = pandas_to_dask_optimal(df)
    print(f"Dask DataFrame created with {ddf.npartitions} partitions")
