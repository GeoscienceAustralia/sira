"""
Parallel computing configuration module for SIRA.
Provides automatic detection and configuration of parallel computing environments.
"""

import json
import logging
import multiprocessing as mp
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


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

    # Check if we're in a container with MPI setup
    if os.path.exists("/.dockerenv") or os.path.exists("/singularity"):
        # In container - check for MPI setup
        if any(var in os.environ for var in mpi_runtime_vars + slurm_vars + pbs_vars):
            return True

    return False


class ParallelConfig:
    """
    Configuration manager for parallel computing in SIRA.

    Automatically detects and configures:
    - Available computing resources (CPUs, memory, GPUs)
    - HPC environment (SLURM, PBS, etc.)
    - MPI configuration
    - Optimal parallelisation parameters
    """

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialise parallel configuration.

        Parameters
        ----------
        config_file : Path, optional
            Path to configuration file. If None, auto-detect settings.
        """
        self.config = {}
        self.environment = self._detect_environment()

        # Load configuration
        if config_file and Path(config_file).exists():
            self._load_config(config_file)
        else:
            self._auto_configure()

        # Validate configuration
        self._validate_config()

    def _detect_environment(self) -> Dict[str, Any]:
        """Detect computing environment characteristics."""
        env = {
            "hostname": socket.gethostname(),
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": mp.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "is_hpc": False,
            "hpc_type": None,
            "mpi_available": False,
            "mpi_environment": False,
            "gpu_available": False,
            "gpu_count": 0,
        }

        # Use the centralised MPI environment detection
        env["mpi_environment"] = is_mpi_environment()

        # Detect HPC environment
        if any(var in os.environ for var in ["SLURM_JOB_ID", "SLURM_NODELIST"]):
            env["is_hpc"] = True
            env["hpc_type"] = "slurm"
            env["slurm_nodes"] = os.environ.get("SLURM_JOB_NODELIST", "")
            env["slurm_tasks"] = int(os.environ.get("SLURM_NTASKS", 1))
            env["slurm_cpus_per_task"] = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        elif "PBS_JOBID" in os.environ:
            env["is_hpc"] = True
            env["hpc_type"] = "pbs"
            env["pbs_nodes"] = os.environ.get("PBS_NODEFILE", "")
            env["pbs_ncpus"] = int(os.environ.get("PBS_NCPUS", mp.cpu_count()))
            # Check if NCI
            if "nci.org.au" in env["hostname"] or "PBS_O_HOST" in os.environ:
                env["hpc_subtype"] = "nci"
        elif "LSB_JOBID" in os.environ:
            env["is_hpc"] = True
            env["hpc_type"] = "lsf"

        # Only try to detect MPI if we're in an MPI environment
        if env["mpi_environment"]:
            try:
                # Set environment variable to prevent auto-initialisation
                os.environ["MPI4PY_RC_INITIALIZE"] = "False"

                from mpi4py import MPI

                env["mpi_available"] = True

                # Check if already initialised
                if MPI.Is_initialized():
                    comm = MPI.COMM_WORLD
                    env["mpi_size"] = comm.Get_size()
                    env["mpi_rank"] = comm.Get_rank()
                else:
                    # Don't initialise here - just note it's available
                    env["mpi_size"] = 1
                    env["mpi_rank"] = 0

            except ImportError:
                env["mpi_available"] = False
            except Exception as e:
                logger.warning(f"MPI detection failed: {e}")
                env["mpi_available"] = False
        else:
            env["mpi_available"] = False

        # Detect GPUs (disabled by default). Enable via env var SIRA_ENABLE_GPU_DETECT=1
        if os.environ.get("SIRA_ENABLE_GPU_DETECT", "0") == "1":
            try:
                import torch  # optional, used only for detection

                if torch.cuda.is_available():
                    env["gpu_available"] = True
                    env["gpu_count"] = torch.cuda.device_count()
                    env["gpu_names"] = [
                        torch.cuda.get_device_name(i) for i in range(env["gpu_count"])
                    ]
            except ImportError:
                # Try with tensorflow (optional, detection only)
                try:
                    import tensorflow as tf  # type: ignore

                    gpus = tf.config.list_physical_devices("GPU")
                    if gpus:
                        env["gpu_available"] = True
                        env["gpu_count"] = len(gpus)
                except ImportError:
                    pass

        return env

    def _auto_configure(self):
        """Automatically configure parallel computing settings."""
        env = self.environment

        # Select backend based on environment
        if env["mpi_environment"] and env["mpi_available"]:
            backend = "mpi"
        else:
            backend = "multiprocessing"

        # Configure based on backend
        if backend == "mpi":
            self.config = self._configure_mpi()
        else:
            self.config = self._configure_multiprocessing()

        # Add common settings
        self.config.update(
            {
                "backend": backend,
                "environment": env,
                "recovery_method": self.config.get("recovery_method", "max"),
                "num_repair_streams": min(100, env["cpu_count"] * 2),
                "save_checkpoints": True,
                "checkpoint_interval": 1000,
                "use_compression": True,
                "compression_type": "snappy",
                "log_level": "INFO",
            }
        )

    def _configure_mpi(self) -> Dict[str, Any]:
        """Configure MPI-specific settings."""
        env = self.environment

        config = {
            "mpi_spawn_method": "fork" if sys.platform != "win32" else "spawn",
            "mpi_buffer_size": "256MB",
            "mpi_collective_ops": True,
        }

        if env["hpc_type"] == "slurm":
            config.update(
                {
                    "nodes": env.get("slurm_nodes", 1),
                    "tasks_per_node": env.get("slurm_tasks", 1),
                    "cpus_per_task": env.get("slurm_cpus_per_task", 1),
                }
            )

        return config

    def _configure_multiprocessing(self) -> Dict[str, Any]:
        """Configure multiprocessing-specific settings."""
        env = self.environment

        # Calculate optimal process count
        n_processes = min(env["cpu_count"], 16)

        config = {
            "mp_n_processes": n_processes,
            "mp_chunk_size": "auto",
            "mp_start_method": "spawn" if sys.platform == "win32" else "fork",
            "mp_maxtasksperchild": 1000,  # Restart workers periodically
            "use_shared_memory": env["memory_gb"] > 16,  # Use if enough RAM
        }

        return config

    def _validate_config(self):
        """Validate configuration settings."""
        required_keys = ["backend", "environment"]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate backend-specific settings
        backend = self.config["backend"]

        if backend == "mpi" and not self.environment["mpi_available"]:
            logger.warning(
                "MPI backend selected but MPI4Py not available. Falling back to multiprocessing."
            )
            self.config["backend"] = "multiprocessing"

    def _load_config(self, config_file: Path):
        """Load configuration from file."""
        with open(config_file, "r") as f:
            self.config = json.load(f)

    def save_config(self, output_file: Path):
        """Save configuration to file."""
        # Create a copy for serialisation, excluding non-serialisable items
        config_copy = {}
        for key, value in self.config.items():
            try:
                json.dumps(value)  # Test if serialisable
                config_copy[key] = value
            except TypeError:
                # Skip non-serialisable items
                config_copy[key] = str(value)

        with open(output_file, "w") as f:
            json.dump(config_copy, f, indent=2)

    def get_optimal_batch_size(self, total_items: int, item_size_mb: float = 1.0) -> int:
        """
        Calculate optimal batch size based on available resources.

        Parameters
        ----------
        total_items : int
            Total number of items to process
        item_size_mb : float
            Estimated size of each item in MB

        Returns
        -------
        int
            Optimal batch size
        """
        available_memory_mb = self.environment["available_memory_gb"] * 1024

        # Use 50% of available memory for batches
        batch_memory_mb = available_memory_mb * 0.5

        # Calculate batch size
        batch_size = int(batch_memory_mb / item_size_mb)

        # Apply constraints
        min_batch = 10
        max_batch = min(10000, total_items // 4)  # At least 4 batches

        return max(min_batch, min(batch_size, max_batch))

    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits for current environment."""
        return {
            "max_workers": self.config.get("mp_n_processes", 1),
            "max_memory_gb": self.environment["available_memory_gb"] * 0.8,
            "max_threads": self.environment["cpu_count"],
            "gpu_available": self.environment["gpu_available"],
            "gpu_count": self.environment["gpu_count"],
        }

    def optimise_for_scenario(self, scenario_type: str) -> Dict[str, Any]:
        """
        Get optimised settings for specific scenario types.

        Parameters
        ----------
        scenario_type : str
            Type of scenario: 'small', 'medium', 'large', 'xlarge'

        Returns
        -------
        dict
            Optimised configuration for scenario
        """
        base_config = self.config.copy()

        scenario_configs = {
            "small": {  # < 1000 events
                "backend": "multiprocessing",
                "mp_n_processes": min(4, self.environment["cpu_count"]),
                "batch_size": 100,
                "use_compression": False,
            },
            "medium": {  # 1000-10000 events
                "backend": "multiprocessing",
                "mp_n_processes": min(8, self.environment["cpu_count"]),
                "batch_size": 500,
                "use_compression": True,
            },
            "large": {  # 10000-100000 events
                "backend": "multiprocessing",
                "mp_n_processes": min(16, self.environment["cpu_count"]),
                "batch_size": 1000,
                "use_compression": True,
                "checkpoint_interval": 5000,
            },
            "xlarge": {  # > 100000 events
                "backend": "mpi" if self.environment["mpi_available"] else "multiprocessing",
                "mp_n_processes": self.environment["cpu_count"],
                "batch_size": 5000,
                "use_compression": True,
                "checkpoint_interval": 10000,
            },
        }

        if scenario_type in scenario_configs:
            # Only use MPI backend if we're actually in an MPI environment
            if (
                scenario_configs[scenario_type].get("backend") == "mpi"
                and not self.environment["mpi_environment"]
            ):
                scenario_configs[scenario_type]["backend"] = "multiprocessing"

            base_config.update(scenario_configs[scenario_type])

        return base_config

    def print_config_summary(self):
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print("SIRA Parallel Computing Configuration")
        print("=" * 60)

        env = self.environment
        print("\nEnvironment:")
        print(f"  Platform: {env['platform']}")
        print(f"  Hostname: {env['hostname']}")
        print(f"  CPUs: {env['cpu_count']} (logical), {env['physical_cores']} (physical)")
        print(
            f"  Memory: {env['memory_gb']:.1f} GB total, "
            f"{env['available_memory_gb']:.1f} GB available"
        )
        print(f"  HPC: {'Yes' if env['is_hpc'] else 'No'}", end="")
        if env["is_hpc"]:
            print(f" ({env['hpc_type']})")
        else:
            print()
        print(f"  MPI Environment: {'Yes' if env['mpi_environment'] else 'No'}")
        print(f"  MPI Available: {'Yes' if env['mpi_available'] else 'No'}")
        print(f"  GPU: {'Available' if env['gpu_available'] else 'Not available'}", end="")
        if env["gpu_available"]:
            print(f" ({env['gpu_count']} devices)")
        else:
            print()

        print("\nConfiguration:")
        print(f"  Backend: {self.config['backend']}")
        print(f"  Recovery method: {self.config['recovery_method']}")
        print(f"  Repair streams: {self.config['num_repair_streams']}")

        if self.config["backend"] == "multiprocessing":
            print(f"  Processes: {self.config.get('mp_n_processes', 'auto')}")

        print("=" * 60 + "\n")


def setup_parallel_environment(
    scenario_size: str = "auto", config_file: Optional[Path] = None, verbose: bool = True
) -> ParallelConfig:
    """
    Setup and configure parallel computing environment.

    Parameters
    ----------
    scenario_size : str
        Size of scenario: 'auto', 'small', 'medium', 'large', 'xlarge'
    config_file : Path, optional
        Path to configuration file
    verbose : bool
        Print configuration summary

    Returns
    -------
    ParallelConfig
        Configured parallel computing environment
    """
    # Create configuration
    config = ParallelConfig(config_file)

    # Optimise for scenario size
    if scenario_size != "auto":
        optimised = config.optimise_for_scenario(scenario_size)
        config.config.update(optimised)

    # Print summary if requested
    if verbose:
        config.print_config_summary()

    return config


# Utility functions for common patterns


def get_batch_iterator(items, batch_size=None, config=None):
    """
    Create an iterator for processing items in batches.

    Parameters
    ----------
    items : list
        Items to process
    batch_size : int, optional
        Batch size. If None, auto-calculate.
    config : ParallelConfig, optional
        Configuration object

    Yields
    ------
    list
        Batch of items
    """
    if batch_size is None:
        if config:
            batch_size = config.get_optimal_batch_size(len(items))
        else:
            batch_size = 1000

    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def parallelise_dataframe(df, func, config=None, **kwargs):
    """
    Apply function to DataFrame in parallel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to process
    func : callable
        Function to apply
    config : ParallelConfig, optional
        Configuration object
    **kwargs
        Additional arguments for func

    Returns
    -------
    pd.DataFrame
        Processed DataFrame
    """
    if config is None:
        config = ParallelConfig()

    # Use multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    n_workers = config.config.get("mp_n_processes", 4)

    # Split DataFrame
    chunks = np.array_split(df, n_workers)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(pd.DataFrame(chunk).apply, func, axis=1, **kwargs)
            for chunk in chunks
        ]

        results = [future.result() for future in futures]

    return pd.concat(results)
