"""
Parallel computing configuration module for SIRA.
Provides automatic detection and configuration of parallel computing environments.
"""

import os
import sys
import logging
import psutil
import json
import numpy as np
import pandas as pd
import socket
from pathlib import Path
from typing import Dict, Optional, Union, Any
import multiprocessing as mp

logger = logging.getLogger(__name__)


class ParallelConfig:
    """
    Configuration manager for parallel computing in SIRA.

    Automatically detects and configures:
    - Available computing resources (CPUs, memory, GPUs)
    - HPC environment (SLURM, PBS, etc.)
    - MPI configuration
    - Dask cluster settings
    - Optimal parallelization parameters
    """

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize parallel configuration.

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
            'hostname': socket.gethostname(),
            'platform': sys.platform,
            'python_version': sys.version,
            'cpu_count': mp.cpu_count(),
            'physical_cores': psutil.cpu_count(logical=False),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'is_hpc': False,
            'hpc_type': None,
            'mpi_available': False,
            'gpu_available': False,
            'gpu_count': 0
        }

        # Detect HPC environment
        if any(var in os.environ for var in ['SLURM_JOB_ID', 'SLURM_NODELIST']):
            env['is_hpc'] = True
            env['hpc_type'] = 'slurm'
            env['slurm_nodes'] = os.environ.get('SLURM_JOB_NODELIST', '')
            env['slurm_tasks'] = int(os.environ.get('SLURM_NTASKS', 1))
            env['slurm_cpus_per_task'] = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
        elif 'PBS_JOBID' in os.environ:
            env['is_hpc'] = True
            env['hpc_type'] = 'pbs'
            env['pbs_nodes'] = os.environ.get('PBS_NODEFILE', '')
            env['pbs_ncpus'] = int(os.environ.get('PBS_NCPUS', mp.cpu_count()))
            # Check if NCI
            if 'nci.org.au' in env['hostname'] or 'PBS_O_HOST' in os.environ:
                env['hpc_subtype'] = 'nci'
        elif 'LSB_JOBID' in os.environ:
            env['is_hpc'] = True
            env['hpc_type'] = 'lsf'

        # Detect MPI
        try:
            from mpi4py import MPI
            env['mpi_available'] = True
            comm = MPI.COMM_WORLD
            env['mpi_size'] = comm.Get_size()
            env['mpi_rank'] = comm.Get_rank()
        except ImportError:
            pass

        # Detect GPUs
        try:
            import torch
            if torch.cuda.is_available():
                env['gpu_available'] = True
                env['gpu_count'] = torch.cuda.device_count()
                env['gpu_names'] = [
                    torch.cuda.get_device_name(i)
                    for i in range(env['gpu_count'])
                ]
        except ImportError:
            # Try with tensorflow
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    env['gpu_available'] = True
                    env['gpu_count'] = len(gpus)
            except ImportError:
                pass

        return env

    def _auto_configure(self):
        """Automatically configure parallel computing settings."""
        env = self.environment

        # Select backend
        if env['is_hpc'] and env['mpi_available']:
            backend = 'mpi'
        elif env['is_hpc'] or env['cpu_count'] > 16:
            backend = 'dask'
        else:
            backend = 'multiprocessing'

        # Configure based on backend
        if backend == 'mpi':
            self.config = self._configure_mpi()
        elif backend == 'dask':
            self.config = self._configure_dask()
        else:
            self.config = self._configure_multiprocessing()

        # Add common settings
        self.config.update({
            'backend': backend,
            'environment': env,
            'recovery_method': 'parallel_streams' if env['cpu_count'] > 4 else 'max',
            'num_repair_streams': min(100, env['cpu_count'] * 2),
            'save_checkpoints': True,
            'checkpoint_interval': 1000,
            'use_compression': True,
            'compression_type': 'snappy',
            'log_level': 'INFO'
        })

    def _configure_mpi(self) -> Dict[str, Any]:
        """Configure MPI-specific settings."""
        env = self.environment

        config = {
            'mpi_spawn_method': 'fork' if sys.platform != 'win32' else 'spawn',
            'mpi_buffer_size': '256MB',
            'mpi_collective_ops': True
        }

        if env['hpc_type'] == 'slurm':
            config.update({
                'nodes': env.get('slurm_nodes', 1),
                'tasks_per_node': env.get('slurm_tasks', 1),
                'cpus_per_task': env.get('slurm_cpus_per_task', 1)
            })

        return config

    def _configure_dask(self) -> Dict[str, Any]:
        """Configure Dask-specific settings."""
        env = self.environment

        # Calculate optimal worker configuration
        available_memory = env['available_memory_gb']
        n_workers = min(env['cpu_count'], 8)  # Cap at 8 workers

        # Reserve 20% memory for system
        worker_memory = (available_memory * 0.8) / n_workers

        config = {
            'dask_scheduler': os.environ.get('DASK_SCHEDULER_ADDRESS', 'local'),
            'dask_n_workers': n_workers,
            'dask_threads_per_worker': max(1, env['cpu_count'] // n_workers),
            'dask_memory_limit': f'{int(worker_memory)}GB',
            'dask_dashboard': True,
            'dask_dashboard_port': 8787,
            'dask_chunk_size': '128MB',
            'dask_dataframe_partitions': n_workers * 2
        }

        # Adjust for HPC
        if env['is_hpc']:
            config['dask_interface'] = 'ib0'  # InfiniBand if available
            config['dask_scheduler_file'] = 'scheduler.json'

        return config

    def _configure_multiprocessing(self) -> Dict[str, Any]:
        """Configure multiprocessing-specific settings."""
        env = self.environment

        # Calculate optimal process count
        n_processes = min(env['cpu_count'], 16)  # Cap at 16 processes

        config = {
            'mp_n_processes': n_processes,
            'mp_chunk_size': 'auto',
            'mp_start_method': 'spawn' if sys.platform == 'win32' else 'fork',
            'mp_maxtasksperchild': 1000,  # Restart workers periodically
            'use_shared_memory': env['memory_gb'] > 16  # Use if enough RAM
        }

        return config

    def _validate_config(self):
        """Validate configuration settings."""
        required_keys = ['backend', 'environment']

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        # Validate backend-specific settings
        backend = self.config['backend']

        if backend == 'mpi' and not self.environment['mpi_available']:
            logger.warning("MPI backend selected but MPI4Py not available. Falling back to Dask.")
            self.config['backend'] = 'dask'

        # Validate memory settings
        if 'dask_memory_limit' in self.config:
            mem_str = self.config['dask_memory_limit']
            mem_value = float(mem_str.replace('GB', '').replace('MB', ''))
            if 'GB' in mem_str and mem_value > self.environment['available_memory_gb']:
                logger.warning(
                    f"Dask memory limit ({mem_value}GB) exceeds available memory "
                    f"({self.environment['available_memory_gb']:.1f}GB). Adjusting..."
                )
                self.config['dask_memory_limit'] = \
                    f"{int(self.environment['available_memory_gb'] * 0.8)}GB"

    def _load_config(self, config_file: Path):
        """Load configuration from file."""
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def save_config(self, output_file: Path):
        """Save configuration to file."""
        with open(output_file, 'w') as f:
            json.dump(self.config, f, indent=2)

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
        available_memory_mb = self.environment['available_memory_gb'] * 1024

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
            'max_workers': self.config.get(
                'dask_n_workers', self.config.get('mp_n_processes', 1)),
            'max_memory_gb': self.environment['available_memory_gb'] * 0.8,
            'max_threads': self.environment['cpu_count'],
            'gpu_available': self.environment['gpu_available'],
            'gpu_count': self.environment['gpu_count']
        }

    def optimize_for_scenario(self, scenario_type: str) -> Dict[str, Any]:
        """
        Get optimized settings for specific scenario types.

        Parameters
        ----------
        scenario_type : str
            Type of scenario: 'small', 'medium', 'large', 'xlarge'

        Returns
        -------
        dict
            Optimized configuration for scenario
        """
        base_config = self.config.copy()

        scenario_configs = {
            'small': {  # < 1000 events
                'backend': 'multiprocessing',
                'mp_n_processes': min(4, self.environment['cpu_count']),
                'batch_size': 100,
                'use_compression': False
            },
            'medium': {  # 1000-10000 events
                'backend': 'multiprocessing' if not self.environment['is_hpc'] else 'dask',
                'mp_n_processes': min(8, self.environment['cpu_count']),
                'dask_n_workers': 4,
                'batch_size': 500,
                'use_compression': True
            },
            'large': {  # 10000-100000 events
                'backend': 'dask',
                'dask_n_workers': min(16, self.environment['cpu_count']),
                'batch_size': 1000,
                'use_compression': True,
                'checkpoint_interval': 5000
            },
            'xlarge': {  # > 100000 events
                'backend': 'mpi' if self.environment['mpi_available'] else 'dask',
                'dask_n_workers': self.environment['cpu_count'],
                'batch_size': 5000,
                'use_compression': True,
                'checkpoint_interval': 10000
            }
        }

        if scenario_type in scenario_configs:
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
        print(f"  Memory: {env['memory_gb']:.1f} GB total, {env['available_memory_gb']:.1f} GB available")
        print(f"  HPC: {'Yes' if env['is_hpc'] else 'No'}", end='')
        if env['is_hpc']:
            print(f" ({env['hpc_type']})")
        else:
            print()
        print(f"  MPI: {'Available' if env['mpi_available'] else 'Not available'}")
        print(f"  GPU: {'Available' if env['gpu_available'] else 'Not available'}", end='')
        if env['gpu_available']:
            print(f" ({env['gpu_count']} devices)")
        else:
            print()

        print("\nConfiguration:")
        print(f"  Backend: {self.config['backend']}")
        print(f"  Recovery method: {self.config['recovery_method']}")
        print(f"  Repair streams: {self.config['num_repair_streams']}")

        if self.config['backend'] == 'dask':
            print(f"  Dask workers: {self.config.get('dask_n_workers', 'auto')}")
            print(f"  Worker memory: {self.config.get('dask_memory_limit', 'auto')}")
        elif self.config['backend'] == 'multiprocessing':
            print(f"  Processes: {self.config.get('mp_n_processes', 'auto')}")

        print("=" * 60 + "\n")


def setup_parallel_environment(
    scenario_size: str = 'auto',
    config_file: Optional[Path] = None,
    verbose: bool = True
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

    # Optimize for scenario size
    if scenario_size != 'auto':
        optimized = config.optimize_for_scenario(scenario_size)
        config.config.update(optimized)

    # Print summary if requested
    if verbose:
        config.print_config_summary()

    # Set environment variables
    if config.config['backend'] == 'dask':
        os.environ['DASK_DISTRIBUTED__SCHEDULER__WORK_STEALING'] = 'True'
        os.environ['DASK_DISTRIBUTED__SCHEDULER__BANDWIDTH'] = '100 MB/s'

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
        yield items[i:i + batch_size]


def parallelize_dataframe(df, func, config=None, **kwargs):
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

    if config.config['backend'] == 'dask':
        import dask.dataframe as dd

        # Convert to Dask DataFrame
        ddf = dd.from_pandas(  # type: ignore
            df, npartitions=config.config.get('dask_dataframe_partitions', 4)
        )

        # Apply function
        result = ddf.apply(func, axis=1, **kwargs)

        # Compute result
        return result.compute()
    else:
        # Use multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        n_workers = config.config.get('mp_n_processes', 4)

        # Split DataFrame
        chunks = np.array_split(df, n_workers)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(pd.DataFrame(chunk).apply, func, axis=1, **kwargs)
                for chunk in chunks
            ]

            results = [future.result() for future in futures]

        return pd.concat(results)
