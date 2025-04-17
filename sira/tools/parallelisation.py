import multiprocessing
import os
import pandas as pd
import dask.dataframe as dd    # type: ignore
import math


def get_available_cores():
    """
    Determine the number of available CPU cores, accounting for HPC environment limitations.
    """
    # Get the number of CPU cores using multiprocessing
    cpu_count = multiprocessing.cpu_count()

    # Check for environment variables that might limit cores (common in HPC environments)
    env_cores = None

    # Check PBS/Torque environment variables (used by NCI)
    if 'PBS_NCPUS' in os.environ:
        env_cores = int(os.environ['PBS_NCPUS'])
    elif 'PBS_NP' in os.environ:
        env_cores = int(os.environ['PBS_NP'])

    # Check SLURM environment variables
    elif 'SLURM_CPUS_PER_TASK' in os.environ:
        env_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    elif 'SLURM_NTASKS' in os.environ:
        env_cores = int(os.environ['SLURM_NTASKS'])

    # Check SGE environment variables
    elif 'NSLOTS' in os.environ:
        env_cores = int(os.environ['NSLOTS'])

    # Return the smaller of the detected values (if env_cores is set)
    if env_cores is not None:
        return min(cpu_count, env_cores)

    return cpu_count

def recommend_partitions(df, task_type='balanced', partition_size_mb=50, override_cores=None):
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
    if task_type == 'cpu_bound':
        # For CPU-bound tasks, use 1-2x cores
        core_multiplier = 1.5
    elif task_type == 'io_bound':
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

def pandas_to_dask_optimal(df, task_type='balanced', partition_size_mb=50, override_cores=None):
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

# Example usage
if __name__ == "__main__":
    # Create a sample DataFrame
    df = pd.DataFrame({'a': range(1000000), 'b': range(1000000)})

    # Get recommended partitions
    rec_parts = recommend_partitions(df)
    print(f"Recommended partitions: {rec_parts}")

    # Convert to Dask with optimal partitioning
    ddf = pandas_to_dask_optimal(df)
    print(f"Dask DataFrame created with {ddf.npartitions} partitions")
