import logging
import multiprocessing
import os

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
