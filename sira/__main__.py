#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
title:        __main__.py
description:  entry point for core sira component with parallel computing support
usage:
    python sira [OPTIONS]
    -h                    Display this usage message
    -d [input_directory]  Specify the directory with the required
                            config and model files
    -s                    Run simulation
    -f                    Conduct model fitting. Must be done
                            after a complete run with `-s` flag
    -l                    Conduct loss analysis. Must be done
                            after a complete run with `-s` flag
    -v [LEVEL]            Choose `verbose` mode, or choose logging
                            level DEBUG, INFO, WARNING, ERROR, CRITICAL

    Parallel Computing Options:
    --parallel-backend    Choose backend: auto, mpi, dask, multiprocessing
    --max-workers         Maximum number of workers/processes
    --scenario-size       Scenario size: auto, small, medium, large, xlarge
    --disable-parallel    Disable parallel processing
    --checkpoint          Enable checkpointing for recovery
    --parallel-config     Path to parallel configuration file

python_version  : 3.11
"""

import argparse
import logging
import os
import re
import sys
from datetime import timedelta
from pathlib import Path
from time import localtime, strftime, time

import numpy as np
from colorama import Fore, init

# Import SIRA modules
from sira.configuration import Configuration
from sira.fit_model import fit_prob_exceed_model
from sira.infrastructure_response import (
    consolidate_streamed_results,
    exceedance_prob_by_component_class,
    write_system_response,
)
from sira.logger import configure_logger
from sira.loss_analysis import run_scenario_loss_analysis
from sira.model_ingest import ingest_model
from sira.modelling.hazard import HazardsContainer
from sira.modelling.system_topology import SystemTopologyGenerator

# Import parallel computing modules
from sira.parallel_config import ParallelConfig, setup_parallel_environment
from sira.scenario import Scenario
from sira.simulation import calculate_response
from sira.tools import utils

logging.basicConfig(level=logging.DEBUG)
rootLogger = logging.getLogger(__name__)

# Add the source dir to system path
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

init()
np.seterr(divide="print", invalid="raise")

# ==============================================================================


def diagnose_serialisation_issues(scenario, infrastructure, hazards_container):
    """Diagnose what's preventing multiprocessing from working."""
    import pickle

    print("\nChecking for serialisation issues...")

    # Test scenario
    try:
        pickle.dumps(scenario)
        print("[OK] Scenario object is picklable")
    except Exception as e:
        print(f"[x] Scenario object is NOT picklable: {e}")

    # Test infrastructure
    try:
        pickle.dumps(infrastructure)
        print("[OK] Infrastructure object is picklable")
    except Exception as e:
        print(f"[x] Infrastructure object is NOT picklable: {e}")

    # Test first hazard
    hazard_list = list(hazards_container.listOfhazards)
    if hazard_list:
        try:
            pickle.dumps(hazard_list[0])
            print("[OK] Hazard object is picklable")
        except Exception as e:
            print(f"[x] Hazard object is NOT picklable: {e}")

    print("Diagnosis complete.\n")


# ==============================================================================


def is_mpi_environment():
    """
    Detect if we're in a proper MPI environment.

    Returns
    -------
    bool
        True if in proper MPI environment, False otherwise
    """
    # Check for SLURM (most common HPC scheduler)
    slurm_vars = ["SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_PROCID", "SLURM_NODELIST"]
    if any(var in os.environ for var in slurm_vars):
        rootLogger.info("Detected SLURM environment - MPI available")
        return True

    # Check for PBS/Torque (another common HPC scheduler)
    pbs_vars = ["PBS_JOBID", "PBS_NCPUS", "PBS_NODEFILE"]
    if any(var in os.environ for var in pbs_vars):
        rootLogger.info("Detected PBS/Torque environment - MPI available")
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
        rootLogger.info("Detected MPI runtime environment variables - MPI available")
        return True

    # Check if we're being launched with mpirun/mpiexec
    parent_process = os.environ.get("_", "")
    if any(launcher in parent_process for launcher in ["mpirun", "mpiexec", "srun"]):
        rootLogger.info("Detected MPI launcher in parent process - MPI available")
        return True

    # Check for HPC-specific hostnames or environments
    hostname = os.environ.get("HOSTNAME", "")
    if any(pattern in hostname.lower() for pattern in ["hpc", "cluster", "node", "compute"]):
        rootLogger.info(f"Detected HPC-like hostname: {hostname} - MPI may be available")
        return True

    # Check if we're in a container with MPI setup
    if os.path.exists("/.dockerenv") or os.path.exists("/singularity"):
        # In container - check for MPI setup
        if any(var in os.environ for var in mpi_runtime_vars + slurm_vars + pbs_vars):
            rootLogger.info("Detected container with MPI environment - MPI available")
            return True

    rootLogger.info("No MPI environment detected - running on regular workstation/laptop")
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
        os.environ["MPI4PY_RC_INITIALIZE"] = "False"

        from mpi4py import MPI

        # Check if already initialised
        if not MPI.Is_initialized():
            MPI.Init()
            # Register finaliser to ensure cleanup
            import atexit

            atexit.register(MPI.Finalize)

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


def detect_environment_and_setup_parallel(args, input_dir, scenario, hazards_container):
    """
    Detect computing environment and setup parallel configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    scenario : Scenario
        SIRA scenario object
    hazards_container : HazardsContainer
        Hazards data

    Returns
    -------
    ParallelConfig
        Configured parallel computing environment
    """

    # Skip if parallel processing is disabled
    if args.disable_parallel:
        rootLogger.info("Parallel processing disabled by user")
        return None

    # Force disable MPI if not in proper environment
    in_mpi_env = is_mpi_environment()

    if args.parallel_backend == "mpi" and not in_mpi_env:
        rootLogger.warning("MPI backend requested but not in MPI environment. Switching to auto.")
        args.parallel_backend = "auto"

    # Determine scenario size if auto
    if args.scenario_size == "auto":
        num_events = len(hazards_container.hazard_scenario_list)
        num_samples = scenario.num_samples
        total_computations = num_events * num_samples

        if total_computations < 1000:
            scenario_size = "small"
        elif total_computations < 10000:
            scenario_size = "medium"
        elif total_computations < 100000:
            scenario_size = "large"
        else:
            scenario_size = "xlarge"

        rootLogger.info(
            f"Auto-detected scenario size: {scenario_size} "
            f"({num_events} events {chr(0x00D7)} {num_samples} samples "
            f"= {total_computations:,} computations)"
        )
    else:
        scenario_size = args.scenario_size

    # Setup parallel configuration
    if args.parallel_config and Path(args.parallel_config).exists():
        # Load from file
        rootLogger.info(f"Loading parallel config from: {args.parallel_config}")
        pconfig = ParallelConfig(config_file=Path(args.parallel_config))
    else:
        # Auto-configure based on environment
        rootLogger.info("No parallel config file provided, auto-configuring")
        pconfig = setup_parallel_environment(scenario_size=scenario_size, verbose=True)

    # Override with command line options
    if args.parallel_backend != "auto":
        if args.parallel_backend == "mpi" and not in_mpi_env:
            rootLogger.warning(
                "MPI backend forced but environment not suitable. Using Dask instead."
            )
            pconfig.config["backend"] = "dask"
        else:
            pconfig.config["backend"] = args.parallel_backend
            rootLogger.info(f"Overriding backend to: {args.parallel_backend}")

    if args.max_workers:
        if pconfig.config["backend"] == "dask":
            pconfig.config["dask_n_workers"] = args.max_workers
        elif pconfig.config["backend"] == "multiprocessing":
            pconfig.config["mp_n_processes"] = args.max_workers
        rootLogger.info(f"Setting max workers to: {args.max_workers}")

    # Optional Dask threads-per-worker override
    if getattr(args, "dask_threads_per_worker", None) is not None:
        if pconfig.config["backend"] == "dask":
            if args.dask_threads_per_worker and args.dask_threads_per_worker > 0:
                pconfig.config["dask_threads_per_worker"] = args.dask_threads_per_worker
                rootLogger.info(
                    f"Setting Dask threads per worker to: {args.dask_threads_per_worker}"
                )
            else:
                rootLogger.warning("Ignoring --dask-threads-per-worker because value must be >= 1")

    # Apply scenario-specific settings
    if hasattr(scenario, "recovery_method"):
        pconfig.config["recovery_method"] = scenario.recovery_method
    if hasattr(scenario, "num_repair_streams"):
        pconfig.config["num_repair_streams"] = scenario.num_repair_streams

    # Enable checkpointing if requested
    if args.checkpoint:
        pconfig.config["save_checkpoints"] = True
        pconfig.config["checkpoint_dir"] = Path(scenario.output_path, "checkpoints")
        pconfig.config["checkpoint_dir"].mkdir(exist_ok=True)
        rootLogger.info(f"Checkpointing enabled: {pconfig.config['checkpoint_dir']}")

    # Save configuration for reference
    config_save_path = Path(input_dir, "parallel_config.json")
    pconfig.save_config(config_save_path)
    config_save_path_wrapped = utils.wrap_file_path(str(config_save_path))
    rootLogger.info(
        f"Saved parallel configuration to: \n{Fore.YELLOW}{config_save_path_wrapped}{Fore.RESET}"
    )

    return pconfig


def initialise_parallel_backend(para_config):
    """
    Initialise the selected parallel backend.

    Parameters
    ----------
    para_config : ParallelConfig
        Parallel configuration

    Returns
    -------
    dict
        Backend-specific initialisation data
    """
    backend_data = {}

    if para_config.config["backend"] == "mpi":
        # Use safe MPI import
        MPI, comm, rank, size = safe_mpi_import()

        if MPI is not None:
            backend_data["MPI"] = MPI
            backend_data["comm"] = comm
            backend_data["rank"] = rank
            backend_data["size"] = size

            if rank == 0:
                rootLogger.info(f"MPI initialised with {size} processes")

            # All processes (including workers) should continue -
            # the MPI backend will handle coordination internally
        else:
            # When MPI is explicitly requested but fails, fall back to multiprocessing
            rootLogger.warning("MPI initialisation failed but MPI backend was explicitly requested")
            rootLogger.warning("Ensure mpirun/mpiexec is used and mpi4py is installed")
            rootLogger.warning("Falling back to multiprocessing backend")
            para_config.config["backend"] = "multiprocessing"

    if para_config.config["backend"] == "multiprocessing":
        rootLogger.info("Using multiprocessing backend (no additional setup required)")

    if para_config.config["backend"] == "dask":
        from sira.tools.parallelisation import DaskClientManager

        n_workers = para_config.config.get("dask_n_workers")
        threads_per_worker = para_config.config.get("dask_threads_per_worker")
        memory_limit = para_config.config.get("dask_memory_limit")

        dask_manager = DaskClientManager(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
        )
        backend_data["dask_client"] = dask_manager.client
        backend_data["dask_manager"] = dask_manager

        rootLogger.info(f"Dask client initialised: {dask_manager.client}")

    return backend_data


def main(args=None):
    """
    Main entry point for SIRA with parallel computing support.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Pre-parsed arguments (mainly for MPI)
    """

    # rootLogger = logging.getLogger(__name__)
    # ---------------------------------------------------------------------------------
    # define arg parser
    parser = argparse.ArgumentParser(
        prog="sira", description="run sira with parallel computing support", add_help=True
    )

    # ---------------------------------------------------------------------------------
    # [Either] Supply config file and model file directly:
    parser.add_argument("-c", "--config_file", type=str)
    parser.add_argument("-m", "--model_file", type=str)

    # [Or] Supply only the directory where the input files reside
    parser.add_argument("-d", "--input_directory", type=str)

    # ---------------------------------------------------------------------------------
    # Define the sim arguments - tell the code what tasks to do

    VERBOSITY_CHOICES = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    parser.add_argument("-s", "--simulation", action="store_true", default=False)
    parser.add_argument("-t", "--draw_topology", action="store_true", default=False)
    parser.add_argument("-f", "--fit", action="store_true", default=False)
    parser.add_argument("-l", "--loss_analysis", action="store_true", default=False)
    parser.add_argument("-r", "--recovery_analysis", action="store_true", default=False)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        type=str,
        choices=VERBOSITY_CHOICES,
        default="INFO",
        help=f"Choose one of these options for logging: \n{VERBOSITY_CHOICES}",
    )

    # ---------------------------------------------------------------------------------
    # Parallel computing arguments

    parser.add_argument(
        "--parallel-backend",
        type=str,
        choices=["auto", "mpi", "dask", "multiprocessing"],
        default="auto",
        help="Choose parallel computing backend (default: auto)",
    )

    parser.add_argument(
        "--max-workers", type=int, default=None, help="Maximum number of parallel workers/processes"
    )

    parser.add_argument(
        "--dask-threads-per-worker",
        type=int,
        default=None,
        help="For Dask backend: number of threads per worker",
    )

    parser.add_argument(
        "--scenario-size",
        type=str,
        choices=["auto", "small", "medium", "large", "xlarge"],
        default="auto",
        help="Scenario size for optimisation (default: auto-detect)",
    )

    parser.add_argument(
        "--disable-parallel", action="store_true", default=False, help="Disable parallel processing"
    )

    parser.add_argument(
        "--checkpoint",
        action="store_true",
        default=False,
        help="Enable checkpointing for recovery analysis",
    )

    parser.add_argument(
        "--parallel-config", type=str, default=None, help="Path to parallel configuration JSON file"
    )

    # Streaming of large intermediate arrays to disk to reduce driver memory
    parser.add_argument(
        "--stream-results",
        action="store_true",
        default=False,
        help=(
            "Stream per-sample arrays to disk (Parquet) during simulation to reduce memory. "
            "Recommended for large runs/HPC."
        ),
    )

    # Parse arguments
    if args is None:
        args = parser.parse_args()

    # ---------------------------------------------------------------------------------
    # Initialise MPI early for proper coordination

    MPI = None
    comm = None
    rank = 0
    size = 1

    if args.parallel_backend in ["auto", "mpi"] and is_mpi_environment():
        MPI, comm, rank, size = safe_mpi_import()
        if MPI is not None:
            if rank == 0:
                rootLogger.info(f"MPI initialised: rank {rank} of {size}")
            # Broadcast arguments to all processes to ensure consistency
            args = comm.bcast(args, root=0)

    # Only rank 0 does argument validation and logging to avoid conflicts
    if rank == 0:
        rootLogger.info(f"args: {args}")

        # ---------------------------------------------------------------------------------
        # error handling

        if not any(
            [
                args.simulation,
                args.fit,
                args.loss_analysis,
                args.draw_topology,
                args.recovery_analysis,
            ]
        ):
            parser.error(
                "\nOne of these flags is required:\n"
                " --simulation (-s) or \n"
                " --fit (-f) or \n"
                " --draw_topology (-t).\n"
                " --loss_analysis (-l) or \n"
                " --recovery_analysis (-r).\n"
                " The options for `fit`,`loss_analysis`, and `recovery_analysis` require "
                " the -s flag,or a previously completed run with the -s flag.\n"
            )
            if comm is not None:
                comm.Abort()
            sys.exit(2)

        proj_root_dir = args.input_directory

        if not os.path.isdir(proj_root_dir):
            print("Invalid path supplied:\n {}".format(proj_root_dir))
            if comm is not None:
                comm.Abort()
            sys.exit(1)
    else:
        # Worker processes wait for validation to complete
        if comm is not None:
            proj_root_dir = args.input_directory  # Use args value for now
        else:
            # Fallback if MPI failed
            proj_root_dir = args.input_directory

    # Broadcast validated arguments to all processes
    if comm is not None:
        proj_root_dir = comm.bcast(proj_root_dir, root=0)

    # ---------------------------------------------------------------------------------

    class InvalidOrMissingInputFile(Exception):
        def __init__(self, type_of_file):
            super(InvalidOrMissingInputFile, self).__init__()
            raise NameError(
                f"\n{str(type_of_file)} file error: invalid or missing input file."
                "\n  A valid model file name must begin the term `model`, "
                "\n  A valid config file name must begin with the term `config`, and"
                "\n  It must be in JSON format.\n"
            )

    # ---------------------------------------------------------------------------------
    # File discovery and model loading - coordinated across MPI processes

    try:
        proj_input_dir = Path(proj_root_dir, "input").resolve()
    except (IOError, OSError):
        if rank == 0:
            rootLogger.error("Invalid path")
        if comm is not None:
            comm.Abort()
        raise IOError("Invalid path")

    if not Path(proj_input_dir).exists():
        if rank == 0:
            rootLogger.error("Invalid path")
        if comm is not None:
            comm.Abort()
        raise IOError("Invalid path")

    # Only rank 0 does file discovery to avoid race conditions
    config_file_name: str = ""
    model_file_name: str = ""

    if rank == 0:
        for fname in os.listdir(proj_input_dir):
            confmatch = re.search(r"(?i)^config.*\.json$", fname)
            if confmatch is not None:
                config_file_name = confmatch.string

            modelmatch = re.search(r"(?i)^model.*\.json$", fname)
            if modelmatch is not None:
                model_file_name = modelmatch.string

        if not config_file_name:
            rootLogger.error("CONFIG file not found")
            if comm is not None:
                comm.Abort()
            raise InvalidOrMissingInputFile("CONFIG")

        if not model_file_name:
            rootLogger.error("MODEL file not found")
            if comm is not None:
                comm.Abort()
            raise InvalidOrMissingInputFile("MODEL")

    # Broadcast file names to all processes
    if comm is not None:
        config_file_name = comm.bcast(config_file_name, root=0)
        model_file_name = comm.bcast(model_file_name, root=0)

    # ---------------------------------------------------------------------------------
    # Define paths for CONFIG and MODEL

    config_file_path = Path(proj_input_dir, config_file_name).resolve()
    model_file_path = Path(proj_input_dir, model_file_name).resolve()

    # ---------------------------------------------------------------------------------
    # Configure simulation model.
    # Read data and control parameters and construct objects.
    # All processes need these objects for the simulation
    # ---------------------------------------------------------------------------------
    config = Configuration(str(config_file_path), str(model_file_path))
    scenario = Scenario(config)
    infrastructure = ingest_model(config)
    hazards_container = HazardsContainer(config, model_file_path)
    output_path = config.OUTPUT_DIR

    # Expose OUTPUT_DIR to workers and helpers (e.g., MPI comptype partials directory resolution)
    try:
        os.environ["SIRA_OUTPUT_DIR"] = str(output_path)
    except Exception:
        pass

    # Configure streaming mode on scenario so downstream modules can react
    setattr(scenario, "stream_results", bool(getattr(args, "stream_results", False)))
    if getattr(scenario, "stream_results", False):
        env_stream_dir = os.environ.get("SIRA_STREAM_DIR")
        if env_stream_dir:
            setattr(scenario, "stream_dir", Path(env_stream_dir))
        else:
            stream_dir = Path(output_path) / "stream"
            stream_dir.mkdir(parents=True, exist_ok=True)
            setattr(scenario, "stream_dir", stream_dir)

    # ---------------------------------------------------------------------------------
    # Set up logging (only on rank 0 to avoid conflicts)
    # ---------------------------------------------------------------------------------
    if rank == 0:
        start_time = time()
        start_time_strf = strftime("%Y-%m-%d %H:%M:%S", localtime(start_time))
        log_path = os.path.join(output_path, "log.txt")
        configure_logger(log_path, args.loglevel)
        print("\n")
        rootLogger.info(f"{Fore.GREEN}Simulation initiated at: {start_time_strf}{Fore.RESET}\n")
        print("-" * 80)
    else:
        # Workers still need timing info for coordination
        start_time = time()

    # Synchronise all processes before proceeding
    if comm is not None:
        comm.barrier()

    # ---------------------------------------------------------------------------------
    # Setup parallel computing environment (coordinated across MPI processes)
    # ---------------------------------------------------------------------------------

    parallel_config = None
    backend_data = {}

    # Note: -t/-f/-l operations always run sequentially to prevent Dask interference
    has_sequential_operations = args.fit or args.draw_topology or args.loss_analysis
    only_sequential_operations = has_sequential_operations and not args.simulation

    if has_sequential_operations and rank == 0:
        if only_sequential_operations:
            rootLogger.info(
                f"{Fore.CYAN}Running in sequential-only mode for -t/-f/-l operations{Fore.RESET}"
            )
        else:
            rootLogger.info(
                f"{Fore.CYAN}Sequential mode enforced for -t/-f/-l operations{Fore.RESET}"
            )

    # Only setup parallel environment if sequential-only mode is not active
    if not args.disable_parallel and not only_sequential_operations:
        if rank == 0:
            rootLogger.info(f"{Fore.CYAN}Setting up computing environment...{Fore.RESET}")

        # Detect and configure parallel environment
        parallel_config = detect_environment_and_setup_parallel(
            args, proj_input_dir, scenario, hazards_container
        )

        if parallel_config:
            # Store MPI information in parallel config for the MPI backend
            if parallel_config.config["backend"] == "mpi" and comm is not None:
                backend_data = {"MPI": MPI, "comm": comm, "rank": rank, "size": size}
            else:
                # Initialise other backends (only on rank 0 to avoid conflicts)
                if rank == 0:
                    backend_data = initialise_parallel_backend(parallel_config)

            # Store parallel config in scenario for use by other modules
            setattr(scenario, "parallel_config", parallel_config)
            setattr(scenario, "parallel_backend_data", backend_data)

        # Synchronise after parallel setup
        if comm is not None:
            comm.barrier()

    # ---------------------------------------------------------------------------------
    # RECOVERY ANALYSIS
    # ---------------------------------------------------------------------------------
    if args.recovery_analysis:
        if rank == 0:
            print()
            rootLogger.info(f"{Fore.CYAN}Flag set for system recovery approximation...{Fore.RESET}")
        CALC_SYSTEM_RECOVERY_FLAG = True
    else:
        CALC_SYSTEM_RECOVERY_FLAG = False

    # ---------------------------------------------------------------------------------
    # SIMULATION
    # Get the results of running a simulation (all processes participate)
    # ---------------------------------------------------------------------------------
    if args.simulation:
        if rank == 0:
            diagnose_serialisation_issues(scenario, infrastructure, hazards_container)

        # Synchronise before simulation
        if comm is not None:
            comm.barrier()

        response_list = calculate_response(
            hazards_container,
            scenario,
            infrastructure,
            dask_client=backend_data.get("dask_client") if parallel_config else None,
            mpi_comm=backend_data.get("comm") if parallel_config else None,
        )

        # Post simulation processing (only rank 0 to avoid conflicts)
        # For MPI, worker ranks may return None, so we need to check response_list exists
        if rank == 0 and response_list is not None:
            if getattr(scenario, "stream_results", False):
                # In streaming mode, calculate_response returns a manifest dict rather than the
                # legacy response list structure. Defer heavy in-memory post-processing here.
                # Check if SIRA_STREAM_DIR was used (HPC JobFS override)
                env_stream_dir = os.environ.get("SIRA_STREAM_DIR")
                if env_stream_dir:
                    stream_dir_msg = str(env_stream_dir)
                else:
                    stream_dir_msg = str(getattr(scenario, "stream_dir", config.OUTPUT_DIR))
                rootLogger.info(
                    "Streaming mode: Skipping write_system_response. Per-event artifacts "
                    "are persisted under: %s. A consolidation step will aggregate outputs.",
                    stream_dir_msg,
                )
                # Optionally: write a small pointer file for downstream tooling
                try:
                    from pathlib import Path as _Path

                    _Path(config.OUTPUT_DIR, "STREAMING_MANIFEST.txt").write_text(
                        f"stream_dir={stream_dir_msg}\n",
                        encoding="utf-8",
                    )
                except Exception:
                    pass

                # Allow HPC workflows to defer consolidation (stage-out from per-node storage first)
                defer_consolidation = os.environ.get("SIRA_DEFER_CONSOLIDATION", "0") == "1"
                if defer_consolidation:
                    rootLogger.info(
                        "SIRA_DEFER_CONSOLIDATION=1 set. Skipping in-process consolidation. "
                        "A separate post-stage step should call consolidate_streamed_results()."
                    )
                else:
                    # Immediately consolidate streamed artifacts into final CSVs
                    rootLogger.info("Starting streaming consolidation to generate output files...")
                    try:
                        consolidate_streamed_results(
                            stream_dir_msg,
                            infrastructure,
                            scenario,
                            config,
                            hazards_container,
                            CALC_SYSTEM_RECOVERY=CALC_SYSTEM_RECOVERY_FLAG,
                        )
                        rootLogger.info("Streaming consolidation completed successfully")
                    except Exception as e:
                        rootLogger.error(f"Streaming consolidation failed: {e}")
                        import traceback

                        rootLogger.debug(traceback.format_exc())
            else:
                write_system_response(
                    response_list,
                    infrastructure,
                    scenario,
                    config,
                    hazards_container,
                    CALC_SYSTEM_RECOVERY=CALC_SYSTEM_RECOVERY_FLAG,
                )
                # Skip memory-heavy component-class exceedance calculation when streaming
                exceedance_prob_by_component_class(
                    response_list, infrastructure, scenario, hazards_container
                )

            print("\n")
            rootLogger.info("Hazard impact simulation completed...")

    if args.draw_topology and rank == 0:
        # Construct visualisation for system topology (only on rank 0)
        rootLogger.info(f"{Fore.CYAN}Drawing system topology...{Fore.RESET}")
        sys_topology_view = SystemTopologyGenerator(infrastructure, output_path)
        sys_topology_view.draw_sys_topology()

    # ---------------------------------------------------------------------------------
    # FIT MODEL to simulation output data (only on rank 0, always sequential)
    # ---------------------------------------------------------------------------------
    if args.fit and rank == 0:
        args.pe_sys = None
        existing_models = [
            "potablewatertreatmentplant",
            "pwtp",
            "wastewatertreatmentplant",
            "wwtp",
            "watertreatmentplant",
            "wtp",
            "powerstation",
            "substation",
            "potablewaterpumpstation",
            "modelteststructure",
        ]

        if (
            infrastructure.system_class is not None
            and infrastructure.system_class.lower() == "substation"
        ):
            args.pe_sys = os.path.join(config.RAW_OUTPUT_DIR, "pe_sys_cpfailrate.npy")

        elif (
            infrastructure.system_class is not None
            and infrastructure.system_class.lower() in existing_models
        ):
            args.pe_sys = os.path.join(config.RAW_OUTPUT_DIR, "pe_sys_econloss.npy")

        config_data_dict = dict(
            model_name=config.MODEL_NAME,
            x_param=config.HAZARD_INTENSITY_MEASURE_PARAM,
            x_unit=config.HAZARD_INTENSITY_MEASURE_UNIT,
            scenario_metrics=config.FOCAL_HAZARD_SCENARIOS,
            scneario_names=config.FOCAL_HAZARD_SCENARIO_NAMES,
        )

        rootLogger.info(
            f"{Fore.CYAN}Initiating model fitting for simulated system "
            f"fragility data...{Fore.RESET}"
        )
        if args.pe_sys is not None:
            hazard_events = hazards_container.hazard_intensity_list
            sys_limit_states = infrastructure.get_system_damage_states()
            pe_sys = np.load(args.pe_sys)

            print()
            rootLogger.info(f"Infrastructure type: {infrastructure.system_class}")
            rootLogger.info(f"System Limit States: {sys_limit_states}")
            rootLogger.info(
                f"System Limit State Bounds: {infrastructure.get_system_damage_state_bounds()}"
            )

            # Calculate & Plot Fitted Models:
            fit_prob_exceed_model(
                hazard_events,
                pe_sys,
                sys_limit_states,
                config_data_dict,
                output_path=config.OUTPUT_DIR,
            )
            rootLogger.info("Model fitting complete.")
        else:
            rootLogger.error(f"Input pe_sys file not found: {str(output_path)}")

    # ---------------------------------------------------------------------------------
    # SCENARIO LOSS ANALYSIS (only on rank 0)
    # ---------------------------------------------------------------------------------
    if args.loss_analysis and rank == 0:
        print()
        rootLogger.info(f"{Fore.CYAN}Calculating system loss metrics...{Fore.RESET}")

        args.ct = Path(config.OUTPUT_DIR, "comptype_response.csv")
        args.cp = Path(config.OUTPUT_DIR, "component_response.csv")

        if config.FOCAL_HAZARD_SCENARIOS:
            if args.ct is not None and args.cp is not None:
                run_scenario_loss_analysis(
                    scenario, hazards_container, infrastructure, config, args.ct, args.cp
                )
            else:
                if args.ct is None:
                    rootLogger.error(f"Input files not found: {' ' * 9}\n{str(args.ct)}")
                if args.cp is None:
                    rootLogger.error(f"Input files not found: {' ' * 9}\n{str(args.cp)}")

    # Synchronise all processes before cleanup
    if comm is not None:
        comm.barrier()

    # ---------------------------------------------------------------------------------
    # Cleanup parallel resources (only on rank 0)
    # ---------------------------------------------------------------------------------
    if rank == 0:
        if backend_data.get("dask_manager"):
            backend_data["dask_manager"].close()
            rootLogger.info("Dask client closed")

        # MPI finalisation is handled automatically by mpi4py at exit

        # ---------------------------------------------------------------------------------
        rootLogger.info("RUN COMPLETE.")
        print("-" * 80)

        rootLogger.info(f"Config file name  : {str(Path(config_file_path).name)}")
        rootLogger.info(f"Model  file name  : {str(Path(model_file_path.name))}")

        if config.HAZARD_INPUT_METHOD in ["hazard_file", "scenario_file"]:
            scnfile_wrap = utils.wrap_file_path(str(config.HAZARD_INPUT_FILE), max_width=95)
            rootLogger.info(f"Hazard input file : \n{scnfile_wrap}")

        outfolder_wrapped = utils.wrap_file_path(str(output_path))
        rootLogger.info(f"Outputs saved in  : \n{Fore.YELLOW}{outfolder_wrapped}{Fore.RESET}\n")

        completion_time = time()
        completion_time_strf = strftime("%Y-%m-%d %H:%M:%S", localtime(completion_time))
        print("-" * 80)
        rootLogger.info(f"{Fore.GREEN}Simulation completed at : {completion_time_strf}{Fore.RESET}")
        run_duration = timedelta(seconds=completion_time - start_time)
        rootLogger.info(f"{Fore.GREEN}Run time : {run_duration}\n{Fore.RESET}")

        # Report parallel computing statistics if available
        if parallel_config:
            print("-" * 80)
            rootLogger.info(f"{Fore.CYAN}Parallel Computing Summary:{Fore.RESET}")
            rootLogger.info(f"  Backend used: {parallel_config.config['backend']}")

            if parallel_config.config["backend"] == "mpi":
                size = backend_data.get("size", 1)
                rootLogger.info(f"  MPI processes: {size}")
            elif parallel_config.config["backend"] == "dask":
                n_workers = parallel_config.config.get("dask_n_workers", "auto")
                rootLogger.info(f"  Dask workers: {n_workers}")
            elif parallel_config.config["backend"] == "multiprocessing":
                n_procs = parallel_config.config.get("mp_n_processes", "auto")
                rootLogger.info(f"  Processes: {n_procs}")

            # Check for recovery analysis time
            recovery_time = getattr(scenario, "recovery_analysis_time", None)
            if recovery_time is not None and recovery_time > 0:
                rootLogger.info(f"  Recovery analysis time: {recovery_time:.2f} seconds")

    # ------------------------------------------------------------------------------
    # MPI cleanup is handled automatically by mpi4py at exit
    # ------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
