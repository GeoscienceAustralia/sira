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
rootLogger = logging.getLogger(__name__)

import os
import re
import sys
from time import time, strftime, localtime
from datetime import timedelta
from pathlib import Path
import numpy as np
from colorama import Fore, init
from sympy import root

# Add the source dir to system path
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

# Import SIRA modules
from sira.configuration import Configuration
rootLogger.info("Imported Configuration class")
from sira.infrastructure_response import (
    exceedance_prob_by_component_class,
    write_system_response)
from sira.logger import configure_logger
from sira.loss_analysis import run_scenario_loss_analysis
from sira.model_ingest import ingest_model
from sira.simulation import calculate_response
from sira.fit_model import fit_prob_exceed_model
from sira.modelling.hazard import HazardsContainer
from sira.modelling.system_topology import SystemTopologyGenerator
from sira.scenario import Scenario
from sira.tools import utils

# Import parallel computing modules
from sira.parallel_config import ParallelConfig, setup_parallel_environment

init()
np.seterr(divide='print', invalid='raise')

# ==============================================================================

# Add this after importing modules in __main__.py
def diagnose_serialisation_issues(scenario, infrastructure, hazards_container):
    """Diagnose what's preventing multiprocessing from working."""
    import pickle

    print("Diagnosing serialisation issues...")

    # Test scenario
    try:
        pickle.dumps(scenario)
        print("✓ Scenario object is picklable")
    except Exception as e:
        print(f"✗ Scenario object is NOT picklable: {e}")

    # Test infrastructure
    try:
        pickle.dumps(infrastructure)
        print("✓ Infrastructure object is picklable")
    except Exception as e:
        print(f"✗ Infrastructure object is NOT picklable: {e}")

    # Test first hazard
    hazard_list = list(hazards_container.listOfhazards)
    if hazard_list:
        try:
            pickle.dumps(hazard_list[0])
            print("✓ Hazard object is picklable")
        except Exception as e:
            print(f"✗ Hazard object is NOT picklable: {e}")

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
    slurm_vars = ['SLURM_JOB_ID', 'SLURM_NTASKS', 'SLURM_PROCID', 'SLURM_NODELIST']
    if any(var in os.environ for var in slurm_vars):
        rootLogger.info("Detected SLURM environment - MPI available")
        return True

    # Check for PBS/Torque (another common HPC scheduler)
    pbs_vars = ['PBS_JOBID', 'PBS_NCPUS', 'PBS_NODEFILE']
    if any(var in os.environ for var in pbs_vars):
        rootLogger.info("Detected PBS/Torque environment - MPI available")
        return True

    # Check for explicit MPI runtime variables
    mpi_runtime_vars = [
        'OMPI_COMM_WORLD_SIZE', 'OMPI_COMM_WORLD_RANK',  # OpenMPI
        'PMI_SIZE', 'PMI_RANK',                          # MPICH
        'MPI_LOCALRANKID', 'MPI_LOCALNRANKS'             # Intel MPI
    ]
    if any(var in os.environ for var in mpi_runtime_vars):
        rootLogger.info("Detected MPI runtime environment variables - MPI available")
        return True

    # Check if we're being launched with mpirun/mpiexec
    parent_process = os.environ.get('_', '')
    if any(launcher in parent_process for launcher in ['mpirun', 'mpiexec', 'srun']):
        rootLogger.info("Detected MPI launcher in parent process - MPI available")
        return True

    # Check for HPC-specific hostnames or environments
    hostname = os.environ.get('HOSTNAME', '')
    if any(pattern in hostname.lower() for pattern in ['hpc', 'cluster', 'node', 'compute']):
        rootLogger.info(f"Detected HPC-like hostname: {hostname} - MPI may be available")
        return True

    # Check if we're in a container with MPI setup
    if os.path.exists('/.dockerenv') or os.path.exists('/singularity'):
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
        os.environ['MPI4PY_RC_INITIALIZE'] = 'False'

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


def detect_environment_and_setup_parallel(args, scenario, hazards_container):
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

    if args.parallel_backend == 'mpi' and not in_mpi_env:
        rootLogger.warning("MPI backend requested but not in MPI environment. Switching to auto.")
        args.parallel_backend = 'auto'

    # Determine scenario size if auto
    if args.scenario_size == 'auto':
        num_events = len(hazards_container.hazard_scenario_list)
        num_samples = scenario.num_samples
        total_computations = num_events * num_samples

        if total_computations < 1000:
            scenario_size = 'small'
        elif total_computations < 10000:
            scenario_size = 'medium'
        elif total_computations < 100000:
            scenario_size = 'large'
        else:
            scenario_size = 'xlarge'

        rootLogger.info(
            f"Auto-detected scenario size: {scenario_size} "
            f"({num_events} events {chr(0x00D7)} {num_samples} samples = {total_computations:,} computations)"
        )
    else:
        scenario_size = args.scenario_size

    # Setup parallel configuration
    if args.parallel_config and Path(args.parallel_config).exists():
        # Load from file
        rootLogger.info(f"Loading parallel config from: {args.parallel_config}")
        config = ParallelConfig(config_file=Path(args.parallel_config))
    else:
        # Auto-configure
        rootLogger.info("No parallel config file provided, attempting auto-configure")
        config = setup_parallel_environment(
            scenario_size=scenario_size,
            verbose=True
        )

    # Override with command line options
    if args.parallel_backend != 'auto':
        if args.parallel_backend == 'mpi' and not in_mpi_env:
            rootLogger.warning("MPI backend forced but environment not suitable. Using Dask instead.")
            config.config['backend'] = 'dask'
        else:
            config.config['backend'] = args.parallel_backend
            rootLogger.info(f"Overriding backend to: {args.parallel_backend}")

    if args.max_workers:
        if config.config['backend'] == 'dask':
            config.config['dask_n_workers'] = args.max_workers
        elif config.config['backend'] == 'multiprocessing':
            config.config['mp_n_processes'] = args.max_workers
        rootLogger.info(f"Setting max workers to: {args.max_workers}")

    # Apply scenario-specific settings
    if hasattr(scenario, 'recovery_method'):
        config.config['recovery_method'] = scenario.recovery_method
    if hasattr(scenario, 'num_repair_streams'):
        config.config['num_repair_streams'] = scenario.num_repair_streams

    # Enable checkpointing if requested
    if args.checkpoint:
        config.config['save_checkpoints'] = True
        config.config['checkpoint_dir'] = Path(scenario.output_path, 'checkpoints')
        config.config['checkpoint_dir'].mkdir(exist_ok=True)
        rootLogger.info(f"Checkpointing enabled: {config.config['checkpoint_dir']}")

    # Save configuration for reference
    config_save_path = Path(scenario.output_path, 'parallel_config.json')
    config.save_config(config_save_path)
    config_save_path_wrapped = utils.wrap_file_path(str(config_save_path))
    rootLogger.info(
        f"Saved parallel configuration to: \n"
        f"{Fore.YELLOW}{config_save_path_wrapped}{Fore.RESET}")

    return config


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

    if para_config.config['backend'] == 'mpi':
        # Use safe MPI import
        MPI, comm, rank, size = safe_mpi_import()

        if MPI is not None:
            backend_data['MPI'] = MPI
            backend_data['comm'] = comm
            backend_data['rank'] = rank
            backend_data['size'] = size

            if rank == 0:
                rootLogger.info(f"MPI initialised with {size} processes")

            # Only rank 0 should continue with normal execution
            if rank != 0:
                backend_data['is_worker'] = True
        else:
            rootLogger.warning("MPI initialisation failed, falling back to Dask")
            para_config.config['backend'] = 'dask'

    if para_config.config['backend'] == 'dask':
        from sira.infrastructure_response import DaskClientManager

        dask_manager = DaskClientManager()
        backend_data['dask_client'] = dask_manager.client
        backend_data['dask_manager'] = dask_manager

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
    # ------------------------------------------------------------------------------
    # define arg parser
    parser = argparse.ArgumentParser(
        prog='sira', description="run sira with parallel computing support", add_help=True)

    # ------------------------------------------------------------------------------
    # [Either] Supply config file and model file directly:
    parser.add_argument("-c", "--config_file", type=str)
    parser.add_argument("-m", "--model_file", type=str)

    # [Or] Supply only the directory where the input files reside
    parser.add_argument("-d", "--input_directory", type=str)

    # ------------------------------------------------------------------------------
    # Define the sim arguments - tell the code what tasks to do

    VERBOSITY_CHOICES = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    parser.add_argument(
        "-s", "--simulation", action='store_true', default=False)
    parser.add_argument(
        "-t", "--draw_topology", action='store_true', default=False)
    parser.add_argument(
        "-f", "--fit", action='store_true', default=False)
    parser.add_argument(
        "-l", "--loss_analysis", action='store_true', default=False)
    parser.add_argument(
        "-v", "--verbose",
        dest="loglevel",
        type=str,
        choices=VERBOSITY_CHOICES,
        default="INFO",
        help=f"Choose one of these options for logging: \n{VERBOSITY_CHOICES}"
    )

    # ------------------------------------------------------------------------------
    # Parallel computing arguments

    parser.add_argument(
        "--parallel-backend",
        type=str,
        choices=['auto', 'mpi', 'dask', 'multiprocessing'],
        default='auto',
        help="Choose parallel computing backend (default: auto)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers/processes"
    )

    parser.add_argument(
        "--scenario-size",
        type=str,
        choices=['auto', 'small', 'medium', 'large', 'xlarge'],
        default='auto',
        help="Scenario size for optimisation (default: auto-detect)"
    )

    parser.add_argument(
        "--disable-parallel",
        action='store_true',
        default=False,
        help="Disable parallel processing"
    )

    parser.add_argument(
        "--checkpoint",
        action='store_true',
        default=False,
        help="Enable checkpointing for recovery analysis"
    )

    parser.add_argument(
        "--parallel-config",
        type=str,
        default=None,
        help="Path to parallel configuration JSON file"
    )

    # Parse arguments
    if args is None:
        args = parser.parse_args()

    rootLogger.info(f"args: {args}")

    # ------------------------------------------------------------------------------
    # Early MPI handling - check if we're a worker process BEFORE any imports

    if args.parallel_backend in ['auto', 'mpi'] and is_mpi_environment():
        MPI, comm, rank, size = safe_mpi_import()

        if MPI is not None and rank != 0:
            # Worker processes should wait for tasks
            from sira.recovery_analysis import RecoveryAnalysisEngine
            # Worker loop - this will be controlled by the master process
            # The actual implementation is handled within the analysis functions
            return

    # ------------------------------------------------------------------------------
    # error handling

    if not any([args.simulation, args.fit, args.loss_analysis, args.draw_topology]):
        parser.error(
            "\nOne of these flags is required:\n"
            " --simulation (-s) or \n"
            " --fit (-f) or \n"
            " --loss_analysis (-l) or \n"
            " --draw_topology (-t).\n"
            " The options for fit or loss_analysis requires the -s flag,\n"
            " or a previously completed run with the -s flag.\n")
        sys.exit(2)

    proj_root_dir = args.input_directory

    if not os.path.isdir(proj_root_dir):
        print("Invalid path supplied:\n {}".format(proj_root_dir))
        sys.exit(1)

    # ------------------------------------------------------------------------------

    class InvalidOrMissingInputFile(Exception):
        def __init__(self, type_of_file):
            super(InvalidOrMissingInputFile, self).__init__()
            raise NameError(
                f"\n{str(type_of_file)} file error: invalid or missing input file."
                "\n  A valid model file name must begin the term `model`, "
                "\n  A valid config file name must begin with the term `config`, and"
                "\n  It must be in JSON format.\n")

    # ------------------------------------------------------------------------------
    # Locate input files

    try:
        proj_input_dir = Path(proj_root_dir, "input").resolve()
    except (IOError, OSError):
        raise IOError("Invalid path")

    if not Path(proj_input_dir).exists():
        raise IOError("Invalid path")

    config_file_name = None
    model_file_name = None

    for fname in os.listdir(proj_input_dir):

        confmatch = re.search(r"(?i)^config.*\.json$", fname)
        if confmatch is not None:
            config_file_name = confmatch.string

        modelmatch = re.search(r"(?i)^model.*\.json$", fname)
        if modelmatch is not None:
            model_file_name = modelmatch.string

    if config_file_name is None:
        raise InvalidOrMissingInputFile("CONFIG")

    if model_file_name is None:
        raise InvalidOrMissingInputFile("MODEL")

    # ------------------------------------------------------------------------------
    # Define paths for CONFIG and MODEL

    config_file_path = Path(proj_input_dir, config_file_name).resolve()
    model_file_path = Path(proj_input_dir, model_file_name).resolve()

    # ---------------------------------------------------------------------------------
    # Configure simulation model.
    # Read data and control parameters and construct objects.
    # ---------------------------------------------------------------------------------
    config = Configuration(str(config_file_path), str(model_file_path))
    scenario = Scenario(config)
    infrastructure = ingest_model(config)
    hazards_container = HazardsContainer(config, model_file_path)
    output_path = config.OUTPUT_DIR

    # ---------------------------------------------------------------------------------
    # Set up logging
    # ---------------------------------------------------------------------------------
    start_time = time()
    start_time_strf = strftime('%Y-%m-%d %H:%M:%S', localtime(start_time))
    log_path = os.path.join(output_path, "log.txt")
    configure_logger(log_path, args.loglevel)
    print("\n")
    rootLogger.info(
        f"{Fore.GREEN}Simulation initiated at: {start_time_strf}{Fore.RESET}\n")
    print("-" * 80)

    # ---------------------------------------------------------------------------------
    # Setup parallel computing environment
    # ---------------------------------------------------------------------------------

    parallel_config = None
    backend_data = {}

    if not args.disable_parallel:
        rootLogger.info(
            f"{Fore.CYAN}Setting up computing environment...{Fore.RESET}")

        # Detect and configure parallel environment
        parallel_config = detect_environment_and_setup_parallel(
            args, scenario, hazards_container
        )

        if parallel_config:
            # Initialise backend
            backend_data = initialise_parallel_backend(parallel_config)

            # Check if this is a worker process (MPI)
            if backend_data.get('is_worker', False):
                # Worker processes are handled within the parallel functions
                return

            # Store parallel config in scenario for use by other modules
            setattr(scenario, 'parallel_config', parallel_config)
            setattr(scenario, 'parallel_backend_data', backend_data)

    # ---------------------------------------------------------------------------------
    # SIMULATION
    # Get the results of running a simulation
    # ---------------------------------------------------------------------------------
    if args.simulation:
        diagnose_serialisation_issues(scenario, infrastructure, hazards_container)
        response_list = calculate_response(hazards_container, scenario, infrastructure)

        # Post simulation processing with parallel support
        pe_sys_econloss = write_system_response(
            response_list, infrastructure, scenario, hazards_container)
        exceedance_prob_by_component_class(
            response_list, infrastructure, scenario, hazards_container)

        print("\n")
        rootLogger.info('Hazard impact simulation completed...')

    if args.draw_topology:
        # Construct visualisation for system topology
        rootLogger.info(
            f"{Fore.CYAN}Attempting to draw system topology ...{Fore.RESET}")
        sys_topology_view = SystemTopologyGenerator(infrastructure, output_path)
        sys_topology_view.draw_sys_topology()

    # ---------------------------------------------------------------------------------
    # FIT MODEL to simulation output data
    # ---------------------------------------------------------------------------------
    if args.fit:

        args.pe_sys = None
        existing_models = [
            "potablewatertreatmentplant", "pwtp",
            "wastewatertreatmentplant", "wwtp",
            "watertreatmentplant", "wtp",
            "powerstation",
            "substation",
            "potablewaterpumpstation",
            "modelteststructure"
        ]

        if infrastructure.system_class is not None and infrastructure.system_class.lower() == 'substation':
            args.pe_sys = os.path.join(
                config.RAW_OUTPUT_DIR, 'pe_sys_cpfailrate.npy')

        elif infrastructure.system_class is not None and infrastructure.system_class.lower() in existing_models:
            args.pe_sys = os.path.join(
                config.RAW_OUTPUT_DIR, 'pe_sys_econloss.npy')

        config_data_dict = dict(
            model_name=config.MODEL_NAME,
            x_param=config.HAZARD_INTENSITY_MEASURE_PARAM,
            x_unit=config.HAZARD_INTENSITY_MEASURE_UNIT,
            scenario_metrics=config.FOCAL_HAZARD_SCENARIOS,
            scneario_names=config.FOCAL_HAZARD_SCENARIO_NAMES
        )

        rootLogger.info(
            f"{Fore.CYAN}Initiating model fitting for simulated system "
            f"fragility data...{Fore.RESET}")
        if args.pe_sys is not None:
            hazard_events = hazards_container.hazard_intensity_list
            sys_limit_states = infrastructure.get_system_damage_states()
            pe_sys = np.load(args.pe_sys)

            print()
            rootLogger.info(f"Infrastructure type: {infrastructure.system_class}")
            rootLogger.info(f"System Limit States: {sys_limit_states}")
            rootLogger.info(
                f"System Limit State Bounds: "
                f"{infrastructure.get_system_damage_state_bounds()}")

            # Calculate & Plot Fitted Models:
            fit_prob_exceed_model(
                hazard_events,
                pe_sys,
                sys_limit_states,
                config_data_dict,
                output_path=config.OUTPUT_DIR
            )
            rootLogger.info('Model fitting complete.')
        else:
            rootLogger.error(f"Input pe_sys file not found: {str(output_path)}")

    # ---------------------------------------------------------------------------------
    # SCENARIO LOSS ANALYSIS
    # ---------------------------------------------------------------------------------
    if args.loss_analysis:

        print()
        rootLogger.info(f"{Fore.CYAN}Calculating system loss metrics...{Fore.RESET}")

        args.ct = Path(config.OUTPUT_DIR, 'comptype_response.csv')
        args.cp = Path(config.OUTPUT_DIR, 'component_response.csv')

        if config.FOCAL_HAZARD_SCENARIOS:
            if args.ct is not None and args.cp is not None:
                run_scenario_loss_analysis(
                    scenario, hazards_container, infrastructure,
                    config, args.ct, args.cp)
            else:
                if args.ct is None:
                    rootLogger.error(
                        f"Input files not found: {' ' * 9}\n{str(args.ct)}")
                if args.cp is None:
                    rootLogger.error(
                        f"Input files not found: {' ' * 9}\n{str(args.cp)}")

    # ---------------------------------------------------------------------------------
    # Cleanup parallel resources
    # ---------------------------------------------------------------------------------

    if backend_data.get('dask_manager'):
        backend_data['dask_manager'].close()
        rootLogger.info("Dask client closed")

    # MPI finalisation is handled automatically by mpi4py at exit

    # ---------------------------------------------------------------------------------
    rootLogger.info("RUN COMPLETE.")
    print("-" * 80)

    rootLogger.info(f"Config file name  : {str(Path(config_file_path).name)}")
    rootLogger.info(f"Model  file name  : {str(Path(model_file_path.name))}")

    if config.HAZARD_INPUT_METHOD in ['hazard_file', 'scenario_file']:
        scnfile_wrap = utils.wrap_file_path(
            str(config.HAZARD_INPUT_FILE), max_width=95)
        rootLogger.info(f"Hazard input file : \n{scnfile_wrap}")

    outfolder_wrapped = utils.wrap_file_path(str(output_path))
    rootLogger.info(
        f"Outputs saved in  : \n{Fore.YELLOW}{outfolder_wrapped}{Fore.RESET}\n")

    completion_time = time()
    completion_time_strf = strftime('%Y-%m-%d %H:%M:%S', localtime(completion_time))
    print("-" * 80)
    rootLogger.info(
        f"{Fore.GREEN}Simulation completed at : {completion_time_strf}{Fore.RESET}")
    run_duration = timedelta(seconds=completion_time - start_time)
    rootLogger.info(
        f"{Fore.GREEN}Run time : {run_duration}\n{Fore.RESET}")

    # Report parallel computing statistics if available
    if parallel_config:
        print("-" * 80)
        rootLogger.info(f"{Fore.CYAN}Parallel Computing Summary:{Fore.RESET}")
        rootLogger.info(f"  Backend used: {parallel_config.config['backend']}")

        if parallel_config.config['backend'] == 'mpi':
            size = backend_data.get('size', 1)
            rootLogger.info(f"  MPI processes: {size}")
        elif parallel_config.config['backend'] == 'dask':
            n_workers = parallel_config.config.get('dask_n_workers', 'auto')
            rootLogger.info(f"  Dask workers: {n_workers}")
        elif parallel_config.config['backend'] == 'multiprocessing':
            n_procs = parallel_config.config.get('mp_n_processes', 'auto')
            rootLogger.info(f"  Processes: {n_procs}")

        # Check for recovery analysis time
        recovery_time = getattr(scenario, 'recovery_analysis_time', None)
        if recovery_time is not None and recovery_time > 0:
            rootLogger.info(f"  Recovery analysis time: {recovery_time:.2f} seconds")

    # ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
