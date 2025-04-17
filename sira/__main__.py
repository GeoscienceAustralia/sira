#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
title:        __main__.py
description:  entry point for core sira component
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

python_version  : 3.11
"""

import argparse
import logging
import logging.config
import os
import re
import sys
from time import time, strftime, localtime
from datetime import timedelta
from pathlib import Path
import numpy as np
from colorama import Fore, init

# Add the source dir to system path
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

# Import SIRA modules
from sira.configuration import Configuration
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

init()
np.seterr(divide='print', invalid='raise')


def main():

    # ------------------------------------------------------------------------------
    # define arg parser
    parser = argparse.ArgumentParser(
        prog='sira', description="run sira", add_help=True)

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
    # parser.add_argument(
    #     "-x", "--donot_draw_topology", action='store_true', default=False)
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

    args = parser.parse_args()

    # ------------------------------------------------------------------------------
    # # error handling
    # if args.input_directory and (args.config_file or args.model_file):
    #     parser.error("--input_directory and [--config_file and --model_file]"
    #                  " are mutually exclusive ...")
    #     sys.exit(2)
    # ------------------------------------------------------------------------------

    # error handling
    if not any([args.simulation, args.fit, args.loss_analysis, args.draw_topology]):
        parser.error(
            "\nOne of these flags is required:\n"
            " --simulation (-s) or \n"
            " --fit (-f) or \n"
            " --loss_analysis (-s) or \n"
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
    rootLogger = logging.getLogger(__name__)
    print("\n")
    rootLogger.info(
        f"{Fore.GREEN}Simulation initiated at: {start_time_strf}{Fore.RESET}\n")
    print("-" * 80)

    # ---------------------------------------------------------------------------------
    # SIMULATION
    # Get the results of running a simulation
    #
    # response_list = [
    #     {},  # [0] hazard level vs component damage state index
    #     {},  # [1] hazard level vs infrastructure output
    #     {},  # [2] hazard level vs component response
    #     {},  # [3] hazard level vs component type response
    #     [],  # [4] array of infrastructure output per sample
    #     [],  # [5] array of infrastructure econ loss per sample
    #     {},  # [6] hazard level vs component class dmg level pct
    #     {}]  # [7] hazard level vs component class expected damage index
    # ---------------------------------------------------------------------------------
    if args.simulation:

        response_list = calculate_response(hazards_container, scenario, infrastructure)

        # Post simulation processing.
        # After the simulation has run the results are aggregated, saved
        # and the system fragility is calculated.
        pe_sys_econloss = write_system_response(
            response_list, infrastructure, scenario, hazards_container)

        # if str(config.HAZARD_INPUT_METHOD).lower() == "calculated_array":
        exceedance_prob_by_component_class(
            response_list, infrastructure, scenario, hazards_container)
        print("\n")
        rootLogger.info('Hazard impact simulation completed...')

    if args.draw_topology:

        # Construct visualization for system topology
        rootLogger.info(
            f"{Fore.CYAN}Attempting to draw system topology ...{Fore.RESET}")
        sys_topology_view = SystemTopologyGenerator(infrastructure, output_path)
        sys_topology_view.draw_sys_topology()  # noqa:E1101

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

        if infrastructure.system_class.lower() == 'substation':
            args.pe_sys = os.path.join(
                config.RAW_OUTPUT_DIR, 'pe_sys_cpfailrate.npy')

        elif infrastructure.system_class.lower() in existing_models:
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
    # ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
