#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
title        : __main__.py
description  : entry point for core sira component
usage        : python sira [OPTIONS]
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

python_version  : 3.7
"""

import argparse
import logging
import logging.config
import os
import re
import sys
import time
from pathlib import Path
# from typing import IO

import numpy as np
from colorama import Fore, init

# Add the source dir to system path
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

# Import SIRA modules
from sira.configuration import Configuration
from sira.infrastructure_response import (pe_by_component_class,
                                          plot_mean_econ_loss,
                                          write_system_response)
from sira.logger import configure_logger
from sira.loss_analysis import run_scenario_loss_analysis
from sira.model_ingest import ingest_model
from sira.modelling.hazard import HazardsContainer
from sira.modelling.system_topology import SystemTopologyGenerator
from sira.scenario import Scenario
from sira.simulation import calculate_response
from sira.fit_model import fit_prob_exceed_model

init()
np.seterr(divide='print', invalid='raise')


def main():

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
    # Tell the code what tasks to do

    parser.add_argument(
        "-s", "--simulation", action='store_true', default=False)
    parser.add_argument(
        "-f", "--fit", action='store_true', default=False)
    parser.add_argument(
        "-l", "--loss_analysis", action='store_true', default=False)

    parser.add_argument(
        "-v", "--verbose", dest="loglevel", type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default="INFO",
        help="Choose option for logging level from: \n"
             "DEBUG, INFO, WARNING, ERROR, CRITICAL.")

    args = parser.parse_args()

    # ------------------------------------------------------------------------------
    # # error handling
    # if args.input_directory and (args.config_file or args.model_file):
    #     parser.error("--input_directory and [--config_file and --model_file]"
    #                  " are mutually exclusive ...")
    #     sys.exit(2)

    # error handling
    if not any([args.simulation, args.fit, args.loss_analysis]):
        parser.error(
            "\nAt least one of these three flags is required:\n"
            " --simulation (-s) or --fit (-f) or --loss_analysis (-s).\n"
            " The options for fit or loss_analysis requires the -s flag, "
            " or a previous completed run with the -s flag.")
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

    # --------------------------------------------------------------------
    # Define paths for CONFIG and MODEL

    config_file_path = Path(proj_input_dir, config_file_name).resolve()
    model_file_path = Path(proj_input_dir, model_file_name).resolve()

    # ------------------------------------------------------------------------------
    # Check output path

    output_path = Path(args.input_directory, "output").resolve()
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    except (FileNotFoundError, OSError):
        parser.error(
            "Unable to create output folder " + str(output_path) + " ...")
        sys.exit(2)

    # ------------------------------------------------------------------------------
    # Set up logging
    # ------------------------------------------------------------------------------
    timestamp = time.strftime('%Y.%m.%d %H:%M:%S')
    log_path = os.path.join(output_path, "log.txt")
    configure_logger(log_path, args.loglevel)
    rootLogger = logging.getLogger(__name__)
    print("\n")
    rootLogger.info(
        Fore.GREEN + 'Simulation initiated at: {}\n'.format(timestamp) + Fore.RESET
    )

    # ------------------------------------------------------------------------------
    # Configure simulation model.
    # Read data and control parameters and construct objects.
    # ------------------------------------------------------------------------------
    config = Configuration(str(config_file_path), str(model_file_path), str(output_path))
    scenario = Scenario(config)
    infrastructure = ingest_model(config)
    hazards = HazardsContainer(config, model_file_path)

    # ------------------------------------------------------------------------------
    # SIMULATION
    # Get the results of running a simulation
    # ------------------------------------------------------------------------------
    # response_list = [
    #     {},  # [0] hazard level vs component damage state index
    #     {},  # [1] hazard level vs infrastructure output
    #     {},  # [2] hazard level vs component response
    #     {},  # [3] hazard level vs component type response
    #     [],  # [4] array of infrastructure output per sample
    #     [],  # [5] array of infrastructure econ loss per sample
    #     {},  # [6] hazard level vs component class dmg level pct
    #     {}]  # [7] hazard level vs component class expected damage index
    if args.simulation:

        response_list = calculate_response(hazards, scenario, infrastructure)

        # ---------------------------------------------------------------------
        # Post simulation processing.
        # After the simulation has run the results are aggregated, saved
        # and the system fragility is calculated.
        # ---------------------------------------------------------------------
        write_system_response(response_list, infrastructure, scenario, hazards)
        economic_loss_array = response_list[5]
        plot_mean_econ_loss(scenario, economic_loss_array, hazards)

        if str(config.HAZARD_INPUT_METHOD).lower() == "calculated_array":
            pe_by_component_class(
                response_list, infrastructure, scenario, hazards)

        # ---------------------------------------------------------------------
        # Visualizations
        # Construct visualization for system topology
        # ---------------------------------------------------------------------
        sys_topology_view = SystemTopologyGenerator(infrastructure, output_path)
        sys_topology_view.draw_sys_topology()  # noqa:E1101
        rootLogger.info('Simulation completed...')

    # ------------------------------------------------------------------------------
    # FIT MODEL to simulation output data
    # ------------------------------------------------------------------------------
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

        if infrastructure.system_class.lower() == 'powerstation':
            args.pe_sys = os.path.join(
                config.RAW_OUTPUT_DIR, 'pe_sys_econloss.npy')

        elif infrastructure.system_class.lower() == 'substation':
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

        if args.pe_sys is not None:
            rootLogger.info('Initiating model fitting for simulated system fragility data...')
            hazard_scenarios = hazards.hazard_scenario_list
            sys_limit_states = infrastructure.get_system_damage_states()
            pe_sys = np.load(args.pe_sys)
            # Calculate & Plot Fitted Models:
            fit_prob_exceed_model(
                hazard_scenarios,
                pe_sys,
                sys_limit_states,
                config_data_dict,
                output_path=config.OUTPUT_PATH,
                distribution='normal_cdf'
            )
            rootLogger.info('Model fitting complete.')
        else:
            rootLogger.error("Input pe_sys file not found: %s", str(output_path))

    # ------------------------------------------------------------------------------
    # SCENARIO LOSS ANALYSIS
    # ------------------------------------------------------------------------------
    if args.loss_analysis:

        args.ct = os.path.join(config.OUTPUT_PATH, 'comptype_response.csv')
        args.cp = os.path.join(config.OUTPUT_PATH, 'component_response.csv')

        if args.ct is not None and args.cp is not None:
            run_scenario_loss_analysis(
                scenario, hazards, infrastructure, config, args.ct, args.cp)
        else:
            if args.ct is None:
                rootLogger.error("Input files not found: %s", str(args.ct))
            if args.cp is None:
                rootLogger.error("Input files not found: %s", str(args.cp))

    rootLogger.info("RUN COMPLETE.\n")
    rootLogger.info("Config file used : %s", str(config_file_path))
    rootLogger.info("Model file used  : %s", str(model_file_path))
    rootLogger.info("Outputs saved in : %s%s%s\n",
                    Fore.YELLOW, str(output_path), Fore.RESET)

if __name__ == "__main__":
    main()
