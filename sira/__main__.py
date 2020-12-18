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

import numpy as np
from colorama import Fore, init

# Add the source dir to system path
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

# Import SIRA modules
from sira.configuration import Configuration
from sira.fit_model import fit_prob_exceed_model
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
    # Locate input files
    proj_input_dir = os.path.join(proj_root_dir, "input")
    config_file_name = None
    model_file_name = None

    for fname in os.listdir(proj_input_dir):
        confmatch = re.search(r"(?i)^config.*\.json$", fname)
        if confmatch is not None:
            config_file_name = confmatch.string
        modelmatch = re.search(r"(?i)^model.*\.json$", fname)
        if modelmatch is not None:
            model_file_name = modelmatch.string

    # ------------------------------------------------------------------------------
    # Test that CONFIG file is identifiable, and is valid
    if config_file_name is None:
        parser.error(
            "Config file name does not meet required naming criteria. "
            "A valid config file name must begin with the term `config`, "
            "and must be a JSON file.\n")
        sys.exit(2)

    config_file_path = Path(proj_input_dir, config_file_name).resolve()
    if not os.path.isfile(config_file_path):
        parser.error(
            "Unable to locate config file " + str(config_file_path) + " ...")
        sys.exit(2)

    # ------------------------------------------------------------------------------
    # Test that MODEL file is identifiable, and is valid
    if model_file_name is None:
        parser.error(
            "Model file name does not meet required naming criteria. "
            "A valid model file name must begin the term `model`, "
            "and must be a JSON file.\n")
        sys.exit(2)

    # ------------------------------------------------------------------------------
    # Check output path
    model_file_path = Path(proj_input_dir, model_file_name).resolve()
    if not os.path.isfile(model_file_path):
        parser.error(
            "Unable to locate model file " + str(model_file_path) + " ...")
        sys.exit(2)

    output_path = Path(args.input_directory, "output").resolve()
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    except Exception:
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
        sys_topology_view.draw_sys_topology()
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

        if args.pe_sys is not None:
            rootLogger.info('Start: Attempting to fit MODEL to simulation '
                            'data...')

            hazard_scenarios = hazards.hazard_scenario_list
            sys_limit_states = infrastructure.get_system_damage_states()
            pe_sys = np.load(args.pe_sys)
            # Calculate & Plot Fitted Models
            fit_prob_exceed_model(
                hazard_scenarios,
                pe_sys,
                sys_limit_states,
                config.OUTPUT_PATH,
                config)

            rootLogger.info('End: Model fitting complete.')
        else:
            rootLogger.error(f"Input  pe_sys file not found: {str(output_path)}")

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
                rootLogger.error("Input files not found: ", str(args.ct))
            if args.cp is None:
                rootLogger.error("Input files not found: ", str(args.cp))

    rootLogger.info('RUN COMPLETE.\n')
    rootLogger.info(f"Config file used : {str(config_file_path)}")
    rootLogger.info(f"Model file used  : {str(model_file_path)}")
    rootLogger.info(f"Outputs saved in : {Fore.YELLOW} {str(output_path)}\
        {Fore.RESET} \n")

if __name__ == "__main__":
    main()
