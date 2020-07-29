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

from __future__ import print_function
import sys
import numpy as np
np.seterr(divide='print', invalid='raise')
import time
import re

from colorama import init, Fore, Back, Style
init()

import os
import argparse

from sira.logger import configure_logger
import logging
import logging.config

from sira.configuration import Configuration
from sira.scenario import Scenario
from sira.modelling.hazard import HazardsContainer
from sira.model_ingest import ingest_model
from sira.simulation import calculate_response
from sira.modelling.system_topology import SystemTopology
from sira.infrastructure_response import (
    write_system_response,
    plot_mean_econ_loss,
    pe_by_component_class
    )
from sira.fit_model import fit_prob_exceed_model

from sira.loss_analysis import run_scenario_loss_analysis

import numpy as np

def main():

    # define arg parser
    parser = argparse.ArgumentParser(
        prog='sira', description="run sira", add_help=True)

    # [Either] Supply config file and model file directly:
    parser.add_argument("-c", "--config_file", type=str)
    parser.add_argument("-m", "--model_file", type=str)

    # [Or] Supply only the directory where the input files reside
    parser.add_argument("-d", "--input_directory", type=str) 

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
        help="Choose option for logging level from: \n"+
             "DEBUG, INFO, WARNING, ERROR, CRITICAL.")

    args = parser.parse_args()

    # error handling
    if args.input_directory and (args.config_file or args.model_file):
        parser.error("--input_directory and [--config_file and --model_file]"
                     " are mutually exclusive ...")
        sys.exit(2)

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

    if config_file_name is None:
        parser.error(
            "Config file not found. "
            "A valid config file name must begin with the term `config`, "
            "and must be a JSON file.\n")
        sys.exit(2)

    if model_file_name is None:
        parser.error(
            "Model file not found. "
            "A valid model file name must begin the term `model`, "
            "and must be a JSON file.\n")
        sys.exit(2)

    args.config_file = os.path.join(proj_input_dir, config_file_name)
    args.model_file = os.path.join(proj_input_dir, model_file_name)
    args.output = os.path.join(args.input_directory, "output")

    if not os.path.isfile(args.config_file):
        parser.error(
            "Unable to locate config file "+str(args.config_file)+" ...")
        sys.exit(2)

    if not os.path.isfile(args.model_file):
        parser.error(
            "Unable to locate model file "+str(args.model_file)+" ...")
        sys.exit(2)

    args.output = os.path.join(
        os.path.dirname(os.path.dirname(args.config_file)), "output")
    try:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
    except Exception:
        parser.error(
            "Unable to create output folder " + str(args.output) + " ...")
        sys.exit(2)

    # ---------------------------------------------------------------------
    # Set up logging
    # ---------------------------------------------------------------------
    timestamp = time.strftime('%Y.%m.%d %H:%M:%S')
    log_path = os.path.join(args.output, "log.txt")
    configure_logger(log_path, args.loglevel)
    rootLogger = logging.getLogger(__name__)
    print("\n")
    rootLogger.info(Fore.GREEN +
                    'Simulation initiated at: {}\n'.format(timestamp) +
                    Fore.RESET)

    # ---------------------------------------------------------------------
    # Configure simulation model.
    # Read data and control parameters and construct objects.
    # ---------------------------------------------------------------------
    config = Configuration(args.config_file, args.model_file, args.output)
    scenario = Scenario(config)
    hazards = HazardsContainer(config)
    infrastructure = ingest_model(config)

    # ---------------------------------------------------------------------
    # SIMULATION
    # Get the results of running a simulation
    # ---------------------------------------------------------------------
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

        if config.HAZARD_INPUT_METHOD == "hazard_array":
            pe_by_component_class(
                response_list, infrastructure, scenario, hazards)

        # ---------------------------------------------------------------------
        # Visualizations
        # Construct visualization for system topology
        # ---------------------------------------------------------------------
        sys_topology_view = SystemTopology(infrastructure, scenario)
        sys_topology_view.draw_sys_topology(viewcontext="as-built")
        rootLogger.info('Simulation completed...')

    # -------------------------------------------------------------------------
    # FIT MODEL ANALYSIS
    # -------------------------------------------------------------------------
    if args.fit:

        args.pe_sys = None
        existing_models = [
            "potablewatertreatmentplant", "pwtp",
            "wastewatertreatmentplant", "wwtp",
            "watertreatmentplant", "wtp",
            "powerstation",
            "substation",
            "potablewaterpumpstation"
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
            rootLogger.error("Input  pe_sys file not found: " +
                             str(args.output))

    # -------------------------------------------------------------------------
    # SCENARIO LOSS ANALYSIS
    # -------------------------------------------------------------------------
    if args.loss_analysis:

        args.ct = os.path.join(config.OUTPUT_PATH, 'comptype_response.csv')
        args.cp = os.path.join(config.OUTPUT_PATH, 'component_response.csv')

        if args.ct is not None and args.cp is not None:
            run_scenario_loss_analysis(
                scenario, hazards, infrastructure, config, args.ct, args.cp)
        else:
            if args.ct is None:
                rootLogger.error("Input files not found: " + str(args.ct))
            if args.cp is None:
                rootLogger.error("Input files not found: " + str(args.cp))

    rootLogger.info('RUN COMPLETE.\n')
    rootLogger.info("Config file used : " + args.config_file)
    rootLogger.info("Model file used  : " + args.model_file)
    rootLogger.info("Outputs saved in : " +
                    Fore.YELLOW + args.output + Fore.RESET + '\n')

if __name__ == "__main__":
    main()
