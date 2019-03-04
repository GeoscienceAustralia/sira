#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
title        : __main__.py
description  : entry point for core sifra component
usage        : python sifra [OPTIONS]
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
import pandas as pd

from colorama import init
init()

import os
import argparse

from sifra.logger import configure_logger
import logging
import logging.config

from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.modelling.hazard import HazardsContainer
from sifra.model_ingest import ingest_model
from sifra.simulation import calculate_response
from sifra.modelling.system_topology import SystemTopology
from sifra.infrastructure_response import (
    write_system_response,
    plot_mean_econ_loss,
    pe_by_component_class
    )
from sifra.fit_model import fit_prob_exceed_model

from sifra.loss_analysis import run_scenario_loss_analysis

import numpy as np

def main():

    # define arg parser
    parser = argparse.ArgumentParser(
        prog='sifra', description="run sifra", add_help=True)

    # [Either] Supply config file and model file directly:
    parser.add_argument("-c", "--config_file", type=str)
    parser.add_argument("-m", "--model_file", type=str)
    # [Or] Supply only the directory where the input files reside
    parser.add_argument("-d", "--input_directory", type=str)

    parser.add_argument("-s", "--simulation",
                        action='store_true', default=False)
    parser.add_argument("-f", "--fit",
                        action='store_true', default=False)
    parser.add_argument("-l", "--loss_analysis",
                        action='store_true', default=False)

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

    if not any([args.simulation, args.fit, args.loss_analysis]):
        parser.error("either --simulation or --fit or --loss_analysis"
                     " is required ...")
        sys.exit(2)

    proj_root_dir = args.input_directory
    if os.path.exists(proj_root_dir):
        proj_input_dir = os.path.join(proj_root_dir, "input")

        for fname in os.listdir(proj_input_dir):
            confmatch = re.search(r"^[config].*[.json]$", fname)
            if confmatch is not None:
                config_file_name = confmatch.string
            modelmatch = re.search(r"^[model].*[.json]$", fname)
            if modelmatch is not None:
                model_file_name = modelmatch.string
    else:
        parser.error(
            "Given input directory does not exist: " +
            str(proj_root_dir))
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
    print(args.output )
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
    rootLogger.info('Simulation initiated at: {}\n'.format(timestamp))

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
    #     {},  # hazard level vs component damage state index
    #     {},  # hazard level vs infrastructure output
    #     {},  # hazard level vs component response
    #     {},  # hazard level vs component type response
    #     [],  # array of infrastructure output per sample
    #     [],  # array infrastructure econ loss per sample
    #     {},  # hazard level vs component class dmg level pct
    #     {}]  # hazard level vs component class expected damage index
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
            "substation"
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
    # print("\n\nLoglevel was: {}.\n".format(str(args.loglevel)))


if __name__ == "__main__":
    main()
