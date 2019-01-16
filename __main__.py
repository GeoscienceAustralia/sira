#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
title           : __main__.py
description     : entry point for core sifra component
usage           : python sifra [OPTIONS]
                  -s                Display this usage message
                  -l [LEVEL]        Choose logging level DEBUG, INFO,
                                    WARNING, ERROR, CRITICAL

python_version  : 2.7
"""
import os
import argparse
from sifra.logger import rootLogger
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
from sifra import fit_model
import numpy as np

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-o", "--output", type=str)

    parser.add_argument("-v", "--verbose",  type=str,
                        help="Choose option for logging level from: \n"
                             "DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    args = parser.parse_args()

    rootLogger.set_log_level(args.verbose)

    if args.config is not None and args.model is not None and args.output is not None:

        rootLogger.set_log_file_path(os.path.join(args.output, "log.txt"))
        rootLogger.info('Simulation initiated...')

        # ---------------------------------------------------------------------
        # Configure simulation model.
        # Read data and control parameters and construct objects.
        # ---------------------------------------------------------------------

        config = Configuration(args.config, args.model, args.output)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)

        # ---------------------------------------------------------------------
        # Run simulation.
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

        response_list = calculate_response(hazards, scenario, infrastructure)

        # ---------------------------------------------------------------------
        # Post simulation processing.
        # After the simulation has run the results are aggregated, saved
        # and the system fragility is calculated.

        write_system_response(response_list, infrastructure, scenario, hazards)
        economic_loss_array = response_list[5]
        plot_mean_econ_loss(scenario, economic_loss_array, hazards)

        if config.HAZARD_INPUT_METHOD == "hazard_array":
            pe_by_component_class(response_list, infrastructure,scenario, hazards)

        # ---------------------------------------------------------------------
        # Visualizations
        # Construct visualization for system topology
        # ---------------------------------------------------------------------
        sys_topology_view = SystemTopology(infrastructure, scenario)
        sys_topology_view.draw_sys_topology(viewcontext="as-built")

        rootLogger.info('Simulation completed...')
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # FIT MODEL ANALYSIS
        # ---------------------------------------------------------------------
        rootLogger.info('Start: FIT MODEL ANALYSIS')

        # hazard_scenarios = hazards.hazard_scenario_list
        # sys_limit_states = infrastructure.get_system_damage_states()
        # FIT_PE_DATA = scenario.fit_pe_data
        #
        #
        # come_models=["potablewatertreatmentplant", "pwtp", "wastewatertreatmentplant", "wwtp", "watertreatmentplant", "wtp"]
        # if infrastructure.system_class.lower() == 'powerstation':
        #     pe_sys = np.load(os.path.join(config.RAW_OUTPUT_DIR, 'pe_sys_econloss.npy'))
        # elif infrastructure.system_class.lower() == 'substation':
        #     pe_sys = np.load(os.path.join(config.RAW_OUTPUT_DIR, 'pe_sys_cpfailrate.npy'))
        #
        # elif infrastructure.system_class.lower() in come_models:
        #     pe_sys = np.load(os.path.join(config.RAW_OUTPUT_DIR, 'pe_sys_econloss.npy'))
        #
        # # --------------------------------------------------------------------------
        # # Calculate & Plot Fitted Models
        #
        # if FIT_PE_DATA:
        #     fit_model.fit_prob_exceed_model(hazard_scenarios, pe_sys, sys_limit_states, config.OUTPUT_PATH, config)

        rootLogger.info('End: FIT MODEL ANALYSIS')

        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # SCENARIO LOSS ANALYSIS
        # ---------------------------------------------------------------------
        rootLogger.info('Start: SCENARIO LOSS ANALYSIS')
        rootLogger.info('End: SCENARIO LOSS ANALYSIS')

        # ---------------------------------------------------------------------

    else:
        print("Input file not found: " + str(args.setup))

    rootLogger.info('End')


if __name__ == "__main__":
    main()
