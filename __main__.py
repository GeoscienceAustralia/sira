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
import argparse
from sifra.logger import logging, rootLogger
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--setup", type=str,
                        help="Setup file for simulation scenario, and \n"
                             "locations of inputs, outputs, and system model.")
    parser.add_argument("-v", "--verbose",  type=str,
                        help="Choose option for logging level from: \n"
                             "DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    args = parser.parse_args()

    level = logging.DEBUG

    if args.verbose is not None:
        if args.verbose.upper() == "DEBUG":
            level = logging.DEBUG

        elif args.verbose.upper() == "INFO":
            level = logging.INFO

        elif args.verbose.upper() == "WARNING":
            level = logging.WARNING

        elif args.verbose.upper() == "ERROR":
            level = logging.ERROR

        elif args.verbose.upper() == "CRITICAL":
            level = logging.CRITICAL

    rootLogger.set_log_level(level)

    if args.setup is not None:
        rootLogger.info('Simulation initiated...')

        # ---------------------------------------------------------------------
        # Configure simulation model.
        # Read data and control parameters and construct objects.

        config = Configuration(args.setup)
        scenario = Scenario(config)
        hazards = HazardsContainer(config)
        infrastructure = ingest_model(config)

        # ---------------------------------------------------------------------
        # Run simulation.
        # Get the results of running a simulation
        #
        # response_list = [
        #     {},  # hazard level vs component damage state index
        #     {},  # hazard level vs infrastructure output
        #     {},  # hazard level vs component response
        #     {},  # hazard level vs component type response
        #     [],  # array of infrastructure output for each sample
        #     []]  # array infrastructure econ loss for each sample

        response_list = calculate_response(hazards, scenario, infrastructure)

        # ---------------------------------------------------------------------
        # Post simulation processing.
        # After the simulation has run the results are aggregated, saved
        # and the system fragility is calculated.

        write_system_response(response_list, infrastructure, scenario, hazards)
        economic_loss_array = response_list[5]
        plot_mean_econ_loss(scenario, economic_loss_array, hazards)

        if config.HAZARD_INPUT_METHOD == "hazard_array":
            pe_by_component_class(response_list, infrastructure,
                                  scenario, hazards)

        # ---------------------------------------------------------------------
        # Visualizations
        # Construct visualization for system topology

        sys_topology_view = SystemTopology(infrastructure, scenario)
        sys_topology_view.draw_sys_topology(viewcontext="as-built")
        # ---------------------------------------------------------------------

    else:
        print("Input file not found: " + str(args.setup))

    rootLogger.info('End')


if __name__ == "__main__":
    main()
