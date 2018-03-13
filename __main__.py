#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
title           : __main__.py
description     : entry point for core sifra component
usage           : python sifra  [OPTIONS]
                  -f                    Display this usage message
                  -c config             Hostname to connect to
                  -l [LEVEL]            Choose logging level DEBUG, INFO, WARNING, ERROR, CRITICAL

python_version  : 2.7
"""
import argparse
import os
import logging
from sifra.logger import rootLogger
from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.model_ingest import ingest_model
from sifra.modelling.system_topology import SystemTopology
from sifra.infrastructure_response import calculate_response, post_processing


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--ifile", type=str,
                        help="choose option for logging level from: DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    parser.add_argument("-v", "--verbose",  type=str,
                        help="choose option for logging level from: DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    args = parser.parse_args()

    level = logging.INFO

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

    rootLogger.info('Start')

    if args.ifile is not None:
        configuration_file_path = args.ifile
        config = Configuration(configuration_file_path)
        scenario = Scenario(config)
        infrastructure, algorithm_factory = ingest_model(config)
        scenario.algorithm_factory = algorithm_factory
        sys_topology_view = SystemTopology(infrastructure, scenario)
        sys_topology_view.draw_sys_topology(viewcontext="as-built")
        post_processing_list = calculate_response(scenario, infrastructure)
        post_processing(infrastructure, scenario, post_processing_list)
    else:
        print("Input file not found: " + args.ifile )


    rootLogger.info('End')


if __name__ == "__main__":
    main()
