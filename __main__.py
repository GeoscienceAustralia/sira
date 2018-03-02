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
from sifra.logger import rootLogger
import logging

from sifra.infrastructure_response import run_scenario


def main():

    parser = argparse.ArgumentParser()
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

    SETUPFILE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "tests/test_scenario_ps_coal.conf"))
    run_scenario(SETUPFILE)
    # setup_file = "C:\\Users\\u12089\\Desktop\\sifra-v0.2.0\\tests\\test_scenario_ps_coal.conf"

    run_scenario(setup_file)

    rootLogger.info('End')


if __name__ == "__main__":

    main()
