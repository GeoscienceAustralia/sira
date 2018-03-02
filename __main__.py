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

import logging
import time
import argparse

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

logPath='sifra/logs'
time_start = time.strftime("%Y%m%d-%H%M%S")
fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, time_start))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

# from sifra.infrastructure_response import run_scenario

def main():

    # time_start = time.strftime("%Y%m%d-%H%M%S")
    time_start = "00"

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",  type=str,
                        help="choose option for logging level from: DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    args = parser.parse_args()

    if args.verbose is not None:
        if args.verbose.upper() == "DEBUG":
            rootLogger.setLevel(logging.DEBUG)

        elif args.verbose.upper() == "INFO":
            rootLogger.setLevel(logging.INFO)

        elif args.verbose.upper() == "WARNING":
            rootLogger.setLevel(logging.WARNING)

        elif args.verbose.upper() == "ERROR":
            rootLogger.setLevel(logging.ERROR)

        elif args.verbose.upper() == "CRITICAL":
            rootLogger.setLevel(logging.CRITICAL)
    else:
        # default option
        rootLogger.setLevel(logging.INFO)

    logging.info('Start')
    SETUPFILE = "C:\\Users\\u12089\\Desktop\\sifra-v0.2.0\\tests\\test_simple_series_struct_dep.conf"
    # run_scenario(SETUPFILE)

    logging.info('End')


if __name__ == "__main__":

    main()
    # SETUPFILE = "C:\\Users\\u12089\\Desktop\\sifra-v0.2.0\\simulation_setup\\test_scenario_ps_coal.conf"
