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

import time
import argparse
from sifra.rootlog import rootLogger


from sifra.infrastructure_response import run_scenario

def main():

    # time_start = time.strftime("%Y%m%d-%H%M%S")
    time_start = "00"

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose",  type=str,
                        help="choose option for logging level from: DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    args = parser.parse_args()

    rootLogger.info('Start')
    SETUPFILE = "C:\\Users\\u12089\\Desktop\\sifra-v0.2.0\\tests\\test_simple_series_struct_dep.conf"
    run_scenario(SETUPFILE)

    rootLogger.info('End')


if __name__ == "__main__":

    main()
    # SETUPFILE = "C:\\Users\\u12089\\Desktop\\sifra-v0.2.0\\simulation_setup\\test_scenario_ps_coal.conf"
