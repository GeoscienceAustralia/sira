from __future__ import print_function

import unittest
import logging

from sifra.infrastructure_response import run_scenario


class TestSifra(unittest.TestCase):
    def test_run_ps_coal_scenario(self):

        # SETUPFILE = '/opt/project/tests/test_scenario_ps_coal.conf'
        SETUPFILE = './tests/test_scenario_ps_coal.conf'
        run_scenario(SETUPFILE)

    def test_run_series_scenario(self):

        # SETUPFILE = '/opt/project/tests/test_simple_series_struct.conf'
        SETUPFILE = './tests/test_simple_series_struct.conf'
        run_scenario(SETUPFILE)

    def test_run_series_dep_scenario(self):

        # SETUPFILE = '/opt/project/tests/test_simple_series_struct_dep.conf'
        SETUPFILE = './tests/test_simple_series_struct_dep.conf'
        run_scenario(SETUPFILE)

    def test_run_parallel_scenario(self):

        # SETUPFILE = '/opt/project/tests/test_simple_parallel_struct.conf'
        SETUPFILE = './tests/test_simple_parallel_struct.conf'
        run_scenario(SETUPFILE)
