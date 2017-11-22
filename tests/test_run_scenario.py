from __future__ import print_function

import unittest
import logging

from sifra.infrastructure_response import run_scenario


class TestSifra(unittest.TestCase):
    def test_run_scenario(self):

        SETUPFILE = '/opt/project/tests/test_scenario_ps_coal.conf'
        run_scenario(SETUPFILE)

        # TODO determine success parameters