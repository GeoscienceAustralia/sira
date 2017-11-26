from __future__ import print_function

import unittest
import logging

import json
from sifra.modelling.utils import pythonify
from model_ingest import ingest_spreadsheet


class TestSifraWeb(unittest.TestCase):

    def test_save_response_models(self):
        request_data = """{"class": ["sifra.modelling.responsemodels", "DamageState"],
                        "damage_ratio": 0.5,
                        "damage_state": "2",
                        "damage_state_description": "Level 3 significant damage",
                        "fragility_source": "arxiv 33092409",
                        "functionality": 0.3,
                        "mode": 2,
                        "component_sector": {
                            "hazard": "earthquake",
                            "sector": "electric power station",
                            "facility_type": "Generation Plant",
                            "component": "Generator"},
                        "attributes": {
                            "name": "gen-1",
                            "description": "Number 1 generator for the power station"
                            }
                        }"""

        data = json.loads(request_data)
        category = data.pop('component_sector', None)
        attrs = data.pop('attributes', None)
        inst = pythonify(data)
        oid = inst.save(
            category=category,
            attributes=attrs)

        self.assertTrue(oid is not None)

    def test_save_spreadsheet(self):
        config_file = '/opt/project/tests/test_scenario_ps_coal.conf'
        infrastructure = ingest_spreadsheet(config_file)
        category = infrastructure.name
        attrs = None
        oid = infrastructure.save(
            category=category,
            attributes=attrs)

        self.assertTrue(oid is not None)

