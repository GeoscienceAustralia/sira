from __future__ import print_function

import unittest
import logging

import json
from sifra.modelling.utils import pythonify
from sifra.modelling.sifra_db import save_model, load_model, save_component
from sifra.modelling.structural import Base
from model_ingest import ingest_spreadsheet


class TestSifraDB(unittest.TestCase):

    def setUp(self):
        # delete it all!
        Base._provider.delete_db()

    def tearDown(self):
        pass

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
        inst = pythonify(data)
        oid = inst.save(inst)

        self.assertTrue(oid is not None)

    def test_save_spreadsheet(self):
        config_file = '/opt/project/tests/test_scenario_ps_coal.conf'
        infrastructure = ingest_spreadsheet(config_file)
        oid = save_model(infrastructure)

        self.assertTrue(oid is not None)

        existing_components = set()

        for component in infrastructure.components.values():
            if component.component_type in ('SYSTEM_INPUT', 'SYSTEM_OUTPUT'):
                continue

            if component.component_type not in existing_components:
                existing_components.add(component.component_type)
                # remove the unwanted attributes
                del(component.component_id)
                del(component.operating_capacity)
                del(component.destination_components)
                del(component.cost_fraction)
                del(component.node_cluster)
                del(component.node_type)
                component.hazard_type = 'earthquake'
                comp_oid = save_component(component)

            self.assertTrue(comp_oid is not None)
