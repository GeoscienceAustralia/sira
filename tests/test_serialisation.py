from __future__ import print_function

import unittest
import logging

import json
from sifra.modelling.utils import pythonify
from sifra.modelling.sifra_db import save_model, load_model, save_component
from sifra.modelling.structural import Base
from sifra.model_ingest import ingest_spreadsheet


class TestSifraDB(unittest.TestCase):

    def setUp(self):
        # delete it all!
        Base._provider.delete_db()

    def tearDown(self):
        pass

    def test_save_spreadsheet(self):
        config_file = './test_scenario_ps_coal.conf'
        infrastructure, _ = ingest_spreadsheet(config_file)
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
