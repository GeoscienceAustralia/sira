import numpy as np
from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.model_ingest import ingest_model
from sifra.modelling.system_topology import SystemTopology
from sifra.infrastructure_response import calculate_response, post_processing


#test algorithms class and functions



# main program
jsonFileName = "config_test.json"
config = Configuration(jsonFileName)
scenario=Scenario(config)
infrastructure, algorithm_factory = ingest_model(config)
scenario.algorithm_factory = algorithm_factory
sys_topology_view = SystemTopology(infrastructure, scenario)
sys_topology_view.draw_sys_topology(viewcontext="as-built")
post_processing_list = calculate_response(scenario, infrastructure)
post_processing(infrastructure, scenario, post_processing_list)
