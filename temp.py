import numpy as np
from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.model_ingest import ingest_model
from sifra.modelling.system_topology import SystemTopology
from sifra.infrastructure_response import calculate_response, post_processing
from sifra.modelling.hazard import Hazard

# main program
jsonFileName = "config_test.json"
config = Configuration(jsonFileName)
scenario = Scenario(config)
hazard = Hazard(config)

infrastructure = ingest_model(config)

post_processing_list = calculate_response(scenario, infrastructure, hazard)

# response
post_processing(infrastructure, scenario, post_processing_list,hazard)

# graphs
sys_topology_view = SystemTopology(infrastructure, scenario)
sys_topology_view.draw_sys_topology(viewcontext="as-built")

#
# class Base(object):
#     def __init__(self, *arg, **kwargs):
#
#         for a in arg:
#             print(a)
#         for k, v in kwargs.iteritems():
#             setattr(self, k, v)
#
#
# dictnry = {"a": 1, "b": 2}
# lst=['a','b']
# new = Base(*lst, **dictnry)
# print(new.a)
# print(new.b)