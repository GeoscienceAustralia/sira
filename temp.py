import numpy as np
from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.model_ingest import ingest_model
from sifra.modelling.system_topology import SystemTopology
from sifra.infrastructure_response import calculate_response, post_processing


#test algorithms class and functions

class DamageState():
    """
    The allocated damage state for a given component
    """
    damage_state = 'test damage state'
    # damage_state_description = Element('str', 'A description of what the damage state means')
    mode = 2
    functionality = 1.1
    # fragility_source = Element('str', 'The source of the parameter values')
    damage_ratio = 0.1
    hazard_intensity = 1



damage_states = []
damage_states.append(DamageState())
damage_states.append(DamageState())
damage_states.append(DamageState())

pe_ds = np.zeros(len(damage_states))

for offset, damage_state in enumerate(damage_states):
    # if damage_state.mode != 1:
    #     raise RuntimeError("Mode {} not implemented".format(damage_state.mode))
    pe_ds[offset] = damage_state.hazard_intensity

print(pe_ds)


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
