import csv
import os
import numpy as np
from sifra.configuration import Configuration
from sifra.scenario import Scenario
from sifra.simulation import calculate_response
from sifra.model_ingest import ingest_model
from sifra.modelling.system_topology import SystemTopology
from sifra.infrastructure_response import  write_system_response, loss_by_comp_type, plot_mean_econ_loss, pe_by_component_class
from sifra.modelling.hazard import Hazards

conf_file_paths = []
parent_folder_name = os.path.dirname(os.getcwd())

for root, dir_names, file_names in os.walk(parent_folder_name):
    for file_name in file_names:
        if file_name.endswith('.json'):
            if 'simulation_setup' in root:
                conf_file_path = os.path.join(root, file_name)
                conf_file_paths.append(conf_file_path)

for jsonFileName in conf_file_paths:
    """
    Configure simulation model.
    Read data and control parameters and construct objects.
    """
    jsonFileName = "test_config.json"
    config = Configuration(jsonFileName)
    scenario = Scenario(config)
    hazard = Hazards(config)
    infrastructure = ingest_model(config)

    """
    Run simulation.
    Get the results of running a simulation
    """
    response_list = calculate_response(scenario, infrastructure, hazard)

    """
    Post simulation processing.
    After the simulation has run the results are aggregated, saved
    and the system fragility is calculated.
    """
    write_system_response(response_list, scenario)
    loss_by_comp_type(response_list, infrastructure, scenario, hazard)
    economic_loss_array = response_list[4]
    plot_mean_econ_loss(scenario, economic_loss_array, hazard)

    if config.HAZARD_INPUT_METHOD == "hazard_array":
        pe_by_component_class(response_list, infrastructure, scenario, hazard)

    # graphs
    sys_topology_view = SystemTopology(infrastructure, scenario)
    sys_topology_view.draw_sys_topology(viewcontext="as-built")
