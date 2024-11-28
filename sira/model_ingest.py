import json
import logging
from collections import OrderedDict
from pathlib import Path

from sira.modelling.component import Component, ConnectionValues
from sira.modelling.infrastructure import InfrastructureFactory
from sira.tools.convert_excel_files_to_json import (read_excel_to_json,
                                                    update_json_structure)

rootLogger = logging.getLogger(__name__)


def ingest_model(config):
    """
    Reads a model file into python objects
    :param config: path to json or xlsx file containing system model
    :return:
        - List of algorithms for each component in particular damage state
        - Object of class infrastructure
    """
    extension = Path(config.INPUT_MODEL_PATH).suffix[1:].lower()

    if extension == 'json':
        with open(config.INPUT_MODEL_PATH, 'r') as f:
            # ensure that damage states are ordered
            model = json.load(f, object_pairs_hook=OrderedDict)
        return read_model_from_json(config, model)

    elif extension == 'xlsx':
        json_obj = json.loads(
            read_excel_to_json(config.INPUT_MODEL_PATH),
            object_pairs_hook=OrderedDict)
        model = update_json_structure(json_obj)
        return read_model_from_json(config, model)

    else:
        rootLogger.critical(
            "Invalid model file type! "
            "Accepted types are json or xlsx.")
        raise ValueError(
            "Invalid model file type! "
            "Accepted types are json or xlsx. "
            "File supplied: " + config.SYS_CONF_FILE)


def read_model_from_json(config, model):
    """
    Create an infrastructure_model and AlgorithmFactory from the
    infrastructure model in json file
    :param config:
    :return:
    """
    system_class = config.SYSTEM_CLASS
    system_subclass = config.SYSTEM_SUBCLASS

    # read the lists from json
    system_meta = model['system_meta']
    component_list = model['component_list']
    node_conn_df = model['node_conn_df']
    sysinp_setup = model['sysinp_setup']
    sysout_setup = model['sysout_setup']

    system_components = {}

    for component_id in component_list:
        component_values = {}
        component_values['component_id'] = component_id

        for param in component_list[component_id].keys():
            component_values[param] = component_list[component_id][param]

        # list of damage states with a function assignment!
        system_components[component_id] = Component(**component_values)

    # TODO refactor code below, combine the two high level variables
    # in input json and make corresponding changes in code below

    # now we add children!
    for index in node_conn_df:
        component_id = node_conn_df[index]['origin']
        system_component = system_components[component_id]

        if not system_component.destination_components:
            system_component.destination_components = {}
        edge_values = {}
        edge_values['link_capacity'] \
            = float(node_conn_df[index]['link_capacity'])
        edge_values['weight'] = float(node_conn_df[index]['weight'])
        system_component.\
            destination_components[node_conn_df[index]['destination']] \
            = ConnectionValues(**edge_values)

    infrastructure_system_constructor = dict()
    infrastructure_system_constructor['name'] = \
        system_class + " : " + system_subclass
    infrastructure_system_constructor['components'] = system_components
    infrastructure_system_constructor['system_meta'] = dict(system_meta)

    # create the supply and output node dictionaries
    supply_nodes = {}
    for index in sysinp_setup:
        sv_dict = {}
        sv_dict['input_capacity'] \
            = sysinp_setup[index]['input_capacity']
        sv_dict['capacity_fraction'] \
            = float(sysinp_setup[index]['capacity_fraction'])
        sv_dict['commodity_type'] \
            = sysinp_setup[index]['commodity_type']

        supply_nodes[index] = sv_dict

    infrastructure_system_constructor['supply_nodes'] = supply_nodes

    output_nodes = {}
    for index in sysout_setup:
        op_dict = {}
        op_dict['production_node'] \
            = sysout_setup[index]['production_node']
        op_dict['output_node_capacity'] \
            = sysout_setup[index]['output_node_capacity']
        op_dict['capacity_fraction'] \
            = float(sysout_setup[index]['capacity_fraction'])
        op_dict['priority'] = sysout_setup[index]['priority']
        output_nodes[index] = op_dict

    infrastructure_system_constructor['system_dmg_states'] = []
    for key in component_list:
        for damages_state in component_list[key]["damages_states_constructor"]:
            if damages_state not in \
                    infrastructure_system_constructor['system_dmg_states']:
                infrastructure_system_constructor['system_dmg_states'].\
                    append(damages_state)

    infrastructure_system_constructor['output_nodes'] = output_nodes

    # set the system class
    infrastructure_system_constructor['system_class'] = system_class

    return InfrastructureFactory.create_model(infrastructure_system_constructor)
