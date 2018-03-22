import os
import pandas as pd
import copy
import json
from sifra.logger import rootLogger
from collections import OrderedDict
from sifra.modelling.infrastructure import InfrastructureFactory
from sifra.modelling.component import (Component, ConnectionValues)


def ingest_model(config):
    """
    Reads a model file into python objects
    :param config: path to json or xlsx file containing system model
    :return:    -List of algorithms for each component in particular damage state
                -Object of class infrastructure
    """
    extension = os.path.splitext(config.SYS_CONF_FILE)[1][1:].strip().lower()

    if extension == 'json':
        return read_model_from_json(config)

    elif extension == 'xlsx':
        return read_model_from_xlxs(config)
    else:
        rootLogger.critical("Invalid model file type! Accepted types are json or xlsx.")
        raise ValueError('Invalid model file type! Accepted types are json or xlsx. File supplied: '+config.SYS_CONF_FILE)


def read_model_from_json(config):
    """
    Create an infrastructure_model and AlgorithmFactory from the infrastructure model in json file
    :param config:
    :return:
    """
    component_dict = {}

    system_class = config.SYSTEM_CLASS
    system_subclass = config.SYSTEM_SUBCLASS

    with open(config.SYS_CONF_FILE, 'r') as f:
        # ensure that damage states are ordered
        model = json.load(f, object_pairs_hook=OrderedDict)

    # read the lists from json
    component_list = model['component_list']
    node_conn_df = model['node_conn_df']
    sysinp_setup = model['sysinp_setup']
    sysout_setup = model['sysout_setup']


    system_components = {}

    for component_id in component_list:
        component_values = {}

        component_values['component_id'] = component_id
        component_values['component_class'] = component_list[component_id]['component_class']
        component_values['component_type'] = component_list[component_id]['component_type']
        component_values['cost_fraction'] = component_list[component_id]['cost_fraction']
        component_values['node_cluster'] = component_list[component_id]['node_cluster']
        component_values['node_type'] = component_list[component_id]['node_type']
        component_values['operating_capacity'] = component_list[component_id]['operating_capacity']
        component_values['longitude'] = component_list[component_id]['longitude']
        component_values['latitude'] = component_list[component_id]['latitude']

        component_values['damages_states_constructor'] = component_list[component_id]['damages_states_constructor']

        # list of damage states with a function assignment!

        system_components[component_id] = Component(**component_values)

    # TODO refractor code below, combine the two high level variables in input json
    # and make corresponding changes in code below

    # now we add children!
    for index in node_conn_df:
        component_id = node_conn_df[index]['origin']
        system_component = system_components[component_id]

        if not system_component.destination_components:
            system_component.destination_components = {}
        edge_values = {}
        edge_values['link_capacity'] = float(node_conn_df[index]['link_capacity'])
        edge_values['weight'] = float(node_conn_df[index]['weight'])
        system_component.destination_components[node_conn_df[index]['destination']] = ConnectionValues(**edge_values)

    infrastructure_system_constructor = dict()
    infrastructure_system_constructor['name'] = system_class + " : " + system_subclass
    infrastructure_system_constructor['components'] = system_components

    # create the supply and output node dictionaries
    supply_nodes = {}
    for index in sysinp_setup:
        sv_dict = {}
        sv_dict['input_capacity'] = sysinp_setup[index]['input_capacity']
        sv_dict['capacity_fraction'] = float(sysinp_setup[index]['capacity_fraction'])
        sv_dict['commodity_type'] = sysinp_setup[index]['commodity_type']
        supply_nodes[index] = sv_dict

    infrastructure_system_constructor['supply_nodes'] = supply_nodes

    output_nodes = {}
    for index in sysout_setup:
        op_dict = {}
        op_dict['production_node'] = sysout_setup[index]['production_node']
        op_dict['output_node_capacity'] = sysout_setup[index]['output_node_capacity']
        op_dict['capacity_fraction'] = float(sysout_setup[index]['capacity_fraction'])
        op_dict['priority'] = sysout_setup[index]['priority']
        output_nodes[index] = op_dict

    infrastructure_system_constructor['sys_dmg_states'] = []
    for key in component_list:
        for damages_state in component_list[key]["damages_states_constructor"]:
            if damages_state not in infrastructure_system_constructor['sys_dmg_states']:
                infrastructure_system_constructor['sys_dmg_states'].append(damages_state)

    infrastructure_system_constructor['output_nodes'] = output_nodes

    # set the system class
    infrastructure_system_constructor['system_class'] = system_class

    return InfrastructureFactory.create_model(infrastructure_system_constructor)

def read_model_from_xlxs(config):
    return None
# def read_model_from_xlxs(config):
#     """
#     Create an infrastructure_model and AlgorithmFactory from the infrastructure model in xlsx file
#     :param config:
#     :return:
#     """
#     component_dict = {}
#
#     system_class = config.SYSTEM_CLASS
#     system_subclass = config.SYSTEM_SUBCLASS
#
#     damage_state_def = pd.read_excel(
#         config.SYS_CONF_FILE, sheet_name='damage_state_def',
#         index_col=[0, 1], header=0,
#         skiprows=0, skipinitialspace=True)
#
#     component_connections = pd.read_excel(
#         config.SYS_CONF_FILE, sheet_name='component_connections',
#         index_col=None, header=0,
#         skiprows=0, skipinitialspace=True)
#
#     component_list = pd.read_excel(
#         config.SYS_CONF_FILE, sheet_name='component_list',
#         index_col='component_id', header=0,
#         skiprows=0, skipinitialspace=True)
#
#     output_setup = pd.read_excel(
#         config.SYS_CONF_FILE, sheet_name='output_setup',
#         index_col='output_node', header=0,
#         skiprows=0, skipinitialspace=True).sort_values(by='priority', ascending=True)
#
#     supply_setup = pd.read_excel(
#         config.SYS_CONF_FILE, sheet_name='supply_setup',
#         index_col='input_node', header=0,
#         skiprows=0, skipinitialspace=True)
#
#     comp_type_dmg_algo = pd.read_excel(
#         config.SYS_CONF_FILE, sheet_name='comp_type_dmg_algo',
#         index_col=[0, 1], header=0,
#         skiprows=0, skipinitialspace=True)
#
#
#     damage_def_dict = {}
#     for index, damage_def in damage_state_def.iterrows():
#         damage_def_dict[index] = damage_def
#
#     for index, damage_state in comp_type_dmg_algo.iterrows():
#         component_type = index[0]
#         if component_type not in component_dict:
#             damage_algorithm_vals = IODict()
#             # damage_algorithm_vals[u'DS0 None'] = Level0Response()
#             recovery_algorithm_vals = {}
#             # recovery_algorithm_vals[u'DS0 None'] = Level0Recovery()
#             # store the current values in the Algorithms
#             component_dict[component_type] = {}
#             component_dict[component_type]['component_type'] = component_type
#             component_dict[component_type]['frag_func'] = DamageAlgorithm(damage_states=damage_algorithm_vals)
#             component_dict[component_type]['recovery_func'] = RecoveryAlgorithm(recovery_states=recovery_algorithm_vals)
#
#         else:
#             damage_algorithm_vals = component_dict[component_type]['frag_func'].damage_states
#             recovery_algorithm_vals = component_dict[component_type]['recovery_func'].recovery_states
#
#         ds_level = index[1]
#         damage_def_state = "NA"
#         if index in damage_def_dict:
#             damage_def_state = damage_def_dict[index]
#
#         response_params = {}
#         response_params['damage_ratio'] = damage_state['damage_ratio']
#         response_params['functionality'] = damage_state['functionality']
#         response_params['fragility_source'] = damage_state['fragility_source']
#
#         response_params['damage_state_description'] = damage_def_state
#
#         if damage_state['damage_function'] == 'Lognormal':
#             # translate the column names
#             response_params['median'] = damage_state['median']
#             response_params['beta'] = damage_state['beta']
#             response_params['mode'] = damage_state['mode']
#             response_model = LogNormalCDF(**response_params)
#         elif damage_state['damage_function'] == 'Normal':
#             response_model = NormalCDF(**response_params)
#         elif damage_state['damage_function'] == 'StepFunc':
#             response_model = StepFunc(**response_params)
#         else:
#             raise ValueError("No response model "
#                              "matches {}".format(damage_state['damage_function']))
#
#         # add the response model to damage algorithm
#         damage_algorithm_vals[ds_level] = response_model
#         # create the recovery_model
#         recovery_columns = ('recovery_std', 'recovery_mean', 'recovery_95percentile')
#         recovery_params = {key: damage_state[key] for key in recovery_columns}
#         recovery_algorithm_vals[ds_level] = RecoveryState(**recovery_params)
#
#
#
#     # add the other component attributes and make a component dict
#     system_components = {}
#     for component_id, component_details in component_list.iterrows():
#         component_type = component_details['component_type']
#         if component_type in component_dict:
#             component_values = copy.deepcopy(component_dict[component_type])
#         else:
#             print("Unknown component {}".format(component_type))
#             continue
#
#         component_values['component_id'] = component_id
#         component_values['component_type'] = component_details['component_type']
#         component_values['component_class'] = component_details['component_class']
#         component_values['cost_fraction'] = component_details['cost_fraction']
#         component_values['node_type'] = component_details['node_type']
#         component_values['node_cluster'] = component_details['node_cluster']
#         component_values['operating_capacity'] = component_details['op_capacity']
#
#         system_components[component_id] = Component(**component_values)
#
#
#     # now we add children!
#     for index, connection_values in component_connections.iterrows():
#         component_id = connection_values['origin']
#         system_component = system_components[component_id]
#
#         if not system_component.destination_components:
#             system_component.destination_components = {}
#         edge_values = {}
#         edge_values['link_capacity'] = float(connection_values['link_capacity'])
#         edge_values['weight'] = float(connection_values['weight'])
#         system_component.destination_components[connection_values['destination']] = ConnectionValues(**edge_values)
#
#     if_system_values = dict()
#     if_system_values['name'] = system_class + " : " + system_subclass
#     if_system_values['components'] = system_components
#
#     # create the supply and output node dictionaries
#     supply_nodes = {}
#     for index, supply_values in supply_setup.iterrows():
#         sv_dict = {}
#         sv_dict['input_capacity'] = supply_values['input_capacity']
#         sv_dict['capacity_fraction'] = float(supply_values['capacity_fraction'])
#         sv_dict['commodity_type'] = supply_values['commodity_type']
#         supply_nodes[index] = sv_dict
#
#     if_system_values['supply_nodes'] = supply_nodes
#
#     output_nodes = {}
#     for index, output_values in output_setup.iterrows():
#         op_dict = {}
#         op_dict['production_node'] = output_values['production_node']
#         op_dict['output_node_capacity'] = output_values['output_node_capacity']
#         op_dict['capacity_fraction'] = float(output_values['capacity_fraction'])
#         op_dict['priority'] = output_values['priority']
#         output_nodes[index] = op_dict
#
#     if_system_values['output_nodes'] = output_nodes
#
#     # set the system class
#     if_system_values['system_class'] = system_class
#
#     if_system_values['sys_dmg_states'] = []
#     for index, damage_state in comp_type_dmg_algo.iterrows():
#         state = index[1]
#         if state not in if_system_values['sys_dmg_states']:
#             if_system_values['sys_dmg_states'].append(state)
#
#     return InfrastructureFactory.create_model(if_system_values)
