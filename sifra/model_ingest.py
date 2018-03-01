import pandas as pd

from sifraclasses import _FacilityDataGetter

from sifra.modelling.component import (Component,
                                       ConnectionValues)

from sifra.modelling.iodict import IODict

from sifra.modelling.infrastructure_system import IFSystemFactory

from sifra.modelling.responsemodels import (LogNormalCDF, NormalCDF, StepFunc,
                                            Level0Response, Level0Recovery,
                                            DamageAlgorithm, RecoveryState,
                                            RecoveryAlgorithm, AlgorithmFactory)

import copy


def ingest_spreadsheet(config):
    """
    Create an infrastructure_model from the config 
    :param config:
    :return:
    """
    facility_data = _FacilityDataGetter(config)
    component_dict = IODict()
    algorithm_factory = AlgorithmFactory()

    damage_state_df = pd.read_excel(
        facility_data.sys_config_file, sheetname='damage_state_def',
        index_col=[0, 1], header=0,
        skiprows=3, skipinitialspace=True)

    damage_def_dict = {}
    for index, damage_def in damage_state_df.iterrows():
        damage_def_dict[index] = damage_def

    for index, damage_state in facility_data.fragility_data.iterrows():
        component_type = index[0]

        if component_type not in component_dict:
            damage_algorithm_vals = IODict()
            damage_algorithm_vals[u'DS0 None'] = Level0Response()
            recovery_algorithm_vals = IODict()
            recovery_algorithm_vals[u'DS0 None'] = Level0Recovery()
            # store the current values in the Algorithms
            component_dict[component_type] = {}
            component_dict[component_type]['component_type'] = component_type
            component_dict[component_type]['frag_func'] = DamageAlgorithm(damage_states=damage_algorithm_vals)
            component_dict[component_type]['recovery_func'] = RecoveryAlgorithm(recovery_states=recovery_algorithm_vals)
        else:
            damage_algorithm_vals = component_dict[component_type]['frag_func'].damage_states
            recovery_algorithm_vals = component_dict[component_type]['recovery_func'].recovery_states

        ds_level = index[1]
        if index in damage_def_dict:
            damage_def_state = damage_def_dict[index]

        response_params = {}
        response_params['damage_ratio'] = damage_state['damage_ratio']
        response_params['functionality'] = damage_state['functionality']
        response_params['fragility_source'] = damage_state['fragility_source']
        response_params['damage_state_description'] = damage_def_state['damage_state_definition']
        if damage_state['damage_function'] == 'Lognormal':
            # translate the column names
            response_params['median'] = damage_state['damage_median']
            response_params['beta'] = damage_state['damage_logstd']
            response_params['mode'] = damage_state['mode']
            response_model = LogNormalCDF(**response_params)
        elif damage_state['damage_function'] == 'Normal':
            response_model = NormalCDF(**response_params)
        elif damage_state['damage_function'] == 'StepFunc':
            response_model = StepFunc(**response_params)
        else:
            raise ValueError("No response model "
                             "matches {}".format(damage_state['damage_function']))

        # add the response model to damage algorithm
        damage_algorithm_vals[ds_level] = response_model

        # create the recovery_model
        recovery_columns = ('recovery_std', 'recovery_mean', 'recovery_95percentile')
        recovery_params = {key: damage_state[key] for key in recovery_columns}
        recovery_algorithm_vals[ds_level] = RecoveryState(**recovery_params)

    # testing
    print(component_type, component_dict.__jsonify__())
    raw_input("Press Enter to continue...")

    # Create a damage algorithm in the AlgorithmFactory for the components
    for component_type in component_dict.keys():
        algorithm_factory.add_response_algorithm(component_type,
                                                 'earthquake',
                                                 component_dict[component_type]['frag_func'])

        algorithm_factory.add_recovery_algorithm(component_type,
                                                 'earthquake',
                                                 component_dict[component_type]['recovery_func'])

    # add the other component attributes and make a component dict
    system_components = IODict()
    for component_id, component_details in facility_data.comp_df.iterrows():
        component_type = component_details['component_type']
        if component_type in component_dict:
            component_values = copy.deepcopy(component_dict[component_type])
        else:
            print("Unknown component {}".format(component_type))
            continue

        component_values['component_id'] = component_id
        component_values['component_type'] = component_details['component_type']
        component_values['component_class'] = component_details['component_class']
        component_values['cost_fraction'] = component_details['cost_fraction']
        component_values['node_type'] = component_details['node_type']
        component_values['node_cluster'] = component_details['node_cluster']
        component_values['operating_capacity'] = component_details['op_capacity']

        system_components[component_id] = Component(**component_values)

    # now we add children!
    for index, connection_values in facility_data.node_conn_df.iterrows():
        component_id = connection_values['origin']
        system_component = system_components[component_id]
        destiny = system_component.destination_components
        if not destiny:
            destiny = system_component.destination_components = IODict()
        edge_values = {}
        edge_values['link_capacity'] = float(connection_values['link_capacity'])
        edge_values['weight'] = float(connection_values['weight'])
        destiny[connection_values['destination']] = ConnectionValues(**edge_values)

    if_system_values = dict()
    if_system_values['name'] = facility_data.system_class + " : " \
                               + facility_data.system_subclass
    if_system_values['components'] = system_components

    # create the supply and output node dictionaries
    supply_nodes = {}
    for index, supply_values in facility_data.sysinp_setup.iterrows():
        sv_dict = {}
        sv_dict['input_capacity'] = supply_values['input_capacity']
        sv_dict['capacity_fraction'] = float(supply_values['capacity_fraction'])
        sv_dict['commodity_type'] = supply_values['commodity_type']
        supply_nodes[index] = sv_dict

    if_system_values['supply_nodes'] = supply_nodes

    output_nodes = {}
    for index, output_values in facility_data.sysout_setup.iterrows():
        op_dict = {}
        op_dict['production_node']=output_values['production_node']
        op_dict['output_node_capacity'] = output_values['output_node_capacity']
        op_dict['capacity_fraction'] = float(output_values['capacity_fraction'])
        op_dict['priority'] = output_values['priority']
        output_nodes[index] = op_dict

    if_system_values['output_nodes'] = output_nodes

    # set the system class
    if_system_values['system_class'] = facility_data.system_class

    return IFSystemFactory.create_model(if_system_values), algorithm_factory
