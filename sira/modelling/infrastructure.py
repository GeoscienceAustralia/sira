import sys

import numpy as np
from sira.modelling.component_graph import ComponentGraph
from sira.modelling.iodict import IODict
from sira.modelling.structural import Base, Element

np.set_printoptions(threshold=sys.maxsize)


class InfrastructureFactory(object):
    @staticmethod
    def create_model(config):
        if config['system_class'].lower() == 'substation':
            return Substation(**config)
        elif config['system_class'].lower() == 'powerstation':
            return PowerStation(**config)
        elif config['system_class'].lower() in ['potablewatertreatmentplant']:
            return PotableWaterTreatmentPlant(**config)
        elif config['system_class'].lower() == 'PotableWaterPumpStation'.lower():
            return PotableWaterPumpStation(**config)
        elif config['system_class'].lower() == 'ModelTestStructure'.lower():
            return ModelTestStructure(**config)
        elif config['system_class'].lower() in ['railnetwork', 'rail_network']:
            return RailNetwork(**config)


class Infrastructure(Base):
    """
    The top level representation of a system that can respond to a
    range of hazards. It encapsulates a number of components that
    are used to estimate the response to various hazard levels.
    """
    supply_nodes = Element(
        'dict',
        'The components that supply the infrastructure system',
        dict)
    output_nodes = Element(
        'dict',
        'The components that output from the infrastructure system',
        dict)

    # supply_total = None
    _component_graph = None
    if_nominal_output = None
    system_class = None
    sys_dmg_states = None

    def __init__(self, **kwargs):
        """
        Construct the infrastructure object
        :param kwargs: Objects making up the infrastructure
        """
        super(Infrastructure, self).__init__(**kwargs)

        if not getattr(self, "components", None):
            self.components = IODict()

        self._component_graph = ComponentGraph(self.components)

    def calc_output_loss(self, scenario, component_damage_state_ind):
        """
        Calculate the results to the infrastructure given the damage state
        parameter
        :param scenario: Details of the scenario being run
        :param component_damage_state_ind:  The array of the component's
                                            damage state samples
        :return: 5 lists of calculations
        """
        # Component loss caused by the damage
        if_level_loss = np.zeros(
            (scenario.num_samples, len(self.components)), dtype=float)
        # Infrastructure loss: sum of component loss
        if_level_economic_loss = np.zeros(
            scenario.num_samples, dtype=float)
        # Component functionality
        if_level_functionality = np.zeros(
            (scenario.num_samples, len(self.components)), dtype=float)
        # output for the level of damage
        if_level_output = np.zeros(
            (scenario.num_samples, len(self.output_nodes)), dtype=float)

        # ********************
        # NOT YET IMPLEMENTED:
        # output available as recovery progresses
        # if_output_given_recovery = \
        #     np.zeros((scenario.num_samples, scenario.num_time_steps),
        #              dtype=float)

        # iterate through the samples
        for sample_index in range(scenario.num_samples):
            # initialise the function and loss arrays for the sample
            # component_function_at_time = []
            comp_sample_loss = np.zeros(len(self.components))
            comp_sample_func = np.zeros(len(self.components))
            # Extract the array of damage states for this sample

            component_ds = component_damage_state_ind[sample_index, :]
            # iterate through the components
            for component_index, comp_key in \
                    enumerate(sorted(self.components.keys())):
                component = self.components[comp_key]
                # get the damage state for the component
                damage_state \
                    = component.get_damage_state(component_ds[component_index])
                # use the damage state attributes to calculate the loss and
                # functionality for the component sample
                loss = damage_state.damage_ratio * component.cost_fraction
                comp_sample_loss[component_index] = loss
                comp_sample_func[component_index] = damage_state.functionality

            # save this sample's component loss and functionality
            if_level_loss[sample_index, :] = comp_sample_loss
            if_level_functionality[sample_index, :] = comp_sample_func
            # the infrastructure's economic loss for this sample is the sum
            # of all component losses
            if_level_economic_loss[sample_index] = np.sum(comp_sample_loss)
            # estimate the output for this sample's component functionality
            if_level_output[sample_index, :] = \
                self.compute_output_given_ds(comp_sample_func)

        return if_level_loss, if_level_functionality, \
            if_level_output, if_level_economic_loss

    def get_nominal_output(self):
        """
        Estimate the output of the undamaged infrastructure output
        nodes.
        :return: Numeric value of aggregated output.
        """
        if not self.if_nominal_output:
            self.if_nominal_output = 0
            # `output_nodes` is a dict
            for output_comp in list(self.output_nodes.values()):   # noqa: E1101
                self.if_nominal_output += output_comp['output_node_capacity']

        return self.if_nominal_output

    def compute_output_given_ds(self, comp_level_func):
        """
        Using the graph of components, the output is calculated
        from the component functionality parameter.
        :param comp_level_func: An array that indicates the functionality
                                level for each component.
        :return: An array of the output level for each output node.
        """
        # Create the component graph if one does not yet exist
        # if not self._component_graph:
        #     self._component_graph = ComponentGraph(
        #         self.components, comp_level_func)
        # else:
        #     self._component_graph.update_capacity(
        #         self.components, comp_level_func)
        self._component_graph.update_capacity(
            self.components, comp_level_func)

        # calculate the capacity
        system_outflows_sample = np.zeros(len(self.output_nodes))
        for output_index, (output_comp_id, output_comp) in \
                enumerate(self.output_nodes.items()):   # noqa: E1101
            # track the outputs by source type (e.g. water or coal)
            total_supply_flow_by_source = {}
            for (supply_comp_id, supply_comp) in \
                    self.supply_nodes.items():   # noqa: E1101
                if_flow_fraction = self._component_graph.maxflow(
                    supply_comp_id, output_comp_id
                )
                if_sample_flow = if_flow_fraction * \
                    supply_comp['capacity_fraction']

                if supply_comp['commodity_type'] not in \
                        total_supply_flow_by_source:
                    total_supply_flow_by_source[supply_comp['commodity_type']] \
                        = if_sample_flow
                else:
                    total_supply_flow_by_source[supply_comp['commodity_type']] \
                        += if_sample_flow

            total_available_flow = min(total_supply_flow_by_source.values())

            estimated_capacity_fraction \
                = min(total_available_flow, output_comp['capacity_fraction'])
            system_outflows_sample[output_index] \
                = estimated_capacity_fraction * self.get_nominal_output()

        return system_outflows_sample

    # pylint: disable=fixme
    # TODO: FIX `calc_recov_time_given_comp_ds`
    # def calc_recov_time_given_comp_ds(self, component, damage_state, scenario):
    #     '''
    #     Calculates the recovery time of a component, given damage state index
    #     '''
    #     import scipy.stats as stats
    #     recovery_parameters = component.get_recovery(damage_state)
    #     damage_parameters = component.get_damage_state(damage_state)
    #
    #     m = recovery_parameters.recovery_param1  # loc = mean
    #     s = recovery_parameters.recovery_param2  # scale = standard deviation
    #     fn = damage_parameters.functionality
    #     cdf = stats.norm.cdf(scenario.restoration_time_range, loc=m, scale=s)
    #     return cdf + (1.0 - cdf) * fn

    def calc_response(self,
                      component_loss,
                      comp_sample_func,
                      component_damage_state_ind):
        """
        Convert the arrays into dicts for subsequent analysis
        :param component_loss: The array of component loss values
        :param comp_sample_func: The array of component functionality values
        :param component_damage_state_ind: The array of component damage state
            indicators
        :return: A dict of component response statistics
        """
        component_list_sorted = np.sort(list(self.components.keys()))
        num_samples = np.shape(component_loss)[0]
        comp_resp_dict = dict()
        comptype_resp_dict = dict()

        # ---------------------------------------------------------------
        # sys_ds_levels = self.get_system_damage_states()
        comp_ds_levels = []

        component_damage_state_array = np.array(component_damage_state_ind)
        comp_damage_state_ind_binned = dict()

        # Collate response of individual components:
        for comp_index, comp_id in enumerate(component_list_sorted):
            component = self.components[comp_id]
            comp_ds_levels = component.damage_states

            comp_damage_state_ind_binned[comp_id] \
                = np.bincount(component_damage_state_array[:, comp_index],
                              minlength=len(comp_ds_levels))

            comp_resp_dict[(comp_id, 'loss_mean')] \
                = np.mean(component_loss[:, comp_index])

            comp_resp_dict[(comp_id, 'loss_std')] \
                = np.std(component_loss[:, comp_index])

            comp_resp_dict[(comp_id, 'func_mean')] \
                = np.mean(comp_sample_func[:, comp_index])

            comp_resp_dict[(comp_id, 'func_std')] \
                = np.std(comp_sample_func[:, comp_index])

            comp_resp_dict[(comp_id, 'num_failures')] \
                = np.mean(component_damage_state_ind[:, comp_index]
                          >= (len(component.damage_states) - 1))  # noqa:W503

        # ---------------------------------------------------------------
        # Collate aggregate response of component grouped by their TYPE:
        for ct_id in self.get_component_types():
            comps_of_a_type = sorted(list(self.get_components_for_type(ct_id)))
            ct_pos_index = [list(component_list_sorted).index(x)
                            for x in comps_of_a_type]

            comptype_resp_dict[(ct_id, 'loss_mean')] \
                = np.mean(component_loss[:, ct_pos_index])

            comptype_resp_dict[(ct_id, 'loss_std')] \
                = np.std(component_loss[:, ct_pos_index])

            comptype_resp_dict[(ct_id, 'loss_tot')] \
                = np.sum(component_loss[:, ct_pos_index]) / float(num_samples)

            comptype_resp_dict[(ct_id, 'func_mean')] \
                = np.mean(comp_sample_func[:, ct_pos_index])

            comptype_resp_dict[(ct_id, 'func_std')] \
                = np.std(comp_sample_func[:, ct_pos_index])

            acomponent = self.components[comps_of_a_type[0]]
            comptype_resp_dict[(ct_id, 'num_failures')] \
                = np.mean(component_damage_state_ind[:, ct_pos_index]
                          >= (len(acomponent.damage_states) - 1))  # noqa:W503
        # ---------------------------------------------------------------

        # comp_dmg_state_array_exp = \
        #     np.around(component_damage_state_array.mean(0))
        # comp_dmg_state_array_std = component_damage_state_array.std(0)

        rows, _ = np.shape(component_damage_state_array)

        comp_cls_dmg_index_binned = dict()
        comp_cls_dmg_index_expected = dict()
        comp_cls_dmg_level_percentages = dict()

        for cls_id in self.get_component_classes():
            comps_of_a_cls = \
                sorted(list(self.get_components_for_class(cls_id)))
            comp_cls_pos_index = \
                [list(component_list_sorted).index(x) for x in comps_of_a_cls]

            comp_cls_dmg_indices_tmp = \
                component_damage_state_array[:, comp_cls_pos_index]

            # Iterate over number of samples
            tmparr = np.zeros(shape=(rows, len(comp_ds_levels)))
            for i in range(rows):
                tmparr[i] = np.bincount(comp_cls_dmg_indices_tmp[i],
                                        minlength=len(comp_ds_levels))

            comp_cls_dmg_level_percentages_matrix \
                = 100 * (tmparr / float(len(comps_of_a_cls)))
            comp_cls_dmg_level_percentages[(cls_id, 'mean')] \
                = comp_cls_dmg_level_percentages_matrix.mean(0)
            comp_cls_dmg_level_percentages[(cls_id, 'std')] \
                = comp_cls_dmg_level_percentages_matrix.std(0)

            comp_cls_dmg_index_binned[cls_id] = tmparr.mean(0)
            tmp = comp_cls_dmg_index_binned[cls_id]
            comp_cls_dmg_index_expected[cls_id] \
                = np.max(np.where(tmp == np.max(tmp)))

        return comp_resp_dict,\
            comptype_resp_dict,\
            comp_cls_dmg_level_percentages,\
            comp_cls_dmg_index_expected

    def get_component_types(self):
        """
        Convenience method to get the list of components that are
        costed.

        :return: list of costed component types
        """
        uncosted_comptypes = {'CONN_NODE', 'SYSTEM_INPUT', 'SYSTEM_OUTPUT'}

        component_types = set()

        for component in list(self.components.values()):
            if component.component_type not in uncosted_comptypes:
                component_types.add(component.component_type)

        return list(component_types)

    def get_components_for_type(self, component_type):
        """
        Return a list of components for the passed component type.
        :param component_type: A string representing a component type
        :return: List of components with the matching component type.
        """
        for component in list(self.components.values()):
            if component.component_type == component_type:
                yield component.component_id

    def get_component_classes(self):
        """
        Convenience method to get the list of components that belong to the
        same `component_class`.

        :return: list of costed component classes
        """
        uncosted_compclasses = [
            'JUNCTION', 'JUNCTION POINT', 'JUNCTION NODE',
            'MODEL ARTEFACT',
            'SYSTEM INPUT',
            'SYSTEM OUTPUT',
            'Generator']

        component_classes = set()

        for component in list(self.components.values()):
            if component.component_class not in uncosted_compclasses:
                component_classes.add(component.component_class)

        return list(component_classes)

    def get_components_for_class(self, component_class):
        """
        Return a list of components for the passed component class.
        :param component_class: A string representing a component class
        :return: List of components with the matching component class.
        """
        for component in list(self.components.values()):
            if component.component_class == component_class:
                yield component.component_id

    def get_system_damage_states(self):
        """
        Return a list of the damage state labels
        :return: List of strings detailing the system damage levels.
        """
        # TODO: THIS IS A HACK. NEED A BETTER SOLUTION!
        one_comp_obj = list(self.components.values())[0]
        self.sys_dmg_states = [one_comp_obj.damage_states[ds].damage_state_name
                               for ds in one_comp_obj.damage_states]
        return self.sys_dmg_states

    def get_dmg_scale_bounds(self):
        """
        An array of damage scale boundaries
        :param scenario: The values for the simulation scenarios
        :return:  Array of real numbers representing damage state boundaries
        """
        # todo introduce system subclass to infrastructure
        return [0.01, 0.05, 0.40, 0.70, 1.00]

    def get_component_class_list(self):
        """
        Return the list of component classes from the components.
        Not sure why duplicates are returned and then stripped out,
        it seems unnecessary
        :return: A generator for the list.
        """
        for component in list(self.components.values()):
            yield component.component_class

    def get_component_graph(self):
        return self._component_graph.digraph


class Substation(Infrastructure):
    def __init__(self, **kwargs):
        # Initialise the substation
        super(Substation, self).__init__(**kwargs)
        self.uncosted_classes = ['JUNCTION', 'JUNCTION POINT', 'JUNCTION NODE',
                                 'MODEL ARTEFACT',
                                 'SYSTEM INPUT',
                                 'SYSTEM OUTPUT',
                                 'Generator', 'Generation Source']
        self.ds_lims_compclasses = {
            'Bus':
                [0.05, 0.40, 0.70, 1.00],
            'Control Building':
                [0.05, 0.40, 0.70, 1.00],
            'Circuit Breaker':
                [0.05, 0.40, 0.70, 1.00],
            'Disconnect Switch':
                [0.05, 0.40, 0.70, 1.00],
            'Current Transformer':
                [0.05, 0.40, 0.70, 1.00],
            'Voltage Transformer':
                [0.05, 0.40, 0.70, 1.00],
            'Potential Transformer':
                [0.05, 0.40, 0.70, 1.00],
            'Autotransformer':
                [0.05, 0.40, 0.70, 1.00],
            'Power Transformer':
                [0.05, 0.40, 0.70, 1.00],
            'Surge Protection':
                [0.05, 0.40, 0.70, 1.00],
            'System Control':
                [0.05, 0.40, 0.70, 1.00],
            'Measurement Instrumentation':
                [0.05, 0.40, 0.70, 1.00],
            'Power Factor Correction':
                [0.05, 0.40, 0.70, 1.00],
            'Lightning Arrester':
                [0.05, 0.40, 0.70, 1.00],
        }

    def get_system_damage_states(self):
        """
        Return a list of the damage state labels
        :return: List of strings detailing the system damage levels.
        """
        self.sys_dmg_states = ['DS0 None',
                               'DS1 Minor',
                               'DS2 Extensive',
                               'DS3 Complete']
        return self.sys_dmg_states

    def get_dmg_scale_bounds(self):
        """
        An array of damage scale boundaries
        :param scenario: The values for the simulation scenarios
        :return:  Array of real numbers representing damage state boundaries
        """
        # [0.1, 0.5, 0.8, 1.0]
        # return [0.01, 0.05, 0.40, 0.70, 1.00]
        return [0.05, 0.4, 0.7, 1.0]


class PowerStation(Infrastructure):
    def __init__(self, **kwargs):
        super(PowerStation, self).__init__(**kwargs)
        # Initiate the power station values, which have been used in all
        # current testing
        self.uncosted_classes = ['SYSTEM INPUT',
                                 'SYSTEM OUTPUT',
                                 'JUNCTION', 'JUNCTION POINT',
                                 'MODEL ARTEFACT']
        self.ds_lims_compclasses = {
            'Boiler': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Control Building': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Emission Management': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Fuel Delivery and Storage': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Fuel Movement': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Generator': [0.0, 0.05, 0.40, 0.70, 1.00],
            'SYSTEM OUTPUT': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Stepup Transformer': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Turbine': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Water System': [0.0, 0.05, 0.40, 0.70, 1.00]
        }

    def get_system_damage_states(self):
        """
        Return a list of the damage state labels
        :return: List of strings detailing the system damage levels.
        """
        self.sys_dmg_states = ["DS0 None",
                               "DS1 Slight", "DS2 Moderate",
                               "DS3 Extensive", "DS4 Complete"]
        return self.sys_dmg_states


class PotableWaterTreatmentPlant(Infrastructure):
    def __init__(self, **kwargs):
        super(PotableWaterTreatmentPlant, self).__init__(**kwargs)
        # Initiate the water treatment plant values, which have been used
        # in all current testing
        self.uncosted_classes = ['SYSTEM INPUT',
                                 'SYSTEM OUTPUT',
                                 'JUNCTION POINT', 'JUNCTION',
                                 'MODEL ARTEFACT']
        self.ds_lims_compclasses = {
            'SYSTEM OUTPUT': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Air Compression Equipment': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Backup Generator': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Basins': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Buildings': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Chlorination Equipment': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Electric Power Equipment': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Electric Power Supply': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Electric Power Commercial': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Electric Power Backup': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Electrical Control Equipment': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Pump Station': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Pumps': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Horizontal Pumps': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Vertical Pumps': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Elevated Pipes': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Internal Pipework': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Chemical Tanks': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Concrete Tanks': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Steel Tanks': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Rectangular Sedimentation Tank': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Circular Clarification Tank': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Wells': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Sediment Flocculation': [0.0, 0.05, 0.40, 0.70, 1.00],
        }

    def get_system_damage_states(self):
        """
        Return a list of the damage state labels
        :return: List of strings detailing the system damage levels.
        """
        self.sys_dmg_states = ["DS0 None",
                               "DS1 Slight",
                               "DS2 Moderate",
                               "DS3 Extensive",
                               "DS4 Complete"]
        return self.sys_dmg_states

    def get_dmg_scale_bounds(self):
        return [0.01, 0.05, 0.40, 0.70, 1.00]


class PotableWaterPumpStation(Infrastructure):
    def __init__(self, **kwargs):
        super(PotableWaterPumpStation, self).__init__(**kwargs)
        # Initiate the water treatment plant values, which have been used
        # in all current testing
        self.uncosted_classes = ['SYSTEM INPUT',
                                 'SYSTEM OUTPUT',
                                 'JUNCTION POINT', 'JUNCTION',
                                 'MODEL ARTEFACT']
        self.ds_lims_compclasses = {
            'SYSTEM INPUT':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'SYSTEM OUTPUT':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Generator':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Basin':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Building':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Chlorination Equipment':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Electric Power Equipment':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Electric Power Supply Commercial':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Electric Power Supply Backup':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Electrical Control Equipment':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Pump':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Pipework':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Chemical Tank':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Concrete Tank':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Steel Tank':
                [0.0, 0.05, 0.40, 0.70, 1.00],
            'Well':
                [0.0, 0.05, 0.40, 0.70, 1.00],
        }

    def get_system_damage_states(self):
        """
        Return a list of the damage state labels
        :return: List of strings detailing the system damage levels.
        """
        self.sys_dmg_states = ["DS0 None",
                               "DS1 Slight",
                               "DS2 Moderate",
                               "DS3 Extensive",
                               "DS4 Complete"]
        return self.sys_dmg_states


class ModelTestStructure(Infrastructure):
    def __init__(self, **kwargs):
        super(ModelTestStructure, self).__init__(**kwargs)
        self.uncosted_classes = ['SYSTEM INPUT',
                                 'SYSTEM OUTPUT',
                                 'JUNCTION', 'JUNCTION POINT',
                                 'MODEL ARTEFACT']
        self.ds_lims_compclasses = {
            'Boiler': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Control Building': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Emission Management': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Fuel Delivery and Storage': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Fuel Movement': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Generator': [0.0, 0.05, 0.40, 0.70, 1.00],
            'SYSTEM OUTPUT': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Stepup Transformer': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Turbine': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Water System': [0.0, 0.05, 0.40, 0.70, 1.00]
        }

    def get_system_damage_states(self):
        """
        Return a list of the damage state labels
        :return: List of strings detailing the system damage levels.
        """
        self.sys_dmg_states = ["DS0 None",
                               "DS1 Slight",
                               "DS2 Moderate",
                               "DS3 Extensive",
                               "DS4 Complete"]
        return self.sys_dmg_states


class RailNetwork(Infrastructure):
    def __init__(self, **kwargs):
        super(RailNetwork, self).__init__(**kwargs)
        self.uncosted_classes = [
            'SYSTEM INPUT',
            'SYSTEM OUTPUT',
            'JUNCTION',
            'JUNCTION POINT',
            'MODEL ARTEFACT'
        ]
        self.ds_lims_compclasses = {
            'SYSTEM OUTPUT': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Regional Railway Station': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Urban Station': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Control Building': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Backup Power Supply': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Commercial Electricity Supply': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Commercial Water Supply': [0.0, 0.05, 0.40, 0.70, 1.00],
            'Rail Line Segment': [0.0, 0.05, 0.40, 0.70, 1.00]
        }

    def get_system_damage_states(self):
        """
        Return a list of the damage state labels
        :return: List of strings detailing the system damage levels.
        """
        self.sys_dmg_states = ["DS0 None",
                               "DS1 Slight",
                               "DS2 Moderate",
                               "DS3 Extensive",
                               "DS4 Complete"]
        return self.sys_dmg_states
