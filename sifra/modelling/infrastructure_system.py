import numpy as np
import time
from datetime import timedelta
import logging

# these are required for defining the data model
from sifra.modelling.structural import (
    Element,
    Info)

from sifra.modelling.component import Component
from sifra.modelling.component_graph import ComponentGraph
from sifra.modelling.infrastructure_model import Model
from sifra.modelling.iodict import IODict


class IFSystem(Model):
    """
    The top level representation of a system that can respond to a
    range of hazards. It encapsulates a number of components that are
    used to estimate the response to various hazard levels.
    """

    supply_nodes = Element('dict', 'The components that supply the infrastructure system', dict)
    output_nodes = Element('dict', 'The components that output from the infrastructure system', dict)

    supply_total = None
    component_graph = None
    if_nominal_output = None
    system_class = None

    sys_dmg_states = ['DS0 None',
                      'DS1 Slight',
                      'DS2 Moderate',
                      'DS3 Extensive',
                      'DS4 Complete']

    def __init__(self, **kwargs):
        """
        Construct the infrastructure object
        :param kwargs: Objects making up the infrastructure
        """
        super(IFSystem, self).__init__(**kwargs)

        if not self.components:
            self.components = IODict()

        self.component_graph = ComponentGraph(self.components)

        # TODO This assignment should be replaced by subclasses of IFSystem
        # instantiated directly by the model creator
        # This initiates some instance members with the correct configuration.
        # Not sure why these aren't in the configuration file
        # TODO Need to move these damage state limits into config file (MR)

        if self.system_class.lower() == 'substation':
            # Initiate the substation, note: this may not have been tested in this
            # version of the code.
            self.uncosted_classes = ['JUNCTION POINT',
                                     'SYSTEM INPUT', 'SYSTEM OUTPUT',
                                     'Generator', 'Bus', 'Lightning Arrester']
            self.ds_lims_compclasses = {
                'Disconnect Switch': [0.05, 0.40, 0.70, 0.99, 1.00],
                'Circuit Breaker': [0.05, 0.40, 0.70, 0.99, 1.00],
                'Current Transformer': [0.05, 0.40, 0.70, 0.99, 1.00],
                'Voltage Transformer': [0.05, 0.40, 0.70, 0.99, 1.00],
                'Power Transformer': [0.05, 0.40, 0.70, 0.99, 1.00],
                'Control Building': [0.06, 0.30, 0.75, 0.99, 1.00]
            }
        elif self.system_class.lower() == 'powerstation':
            # Initiate the power station values, which have been used in all current
            # testing
            self.uncosted_classes = ['JUNCTION POINT', 'SYSTEM INPUT', 'SYSTEM OUTPUT']
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

    def expose_to(self, hazard_level, scenario):
        """
        Exposes the components of the infrastructure to a hazard level
        within a scenario.
        :param hazard_level: The hazard level that the infrastructure is to be exposed to.
        :param scenario: The parameters for the scenario being simulated.
        :return: The state of the infrastructure after the exposure.
        """

        code_start_time = time.time() # keep track of the length of time the exposure takes

        # calculate the damage state probabilities
        component_damage_state_ind = self.probable_ds_hazard_level(hazard_level, scenario)

        # calculate the component loss, functionality, output,
        #  economic loss and recovery output over time
        component_sample_loss, \
        comp_sample_func, \
        if_sample_output, \
        if_sample_economic_loss, \
        if_output_given_recovery = self.calc_output_loss(scenario, component_damage_state_ind)

        # Construct the dictionary containing the statisitics of the response
        component_response = self.calc_response(component_sample_loss,
                                                comp_sample_func,
                                                component_damage_state_ind)

        # determine average output for the output components
        if_output = {}
        for output_index, (output_comp_id, output_comp) in enumerate(self.output_nodes.iteritems()):
            if_output[output_comp_id] = np.mean(if_sample_output[:, output_index])

        # log the elapsed time for this hazard level
        elapsed = timedelta(seconds=(time.time() - code_start_time))
        logging.info("Hazard {} run time: {}".format(hazard_level.hazard_intensity,
                                                     str(elapsed)))

        # We combine the result data into a dictionary for ease of use
        response_dict = {hazard_level.hazard_intensity: [component_damage_state_ind,
                                                         if_output,
                                                         component_response,
                                                         if_sample_output,
                                                         if_sample_economic_loss,
                                                         if_output_given_recovery]}

        return response_dict

    def probable_ds_hazard_level(self, hazard_level, scenario):
        """
        Calculate the probability that being exposed to a hazard level
        will exceed the given damage levels for each component. A monte
        carlo approach is taken by simulating the exposure for the number
        of samples given in the scenario.
        :param hazard_level: Level of the hazard
        :param scenario: Parameters for the scenario
        :return: An array of the probability that each of the damage states were exceeded.
        """
        if scenario.run_context:
            # Use seeding for this test run for reproducibility, the seeding
            # is generated by converting the hazard intensity to an integer
            # after shifting by two decimal places.
            prng = np.random.RandomState(int(hazard_level.hazard_intensity*100))
        else:
            # seeding was not used
            prng = np.random.RandomState()

        # record the number of elements for use
        num_elements = len(self.components)

        # construct a zeroed numpy array that can contain the number of samples for
        # each element.
        component_damage_state_ind = np.zeros((scenario.num_samples, num_elements),
                                              dtype=int)
        # create another numpy array of random uniform [0,1.0) numbers.
        rnd = prng.uniform(size=(scenario.num_samples, num_elements))
        logging.debug("Hazard Intensity {}".format(hazard_level.hazard_intensity))
        # iterate through the components
        for index, comp_key in enumerate(sorted(self.components.keys())):
            component = self.components[comp_key]
            # use the components expose_to method to retrieve the probabilities
            # of this hazard level exceeding each of the components damage levels
            component_pe_ds = np.sort(component.expose_to(hazard_level, scenario))
            logging.debug("Component {} : pe_ds {}".format(component.component_id,
                                                           component_pe_ds))

            # This little piece of numpy magic calculates the damage level by summing
            # how many damage states were exceeded.
            # Unpacking the calculation:
            # component_pe_ds is usually something like [0.01, 0.12, 0.21, 0.33]
            # rnd[:, index] gives the random numbers created for this component
            # with the first axis (denoted by :) containing the samples for this
            # hazard intensity. The [:, np.newaxis] broadcasts
            # (https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)
            # each randomn number across the supplied component_pe_ds. If the sample number
            # is greater than the first two numbers in component_pe_ds, the comparison > will
            # return [True, True, False, False]
            # the np.sum will convert this to [1, 1, 0, 0] and return 2. This is the resulting
            # damage level. This will complete the comparison for all of the samples
            # for this component
            component_damage_state_ind[:, index] = \
                np.sum(component_pe_ds > rnd[:, index][:, np.newaxis], axis=1)

        return component_damage_state_ind

    def calc_output_loss(self, scenario, component_damage_state_ind):
        """
        Calculate the results to the infrastructure given the damage state
        parameter
        :param scenario: Details of the scenario being run
        :param component_damage_state_ind: The array of the component's damage state samples
        :return: 5 lists of calculations
        """
        # Component loss caused by the damage
        if_level_loss = np.zeros((scenario.num_samples, len(self.components)),
                                 dtype=np.float64)
        # Infrastructure loss: sum of component loss
        if_level_economic_loss = np.zeros(scenario.num_samples,
                                          dtype=np.float64)
        # Component functionality
        if_level_functionality = np.zeros((scenario.num_samples, len(self.components)),
                                          dtype=np.float64)
        # output for the level of damage
        if_level_output = np.zeros((scenario.num_samples, len(self.output_nodes)),
                                   dtype=np.float64)
        # output available as recovery progresses
        if_output_given_recovery = np.zeros((scenario.num_samples, scenario.num_time_steps), dtype=np.float64)

        # iterate through the samples
        for sample_index in range(scenario.num_samples):
            # initialise the function and loss arrays for the sample
            component_function_at_time = []
            comp_sample_loss = np.zeros(len(self.components))
            comp_sample_func = np.zeros(len(self.components))
            # Extract the array of damage states for this sample
            component_ds = component_damage_state_ind[sample_index, :]
            # iterate through the components
            for component_index, comp_key in enumerate(sorted(self.components.keys())):
                component = self.components[comp_key]
                # get the damage state for the component
                damage_state = component.get_damage_state(component_ds[component_index])
                # use the damage state attributes to calculate the loss and functionality for the component sample
                loss = damage_state.damage_ratio * component.cost_fraction
                comp_sample_loss[component_index] = loss
                comp_sample_func[component_index] = damage_state.functionality
                # calculate the recovery time
                component_function_at_time.append(self.calc_recov_time_given_comp_ds(component,
                                                                                     component_ds[component_index],
                                                                                     scenario))
            # save this sample's component loss and functionality
            if_level_loss[sample_index, :] = comp_sample_loss
            if_level_functionality[sample_index, :] = comp_sample_func
            # the infrastructure's economic loss for this sample is the sum of all
            # component losses
            if_level_economic_loss[sample_index] = np.sum(comp_sample_loss)
            # estimate the output for this sample's component functionality
            if_level_output[sample_index, :] = self.compute_output_given_ds(comp_sample_func)

            # calculate the restoration output
            component_function_at_time = np.array(component_function_at_time)
            for time_step in range(scenario.num_time_steps):
                if_output_given_recovery[sample_index, time_step] = \
                    sum(self.compute_output_given_ds(component_function_at_time[:, time_step]))

        return if_level_loss, \
               if_level_functionality, \
               if_level_output, \
               if_level_economic_loss, \
               if_output_given_recovery

    def get_nominal_output(self):
        """
        Estimate the output of the undamaged infrastructure output
        nodes.
        :return: Numeric value of aggregated output.
        """
        if not self.if_nominal_output:
            self.if_nominal_output = 0
            for output_comp_id, output_comp in self.output_nodes.iteritems():
                self.if_nominal_output += output_comp['output_node_capacity']

        return self.if_nominal_output

    def compute_output_given_ds(self, comp_level_func):
        """
        Using the graph of components, the output is calculated
        from the component functionality parameter.
        :param comp_level_func: An array that indicates the functionality level for each component.
        :return: An array of the output level for each output node.
        """
        # Create the component graph if one does not yet exist
        if not self.component_graph:
            self.component_graph = ComponentGraph(self.components, comp_level_func)
        else:
            self.component_graph.update_capacity(self.components, comp_level_func)

        # calculate the capacity
        system_flows_sample = []
        system_outflows_sample = np.zeros(len(self.output_nodes))
        for output_index, (output_comp_id, output_comp) in enumerate(self.output_nodes.iteritems()):
            # track the outputs by source type (e.g. water or coal)
            total_supply_flow_by_source = {}
            for supply_index, (supply_comp_id, supply_comp) in enumerate(self.supply_nodes.iteritems()):
                if_flow_fraction = self.component_graph.maxflow(supply_comp_id, output_comp_id)
                if_sample_flow = if_flow_fraction * supply_comp['capacity_fraction']

                if supply_comp['commodity_type'] not in total_supply_flow_by_source:
                    total_supply_flow_by_source[supply_comp['commodity_type']] = if_sample_flow
                else:
                    total_supply_flow_by_source[supply_comp['commodity_type']] += if_sample_flow

            total_available_flow = min(total_supply_flow_by_source.itervalues())

            estimated_capacity_fraction = min(total_available_flow, output_comp['capacity_fraction'])
            system_outflows_sample[output_index] = estimated_capacity_fraction * self.get_nominal_output()

        return system_outflows_sample

    def calc_recov_time_given_comp_ds(self, component, damage_state, scenario):
        '''
        Calculates the recovery time of a component, given damage state index
        '''
        import scipy.stats as stats
        recovery_parameters = component.get_recovery(damage_state)
        damage_parameters = component.get_damage_state(damage_state)

        m = recovery_parameters.recovery_mean
        s = recovery_parameters.recovery_std
        fn = damage_parameters.functionality
        cdf = stats.norm.cdf(scenario.restoration_time_range, loc=m, scale=s)
        return cdf + (1.0 - cdf) * fn

    def calc_response(self, component_loss, comp_sample_func, component_damage_state_ind):
        """
        Convert the arrays into dicts for subsequent analysis
        :param component_loss: The array of component loss values
        :param comp_sample_func: The array of component functionality values
        :param component_damage_state_ind: The array of component damage state indicators
        :return: A dict of component response statistics
        """
        comp_resp_dict = dict()

        for comp_index, comp_id in enumerate(np.sort(self.components.keys())):
            component = self.components[comp_id]
            comp_resp_dict[(comp_id, 'loss_mean')] \
                = np.mean(component_loss[:, comp_index])

            comp_resp_dict[(comp_id, 'loss_std')] \
                = np.std(component_loss[:, comp_index])

            comp_resp_dict[(comp_id, 'func_mean')] \
                = np.mean(comp_sample_func[:, comp_index])

            comp_resp_dict[(comp_id, 'func_std')] \
                = np.std(comp_sample_func[:, comp_index])

            comp_resp_dict[(comp_id, 'num_failures')] \
                = np.mean(component_damage_state_ind[:, comp_index] >= (len(component.frag_func.damage_states) - 1))

        return comp_resp_dict

    def get_component_types(self):
        """
        Convenience method to get the list of components that are
        costed.

        :return: list of costed component types
        """
        uncosted_comptypes = {'CONN_NODE', 'SYSTEM_INPUT', 'SYSTEM_OUTPUT'}

        component_types = set()

        for component in self.components.itervalues():
            if component.component_type not in uncosted_comptypes:
                component_types.add(component.component_type)

        return list(component_types)

    def get_components_for_type(self, component_type):
        """
        Return a list of components for the passed component type.
        :param component_type: A string representing a component type
        :return: List of components with the matching component type.
        """
        for component in self.components.itervalues():
            if component.component_type == component_type:
                yield component.component_id

    def get_system_damage_states(self):
        """
        Return a list of the damage state labels
        :return: List of strings detailing the system damage levels.
        """
        return self.sys_dmg_states


    def get_dmg_scale_bounds(self, scenario):
        """
        An array of damage scale boundaries
        :param scenario: The values for the simulation scenarios
        :return:  Array of real numbers representing damage state boundaries
        """
        # todo introduce system subclass to infrastructure
        return [0.01, 0.15, 0.4, 0.8, 1.0]

    def get_component_class_list(self):
        """
        Return the list of component classes from the components.
        Not sure why duplicates are returned and then stripped out,
        it seems unnecessary
        :return: A generator for the list.
        """
        for component in self.components.itervalues():
            yield component.component_class

