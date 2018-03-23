import numpy as np

from sifra.configuration import Configuration
from sifra.logger import rootLogger
from sifra.modelling.hazard import Hazard
from sifra.scenario import Scenario
from sifra.model_ingest import ingest_model
import time
from datetime import timedelta


class Simulation:

    def __init__(self, configuration_file_path):

        # a variable to store a list of simulation_results

        # think weather copies of these variable need to be kept this is direct binding very bad practise
        self.config = Configuration(configuration_file_path)
        self.scenario = Scenario(self.config)
        self.infrastructure, algorithm_factory = ingest_model(self.config)
        self.hazard = Hazard(self.config)

        # variable to hold the states of all the components

    # a function to get a single simulation result ie probability of components
    # a function to loop over different types of hazard values

    # def main_container(self, configuration_file_path):
    #     """
    #     Run a scenario by constructing a facility, and executing a scenario, with
    #     the parameters read from the config file.
    #     :param configuration_file_path: Scenario setting values and the infrastructure configuration file path
    #     :return: None
    #     """
    #
    #     rootLogger.info("Loading scenario config... ")
    #     config = Configuration(configuration_file_path)
    #     rootLogger.info("Done.")
    #
    #
    #     rootLogger.info("constructing scenario... ")
    #     scenario = Scenario(config)
    #     rootLogger.info("Done.")
    #
    #     hazard_level_response = []
    #     for hazard_level in self.hazard.hazard_range:
    #
    #         hazard_level_response.append(infrastructure.expose_to(hazard_level, scenario))
    #
    #         component_damage_state_ind = self.probable_ds_hazard_level(hazard_level, scenario)
    #
    #             run_simulation(infrastructure, hazard_level, sceanario)
    #
    #         print()


def calculate_response(scenario, infrastructure, hazard):
    """
    The response will be calculated by creating the hazard_levels,
    iterating through the range of hazards and calling the infrastructure systems
    expose_to method. This will return the results of the infrastructure to each hazard level
    exposure. A parameter in the scenario file determines whether the parmap.map function spawns threads
    that will perform parallel calculations.
    :param scenario: Parameters for the simulation.
    :param infrastructure: Model of the infrastructure.
    :param hazard: hazards container.
    :return: List of results for each hazard level.
    """

    # code_start_time = time.time() # start of the overall response calculation
    # capture the results from the map call in a list
    hazard_level_response = []
    # Use the parallel option in the scenario to determine how to run

    print("Start Remove parallel run")
    for hazard_scenario_name in hazard.hazard_scenario_name:
        hazard_level_response.append(infrastructure_expose_to(infrastructure, hazard_scenario_name, scenario,hazard))
    print("End Remove parallel run")
    #
    # hazard_level_response.extend(parmap.map(run_para_scen,
    #                                         hazard_levels.hazard_range(),
    #                                         infrastructure,
    #                                         scenario,
    #                                         parallel=scenario.run_parallel_proc))


    # TODO add test case to compare the hazard response values!

    # combine the responses into one list
    post_processing_list = [{},  # hazard level vs component damage state index
                            {},  # hazard level vs infrastructure output
                            {},  # hazard level vs component response
                            [],  # infrastructure output for sample
                            []]  # infrastructure econ loss for sample

    # iterate through the hazard levels
    for hazard_level_values in hazard_level_response:
        # iterate through the hazard level lists
        for key, value_list in hazard_level_values.items():
            for list_number in range(5):
                # the first three lists are dicts
                if list_number <= 2:
                    post_processing_list[list_number][key] \
                        = value_list[list_number]
                else:
                    # the last three are lists
                    post_processing_list[list_number]. \
                        append(value_list[list_number])

    # Convert the last 3 lists into arrays
    for list_number in range(3, 5):
        post_processing_list[list_number] \
            = np.array(post_processing_list[list_number])

    # Convert the calculated output array into the correct format
    post_processing_list[3] = np.sum(post_processing_list[3], axis=2).transpose()
    post_processing_list[4] = post_processing_list[4].transpose()

    # elapsed = timedelta(seconds=(time.time() - code_start_time))
    # logging.info("[ Run time: %s ]\n" % str(elapsed))

    return post_processing_list


def infrastructure_expose_to(infrastructure, hazard_scenario_name, scenario, hazard):
    """
    Exposes the components of the infrastructure to a hazard level
    within a scenario.
    :param infrastructure: containing for components
    :param hazard_level: The hazard level that the infrastructure is to be exposed to.
    :param scenario: The parameters for the scenario being simulated.
    :return: The state of the infrastructure after the exposure.
    """

    code_start_time = time.time() # keep track of the length of time the exposure takes

    # calculate the damage state probabilities
    print("START")

    print("Calculate System Response")
    component_damage_state_ind = probable_ds_hazard_level(infrastructure, hazard_scenario_name, scenario, hazard)

    print("System Response: ")

    # calculate the component loss, functionality, output,
    #  economic loss and recovery output over time
    component_sample_loss, comp_sample_func, if_sample_output, if_sample_economic_loss \
        = infrastructure.calc_output_loss(scenario, component_damage_state_ind)

    # Construct the dictionary containing the statistics of the response
    component_response = \
        infrastructure.calc_response(component_sample_loss, comp_sample_func, component_damage_state_ind)

    # determine average output for the output components
    if_output = {}
    for output_index, (output_comp_id, output_comp) in enumerate(infrastructure.output_nodes.iteritems()):
        if_output[output_comp_id] = np.mean(if_sample_output[:, output_index])

    # log the elapsed time for this hazard level
    elapsed = timedelta(seconds=(time.time() - code_start_time))
    rootLogger.info("Hazard {} run time: {}".format(hazard_scenario_name, str(elapsed)))

    # We combine the result data into a dictionary for ease of use
    response_dict = {hazard_scenario_name: [component_damage_state_ind,
                                                     if_output,
                                                     component_response,
                                                     if_sample_output,
                                                     if_sample_economic_loss]}
    return response_dict


def probable_ds_hazard_level(infrastructure, hazard_scenario_name, scenario,hazard):
    """
    Calculate the probability that being exposed to a hazard level
    will exceed the given damage levels for each component. A monte
    carlo approach is taken by simulating the exposure for the number
    of samples given in the scenario.
    :param infrastructure: containing for components
    :param hazard_level: Level of the hazard
    :param scenario: Parameters for the scenario
    :return: An array of the probability that each of the damage states were exceeded.
    """
    # if scenario.run_context:
    #     # Use seeding for this test run for reproducibility, the seeding
    #     # is generated by converting the hazard intensity to an integer
    #     # after shifting by two decimal places.
    #     random_number = np.random.RandomState(int(hazard_level.hazard_intensity * 100))
    # else:
    #     # seeding was not used
    random_number = np.random.RandomState()

    # record the number of elements for use
    num_components = len(infrastructure.components)

    # construct a zeroed numpy array that can contain the number of samples for
    # each element.
    component_damage_state_ind = np.zeros((scenario.num_samples, num_components),
                                          dtype=int)
    # create another numpy array of random uniform [0,1.0) numbers.
    rnd = random_number.uniform(size=(scenario.num_samples, num_components))
    rootLogger.debug("Hazard Intensity {}".format(hazard_scenario_name))
    # iterate through the components
    for index, comp_key in enumerate(sorted(infrastructure.components.keys())):
        component = infrastructure.components[comp_key]
        # use the components expose_to method to retrieve the probabilities
        # of this hazard level exceeding each of the components damage levels

        print("Start calculating probability of component in a damage state.")

        component_pe_ds = np.zeros(len(component.damage_states))

        for damage_state_index in component.damage_states.keys():
            long, lat = component.get_location()
            hazard_intensity = hazard.get_hazard_intensity_at_location(hazard_scenario_name, long, lat)
            component_pe_ds[damage_state_index] = component.damage_states[damage_state_index].response_function(hazard_intensity)

        component_pe_ds = component_pe_ds[1:]
        print("Calculate System Response")
        rootLogger.debug("Component {} : pe_ds {}".format(component.component_id,
                                                          component_pe_ds))

        # This little piece of numpy magic calculates the damage level by summing
        # how many damage states were exceeded.
        # Unpacking the calculation:
        # component_pe_ds is usually something like [0.01, 0.12, 0.21, 0.33]
        # rnd[:, index] gives the random numbers created for this component
        # with the first axis (denoted by :) containing the samples for this
        # hazard intensity. The [:, np.newaxis] broadcasts
        # (https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)
        # each random number across the supplied component_pe_ds. If the last two
        # numbers in component_pe_ds are greater than the sample number, the
        # comparison > will return [False, False, True, True]
        # the np.sum will convert this to [1, 1, 0, 0] and return 2. This is the resulting
        # damage level. This will complete the comparison for all of the samples
        # for this component

        component_damage_state_ind[:, index] = \
            np.sum(component_pe_ds > rnd[:, index][:, np.newaxis], axis=1)
        print(component_damage_state_ind)
    return component_damage_state_ind
