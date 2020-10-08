"""simulation.py
This module applies a Monte Carlo process to calculate system output
levels and losses for given hazard levels and given number of iterations.
"""

import logging
import time
from datetime import timedelta

import numpy as np

# import parmap
rootLogger = logging.getLogger(__name__)


def calculate_response(hazards, scenario, infrastructure):
    """
    The response will be calculated by creating the hazard_levels,
    iterating through the range of hazards and calling the infrastructure
    systems expose_to method. This will return the results of the
    infrastructure to each hazard level exposure. A parameter in the
    scenario file determines whether the parmap.map function spawns threads
    that will perform parallel calculations.
    :param scenario: Parameters for the simulation.
    :param infrastructure: Model of the infrastructure.
    :param hazards: hazards container.
    :return: List of results for each hazard level.
    """

    # code_start_time = time.time() # start of the overall response calculation
    # capture the results from the map call in a list
    hazards_response = []

    # # Use the parallel option in the scenario to determine how to run
    # if scenario.run_parallel_proc:
    #     rootLogger.info("Start parallel run")
    #     hazards_response.extend(
    #         parmap.map(calculate_response_for_hazard,
    #         hazards.get_listOfhazards(),
    #         scenario,
    #         infrastructure,
    #         parallel=scenario.run_parallel_proc))
    #     rootLogger.info("End parallel run")
    # else:
    #     rootLogger.info("Start serial run")
    #     for hazard in hazards.listOfhazards:
    #         hazards_response.append(
    #             calculate_response_for_hazard(
    #                 hazard, scenario, infrastructure))
    #     rootLogger.info("End serial run")
    rootLogger.info("Start serial run")
    for hazard in hazards.listOfhazards:
        hazards_response.append(
            calculate_response_for_hazard(hazard, scenario, infrastructure))

    # combine the responses into one list
    post_processing_list = [
        {},  # [0] hazard level vs component damage state index
        {},  # [1] hazard level vs infrastructure output
        {},  # [2] hazard level vs component response
        {},  # [3] hazard level vs component type response
        [],  # [4] array of infrastructure output per sample
        [],  # [5] array of infrastructure econ loss per sample
        {},  # [6] hazard level vs component class dmg level pct
        {}   # [7] hazard level vs component class expected damage index
    ]
    # iterate through the hazards
    for hazard_response in hazards_response:
        # iterate through the hazard response dictionary
        for key, value_list in hazard_response.items():
            # for list_number in range(len(post_processing_list)):
            for list_number, _ in enumerate(post_processing_list):
                # the first four are dicts
                if list_number <= 3:
                    post_processing_list[list_number][key] \
                        = value_list[list_number]
                elif list_number in [6, 7]:
                    post_processing_list[list_number][key] \
                        = value_list[list_number]
                else:
                    # the last two are lists
                    post_processing_list[list_number]. \
                        append(value_list[list_number])

    # Convert the last 2 lists into arrays
    for list_number in range(4, 6):
        post_processing_list[list_number] \
            = np.array(post_processing_list[list_number])

    # Convert the calculated output array into the correct format
    system_output_arr_per_sample = post_processing_list[4]
    post_processing_list[4] = \
        np.sum(system_output_arr_per_sample, axis=2).transpose()

    system_econ_loss_arr_per_sample = np.array(post_processing_list[5])
    post_processing_list[5] = system_econ_loss_arr_per_sample.transpose()

    # elapsed = timedelta(seconds=(time.time() - code_start_time))
    # logging.info("[ Run time: %s ]\n" % str(elapsed))

    return post_processing_list


def calculate_response_for_hazard(hazard, scenario, infrastructure):
    """
    Exposes the components of the infrastructure to a hazard level
    within a scenario.
    :param infrastructure: containing for components
    :param hazard: The hazard  that the infrastructure is to be exposed to.
    :param scenario: The parameters for the scenario being simulated.
    :return: The state of the infrastructure after the exposure.
    """

    # keep track of the length of time the exposure takes
    code_start_time = time.time()

    # calculate the damage state probabilities
    rootLogger.info(
        "Initiating damage state calculations for: %s",
        hazard.hazard_scenario_name)
    expected_damage_state_of_components_for_n_simulations = \
        calc_component_damage_state_for_n_simulations(
            infrastructure, scenario, hazard)

    # calculate the component loss, functionality, output,
    #  economic loss and recovery output over time
    (component_sample_loss,
     comp_sample_func,
     infrastructure_sample_output,
     infrastructure_sample_economic_loss) = \
        infrastructure.calc_output_loss(
            scenario,
            expected_damage_state_of_components_for_n_simulations)

    # Construct the dictionary containing the statistics of the response
    (component_response_dict,
     comptype_response_dict,
     compclass_dmg_level_percentages,
     compclass_dmg_index_expected) = \
        infrastructure.calc_response(
            component_sample_loss,
            comp_sample_func,
            expected_damage_state_of_components_for_n_simulations)

    # determine average output for the output components
    rootLogger.debug(
        "Calculating system output for hazard %s",
        hazard.hazard_scenario_name)
    infrastructure_output = {}
    output_nodes = infrastructure.output_nodes.keys()
    for output_index, output_comp_id in enumerate(output_nodes):
        infrastructure_output[output_comp_id] = np.mean(
            infrastructure_sample_output[:, output_index])

    # log the elapsed time for this hazard level
    elapsed = timedelta(seconds=(time.time() - code_start_time))
    rootLogger.info(
        "Run time for hazard scenario %s : %s",
        hazard.hazard_scenario_name, str(elapsed)
    )

    # We combine the result data into a dictionary for ease of use
    response_for_a_hazard = {
        hazard.hazard_scenario_name: [
            expected_damage_state_of_components_for_n_simulations,
            infrastructure_output,
            component_response_dict,
            comptype_response_dict,
            infrastructure_sample_output,
            infrastructure_sample_economic_loss,
            compclass_dmg_level_percentages,
            compclass_dmg_index_expected
        ]
    }
    return response_for_a_hazard


def calc_component_damage_state_for_n_simulations(
        infrastructure, scenario, hazard):
    """
    Calculate the probability that being exposed to a hazard level
    will exceed the given damage levels for each component. A monte
    carlo approach is taken by simulating the exposure for the number
    of samples given in the scenario.
    :param infrastructure: containing for components
    :param hazard: Level of the hazard
    :param scenario: Parameters for the scenario
    :return: An array of the probability that each of the damage states
             were exceeded.
    """
    if scenario.run_context:
        # use seed for actual runs or not
        random_number = np.random.RandomState(seed=hazard.get_seed())
    else:
        # seeding was not used
        random_number = np.random.RandomState(seed=2)

    # record the number of component in infrastructure
    number_of_components = len(infrastructure.components)

    # construct a zeroed numpy array that can contain the number of samples
    # for each element.
    component_damage_state_ind = np.zeros(
        (scenario.num_samples, number_of_components), dtype=int)

    # create numpy array of uniformly distributed random numbers between (0,1)
    rnd = random_number.uniform(
        size=(scenario.num_samples, number_of_components))
    rootLogger.info("Hazard Intensity %s", hazard.hazard_scenario_name)

    # iterate through the components
    rootLogger.info("Calculating component damage state probability.")
    for index, component_key in enumerate(
            sorted(infrastructure.components.keys())):
        component = infrastructure.components[component_key]
        # use the components expose_to method to retrieve the probabilities
        # of this hazard level exceeding each of the components damage levels

        # create numpy array of length equal to the number of
        # damage states for the component
        component_pe_ds = np.zeros(len(component.damage_states))

        # iterate through each damage state for the component
        rootLogger.info("Calculating Component Response...")
        for damage_state_index in component.damage_states.keys():
            # find the hazard intensity component is exposed too
            pos_x, pos_y = component.get_location()
            hazard_intensity = hazard.get_hazard_intensity(
                pos_x, pos_y)

            component_pe_ds[damage_state_index] \
                = component.damage_states[damage_state_index].\
                response_function(hazard_intensity)

        # Drop the default state (DS0 None) because its response will
        # always be zero which will be always be greater than random variable
        component_pe_ds = component_pe_ds[1:]

        rootLogger.info(
            "PE_DS for Component: %s  %r",
            component.component_id, component_pe_ds)

        # This little piece of numpy magic calculates the damage level by
        # summing how many damage states were exceeded.
        #
        # Unpacking the calculation:
        # component_pe_ds is usually something like [0.01, 0.12, 0.21, 0.33]
        # rnd[:, index] gives the random numbers created for this component
        # with the first axis (denoted by :) containing the samples for this
        # hazard intensity. The [:, np.newaxis] broadcasts each
        # random number across the supplied component_pe_ds.
        #
        # LINK:
        # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
        #
        # If the last two numbers in component_pe_ds are greater than the
        # sample number, the comparison > will return:
        #       [False, False, True, True]
        # The np.sum will convert this to [0, 0, 1, 1] and return 2.
        #
        # This is the resulting damage level.
        # This will complete the comparison for all of the samples
        # for this component.

        component_pe_ds[np.isnan(component_pe_ds)] = -np.inf
        component_damage_state_ind[:, index] = \
            np.sum(component_pe_ds > rnd[:, index][:, np.newaxis], axis=1)

    return component_damage_state_ind
