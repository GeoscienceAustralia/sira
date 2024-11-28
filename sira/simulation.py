"""simulation.py
This module applies a Monte Carlo process to calculate system output
levels and losses for given hazard levels and given number of iterations.
"""

import traceback
import numpy as np
import multiprocessing as mp
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

rootLogger = logging.getLogger(__name__)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# def calculate_response_for_hazard(hazard, scenario, infrastructure):
#     """
#     Exposes the components of the infrastructure to a hazard level
#     within a scenario.
#     :param infrastructure: containing for components
#     :param hazard: The hazard  that the infrastructure is to be exposed to.
#     :param scenario: The parameters for the scenario being simulated.
#     :return: The state of the infrastructure after the exposure.
#     """

#     # calculate the damage state probabilities
#     expected_damage_state_of_components_for_n_simulations = \
#         calc_component_damage_state_for_n_simulations(
#             infrastructure, scenario, hazard)

#     # Calculate the component loss, functionality, output,
#     #  economic loss and recovery output over time
#     (
#         component_sample_loss,
#         comp_sample_func,
#         infrastructure_sample_output,
#         infrastructure_sample_economic_loss
#     ) = \
#         infrastructure.calc_output_loss(
#             scenario,
#             expected_damage_state_of_components_for_n_simulations)

#     # Construct the dictionary containing the statistics of the response
#     (
#         component_response_dict,
#         comptype_response_dict,
#         compclass_dmg_level_percentages,
#         compclass_dmg_index_expected
#     ) = \
#         infrastructure.calc_response(
#             component_sample_loss,
#             comp_sample_func,
#             expected_damage_state_of_components_for_n_simulations
#     )

#     # determine average output for the output components
#     infrastructure_output = {}
#     output_nodes = infrastructure.output_nodes.keys()
#     for output_index, output_comp_id in enumerate(output_nodes):
#         infrastructure_output[output_comp_id] = np.mean(
#             infrastructure_sample_output[:, output_index])

#     # We combine the result data into a dictionary for ease of use
#     response_for_a_hazard = {
#         hazard.hazard_event_id: [
#             expected_damage_state_of_components_for_n_simulations,
#             infrastructure_output,
#             component_response_dict,
#             comptype_response_dict,
#             infrastructure_sample_output,
#             infrastructure_sample_economic_loss,
#             compclass_dmg_level_percentages,
#             compclass_dmg_index_expected
#         ]
#     }
#     return response_for_a_hazard

# # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

def calculate_response_for_single_hazard(hazard, scenario, infrastructure):
    """
    Exposes the components of the infrastructure to a hazard level
    within a scenario.

    Parameters:
    -----------
    infrastructure: `Infrastructure` object
        container for components
    hazard : `Hazard` object
        The hazard that the infrastructure is to be exposed to.
    scenario : `Scenario` object
        The parameters for the scenario being simulated.

    Returns:
    --------
    response_for_a_hazard : list of calculated 'responses' to hazard for system
        The state of the infrastructure after the exposure.
    """

    # Calculate the damage state probabilities
    expected_damage_state_of_components_for_n_simulations = \
        calc_component_damage_state_for_n_simulations(
            infrastructure, scenario, hazard)

    # Calculate the component loss, functionality, output, economic loss,
    #   and recovery output over time
    (
        component_sample_loss,
        comp_sample_func,
        infrastructure_sample_output,
        infrastructure_sample_economic_loss
    ) = infrastructure.calc_output_loss(
        scenario,
        expected_damage_state_of_components_for_n_simulations)

    # Construct the dictionary containing the statistics of the response
    (
        component_response_dict,
        comptype_response_dict,
        compclass_dmg_level_percentages,
        compclass_dmg_index_expected
    ) = infrastructure.calc_response(
        component_sample_loss,
        comp_sample_func,
        expected_damage_state_of_components_for_n_simulations
    )

    # Determine average output for the output components
    infrastructure_output = {}
    output_nodes = infrastructure.output_nodes.keys()
    for output_index, output_comp_id in enumerate(output_nodes):
        infrastructure_output[output_comp_id] = np.mean(
            infrastructure_sample_output[:, output_index])

    # Combine the result data into a dictionary for ease of use
    response_for_a_hazard = {
        hazard.hazard_event_id: [
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


def process_hazard_chunk(chunk_data):
    """
    Process a chunk of hazards
    """
    hazard_chunk, scenario, infrastructure = chunk_data
    chunk_response = {}
    for hazard in hazard_chunk:
        result = calculate_response_for_single_hazard(hazard, scenario, infrastructure)
        chunk_response.update(result)
    return chunk_response


def calculate_response(hazards, scenario, infrastructure):
    """
    The response will be calculated by creating the hazard_levels,
    iterating through the range of hazards and calling the infrastructure
    systems expose_to method.
    It returns the results of the infrastructure to each hazard event.

    A parameter in the scenario file determines whether the parallelising
    function spawns threads that will perform parallel calculations.

    :param scenario: Object for simulation parameters.
    :param infrastructure: Object of infrastructure model.
    :param hazards: `HazardsContainer` object.
    :return: List of results for each hazard level.
    """

    # ---------------------------------------------------------------------------------
    # capture the results from the map call in a list

    hazards_response = []
    rootLogger.info(
        "Initiating calculation of component damage states for hazard event set."
    )

    if scenario.run_parallel_proc:
        # --- Parallel execution ---
        rootLogger.info("Starting parallel run")

        # Convert hazards to list and prepare chunks
        hazard_list = list(hazards.listOfhazards)
        num_processes = mp.cpu_count()
        total_hazards = len(hazard_list)

        # Calculate chunk size - ensure at least 1 hazard per chunk
        chunk_size = max(1, total_hazards // (num_processes * 4))

        # Create chunks
        hazard_chunks = [
            hazard_list[i:i + chunk_size]
            for i in range(0, len(hazard_list), chunk_size)
        ]

        # Prepare chunk data
        chunk_data = [
            (chunk, scenario, infrastructure)
            for chunk in hazard_chunks
        ]

        rootLogger.info(f"Allocating {len(hazard_chunks)} chunks across {num_processes} processes")

        # Setup progress bar
        print()
        with tqdm(total=total_hazards, desc="Processing hazards", unit="hazard") as pbar:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit chunks for processing
                futures = []
                for data in chunk_data:
                    future = executor.submit(process_hazard_chunk, data)
                    futures.append(future)

                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        chunk_result = future.result()
                        hazards_response.append(chunk_result)
                        # Update progress bar
                        pbar.update(chunk_size)
                    except Exception as e:
                        rootLogger.error(f"Chunk processing failed: {str(e)}")
                        rootLogger.error(f"Exception details: {traceback.format_exc()}")
                        exit()
                        raise
        print()
        rootLogger.info("Completed parallel run")

    else:
        # --- Sequential execution ---
        rootLogger.info("Start serial run")
        print()
        for hazard in tqdm(hazards.listOfhazards, desc="Processing hazards"):
            hazards_response.append(
                calculate_response_for_single_hazard(hazard, scenario, infrastructure))
        print()
        rootLogger.info("End serial run")

    # ---------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------
    # iterate through the hazards
    # ----------------------------------------------------------------------------------

    for hazard_response in hazards_response:
        for key, value_list in hazard_response.items():
            for list_number, _ in enumerate(post_processing_list):
                if (list_number <= 3) or (list_number in [6, 7]):
                    post_processing_list[list_number][key] = value_list[list_number]
                else:
                    post_processing_list[list_number].append(value_list[list_number])

    # Convert the two lists to arrays
    for list_number in range(4, 6):
        post_processing_list[list_number] = np.array(post_processing_list[list_number])

    # ---------------------------------------------------------------------------------
    # Convert the calculated output array into the correct format
    system_output_arr_per_sample = post_processing_list[4]
    post_processing_list[4] = np.sum(system_output_arr_per_sample, axis=2).transpose()
    system_econ_loss_arr_per_sample = np.array(post_processing_list[5])
    post_processing_list[5] = system_econ_loss_arr_per_sample.transpose()

    return post_processing_list


# def progress_monitor(progress_queue, total_components, update_interval=10):
#     """
#     Monitor progress queue and update progress bar.
#     Runs in a separate thread to avoid blocking the main process.
#     """
#     pbar = tqdm(total=total_components, desc="Processing components")
#     last_count = 0

#     while True:
#         try:
#             # Non-blocking queue check
#             msg = progress_queue.get_nowait()
#             if msg == "DONE":
#                 pbar.update(total_components - last_count)  # Ensure we reach 100%
#                 pbar.close()
#                 break
#             # Update progress bar with number of components processed
#             completed = msg - last_count
#             pbar.update(completed)
#             last_count = msg
#         except Empty:
#             time.sleep(update_interval)  # Reduce CPU usage
#             continue


def process_component_batch(batch_data, infrastructure, scenario, hazard, rnd):
    """
    Process a batch of components - moved to module level for pickling
    """
    start_idx, component_keys = batch_data
    batch_size = len(component_keys)
    batch_damage_states = np.zeros((scenario.num_samples, batch_size), dtype=int)

    for local_idx, component_key in enumerate(component_keys):
        component = infrastructure.components[component_key]
        global_idx = start_idx + local_idx

        # Calculate damage states
        component_pe_ds = np.zeros(len(component.damage_states))

        for damage_state_index in component.damage_states.keys():
            loc_params = component.get_location()
            hazard_intensity = hazard.get_hazard_intensity(*loc_params)
            component_pe_ds[damage_state_index] = \
                component.damage_states[damage_state_index].response_function(
                    hazard_intensity)

        component_pe_ds = component_pe_ds[1:]
        component_pe_ds[np.isnan(component_pe_ds)] = -np.inf

        batch_damage_states[:, local_idx] = \
            np.sum(component_pe_ds > rnd[:, global_idx][:, np.newaxis], axis=1)

    return start_idx, batch_damage_states


def calc_component_damage_state_for_n_simulations(infrastructure, scenario, hazard):
    """
    Calculate the probability that being exposed to a hazard level
    will exceed the given damage levels for each component. A monte
    carlo approach is taken by simulating the exposure for the number
    of samples given in the scenario.

    Parameters:
        infrastructure: Container for components
        scenario: Parameters for the scenario
        hazard: Level of the hazard
    Returns:
        numpy.ndarray: Array of damage state indices for each sample and component
    """
    if scenario.run_context:
        random_number = np.random.RandomState(seed=hazard.get_seed())
    else:
        random_number = np.random.RandomState(seed=2)

    number_of_components = len(infrastructure.components)
    component_damage_state_ind = np.zeros(
        (scenario.num_samples, number_of_components), dtype=int)

    # Generate all random numbers at once
    rnd = random_number.uniform(size=(scenario.num_samples, number_of_components))

    if number_of_components >= 150:
        component_keys = sorted(infrastructure.components.keys())
        num_cores = mp.cpu_count()
        batch_size = max(100, number_of_components // (num_cores * 4))

        # Prepare batches
        batches = []
        for start_idx in range(0, number_of_components, batch_size):
            end_idx = min(start_idx + batch_size, number_of_components)
            batch_keys = component_keys[start_idx:end_idx]
            batches.append((start_idx, batch_keys))

        # Process batches in parallel with progress bar
        print(f"\nComponent processing in batch of {batch_size}...")
        with tqdm(total=number_of_components, desc="Processing components ") as pbar:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Submit all batches for processing
                futures = []
                for batch in batches:
                    future = executor.submit(
                        process_component_batch,
                        batch,
                        infrastructure,
                        scenario,
                        hazard,
                        rnd
                    )
                    futures.append(future)

                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        start_idx, batch_results = future.result()
                        batch_size = batch_results.shape[1]
                        component_damage_state_ind[:, start_idx:start_idx + batch_size] = \
                            batch_results
                        pbar.update(batch_size)
                    except Exception as exc:
                        rootLogger.error(f"Batch processing failed: {str(exc)}")
                        raise

    else:
        # Sequential processing for smaller component sets
        for index, component_key in enumerate(tqdm(
            sorted(infrastructure.components.keys()),
            desc="Processing components"
        )):
            component = infrastructure.components[component_key]
            component_pe_ds = np.zeros(len(component.damage_states))

            for damage_state_index in component.damage_states.keys():
                loc_params = component.get_location()
                hazard_intensity = hazard.get_hazard_intensity(*loc_params)
                component_pe_ds[damage_state_index] = \
                    component.damage_states[damage_state_index].response_function(
                        hazard_intensity)

            # ======================================================================
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
            # ======================================================================

            component_pe_ds = component_pe_ds[1:]
            component_pe_ds[np.isnan(component_pe_ds)] = -np.inf

            component_damage_state_ind[:, index] = \
                np.sum(component_pe_ds > rnd[:, index][:, np.newaxis], axis=1)

    return component_damage_state_ind
