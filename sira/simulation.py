import numpy as np
import math
import multiprocessing as mp
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback
from tqdm import tqdm
from sira.tools.parallelisation import get_available_cores

rootLogger = logging.getLogger(__name__)


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
    # Set run_parallel_proc to False to avoid nested parallelization
    # We'll store the original value and restore it later
    original_parallel_setting = scenario.run_parallel_proc
    scenario.run_parallel_proc = False

    try:
        # Calculate the damage state probabilities with sequential processing
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
    finally:
        # Restore the original parallel setting
        scenario.run_parallel_proc = original_parallel_setting


def process_hazard_chunk(chunk_data):
    """
    Process a chunk of hazards

    Parameters:
    -----------
    chunk_data : tuple
        (hazard_chunk, scenario, infrastructure, chunk_id)

    Returns:
    --------
    tuple : (chunk_id, chunk_response, processed_count)
        Results, chunk ID, and number of hazards processed
    """
    hazard_chunk, scenario, infrastructure, chunk_id = chunk_data

    # Create a worker copy of the scenario to ensure clean state
    worker_scenario = scenario.create_worker_copy()

    chunk_response = {}

    # Process each hazard in the chunk
    for hazard in hazard_chunk:
        # Call the single hazard processing function
        result = calculate_response_for_single_hazard(hazard, worker_scenario, infrastructure)
        chunk_response.update(result)

    # Return chunk ID, results, and count of processed hazards
    return chunk_id, chunk_response, len(hazard_chunk)


def calculate_response(hazards, scenario, infrastructure):
    """
    The response will be calculated by creating the hazard_levels,
    iterating through the range of hazards and calling the infrastructure
    systems expose_to method.
    It returns the results of the infrastructure to each hazard event.

    A parameter in the scenario file determines whether the parallelising
    function spawns processes that will perform parallel calculations.

    Parameters:
    -----------
    scenario: Object for simulation parameters.
    infrastructure: Object of infrastructure model.
    hazards: `HazardsContainer` object.

    Returns:
    --------
    List of results for each hazard level.
    """
    hazards_response = []
    rootLogger.info(
        "Initiating calculation of component damage states for hazard event set."
    )

    # Convert hazards to list
    hazard_list = list(hazards.listOfhazards)
    total_hazards = len(hazard_list)

    if scenario.run_parallel_proc:
        # --- Parallel execution ---
        rootLogger.info("Starting parallel hazard processing")

        # Get number of cores to use
        num_cores = get_available_cores()
        rootLogger.info(f"Available cores: {num_cores}")

        # Calculate optimal chunk size - aim for ~4 chunks per core
        # This gives good load balancing while minimizing overhead
        target_chunks_per_core = 4
        chunk_size = max(1, math.ceil(total_hazards / (num_cores * target_chunks_per_core)))

        # Create chunks with unique IDs
        hazard_chunks = []
        for i in range(0, total_hazards, chunk_size):
            end_idx = min(i + chunk_size, total_hazards)
            # Store the chunk index for reference
            hazard_chunks.append((i // chunk_size, hazard_list[i:end_idx]))

        rootLogger.info(
            f"Processing {total_hazards} hazards in {len(hazard_chunks)} chunks "
            f"with ~{chunk_size} hazards per chunk using {num_cores} cores")

        # Initialize progress variables
        processed_hazards = 0

        # Print initial progress
        print(f"\nSimulating impact of hazards: 0% complete (0/{total_hazards})", end='', flush=True)

        # Dictionary to collect results by chunk ID to maintain order
        collected_results = {}

        # Create a lock for updating progress
        progress_lock = threading.Lock()

        # Process using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit all chunks for processing
            futures = []
            for chunk_id, chunk in hazard_chunks:
                future = executor.submit(
                    process_hazard_chunk,
                    (chunk, scenario, infrastructure, chunk_id)
                )
                futures.append(future)

            # Process results as they complete
            for future in as_completed(futures):
                try:
                    chunk_id, chunk_result, hazards_processed = future.result()
                    collected_results[chunk_id] = chunk_result

                    # Update progress
                    with progress_lock:
                        processed_hazards += hazards_processed
                        percent = int((processed_hazards / total_hazards) * 100)
                        print(
                            f"\rSimulating impact of hazards: {percent}% complete "
                            f"({processed_hazards}/{total_hazards})",
                            end='', flush=True)

                except Exception as e:
                    rootLogger.error(f"Chunk processing failed: {str(e)}")
                    rootLogger.error(f"Exception details: {traceback.format_exc()}")
                    raise

            # Convert results from dictionary to list in original chunk order
            hazards_response = [
                collected_results[i] for i in range(len(hazard_chunks))
                if i in collected_results]

        # Final newline after all processing is complete
        print("\nHazard processing complete.\n")
        rootLogger.info("Completed parallel hazard processing")

    else:
        # --- Sequential execution ---
        rootLogger.info("Starting sequential hazard processing")

        # Set up progress tracking
        print(f"\nSimulating impact of hazards: 0% complete (0/{total_hazards})", end='', flush=True)

        # Process hazards sequentially with progress updates
        for i, hazard in enumerate(hazard_list):
            # Process single hazard
            result = calculate_response_for_single_hazard(hazard, scenario, infrastructure)
            hazards_response.append(result)

            # Update progress
            percent = int(((i + 1) / total_hazards) * 100)
            print(
                f"\rSimulating impact of hazards: {percent}% complete ({i + 1}/{total_hazards})",
                end='', flush=True)

        # Final newline after all processing is complete
        print("\nHazard processing complete.\n")
        rootLogger.info("Completed sequential hazard processing")

    # Combine the responses into one list
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

    # Iterate through the hazards
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

    # Convert the calculated output array into the correct format
    system_output_arr_per_sample = post_processing_list[4]
    post_processing_list[4] = np.sum(system_output_arr_per_sample, axis=2).transpose()
    system_econ_loss_arr_per_sample = np.array(post_processing_list[5])
    post_processing_list[5] = system_econ_loss_arr_per_sample.transpose()

    return post_processing_list


def calc_component_damage_state_for_n_simulations(infrastructure, scenario, hazard):
    """
    Calculate the probability that being exposed to a hazard level
    will exceed the given damage levels for each component. A monte
    carlo approach is taken by simulating the exposure for the number
    of samples given in the scenario.

    Parameters:
    -----------
    infrastructure: Container for components
    scenario: Parameters for the scenario
    hazard: Level of the hazard

    Returns:
    --------
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

    # Component keys sorted for consistent processing order
    component_keys = sorted(infrastructure.components.keys())

    # Check if this is being called directly (not from a hazard chunk)
    # This affects whether we show progress or not
    show_progress = False
    try:
        stack = traceback.extract_stack()
        # If calculate_response_for_single_hazard is in the stack but process_hazard_chunk isn't,
        # or if neither is in the stack, show progress
        in_single_hazard = any(
            getattr(frame, 'name', None) == 'calculate_response_for_single_hazard' or\
            getattr(frame, 'function', str(frame)) == 'calculate_response_for_single_hazard'
            for frame in stack
        )
        in_hazard_chunk = any(
            getattr(frame, 'name', None) == 'process_hazard_chunk' or\
            getattr(frame, 'function', str(frame)) == 'process_hazard_chunk'
            for frame in stack
        )

        # Show progress if we're processing a single hazard directly
        # but not if we're in a parallel hazard chunk
        show_progress = in_single_hazard and not in_hazard_chunk
        if not in_single_hazard and not in_hazard_chunk:
            # Direct call to this function, show progress
            show_progress = True
    except Exception:
        # If stack inspection fails, assume we're not in a worker process
        show_progress = True

    # Set up simple progress reporting
    if show_progress:
        # Get hazard ID for context if available
        hazard_id = getattr(hazard, 'hazard_event_id', "").split('_')[-1]
        prefix = f"[Hazard {hazard_id}] " if hazard_id else ""
        print(f"\n{prefix}Processing components sequentially...")
        print(f"{prefix}Processing components: 0% complete (0/{number_of_components})", end='', flush=True)

    # Process components sequentially
    for index, component_key in enumerate(component_keys):
        component = infrastructure.components[component_key]
        component_pe_ds = np.zeros(len(component.damage_states))

        for damage_state_index in component.damage_states.keys():
            loc_params = component.get_location()
            hazard_intensity = hazard.get_hazard_intensity(*loc_params)
            component_pe_ds[damage_state_index] = \
                component.damage_states[damage_state_index].response_function(
                    hazard_intensity)

        component_pe_ds = component_pe_ds[1:]
        component_pe_ds[np.isnan(component_pe_ds)] = -np.inf

        component_damage_state_ind[:, index] = \
            np.sum(component_pe_ds > rnd[:, index][:, np.newaxis], axis=1)

        # Update progress at regular intervals if showing progress
        if show_progress and (index + 1) % max(1, number_of_components // 20) == 0:
            percent = int(((index + 1) / number_of_components) * 100)
            hazard_id = getattr(hazard, 'hazard_event_id', "").split('_')[-1]
            prefix = f"[Hazard {hazard_id}] " if hazard_id else ""
            print(
                f"\r{prefix}Processing components: {percent}% complete "
                f"({index + 1}/{number_of_components})",
                end='', flush=True)

    # Final update and newline
    if show_progress:
        hazard_id = getattr(hazard, 'hazard_event_id', "").split('_')[-1]
        prefix = f"[Hazard {hazard_id}] " if hazard_id else ""
        print(f"\r{prefix}Processing components: 100% complete ({number_of_components}/{number_of_components})")
        print(f"{prefix}Component processing complete.")

    return component_damage_state_ind
