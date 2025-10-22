import logging
import math
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

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
    # Disable nested parallelisation, store original setting to restore later
    original_parallel_setting = scenario.run_parallel_proc
    scenario.run_parallel_proc = False

    try:
        # Calculate the damage state probabilities with sequential processing
        expected_damage_state_of_components_for_n_simulations = (
            calc_component_damage_state_for_n_simulations(infrastructure, scenario, hazard)
        )

        # Calculate the component loss, functionality, output, economic loss,
        #   and recovery output over time
        (
            component_sample_loss,
            comp_sample_func,
            infrastructure_sample_output,
            infrastructure_sample_economic_loss,
        ) = infrastructure.calc_output_loss(
            scenario, expected_damage_state_of_components_for_n_simulations
        )

        # Construct the dictionary containing the statistics of the response
        (
            component_response_dict,
            comptype_response_dict,
            compclass_dmg_level_percentages,
            compclass_dmg_index_expected,
        ) = infrastructure.calc_response(
            component_sample_loss,
            comp_sample_func,
            expected_damage_state_of_components_for_n_simulations,
        )

        # Determine average output for the output components
        infrastructure_output = {}
        output_nodes = infrastructure.output_nodes.keys()
        for output_index, output_comp_id in enumerate(output_nodes):
            infrastructure_output[output_comp_id] = np.mean(
                infrastructure_sample_output[:, output_index]
            )

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
                compclass_dmg_index_expected,
            ]
        }
        return response_for_a_hazard
    finally:
        # Restore the original parallel setting
        scenario.run_parallel_proc = original_parallel_setting


def calculate_response_for_hazard_batch(hazard_chunk, scenario, infrastructure):
    """
    Process a batch of hazards efficiently by minimizing per-hazard overhead.

    This function processes hazards individually but with optimisations to reduce
    the overhead of repeated function calls and data structure creation.

    Parameters:
    -----------
    hazard_chunk : list
        List of hazard objects to process
    scenario : Scenario object
        The parameters for the scenario being simulated
    infrastructure : Infrastructure object
        Container for components

    Returns:
    --------
    dict : Combined response dictionary for all hazards in the chunk
    """
    if not hazard_chunk:
        return {}

    # Pre-allocate the response dictionary with expected size
    batch_response = {}

    # Set up optimised environment for batch processing
    original_parallel_setting = scenario.run_parallel_proc
    scenario.run_parallel_proc = False  # Avoid nested parallelisation

    try:
        # Process each hazard with minimal overhead
        for hazard in hazard_chunk:
            # Call optimised single hazard processing
            hazard_response = calculate_response_for_single_hazard_optimised(
                hazard, scenario, infrastructure
            )
            batch_response.update(hazard_response)
    finally:
        scenario.run_parallel_proc = original_parallel_setting

    return batch_response


def calculate_response_for_single_hazard_optimised(hazard, scenario, infrastructure):
    """
    Optimised version of single hazard processing with reduced overhead.
    """
    # Pre-calculate damage states with optimised path
    expected_damage_state_of_components_for_n_simulations = (
        calc_component_damage_state_for_n_simulations(infrastructure, scenario, hazard)
    )

    # Calculate outputs more efficiently
    (
        component_sample_loss,
        comp_sample_func,
        infrastructure_sample_output,
        infrastructure_sample_economic_loss,
    ) = infrastructure.calc_output_loss(
        scenario, expected_damage_state_of_components_for_n_simulations
    )

    # Calculate response efficiently
    (
        component_response_dict,
        comptype_response_dict,
        compclass_dmg_level_percentages,
        compclass_dmg_index_expected,
    ) = infrastructure.calc_response(
        component_sample_loss,
        comp_sample_func,
        expected_damage_state_of_components_for_n_simulations,
    )

    # Calculate infrastructure output efficiently
    infrastructure_output = {}
    output_nodes = infrastructure.output_nodes.keys()
    for output_index, output_comp_id in enumerate(output_nodes):
        infrastructure_output[output_comp_id] = np.mean(
            infrastructure_sample_output[:, output_index]
        )

    # Return optimised response structure
    return {
        hazard.hazard_event_id: [
            expected_damage_state_of_components_for_n_simulations,
            infrastructure_output,
            component_response_dict,
            comptype_response_dict,
            infrastructure_sample_output,
            infrastructure_sample_economic_loss,
            compclass_dmg_level_percentages,
            compclass_dmg_index_expected,
        ]
    }


def process_hazard_chunk(chunk_data):
    """
    Process a chunk of hazards using vectorised batch processing

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

    # Disable nested parallelisation in workers
    worker_scenario.run_parallel_proc = False

    chunk_response = {}

    # Use vectorised batch processing for efficiency
    try:
        result = calculate_response_for_hazard_batch(hazard_chunk, worker_scenario, infrastructure)
        chunk_response.update(result)
    except Exception as e:
        # Fallback to individual processing if batch processing fails
        print(
            f"Batch processing failed for chunk {chunk_id}, "
            f"falling back to individual processing: {e}"
        )
        for hazard in hazard_chunk:
            result = calculate_response_for_single_hazard(hazard, worker_scenario, infrastructure)
            chunk_response.update(result)

    # Return chunk ID, results, and count of processed hazards
    return chunk_id, chunk_response, len(hazard_chunk)


def calculate_response(hazards, scenario, infrastructure, dask_client=None, mpi_comm=None):
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
    dask_client: Dask distributed client (optional)
    mpi_comm: MPI communicator (optional)

    Returns:
    --------
    List of results for each hazard level.
    """

    hazards_response: list[dict] = []
    streaming = bool(getattr(scenario, "stream_results", False))
    stream_dir: Path | None = None
    if streaming:
        # Allow environment override (e.g., to point to job-local storage on HPC)
        env_stream_dir = os.environ.get("SIRA_STREAM_DIR")
        if env_stream_dir:
            stream_dir = Path(env_stream_dir)
        else:
            stream_dir = Path(getattr(scenario, "stream_dir", Path("stream")))
        stream_dir.mkdir(parents=True, exist_ok=True)

    rootLogger.info("Initiating calculation of component damage states for hazard event set.")

    hazard_list = list(hazards.listOfhazards)
    total_hazards = len(hazard_list)

    # ------------------------------------------------------------------
    # Parallel path
    # ------------------------------------------------------------------
    if scenario.run_parallel_proc and total_hazards > 0:
        rootLogger.info("Starting parallel hazard processing")

        # Check if MPI is available and prioritize it for HPC environments
        if mpi_comm is not None:
            rootLogger.info("Using MPI backend for parallel processing")
            return _calculate_response_mpi(
                hazards,
                scenario,
                infrastructure,
                mpi_comm,
                hazard_list,
                total_hazards,
                streaming,
                stream_dir,
            )

        # Multiprocessing fallback (Dask removed due to memory issues on HPC)
        rootLogger.info("Using multiprocessing backend for parallel processing")

        # Determine optimal processing strategy based on problem size
        target_chunks_per_slot = int(os.environ.get("SIRA_CHUNKS_PER_SLOT", "2"))
        num_slots = get_available_cores()
        rootLogger.info(f"Available local cores: {num_slots}")

        # For very small simulations, consider serial processing to avoid overhead
        min_hazards_for_parallel = int(os.environ.get("SIRA_MIN_HAZARDS_FOR_PARALLEL", "50"))
        if total_hazards < min_hazards_for_parallel:
            rootLogger.info(
                f"Small simulation ({total_hazards} hazards) - "
                f"considering serial processing to avoid multiprocessing overhead"
            )
            # Can be enabled with SIRA_FORCE_SERIAL=1 environment variable
            if os.environ.get("SIRA_FORCE_SERIAL", "0") == "1":
                rootLogger.info("Serial processing enabled via SIRA_FORCE_SERIAL")
                return _calculate_response_serial(hazard_list, scenario, infrastructure)

        # Optimised chunk sizing to reduce multiprocessing overhead
        chunk_size = max(1, math.ceil(total_hazards / (num_slots * target_chunks_per_slot)))

        # For small numbers of hazards, use fewer larger chunks to reduce overhead
        if total_hazards < num_slots * 2:
            chunk_size = max(1, math.ceil(total_hazards / num_slots))
            target_chunks_per_slot = 1
            rootLogger.info(
                f"Small hazard count detected, using {num_slots} chunks of size {chunk_size}"
            )

        hazard_chunks: list[tuple[int, list]] = []
        for i in range(0, total_hazards, chunk_size):
            end_idx = min(i + chunk_size, total_hazards)
            hazard_chunks.append((i // chunk_size, hazard_list[i:end_idx]))

        rootLogger.info(
            f"Processing {total_hazards} hazards in {len(hazard_chunks)} chunks "
            f"with ~{chunk_size} hazards per chunk across ~{num_slots} slots"
        )

        processed_hazards = 0
        print(
            f"\nSimulating impact of hazards: 0% complete (0/{total_hazards})", end="", flush=True
        )

        collected_results: dict[int, dict] | None = {} if not streaming else None
        progress_lock = threading.Lock()

        # Use multiprocessing for parallel execution
        with ProcessPoolExecutor(max_workers=num_slots) as executor:
            futures = [
                executor.submit(process_hazard_chunk, (chunk, scenario, infrastructure, chunk_id))
                for chunk_id, chunk in hazard_chunks
            ]
            for future in as_completed(futures):
                chunk_id, chunk_result, hazards_processed = future.result()
                if streaming and stream_dir is not None:
                    _persist_chunk_result(chunk_id, chunk_result, stream_dir)
                else:
                    assert collected_results is not None
                    collected_results[chunk_id] = chunk_result

                with progress_lock:
                    processed_hazards += hazards_processed
                    percent = int((processed_hazards / total_hazards) * 100)
                    print(
                        f"\rSimulating impact of hazards: {percent}% complete "
                        f"({processed_hazards}/{total_hazards})",
                        end="",
                        flush=True,
                    )

        print("\nHazard processing complete.\n")
        rootLogger.info("Completed parallel hazard processing")

        # Consolidate manifests if streaming (same as MPI path)
        if streaming and stream_dir is not None:
            main_manifest_path = stream_dir / "manifest.jsonl"
            try:
                with open(main_manifest_path, "w", encoding="utf-8") as main_mf:
                    for chunk_id in range(len(hazard_chunks)):
                        rank_manifest_path = stream_dir / f"manifest_rank_{chunk_id:06d}.jsonl"
                        if rank_manifest_path.exists():
                            with open(rank_manifest_path, "r", encoding="utf-8") as rank_mf:
                                for line in rank_mf:
                                    main_mf.write(line)
                        # Clean up chunk-specific manifest
                        rank_manifest_path.unlink()

                rootLogger.info(
                    f"Consolidated {len(hazard_chunks)} chunk manifests into {main_manifest_path}"
                )
            except Exception as e:
                rootLogger.error(f"Failed to consolidate manifests: {e}")

        if not streaming and collected_results is not None:
            hazards_response = [
                collected_results[i] for i in range(len(hazard_chunks)) if i in collected_results
            ]

    # ------------------------------------------------------------------
    # Sequential path
    # ------------------------------------------------------------------
    else:
        rootLogger.info("Starting sequential hazard processing")
        print(
            f"\nSimulating impact of hazards: 0% complete (0/{total_hazards})", end="", flush=True
        )
        for i, hazard in enumerate(hazard_list):
            result = calculate_response_for_single_hazard(hazard, scenario, infrastructure)
            if streaming and stream_dir is not None:
                _persist_chunk_result(i, result, stream_dir)
            else:
                hazards_response.append(result)

            percent = int(((i + 1) / total_hazards) * 100) if total_hazards else 100
            print(
                (f"\rSimulating impact of hazards: {percent}% complete ({i + 1}/{total_hazards})"),
                end="",
                flush=True,
            )
        print("\nHazard processing complete.\n")
        rootLogger.info("Completed sequential hazard processing")

    # ------------------------------------------------------------------
    # Return section
    # ------------------------------------------------------------------
    if streaming and stream_dir is not None:
        num_chunks = len(hazards_response) if not scenario.run_parallel_proc else None
        # For sequential streaming we treat each hazard as its own chunk; for parallel we can
        # infer chunk directories later. We therefore include total hazards always.
        return {
            "streaming": True,
            "stream_dir": str(stream_dir),
            "total_hazards": total_hazards,
            "num_chunks": num_chunks,
        }

    # Legacy in-memory aggregation
    response_data = [
        {},  # [0]
        {},  # [1]
        {},  # [2]
        {},  # [3]
        [],  # [4]
        [],  # [5]
        {},  # [6]
        {},  # [7]
    ]

    for hazard_response in hazards_response:
        for key, value_list in hazard_response.items():
            for list_number, _ in enumerate(response_data):
                if (list_number <= 3) or (list_number in [6, 7]):
                    response_data[list_number][key] = value_list[list_number]
                else:
                    response_data[list_number].append(value_list[list_number])

    for list_number in range(4, 6):
        response_data[list_number] = np.array(response_data[list_number])

    system_output_arr_per_sample = response_data[4]
    response_data[4] = np.sum(system_output_arr_per_sample, axis=2).transpose()
    system_econ_loss_arr_per_sample = np.array(response_data[5])
    response_data[5] = system_econ_loss_arr_per_sample.transpose()

    return response_data


def _calculate_response_serial(hazard_list, scenario, infrastructure):
    """
    Serial processing fallback for small simulations to avoid multiprocessing overhead.

    Parameters:
    -----------
    hazard_list: List of hazard objects
    scenario: Scenario object
    infrastructure: Infrastructure object

    Returns:
    --------
    List of results for each hazard level
    """

    rootLogger.info("Using optimised serial processing")
    total_hazards = len(hazard_list)
    hazards_response = []

    print(f"\nSimulating impact of hazards: 0% complete (0/{total_hazards})", end="", flush=True)

    for i, hazard in enumerate(hazard_list):
        result = calculate_response_for_single_hazard_optimised(hazard, scenario, infrastructure)
        hazards_response.append(result)

        percent = int(((i + 1) / total_hazards) * 100) if total_hazards else 100
        print(
            f"\rSimulating impact of hazards: {percent}% complete ({i + 1}/{total_hazards})",
            end="",
            flush=True,
        )

    print("\nSerial hazard processing complete.\n")
    rootLogger.info("Completed optimised serial hazard processing")

    # Return in the same format as parallel processing
    return hazards_response


def _distribute_hazards_mpi(total_hazards, mpi_size):
    """
    Distribute hazards across MPI ranks as evenly as possible.

    Parameters:
    -----------
    total_hazards: Total number of hazards to distribute
    mpi_size: Number of MPI processes

    Returns:
    --------
    List of (start_idx, end_idx) tuples for each rank
    """
    hazards_per_rank = total_hazards // mpi_size
    remainder = total_hazards % mpi_size

    distribution = []
    current_start = 0

    for rank in range(mpi_size):
        if rank < remainder:
            rank_count = hazards_per_rank + 1
        else:
            rank_count = hazards_per_rank

        distribution.append((current_start, current_start + rank_count))
        current_start += rank_count

    return distribution


def _calculate_response_mpi(
    hazards, scenario, infrastructure, mpi_comm, hazard_list, total_hazards, streaming, stream_dir
):
    """
    MPI-based hazard processing for HPC environments.

    Parameters:
    -----------
    hazards: HazardsContainer object
    scenario: Scenario object
    infrastructure: Infrastructure object
    mpi_comm: MPI communicator
    hazard_list: List of hazard objects
    total_hazards: Total number of hazards
    streaming: Whether streaming is enabled
    stream_dir: Stream directory path

    Returns:
    --------
    Aggregated results (only on rank 0), None on worker ranks
    """
    rank = mpi_comm.Get_rank()
    size = mpi_comm.Get_size()

    rootLogger.info(f"MPI rank {rank} of {size}: Starting MPI hazard processing")

    # Distribute hazards across ranks
    distribution = _distribute_hazards_mpi(total_hazards, size)
    start_idx, end_idx = distribution[rank]
    my_hazards = hazard_list[start_idx:end_idx]

    rootLogger.info(
        f"MPI rank {rank}: Processing {len(my_hazards)} hazards (indices {start_idx}-{end_idx - 1})"
    )

    # Process hazards assigned to this rank
    my_results = {}
    failed_hazards = []
    successful_hazards = 0

    for i, hazard in enumerate(my_hazards):
        try:
            result = calculate_response_for_single_hazard(hazard, scenario, infrastructure)
            my_results.update(result)
            successful_hazards += 1

            # Progress reporting (only rank 0 reports, less frequently)
            if rank == 0 and len(my_hazards) > 9 and (i + 1) % max(1, len(my_hazards) // 10) == 0:
                global_progress = int(((i + 1) / len(my_hazards)) * 100)
                print(f"\rMPI rank 0 progress: {global_progress}% complete", end="", flush=True)

        except Exception as e:
            failed_hazards.append((hazard.hazard_event_id, str(e)))
            rootLogger.error(
                f"MPI rank {rank}: Error processing hazard {hazard.hazard_event_id}: {e}"
            )
            # Continue processing other hazards instead of aborting immediately
            continue

    # Report failures if any occurred
    if failed_hazards:
        rootLogger.warning(
            f"MPI rank {rank}: {len(failed_hazards)} hazards failed out of {len(my_hazards)}"
        )
        for hazard_id, error in failed_hazards:
            rootLogger.warning(f"  - {hazard_id}: {error}")

    # Gather failure statistics across all ranks
    failure_counts = mpi_comm.gather(len(failed_hazards), root=0)
    total_successful = mpi_comm.gather(successful_hazards, root=0)

    if rank == 0 and failure_counts:
        total_failures = sum(failure_counts)
        total_successes = sum(total_successful) if total_successful else 0
        if total_failures > 0:
            rootLogger.warning(
                f"MPI processing completed with {total_failures} failed hazards "
                f"out of {total_hazards} total ({total_successes} successful)"
            )

    # Handle streaming if enabled
    if streaming and stream_dir is not None:
        # Each rank writes its own chunk
        _persist_chunk_result(rank, my_results, stream_dir)

        # Synchronise all ranks
        mpi_comm.barrier()

        # Only rank 0 consolidates manifests and returns the streaming info
        if rank == 0:
            import time

            # Allow time for distributed filesystem synchronization across nodes
            rootLogger.info("Waiting for file system synchronization...")
            time.sleep(5)

            # Check stream directory contents for diagnostics
            print(f"\n{'=' * 80}", flush=True)
            print(f"Stream directory check: {stream_dir}\n", flush=True)

            try:
                all_files = list(stream_dir.glob("*"))
                manifest_files = sorted(stream_dir.glob("manifest_rank_*.jsonl"))
                chunk_dirs = sorted([d for d in stream_dir.iterdir() if d.is_dir()])

                print(f"Total files/dirs in stream_dir: {len(all_files)}", flush=True)
                print(f"Manifest files found: {len(manifest_files)}", flush=True)
                print(f"Chunk directories found: {len(chunk_dirs)}", flush=True)

                if manifest_files:
                    print("\nFirst 10 manifest files:", flush=True)
                    for mf in manifest_files[:10]:
                        file_size = mf.stat().st_size
                        print(f"  {mf.name}: {file_size:,} bytes", flush=True)

                    if len(manifest_files) > 10:
                        print(f"  ... and {len(manifest_files) - 10} more", flush=True)

                    print("\nLast 10 manifest files:", flush=True)
                    for mf in manifest_files[-10:]:
                        file_size = mf.stat().st_size
                        print(f"  {mf.name}: {file_size:,} bytes", flush=True)
                else:
                    print("Warning: No manifest files found", flush=True)

                print(f"{'=' * 80}\n", flush=True)

                rootLogger.info(
                    f"Directory check: Found {len(manifest_files)} manifest files "
                    f"out of {size} expected"
                )
            except Exception as e:
                print(f"Error during directory check: {e}", flush=True)
                rootLogger.error(f"Failed to list stream directory contents: {e}")

            # Consolidate rank-specific manifests into main manifest
            main_manifest_path = stream_dir / "manifest.jsonl"
            try:
                manifests_found = 0
                total_events_consolidated = 0
                missing_ranks = []

                with open(main_manifest_path, "w", encoding="utf-8") as main_mf:
                    for rank_id in range(size):
                        rank_manifest_path = stream_dir / f"manifest_rank_{rank_id:06d}.jsonl"

                        # Retry with exponential backoff for distributed filesystems
                        manifest_found = False
                        for attempt in range(3):
                            if rank_manifest_path.exists():
                                manifest_found = True
                                break
                            if attempt < 2:
                                time.sleep(2**attempt)

                        if manifest_found:
                            manifests_found += 1
                            events_in_rank = 0
                            with open(rank_manifest_path, "r", encoding="utf-8") as rank_mf:
                                for line in rank_mf:
                                    main_mf.write(line)
                                    events_in_rank += 1
                            total_events_consolidated += events_in_rank
                            rootLogger.debug(
                                f"Consolidated {events_in_rank} events from rank {rank_id}"
                            )
                            # Clean up rank-specific manifest
                            rank_manifest_path.unlink()
                        else:
                            missing_ranks.append(rank_id)
                            rootLogger.warning(f"Rank manifest not found: {rank_manifest_path}")

                rootLogger.info(
                    f"Consolidated {total_events_consolidated} events from "
                    f"{manifests_found}/{size} rank manifests into {main_manifest_path}"
                )

                if missing_ranks:
                    rootLogger.error(
                        f"Missing {len(missing_ranks)} rank manifests: {missing_ranks[:10]}..."
                        if len(missing_ranks) > 10
                        else f"Missing {len(missing_ranks)} rank manifests: {missing_ranks}"
                    )

                if total_events_consolidated != total_hazards:
                    rootLogger.error(
                        f"Event count mismatch! Expected {total_hazards} events, "
                        f"but consolidated {total_events_consolidated} events. "
                        f"Missing {total_hazards - total_events_consolidated} events."
                    )
            except Exception as e:
                rootLogger.error(f"Failed to consolidate manifests: {e}")
                import traceback

                rootLogger.error(traceback.format_exc())

            return {
                "streaming": True,
                "stream_dir": str(stream_dir),
                "total_hazards": total_hazards,
                "num_chunks": size,  # One chunk per MPI rank
            }
        else:
            return None

    # Gather all results to rank 0 for non-streaming case
    all_results = mpi_comm.gather(my_results, root=0)

    if rank == 0:
        # Aggregate results from all ranks
        hazards_response = []
        for rank_results in all_results:
            if rank_results:
                hazards_response.append(rank_results)

        print(f"\nMPI processing complete. Processed {total_hazards} hazards across {size} ranks.")
        rootLogger.info(f"MPI rank 0: Completed aggregation of results from {size} ranks")

        return hazards_response
    else:
        # Worker ranks return None
        return None


def _persist_chunk_result(chunk_id: int, chunk_result: dict, stream_dir: Path) -> None:
    """Persist chunk results to NPY and JSON files.

    Writes streaming data for each event in the chunk:
      - Economic loss samples (NPY)
      - System output samples (NPY)
      - Component damage states (NPY)
      - Component response data (JSON)
      - Component type response data (JSON)

    Creates a manifest file tracking all persisted artifacts.
    """
    import json
    import logging

    import numpy as np

    rootLogger = logging.getLogger(__name__)

    # Paths
    chunk_base = stream_dir / f"chunk_{chunk_id:06d}"
    chunk_base.mkdir(parents=True, exist_ok=True)

    # Persist each event in this chunk separately and record manifest lines
    def _sanitize(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(name))

    entries = []

    def _serialise_response_dict(resp_dict: dict) -> dict:
        """Convert tuple-keyed response dicts into JSON-serialisable nested dicts."""

        serialisable: dict[str, dict[str, float]] = {}
        for key, value in resp_dict.items():
            if isinstance(key, (list, tuple)) and len(key) == 2:
                comp_key, response_key = key
                comp_str = str(comp_key)
                response_str = str(response_key)
                metrics = serialisable.setdefault(comp_str, {})
                try:
                    metrics[response_str] = float(value)
                except (TypeError, ValueError):
                    metrics[response_str] = 0.0
            else:
                comp_str = str(key)
                metrics = serialisable.setdefault(comp_str, {})
                try:
                    metrics["value"] = float(value)
                except (TypeError, ValueError):
                    metrics["value"] = 0.0
        return serialisable

    for event_id, vlist in chunk_result.items():
        try:
            damage_states = vlist[0]
            sys_output_samples = vlist[4]
            sys_econ_loss_samples = vlist[5]
            comp_resp = vlist[2]
            comptype_resp = vlist[3]

            safe_id = _sanitize(event_id)

            # Economic loss: write directly to NPY (simpler and faster than parquet)
            econ_path = chunk_base / f"{safe_id}__econ.npy"
            try:
                np.save(econ_path, np.asarray(sys_econ_loss_samples))
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(
                    f"Failed to write economic loss for event {event_id} to {econ_path}: {e}"
                )

            # System output: write to NPY format
            out_paths = []
            try:
                out_p = chunk_base / f"{safe_id}__sysout.npy"
                np.save(out_p, np.asarray(sys_output_samples))
                out_paths = [str(out_p)]
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Failed to write system output for event {event_id}: {e}")
                out_paths = []

            # Persist component damage state indicators for recovery/statistics reconstruction
            damage_path = chunk_base / f"{safe_id}__damage.npy"
            try:
                np.save(damage_path, np.asarray(damage_states, dtype=np.int16))
            except Exception:
                # Remove partially written file if save fails
                if damage_path.exists():
                    try:
                        damage_path.unlink()
                    except Exception:
                        pass

            # Store component and component type response data as JSON
            comp_resp_json = _serialise_response_dict(comp_resp)
            comptype_resp_json = _serialise_response_dict(comptype_resp)
            try:
                if comp_resp_json:
                    with open(chunk_base / f"{safe_id}__comp_response.json", "w") as f:
                        json.dump(comp_resp_json, f)
                if comptype_resp_json:
                    with open(chunk_base / f"{safe_id}__comptype_response.json", "w") as f:
                        json.dump(comptype_resp_json, f)
            except Exception:
                pass

            entries.append(
                {
                    "event_id": str(event_id),
                    "econ": str(econ_path),
                    "econ_size": econ_path.stat().st_size if econ_path.exists() else 0,
                    "sys_output": out_paths,
                    "sys_output_sizes": [
                        Path(p).stat().st_size if Path(p).exists() else 0 for p in out_paths
                    ],
                    "damage": str(damage_path) if damage_path.exists() else None,
                    "damage_size": damage_path.stat().st_size if damage_path.exists() else 0,
                }
            )
        except Exception:
            continue

    # Append to rank-specific manifest to avoid MPI race conditions
    try:
        rank_manifest_path = stream_dir / f"manifest_rank_{chunk_id:06d}.jsonl"
        with open(rank_manifest_path, "w", encoding="utf-8") as mf:
            for rec in entries:
                mf.write(json.dumps(rec) + "\n")
            # Explicitly flush to OS buffer
            mf.flush()
            # Force OS to write to disk (critical for distributed filesystems)
            import os

            os.fsync(mf.fileno())

        # Verify manifest was written successfully
        if rank_manifest_path.exists():
            manifest_size = rank_manifest_path.stat().st_size
            rootLogger.info(
                f"Rank {chunk_id}: Wrote manifest with {len(entries)} events "
                f"({manifest_size} bytes) to {rank_manifest_path.name}"
            )
        else:
            rootLogger.error(f"Rank {chunk_id}: Manifest file not found: {rank_manifest_path}")
    except Exception as e:
        rootLogger.error(f"Failed to write manifest for chunk {chunk_id}: {e}")
        import traceback

        rootLogger.error(traceback.format_exc())


def get_vectorised_damage_probabilities(component, hazard_intensity):
    """
    Get damage state probabilities for a component using vectorised calls.

    This function leverages the SciPy-based response functions in
    responsemodels which are already vectorised, but optimises the function call
    dispatch by collecting all damage state evaluations into a single operation.

    Parameters:
    -----------
    component: Component object with damage_states
    hazard_intensity: float, hazard intensity value

    Returns:
    --------
    np.ndarray: Exceedance probabilities for all damage states (excluding DS0)
    """
    damage_state_indices = sorted([k for k in component.damage_states.keys() if k != 0])

    if not damage_state_indices:
        return np.array([])

    # Extract response functions for all damage states
    response_funcs = []
    for ds_idx in damage_state_indices:
        try:
            response_funcs.append(component.damage_states[ds_idx].response_function)
        except (KeyError, AttributeError):
            # Handle missing or invalid damage states
            response_funcs.append(lambda x: 0.0)

    # Apply response functions to hazard intensity (vectorised via SciPy)
    try:
        probabilities = np.array([func(hazard_intensity) for func in response_funcs])

        # Handle any NaN or invalid results
        probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)

        return probabilities

    except Exception:
        # Fallback: return zeros for this component
        return np.zeros(len(response_funcs))


def batch_process_similar_components(
    component_group, hazard_intensities, rnd, component_damage_state_ind, show_progress=False
):
    """
    Advanced batch processing for components with similar response function types.

    This function identifies components that use the same response function types
    and parameters, then processes them together using vectorised SciPy calls
    for maximum efficiency.

    Parameters:
    -----------
    component_group: list of (index, component_key, component) tuples
    hazard_intensities: np.ndarray of hazard intensity values
    rnd: np.ndarray of random numbers for Monte Carlo simulation
    component_damage_state_ind: np.ndarray to store results
    show_progress: bool, whether to show progress updates

    Returns:
    --------
    int: Number of components processed
    """
    from sira.modelling.responsemodels import LogNormalCDF, NormalCDF

    processed_count = 0

    # HPC optimisation: adjust batch sizes based on memory constraints
    hpc_mode = int(os.environ.get("SIRA_HPC_MODE", "0"))
    max_batch_size = int(os.environ.get("SIRA_MAX_BATCH_SIZE", "1000"))

    # For HPC environments, use larger batches but respect memory limits
    if hpc_mode and len(component_group) > max_batch_size:
        # Process in chunks to avoid memory issues
        chunk_size = max_batch_size
        for i in range(0, len(component_group), chunk_size):
            chunk = component_group[i : i + chunk_size]
            processed_count += batch_process_similar_components(
                chunk, hazard_intensities, rnd, component_damage_state_ind, False
            )
        return processed_count

    # Group components by response function type for batch processing
    lognormal_batch = []
    normal_batch = []
    other_components = []

    for index, component_key, component in component_group:
        # Check the type of the first damage state function (most components are homogeneous)
        damage_states = [k for k in component.damage_states.keys() if k != 0]
        if not damage_states:
            other_components.append((index, component_key, component))
            continue

        first_ds = component.damage_states[damage_states[0]]
        if isinstance(first_ds.response_function, LogNormalCDF):
            lognormal_batch.append((index, component_key, component))
        elif isinstance(first_ds.response_function, NormalCDF):
            normal_batch.append((index, component_key, component))
        else:
            other_components.append((index, component_key, component))

    # Process lognormal components in batch if we have enough
    if len(lognormal_batch) >= 3:  # Batch processing threshold
        processed_count += process_lognormal_batch(
            lognormal_batch, hazard_intensities, rnd, component_damage_state_ind
        )
    else:
        other_components.extend(lognormal_batch)

    # Process normal components in batch if we have enough
    if len(normal_batch) >= 3:  # Batch processing threshold
        processed_count += process_normal_batch(
            normal_batch, hazard_intensities, rnd, component_damage_state_ind
        )
    else:
        other_components.extend(normal_batch)

    # Process remaining components individually using vectorised approach
    for index, component_key, component in other_components:
        hazard_intensity = hazard_intensities[index]
        component_pe_ds = get_vectorised_damage_probabilities(component, hazard_intensity)

        if len(component_pe_ds) == 0:
            component_damage_state_ind[:, index] = 0
        else:
            component_pe_ds_stable = np.where(component_pe_ds == 0, -np.inf, component_pe_ds)
            component_damage_state_ind[:, index] = np.sum(
                component_pe_ds_stable >= rnd[:, index][:, np.newaxis], axis=1
            )

        processed_count += 1

    return processed_count


def process_lognormal_batch(component_batch, hazard_intensities, rnd, component_damage_state_ind):
    """Process a batch of components with LogNormalCDF response functions
    using true vectorisation."""
    import scipy.stats as stats

    from sira.modelling.responsemodels import LogNormalCDF

    processed_count = 0

    # HPC optimisation: adjust thresholds based on batch size
    hpc_mode = int(os.environ.get("SIRA_HPC_MODE", "0"))
    min_batch_size = 2 if hpc_mode else 3

    if len(component_batch) < min_batch_size:
        # Fall back to individual processing for small batches
        for index, component_key, component in component_batch:
            hazard_intensity = hazard_intensities[index]
            component_pe_ds = get_vectorised_damage_probabilities(component, hazard_intensity)

            if len(component_pe_ds) == 0:
                component_damage_state_ind[:, index] = 0
            else:
                component_pe_ds_stable = np.where(component_pe_ds == 0, -np.inf, component_pe_ds)
                component_damage_state_ind[:, index] = np.sum(
                    component_pe_ds_stable >= rnd[:, index][:, np.newaxis], axis=1
                )
            processed_count += 1
        return processed_count

    # Group components by identical LogNormalCDF parameters for true batch processing
    param_groups = {}

    for index, component_key, component in component_batch:
        # Get LogNormalCDF parameters for all damage states
        ds_params = []
        damage_states = [k for k in component.damage_states.keys() if k != 0]

        for ds_idx in damage_states:
            ds = component.damage_states[ds_idx]
            if isinstance(ds.response_function, LogNormalCDF):
                # Extract parameters (median, beta, location)
                rf = ds.response_function
                params = (rf.median, rf.beta, rf.location)
                ds_params.append(params)
            else:
                ds_params = None  # Mixed types, can't batch efficiently
                break

        if ds_params:
            param_key = tuple(ds_params)
            if param_key not in param_groups:
                param_groups[param_key] = []
            param_groups[param_key].append((index, component_key, component))

    # Process each parameter group with vectorised SciPy calls
    for param_key, group_components in param_groups.items():
        if len(group_components) >= 3:  # Only batch if we have enough components
            # Extract indices and hazard intensities for this group
            indices = [idx for idx, _, _ in group_components]
            group_hazard_intensities = hazard_intensities[indices]

            # Get damage state parameters
            num_damage_states = len(param_key)

            # Create arrays for batch processing
            # Shape: (num_components, num_damage_states)
            all_probabilities = np.zeros((len(group_components), num_damage_states))

            # Process each damage state across all components in batch
            for ds_idx, (median, beta, location) in enumerate(param_key):
                # Use the same parameters as the original LogNormalCDF implementation
                # scipy.stats.lognorm.cdf(data_point, beta, loc=location, scale=median)

                # Vectorised call to SciPy - single call for all components!
                batch_probs = stats.lognorm.cdf(
                    group_hazard_intensities, s=beta, loc=location, scale=median
                )
                all_probabilities[:, ds_idx] = batch_probs

            # Apply Monte Carlo simulation for each component
            for comp_idx, (index, _, _) in enumerate(group_components):
                component_pe_ds = all_probabilities[comp_idx]

                # Handle numerical stability
                component_pe_ds_stable = np.where(component_pe_ds == 0, -np.inf, component_pe_ds)

                # Monte Carlo simulation
                component_damage_state_ind[:, index] = np.sum(
                    component_pe_ds_stable >= rnd[:, index][:, np.newaxis], axis=1
                )

                processed_count += 1
        else:
            # Process small groups individually
            for index, component_key, component in group_components:
                hazard_intensity = hazard_intensities[index]
                component_pe_ds = get_vectorised_damage_probabilities(component, hazard_intensity)

                if len(component_pe_ds) == 0:
                    component_damage_state_ind[:, index] = 0
                else:
                    component_pe_ds_stable = np.where(
                        component_pe_ds == 0, -np.inf, component_pe_ds
                    )
                    component_damage_state_ind[:, index] = np.sum(
                        component_pe_ds_stable >= rnd[:, index][:, np.newaxis], axis=1
                    )
                processed_count += 1

    return processed_count


def process_normal_batch(component_batch, hazard_intensities, rnd, component_damage_state_ind):
    """Process a batch of components with NormalCDF response functions
    using true vectorisation."""
    import scipy.stats as stats

    from sira.modelling.responsemodels import NormalCDF

    processed_count = 0

    # HPC optimisation: adjust thresholds based on batch size
    hpc_mode = int(os.environ.get("SIRA_HPC_MODE", "0"))
    min_batch_size = 2 if hpc_mode else 3

    if len(component_batch) < min_batch_size:
        # Fall back to individual processing for small batches
        for index, component_key, component in component_batch:
            hazard_intensity = hazard_intensities[index]
            component_pe_ds = get_vectorised_damage_probabilities(component, hazard_intensity)

            if len(component_pe_ds) == 0:
                component_damage_state_ind[:, index] = 0
            else:
                component_pe_ds_stable = np.where(component_pe_ds == 0, -np.inf, component_pe_ds)
                component_damage_state_ind[:, index] = np.sum(
                    component_pe_ds_stable >= rnd[:, index][:, np.newaxis], axis=1
                )
            processed_count += 1
        return processed_count

    # Group components by identical NormalCDF parameters for true batch processing
    param_groups = {}

    for index, component_key, component in component_batch:
        # Get NormalCDF parameters for all damage states
        ds_params = []
        damage_states = [k for k in component.damage_states.keys() if k != 0]

        for ds_idx in damage_states:
            ds = component.damage_states[ds_idx]
            if isinstance(ds.response_function, NormalCDF):
                # Extract parameters (mean, stddev)
                rf = ds.response_function
                params = (rf.mean, rf.stddev)
                ds_params.append(params)
            else:
                ds_params = None  # Mixed types, can't batch efficiently
                break

        if ds_params:
            param_key = tuple(ds_params)
            if param_key not in param_groups:
                param_groups[param_key] = []
            param_groups[param_key].append((index, component_key, component))

    # Process each parameter group with vectorised SciPy calls
    for param_key, group_components in param_groups.items():
        if len(group_components) >= 3:  # Only batch if we have enough components
            # Extract indices and hazard intensities for this group
            indices = [idx for idx, _, _ in group_components]
            group_hazard_intensities = hazard_intensities[indices]

            # Get damage state parameters
            num_damage_states = len(param_key)

            # Create arrays for batch processing
            # Shape: (num_components, num_damage_states)
            all_probabilities = np.zeros((len(group_components), num_damage_states))

            # Process each damage state across all components in batch
            for ds_idx, (mean, stddev) in enumerate(param_key):
                # Vectorised call to SciPy - single call for all components!
                batch_probs = stats.norm.cdf(group_hazard_intensities, loc=mean, scale=stddev)
                all_probabilities[:, ds_idx] = batch_probs

            # Apply Monte Carlo simulation for each component
            for comp_idx, (index, _, _) in enumerate(group_components):
                component_pe_ds = all_probabilities[comp_idx]

                # Handle numerical stability
                component_pe_ds_stable = np.where(component_pe_ds == 0, -np.inf, component_pe_ds)

                # Monte Carlo simulation
                component_damage_state_ind[:, index] = np.sum(
                    component_pe_ds_stable >= rnd[:, index][:, np.newaxis], axis=1
                )

                processed_count += 1
        else:
            # Process small groups individually
            for index, component_key, component in group_components:
                hazard_intensity = hazard_intensities[index]
                component_pe_ds = get_vectorised_damage_probabilities(component, hazard_intensity)

                if len(component_pe_ds) == 0:
                    component_damage_state_ind[:, index] = 0
                else:
                    component_pe_ds_stable = np.where(
                        component_pe_ds == 0, -np.inf, component_pe_ds
                    )
                    component_damage_state_ind[:, index] = np.sum(
                        component_pe_ds_stable >= rnd[:, index][:, np.newaxis], axis=1
                    )
                processed_count += 1

    return processed_count


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
    component_damage_state_ind = np.zeros((scenario.num_samples, number_of_components), dtype=int)

    # Generate all random numbers at once
    rnd = random_number.uniform(size=(scenario.num_samples, number_of_components))

    # Component keys sorted for consistent processing order
    component_keys = sorted(infrastructure.components.keys())

    # Optimised progress reporting check
    show_progress = False
    quiet_mode = os.environ.get("SIRA_QUIET_MODE", "").lower() in ("1", "true", "yes")

    # For HPC with many hazards, disable progress by default to reduce overhead
    hpc_mode = int(os.environ.get("SIRA_HPC_MODE", "0"))
    if hpc_mode or number_of_components > 1000:
        show_progress = False
    elif not quiet_mode:
        # More efficient context detection using globals/locals inspection
        import inspect

        frame = inspect.currentframe()
        try:
            # Check calling function names in a more efficient way
            caller_names = []
            f = frame.f_back
            depth = 0
            while f and depth < 5:  # Limit depth to avoid excessive overhead
                caller_names.append(f.f_code.co_name)
                f = f.f_back
                depth += 1

            # Only show progress for direct calls, not in parallel contexts
            parallel_context_names = [
                "process_hazard_chunk",
                "_calculate_response_mpi",
                "worker_function",
            ]
            show_progress = not any(name in caller_names for name in parallel_context_names)
        finally:
            del frame  # Prevent reference cycles

    # Cache hazard intensities by location to avoid recalculation
    location_cache = {}
    component_locations = []
    component_hazard_intensities = np.zeros(number_of_components)

    # Pre-calculate hazard intensities for all components
    for index, component_key in enumerate(component_keys):
        component = infrastructure.components[component_key]
        loc_params = component.get_location()
        loc_key = tuple(loc_params)

        if loc_key not in location_cache:
            location_cache[loc_key] = hazard.get_hazard_intensity(*loc_params)

        component_locations.append(loc_params)
        component_hazard_intensities[index] = location_cache[loc_key]

    # Set up progress reporting only if needed
    if show_progress:
        hazard_id = getattr(hazard, "hazard_event_id", "").split("_")[-1]
        prefix = f"[Hazard {hazard_id}] " if hazard_id else ""
        print(f"\n{prefix}Processing components with optimised algorithm...")
        print(
            f"{prefix}Processing components: 0% complete (0/{number_of_components})",
            end="",
            flush=True,
        )

    # Optimised processing: batch components by damage state structure
    component_groups = {}
    for index, component_key in enumerate(component_keys):
        component = infrastructure.components[component_key]
        num_damage_states = len(component.damage_states)
        if num_damage_states not in component_groups:
            component_groups[num_damage_states] = []
        component_groups[num_damage_states].append((index, component_key, component))

    processed_components = 0
    progress_interval = max(1, number_of_components // 20)

    # HPC optimisation: adjust batch processing thresholds based on problem size
    hpc_mode = int(os.environ.get("SIRA_HPC_MODE", "0"))
    if hpc_mode or number_of_components > 1000:
        # More aggressive batching for HPC environments
        batch_threshold = max(2, min(10, number_of_components // 100))
    else:
        batch_threshold = 5

    # Process components in groups with similar damage state structures
    for num_damage_states, component_group in component_groups.items():
        # Use advanced batch processing for larger groups
        if len(component_group) >= batch_threshold:
            batch_processed = batch_process_similar_components(
                component_group,
                component_hazard_intensities,
                rnd,
                component_damage_state_ind,
                show_progress,
            )
            processed_components += batch_processed
        else:
            # Process smaller groups using individual vectorised approach
            for index, component_key, component in component_group:
                hazard_intensity = component_hazard_intensities[index]

                # Use vectorised function to get all damage state probabilities at once
                component_pe_ds = get_vectorised_damage_probabilities(component, hazard_intensity)

                # Handle empty result (component with only DS0 or errors)
                if len(component_pe_ds) == 0:
                    component_damage_state_ind[:, index] = 0
                else:
                    # Convert probabilities to -inf for numerical stability
                    # (matching original logic)
                    component_pe_ds_stable = np.where(
                        component_pe_ds == 0, -np.inf, component_pe_ds
                    )

                    # Vectorised damage state calculation using broadcast comparison
                    component_damage_state_ind[:, index] = np.sum(
                        component_pe_ds_stable >= rnd[:, index][:, np.newaxis], axis=1
                    )

                processed_components += 1

        # Update progress less frequently
        if show_progress and processed_components % progress_interval == 0:
            percent = int((processed_components / number_of_components) * 100)
            hazard_id = getattr(hazard, "hazard_event_id", "").split("_")[-1]
            prefix = f"[Hazard {hazard_id}] " if hazard_id else ""
            print(
                f"\r{prefix}Processing components: {percent}% complete "
                f"({processed_components}/{number_of_components})",
                end="",
                flush=True,
            )

    # Final update and newline
    if show_progress:
        hazard_id = getattr(hazard, "hazard_event_id", "").split("_")[-1]
        prefix = f"[Hazard {hazard_id}] " if hazard_id else ""
        print(
            f"\r{prefix}Processing components: 100% complete "
            f"({number_of_components}/{number_of_components})"
        )
        print(f"{prefix}Component processing complete.")

    return component_damage_state_ind
