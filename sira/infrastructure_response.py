import os
import sys

from sira import recovery_analysis
from sira.tools import utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

if sys.platform == "win32":
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
from pathlib import Path
from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Fore, init
from numba import njit

init()

matplotlib.use("Agg")
plt.switch_backend("agg")

rootLogger = logging.getLogger(__name__)
logging.getLogger("distributed.batched").setLevel(logging.WARNING)
logging.getLogger("distributed.comm").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
# mpl_looger = logging.getLogger("matplotlib")
# mpl_looger.setLevel(logging.WARNING)


# ****************************************************************************


# ****************************************************************************
# BEGIN POST-PROCESSING ...
# ****************************************************************************


def calc_tick_vals(val_list, xstep=0.1):
    num_ticks = int(round(len(val_list) / xstep)) + 1
    if (num_ticks > 12) and (num_ticks <= 20):
        xstep = 0.2
        num_ticks = int(round(len(val_list) / xstep)) + 1
    elif num_ticks > 20:
        num_ticks = 11
    tick_labels = val_list[:: (num_ticks - 1)]
    if isinstance(tick_labels[0], float):
        tick_labels = ["{:.3f}".format(val) for val in tick_labels]
    return tick_labels


def plot_mean_econ_loss(
    hazard_intensity_list: Union[List[float], np.ndarray],
    loss_array: Union[List[float], np.ndarray],
    x_label: str = "Hazard Intensity",
    y_label: str = "Direct Loss Fraction",
    fig_title: str = "Loss Ratio",
    fig_name: str = "fig_lossratio_boxplot",
    output_path: Union[str, Path] = ".",
) -> None:
    """Draws and saves a boxplot of mean economic loss"""

    # --------------------------------------------------------------------------

    econ_loss = np.array(loss_array)
    # econ_loss = econ_loss.transpose()

    x_values = list(hazard_intensity_list)
    y_values_list = econ_loss

    x_max = max(x_values)
    x_min = min(x_values)
    x_diff = x_max - x_min

    if x_diff <= 0.1:
        bin_width = 0.02
    elif x_diff <= 0.6:
        bin_width = 0.05
    elif x_diff <= 1.1:
        bin_width = 0.1
    elif x_diff <= 2.1:
        bin_width = 0.2
    elif x_diff <= 3.1:
        bin_width = 0.25
    elif x_diff <= 5.1:
        bin_width = 0.5
    elif x_diff <= 10:
        bin_width = 1
    elif x_diff <= 20:
        bin_width = 2
    elif x_diff <= 50:
        bin_width = 5
    elif x_diff <= 100:
        bin_width = 10
    else:
        bin_width = int(x_diff / 10)

    if x_diff <= 0.25:
        precision_digits = 3
    elif x_diff <= 0.5:
        precision_digits = 2
    elif x_diff <= 1.0:
        precision_digits = 2
    else:
        precision_digits = 1

    bin_edges = np.arange(0, x_max + bin_width, bin_width)
    if bin_edges[-1] > x_max:
        bin_edges[-1] = x_max
    binned_x = np.digitize(x_values, bin_edges, right=True)
    binned_x = binned_x * bin_width
    binned_x = binned_x[1:]

    all_x = []
    all_y = []

    for x, y_array in zip(binned_x, y_values_list):
        all_x.extend([x] * len(y_array))
        all_y.extend(y_array)

    pts = str(precision_digits)
    format_string = "{:." + pts + "f}-{:." + pts + "f}"

    bin_labels = [
        format_string.format(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)
    ]

    # --------------------------------------------------------------------------

    fig, ax = plt.subplots(1, figsize=(12, 7), facecolor="white")
    sns.set_theme(style="ticks", palette="Set2")
    sns.boxplot(
        ax=ax,
        x=all_x,
        y=all_y,
        linewidth=0.8,
        width=0.35,
        color="whitesmoke",
        showmeans=True,
        showfliers=True,
        meanprops=dict(marker="o", markeredgecolor="coral", markerfacecolor="coral"),
        flierprops=dict(marker="x", markerfacecolor="#BBB", markersize=6, linestyle="none"),
    )

    # --------------------------------------------------------------------------

    sns.despine(bottom=False, top=True, left=True, right=True, offset=None, trim=True)

    ax.spines["bottom"].set(linewidth=1.0, color="#444444", position=("axes", -0.02))

    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.5, color="#B6B6B6")

    ax.tick_params(axis="x", bottom=True, top=False, width=1.0, labelsize=10, color="#444444")

    ax.tick_params(axis="y", left=False, right=False, width=1.0, labelsize=10, color="#444444")

    ax.set_xticks(range(len(bin_labels)), bin_labels, rotation=45, ha="right")

    _, y_max = ax.get_ylim()
    y_max = np.round(y_max, 1)
    y_ticks = np.arange(0.0, y_max + 0.1, 0.2)
    ax.set_yticks(y_ticks)

    # --------------------------------------------------------------------------

    ax.set_xlabel(x_label, labelpad=9, size=11)
    ax.set_ylabel(y_label, labelpad=9, size=11)
    ax.set_title(fig_title, loc="center", y=1.04, fontsize=12, weight="bold")

    # --------------------------------------------------------------------------
    fig_name = fig_name + ".png"
    figfile = Path(output_path, fig_name)
    plt.margins(0.05)
    plt.savefig(figfile, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    # --------------------------------------------------------------------------


# =====================================================================================


# =====================================================================================


def calculate_loss_stats(df, progress_bar=True):
    """Calculate summary statistics for loss -- using pandas dataframe"""
    print()
    rootLogger.info(f"\n{Fore.CYAN}Calculating summary stats for system loss...{Fore.RESET}")
    return {
        "Mean": df.loss_mean.mean(),
        "Std": df.loss_mean.std(),
        "Min": df.loss_mean.min(),
        "Max": df.loss_mean.max(),
        "Median": df.loss_mean.quantile(0.5),
    }


def calculate_output_stats(df, progress_bar=True):
    """Calculate summary statistics for output -- using pandas dataframe"""
    print()
    rootLogger.info(f"{Fore.CYAN}Calculating summary stats for system output...{Fore.RESET}")
    return {
        "Mean": df.output_mean.mean(),
        "Std": df.output_mean.std(),
        "Min": df.output_mean.min(),
        "Max": df.output_mean.max(),
        "Median": df.output_mean.quantile(0.5),
    }


def calculate_recovery_stats(df, progress_bar=True):
    """Calculate summary statistics for recovery time -- using pandas dataframe"""
    print()
    rootLogger.info(f"\n{Fore.CYAN}Calculating summary stats for system recovery...{Fore.RESET}")
    return {
        "Mean": df.recovery_time_100pct.mean(),
        "Std": df.recovery_time_100pct.std(),
        "Min": df.recovery_time_100pct.min(),
        "Max": df.recovery_time_100pct.max(),
        "Median": df.recovery_time_100pct.quantile(0.5),
        # "Q1": df.recovery_time_100pct.quantile(0.25),
        # "Q3": df.recovery_time_100pct.quantile(0.75),
    }


def calculate_summary_statistics(df, calc_recovery=False):
    """Combine all summary statistics"""

    summary_stats = {
        "Loss": calculate_loss_stats(df),
        "Output": calculate_output_stats(df),
    }
    if calc_recovery:
        summary_stats["Recovery Time"] = calculate_recovery_stats(df)

    return summary_stats


def consolidate_streamed_results(
    stream_dir,
    infrastructure,
    scenario,
    config,
    hazards,
    CALC_SYSTEM_RECOVERY=False,
):
    """
    Consolidate per-event streamed artifacts into final CSV summaries.

    This consumes the manifest.jsonl plus the per-event Parquet/NPY files written
    during simulation when streaming mode is enabled, and produces at least:
      - system_response.csv (loss_mean/std, output_mean/std, optional recovery)
      - system_output_vs_hazard_intensity.csv (per-line mean capacity in %)
      - risk_summary_statistics.csv/.json

    Parameters
    ----------
    stream_dir : str | Path
        Directory containing manifest.jsonl and chunk_* subfolders
    infrastructure, scenario, config, hazards : as per write_system_response
    CALC_SYSTEM_RECOVERY : bool
        If True, attempts recovery analysis. Currently uses a safe fallback (zeros)
        to avoid high memory usage; a richer reconstruction can be added later.
    """
    import json
    from pathlib import Path as _Path

    import numpy as np
    import pandas as pd

    stream_dir = _Path(stream_dir)
    manifest_path = stream_dir / "manifest.jsonl"

    if not manifest_path.exists():
        rootLogger.warning(
            f"Streaming manifest not found at {manifest_path}. Skipping consolidation."
        )
        return

    rootLogger.info(
        "Consolidating streamed results from: %s",
        utils.wrap_file_path(str(stream_dir)),
    )

    # Get hazard event order from hazard file (CRITICAL for maintaining correct order)
    hazard_events = [str(eid) for eid in hazards.hazard_scenario_list]

    # Collect data in a dict keyed by event_id, then reorder by hazard file order
    event_data: dict[str, dict] = {}

    # Output line metadata (order and capacities)
    output_line_ids = list(infrastructure.output_nodes.keys())
    out_nodes_df = pd.DataFrame(infrastructure.output_nodes).T
    if "output_node_capacity" in out_nodes_df.columns:
        # Ensure capacities are numeric, converting any non-numeric to 1.0
        capacities_raw = out_nodes_df.loc[output_line_ids, "output_node_capacity"]
        line_capacities = (
            pd.to_numeric(capacities_raw, errors="coerce").fillna(1.0).to_numpy(dtype=float)
        )
    else:
        # Fallback: treat all capacities as 1.0 if field missing
        line_capacities = np.ones(len(output_line_ids), dtype=float)

    # Iterate manifest entries and collect data by event_id
    with open(manifest_path, "r", encoding="utf-8") as mf:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            event_id = str(rec.get("event_id"))
            econ_path = rec.get("econ")
            out_paths = rec.get("sys_output", []) or []

            # Economic loss: compressed NPZ format
            econ_vals: np.ndarray | None = None
            try:
                if econ_path and _Path(econ_path).exists():
                    # Handle both .npz (compressed) and .npy (legacy) formats
                    if econ_path.endswith(".npz"):
                        with np.load(econ_path, allow_pickle=False) as data:
                            econ_vals = data["data"].reshape(-1)
                    else:
                        econ_vals = np.load(econ_path, allow_pickle=False).reshape(-1)
            except Exception as e:
                rootLogger.warning(f"Failed reading econ for event {event_id}: {e}")

            if econ_vals is None or econ_vals.size == 0:
                # If missing, set to zeros (conservative)
                rootLogger.warning(f"No economic loss data for event {event_id}, using zeros")
                econ_vals = np.zeros(1, dtype=float)

            # Store economic loss stats in event_data
            event_record = {
                "loss_mean": float(np.mean(econ_vals)),
                "loss_std": float(np.std(econ_vals)),
            }

            # System output: handle both NPZ (compressed) and NPY formats
            sys_df: pd.DataFrame | None = None
            try:
                if isinstance(out_paths, list) and len(out_paths) > 0:
                    # Load all NPY files and concatenate
                    parts = []
                    for p in out_paths:
                        p_path = _Path(p)
                        if p_path.exists():
                            if str(p_path).endswith(".npz"):
                                with np.load(p_path, allow_pickle=False) as data:
                                    # Preferred: full samples stored under key 'data'
                                    if "data" in data.files:
                                        arr = data["data"]
                                    # Stats-only mode: synthesise samples from per-line means
                                    elif "mean" in data.files:
                                        mean = np.asarray(data["mean"])
                                        n = int(getattr(scenario, "num_samples", 1) or 1)
                                        arr = np.tile(mean, (n, 1))
                                    else:
                                        # Unknown NPZ structure; skip gracefully
                                        arr = None
                                    if isinstance(arr, np.ndarray):
                                        parts.append(pd.DataFrame(arr))
                            else:
                                parts.append(pd.DataFrame(np.load(p_path, allow_pickle=False)))
                    if parts:
                        sys_df = pd.concat(parts, axis=1, ignore_index=True)
                elif isinstance(out_paths, str) and _Path(out_paths).exists():
                    p_path = _Path(out_paths)
                    if str(p_path).endswith(".npz"):
                        with np.load(p_path, allow_pickle=False) as data:
                            if "data" in data.files:
                                arr = data["data"]
                            elif "mean" in data.files:
                                mean = np.asarray(data["mean"])
                                n = int(getattr(scenario, "num_samples", 1) or 1)
                                arr = np.tile(mean, (n, 1))
                            else:
                                arr = None
                            if isinstance(arr, np.ndarray):
                                sys_df = pd.DataFrame(arr)
                    else:
                        sys_df = pd.DataFrame(np.load(p_path, allow_pickle=False))
            except Exception as e:
                rootLogger.warning(f"Failed reading sys_output for event {event_id}: {e}")

            if sys_df is None or sys_df.empty:
                # When output data is missing, use zeros with correct dimensions
                rootLogger.error(
                    f"No system output data available for event {event_id}. "
                    f"Expected paths: {out_paths}. Using zero output."
                )
                num_samples = scenario.num_samples
                per_line_mean = np.zeros(len(output_line_ids), dtype=float)
                tot_out_per_sample = np.zeros(num_samples, dtype=float)
            else:
                # Ensure number of columns matches lines (best effort)
                # If fewer columns than lines, pad with zeros. If more, truncate.
                n_cols = sys_df.shape[1]
                if n_cols < len(output_line_ids):
                    pad = pd.DataFrame(
                        np.zeros((len(sys_df), len(output_line_ids) - n_cols), dtype=float)
                    )
                    sys_df = pd.concat([sys_df, pad], axis=1)
                elif n_cols > len(output_line_ids):
                    sys_df = sys_df.iloc[:, : len(output_line_ids)]

                # Per-line mean across samples
                per_line_mean = sys_df.mean(axis=0).to_numpy(dtype=float)
                # Total output per sample (sum across lines)
                tot_out_per_sample = sys_df.sum(axis=1).to_numpy(dtype=float)

            # Compute output fraction (0..1) relative to total capacity
            denom = float(infrastructure.system_output_capacity) or 1.0
            out_frac_per_sample = tot_out_per_sample / denom
            event_record["output_mean"] = float(np.mean(out_frac_per_sample))
            event_record["output_std"] = float(np.std(out_frac_per_sample))

            # For per-line percent of capacity - ensure all arrays are float64
            with np.errstate(divide="ignore", invalid="ignore"):
                per_line_mean = np.asarray(per_line_mean, dtype=np.float64)
                line_capacities_array = np.asarray(line_capacities, dtype=np.float64)
                perc = (per_line_mean / line_capacities_array) * 100.0
                perc = np.clip(perc, 0.0, 100.0)
                # Safe handling of non-finite values
                finite_mask = np.isfinite(perc)
                perc[~finite_mask] = 0.0

            event_record["per_line_means"] = perc  # type: ignore
            event_data[event_id] = event_record

    # Reconstruct arrays in hazard file order, not manifest order
    missing_events = [eid for eid in hazard_events if eid not in event_data]
    if missing_events:
        rootLogger.error(
            f"Missing {len(missing_events)} events in consolidation. "
            f"First 10 missing: {missing_events[:10]}"
        )

    event_ids: list[str] = []
    loss_mean: list[float] = []
    loss_std: list[float] = []
    output_mean: list[float] = []
    output_std: list[float] = []
    per_event_line_means: dict[str, np.ndarray] = {}

    for event_id in hazard_events:
        record = event_data.get(event_id)
        if not record:
            rootLogger.warning(f"No data for event {event_id}, using zeros")
            record = {
                "loss_mean": 0.0,
                "loss_std": 0.0,
                "output_mean": 0.0,
                "output_std": 0.0,
                "per_line_means": np.zeros(len(output_line_ids), dtype=float),
            }

        event_ids.append(event_id)
        loss_mean.append(record["loss_mean"])
        loss_std.append(record["loss_std"])
        output_mean.append(record["output_mean"])
        output_std.append(record["output_std"])
        per_event_line_means[event_id] = record["per_line_means"]

    # Build system_response DataFrame (now in correct hazard file order)
    hazard_col = hazards.HAZARD_INPUT_HEADER

    df_sys_response = pd.DataFrame(
        {
            "event_id": event_ids,
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "output_mean": output_mean,
            "output_std": output_std,
        }
    )

    precomputed_recovery_times: np.ndarray | list | None = None
    if CALC_SYSTEM_RECOVERY:
        hazard_event_list = [str(eid) for eid in hazards.hazard_scenario_list]
        components_costed = [
            comp_id
            for comp_id, component in infrastructure.components.items()
            if component.component_class not in infrastructure.uncosted_classes
        ]

        prev_force_no_mpi = os.environ.get("SIRA_FORCE_NO_MPI")
        os.environ["SIRA_FORCE_NO_MPI"] = "1"

        par_cfg = getattr(scenario, "parallel_config", None)
        original_backend = None
        par_cfg_backend_present = False
        try:
            if par_cfg is not None and hasattr(par_cfg, "config"):
                par_cfg_backend_present = "backend" in par_cfg.config
                original_backend = par_cfg.config.get("backend")
                par_cfg.config["backend"] = "multiprocessing"

            rootLogger.info("Calculating system recovery information from streamed results...")
            recovery_times = recovery_analysis.parallel_recovery_analysis(
                config=config,
                hazards=hazards,
                components=infrastructure.components,
                infrastructure=infrastructure,
                scenario=scenario,
                components_costed=components_costed,
                recovery_method=config.RECOVERY_METHOD,
                num_repair_streams=config.NUM_REPAIR_STREAMS,
                max_workers=getattr(scenario, "recovery_max_workers", None),
                batch_size=getattr(scenario, "recovery_batch_size", None),
            )
            if recovery_times is None:
                raise RuntimeError("Recovery analysis returned None")

            precomputed_recovery_times = recovery_times

            recovery_map = {
                str(event_id): float(value)
                for event_id, value in zip(hazard_event_list, recovery_times)
            }
            df_sys_response["recovery_time_100pct"] = [
                recovery_map.get(str(evt_id), 0.0) for evt_id in event_ids
            ]

        except Exception as exc:
            rootLogger.error("Recovery analysis during streaming consolidation failed: %s", exc)
            df_sys_response["recovery_time_100pct"] = 0.0
            precomputed_recovery_times = None
        finally:
            if par_cfg is not None and hasattr(par_cfg, "config"):
                if par_cfg_backend_present:
                    par_cfg.config["backend"] = original_backend
                else:
                    par_cfg.config.pop("backend", None)

            if prev_force_no_mpi is None:
                os.environ.pop("SIRA_FORCE_NO_MPI", None)
            else:
                os.environ["SIRA_FORCE_NO_MPI"] = prev_force_no_mpi

    else:
        df_sys_response["recovery_time_100pct"] = 0.0

    # Optionally insert hazard intensity values for facility-level models
    if (config.INFRASTRUCTURE_LEVEL).lower() == "facility":
        try:
            if config.HAZARD_INPUT_METHOD in ["calculated_array", "hazard_array"]:
                site_id = "0"
            else:
                component1 = list(infrastructure.components.values())[0]
                site_id = str(component1.site_id)
            haz_vals = hazards.hazard_data_df[site_id].values
            # Align hazard values to event_ids ordering if possible
            # If hazard_data_df indexed by event_id, map; else just append in original order
            if hasattr(hazards.hazard_data_df.index, "isin") and all(
                x in hazards.hazard_data_df.index for x in event_ids
            ):
                haz_series = hazards.hazard_data_df.loc[event_ids, site_id].reset_index(drop=True)
                df_sys_response.insert(1, hazard_col, haz_series.values)
            else:
                df_sys_response.insert(1, hazard_col, haz_vals[: len(df_sys_response)])
        except Exception as e:
            rootLogger.warning(f"Could not insert hazard intensity column: {e}")

    # Sort and write system_response
    df_sys_response = df_sys_response.sort_values("loss_mean", ascending=True)
    outfile_sys_response = _Path(config.OUTPUT_DIR, "system_response.csv")
    outpath_wrapped = utils.wrap_file_path(str(outfile_sys_response))
    rootLogger.info(
        f"Writing {Fore.CYAN}system hazard response data{Fore.RESET} to:\n"
        f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}"
    )
    df_sys_response.to_csv(outfile_sys_response, index=False)
    rootLogger.info("Done.\n")

    # Build and write system_output_vs_hazard_intensity.csv
    try:
        sys_output_df = pd.DataFrame.from_dict(per_event_line_means, orient="index")
        sys_output_df.columns = output_line_ids
        outfile_sysoutput = Path(config.OUTPUT_DIR, "system_output_vs_hazard_intensity.csv")
        outpath_wrapped = utils.wrap_file_path(str(outfile_sysoutput))
        rootLogger.info(
            f"Writing {Fore.CYAN}system line capacity data{Fore.RESET} to: \n"
            f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}"
        )
        sys_output_df.to_csv(outfile_sysoutput, sep=",", index_label=["hazard_event"])
        rootLogger.info("Done.\n")
    except Exception as e:
        rootLogger.error(f"Failed to write system_output_vs_hazard_intensity.csv: {e}")
        rootLogger.info("Continuing without system output vs hazard intensity file")
        print("-" * 81)

    # Risk summary statistics for streaming consolidation
    try:
        summary_stats = calculate_summary_statistics(
            df_sys_response, calc_recovery=CALC_SYSTEM_RECOVERY
        )
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(_Path(config.OUTPUT_DIR, "risk_summary_statistics.csv"))
        summary_df.to_json(_Path(config.OUTPUT_DIR, "risk_summary_statistics.json"))
        print(f"\n{Fore.CYAN}Summary Statistics:{Fore.RESET}")
        print(summary_df.round(4), "\n")
    except Exception as e:
        rootLogger.warning(f"Failed calculating risk summary statistics: {e}")

    # ================================================================================
    # COMPLETE Response Reconstruction from Streaming Data
    # ================================================================================
    try:
        rootLogger.info("Reconstructing complete response data from streaming artifacts...")

        # Reconstruct response_list structure from streaming data to enable
        # full write_system_response
        response_list = reconstruct_response_list_from_streaming(
            stream_dir, manifest_path, infrastructure, scenario, hazards, config
        )

        if response_list is not None:
            rootLogger.info(
                "Successfully reconstructed response data. Generating all output files..."
            )

            # Call the full write_system_response function with reconstructed data
            # Disable recovery calculation since it was already done during basic consolidation
            write_system_response(
                response_list,
                infrastructure,
                scenario,
                config,
                hazards,
                CALC_SYSTEM_RECOVERY=False,  # Disable to prevent MPI conflicts
                precomputed_recovery_times=precomputed_recovery_times,
            )
            if precomputed_recovery_times is not None:
                rootLogger.info("[OK] All output files generated successfully (recovery reused)")
            elif CALC_SYSTEM_RECOVERY:
                rootLogger.info(
                    "[OK] All output files generated successfully (recovery calculated)"
                )
            else:
                rootLogger.info("[OK] All output files generated successfully")
        else:
            rootLogger.warning(
                "Failed to reconstruct response data. Falling back to basic outputs."
            )
            # Keep the existing basic outputs we already generated above

    except Exception as e:
        rootLogger.warning(f"Failed to reconstruct complete response data: {e}")
        import traceback

        rootLogger.debug(traceback.format_exc())

    rootLogger.info("Streaming consolidation complete.")


def reconstruct_response_list_from_streaming(
    stream_dir, manifest_path, infrastructure, scenario, hazards, config
):
    """
    Reconstruct the response_list structure from streaming data to enable full write_system_response

    Returns response_list in the format expected by write_system_response:
    [
        event_vs_dmg_indices,
        sys_output_dict,
        comp_response_dict,
        comptype_resp_dict,
        sys_output_array,
        sys_economic_loss_array
    ]
    """
    import json
    from pathlib import Path as _Path

    import numpy as np

    try:
        rootLogger.info("Reconstructing response arrays from streaming data...")

        save_component_response = os.environ.get("SIRA_SAVE_COMPONENT_RESPONSE", "1") == "1"
        save_comptype_response = os.environ.get("SIRA_SAVE_COMPTYPE_RESPONSE", "1") == "1"

        hazard_events = [str(eid) for eid in hazards.hazard_scenario_list]
        num_samples = scenario.num_samples  # Use correct value - NO fallback
        component_ids_sorted = list(np.sort(list(infrastructure.components.keys())))
        component_types_sorted = sorted(infrastructure.get_component_types())
        num_components = len(component_ids_sorted)
        num_lines = len(infrastructure.output_nodes)

        event_records: dict[str, dict] = {}

        with open(manifest_path, "r", encoding="utf-8") as mf:
            for raw_line in mf:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    event_id = str(entry.get("event_id"))
                    econ_path = _Path(entry.get("econ", ""))
                    chunk_dir = econ_path.parent if econ_path else _Path(stream_dir)

                    safe_event_id = "".join(
                        ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in event_id
                    )

                    record: dict[str, object] = {}

                    # --- Economic loss samples ---
                    # All streaming data now uses NPY format (simpler and more reliable)
                    expected_econ_size = entry.get("econ_size", 0)
                    econ_values = np.zeros(num_samples, dtype=float)
                    if econ_path and econ_path.exists():
                        try:
                            # Check for file size mismatch (potential corruption)
                            actual_size = econ_path.stat().st_size
                            if expected_econ_size > 0 and actual_size != expected_econ_size:
                                rootLogger.warning(
                                    f"Economic loss file size mismatch for event {event_id}: "
                                    f"expected {expected_econ_size}, got {actual_size}. "
                                    f"File may be corrupted."
                                )

                            # Read compressed or uncompressed format
                            if str(econ_path).endswith(".npz"):
                                with np.load(econ_path, allow_pickle=False) as data:
                                    econ_values = data["data"].flatten()
                            else:
                                econ_values = np.load(econ_path, allow_pickle=False).flatten()
                        except Exception as exc:
                            rootLogger.warning(
                                f"Failed to load economic loss for event {event_id}: {exc}"
                            )

                    if econ_values.size != num_samples:
                        econ_values = np.resize(econ_values, num_samples)
                    record["econ"] = econ_values

                    # --- System output samples ---
                    sys_output_files = entry.get("sys_output", []) or []
                    expected_sizes = entry.get("sys_output_sizes", [])
                    output_totals = np.zeros(num_samples, dtype=float)
                    line_means = np.zeros(num_lines, dtype=float)

                    output_parts: list[np.ndarray] = []
                    for idx, out_file in enumerate(sys_output_files):
                        try:
                            out_path = _Path(out_file)
                            if not out_path.exists():
                                rootLogger.warning(f"System output file does not exist: {out_path}")
                                continue

                            # Check for file size mismatch (potential corruption)
                            actual_size = out_path.stat().st_size
                            if idx < len(expected_sizes) and expected_sizes[idx] > 0:
                                if actual_size != expected_sizes[idx]:
                                    rootLogger.warning(
                                        f"System output file size mismatch for event {event_id}: "
                                        f"{out_path.name} expected {expected_sizes[idx]}, "
                                        f"got {actual_size}. File may be corrupted."
                                    )

                            # Support multiple formats for backward compatibility:
                            # 1. Compressed NPZ with full samples (.npz with 'data' key)
                            # 2. Statistics-only NPZ (.npz with 'mean', 'std', etc.)
                            # 3. Legacy uncompressed NPY (.npy)
                            if out_path.suffix == ".npz":
                                loaded = np.load(out_path, allow_pickle=False)
                                if "data" in loaded:
                                    # Full samples stored with compression
                                    out_array = loaded["data"]
                                elif "mean" in loaded:
                                    # Statistics-only mode: reconstruct approximate samples
                                    # Use mean +/- std to create pseudo-samples
                                    mean_val = loaded["mean"]
                                    std_val = loaded["std"]
                                    min_val = loaded["min"]
                                    max_val = loaded["max"]

                                    # Create synthetic samples that match the statistics
                                    # Approximation for when full samples weren't stored
                                    n_lines = len(mean_val)
                                    synthetic_samples = np.zeros(
                                        (num_samples, n_lines), dtype=float
                                    )
                                    for i in range(n_lines):
                                        # Generate samples with correct mean/std,
                                        # clipped to min/max
                                        synthetic_samples[:, i] = np.clip(
                                            np.random.normal(mean_val[i], std_val[i], num_samples),
                                            min_val[i],
                                            max_val[i],
                                        )
                                    out_array = synthetic_samples
                                    rootLogger.debug(
                                        f"Reconstructed samples from statistics "
                                        f"for event {event_id}"
                                    )
                                else:
                                    raise ValueError(f"Unrecognized NPZ format in {out_path}")
                            else:
                                # Legacy uncompressed NPY format
                                out_array = np.load(out_path, allow_pickle=False)

                            output_parts.append(out_array)
                        except Exception as exc:
                            rootLogger.warning(
                                f"Failed to load system output for event {event_id} "
                                f"from {out_file}: {exc}"
                            )

                    if not output_parts:
                        rootLogger.error(
                            f"No system output data loaded for event {event_id}. "
                            f"Expected files: {sys_output_files}. Output will be zero."
                        )
                    else:
                        combined_output = np.concatenate(output_parts, axis=1)
                        if combined_output.shape[0] != num_samples:
                            rootLogger.warning(
                                f"Output shape mismatch for event {event_id}: "
                                f"expected {num_samples} samples, "
                                f"got {combined_output.shape[0]}. Resizing."
                            )
                            combined_output = np.resize(
                                combined_output,
                                (num_samples, combined_output.shape[1]),
                            )
                        output_totals = np.sum(combined_output, axis=1)
                        line_means_raw = np.mean(combined_output, axis=0)
                        line_means[: min(len(line_means_raw), num_lines)] = line_means_raw[
                            :num_lines
                        ]

                    record["output_total"] = output_totals
                    record["output_lines"] = {
                        line_id: line_means[idx] if idx < len(line_means) else 0.0
                        for idx, line_id in enumerate(infrastructure.output_nodes.keys())
                    }

                    # --- Damage state indicators ---
                    damage_path_str = entry.get("damage")
                    damage_array = np.zeros((num_samples, num_components), dtype=np.int16)
                    possible_paths = []
                    if damage_path_str:
                        possible_paths.append(_Path(damage_path_str))
                    # Check both compressed (.npz) and legacy (.npy) formats
                    possible_paths.append(chunk_dir / f"{safe_event_id}__damage.npz")
                    possible_paths.append(chunk_dir / f"{safe_event_id}__damage.npy")

                    for dmg_path in possible_paths:
                        if dmg_path and dmg_path.exists():
                            try:
                                # Load compressed or uncompressed format
                                if str(dmg_path).endswith(".npz"):
                                    with np.load(dmg_path, allow_pickle=False) as data:
                                        dmg = data["data"]
                                else:
                                    dmg = np.load(dmg_path, allow_pickle=False)
                                dmg = np.asarray(dmg)
                                if dmg.shape[0] != num_samples:
                                    dmg = np.resize(dmg, (num_samples, dmg.shape[1]))
                                if dmg.shape[1] != num_components:
                                    dmg = np.resize(dmg, (num_samples, num_components))
                                damage_array = dmg.astype(np.int16, copy=False)
                                break
                            except Exception as exc:
                                rootLogger.debug(
                                    f"Failed to load damage indices for event {event_id}: {exc}"
                                )

                    record["damage"] = damage_array

                    # --- Component responses ---
                    if save_component_response:
                        comp_response_file = chunk_dir / f"{safe_event_id}__comp_response.json"
                        if comp_response_file.exists():
                            try:
                                with open(comp_response_file, "r", encoding="utf-8") as fh:
                                    record["comp_response"] = json.load(fh)
                            except Exception as exc:
                                rootLogger.debug(
                                    f"Failed to load component response for event {event_id}: {exc}"
                                )

                    if save_comptype_response:
                        comptype_response_file = chunk_dir / (
                            f"{safe_event_id}__comptype_response.json"
                        )
                        if comptype_response_file.exists():
                            try:
                                with open(comptype_response_file, "r", encoding="utf-8") as fh:
                                    record["comptype_response"] = json.load(fh)
                            except Exception as exc:
                                rootLogger.debug(
                                    (
                                        "Failed to load component type response for "
                                        f"event {event_id}: {exc}"
                                    )
                                )

                    event_records[event_id] = record

                except Exception as exc:
                    rootLogger.debug(f"Failed to process manifest entry: {exc}")
                    continue

        # Align results to hazard event order
        # Validate all events from hazard file are present in reconstruction
        missing_events = [eid for eid in hazard_events if str(eid) not in event_records]
        if missing_events:
            rootLogger.error(
                f"Missing {len(missing_events)} events in streaming reconstruction. "
                f"First 10 missing: {missing_events[:10]}"
            )

        event_vs_dmg_indices: dict[str, np.ndarray] = {}
        sys_output_dict: dict[str, dict] = {}
        comp_response_events: dict[str, dict] = {}
        comptype_response_events: dict[str, dict] = {}
        sys_economic_loss_list: list[np.ndarray] = []
        sys_output_totals_list: list[np.ndarray] = []

        component_metrics = ["loss_mean", "loss_std", "func_mean", "func_std", "failure_rate"]
        comptype_metrics = [
            "loss_mean",
            "loss_std",
            "loss_tot",
            "func_mean",
            "func_std",
            "failure_rate",
        ]

        # Reconstruct arrays in the EXACT order of hazard_events (preserves hazard file order)
        for event_id in hazard_events:
            record = event_records.get(event_id, {})

            if not record:
                # Warn about missing event data
                rootLogger.warning(
                    f"No data found for event {event_id} in streaming files. "
                    f"Using zeros for loss, 100% capacity for output."
                )

            econ_values = np.asarray(record.get("econ", np.zeros(num_samples)), dtype=float)
            if econ_values.size != num_samples:
                econ_values = np.resize(econ_values, num_samples)
            sys_economic_loss_list.append(econ_values)

            output_totals = np.asarray(
                record.get("output_total", 100 * np.ones(num_samples)), dtype=float
            )
            if output_totals.size != num_samples:
                output_totals = np.resize(output_totals, num_samples)
            sys_output_totals_list.append(output_totals)

            output_lines = record.get("output_lines")
            if isinstance(output_lines, dict):
                sys_output_dict[event_id] = output_lines
            else:
                sys_output_dict[event_id] = {
                    line_id: 0.0 for line_id in infrastructure.output_nodes.keys()
                }

            damage_array = record.get("damage")
            if isinstance(damage_array, np.ndarray):
                event_vs_dmg_indices[event_id] = damage_array
            else:
                event_vs_dmg_indices[event_id] = np.zeros(
                    (num_samples, num_components), dtype=np.int16
                )

            # Reconstruct component response dict with tuple keys
            comp_resp_data = {}
            comp_json = record.get("comp_response") if save_component_response else None
            if isinstance(comp_json, dict):
                for comp_id in component_ids_sorted:
                    comp_key = str(comp_id)
                    metrics = comp_json.get(comp_key, {}) if isinstance(comp_json, dict) else {}
                    for metric in component_metrics:
                        value = metrics.get(metric, 0.0)
                        try:
                            comp_resp_data[(comp_id, metric)] = float(value)
                        except (TypeError, ValueError):
                            comp_resp_data[(comp_id, metric)] = 0.0
            comp_response_events[event_id] = comp_resp_data

            # Reconstruct component type response dict with tuple keys
            comptype_resp_data = {}
            comptype_json = record.get("comptype_response") if save_comptype_response else None
            if isinstance(comptype_json, dict):
                for comptype in component_types_sorted:
                    comp_key = str(comptype)
                    metrics = (
                        comptype_json.get(comp_key, {}) if isinstance(comptype_json, dict) else {}
                    )
                    for metric in comptype_metrics:
                        value = metrics.get(metric, 0.0)
                        try:
                            comptype_resp_data[(comptype, metric)] = float(value)
                        except (TypeError, ValueError):
                            comptype_resp_data[(comptype, metric)] = 0.0
            comptype_response_events[event_id] = comptype_resp_data

        if not sys_economic_loss_list:
            rootLogger.warning("No economic loss data reconstructed; cannot build response list")
            return None

        sys_economic_loss_array = np.asarray(sys_economic_loss_list, dtype=float).T
        sys_output_array = np.asarray(sys_output_totals_list, dtype=float).T

        response_list = [
            event_vs_dmg_indices,
            sys_output_dict,
            comp_response_events,
            comptype_response_events,
            sys_output_array,
            sys_economic_loss_array,
        ]

        # ========================================================================
        # FINAL DATA QUALITY SUMMARY
        # ========================================================================
        total_events = len(hazard_events)
        events_with_data = len(event_records)
        missing_count = total_events - events_with_data

        if missing_count > 0:
            missing_pct = (missing_count / total_events) * 100
            rootLogger.error(
                f"\n{'=' * 80}\n"
                f"STREAMING CONSOLIDATION ERROR SUMMARY:\n"
                f"{'=' * 80}\n"
                f"Total events expected:        {total_events:,}\n"
                f"Events with data:             {events_with_data:,}\n"
                f"Events missing data:          {missing_count:,} ({missing_pct:.1f}%)\n"
                f"\n"
                f"IMPACT:\n"
                f"  - Economic loss:  {missing_count:,} events set to ZERO (conservative)\n"
                f"  - System output:  {missing_count:,} events set to 100% (optimistic)\n"
                f"\n"
                f"ROOT CAUSE: Incomplete manifest consolidation during MPI streaming.\n"
                f"            Not all rank manifest files were found during consolidation.\n"
                f"\n"
                f"ACTION REQUIRED: Results are UNRELIABLE. Check simulation.py logs for\n"
                f"                 manifest consolidation errors and missing rank manifests.\n"
                f"{'=' * 80}\n"
            )
        else:
            rootLogger.info(
                f"✓ Data quality check: All {total_events:,} events successfully "
                f"reconstructed from streaming data."
            )

        rootLogger.info("Response list reconstruction completed successfully")
        return response_list

    except Exception as e:
        rootLogger.error(f"Failed to reconstruct response list: {e}")
        import traceback

        rootLogger.debug(traceback.format_exc())
        return None


def write_system_response(
    response_list,
    infrastructure,
    scenario,
    config,
    hazards,
    CALC_SYSTEM_RECOVERY=False,
    precomputed_recovery_times: np.ndarray | list | None = None,
):
    # ---------------------------------------------------------------------------------
    # Hazard response for component types
    # ---------------------------------------------------------------------------------
    # Check environment variable to control comptype response generation
    save_comptype_response = os.environ.get("SIRA_SAVE_COMPTYPE_RESPONSE", "1") == "1"

    if save_comptype_response:
        try:
            comptype_resp_dict = response_list[3]
        except Exception:
            comptype_resp_dict = {}

        if comptype_resp_dict:
            comptype_resp_df = pd.DataFrame(comptype_resp_dict)
            comptype_resp_df.index.names = ["component_type", "response"]
            comptype_resp_df = comptype_resp_df.transpose()
            comptype_resp_df.index.name = "hazard_event"

            outfile_comptype_resp = Path(config.OUTPUT_DIR, "comptype_response.csv")
            print("-" * 81)
            outpath_wrapped = utils.wrap_file_path(str(outfile_comptype_resp))
            rootLogger.info(
                f"Writing {Fore.CYAN}component type response{Fore.RESET} to: \n"
                f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}"
            )
            comptype_resp_df.to_csv(outfile_comptype_resp, sep=",")
            rootLogger.info("Done.\n")
            # Cleanup $OUTPUT_DIR/mpi_comptype_parts if present
            try:
                cleanup_dir = Path(config.OUTPUT_DIR, "mpi_comptype_parts")
                if cleanup_dir.exists() and cleanup_dir.is_dir():
                    import shutil as _shutil

                    _shutil.rmtree(cleanup_dir, ignore_errors=True)
                    rootLogger.info(f"Cleaned up comptype partials directory: {cleanup_dir}")
            except Exception as _ce:
                rootLogger.warning(f"Failed to clean up comptype partials directory: {_ce}")
        else:
            # Merge per-rank partials produced by MPI path if available
            parts_dir_env = os.environ.get("SIRA_MPI_COMPTYPE_PARTS_DIR", "")
            parts_dir = (
                Path(parts_dir_env)
                if parts_dir_env
                else Path(config.OUTPUT_DIR) / "mpi_comptype_parts"
            )
            part_files = sorted(parts_dir.glob("comptype_rank_*.csv")) if parts_dir.exists() else []
            if not part_files:
                rootLogger.error(
                    "comptype_response.csv is required but neither in-memory data "
                    "nor partials were found."
                )
            else:
                frames = []
                for pf in part_files:
                    try:
                        df = pd.read_csv(pf, index_col=0)
                        df.index.name = "hazard_event"
                        frames.append(df)
                    except Exception as e:
                        rootLogger.warning(f"Failed to read comptype partial {pf}: {e}")
                if frames:
                    merged = pd.concat(frames, axis=0)
                    try:
                        merged = merged.sort_index(axis=0)
                    except Exception:
                        pass
                    outfile_comptype_resp = Path(config.OUTPUT_DIR, "comptype_response.csv")
                    outpath_wrapped = utils.wrap_file_path(str(outfile_comptype_resp))
                    rootLogger.info(
                        f"Writing {Fore.CYAN}component type response{Fore.RESET} to: \n"
                        f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}"
                    )
                    merged.to_csv(outfile_comptype_resp, sep=",")
                    rootLogger.info("Done.\n")
                    # Cleanup $OUTPUT_DIR/mpi_comptype_parts after success
                    try:
                        cleanup_dir = Path(config.OUTPUT_DIR, "mpi_comptype_parts")
                        if cleanup_dir.exists() and cleanup_dir.is_dir():
                            import shutil as _shutil

                            _shutil.rmtree(cleanup_dir, ignore_errors=True)
                            rootLogger.info(
                                f"Cleaned up comptype partials directory: {cleanup_dir}"
                            )
                    except Exception as _ce:
                        rootLogger.warning(f"Failed to clean up comptype partials directory: {_ce}")
                else:
                    rootLogger.error(
                        "Unable to construct comptype_response.csv from partials; no valid frames."
                    )
        # try:
        #     comptype_resp_dict = response_list[3]
        #     comptype_resp_df = pd.DataFrame(comptype_resp_dict)
        #     comptype_resp_df.index.names = ["component_type", "response"]
        #     comptype_resp_df = comptype_resp_df.transpose()
        #     comptype_resp_df.index.name = "hazard_event"

        #     outfile_comptype_resp = Path(config.OUTPUT_DIR, "comptype_response.csv")
        #     print("-" * 81)
        #     outpath_wrapped = utils.wrap_file_path(str(outfile_comptype_resp))
        #     rootLogger.info(
        #         f"Writing {Fore.CYAN}component type response{Fore.RESET} to: \n"
        #         f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}"
        #     )
        #     comptype_resp_df.to_csv(outfile_comptype_resp, sep=",")
        #     rootLogger.info("Done.\n")
        # except Exception as e:
        #     rootLogger.error(f"Failed to write comptype_response.csv: {e}")
        #     rootLogger.info("Skipping comptype_response.csv due to error during file generation")
        #     print("-" * 81)
    else:
        rootLogger.info(
            "Skipping comptype_response.csv (disabled by SIRA_SAVE_COMPTYPE_RESPONSE=0)"
        )
        print("-" * 81)

    # ---------------------------------------------------------------------------------
    # Output File -- response of each COMPONENT to hazard
    # ---------------------------------------------------------------------------------
    # Check environment variable to control component response generation
    save_component_response = os.environ.get("SIRA_SAVE_COMPONENT_RESPONSE", "1") == "1"

    if save_component_response:
        try:
            costed_component_ids = set()
            for comp_id, component in infrastructure.components.items():
                if component.component_class not in infrastructure.uncosted_classes:
                    costed_component_ids.add(comp_id)
            costed_component_ids = sorted(list(costed_component_ids))

            comp_response_list = response_list[2]
            component_resp_df = pd.DataFrame(comp_response_list)
            component_resp_df.columns = hazards.hazard_scenario_list
            component_resp_df.index.names = ["component_id", "response"]

            component_ids = component_resp_df.index.get_level_values("component_id").unique()
            component_ids = [str(x) for x in component_ids]

            # Filter for costed components
            component_resp_df = component_resp_df[
                component_resp_df.index.get_level_values("component_id").isin(costed_component_ids)
            ]

            component_resp_df = component_resp_df.transpose()
            component_resp_df.index.names = ["hazard_event"]

            event_vs_dmg_indices = response_list[0]
            dmgidx_medians = {
                k: (np.round(np.median(v, axis=0) + 0.01)).astype(int)
                for k, v in event_vs_dmg_indices.items()
            }

            comp_dmgidx_df = pd.DataFrame.from_dict(dmgidx_medians, orient="index")
            comp_dmgidx_df.index.name = "hazard_event"
            comp_dmgidx_df.columns = component_ids
            comp_dmgidx_df = comp_dmgidx_df[list(costed_component_ids)]

            comp_dmgidx_df_multiindex = comp_dmgidx_df.copy()
            comp_dmgidx_df_multiindex.columns = pd.MultiIndex.from_product(
                [comp_dmgidx_df.columns, ["damage_index"]], names=["component_id", "response"]
            )

            component_resp_df = pd.concat([comp_dmgidx_df_multiindex, component_resp_df], axis=1)
            component_resp_df = component_resp_df.sort_index(axis=1, level=0)

            # ----------------------------------------------------------------------------
            # Get hazard intensities for all components across all events
            hazard_intensities = {}
            component_locations = {
                comp_id: comp.get_location() for comp_id, comp in infrastructure.components.items()
            }

            for comp_id in costed_component_ids:
                loc_params = component_locations[comp_id]
                if config.HAZARD_INPUT_METHOD in ["calculated_array", "hazard_array"]:
                    site_id = "0"
            else:
                site_id = str(loc_params[0]) if isinstance(loc_params, tuple) else "0"

            if site_id in hazards.hazard_data_df.columns:
                hazard_intensities[comp_id] = hazards.hazard_data_df[site_id].values
            else:
                hazard_intensities[comp_id] = np.zeros(len(hazards.hazard_scenario_list))

            # Create hazard intensity DataFrame with multiindex columns
            hazard_df = pd.DataFrame(hazard_intensities, index=hazards.hazard_scenario_list)
            hazard_df_multiindex = hazard_df.copy()
            hazard_df_multiindex.columns = pd.MultiIndex.from_product(
                [hazard_df.columns, ["hazard_intensity"]], names=["component_id", "response"]
            )

            # Concatenate with existing component_resp_df
            component_resp_df = pd.concat([hazard_df_multiindex, component_resp_df], axis=1)
            component_resp_df = component_resp_df.sort_index(axis=1, level=0, sort_remaining=False)

            # ----------------------------------------------------------------------------
            outfile_comp_resp = Path(config.OUTPUT_DIR, "component_response.csv")
            outpath_wrapped = utils.wrap_file_path(str(outfile_comp_resp))
            rootLogger.info(
                f"Writing component hazard response data to: \n"
                f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}"
            )

            component_resp_df.to_csv(outfile_comp_resp, sep=",")
            # component_resp_df.to_parquet(
            #     outfile_comp_resp, engine='pyarrow', index=True, compression='snappy')
            rootLogger.info("Done.\n")
        except Exception as e:
            rootLogger.error(f"Failed to write component_response.csv: {e}")
            rootLogger.info("Skipping component_response.csv due to error during file generation")
            print("-" * 81)
    else:
        rootLogger.info(
            "Skipping component_response.csv (disabled by SIRA_SAVE_COMPONENT_RESPONSE=0)"
        )
        print("-" * 81)

    # =================================================================================
    # System output file (for given hazard transfer parameter value)
    # ---------------------------------------------------------------------------------
    sys_output_dict = response_list[1]

    rootLogger.info("Collating data on output line capacities of system ...")
    sys_output_df = pd.DataFrame(sys_output_dict)
    sys_output_df = sys_output_df.transpose()
    sys_output_df.index.name = "event_id"

    # Get individual line capacities from output_nodes
    output_nodes_dict = infrastructure.output_nodes
    output_nodes_df = pd.DataFrame(output_nodes_dict)
    output_nodes_df = output_nodes_df.transpose()
    output_nodes_df.index.name = "output_node"
    line_capacities = output_nodes_df["output_node_capacity"]

    # Calculate percentage values (0-100) by dividing by each line's capacity
    for line in sys_output_df.columns:
        # Get the capacity for the specific line
        line_capacity = line_capacities[line]
        # Convert to percentage of line capacity
        sys_output_df[line] = (sys_output_df[line] / line_capacity) * 100
        # Ensure values are between 0 and 100
        sys_output_df[line] = sys_output_df[line].clip(0, 100)

    try:
        outfile_sysoutput = Path(config.OUTPUT_DIR, "system_output_vs_hazard_intensity.csv")
        outpath_wrapped = utils.wrap_file_path(str(outfile_sysoutput))
        rootLogger.info(
            f"Writing {Fore.CYAN}system line capacity data{Fore.RESET} to: \n"
            f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}"
        )
        sys_output_df.to_csv(outfile_sysoutput, sep=",", index_label=[sys_output_df.index.name])
        rootLogger.info("Done.\n")
    except Exception as e:
        rootLogger.error(f"Failed to write system_output_vs_hazard_intensity.csv: {e}")
        rootLogger.info("Continuing without system output vs hazard intensity file")
        print("-" * 81)

    # =================================================================================
    # Calculate recovery times for each hazard event FIRST
    # ---------------------------------------------------------------------------------

    hazard_event_list = hazards.hazard_data_df.index.tolist()
    if precomputed_recovery_times is not None:
        try:
            recovery_time_100pct = np.asarray(precomputed_recovery_times, dtype=np.float64)
        except Exception:
            rootLogger.warning(
                "Failed to coerce precomputed recovery times to float; defaulting to zeros"
            )
            recovery_time_100pct = np.zeros(len(hazard_event_list), dtype=np.float64)
        if recovery_time_100pct.size != len(hazard_event_list):
            rootLogger.warning(
                "Precomputed recovery times length mismatch. Adjusting to hazard event list."
            )
            recovery_time_100pct = np.resize(recovery_time_100pct, len(hazard_event_list))
    elif CALC_SYSTEM_RECOVERY:
        components_uncosted = [
            comp_id
            for comp_id, component in infrastructure.components.items()
            if component.component_class in infrastructure.uncosted_classes
        ]
        components_costed = [
            comp_id
            for comp_id in infrastructure.components.keys()
            if comp_id not in components_uncosted
        ]

        #######################################################################
        #######################################################################

        rootLogger.info("Calculating system recovery information ...")

        # Determine recovery method from scenario configuration
        recovery_method = config.RECOVERY_METHOD  # Default is 'max'
        num_repair_streams = config.NUM_REPAIR_STREAMS  # Default is 100

        rootLogger.info(f"Recovery method: {recovery_method}")
        if recovery_method == "parallel_streams":
            rootLogger.info(f"Number of repair streams: {num_repair_streams}")

        rootLogger.info(f"Number of costed components: {len(components_costed)}")
        rootLogger.info(f"Number of hazard events: {len(hazard_event_list)}")
        # Only log component response DataFrame shape if it was created
        if save_component_response:
            try:
                rootLogger.info(f"Component response DataFrame shape: {component_resp_df.shape}")
            except NameError:
                rootLogger.info("Component response DataFrame: not created due to error")
        else:
            rootLogger.info("Component response DataFrame: disabled by environment variable")

        # ----------------------------------------------------------------------------
        # RECOVERY processing -- parallel method with sequential fallback
        # ----------------------------------------------------------------------------
        try:
            rootLogger.info("Attempting optimised parallel recovery analysis...")
            recovery_time_100pct = recovery_analysis.parallel_recovery_analysis(
                config=config,
                hazards=hazards,
                components=infrastructure.components,
                infrastructure=infrastructure,
                scenario=scenario,
                components_costed=components_costed,
                recovery_method=recovery_method,
                num_repair_streams=num_repair_streams,
                max_workers=getattr(scenario, "recovery_max_workers", None),
                batch_size=getattr(scenario, "recovery_batch_size", None),
            )
            rootLogger.info(
                f"{Fore.GREEN}Optimised parallel recovery analysis succeeded.{Fore.RESET}"
            )
        except Exception as e:
            rootLogger.warning(
                f"{Fore.RED}Optimised parallel recovery analysis failed: {e}{Fore.RESET}"
            )
            rootLogger.info("Falling back to sequential recovery analysis...")
            try:
                recovery_time_100pct = recovery_analysis.sequential_recovery_analysis(
                    hazard_event_list,
                    infrastructure,
                    config,
                    hazards,
                    components_costed,
                )
                rootLogger.info("Sequential recovery analysis completed successfully")
            except Exception as e3:
                rootLogger.error(f"*** Sequential recovery analysis also failed: {e3}")
                rootLogger.info("Using zeros for all recovery times")
                recovery_time_100pct = [0] * len(hazard_event_list)

        # ----------------------------------------------------------------------------
        # Version with sequential RECOVERY processing only
        # ----------------------------------------------------------------------------
        # try:
        #     rootLogger.info("Initiating sequential recovery analysis...")
        #     recovery_time_100pct = recovery_analysis.sequential_recovery_analysis(
        #         hazard_event_list,
        #         infrastructure,
        #         config,
        #         hazards,
        #         components_costed,
        #     )
        #     rootLogger.info("Sequential recovery analysis completed successfully")
        # except Exception as e3:
        #     rootLogger.error(f"*** Recovery analysis failed: {e3}")
        #     rootLogger.info("Using zeros for all recovery times")
        #     recovery_time_100pct = [0.0] * len(hazard_event_list)

        # ----------------------------------------------------------------------------
        # Validate & sanitise recovery times (vectorised)
        # ----------------------------------------------------------------------------
        if recovery_time_100pct is not None:
            rootLogger.info(f"Recovery analysis returned: {type(recovery_time_100pct)}")

            # Convert to float64 numpy array efficiently; coerce invalids to NaN
            try:
                if isinstance(recovery_time_100pct, np.ndarray):
                    arr = recovery_time_100pct.astype(np.float64, copy=False)
                elif isinstance(recovery_time_100pct, (list, tuple)):
                    arr = np.asarray(recovery_time_100pct, dtype=np.float64)
                else:
                    arr = np.asarray(list(recovery_time_100pct), dtype=np.float64)
            except Exception:
                # Fallback: robust coercion using pandas (vectorised), invalid -> NaN
                try:
                    coerced = pd.to_numeric(recovery_time_100pct, errors="coerce")
                    arr = np.asarray(coerced, dtype=np.float64)
                except Exception:
                    rootLogger.error(
                        "Could not coerce recovery_time_100pct to numeric; defaulting to zeros"
                    )
                    arr = np.zeros(len(hazard_event_list), dtype=np.float64)

            # Fast invalid detection and replacement
            try:
                # Ensure array is float64 before isfinite check
                arr = np.asarray(arr, dtype=np.float64)
                invalid_mask = ~np.isfinite(arr)
                invalid_count = int(invalid_mask.sum())
                if invalid_count:
                    rootLogger.warning(
                        ("Recovered %d invalid recovery time entries; replaced with 0.0")
                        % invalid_count
                    )
                    arr = arr.copy()
                    arr[invalid_mask] = 0.0
            except Exception as e:
                rootLogger.warning(f"Error in isfinite check: {e}; using zeros for recovery times")
                arr = np.zeros(len(hazard_event_list), dtype=np.float64)

            # Length mismatch is an error: flag and set sentinel values
            if arr.size != len(hazard_event_list):
                rootLogger.error(
                    (
                        "Recovery time length mismatch. Expected: %d, Got: %d. "
                        "Marking all recovery times as error (-99)."
                    )
                    % (len(hazard_event_list), arr.size)
                )
                arr = np.full(len(hazard_event_list), -99.0, dtype=np.float64)

            # Count strictly positive values (vectorised)
            non_zero_count = int(np.count_nonzero(arr > 0.0))
            rootLogger.info(
                "Recovery calculation complete: "
                f"{non_zero_count:,}/{arr.size:,} events have non-zero recovery times."
            )

            recovery_time_100pct = arr
        else:
            # If the overall result is None, add a sentinel value (-99).
            rootLogger.error(
                "Recovery analysis returned None.\nMarking all recovery times as error (-99)."
            )
            recovery_time_100pct = np.full(len(hazard_event_list), -99.0, dtype=np.float64)
    else:
        recovery_time_100pct = [0] * len(hazard_event_list)

    # =================================================================================
    # Output File -- system response summary outputs
    # ---------------------------------------------------------------------------------

    hazard_col = hazards.HAZARD_INPUT_HEADER
    rootLogger.info("Collating data on system loss and output ...")

    out_cols = [
        "event_id",  # [0] Formerly 'INTENSITY_MEASURE'
        "loss_mean",  # [1]
        "loss_std",  # [2]
        "output_mean",  # [3]
        "output_std",  # [4]
        "recovery_time_100pct",  # [5]
    ]

    sys_economic_loss_array = response_list[5]
    sys_output_array = response_list[4]

    output_array = np.divide(sys_output_array, infrastructure.system_output_capacity)
    output_array_mean = np.mean(output_array, axis=0)
    output_array_std = np.std(output_array, axis=0)

    rootLogger.info("Done.\n")

    # ---------------------------------------------------------------------------------
    # Create the complete DataFrame with ALL columns including recovery times
    # ---------------------------------------------------------------------------------

    # Create dataframe with ALL columns at once
    df_sys_response = pd.DataFrame(columns=out_cols)

    df_sys_response[out_cols[0]] = hazard_event_list
    df_sys_response[out_cols[1]] = np.mean(sys_economic_loss_array, axis=0)
    df_sys_response[out_cols[2]] = np.std(sys_economic_loss_array, axis=0)
    df_sys_response[out_cols[3]] = output_array_mean
    df_sys_response[out_cols[4]] = output_array_std
    df_sys_response[out_cols[5]] = recovery_time_100pct

    if (config.INFRASTRUCTURE_LEVEL).lower() == "facility":
        # Determine site_id selection for collocated components (facility-level)
        # - For calculated/hazard_array, the site is always "0"
        # - For scenario_file/hazard_file, select a non-negative site_id from components
        #   (negative site_id components are ignored as per hazard logic)
        try:
            if config.HAZARD_INPUT_METHOD in ["calculated_array", "hazard_array"]:
                site_id = "0"
                haz_source_ok = True
            else:
                # Build a list of valid site_id's and pick a valid (non-negative) value.
                # Collect non-negative site_ids from components
                valid_sites = []
                for comp in infrastructure.components.values():
                    try:
                        if float(comp.site_id) >= 0:
                            valid_sites.append(str(comp.site_id))
                    except Exception:
                        # Non-numeric site ids are ignored
                        continue

                if valid_sites:
                    # Prefer a stable choice (sorted first)
                    site_id = sorted(set(valid_sites))[0]
                    haz_source_ok = True
                else:
                    # No valid sites: hazard contribution is zero for all events
                    site_id = None
                    haz_source_ok = False

            if haz_source_ok and site_id in hazards.hazard_data_df.columns:
                haz_vals = hazards.hazard_data_df[site_id].values
            elif haz_source_ok:
                rootLogger.warning(
                    "Facility site_id %s not found in hazard data; defaulting to zeros",
                    site_id,
                )
                haz_vals = np.zeros(len(hazard_event_list), dtype=float)
            else:
                haz_vals = np.zeros(len(hazard_event_list), dtype=float)

            df_sys_response.insert(1, hazard_col, haz_vals)
        except Exception as e:
            # Do not fail summary writing because of hazard column insertion
            rootLogger.warning("Failed to insert facility hazard column (%s): %s", hazard_col, e)

    df_sys_response = df_sys_response.sort_values("loss_mean", ascending=True)

    try:
        outfile_sys_response = Path(config.OUTPUT_DIR, "system_response.csv")
        outpath_wrapped = utils.wrap_file_path(str(outfile_sys_response))
        rootLogger.info(
            f"Writing {Fore.CYAN}system hazard response data{Fore.RESET} to:\n"
            f"{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}"
        )
        df_sys_response.to_csv(outfile_sys_response, sep=",", index=False)
        rootLogger.info("Done.\n")
    except Exception as e:
        rootLogger.error(f"Failed to write system_response.csv: {e}")
        rootLogger.info("Critical error: system response file is required for analysis")
        raise  # Re-raise as this is a critical file

    # =================================================================================
    # Risk calculations
    # ---------------------------------------------------------------------------------
    # Calculating summary statistics using pandas
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(
        df_sys_response, calc_recovery=CALC_SYSTEM_RECOVERY
    )

    # Convert to DataFrame for better display
    summary_df = pd.DataFrame(summary_stats)
    try:
        summary_df.to_csv(Path(config.OUTPUT_DIR, "risk_summary_statistics.csv"))
        summary_df.to_json(Path(config.OUTPUT_DIR, "risk_summary_statistics.json"))
    except Exception as e:
        rootLogger.error(f"Failed to write risk summary statistics files: {e}")
        rootLogger.info("Continuing without risk summary statistics files")

    # Print summary statistics
    print(f"\n{Fore.CYAN}Summary Statistics:{Fore.RESET}")
    print(summary_df.round(4), "\n")

    # --------------------------------------------------------------------------------
    # Calculate correlations

    plot_env = os.environ.get("SIRA_ENABLE_SYSTEM_PLOTS")
    if plot_env is not None:
        should_generate_plots = plot_env.lower() in ("1", "true", "yes")
    else:
        # RUN_CONTEXT > 0 indicates risk analysis mode for large hazard portfolios
        should_generate_plots = not bool(getattr(config, "RUN_CONTEXT", 0))

    if should_generate_plots:
        print()
        rootLogger.info(
            f"\n{Fore.CYAN}Calculating correlations between loss & output...{Fore.RESET}"
        )

        total_rows = len(df_sys_response)
        desired_sample_size = min(1_000_000, total_rows)
        sample_fraction = desired_sample_size / total_rows
        sample_df = df_sys_response.sample(frac=sample_fraction)

        plot_params = [
            {
                "x": "loss_mean",
                "title": "Distribution of Loss",
                "xlabel": "Loss Ratio",
                "plot_type": "hist",
            },
            {
                "x": "output_mean",
                "title": "Distribution of Output",
                "xlabel": "Output Mean",
                "plot_type": "hist",
            },
            {
                "x": "loss_mean",
                "y": "output_mean",
                "title": "Loss Ratio vs System Output",
                "xlabel": "Loss Ratio",
                "ylabel": "Output Fraction",
                "plot_type": "scatter",
            },
        ]

        try:
            plt.style.use("seaborn-v0_8-ticks")
            for params in plot_params:
                fig, ax = plt.subplots(figsize=(10, 8))

                if params.get("plot_type") == "hist":
                    sns.histplot(data=sample_df, x=params["x"], kde=True, ax=ax)
                else:
                    sns.scatterplot(data=sample_df, x=params["x"], y=params["y"], alpha=0.5, ax=ax)

                ax.set_title(params["title"])
                ax.set_xlabel(params["xlabel"])
                if params.get("ylabel"):
                    ax.set_ylabel(params["ylabel"])

                plt.tight_layout()
                plot_name = params["title"].lower().replace(" ", "_")
                fig.savefig(
                    Path(config.OUTPUT_DIR, f"sys_{plot_name}__counts.png"),
                    dpi=300,
                    format="png",
                )
                plt.close(fig)
        except Exception as e:
            rootLogger.error(f"Failed to create count plots: {e}")
            rootLogger.info("Continuing without count plots")

        try:
            for params in plot_params:
                fig, ax = plt.subplots(figsize=(10, 8))

                if params.get("plot_type") == "hist":
                    sns.histplot(data=sample_df, x=params["x"], kde=True, ax=ax, stat="probability")
                    ax.set_ylim(0, 1)
                    title = f"{params['title']} (Normalised Frequency)"
                else:
                    sns.scatterplot(data=sample_df, x=params["x"], y=params["y"], alpha=0.5, ax=ax)
                    title = f"{params['title']} (Sampled Data)"

                ax.set_title(title)
                ax.set_xlabel(params["xlabel"])
                if params.get("ylabel"):
                    ax.set_ylabel(params["ylabel"])

                plt.tight_layout()
                plot_name = params["title"].lower().replace(" ", "_")
                fig.savefig(
                    Path(config.OUTPUT_DIR, f"sys_{plot_name}__normalised.png"),
                    dpi=300,
                    format="png",
                )
                plt.close(fig)
        except Exception as e:
            rootLogger.error(f"Failed to create normalised plots: {e}")
            rootLogger.info("Continuing without normalised plots")
    else:
        rootLogger.info(
            "Skipping system loss/output plots (risk analysis mode). "
            "Set SIRA_ENABLE_SYSTEM_PLOTS=1 to override."
        )

    # =================================================================================
    # Calculate system fragility & exceedance probabilities
    # ---------------------------------------------------------------------------------

    sys_economic_loss_array = response_list[5]
    sys_ds_bounds = np.array(infrastructure.get_system_damage_state_bounds())

    # Vectorised fragility calculation
    comparisons = sys_economic_loss_array[:, :, np.newaxis] >= sys_ds_bounds
    sys_fragility = np.sum(comparisons, axis=2)

    # Adjust highest state in one operation
    sys_fragility[sys_economic_loss_array >= sys_ds_bounds[-1]] = len(sys_ds_bounds)

    # Prob of exceedance
    num_ds = len(infrastructure.get_system_damage_states())
    pe_sys_econloss = np.array(
        [np.mean(sys_fragility >= ds, axis=0) for ds in range(num_ds)], dtype=np.float32
    )

    if config.SWITCH_SAVE_VARS_NPY:
        try:
            np.save(Path(config.RAW_OUTPUT_DIR, "sys_output_array.npy"), sys_output_array)

            np.save(
                Path(config.RAW_OUTPUT_DIR, "economic_loss_array.npy"),
                sys_economic_loss_array,
            )
        except Exception as e:
            rootLogger.error(f"Failed to save numpy arrays: {e}")
            rootLogger.info("Continuing without saving numpy arrays")

    if not str(config.INFRASTRUCTURE_LEVEL).lower() == "network":
        try:
            path_pe_sys_econloss = Path(config.RAW_OUTPUT_DIR, "pe_sys_econloss.npy")
            outpath_wrapped = utils.wrap_file_path(str(path_pe_sys_econloss), max_width=120)
            print()
            rootLogger.info(
                f"Writing prob of exceedance data to: \n{Fore.YELLOW}{outpath_wrapped}{Fore.RESET}"
            )
            np.save(path_pe_sys_econloss, pe_sys_econloss)
            rootLogger.info("Done.\n")
        except Exception as e:
            rootLogger.error(f"Failed to save probability of exceedance data: {e}")
            rootLogger.info("Continuing without probability of exceedance file")

    # return pe_sys_econloss
    return


# -------------------------------------------------------------------------------------


@njit
def _pe2pb(pe):
    """Numba-optimised version of pe2pb"""
    pex = np.sort(pe)[::-1]
    tmp = -1.0 * np.diff(pex)
    pb = np.zeros(len(pe) + 1)
    pb[1:-1] = tmp
    pb[-1] = pex[-1]
    pb[0] = 1 - pex[0]
    return pb


# -------------------------------------------------------------------------------------


def exceedance_prob_by_component_class(response_list, infrastructure, scenario, hazards):
    """
    Calculates probability of exceedance based on failure of component classes.
    Damage state boundaries for Component Type Failures (Substations) are
    based on HAZUS MH MR3, p 8-66 to 8-68.

    Parameters:
    -----------
    response_list : list
    infrastructure : Infastructure object
    scenario : Scenario object
    hazards : HazardContainer object

    Returns:
    --------
    pe_sys_cpfailrate : numpy array
        array with exceedance probabilities of for component failures
    """
    if not str(infrastructure.system_class).lower() == "substation":
        return None

    # Pre-calculate all indices and mappings once
    component_keys = list(infrastructure.components.keys())
    cp_classes_in_system = np.unique(list(infrastructure.get_component_class_list()))

    # Create mapping of class -> array of component indices
    cp_class_indices = {
        k: np.array(
            [
                component_keys.index(comp_id)
                for comp_id, comp in infrastructure.components.items()
                if comp.component_class == k
            ]
        )
        for k in cp_classes_in_system
    }

    cp_classes_costed = [
        x for x in cp_classes_in_system if x not in infrastructure.uncosted_classes
    ]

    # Convert response data to single numpy array upfront
    num_samples = scenario.num_samples
    num_events = len(hazards.hazard_data_df)
    num_components = len(infrastructure.components)

    # Normalise event id key shape to match response_list[0] keys
    event_response_dict = response_list[0]

    # If we don't have per-event component state data (e.g., MPI compact path), skip gracefully
    if not event_response_dict:
        rootLogger.info(
            "Skipping component-class exceedance calculation: missing per-event component"
            " state data (MPI compact path)."
        )
        return None

    # Pre-allocate the full response array
    response_array = np.zeros((num_samples, num_events, num_components))

    def _coerce_event_key(raw_key):
        """
        Coerce a hazard index value into the key format used by event_response_dict.
        Tries int->str fallbacks to handle mixed key types robustly.
        """
        # Unwrap tuple/list index commonly seen in MultiIndex
        key = raw_key[0] if isinstance(raw_key, (tuple, list)) else raw_key

        # Fast-path: numpy scalars to native int
        try:
            import numpy as _np  # local import to avoid polluting namespace

            if isinstance(key, (_np.integer,)):
                key = int(key)
        except Exception:
            # numpy may not be available here, ignore
            pass

        # If string looks like an int, coerce to int
        if isinstance(key, str):
            s = key.strip()
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                try:
                    key = int(s)
                except Exception:
                    # keep as original string if int conversion fails unexpectedly
                    key = key

        # Resolve against available dict keys with graceful fallbacks
        if key in event_response_dict:
            return key

        # Try alternate representation if possible
        if isinstance(key, int):
            alt = str(key)
            if alt in event_response_dict:
                return alt
        elif isinstance(key, str):
            try:
                alt_int = int(key)
                if alt_int in event_response_dict:
                    return alt_int
            except Exception:
                pass

        # As a last resort, return the original key (will raise later with clearer context)
        return key

    # Fill response array (do this once instead of repeatedly accessing dict)
    for j, scenario_index in enumerate(hazards.hazard_data_df.index):
        # print(f"Processing hazard event {j + 1}/{num_events} (Event ID: {scenario_index})")
        ev_key = _coerce_event_key(scenario_index)
        try:
            response_array[:, j, :] = event_response_dict[ev_key]
        except KeyError as e:
            # Provide a more informative error including sample keys snapshot
            sample_keys = list(event_response_dict.keys())
            sample_preview = sample_keys[:5]
            raise KeyError(
                f"Event key {ev_key!r} not found in response_list[0]. "
                f"Example keys: {sample_preview} (total {len(sample_keys)}). "
                f"Original index value: {scenario_index!r}"
            ) from e

    # --- System fragility - Based on Failure of Component Classes ---
    comp_class_failures = {}
    comp_class_frag = {}

    for compclass in cp_classes_costed:
        indices = cp_class_indices[compclass]
        if len(indices) == 0:
            continue

        # Calculate failures for entire class at once
        failures = (response_array[:, :, indices] >= 2).sum(axis=2) / len(indices)
        comp_class_failures[compclass] = failures

        # Calculate fragility using vectorised operations
        ds_lims = np.array(infrastructure.get_ds_lims_for_compclass(compclass))
        comp_class_frag[compclass] = (failures[:, :, np.newaxis] > ds_lims).sum(axis=2)

    # Probability of Exceedance -- Based on Failure of Component Classes
    pe_sys_cpfailrate = np.zeros((len(infrastructure.system_dmg_states), hazards.num_hazard_pts))

    for d in range(len(infrastructure.system_dmg_states)):
        exceedance_probs = []
        for compclass in cp_classes_costed:
            if compclass in comp_class_frag:
                class_exceed = (comp_class_frag[compclass] >= d).mean(axis=0)
                exceedance_probs.append(class_exceed)

        if exceedance_probs:
            pe_sys_cpfailrate[d, :] = np.median(exceedance_probs, axis=0)

    # Vectorised damage ratio calculations
    exp_damage_ratio = np.zeros((len(infrastructure.components), hazards.num_hazard_pts))
    hazard_data = hazards.hazard_data_df.values

    # Process in large batches for better vectorisation
    for comp_class in cp_classes_costed:
        indices = cp_class_indices[comp_class]
        if len(indices) == 0:
            continue

        batch_components = [infrastructure.components[component_keys[i]] for i in indices]

        for i, component in enumerate(batch_components):
            comp_idx = indices[i]
            try:
                loc_params = component.get_location()
                site_id = str(loc_params[0]) if isinstance(loc_params, tuple) else "0"

                if site_id in hazards.hazard_data_df.columns:
                    site_col_idx = hazards.hazard_data_df.columns.get_loc(site_id)
                    hazard_intensities = hazard_data[:, site_col_idx]

                    # Vectorised response calculation
                    pe_ds = np.zeros((hazards.num_hazard_pts, len(component.damage_states)))
                    valid_mask = ~np.isnan(hazard_intensities)

                    if np.any(valid_mask):
                        for ds_idx in component.damage_states.keys():
                            pe_ds[valid_mask, ds_idx] = component.damage_states[
                                ds_idx
                            ].response_function(hazard_intensities[valid_mask])

                        # Vectorised damage ratio calculation
                        pe_ds = pe_ds[:, 1:]  # Remove first state
                        pb = _pe2pb(pe_ds[0])  # Calculate probability bins
                        dr = np.array(
                            [
                                component.damage_states[int(ds)].damage_ratio
                                for ds in range(len(component.damage_states))
                            ]
                        )
                        exp_damage_ratio[comp_idx, valid_mask] = np.sum(
                            pb * dr * component.cost_fraction
                        )

            except Exception as e:
                rootLogger.warning(f"Error calculating damage ratio for component {comp_idx}: {e}")
                continue

    # Save results
    if scenario.save_vars_npy:
        try:
            np.save(Path(scenario.raw_output_dir, "exp_damage_ratio.npy"), exp_damage_ratio)
        except Exception as e:
            rootLogger.error(f"Failed to save exp_damage_ratio.npy: {e}")

    if scenario.hazard_input_method.lower() in ["calculated_array"]:
        try:
            path_sys_cpfailrate = Path(scenario.raw_output_dir, "pe_sys_cpfailrate.npy")
            np.save(path_sys_cpfailrate, pe_sys_cpfailrate)
        except Exception as e:
            rootLogger.error(f"Failed to save pe_sys_cpfailrate.npy: {e}")

    return pe_sys_cpfailrate
