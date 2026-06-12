"""
Demonstration / characterisation of a caveat in the substation system-fragility
aggregation (``exceedance_prob_by_component_class``, infrastructure_response.py).

A substation does NOT assign a system damage state per realisation. Instead it
computes, per component *class*, an exceedance probability from that class's
failure rate (``#components at DS>=2 / #components in class``), then takes the
**median across classes** as the system curve (``pe_sys_cpfailrate``). This is the
curve a substation actually fits and reports.

The caveat: the median across classes can be outvoted. If a single critical class
(here one Power Transformer, in series) drives the whole substation down, but it is
outnumbered by robust classes (disconnect switches, circuit breakers that rarely
fail), the median ignores the failing class — so the reported substation fragility
says "undamaged" while the substation is physically dead.

Model ``substation__median_caveat`` (series chain):
    SUPPLY -> SW_1 -> SW_2 -> SW_3 -> TX -> CB_1 -> CB_2 -> CB_3 -> OUT
  * TX (Power Transformer): fragile, fails early, and is the series bottleneck.
  * SW_* (Disconnect Switch) and CB_* (Circuit Breaker): robust, ~never fail.
  Three component classes total, so the single failing class is the minority.

This test contrasts three views of the same run at a hazard level where the
transformer is destroyed:
  * actual system output      -> ~0  (substation is down)
  * econloss-based fragility   -> ~1  (correctly flags near-certain damage)
  * cpfailrate (median) curve  -> ~0  (masks it — the caveat)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sira.configuration import Configuration
from sira.infrastructure_response import (
    exceedance_prob_by_component_class,
    write_system_response,
)
from sira.model_ingest import ingest_model
from sira.modelling.hazard import HazardsContainer
from sira.scenario import Scenario
from sira.simulation import calculate_response

MODEL_NAME = "substation__median_caveat"
N_SAMPLES = 1000


@pytest.fixture(scope="module")
def caveat_run(dir_setup, tmp_path_factory):
    _, mdls_dir = dir_setup
    inp = Path(mdls_dir, MODEL_NAME, "input")
    model_path = inp / "model_substation_caveat.json"

    out_dir = tmp_path_factory.mktemp("ss_caveat_out")
    config = Configuration(
        str(inp / "config_substation_caveat.json"), str(model_path), output_path=str(out_dir)
    )
    os.environ["SIRA_QUIET_MODE"] = "1"
    scenario = Scenario(config)
    scenario.run_parallel_proc = False
    scenario.num_samples = N_SAMPLES
    infrastructure = ingest_model(config)
    hazards = HazardsContainer(config, str(model_path))

    response_list = calculate_response(hazards, scenario, infrastructure)
    write_system_response(response_list, infrastructure, scenario, config, hazards,
                          CALC_SYSTEM_RECOVERY=False)
    pe_cpfailrate = exceedance_prob_by_component_class(response_list, infrastructure, scenario, hazards)
    pe_econloss = np.load(Path(config.RAW_OUTPUT_DIR, "pe_sys_econloss.npy"))

    df = pd.read_csv(Path(out_dir, "system_response.csv")).sort_values("hazard_intensity")
    df = df.reset_index(drop=True)
    return {
        "pga": df["hazard_intensity"].to_numpy(),
        "output_mean": df["output_mean"].to_numpy(),
        "pe_econloss": pe_econloss,
        "pe_cpfailrate": pe_cpfailrate,
    }


def _at(pga_array, target):
    return int(np.argmin(np.abs(pga_array - target)))


def test_substation_is_physically_down_at_high_pga(caveat_run):
    """By 0.8 g the fragile series transformer is destroyed -> no system output."""
    j = _at(caveat_run["pga"], 0.8)
    assert caveat_run["output_mean"][j] < 0.02


def test_econloss_fragility_flags_the_failure(caveat_run):
    """Economic-loss fragility (per-realisation system state) correctly reports
    near-certain damage where the substation is down."""
    j = _at(caveat_run["pga"], 0.8)
    assert caveat_run["pe_econloss"][1][j] > 0.95


def test_median_across_classes_masks_the_critical_failure(caveat_run):
    """THE CAVEAT: the median-across-classes substation curve reports the
    substation as essentially undamaged at the same hazard level, because the one
    failing class (Power Transformer) is outvoted by the two robust classes."""
    j = _at(caveat_run["pga"], 0.8)
    assert caveat_run["pe_cpfailrate"][1][j] < 0.05


def test_the_two_fragility_curves_diverge_grossly(caveat_run):
    """Direct contrast: across the band where the transformer fails, the econloss
    curve rises to ~1 while the median-across-classes curve stays ~0."""
    econ = caveat_run["pe_econloss"][1]
    cp = caveat_run["pe_cpfailrate"][1]
    j_hi = _at(caveat_run["pga"], 1.0)
    # Econloss recognises the loss; the class-median does not.
    assert econ[j_hi] - cp[j_hi] > 0.9
