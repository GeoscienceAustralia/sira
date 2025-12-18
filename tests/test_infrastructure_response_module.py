"""
Comprehensive tests for infrastructure_response module.

Covers core functions, utility functions, statistics calculations, plotting, and edge cases.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sira.infrastructure_response import (
    _pe2pb,
    calc_tick_vals,
    calculate_loss_stats,
    calculate_output_stats,
    calculate_recovery_stats,
    calculate_summary_statistics,
    plot_mean_econ_loss,
)


# Test fixtures and helper classes
class SimpleComponent:
    def __init__(self):
        self.cost = 100
        self.time_to_repair = 5
        self.recovery_function = lambda t: min(1.0, t / self.time_to_repair)


class SimpleInfrastructure:
    def __init__(self):
        self.components = {"comp1": SimpleComponent()}
        self.system_output_capacity = 100


class SimpleScenario:
    def __init__(self):
        self.output_path = "test_path"
        self.num_samples = 10


class SimpleHazard:
    def __init__(self):
        self.hazard_scenario_list = ["event1"]


@pytest.fixture
def test_infrastructure():
    return SimpleInfrastructure()


@pytest.fixture
def test_scenario():
    return SimpleScenario()


@pytest.fixture
def test_hazard():
    return SimpleHazard()


@pytest.fixture
def test_output_dir():
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup
    for f in test_dir.glob("*"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    test_dir.rmdir()


@pytest.fixture
def mock_component():
    """Create a mock component with required attributes."""
    comp = Mock()
    comp.component_id = "comp_001"
    comp.component_type = "test_type"
    comp.component_class = "test_class"
    comp.cost_fraction = 0.1
    comp.damage_states = {
        0: Mock(damage_ratio=0.0),
        1: Mock(damage_ratio=0.3),
        2: Mock(damage_ratio=0.7),
        3: Mock(damage_ratio=1.0),
    }
    return comp


@pytest.fixture
def mock_infrastructure(mock_component):
    """Create a mock infrastructure object."""
    infra = Mock()
    infra.components = {"comp_001": mock_component, "comp_002": mock_component}
    infra.output_nodes = {"node_1": {"output_node_capacity": 100.0}}
    infra.system_output_capacity = 100.0
    infra.uncosted_classes = []

    def get_component_types():
        return ["test_type"]

    def get_components_for_type(comp_type):
        return list(infra.components.values())

    infra.get_component_types = get_component_types
    infra.get_components_for_type = get_components_for_type

    return infra


@pytest.fixture
def mock_scenario():
    """Create a mock scenario object."""
    scenario = Mock()
    scenario.num_samples = 10
    scenario.output_path = "."
    scenario.recovery_max_workers = 2
    scenario.recovery_batch_size = 100
    scenario.parallel_config = None
    return scenario


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration object."""
    config = Mock()
    config.OUTPUT_DIR = str(tmp_path)
    config.INFRASTRUCTURE_LEVEL = "network"
    config.HAZARD_INPUT_METHOD = "hazard_file"
    return config


@pytest.fixture
def mock_hazards():
    """Create a mock hazards container."""
    hazards = Mock()
    hazards.HAZARD_INPUT_HEADER = "PGA"
    hazards.hazard_scenario_list = ["event_001", "event_002", "event_003"]
    hazards.hazard_data_df = pd.DataFrame({"0": [0.1, 0.5, 0.9]})
    return hazards


@pytest.fixture
def mock_df():
    return pd.DataFrame(
        {
            "loss_mean": [0.1, 0.2, 0.3],
            "output_mean": [0.5, 0.6, 0.7],
            "recovery_time_100pct": [10, 20, 30],
        }
    )


# -------------------------------------------------------------------------------------
# Tests for _pe2pb
# -------------------------------------------------------------------------------------


def test_pe2pb_basic():
    """Test basic functionality with simple input"""
    pe = np.array([0.9, 0.6, 0.3])
    result = _pe2pb(pe)
    expected = np.array([0.1, 0.3, 0.3, 0.3])
    np.testing.assert_array_almost_equal(result, expected)


def test_pe2pb_single_value():
    """Test with a single value"""
    pe = np.array([0.5])
    result = _pe2pb(pe)
    expected = np.array([0.5, 0.5])
    np.testing.assert_array_almost_equal(result, expected)


def test_pe2pb_identical_values():
    """Test with array containing identical values"""
    pe = np.array([0.3, 0.3, 0.3])
    result = _pe2pb(pe)
    expected = np.array([0.7, 0.0, 0.0, 0.3])
    np.testing.assert_array_almost_equal(result, expected)


def test_pe2pb_zero():
    """Test with zero probability"""
    pe = np.array([0.0, 0.0])
    result = _pe2pb(pe)
    expected = np.array([1.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_pe2pb_one():
    """Test with probability of 1"""
    pe = np.array([1.0, 0.5])
    result = _pe2pb(pe)
    expected = np.array([0.0, 0.5, 0.5])
    np.testing.assert_array_almost_equal(result, expected)


def test_pe2pb_properties():
    """Test mathematical properties that should hold for any valid input"""
    pe = np.array([0.8, 0.5, 0.2])
    result = _pe2pb(pe)

    # Sum of probabilities should be 1
    assert np.isclose(np.sum(result), 1.0)

    # Length should be input length + 1
    assert len(result) == len(pe) + 1

    # All probabilities should be non-negative
    assert np.all(result >= 0)

    # All probabilities should be <= 1
    assert np.all(result <= 1)


def test_pe2pb_different_dtypes():
    """Test with different input data types"""
    inputs = [
        np.array([0.9, 0.6, 0.3], dtype=np.float32),
        np.array([0.9, 0.6, 0.3], dtype=np.float64),
        np.array([0.9, 0.6, 0.3]),  # list converted to array
        np.array([0.9, 0.6, 0.3]),  # tuple converted to array
    ]
    expected = np.array([0.1, 0.3, 0.3, 0.3])
    for inp in inputs:
        result = _pe2pb(inp)
        np.testing.assert_array_almost_equal(result, expected)


# -------------------------------------------------------------------------------------
# Tests for calc_tick_vals
# -------------------------------------------------------------------------------------


def test_calc_tick_vals_normal_case():
    """Test normal case with small list."""
    val_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = calc_tick_vals(val_list)
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)


def test_calc_tick_vals_long_list():
    """Test long list case (> 20 ticks)."""
    long_list = list(range(30))
    result_long = calc_tick_vals(long_list)
    assert len(result_long) <= 11


def test_calc_tick_vals_small_list():
    """Test with small list (< 12 ticks)."""
    val_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = calc_tick_vals(val_list, xstep=0.1)
    assert len(result) > 0
    assert isinstance(result[0], str)


def test_calc_tick_vals_medium_list():
    """Test with medium list (12-20 ticks)."""
    val_list = [float(i) * 0.1 for i in range(15)]
    result = calc_tick_vals(val_list, xstep=0.1)
    assert len(result) > 0


def test_calc_tick_vals_large_list():
    """Test with large list (> 20 ticks)."""
    val_list = [float(i) * 0.1 for i in range(30)]
    result = calc_tick_vals(val_list, xstep=0.1)
    # Large list should be reduced to 11 ticks
    assert len(result) <= 11


def test_calc_tick_vals_string_values():
    """Test with string values."""
    val_list = ["a", "b", "c", "d", "e"]
    result = calc_tick_vals(val_list, xstep=0.2)
    assert len(result) > 0


# -------------------------------------------------------------------------------------
# Tests for plot_mean_econ_loss
# -------------------------------------------------------------------------------------


@patch("matplotlib.pyplot.savefig")
def test_plot_mean_econ_loss(mock_savefig, test_output_dir):
    hazard_data = np.array([0.1, 0.2, 0.3])
    loss_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    plot_mean_econ_loss(hazard_data, loss_data, output_path=test_output_dir)
    mock_savefig.assert_called_once()


def test_plot_mean_econ_loss_basic(tmp_path):
    """Test basic plotting functionality."""
    hazard_intensity = [0.1, 0.5, 0.9]
    loss_array = np.random.rand(10, 3)

    with patch("matplotlib.pyplot.savefig") as mock_save:
        plot_mean_econ_loss(hazard_intensity, loss_array, output_path=tmp_path)
        mock_save.assert_called_once()


def test_plot_mean_econ_loss_different_bin_widths(tmp_path):
    """Test different hazard intensity ranges (different bin widths)."""
    test_cases = [
        ([0.01, 0.02, 0.03], "tiny range"),
        ([0.1, 0.3, 0.5], "small range"),
        ([1.0, 2.0, 3.0], "medium range"),
        ([5.0, 10.0, 15.0], "large range"),
    ]

    for hazard_vals, desc in test_cases:
        loss_array = np.random.rand(5, len(hazard_vals))
        with patch("matplotlib.pyplot.savefig"):
            plot_mean_econ_loss(
                hazard_vals,
                loss_array,
                output_path=tmp_path,
                fig_name=f"test_{desc.replace(' ', '_')}",
            )


# -------------------------------------------------------------------------------------
# Statistics calculation tests
# -------------------------------------------------------------------------------------


def test_calculate_loss_stats(mock_df):
    stats = calculate_loss_stats(mock_df, progress_bar=False)
    assert isinstance(stats, dict)
    assert all(k in stats for k in ["Mean", "Std", "Min", "Max", "Median"])
    assert abs(stats["Mean"] - 0.2) < 0.001


def test_calculate_loss_stats_basic():
    """Test loss statistics calculation."""
    # These functions expect dataframes with pre-calculated columns
    df = pd.DataFrame(
        {
            "loss_mean": [0.1, 0.2, 0.3],
            "loss_std": [0.01, 0.02, 0.03],
            "loss_tot": [10.0, 20.0, 30.0],
        }
    )

    result = calculate_loss_stats(df, progress_bar=False)

    assert isinstance(result, dict)
    assert "Mean" in result
    assert "Std" in result
    assert result["Mean"] == pytest.approx(0.2, rel=0.01)


def test_calculate_loss_stats_with_progress():
    """Test with progress bar enabled."""
    df = pd.DataFrame(
        {
            "loss_mean": [0.1, 0.2, 0.3],
            "loss_std": [0.01, 0.02, 0.03],
        }
    )

    # Should work without errors even with progress_bar=True
    result = calculate_loss_stats(df, progress_bar=True)
    assert isinstance(result, dict)


def test_calculate_output_stats(mock_df):
    stats = calculate_output_stats(mock_df, progress_bar=False)
    assert isinstance(stats, dict)
    assert abs(stats["Mean"] - 0.6) < 0.001


def test_calculate_output_stats_basic():
    """Test output statistics calculation."""
    df = pd.DataFrame(
        {
            "output_mean": [80.0, 85.0, 90.0],
            "output_std": [5.0, 6.0, 7.0],
        }
    )

    result = calculate_output_stats(df, progress_bar=False)

    assert isinstance(result, dict)
    assert "Mean" in result
    assert "Std" in result


def test_calculate_recovery_stats(mock_df):
    stats = calculate_recovery_stats(mock_df, progress_bar=False)
    assert isinstance(stats, dict)
    assert abs(stats["Mean"] - 20) < 0.001


def test_calculate_recovery_stats_basic():
    """Test recovery statistics calculation."""
    df = pd.DataFrame(
        {
            "recovery_time_100pct": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    result = calculate_recovery_stats(df, progress_bar=False)

    assert isinstance(result, dict)
    assert "Mean" in result
    assert "Std" in result


def test_calculate_summary_statistics(mock_df):
    summary = calculate_summary_statistics(mock_df, calc_recovery=True)
    assert isinstance(summary, dict)
    assert all(k in summary for k in ["Loss", "Output", "Recovery Time"])


def test_calculate_summary_statistics_without_recovery():
    """Test summary statistics without recovery."""
    df = pd.DataFrame(
        {
            "loss_mean": [0.1, 0.2, 0.3],
            "output_mean": [80.0, 85.0, 90.0],
        }
    )

    result = calculate_summary_statistics(df, calc_recovery=False)

    assert isinstance(result, dict)
    assert "Loss" in result
    assert "Output" in result
    assert "Recovery Time" not in result


def test_calculate_summary_statistics_with_recovery():
    """Test summary statistics with recovery."""
    df = pd.DataFrame(
        {
            "loss_mean": [0.1, 0.2, 0.3],
            "output_mean": [80.0, 85.0, 90.0],
            "recovery_time_100pct": [5.0, 10.0, 15.0],
        }
    )

    result = calculate_summary_statistics(df, calc_recovery=True)

    assert isinstance(result, dict)
    # Should include recovery statistics
    assert "Loss" in result
    assert "Output" in result
    assert "Recovery Time" in result


# -------------------------------------------------------------------------------------
# Edge cases and error handling
# -------------------------------------------------------------------------------------


def test_calculate_stats_empty_dataframe():
    """Test statistics calculation with empty dataframe."""
    # Create dataframe with required columns but no data
    df = pd.DataFrame({"loss_mean": [], "loss_std": []})

    # Should handle empty dataframe gracefully
    result = calculate_loss_stats(df, progress_bar=False)
    assert isinstance(result, dict)


def test_calculate_stats_single_event():
    """Test with single event."""
    df = pd.DataFrame({"loss_mean": [0.2], "loss_std": [0.01]})

    result = calculate_loss_stats(df, progress_bar=False)
    assert isinstance(result, dict)
    assert result["Mean"] == pytest.approx(0.2, rel=0.01)


def test_calculate_stats_nan_values():
    """Test handling of NaN values."""
    df = pd.DataFrame(
        {
            "loss_mean": [0.1, np.nan, 0.3],
            "loss_std": [0.01, 0.02, 0.03],
        }
    )

    # Should handle NaN values appropriately
    result = calculate_loss_stats(df, progress_bar=False)
    assert isinstance(result, dict)


# -------------------------------------------------------------------------------------
# Integration tests
# -------------------------------------------------------------------------------------


@pytest.mark.integration
def test_stats_calculation_flow(mock_df):
    loss_stats = calculate_loss_stats(mock_df, progress_bar=False)
    output_stats = calculate_output_stats(mock_df, progress_bar=False)
    recovery_stats = calculate_recovery_stats(mock_df, progress_bar=False)

    assert isinstance(loss_stats, dict)
    assert isinstance(output_stats, dict)
    assert isinstance(recovery_stats, dict)

    summary_stats = calculate_summary_statistics(mock_df, calc_recovery=True)
    assert isinstance(summary_stats, dict)
    assert len(summary_stats) == 3


# -------------------------------------------------------------------------------------
# Complex integration tests (skipped)
# -------------------------------------------------------------------------------------


@pytest.mark.skip(reason="Complex integration test - requires full infrastructure setup")
def test_write_system_response_basic(tmp_path, mock_infrastructure, mock_scenario, mock_hazards):
    """Test basic write_system_response functionality."""
    pass


@pytest.mark.skip(reason="Complex integration test - requires full infrastructure setup")
def test_write_system_response_with_recovery(
    tmp_path, mock_infrastructure, mock_scenario, mock_hazards
):
    """Test write_system_response with recovery calculation."""
    pass


@pytest.mark.skip(reason="Complex integration test - requires full infrastructure setup")
def test_reconstruct_response_list_from_streaming(
    tmp_path, mock_infrastructure, mock_scenario, mock_hazards
):
    """Test reconstruction of response list from streaming data."""
    pass


@pytest.mark.skip(reason="Requires correct manifest format")
def test_reconstruct_response_list_missing_manifest(
    tmp_path, mock_infrastructure, mock_scenario, mock_hazards
):
    """Test reconstruction when manifest is missing."""
    pass


@pytest.mark.skip(reason="Requires correct manifest format")
def test_reconstruct_response_list_corrupted_data(
    tmp_path, mock_infrastructure, mock_scenario, mock_hazards
):
    """Test reconstruction with corrupted data files."""
    pass


@pytest.mark.skip(reason="Complex integration test - covered by test_streaming_consolidation.py")
def test_consolidate_streamed_results_basic(
    tmp_path, mock_infrastructure, mock_scenario, mock_config, mock_hazards
):
    """Test basic consolidation of streamed results."""
    pass


@pytest.mark.skip(reason="Complex integration test - covered by test_streaming_consolidation.py")
def test_consolidate_streamed_results_with_recovery(
    tmp_path, mock_infrastructure, mock_scenario, mock_config, mock_hazards
):
    """Test consolidation with recovery calculation."""
    pass


@pytest.mark.skip(reason="Complex integration test - covered by test_streaming_consolidation.py")
def test_consolidate_streamed_results_empty_directory(
    tmp_path, mock_infrastructure, mock_scenario, mock_config, mock_hazards
):
    """Test consolidation with empty stream directory."""
    pass


@pytest.mark.skip(reason="Complex integration test - requires full component graph")
def test_exceedance_prob_by_component_class_basic(mock_infrastructure, mock_scenario, mock_hazards):
    """Test exceedance probability calculation."""
    pass


@pytest.mark.skip(reason="Complex integration test")
def test_exceedance_prob_by_component_class_empty_response(
    mock_infrastructure, mock_scenario, mock_hazards
):
    """Test with empty response list."""
    pass


if __name__ == "__main__":
    pytest.main(["-v"])
