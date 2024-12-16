"""

"""

import pytest
import numpy as np
import numba as nb
import pandas as pd
from pathlib import Path
import os
from unittest.mock import patch
import matplotlib.pyplot as plt
import dask.dataframe as dd  # type: ignore

from sira.infrastructure_response import (
    calc_tick_vals,
    plot_mean_econ_loss,
    calculate_loss_stats,
    calculate_output_stats,
    calculate_recovery_stats,
    calculate_summary_statistics,
    _pe2pb,
    parallel_recovery_analysis
)


# Test fixtures and helper classes
class SimpleComponent:
    def __init__(self):
        self.cost = 100
        self.time_to_repair = 5
        self.recovery_function = lambda t: min(1.0, t / self.time_to_repair)

class SimpleInfrastructure:
    def __init__(self):
        self.components = {'comp1': SimpleComponent()}
        self.system_output_capacity = 100

class SimpleScenario:
    def __init__(self):
        self.output_path = "test_path"
        self.num_samples = 10

class SimpleHazard:
    def __init__(self):
        self.hazard_scenario_list = ['event1']

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
    for f in test_dir.glob('*'):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    test_dir.rmdir()

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
        [0.9, 0.6, 0.3],  # list
        (0.9, 0.6, 0.3),  # tuple
    ]
    expected = np.array([0.1, 0.3, 0.3, 0.3])
    for inp in inputs:
        result = _pe2pb(inp)
        np.testing.assert_array_almost_equal(result, expected)

# -------------------------------------------------------------------------------------

def test_calc_tick_vals():
    # Test normal case
    val_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = calc_tick_vals(val_list)
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)

    # Test long list case
    long_list = list(range(30))
    result_long = calc_tick_vals(long_list)
    assert len(result_long) <= 11

@patch('matplotlib.pyplot.savefig')
def test_plot_mean_econ_loss(mock_savefig, test_output_dir):
    hazard_data = np.array([0.1, 0.2, 0.3])
    loss_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    plot_mean_econ_loss(
        hazard_data,
        loss_data,
        output_path=test_output_dir
    )
    mock_savefig.assert_called_once()

# -------------------------------------------------------------------------------------

# Statistics calculation tests
@pytest.fixture
def mock_dask_df():
    df = pd.DataFrame({
        'loss_mean': [0.1, 0.2, 0.3],
        'output_mean': [0.5, 0.6, 0.7],
        'recovery_time_100pct': [10, 20, 30]
    })
    return dd.from_pandas(df, npartitions=1)

def test_calculate_loss_stats(mock_dask_df):
    stats = calculate_loss_stats(mock_dask_df, progress_bar=False)
    assert isinstance(stats, dict)
    assert all(k in stats for k in ['Mean', 'Std', 'Min', 'Max', 'Median'])
    assert abs(stats['Mean'] - 0.2) < 0.001

def test_calculate_output_stats(mock_dask_df):
    stats = calculate_output_stats(mock_dask_df, progress_bar=False)
    assert isinstance(stats, dict)
    assert abs(stats['Mean'] - 0.6) < 0.001

def test_calculate_recovery_stats(mock_dask_df):
    stats = calculate_recovery_stats(mock_dask_df, progress_bar=False)
    assert isinstance(stats, dict)
    assert abs(stats['Mean'] - 20) < 0.001

def test_calculate_summary_statistics(mock_dask_df):
    summary = calculate_summary_statistics(mock_dask_df, calc_recovery=True)
    assert isinstance(summary, dict)
    assert all(k in summary for k in ['Loss', 'Output', 'Recovery Time'])

# -------------------------------------------------------------------------------------

# class RecoveryTestComponent:
#     def __init__(self):
#         self.cost = 100
#         self.time_to_repair = 5

#     def recovery_function(self, t):
#         return min(1.0, t / self.time_to_repair)

#     def get_location(self):
#         # Returns: self.pos_x, self.pos_y, self.site_id
#         # Return a mock location for testing
#         return 0, 0, 99  # Example: (latitude, longitude)

# class RecoveryTestInfrastructure:
#     def __init__(self):
#         self.components = {'comp1': RecoveryTestComponent()}
#         self.system_output_capacity = 100
#         self.uncosted_classes = []

# class RecoveryTestScenario:
#     def __init__(self):
#         self.output_path = "test_path"
#         self.num_samples = 2

# class RecoveryTestHazard:
#     def __init__(self):
#         self.hazard_scenario_list = ['hazard_1', 'hazard_2']
#         # Make listOfhazards a list instead of dict to match expected structure
#         self.listOfhazards = [
#             MockHazardEvent('hazard_1', 0.5),
#             MockHazardEvent('hazard_2', 0.8)
#         ]

# class MockHazardEvent:
#     def __init__(self, hazard_id, intensity):
#         self.hazard_event_id = hazard_id
#         self.intensity = intensity

#     def get_hazard_intensity(self, *args):
#         return self.intensity

# def test_parallel_recovery_analysis():
#     """Test parallel_recovery_analysis with minimal setup"""
#     hazard_event_list = ['hazard_1', 'hazard_2']

#     # Using multi-index DataFrame to match expected structure
#     index = pd.MultiIndex.from_product(
#         [['comp1'], ['func_mean', 'damage_mean']],
#         names=['component_id', 'response']
#     )
#     component_resp_df = pd.DataFrame(
#         [[1.0, 0.0], [0.5, 0.5]],
#         columns=hazard_event_list,
#         index=index
#     )

#     result = parallel_recovery_analysis(
#         hazard_event_list,
#         RecoveryTestInfrastructure(),
#         RecoveryTestScenario(),
#         RecoveryTestHazard(),
#         component_resp_df,
#         ['comp1'],
#         [],
#         chunk_size=1
#     )

#     assert isinstance(result, list)
#     assert len(result) == 2
#     assert all(isinstance(x, (int, float)) for x in result)

# -------------------------------------------------------------------------------------
# Integration tests

@pytest.mark.integration
def test_stats_calculation_flow(mock_dask_df):
    loss_stats = calculate_loss_stats(mock_dask_df, progress_bar=False)
    output_stats = calculate_output_stats(mock_dask_df, progress_bar=False)
    recovery_stats = calculate_recovery_stats(mock_dask_df, progress_bar=False)

    assert isinstance(loss_stats, dict)
    assert isinstance(output_stats, dict)
    assert isinstance(recovery_stats, dict)

    summary_stats = calculate_summary_statistics(mock_dask_df, calc_recovery=True)
    assert isinstance(summary_stats, dict)
    assert len(summary_stats) == 3

if __name__ == '__main__':
    pytest.main(['-v'])
