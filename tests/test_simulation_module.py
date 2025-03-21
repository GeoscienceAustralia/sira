"""
This test was generated by AI and checked by a human.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import multiprocessing as mp
from sira.simulation import (
    calculate_response_for_single_hazard,
    process_component_batch,
    calculate_response,
    calc_component_damage_state_for_n_simulations
)

# Mock classes modified to be picklable
class MockResponseFunction:
    def __init__(self, return_value):
        self.return_value = return_value

    def __call__(self, x):
        return self.return_value

class MockDamageState:
    def __init__(self, response_value):
        self.response_function = MockResponseFunction(response_value)

class MockComponent:
    def __init__(self, damage_states=None):
        if damage_states is None:
            self.damage_states = {
                0: MockDamageState(0.0),
                1: MockDamageState(0.1),
                2: MockDamageState(0.2)
            }
        else:
            self.damage_states = damage_states

    def get_location(self):
        return (0.0, 0.0)

class MockInfrastructure:
    def __init__(self, num_components=3):
        self.components = {
            f'comp_{i}': MockComponent()
            for i in range(num_components)
        }
        self.output_nodes = {'node1': None}

    def calc_output_loss(self, scenario, damage_states):
        return (
            np.zeros((10, 3)),  # component_sample_loss
            np.ones((10, 3)),   # comp_sample_func
            np.ones((10, 1, 3)),  # infrastructure_sample_output
            np.ones((10, 1))    # infrastructure_sample_economic_loss
        )

    def calc_response(self, *args):
        return (
            {'comp_responses': {}},  # component_response_dict
            {'type_responses': {}},  # comptype_response_dict
            {'dmg_levels': {}},     # compclass_dmg_level_percentages
            {'dmg_indices': {}}     # compclass_dmg_index_expected
        )

class MockHazard:
    def __init__(self, hazard_event_id='TEST_001'):
        self.hazard_event_id = hazard_event_id

    def get_hazard_intensity(self, *args):
        return 0.5

    def get_seed(self):
        return 42

class MockScenario:
    def __init__(self, run_parallel_proc=False):
        self.run_parallel_proc = run_parallel_proc
        self.num_samples = 10
        self.run_context = True

class FailingInfrastructure(MockInfrastructure):
    def calc_output_loss(self, *args):
        raise Exception("Test error")

@pytest.fixture
def mock_scenario():
    return MockScenario()

@pytest.fixture
def mock_infrastructure():
    return MockInfrastructure()

@pytest.fixture
def mock_hazard():
    return MockHazard()

def test_calculate_response_for_single_hazard(mock_scenario, mock_infrastructure, mock_hazard):
    """Test the single hazard calculation function"""
    result = calculate_response_for_single_hazard(
        mock_hazard, mock_scenario, mock_infrastructure
    )

    assert isinstance(result, dict)
    assert mock_hazard.hazard_event_id in result
    assert len(result[mock_hazard.hazard_event_id]) == 8

def test_calculate_response_serial(mock_scenario, mock_infrastructure):
    """Test the main calculate_response function in serial mode"""
    class MockHazardContainer:
        def __init__(self):
            self.listOfhazards = [MockHazard(f'TEST_{i:03d}') for i in range(3)]

    hazards = MockHazardContainer()
    result = calculate_response(hazards, mock_scenario, mock_infrastructure)

    assert len(result) == 8
    assert isinstance(result[0], dict)
    assert isinstance(result[4], np.ndarray)

@pytest.mark.slow
def test_calculate_response_parallel(mock_infrastructure):
    """Test the main calculate_response function in parallel mode"""
    scenario = MockScenario(run_parallel_proc=True)

    class MockHazardContainer:
        def __init__(self):
            self.listOfhazards = [MockHazard(f'TEST_{i:03d}') for i in range(5)]

    hazards = MockHazardContainer()

    with patch('sira.simulation.exit') as mock_exit:  # Prevent actual system exit
        result = calculate_response(hazards, scenario, mock_infrastructure)
        assert not mock_exit.called

    assert len(result) == 8
    assert isinstance(result[0], dict)
    assert isinstance(result[4], np.ndarray)

def test_process_component_batch(mock_scenario, mock_infrastructure, mock_hazard):
    """Test the component batch processing function"""
    rnd = np.random.RandomState(42).uniform(size=(10, 3))
    batch_data = (0, ['comp_0', 'comp_1', 'comp_2'])

    start_idx, batch_results = process_component_batch(
        batch_data, mock_infrastructure, mock_scenario, mock_hazard, rnd
    )

    assert start_idx == 0
    assert isinstance(batch_results, np.ndarray)
    assert batch_results.shape == (10, 3)

@pytest.mark.parametrize("num_components", [50, 200])
def test_calc_component_damage_state_for_n_simulations(mock_scenario, mock_hazard, num_components):
    """Test damage state calculation with different numbers of components"""
    infrastructure = MockInfrastructure(num_components=num_components)

    result = calc_component_damage_state_for_n_simulations(
        infrastructure, mock_scenario, mock_hazard
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (mock_scenario.num_samples, num_components)


class FailingHazard(MockHazard):
    def get_hazard_intensity(self, *args):
        raise Exception("Test error in hazard intensity calculation")

def test_error_handling_in_parallel_processing():
    scenario = MockScenario(run_parallel_proc=True)
    infrastructure = MockInfrastructure(num_components=5)

    class MockHazardContainer:
        def __init__(self):
            self.listOfhazards = [FailingHazard('TEST_001')]

    hazards = MockHazardContainer()

    with patch('sira.simulation.rootLogger.error') as mock_logger:
        try:
            calculate_response(hazards, scenario, infrastructure)
        except Exception as e:
            assert "Test error in hazard intensity calculation" in str(e)
            assert mock_logger.called
            return

    pytest.fail("Expected exception was not raised")


def test_random_number_consistency(mock_infrastructure, mock_hazard):
    """Test consistency of random number generation"""
    scenario = MockScenario()
    scenario.run_context = True

    result1 = calc_component_damage_state_for_n_simulations(
        mock_infrastructure, scenario, mock_hazard
    )
    result2 = calc_component_damage_state_for_n_simulations(
        mock_infrastructure, scenario, mock_hazard
    )

    np.testing.assert_array_equal(result1, result2)
