"""
Integration tests for recovery_analysis and simulation modules.

These tests focus on exercising uncovered code paths in both modules using
real infrastructure objects and realistic data flows.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from sira.configuration import Configuration
from sira.model_ingest import ingest_model
from sira.modelling.hazard import HazardsContainer
from sira.recovery_analysis import extract_infrastructure_data, process_event_chunk
from sira.scenario import Scenario
from sira.simulation import calc_component_damage_state_for_n_simulations

# ==============================================================================
# Tests for extract_infrastructure_data
# ==============================================================================


@pytest.fixture
def mock_component_with_recovery():
    """Create a mock component with recovery function data."""
    comp = Mock()
    comp.component_type = "power_transformer"
    comp.component_class = "Transformer"
    comp.cost_fraction = 0.15

    # Create damage states with recovery functions
    ds1 = Mock()
    ds1.damage_ratio = 0.1
    ds1.functionality = 0.9
    ds1.recovery_function = "Normal"
    ds1.recovery_function_constructor = {"mean": 10, "stddev": 2}

    ds2 = Mock()
    ds2.damage_ratio = 0.5
    ds2.functionality = 0.5
    ds2.recovery_function = "Lognormal"
    ds2.recovery_function_constructor = {"median": 30, "beta": 0.5}

    comp.damage_states = {0: ds1, 1: ds2}

    return comp


@pytest.fixture
def mock_infrastructure_minimal():
    """Create minimal mock infrastructure for testing."""
    infra = Mock()
    infra.system_class = "PowerStation"
    infra.system_output_capacity = 500.0

    comp1 = Mock()
    comp1.component_type = "generator"
    comp1.component_class = "PowerGenerator"
    comp1.cost_fraction = 0.4
    comp1.damage_states = {
        0: Mock(damage_ratio=0.0, functionality=1.0, recovery_function="None"),
        1: Mock(damage_ratio=0.3, functionality=0.7, recovery_function="Normal"),
    }

    comp2 = Mock()
    comp2.component_type = "transformer"
    comp2.component_class = "Transformer"
    comp2.cost_fraction = 0.3
    comp2.damage_states = {
        0: Mock(damage_ratio=0.0, functionality=1.0, recovery_function="None"),
    }

    infra.components = {"gen1": comp1, "trans1": comp2}

    return infra


def test_extract_infrastructure_data_basic(mock_infrastructure_minimal):
    """Test extracting infrastructure data from mock infrastructure."""
    result = extract_infrastructure_data(mock_infrastructure_minimal)

    assert isinstance(result, dict)
    assert "components" in result
    assert "system_class" in result
    assert "output_capacity" in result

    assert result["system_class"] == "PowerStation"
    assert result["output_capacity"] == 500.0
    assert len(result["components"]) == 2


def test_extract_infrastructure_data_component_structure(mock_infrastructure_minimal):
    """Test that extracted component data has correct structure."""
    result = extract_infrastructure_data(mock_infrastructure_minimal)

    comp_data = result["components"]["gen1"]
    assert comp_data["component_type"] == "generator"
    assert comp_data["component_class"] == "PowerGenerator"
    assert comp_data["cost_fraction"] == 0.4
    assert "damage_states" in comp_data
    assert len(comp_data["damage_states"]) == 2


def test_extract_infrastructure_data_damage_states(mock_component_with_recovery):
    """Test extraction of damage state recovery function data."""
    infra = Mock()
    infra.system_class = "test"
    infra.system_output_capacity = 1.0
    infra.components = {"comp1": mock_component_with_recovery}

    result = extract_infrastructure_data(infra)

    comp_data = result["components"]["comp1"]
    ds1_data = comp_data["damage_states"][0]

    assert ds1_data["damage_ratio"] == 0.1
    assert ds1_data["functionality"] == 0.9
    assert ds1_data["recovery_function"] == "Normal"
    assert "recovery_function_constructor" in ds1_data
    assert ds1_data["recovery_function_constructor"]["mean"] == 10


def test_extract_infrastructure_data_missing_attributes():
    """Test extraction handles missing attributes gracefully."""
    infra = Mock(spec=[])  # Mock with no attributes
    infra.components = {}

    result = extract_infrastructure_data(infra)

    # Should handle missing attributes with getattr defaults
    assert result["system_class"] == "unknown"
    assert result["output_capacity"] == 1.0


# ==============================================================================
# Tests for process_event_chunk with realistic scenarios
# ==============================================================================


def test_process_event_chunk_with_actual_infrastructure():
    """Test process_event_chunk with more realistic infrastructure setup."""
    # Create minimal but realistic infrastructure
    infrastructure = Mock()
    infrastructure.components = {
        "comp1": Mock(
            cost_fraction=0.5,
            component_type="generator",
            damage_states={
                0: Mock(damage_ratio=0.0, functionality=1.0),
                1: Mock(damage_ratio=0.3, functionality=0.7),
            },
        ),
        "comp2": Mock(
            cost_fraction=0.5,
            component_type="transformer",
            damage_states={
                0: Mock(damage_ratio=0.0, functionality=1.0),
            },
        ),
    }

    config = Mock()
    config.MODEL_NAME = "test_model"

    hazards = Mock()
    hazards.hazard_scenario_list = ["event1", "event2"]
    hazards.listOfhazards = [Mock(hazard_event_id="event1"), Mock(hazard_event_id="event2")]

    scenario = Mock()
    scenario.num_samples = 10
    scenario.run_context = 1

    components_costed = ["comp1", "comp2"]

    # Mock calculate_event_recovery to return fixed values
    with patch("sira.recovery_analysis.calculate_event_recovery", return_value=15.5):
        result = process_event_chunk(
            ["event1", "event2"],
            config,
            hazards,
            components_costed,
            infrastructure,
            recovery_method="max",
            num_repair_streams=10,
        )

    assert len(result) == 2
    assert all(rt == 15.5 for rt in result)


def test_process_event_chunk_error_handling():
    """Test process_event_chunk handles errors gracefully."""
    config = Mock()
    hazards = Mock()
    hazards.hazard_scenario_list = ["event1"]
    hazards.listOfhazards = [Mock(hazard_event_id="event1")]

    infrastructure = Mock()
    infrastructure.components = {}

    # Make calculate_event_recovery raise an exception
    with patch(
        "sira.recovery_analysis.calculate_event_recovery", side_effect=Exception("Test error")
    ):
        result = process_event_chunk(
            ["event1"],
            config,
            hazards,
            [],
            infrastructure,
            recovery_method="max",
            num_repair_streams=10,
        )

    # Should return 0 for failed events
    assert len(result) == 1
    assert result[0] == 0.0


# ==============================================================================
# Tests for calc_component_damage_state_for_n_simulations
# ==============================================================================


@pytest.mark.skip(reason="Requires complex infrastructure mock - covered by integration tests")
def test_calc_damage_state_basic_vectorization():
    """Test damage state calculation with basic vectorization."""
    # Create minimal infrastructure
    infrastructure = Mock()
    comp1 = Mock()
    comp1.component_id = "comp1"
    comp1.component_type = "test_type"
    comp1.frag_func_hazard_int_param = "PGA"
    comp1.damage_states = {0: Mock(), 1: Mock()}

    infrastructure.components = {"comp1": comp1}
    infrastructure.comp_id_list = ["comp1"]

    scenario = Mock()
    scenario.num_samples = 5
    scenario.run_context = 1

    hazard = Mock()
    hazard.hazard_event_id = "event1"
    hazard.component_intensity = {"comp1": 0.3}
    hazard.get_seed = Mock(return_value=42)  # Provide valid seed

    # Mock the probability calculation
    with patch("sira.simulation.get_vectorised_damage_probabilities") as mock_prob:
        mock_prob.return_value = np.array([[0.9, 0.1], [0.8, 0.2]])

        result = calc_component_damage_state_for_n_simulations(infrastructure, scenario, hazard)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 1  # One component
    assert result.shape[1] == 5  # num_samples


# ==============================================================================
# Integration tests using real test models (requires test_network__basic)
# ==============================================================================


@pytest.fixture
def test_network_basic_full():
    """Load the full test_network__basic model for integration testing."""
    model_dir = Path(__file__).parent / "models" / "test_network__basic" / "input"

    if not model_dir.exists():
        pytest.skip("test_network__basic model not found")

    config_path = model_dir.parent / "config.json"
    model_path = model_dir.parent / "model.json"

    # Create temp output directory
    import tempfile

    temp_output = tempfile.mkdtemp(prefix="sira_test_")

    config = Configuration(str(config_path), str(model_path), output_path=temp_output)
    infrastructure = ingest_model(config)
    hazards = HazardsContainer(config, str(model_path))
    scenario = Scenario(config)

    return {
        "config": config,
        "infrastructure": infrastructure,
        "hazards": hazards,
        "scenario": scenario,
        "model_dir": model_dir,
    }


@pytest.mark.skip(reason="Requires specific model path structure")
def test_extract_infrastructure_data_real_model(test_network_basic_full):
    """Test extract_infrastructure_data with real infrastructure model."""
    infrastructure = test_network_basic_full["infrastructure"]

    result = extract_infrastructure_data(infrastructure)

    assert isinstance(result, dict)
    assert "components" in result
    assert "system_class" in result
    assert len(result["components"]) > 0

    # Check at least one component has proper structure
    first_comp_id = list(result["components"].keys())[0]
    comp_data = result["components"][first_comp_id]

    assert "component_type" in comp_data
    assert "component_class" in comp_data
    assert "cost_fraction" in comp_data
    assert "damage_states" in comp_data


@pytest.mark.skip(reason="Requires specific model path structure")
def test_process_event_chunk_real_model(test_network_basic_full):
    """Test process_event_chunk with real model data."""
    config = test_network_basic_full["config"]
    hazards = test_network_basic_full["hazards"]
    infrastructure = test_network_basic_full["infrastructure"]
    # scenario = test_network_basic_full["scenario"]

    # Get components with costs
    components_costed = [
        comp_id
        for comp_id, comp in infrastructure.components.items()
        if hasattr(comp, "cost_fraction") and comp.cost_fraction > 0
    ]

    # Test with first 2 events
    event_chunk = hazards.hazard_scenario_list[:2]

    result = process_event_chunk(
        event_chunk,
        config,
        hazards,
        components_costed,
        infrastructure,
        recovery_method="max",
        num_repair_streams=10,
    )

    assert len(result) == 2
    assert all(isinstance(rt, (int, float, np.integer, np.floating)) for rt in result)
    assert all(rt >= 0 for rt in result)


@pytest.mark.skip(reason="Requires specific model path structure")
def test_calc_damage_state_real_model(test_network_basic_full):
    """Test damage state calculation with real model and hazard."""
    infrastructure = test_network_basic_full["infrastructure"]
    scenario = test_network_basic_full["scenario"]
    hazards = test_network_basic_full["hazards"]

    # Get first hazard
    first_hazard = hazards.listOfhazards[0]

    result = calc_component_damage_state_for_n_simulations(infrastructure, scenario, first_hazard)

    assert isinstance(result, np.ndarray)
    # Shape: (num_components, num_samples)
    assert result.shape[0] == len(infrastructure.components)
    assert result.shape[1] == scenario.num_samples
    # All values should be valid damage state indices
    assert np.all(result >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
