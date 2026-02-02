"""
Additional tests for recovery_analysis module focusing on main workflow functions.

These tests exercise the RecoveryAnalysisEngine and main analysis functions
using minimal mocks and test fixtures.
"""

from unittest.mock import Mock, patch

import pytest

from sira.recovery_analysis import (
    RecoveryAnalysisEngine,
    calculate_event_recovery,
    process_event_chunk,
)

# ==============================================================================
# Tests for RecoveryAnalysisEngine
# ==============================================================================


def test_recovery_analysis_engine_init_auto():
    """Test RecoveryAnalysisEngine initialization with auto backend."""
    with patch("sira.recovery_analysis.is_mpi_environment", return_value=False):
        engine = RecoveryAnalysisEngine(backend="auto")

        assert engine.backend == "multiprocessing"
        assert engine.rank == 0
        assert engine.size == 1


def test_recovery_analysis_engine_init_multiprocessing():
    """Test explicit multiprocessing backend."""
    engine = RecoveryAnalysisEngine(backend="multiprocessing")

    assert engine.backend == "multiprocessing"
    assert engine.comm is None


def test_recovery_analysis_engine_init_mpi_environment():
    """Test MPI backend selection in MPI environment."""
    with patch("sira.recovery_analysis.is_mpi_environment", return_value=True):
        mock_mpi = Mock()
        mock_mpi.Is_initialized.return_value = False
        mock_comm = Mock()
        mock_comm.Get_size.return_value = 4
        mock_comm.Get_rank.return_value = 0

        with patch(
            "sira.recovery_analysis.safe_mpi_import", return_value=(mock_mpi, mock_comm, 0, 4)
        ):
            engine = RecoveryAnalysisEngine(backend="auto")

            assert engine.backend == "mpi"
            assert engine.size == 4


def test_recovery_analysis_engine_mpi_fallback():
    """Test fallback to multiprocessing when MPI not available."""
    with patch("sira.recovery_analysis.safe_mpi_import", return_value=(None, None, 0, 1)):
        engine = RecoveryAnalysisEngine(backend="mpi")

        assert engine.backend == "multiprocessing"


def test_recovery_analysis_engine_select_backend_auto_no_mpi():
    """Test backend selection without MPI."""
    engine = RecoveryAnalysisEngine()
    with patch("sira.recovery_analysis.is_mpi_environment", return_value=False):
        backend = engine._select_backend("auto")

        assert backend == "multiprocessing"


def test_recovery_analysis_engine_select_backend_auto_with_mpi():
    """Test backend selection with MPI environment."""
    engine = RecoveryAnalysisEngine()
    with patch("sira.recovery_analysis.is_mpi_environment", return_value=True):
        backend = engine._select_backend("auto")

        assert backend == "mpi"


def test_recovery_analysis_engine_select_backend_explicit():
    """Test explicit backend selection."""
    engine = RecoveryAnalysisEngine()

    assert engine._select_backend("multiprocessing") == "multiprocessing"
    assert engine._select_backend("mpi") == "mpi"


# ==============================================================================
# Tests for calculate_event_recovery
# ==============================================================================


def test_calculate_event_recovery_basic():
    """Test basic event recovery calculation."""
    mock_config = Mock()

    mock_component = Mock()
    mock_component.get_location.return_value = (0.0, 0.0, "site1")

    mock_infrastructure = Mock()
    mock_infrastructure.components = {"comp1": mock_component}

    mock_hazard = Mock()
    mock_hazard.get_hazard_intensity.return_value = 0.5

    components_costed = ["comp1"]

    with patch(
        "sira.recovery_analysis.loss_analysis.calc_component_recovery_time", return_value=10
    ):
        recovery_time = calculate_event_recovery(
            mock_config,
            "event_001",
            mock_hazard,
            components_costed,
            mock_infrastructure,
            recovery_method="max",
            num_repair_streams=100,
        )

        assert recovery_time == 10


def test_calculate_event_recovery_multiple_components():
    """Test recovery calculation with multiple components."""
    mock_config = Mock()

    mock_comp1 = Mock()
    mock_comp1.get_location.return_value = (0.0, 0.0, "site1")

    mock_comp2 = Mock()
    mock_comp2.get_location.return_value = (1.0, 1.0, "site2")

    mock_infrastructure = Mock()
    mock_infrastructure.components = {"comp1": mock_comp1, "comp2": mock_comp2}

    mock_hazard = Mock()
    mock_hazard.get_hazard_intensity.return_value = 0.5

    components_costed = ["comp1", "comp2"]

    # Mock different recovery times for different components
    with patch(
        "sira.recovery_analysis.loss_analysis.calc_component_recovery_time",
        side_effect=[15, 25],
    ):
        recovery_time = calculate_event_recovery(
            mock_config,
            "event_001",
            mock_hazard,
            components_costed,
            mock_infrastructure,
            recovery_method="max",
        )

        # Should return max recovery time
        assert recovery_time == 25


def test_calculate_event_recovery_zero_damage():
    """Test recovery calculation when no damage occurs."""
    mock_config = Mock()

    mock_component = Mock()
    mock_component.get_location.return_value = (0.0, 0.0, "site1")

    mock_infrastructure = Mock()
    mock_infrastructure.components = {"comp1": mock_component}

    mock_hazard = Mock()
    mock_hazard.get_hazard_intensity.return_value = 0.1

    components_costed = ["comp1"]

    with patch("sira.recovery_analysis.loss_analysis.calc_component_recovery_time", return_value=0):
        recovery_time = calculate_event_recovery(
            mock_config,
            "event_001",
            mock_hazard,
            components_costed,
            mock_infrastructure,
        )

        assert recovery_time == 0


def test_calculate_event_recovery_error_handling():
    """Test recovery calculation with component errors."""
    mock_config = Mock()

    mock_component = Mock()
    mock_component.get_location.return_value = (0.0, 0.0, "site1")

    mock_infrastructure = Mock()
    mock_infrastructure.components = {"comp1": mock_component}

    mock_hazard = Mock()
    mock_hazard.get_hazard_intensity.return_value = 0.5

    components_costed = ["comp1"]

    # Simulate error in recovery time calculation
    with patch(
        "sira.recovery_analysis.loss_analysis.calc_component_recovery_time",
        side_effect=Exception("Calculation failed"),
    ):
        recovery_time = calculate_event_recovery(
            mock_config,
            "event_001",
            mock_hazard,
            components_costed,
            mock_infrastructure,
        )

        # Should return 0 when error occurs
        assert recovery_time == 0


def test_calculate_event_recovery_non_numeric_result():
    """Test handling of non-numeric recovery time results."""
    mock_config = Mock()

    mock_component = Mock()
    mock_component.get_location.return_value = (0.0, 0.0, "site1")

    mock_infrastructure = Mock()
    mock_infrastructure.components = {"comp1": mock_component}

    mock_hazard = Mock()
    mock_hazard.get_hazard_intensity.return_value = 0.5

    components_costed = ["comp1"]

    # Return non-numeric value
    with patch(
        "sira.recovery_analysis.loss_analysis.calc_component_recovery_time",
        return_value=None,
    ):
        recovery_time = calculate_event_recovery(
            mock_config,
            "event_001",
            mock_hazard,
            components_costed,
            mock_infrastructure,
        )

        assert recovery_time == 0


def test_calculate_event_recovery_float_rounding():
    """Test that float recovery times are rounded correctly."""
    mock_config = Mock()

    mock_component = Mock()
    mock_component.get_location.return_value = (0.0, 0.0, "site1")

    mock_infrastructure = Mock()
    mock_infrastructure.components = {"comp1": mock_component}

    mock_hazard = Mock()
    mock_hazard.get_hazard_intensity.return_value = 0.5

    components_costed = ["comp1"]

    with patch(
        "sira.recovery_analysis.loss_analysis.calc_component_recovery_time",
        return_value=15.7,
    ):
        recovery_time = calculate_event_recovery(
            mock_config,
            "event_001",
            mock_hazard,
            components_costed,
            mock_infrastructure,
        )

        # Should round to nearest integer
        assert recovery_time == 16


# ==============================================================================
# Tests for process_event_chunk
# ==============================================================================


def test_process_event_chunk_basic():
    """Test processing a single event chunk."""
    mock_config = Mock()

    mock_component = Mock()
    mock_component.get_location.return_value = (0.0, 0.0, "site1")

    mock_infrastructure = Mock()
    mock_infrastructure.components = {"comp1": mock_component}

    mock_hazard = Mock()
    mock_hazard.hazard_event_id = "event_001"
    mock_hazard.get_hazard_intensity.return_value = 0.5

    mock_hazards = Mock()
    mock_hazards.hazard_scenario_list = ["event_001"]
    mock_hazards.listOfhazards = [mock_hazard]

    components_costed = ["comp1"]

    with patch(
        "sira.recovery_analysis.loss_analysis.calc_component_recovery_time", return_value=10
    ):
        results = process_event_chunk(
            ["event_001"],
            mock_config,
            mock_hazards,
            components_costed,
            mock_infrastructure,
            "max",
            100,
        )

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0] == 10


def test_process_event_chunk_multiple_events():
    """Test processing multiple events in a chunk."""
    mock_config = Mock()

    mock_component = Mock()
    mock_component.get_location.return_value = (0.0, 0.0, "site1")

    mock_infrastructure = Mock()
    mock_infrastructure.components = {"comp1": mock_component}

    mock_hazard1 = Mock()
    mock_hazard1.hazard_event_id = "event_001"
    mock_hazard1.get_hazard_intensity.return_value = 0.5

    mock_hazard2 = Mock()
    mock_hazard2.hazard_event_id = "event_002"
    mock_hazard2.get_hazard_intensity.return_value = 0.7

    mock_hazards = Mock()
    mock_hazards.hazard_scenario_list = ["event_001", "event_002"]
    mock_hazards.listOfhazards = [mock_hazard1, mock_hazard2]

    components_costed = ["comp1"]

    with patch(
        "sira.recovery_analysis.loss_analysis.calc_component_recovery_time",
        side_effect=[10, 20],
    ):
        results = process_event_chunk(
            ["event_001", "event_002"],
            mock_config,
            mock_hazards,
            components_costed,
            mock_infrastructure,
            "max",
            100,
        )

        assert len(results) == 2
        assert results[0] == 10
        assert results[1] == 20


def test_process_event_chunk_empty():
    """Test processing empty event chunk."""
    mock_config = Mock()
    mock_infrastructure = Mock()
    mock_hazards = Mock()
    components_costed = []

    results = process_event_chunk(
        [],
        mock_config,
        mock_hazards,
        components_costed,
        mock_infrastructure,
        "max",
        100,
    )

    assert isinstance(results, list)
    assert len(results) == 0


# ==============================================================================
# Edge cases and integration-style tests
# ==============================================================================


def test_calculate_event_recovery_no_costed_components():
    """Test recovery calculation with no costed components.

    Note: Current implementation has a bug where max() on empty sequence raises ValueError.
    This test documents the current behavior. The function should return 0 instead.
    """
    mock_config = Mock()
    mock_infrastructure = Mock()
    mock_infrastructure.components = {}
    mock_hazard = Mock()

    components_costed = []

    # Current implementation raises ValueError on empty max()
    with pytest.raises(ValueError, match="max.*empty sequence"):
        calculate_event_recovery(
            mock_config,
            "event_001",
            mock_hazard,
            components_costed,
            mock_infrastructure,
        )


def test_recovery_analysis_engine_analyse_routing():
    """Test that analyse() routes to correct backend."""
    with patch("sira.recovery_analysis.is_mpi_environment", return_value=False):
        engine = RecoveryAnalysisEngine(backend="multiprocessing")

        # Mock the backend-specific method
        with patch.object(engine, "_analyse_multiprocessing", return_value=[1, 2, 3]) as mock_mp:
            mock_config = Mock()
            mock_hazards = Mock()
            mock_hazards.hazard_scenario_list = ["event_001", "event_002"]
            mock_infrastructure = Mock()
            mock_scenario = Mock()
            components_costed = ["comp1"]

            result = engine.analyse(
                mock_config,
                mock_hazards,
                mock_infrastructure,
                mock_scenario,
                components_costed,
                recovery_method="max",
                num_repair_streams=100,
            )

            assert result == [1, 2, 3]
            mock_mp.assert_called_once()


@pytest.mark.skip(reason="Requires full MPI setup")
def test_recovery_analysis_engine_analyse_mpi():
    """Test MPI backend analysis (requires actual MPI setup)."""
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
