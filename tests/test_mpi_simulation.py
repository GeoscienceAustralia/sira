"""
Test MPI simulation functionality.

This test suite validates that the MPI implementation in simulation.py
works correctly by:
- Testing hazard distribution across MPI ranks
- Validating MPI response calculation with and without streaming
- Ensuring correct fallback to multiprocessing when MPI is unavailable
- Verifying result consistency between parallel backends

Note: These tests use mock objects to avoid requiring actual MPI environment.
For real MPI testing, use: mpirun -n 4 pytest test_mpi_simulation.py
"""

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sira.simulation import _calculate_response_mpi, _distribute_hazards_mpi, calculate_response


class MockMPIComm:
    """Mock MPI communicator for testing"""

    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size
        self._gathered_data = []

    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return self._size

    def gather(self, data, root=0):
        """Simple gather implementation for testing"""
        if self._rank == root:
            # Simulate gathering data from all ranks
            return [data] * self._size
        return None

    def allreduce(self, data):
        """Simple allreduce that just returns the data multiplied by size"""
        return data * self._size


class MockHazard:
    def __init__(self, hazard_event_id="TEST_001"):
        self.hazard_event_id = hazard_event_id

    def get_hazard_intensity(self, *args):
        return 0.5

    def get_seed(self):
        return 42


class MockScenario:
    def __init__(self, run_parallel_proc=True):
        self.run_parallel_proc = run_parallel_proc
        self.num_samples = 10
        self.run_context = True

    def create_worker_copy(self):
        return self


class MockComponent:
    """Picklable mock component for multiprocessing tests"""

    def __init__(self, component_id):
        self.component_id = component_id
        self.damage_states = {}

    def get_location(self):
        return (0.0, 0.0)


class MockInfrastructure:
    def __init__(self, num_components=3):
        # Use picklable MockComponent instead of mock.Mock
        self.components = {f"comp_{i}": MockComponent(f"comp_{i}") for i in range(num_components)}
        self.output_nodes = {"node1": None}

    def calc_output_loss(self, scenario, damage_states):
        return (
            np.zeros((10, 3)),  # component_sample_loss
            np.ones((10, 3)),  # comp_sample_func
            np.ones((10, 1)),  # infrastructure_sample_output
            np.ones(10),  # infrastructure_sample_economic_loss
        )

    def calc_response(self, *args):
        return (
            {"comp_responses": {}},  # component_response_dict
            {"type_responses": {}},  # comptype_response_dict
            {"dmg_levels": {}},  # compclass_dmg_level_percentages
            {"dmg_indices": {}},  # compclass_dmg_index_expected
        )


class MockHazardsContainer:
    def __init__(self, num_hazards=10):
        self.listOfhazards = [MockHazard(f"HAZARD_{i:03d}") for i in range(num_hazards)]


def test_distribute_hazards_mpi():
    """Test that hazard distribution across MPI ranks is correct"""

    # Test even distribution
    distribution = _distribute_hazards_mpi(10, 2)
    expected = [(0, 5), (5, 10)]
    assert distribution == expected

    # Test uneven distribution
    distribution = _distribute_hazards_mpi(10, 3)
    expected = [(0, 4), (4, 7), (7, 10)]
    assert distribution == expected

    # Test single rank
    distribution = _distribute_hazards_mpi(10, 1)
    expected = [(0, 10)]
    assert distribution == expected

    # Test more ranks than hazards
    distribution = _distribute_hazards_mpi(3, 5)
    expected = [(0, 1), (1, 2), (2, 3), (3, 3), (3, 3)]
    assert distribution == expected


@pytest.mark.parametrize("rank,size", [(0, 1), (0, 2), (1, 2), (0, 4), (3, 4)])
def test_mpi_calculate_response_no_streaming(rank, size):
    """Test MPI response calculation without streaming"""

    # Setup mocks
    mock_comm = MockMPIComm(rank=rank, size=size)
    mock_scenario = MockScenario()
    mock_infrastructure = MockInfrastructure()
    mock_hazards = MockHazardsContainer(10)

    # Mock the single hazard calculation
    with mock.patch("sira.simulation.calculate_response_for_single_hazard") as mock_calc:
        mock_calc.return_value = {"TEST_001": [None] * 8}  # Standard response format

        result = _calculate_response_mpi(
            mock_hazards,
            mock_scenario,
            mock_infrastructure,
            mock_comm,
            list(mock_hazards.listOfhazards),
            10,
            streaming=False,
            stream_dir=None,
        )

        # Only rank 0 should return results
        if rank == 0:
            assert result is not None
            assert isinstance(result, list)
        else:
            assert result is None


def test_mpi_calculate_response_streaming():
    """Test MPI response calculation with streaming enabled"""

    # Setup mocks
    mock_comm = MockMPIComm(rank=0, size=2)
    mock_scenario = MockScenario()
    mock_infrastructure = MockInfrastructure()
    mock_hazards = MockHazardsContainer(10)

    # Create a temporary directory for streaming
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        stream_dir = Path(temp_dir)

        # Mock the single hazard calculation and persistence
        with (
            mock.patch("sira.simulation.calculate_response_for_single_hazard") as mock_calc,
            mock.patch("sira.simulation._persist_chunk_result"),
        ):
            mock_calc.return_value = {"TEST_001": [None] * 8}

            result = _calculate_response_mpi(
                mock_hazards,
                mock_scenario,
                mock_infrastructure,
                mock_comm,
                list(mock_hazards.listOfhazards),
                10,
                streaming=True,
                stream_dir=stream_dir,
            )

            # Should return streaming manifest for rank 0
            assert result is not None
            assert result["streaming"] is True  # type: ignore
            assert "stream_dir" in result
            assert result["total_hazards"] == 10  # type: ignore


def test_calculate_response_with_mpi():
    """Test that calculate_response correctly routes to MPI when comm is provided"""

    mock_comm = MockMPIComm(rank=0, size=2)
    mock_scenario = MockScenario()
    mock_infrastructure = MockInfrastructure()
    mock_hazards = MockHazardsContainer(5)

    # Mock the MPI function
    with mock.patch("sira.simulation._calculate_response_mpi") as mock_mpi:
        mock_mpi.return_value = [{"TEST_001": [None] * 8}]

        result = calculate_response(
            mock_hazards, mock_scenario, mock_infrastructure, mpi_comm=mock_comm
        )

        # Verify MPI function was called
        mock_mpi.assert_called_once()
        assert result is not None


def test_calculate_response_fallback_to_multiprocessing():
    """Test that calculate_response falls back to multiprocessing when no MPI comm is provided"""

    mock_scenario = MockScenario()
    mock_infrastructure = MockInfrastructure()
    mock_hazards = MockHazardsContainer(2)  # Small number for quick test

    # Mock the single hazard calculation to avoid actual computation
    with mock.patch("sira.simulation.calculate_response_for_single_hazard") as mock_calc:
        mock_calc.return_value = {
            "TEST_001": [
                np.zeros((10, 3)),  # expected_damage_state_of_components_for_n_simulations
                {"node1": 1.0},  # infrastructure_output
                {},  # component_response_dict
                {},  # comptype_response_dict
                np.ones((10, 1)),  # infrastructure_sample_output
                np.ones(10),  # infrastructure_sample_economic_loss
                {},  # compclass_dmg_level_percentages
                {},  # compclass_dmg_index_expected
            ]
        }

        result = calculate_response(mock_hazards, mock_scenario, mock_infrastructure, mpi_comm=None)

        # Should return the standard post-processing list format
        assert isinstance(result, list)
        assert len(result) == 8  # Standard SIRA response format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
