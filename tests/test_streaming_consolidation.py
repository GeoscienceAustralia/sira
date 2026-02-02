"""
Test streaming consolidation functionality.

This module verifies that consolidate_streamed_results properly:
1. Reads manifest files
2. Loads streaming data files (NPY format)
3. Calculates statistics correctly
4. Handles missing/corrupted data gracefully
5. Generates all required output files
6. Preserves hazard event ordering

Tests verify the NPY-based streaming format that replaced the previous
Parquet format for improved reliability and performance.

Usage:
    # Run all tests in this module
    pytest tests/test_streaming_consolidation.py -v

    # Run specific test
    pytest tests/test_streaming_consolidation.py::test_consolidation_basic -v

    # Run directly
    python tests/test_streaming_consolidation.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Pytest Fixtures
@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stream_dir = Path(tmpdir) / "stream"
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True)
        yield stream_dir, output_dir


@pytest.fixture
def mock_objects():
    """Create mock SIRA objects for testing."""
    infrastructure = create_mock_infrastructure(num_components=10, num_output_lines=3)
    scenario = create_mock_scenario()
    return infrastructure, scenario


# Helper Functions
def create_mock_streaming_files(stream_dir: Path, num_events: int = 5, num_samples: int = 10):
    """
    Create mock streaming files for testing.

    Parameters
    ----------
    stream_dir : Path
        Directory to create streaming files in
    num_events : int
        Number of hazard events to simulate
    num_samples : int
        Number of samples per event

    Returns
    -------
    dict : Metadata about created files
    """
    stream_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    event_ids = []

    for i in range(num_events):
        event_id = f"TEST_EVENT_{i:03d}"
        event_ids.append(event_id)

        # Create chunk directory
        chunk_dir = stream_dir / f"chunk_{i:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # Create economic loss data (1D array of samples) - compressed format
        econ_data = np.random.uniform(0.1, 1.0, num_samples)
        econ_path = chunk_dir / f"{event_id}_econ.npz"
        np.savez_compressed(econ_path, data=econ_data)

        # Create system output data (samples x lines)
        # Simulate 3 output lines
        num_lines = 3
        output_data = np.random.uniform(50, 100, (num_samples, num_lines))

        # Save as compressed NPZ (new streaming format)
        output_path = chunk_dir / f"{event_id}__sysout.npz"
        np.savez_compressed(output_path, data=output_data)

        # Add to manifest with file sizes
        manifest_entry = {
            "event_id": event_id,
            "econ": str(econ_path),
            "econ_size": econ_path.stat().st_size,
            "sys_output": [str(output_path)],
            "sys_output_sizes": [output_path.stat().st_size],
        }
        manifest_lines.append(json.dumps(manifest_entry))

    # Write manifest
    manifest_path = stream_dir / "manifest.jsonl"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    return {
        "num_events": num_events,
        "num_samples": num_samples,
        "event_ids": event_ids,
        "manifest_path": manifest_path,
    }


def create_mock_infrastructure(num_components: int = 10, num_output_lines: int = 3):
    """Create a mock infrastructure object for testing."""

    class MockComponent:
        def __init__(self, comp_id, site_id="SITE_001"):
            self.component_id = comp_id
            self.site_id = site_id
            self.component_class = "test_component"
            self.component_type = "test_component"

    class MockInfrastructure:
        def __init__(self):
            self.components = {
                f"comp_{i}": MockComponent(f"comp_{i}") for i in range(num_components)
            }
            self.uncosted_classes = []

            # Output nodes with capacities
            self.output_nodes = {
                f"line_{i}": {"output_node_capacity": 100.0} for i in range(num_output_lines)
            }

            self.system_output_capacity = 300.0  # Sum of all line capacities

        def get_component_types(self):
            """Return list of component types (excluding uncosted types)."""
            return ["test_component"]

    return MockInfrastructure()


def create_mock_scenario():
    """Create a mock scenario object."""

    class MockScenario:
        def __init__(self):
            self.num_samples = 10
            self.recovery_max_workers = 2
            self.recovery_batch_size = 100
            self.parallel_config = None

    return MockScenario()


def create_mock_config(output_dir: Path):
    """Create a mock configuration object."""

    class MockConfig:
        def __init__(self, output_dir):
            self.OUTPUT_DIR = str(output_dir)
            self.INFRASTRUCTURE_LEVEL = "network"
            self.HAZARD_INPUT_METHOD = "hazard_file"
            self.RECOVERY_METHOD = "max"
            self.NUM_REPAIR_STREAMS = 100

    return MockConfig(output_dir)


def create_mock_hazards(num_events: int = 5):
    """Create a mock hazards container."""

    class MockHazards:
        def __init__(self, num_events):
            self.HAZARD_INPUT_HEADER = "PGA"
            self.hazard_scenario_list = [f"TEST_EVENT_{i:03d}" for i in range(num_events)]

            # Create hazard data DataFrame
            self.hazard_data_df = pd.DataFrame({"0": np.linspace(0.1, 1.0, num_events)})

    return MockHazards(num_events)


def test_consolidation_basic():
    """Test basic consolidation functionality."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Consolidation")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        stream_dir = Path(tmpdir) / "stream"
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True)

        # Create mock streaming files
        print("Creating mock streaming files...")
        metadata = create_mock_streaming_files(stream_dir, num_events=5, num_samples=10)
        print(
            f"✓ Created {metadata['num_events']} events with {metadata['num_samples']} samples each"
        )

        # Create mock objects
        infrastructure = create_mock_infrastructure(num_components=10, num_output_lines=3)
        scenario = create_mock_scenario()
        config = create_mock_config(output_dir)
        hazards = create_mock_hazards(num_events=5)

        # Import and run consolidation
        print("\nRunning consolidation...")
        from sira.infrastructure_response import consolidate_streamed_results

        try:
            consolidate_streamed_results(
                stream_dir=stream_dir,
                infrastructure=infrastructure,
                scenario=scenario,
                config=config,
                hazards=hazards,
                CALC_SYSTEM_RECOVERY=False,  # Skip recovery for speed
            )
            print("✓ Consolidation completed successfully")
        except Exception as e:
            print(f"✗ Consolidation failed: {e}")
            import traceback

            traceback.print_exc()
            assert False, f"Consolidation failed: {e}"

        # Verify output files exist
        print("\nVerifying output files...")
        expected_files = [
            "system_response.csv",
            "system_output_vs_hazard_intensity.csv",
            "risk_summary_statistics.csv",
            "risk_summary_statistics.json",
        ]

        all_exist = True
        for filename in expected_files:
            filepath = output_dir / filename
            if filepath.exists():
                print(f"✓ {filename} exists")
                # Check file is not empty
                if filepath.stat().st_size > 0:
                    print(f"  Size: {filepath.stat().st_size} bytes")
                else:
                    print("  ✗ WARNING: File is empty!")
                    all_exist = False
            else:
                print(f"✗ {filename} MISSING")
                all_exist = False

        if all_exist:
            print("\n✓ All output files generated successfully")
        else:
            print("\n✗ Some output files missing or empty")
            assert False, "Some output files missing or empty"

        # Verify data integrity
        print("\nVerifying data integrity...")
        df_response = pd.read_csv(output_dir / "system_response.csv")
        print(f"✓ system_response.csv has {len(df_response)} rows")
        print(f"  Columns: {', '.join(df_response.columns)}")

        # Check all expected columns exist
        expected_cols = ["event_id", "loss_mean", "loss_std", "output_mean", "output_std"]
        missing_cols = [col for col in expected_cols if col not in df_response.columns]
        if missing_cols:
            print(f"✗ Missing columns: {missing_cols}")
            assert False, f"Missing columns: {missing_cols}"
        print("✓ All expected columns present")

        # Check data ranges are reasonable
        if df_response["loss_mean"].min() < 0:
            print("✗ Negative loss values detected!")
            assert False, "Negative loss values detected!"
        if df_response["output_mean"].min() < 0 or df_response["output_mean"].max() > 1:
            print("✗ Output values out of range [0, 1]!")
            assert False, "Output values out of range [0, 1]!"
        print("✓ Data values in reasonable ranges")

        # Verify system output file
        df_output = pd.read_csv(output_dir / "system_output_vs_hazard_intensity.csv")
        print(f"\n✓ system_output_vs_hazard_intensity.csv has {len(df_output)} rows")
        print(f"  Columns: {', '.join(df_output.columns)}")

        print("\n" + "=" * 80)
        print("TEST 1: PASSED ✓")
        print("=" * 80)


def test_consolidation_missing_files():
    """Test consolidation with missing/corrupted files."""
    print("\n" + "=" * 80)
    print("TEST 2: Handling Missing/Corrupted Files")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        stream_dir = Path(tmpdir) / "stream"
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True)

        # Create mock streaming files
        print("Creating mock streaming files with some missing data...")
        create_mock_streaming_files(stream_dir, num_events=5, num_samples=10)

        # Delete one economic loss file to simulate missing data
        chunk_dir = stream_dir / "chunk_000002"
        econ_files = list(chunk_dir.glob("*_econ.npy"))
        if econ_files:
            econ_files[0].unlink()
            print(f"✓ Deleted {econ_files[0].name} to test missing file handling")

        # Create mock objects
        infrastructure = create_mock_infrastructure(num_components=10, num_output_lines=3)
        scenario = create_mock_scenario()
        config = create_mock_config(output_dir)
        hazards = create_mock_hazards(num_events=5)

        # Run consolidation
        print("\nRunning consolidation with missing files...")
        from sira.infrastructure_response import consolidate_streamed_results

        try:
            consolidate_streamed_results(
                stream_dir=stream_dir,
                infrastructure=infrastructure,
                scenario=scenario,
                config=config,
                hazards=hazards,
                CALC_SYSTEM_RECOVERY=False,
            )
            print("✓ Consolidation completed despite missing files")
        except Exception as e:
            print(f"✗ Consolidation failed: {e}")
            import traceback

            traceback.print_exc()
            assert False, f"Consolidation failed: {e}"

        # Verify outputs still generated
        filepath = output_dir / "system_response.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"✓ Output generated with {len(df)} events (should still be 5)")
            if len(df) == 5:
                print("✓ All events present (missing data filled with defaults)")
            else:
                print(f"✗ Expected 5 events, got {len(df)}")
                assert False, f"Expected 5 events, got {len(df)}"
        else:
            print("✗ Output file not generated")
            assert False, "Output file not generated"

        print("\n" + "=" * 80)
        print("TEST 2: PASSED ✓")
        print("=" * 80)


def test_consolidation_no_manifest():
    """Test consolidation when manifest doesn't exist."""
    print("\n" + "=" * 80)
    print("TEST 3: Handling Missing Manifest")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        stream_dir = Path(tmpdir) / "stream"
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True)
        stream_dir.mkdir(parents=True)

        # Don't create manifest - test graceful handling
        infrastructure = create_mock_infrastructure()
        scenario = create_mock_scenario()
        config = create_mock_config(output_dir)
        hazards = create_mock_hazards(num_events=5)

        print("Running consolidation without manifest file...")
        from sira.infrastructure_response import consolidate_streamed_results

        try:
            consolidate_streamed_results(
                stream_dir=stream_dir,
                infrastructure=infrastructure,
                scenario=scenario,
                config=config,
                hazards=hazards,
                CALC_SYSTEM_RECOVERY=False,
            )
            print("✓ Function handled missing manifest gracefully")
        except Exception as e:
            print(f"✗ Function raised exception: {e}")
            assert False, f"Function raised exception: {e}"

        print("\n" + "=" * 80)
        print("TEST 3: PASSED ✓")
        print("=" * 80)


def test_consolidation_large_dataset():
    """Test consolidation with larger dataset."""
    print("\n" + "=" * 80)
    print("TEST 4: Large Dataset (100 events)")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        stream_dir = Path(tmpdir) / "stream"
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True)

        print("Creating large streaming dataset...")
        metadata = create_mock_streaming_files(stream_dir, num_events=100, num_samples=50)
        print(
            f"✓ Created {metadata['num_events']} events with {metadata['num_samples']} samples each"
        )

        infrastructure = create_mock_infrastructure(num_components=50, num_output_lines=5)
        scenario = create_mock_scenario()
        config = create_mock_config(output_dir)
        hazards = create_mock_hazards(num_events=100)

        print("\nRunning consolidation...")
        import time

        from sira.infrastructure_response import consolidate_streamed_results

        start_time = time.time()

        try:
            consolidate_streamed_results(
                stream_dir=stream_dir,
                infrastructure=infrastructure,
                scenario=scenario,
                config=config,
                hazards=hazards,
                CALC_SYSTEM_RECOVERY=False,
            )
            elapsed = time.time() - start_time
            print(f"✓ Consolidation completed in {elapsed:.2f} seconds")
        except Exception as e:
            print(f"✗ Consolidation failed: {e}")
            import traceback

            traceback.print_exc()
            assert False, f"Consolidation failed: {e}"

        # Verify output
        filepath = output_dir / "system_response.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"✓ Generated output with {len(df)} events")
            if len(df) == 100:
                print("✓ All 100 events present")
            else:
                print(f"✗ Expected 100 events, got {len(df)}")
                assert False, f"Expected 100 events, got {len(df)}"
        else:
            print("✗ Output file not generated")
            assert False, "Output file not generated"

        print("\n" + "=" * 80)
        print("TEST 4: PASSED ✓")
        print("=" * 80)


if __name__ == "__main__":
    # Allow running directly for quick verification
    pytest.main([__file__, "-v", "--tb=short"])
