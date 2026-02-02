import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from sira.modelling.hazard import (
    haversine_dist,
    find_nearest,
    Hazard,
    HazardsContainer
)

# Test data fixtures
@pytest.fixture
def sample_hazard_data():
    return pd.DataFrame({
        'event_id': ['EQ_01', 'EQ_02'],
        'site_id': ['1', '1'],
        'hazard_intensity': [0.5, 0.7]
    })

@pytest.fixture
def sample_config():
    config = Mock()
    config.HAZARD_TYPE = "earthquake"
    config.HAZARD_INTENSITY_MEASURE_PARAM = "PGA"
    config.HAZARD_INTENSITY_MEASURE_UNIT = "g"
    config.FOCAL_HAZARD_SCENARIOS = []
    config.HAZARD_INPUT_HEADER = "hazard_intensity"
    config.HAZARD_SCALING_FACTOR = 1.0
    config.HAZARD_INPUT_METHOD = "calculated_array"
    config.INTENSITY_MEASURE_MAX = 1.0
    config.INTENSITY_MEASURE_MIN = 0.0
    config.INTENSITY_MEASURE_STEP = 0.1
    return config

@pytest.fixture
def sample_model_file(tmp_path):
    model_data = {
        "component_list": {
            "comp1": {"pos_x": 150.0, "pos_y": -34.0, "site_id": "1"},
            "comp2": {"pos_x": 151.0, "pos_y": -34.5, "site_id": "2"}
        }
    }
    file_path = tmp_path / "model.json"
    with open(file_path, "w") as f:
        import json
        json.dump(model_data, f)
    return str(file_path)

def test_haversine_dist():
    """Test haversine distance calculation"""
    # Sydney to Melbourne approximate coordinates
    dist = haversine_dist(151.2093, -33.8688, 144.9631, -37.8136)
    assert isinstance(dist, float)
    assert 700 < dist < 800  # Approximate distance in km

def test_find_nearest():
    """Test finding nearest point in dataframe"""
    df = pd.DataFrame({
        'pos_x': [151.0, 145.0],
        'pos_y': [-34.0, -38.0],
        'value': ['A', 'B']
    })
    # Point closer to first location
    result = find_nearest(151.1, -34.1, df, 'value')
    assert result == 'A'

def test_hazard_class_basic():
    """Test basic Hazard class functionality"""
    df = pd.DataFrame({
        'event_id': ['TEST_01', 'TEST_01'],
        'site_id': ['1', '2'],
        'hazard_intensity': [0.5, 0.7]
    })
    df.set_index(['event_id', 'site_id'], inplace=True, drop=True)
    df = df.unstack(level=-1, fill_value=0)
    df = df.droplevel(axis='columns', level=0)

    hazard = Hazard(
        hazard_event_id="TEST_01",
        hazard_input_method="scenario_file",
        hazard_intensity_header="hazard_intensity",
        hazard_data_df=df
    )

    assert hazard.get_hazard_intensity(0, 0, site_id='1') == 0.5
    assert hazard.get_hazard_intensity(0, 0, site_id=-1) == 0
    assert isinstance(hazard.get_seed(), int)

def test_hazard_class_methods():
    """Test different hazard input methods"""
    # Test for calculated_array method
    df_calc = pd.DataFrame({'0': [0.5]}, index=['TEST_01'])
    hazard_calc = Hazard("TEST_01", "calculated_array", "hazard_intensity", df_calc)
    assert hazard_calc.get_hazard_intensity(0, 0) == 0.5

    # Test for hazard_file method
    df_file = pd.DataFrame({
        'pos_x': [151.0],
        'pos_y': [-34.0],
        'TEST_01': [0.7]
    })
    hazard_file = Hazard("TEST_01", "hazard_file", "hazard_intensity", df_file)
    assert hazard_file.get_hazard_intensity(151.0, -34.0) == 0.7

def test_hazards_container_calculated_array(sample_config, sample_model_file):
    """Test HazardsContainer with calculated array method"""
    container = HazardsContainer(sample_config, sample_model_file)

    assert len(container.listOfhazards) == 11  # Based on min/max/step values
    assert len(container.hazard_intensity_list) == 11
    assert container.hazard_type == "earthquake"

def test_hazards_container_hazard_file(sample_config, sample_model_file, tmp_path):
    """Test HazardsContainer with hazard file method"""
    # Create test hazard file
    hazard_file = tmp_path / "hazard.csv"
    pd.DataFrame({
        'event_id': ['EQ_01', 'EQ_02'],
        'site_id': ['1', '1'],
        'hazard_intensity': [0.5, 0.7],
        'pos_x': [151.0, 151.0],
        'pos_y': [-34.0, -34.0]
    }).to_csv(hazard_file, index=False)

    sample_config.HAZARD_INPUT_METHOD = "hazard_file"
    sample_config.HAZARD_INPUT_FILE = str(hazard_file)

    container = HazardsContainer(sample_config, sample_model_file)
    assert len(container.listOfhazards) == 2
    assert len(container.hazard_scenario_list) == 2

def test_hazard_container_scaling(sample_config, sample_model_file, tmp_path):
    """Test hazard scaling factor"""
    hazard_file = tmp_path / "hazard.csv"
    pd.DataFrame({
        'event_id': ['EQ_01'],
        'site_id': ['1'],
        'hazard_intensity': [0.5],
        'pos_x': [151.0],
        'pos_y': [-34.0]
    }).to_csv(hazard_file, index=False)

    sample_config.HAZARD_INPUT_METHOD = "hazard_file"
    sample_config.HAZARD_INPUT_FILE = str(hazard_file)
    sample_config.HAZARD_SCALING_FACTOR = 2.0

    container = HazardsContainer(sample_config, sample_model_file)
    assert container.hazard_intensity_list[0] == 1.0  # Should be scaled by 2.0

def test_hazard_get_listOfhazards(sample_config, sample_model_file):
    """Test hazard list generator"""
    container = HazardsContainer(sample_config, sample_model_file)
    hazard_list = list(container.get_listOfhazards())
    assert len(hazard_list) == len(container.listOfhazards)
    assert all(isinstance(h, Hazard) for h in hazard_list)

def test_invalid_hazard_intensity():
    """Test error handling for invalid hazard intensity calculation"""
    hazard = Hazard("TEST", "invalid_method", "hazard_intensity", pd.DataFrame())
    with pytest.raises(Exception):
        hazard.get_hazard_intensity(0, 0)
