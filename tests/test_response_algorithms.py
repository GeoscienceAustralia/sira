import pytest
import numpy as np
from sira.modelling.responsemodels import Algorithm
from sira.fit_model import fit_prob_exceed_model
from pathlib import Path
import shutil

# ==============================================================================
# TEST DATA for Algorithms

func_data_piecewise1 = {
    "function_name": "PiecewiseFunction",
    "piecewise_function_constructor": [
        {
            "function_name": "normal",
            "mean": 0.3,
            "stddev": 0.5,
            "data_source": "source mysterio",
            "minimum": 'null',
            "lower_limit": 0.2,
            "upper_limit": 0.5,
            "damage_state_definition": "Not Available."
        },
        {
            "function_name": "normal",
            "mean": 0.3,
            "stddev": 0.7,
            "data_source": "source mysterio",
            "minimum": 'null',
            "lower_limit": 0.5,
            "upper_limit": 0.95,
            "damage_state_definition": "Not Available."
        },
        {
            "function_name": "Lognormal",
            "median": 0.4,
            "beta": 0.7,
            "location": 0,
            "data_source": "source mysterio",
            "minimum": 'null',
            "lower_limit": 0.95,
            "upper_limit": 2.0,
            "damage_state_definition": "Not Available."
        }
    ]
}

func_data_piecewise2 = {
    "function_name": "PiecewiseFunction",
    "piecewise_function_constructor": [
        {
            "function_name": "ConstantFunction",
            "amplitude": 1,
            "data_source": "source mysterio",
            "minimum": 'null',
            "lower_limit": 0.3,
            "upper_limit": 0.85,
            "damage_state_definition": "Not Available."
        },
        {
            "function_name": "ConstantFunction",
            "amplitude": 3,
            "data_source": "source mysterio",
            "minimum": 'null',
            "lower_limit": 0.85,
            "upper_limit": 2.0,
            "damage_state_definition": "Not Available."
        }
    ]
}

func_data_rayleigh = {
    "function_name": "RayleighCDF",
    "loc": 5.0,
    "scale": 1.1,
    "data_source": "Not Available.",
    "damage_state_definition": "Not Available."
}

# -----------------------------------------

@pytest.mark.algorithms
def test_algorithm_mdl_pw_parametric():
    """Tests for parametric piecewise defined functions"""
    fn = Algorithm.factory(func_data_piecewise1)
    xvals = np.linspace(0, 1.5, 16, endpoint=True)
    fn_output = fn(xvals)
    # Test for feeding an array as input
    assert "{:.3f}".format(fn_output[4]) == "0.579"
    # Test for scalar values as input
    assert "{:.3f}".format(fn(0.4)) == "0.579"
    assert "{:.3f}".format(fn(0.1)) == "0.000"


@pytest.mark.algorithms
@pytest.mark.parametrize("func_constructor, xval, expected", [
    (func_data_rayleigh, 0.0, "0.000"),
    (func_data_rayleigh, 0.0, "0.000"),
    (func_data_rayleigh, 6.0, "0.338"),
    (func_data_rayleigh, 12.0, "1.000"),
    (func_data_piecewise2, 0.0, "0.000"),
    (func_data_piecewise2, 0.2, "0.000"),
    (func_data_piecewise2, 0.7, "1.000"),
    (func_data_piecewise2, 1.2, "3.000")
])
def test_algorithm_mdls(func_constructor, xval, expected):
    fn = Algorithm.factory(func_constructor)
    assert "{:.3f}".format(fn(xval)) == expected


# ==============================================================================
# Data for model fitting

hazard_scenarios = [
    '0.000', '0.100', '0.200', '0.300', '0.400',
    '0.500', '0.600', '0.700', '0.800', '0.900', '1.000']

sys_limit_states = [
    'DS0 None', 'DS1 Slight', 'DS2 Moderate', 'DS3 Extensive', 'DS4 Complete']

data_for_fitting_no_xover = [
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    [0.00, 0.45, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    [0.00, 0.00, 0.75, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.17, 0.73, 0.96, 1.00, 1.00, 1.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 0.19]
]

data_for_fitting_with_xover = [
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    [0.00, 0.45, 0.70, 0.90, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    [0.00, 0.30, 0.78, 0.85, 0.90, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    [0.00, 0.00, 0.00, 0.17, 0.95, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 0.19]
]

config_data_dict = {
    'model_name': 'System X',
    'x_param': 'PGA',
    'x_unit': 'g',
    'scenario_metrics': ['0.500'],
    'scneario_names': ['Sc A']
}

TEMP_OUTPUT = Path("./temp/output")

# ------------------------------------------------
# Test model fitting, for data with no crossover
# ------------------------------------------------
@pytest.mark.modelfitting
@pytest.mark.parametrize(
    argnames="distribution",
    argvalues=[
        'lognormal_cdf',
        'normal_cdf',
        'rayleigh_cdf'
    ]
)
def test_model_fitting_no_crossover(distribution):
    if not TEMP_OUTPUT.is_dir():
        TEMP_OUTPUT.mkdir(parents=True, exist_ok=True)
    fitted_params_dict = fit_prob_exceed_model(
        hazard_scenarios,
        data_for_fitting_no_xover,
        sys_limit_states,
        config_data_dict,
        output_path=TEMP_OUTPUT,
        distribution=distribution
    )
    assert fitted_params_dict[1]['fit_statistics']['chisqr'] <= 0.1
    assert fitted_params_dict[2]['fit_statistics']['chisqr'] <= 0.1
    shutil.rmtree(TEMP_OUTPUT)

# ------------------------------------------------
# Test model fitting, for data WITH CROSSOVER
# ------------------------------------------------
@pytest.mark.modelfitting
@pytest.mark.parametrize(
    argnames="distribution",
    argvalues=[
        'lognormal_cdf',
        'normal_cdf',
        'rayleigh_cdf'
    ]
)
def test_model_fitting_with_crossover(distribution):
    if not TEMP_OUTPUT.is_dir():
        TEMP_OUTPUT.mkdir(parents=True, exist_ok=True)
    fitted_params_dict = fit_prob_exceed_model(
        hazard_scenarios,
        data_for_fitting_with_xover,
        sys_limit_states,
        config_data_dict,
        output_path=TEMP_OUTPUT,
        distribution=distribution
    )
    assert fitted_params_dict[1]['fit_statistics']['chisqr'] <= 0.1
    assert fitted_params_dict[2]['fit_statistics']['chisqr'] <= 0.1
    shutil.rmtree(TEMP_OUTPUT)


if __name__ == '__main__':
    pytest.main([__file__])
