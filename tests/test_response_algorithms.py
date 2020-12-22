import pytest
import numpy as np
from sira.modelling.responsemodels import Algorithm

# ------------------------------------------------------------------------------
# TEST DATA

func_data_piecewise1 = {
    "function_name": "PiecewiseFunction",
    "piecewise_function_constructor": [
        {
            "function_name": "normal",
            "mean": 0.3,
            "stddev": 0.5,
            "fragility_source": "source mysterio",
            "minimum": 'null',
            "lower_limit": 0.2,
            "upper_limit": 0.5,
            "damage_state_definition": "Not Available."
        },
        {
            "function_name": "normal",
            "mean": 0.3,
            "stddev": 0.7,
            "fragility_source": "source mysterio",
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
            "fragility_source": "source mysterio",
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
            "fragility_source": "source mysterio",
            "minimum": 'null',
            "lower_limit": 0.3,
            "upper_limit": 0.85,
            "damage_state_definition": "Not Available."
        },
        {
            "function_name": "ConstantFunction",
            "amplitude": 3,
            "fragility_source": "source mysterio",
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
    "fragility_source": "Not Available.",
    "damage_state_definition": "Not Available."
}

# ------------------------------------------------------------------------------

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
    ("func_data_rayleigh", 0.0, "0.000"),
    ("func_data_rayleigh", 0.0, "0.000"),
    ("func_data_rayleigh", 6.0, "0.338"),
    ("func_data_rayleigh", 12.0, "1.000"),
    ("func_data_piecewise2", 0.0, "0.000"),
    ("func_data_piecewise2", 0.2, "0.000"),
    ("func_data_piecewise2", 0.7, "1.000"),
    ("func_data_piecewise2", 1.2, "3.000")
])
def test_algorithm_mdls(func_constructor, xval, expected):
    func_params = eval(func_constructor)
    fn = Algorithm.factory(func_params)
    assert "{:.3f}".format(fn(xval)) == expected
