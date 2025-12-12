import pytest

import jax.numpy as jnp

from jaxlogit._variables import ParametersSetup
from jaxlogit._config_data import ConfigData


def test_wrong_set_variables():
    missing_set_variable_config = ConfigData(fixedvars={"fake_variable": 1})
    with pytest.raises(ValueError):
        ParametersSetup(
            jnp.array(["n"]),
            jnp.array([0]),
            jnp.array([True]),
            jnp.array([False]),
            ["real_variable"],
            [0.1],
            missing_set_variable_config,
        )
