import pytest

import jax.numpy as jnp
import numpy as np

from jaxlogit._variables import ParametersSetup
from jaxlogit._config_data import ConfigData


def test_wrong_set_variables():
    missing_set_variable_config = ConfigData(set_vars={"fake_variable": 1})

    with pytest.raises(ValueError, match=r"Variable fake_variable not found in the model."):
        # rvdist, rvidx, rvidx_normal_bases, rvidx_truncnorm_based, coef_names, betas, config: ConfigData
        ParametersSetup(
            np.array(["n"]),
            jnp.array([True]),
            jnp.array([True]),
            jnp.array([False]),
            np.array(["real_variable"]),
            jnp.array([0.1]),
            missing_set_variable_config,
        )


def test_not_enough_variables_for_correlation():
    correlation_only_config = ConfigData(include_correlations=True)

    with pytest.raises(ValueError, match=r"Only [01] normal based variable\(s\)\. Cannot use correlation"):
        # rvdist, rvidx, rvidx_normal_bases, rvidx_truncnorm_based, coef_names, betas, config: ConfigData
        variable_types_groups = [
            ["n"],
            ["trunc_n", "trunc_n", "trunc_n"],
            ["ln", "trunc_n", "trunc_n"],
        ]
        for variable_types in variable_types_groups:
            ParametersSetup(
                np.array(variable_types),
                jnp.array([True for _ in variable_types]),
                jnp.array([var == "n" or var == "ln" for var in variable_types]),
                jnp.array([var == "trunc_n" for var in variable_types]),
                np.array([f"x{i}" for i in enumerate(variable_types)]),
                jnp.array([0.1 for _ in variable_types]),
                correlation_only_config,
            )
