import pytest

import jax.numpy as jnp
import numpy as np

from jaxlogit._variables import ParametersSetup
from jaxlogit._config_data import ConfigData
from jaxlogit.mixed_logit import MixedLogit


def test_wrong_set_variables():
    missing_set_variable_config = ConfigData(set_vars={"real_variable": 1})

    with pytest.raises(ValueError, match=r"Variable real_variable not found in the model."):
        # rvdist, rvidx, rvidx_normal_bases, rvidx_truncnorm_based, coef_names, betas, config: ConfigData
        m = MixedLogit()

        m._setup_randvars_info([""], np.array([]))
        ParametersSetup(
            np.array(["n"]),
            jnp.array([True]),
            jnp.array([True]),
            jnp.array([False]),
            np.array(["fake_variable"]),
            jnp.array([0.1]),
            missing_set_variable_config,
        )


if __name__ == "__main__":
    m = MixedLogit()
    print(1)
