from jaxlogit._optimize import hessian
import numpy as np
import pandas as pd
import jax as jax
import jax.numpy as jnp
import pytest


def test_hessian_no_finite_diff():
    # x is the betas
    def test_function(x, a, b, c, force_positive_chol_diag, num_panels):
        return a ** x[0] + b ** x[1] + a / c + x[2] ** 5

    a = 5.0
    b = 2.0
    c = 3.0
    fpcd = True
    num_panels = 3
    args = (a, b, c, fpcd, num_panels)
    x = np.repeat(0.1, 3)
    expected = np.array([np.array([3.0426044, 0.0, 0.0]), np.array([0.0, 0.5149368, 0.0]), np.array([0.0, 0.0, 0.02])])
    assert expected == pytest.approx(hessian(test_function, x, False, False, *args))  # not hessian_by_row
    assert expected == pytest.approx(hessian(test_function, x, True, False, *args))  # hessian_by_row


def test_hessian_finite_diff():
    def test_function(x, a, b, c, force_positive_chol_diag, num_panels):
        return a ** x[0] + b ** x[1] + a / c + x[2] ** 5

    a = 5.0
    b = 2.0
    c = 3.0
    fpcd = True
    num_panels = 3
    args = (a, b, c, fpcd, num_panels)
    x = jnp.array([0.1, 0.1, 0.1])
    expected = np.array([np.array([3.0426044, 0.0, 0.0]), np.array([0.0, 0.5149368, 0.0]), np.array([0.0, 0.0, 0.02])])
    assert expected == pytest.approx(hessian(test_function, x, False, True, *args), rel=6e-2)  # finite_diff
