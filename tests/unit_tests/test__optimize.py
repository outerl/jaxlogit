from jaxlogit._optimize import hessian, gradient
import numpy as np
import jax as jax
import jax.numpy as jnp
import pytest


# TODO: minimise tests once mixed_logit is sorted out
def test_hessian_no_finite_diff():
    def test_function(x, a, b, c, dummy_1, dummy_2):
        return a ** x[0] + b ** x[1] + a / c + x[2] ** 5

    a = 5.0
    b = 2.0
    c = 3.0
    dummy_1 = 4
    dummy_2 = (1, 2, 3)

    args = (a, b, c, dummy_1, dummy_2)
    x = np.repeat(0.1, 3)
    expected = np.array([np.array([3.0426044, 0.0, 0.0]), np.array([0.0, 0.5149368, 0.0]), np.array([0.0, 0.0, 0.02])])
    assert expected == pytest.approx(
        hessian(test_function, x, False, False, *args, static_argnames=("dummy_1", "dummy_2"))
    )  # not hessian_by_row
    assert expected == pytest.approx(
        hessian(test_function, x, True, False, *args, static_argnames=("dummy_1", "dummy_2"))
    )  # hessian_by_row


def test_hessian_finite_diff():
    def test_function(x, a, b, c, dummy_1, dummy_2):
        return a ** x[0] + b ** x[1] + a / c + x[2] ** 5

    a = 5.0
    b = 2.0
    c = 3.0
    dummy_1 = 4
    dummy_2 = (1, 2, 3)

    args = (a, b, c, dummy_1, dummy_2)
    x = jnp.array([0.1, 0.1, 0.1])
    expected = np.array([np.array([3.0426044, 0.0, 0.0]), np.array([0.0, 0.5149368, 0.0]), np.array([0.0, 0.0, 0.02])])
    assert expected == pytest.approx(
        hessian(test_function, x, False, True, *args, static_argnames=("dummy_1", "dummy_2")), rel=5e-2
    )
