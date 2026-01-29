from jaxlogit._optimize import hessian, _minimize
from jaxlogit.mixed_logit import MixedLogit, ConfigData, neg_loglike, neg_loglike_grad_batched
from jaxlogit.MixedLogitEncoder import optim_res_decoder
import numpy as np
import pandas as pd
import jax as jax
import jax.numpy as jnp

import pytest
import pathlib
import json


def setup_minimize():
    """Performs the same code as in mixed_logit.predict, but stopping after minimize"""
    df = pd.read_csv(pathlib.Path(__file__).parent.parent.parent / "examples/electricity_long.csv")
    varnames = ["pf", "cl", "loc", "wk", "tod", "seas"]
    n_draws = 600
    X = df[varnames]
    y = df["choice"]

    ids = df["chid"]
    alts = df["alt"]
    panels = df["id"]

    randvars = {"pf": "n", "cl": "n", "loc": "n", "wk": "n", "tod": "n", "seas": "n"}

    model = MixedLogit()

    config = ConfigData(
        panels=panels,
        n_draws=n_draws,
        skip_std_errs=True,  # skip standard errors to speed up the example
        batch_size=None,
        optim_method="L-BFGS-B",
    )

    (betas, Xdf, Xdr, panels, weights, avail, num_panels, coef_names, draws, parameter_info, _) = model.data_prep(
        X,
        y,
        varnames,
        alts,
        ids,
        randvars,
        config,
        None,
    )

    fargs = (
        Xdf,
        Xdr,
        panels,
        weights,
        avail,
        num_panels,
        config.force_positive_chol_diag,
        draws,
        parameter_info,
        config.batch_size,
    )

    tol = {
        "ftol": 1e-10,
        "gtol": 1e-6,
    }
    if config.tol_opts is not None:
        tol.update(config.tol_opts)

    fct_to_optimize = neg_loglike if config.batch_size is None else neg_loglike_grad_batched

    optim_res = _minimize(
        fct_to_optimize,
        betas,
        args=fargs,
        method=config.optim_method,
        tol=tol["ftol"],
        options={
            "gtol": tol["gtol"],
            "maxiter": config.maxiter,
        },
        jit_loglik=config.batch_size is None,
    )
    return optim_res


def test__minimize():
    # expected values based on iteration comparable to other models
    with open(pathlib.Path(__file__).parent / "test_data" / "optimize_minimize_output.json", "r") as f:
        expected = json.load(f, object_hook=optim_res_decoder)
    actual = setup_minimize()
    assert expected["message"] == actual["message"]
    assert expected["success"] == actual["success"]
    assert len(expected["x"]) == len(actual.x)
    for i in range(len(expected["x"])):
        assert pytest.approx(expected["x"][i], rel=1e-2) == actual.x[i]
    assert pytest.approx(expected["fun"], rel=1e-3) == actual["fun"]
    assert len(expected["jac"]) == len(actual.jac)

    expected_hi = expected["hess_inv"]
    assert len(expected_hi["sk"]) == len(actual.hess_inv.sk)
    assert len(expected_hi["yk"]) == len(actual.hess_inv.yk)
    assert len(expected_hi["rho"]) == len(actual.hess_inv.rho)


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
    assert expected == pytest.approx(hessian(test_function, x, False, False, args))  # not hessian_by_row
    assert expected == pytest.approx(hessian(test_function, x, True, False, args))  # hessian_by_row


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
    assert expected == pytest.approx(hessian(test_function, x, False, True, args), rel=5e-2)
