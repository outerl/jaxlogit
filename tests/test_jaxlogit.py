import pytest
import numpy as np

import jax
import jax.numpy as jnp
import pickle

from jaxlogit._variables import ParametersSetup
from jaxlogit._config_data import ConfigData

from jaxlogit.mixed_logit import (
    MixedLogit,
    _transform_rand_betas,
    loglike_individual,
    neg_loglike,
    probability_individual,
)

SEED = 0


def make_simple_data():
    N = 6  # individuals
    J = 3  # alternatives
    K = 3  # variables
    np.random.seed(SEED)
    X = np.random.randn(N * J, K)
    # y = np.random.randint(0, 2, size=(N * J,))
    y = np.zeros((N, J))
    y[:, 0] = 1
    _ = [np.random.shuffle(x) for x in list(y)]
    y = y.reshape(-1)
    ids = np.repeat(np.arange(N), J)
    alts = np.tile(np.arange(J), N)
    avail = np.ones((N * J,))
    panels = np.repeat(np.arange(N), J)
    weights = np.ones(N * J)
    return X, y, ids, alts, avail, panels, weights


@pytest.fixture
def simple_data():
    return make_simple_data()


def test_mixed_logit_fit(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data

    varnames = [f"x{i}" for i in range(X.shape[1])]

    model = MixedLogit()
    randvars = {varnames[0]: "n", varnames[1]: "ln", varnames[2]: "n_trunc"}
    fixedvars = {}
    config = ConfigData(
        avail=avail,
        panels=panels,
        weights=weights,
        n_draws=3,
        fixedvars=fixedvars,
        optim_method="L-BFGS-B",
        init_coeff=None,
        skip_std_errs=True,
    )
    result = model.fit(X, y, varnames, alts, ids, randvars, config)

    assert result is not None
    assert "fun" in result


@pytest.mark.skip(reason="different python versions and OSs give different results")
def test_mixed_logit_fit_against_previous_results(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data

    varnames = [f"x{i}" for i in range(X.shape[1])]

    model = MixedLogit()
    randvars = {varnames[0]: "n"}
    fixedvars = {}

    config = ConfigData(
        skip_std_errs=True,
        init_coeff=None,
        optim_method="L-BFGS-B",
        fixedvars=fixedvars,
        n_draws=3,
        weights=weights,
        panels=panels,
        avail=avail,
    )

    model.fit(X=X, y=y, varnames=varnames, ids=ids, alts=alts, randvars=randvars, config=config)

    with open("tests/simple_data_output.pkl", "rb") as f:
        previous_model = pickle.load(f)

    # assert list(model.coeff_names) == list(previous_model.coeff_names)
    assert list(model.coeff_) == pytest.approx(list(previous_model.coeff_), rel=1e-3)
    # assert list(model.stderr) == pytest.approx(list(previous_model.stderr), rel=1e-3)
    # assert list(model.zvalues) == pytest.approx(list(previous_model.zvalues), rel=1e-3)
    assert model.loglikelihood == pytest.approx(previous_model.loglikelihood)
    # could also add model.loglikelihood, model.aic and model.bic


def test_loglike_individual_and_total(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data
    varnames = [f"x{i}" for i in range(X.shape[1])]
    import pandas as pd

    df = pd.DataFrame({name: X[:, i] for i, name in enumerate(varnames)})
    df["choice"] = y
    df["custom_id"] = ids
    df["alt"] = alts
    df["avail"] = avail
    df["person_id_contiguous"] = panels
    df["weight"] = weights

    model = MixedLogit()
    randvars = {varnames[0]: "n"}

    fixedvars = {}
    config = ConfigData(
        avail=np.array(df["avail"]),
        panels=np.array(df["person_id_contiguous"]),
        weights=np.array(df["weight"]),
        n_draws=3,
        fixedvars=fixedvars,
        init_coeff=None,
        include_correlations=False,
    )
    (betas, Xdf, Xdr, panels, weights, avail, num_panels, coef_names, draws, parameter_info) = model.data_prep(
        df[varnames], df["choice"], varnames, df["alt"], df["custom_id"], randvars, config
    )

    ll_indiv = loglike_individual(betas, Xdf, Xdr, panels, weights, avail, num_panels, False, draws, parameter_info)
    assert ll_indiv.shape[0] == num_panels
    assert not jnp.any(jnp.isnan(ll_indiv))

    nll = neg_loglike(betas, Xdf, Xdr, panels, weights, avail, num_panels, False, draws, parameter_info, 0)
    assert np.isscalar(nll) or (isinstance(nll, jnp.ndarray) and nll.shape == ())
    assert np.allclose(-nll, jnp.sum(ll_indiv), atol=1e-5)


def test_probability_individual(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data
    varnames = [f"x{i}" for i in range(X.shape[1])]
    import pandas as pd

    df = pd.DataFrame({name: X[:, i] for i, name in enumerate(varnames)})
    df["choice"] = y
    df["custom_id"] = ids
    df["alt"] = alts
    df["avail"] = avail
    df["person_id_contiguous"] = panels
    df["weight"] = weights

    model = MixedLogit()
    randvars = {varnames[0]: "n"}
    fixedvars = {}
    config = ConfigData(
        avail=np.array(df["avail"]),
        panels=np.array(df["person_id_contiguous"]),
        weights=np.array(df["weight"]),
        n_draws=3,
        fixedvars=fixedvars,
        init_coeff=None,
        include_correlations=False,
    )
    (betas, Xdf, Xdr, panels, weights, avail, num_panels, coef_names, draws, parameter_info) = model.data_prep(
        df[varnames], df["choice"], varnames, df["alt"], df["custom_id"], randvars, config
    )

    probs = probability_individual(betas, Xdf, Xdr, panels, weights, avail, num_panels, False, draws, parameter_info)
    assert probs.shape[0] == Xdf.shape[0]
    assert not jnp.any(jnp.isnan(probs))


def test_transform_rand_betas_shapes():
    Kr = 3
    N = 4
    R = 3
    betas = jnp.arange(Kr + Kr + Kr * (Kr + 1) // 2, dtype=float)
    draws = jnp.ones((N, Kr, R))
    rvdist = np.array(["n", "ln", "n_trunc"])
    rvidx = jnp.array([True, True, True])
    rvidx_normal_bases = jnp.array([True, True, False])
    rvidx_truncnorm_based = jnp.array([False, False, True])
    coef_names = np.array(["normal_variable", "lognormal_variable", "truncated_normal_variable"])
    config = ConfigData(include_correlations=False, force_positive_chol_diag=False)
    parameter_info_no_correlation = ParametersSetup(
        rvdist, rvidx, rvidx_normal_bases, rvidx_truncnorm_based, coef_names, betas, config
    )

    config_corellation = ConfigData(include_correlations=True, force_positive_chol_diag=True)
    parameter_info_correlation = ParametersSetup(
        rvdist, rvidx, rvidx_normal_bases, rvidx_truncnorm_based, coef_names, betas, config_corellation
    )

    out = _transform_rand_betas(betas, False, draws, parameter_info_no_correlation)
    assert out.shape == (N, Kr, R)
    out_corr = _transform_rand_betas(betas, True, draws, parameter_info_correlation)
    assert out_corr.shape == (N, Kr, R)


def test_transform_rand_betas_jit():
    Kr = 2
    N = 3
    R = 2
    betas = jnp.arange(Kr + Kr + Kr * (Kr + 1) // 2, dtype=float)
    draws = jnp.ones((N, Kr, R))
    rvdist = np.array(["n", "n"])
    rvidx = jnp.array([True, True])
    rvidx_normal_bases = jnp.array([True, True])
    rvidx_truncnorm_based = jnp.array([False, False])
    coef_names = np.array(["testvar1", "testvar2"])

    config_corellation = ConfigData(include_correlations=True, force_positive_chol_diag=True)
    parameter_info_correlation = ParametersSetup(
        rvdist, rvidx, rvidx_normal_bases, rvidx_truncnorm_based, coef_names, betas, config_corellation
    )
    fn = jax.jit(_transform_rand_betas, static_argnames=["parameter_info", "force_positive_chol_diag"])
    out = fn(betas, True, draws, parameter_info_correlation)
    assert out.shape == (N, Kr, R)


def save_simple_data_output():
    X, y, ids, alts, avail, panels, weights = make_simple_data()

    varnames = [f"x{i}" for i in range(X.shape[1])]

    model = MixedLogit()
    randvars = {varnames[0]: "n"}
    fixedvars = {}

    config = ConfigData(
        skip_std_errs=True,
        init_coeff=None,
        optim_method="L-BFGS-B",
        fixedvars=fixedvars,
        n_draws=3,
        weights=weights,
        panels=panels,
        avail=avail,
    )

    model.fit(X=X, y=y, varnames=varnames, ids=ids, alts=alts, randvars=randvars, config=config)

    with open("tests/simple_data_output.pkl", "wb") as f:
        pickle.dump(model, f)


def main():
    save_simple_data_output()


if __name__ == "__main__":
    main()
