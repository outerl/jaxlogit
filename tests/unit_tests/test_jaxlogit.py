import pytest
import numpy as np

import jax
import jax.numpy as jnp
import re

from jaxlogit.mixed_logit import (
    MixedLogit,
    _transform_rand_betas,
    loglike_individual,
    neg_loglike,
    probability_individual,
)
from jaxlogit._variables import ParametersSetup
from jaxlogit._config_data import ConfigData

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


def test_no_random_variables_draws(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data
    varnames = [f"x{i}" for i in range(X.shape[1])]

    model = MixedLogit()
    randvars = {}
    config = ConfigData(
        avail=avail,
        panels=panels,
        weights=weights,
        n_draws=3,
        optim_method="L-BFGS-scipy",
        init_coeff=None,
        skip_std_errs=True,
    )
    result = model.fit(X, y, varnames, alts, ids, randvars, config)
    assert result is not None
    assert result.fun is not None

    predict_config = ConfigData(
        avail=avail,
        panels=panels,
        weights=weights,
        init_coeff=result.x,
    )
    probs = model.predict(X, varnames, alts, ids, randvars, predict_config)
    assert probs.shape == (X.shape[0] / X.shape[1], X.shape[1])  # this is true for non-panel data
    assert not jnp.any(jnp.isnan(probs))
    assert not jnp.any(jnp.isinf(probs))
    assert not jnp.any(jnp.isneginf(probs))


def test_no_random_variables(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data
    varnames = [f"x{i}" for i in range(X.shape[1])]

    model = MixedLogit()
    randvars = {}
    config = ConfigData(
        avail=avail,
        panels=panels,
        weights=weights,
        optim_method="L-BFGS-jax",
        init_coeff=None,
        skip_std_errs=True,
    )
    result = model.fit(X, y, varnames, alts, ids, randvars, config)
    assert result is not None
    assert result.fun is not None

    predict_config = ConfigData(
        avail=avail,
        panels=panels,
        weights=weights,
        init_coeff=result.x,
    )
    probs = model.predict(X, varnames, alts, ids, randvars, predict_config)
    assert probs.shape == (X.shape[0] / X.shape[1], X.shape[1])  # this is true for non-panel data
    assert not jnp.any(jnp.isnan(probs))
    assert not jnp.any(jnp.isinf(probs))
    assert not jnp.any(jnp.isneginf(probs))


def test_bad_random_variables(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data
    varnames = [f"x{i}" for i in range(X.shape[1])]

    model = MixedLogit()
    randvars = {varnames[0]: "n", varnames[1]: "fake"}
    config = ConfigData(
        avail=avail,
        panels=panels,
        weights=weights,
        n_draws=3,
        optim_method="L-BFGS-scipy",
        init_coeff=None,
        skip_std_errs=True,
    )
    with pytest.raises(
        ValueError, match="Wrong mixing distribution in 'randvars'. Accepted distrubtions are n, ln, t, u, n_trunc"
    ):
        model.fit(X, y, varnames, alts, ids, randvars, config)


def test_mixed_logit_fit_different_variables(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data
    rand_var_type_combos = [
        ("n", "n", "n"),
        ("n", "ln", "ln"),
        ("n_trunc", "n", "ln"),
        ("n_trunc", "n_trunc", "n_trunc"),
        ("n_trunc", None, None),
        (None, None, None),
        ("n", None, None),
        ("n", "ln", None),
    ]
    varnames = [f"x{i}" for i in range(X.shape[1])]

    for include_correlations in [True, False]:
        for rand_var_types in rand_var_type_combos:
            number_normal_and_lognormal = rand_var_types.count("n") + rand_var_types.count("ln")

            model = MixedLogit()
            randvars = {varnames[i]: rand_var_types[i] for i in range(len(rand_var_types)) if rand_var_types[i]}
            set_vars = {}
            config = ConfigData(
                avail=avail,
                panels=panels,
                weights=weights,
                n_draws=3,
                set_vars=set_vars,
                optim_method="L-BFGS-scipy",
                init_coeff=None,
                include_correlations=include_correlations,
                skip_std_errs=True,
            )
            if include_correlations and number_normal_and_lognormal < 2:
                with pytest.raises(
                    ValueError,
                    match=re.escape(
                        f"Only {number_normal_and_lognormal} normal based variable(s). Cannot use correlation"
                    ),
                ):
                    result = model.fit(X, y, varnames, alts, ids, randvars, config)
                continue

            result = model.fit(X, y, varnames, alts, ids, randvars, config)

            assert result is not None
            assert result.fun is not None
            number_normal_and_lognormal = rand_var_types.count("n") + rand_var_types.count("ln")
            assert (
                len(result.x)
                == 2 * len(rand_var_types)
                - rand_var_types.count(None)
                + number_normal_and_lognormal * (number_normal_and_lognormal - 1) / 2 * include_correlations
            )


def test_mixed_logit_fit_no_panels_weights(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data

    model = MixedLogit()
    varnames = [f"x{i}" for i in range(X.shape[1])]
    rand_var_types = ("n_trunc", "n", "ln")
    randvars = {varnames[i]: rand_var_types[i] for i in range(len(rand_var_types)) if rand_var_types[i]}
    set_vars = {}

    no_weights_or_panel_config = ConfigData(
        avail=avail, n_draws=3, set_vars=set_vars, optim_method="L-BFGS-scipy", init_coeff=None, skip_std_errs=False
    )
    result = model.fit(X, y, varnames, alts, ids, randvars, no_weights_or_panel_config)
    assert result is not None
    assert result.fun is not None


def test_mixed_logit_fit_set_variables(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data
    varnames = [f"x{i}" for i in range(X.shape[1])]

    randvars = {varnames[0]: "n"}
    set_vars = {varnames[1]: 0.0}
    config = ConfigData(
        avail=avail,
        panels=panels,
        weights=weights,
        n_draws=3,
        set_vars=set_vars,
        optim_method="L-BFGS-scipy",
        init_coeff=None,
        skip_std_errs=True,
    )
    model = MixedLogit()
    result = model.fit(X, y, varnames, alts, ids, randvars, config)
    assert result is not None
    assert result.fun is not None
    assert len(result.x) == 4  # two from normal distribution, two from un-parameterised variables


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

    set_vars = {}
    config = ConfigData(
        avail=np.array(df["avail"]),
        panels=np.array(df["person_id_contiguous"]),
        weights=np.array(df["weight"]),
        n_draws=3,
        set_vars=set_vars,
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


def test_no_random_variables_loglikes(simple_data):
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
    randvars = {}

    set_vars = {}
    config = ConfigData(
        avail=np.array(df["avail"]),
        panels=np.array(df["person_id_contiguous"]),
        weights=np.array(df["weight"]),
        n_draws=3,
        set_vars=set_vars,
        init_coeff=None,
        include_correlations=False,
    )
    (betas, Xdf, Xdr, panels, weights, avail, num_panels, coef_names, draws, parameter_info) = model.data_prep(
        df[varnames], df["choice"], varnames, df["alt"], df["custom_id"], randvars, config
    )

    ll_indiv = loglike_individual(betas, Xdf, Xdr, panels, weights, avail, num_panels, False, draws, parameter_info)
    assert not jnp.any(jnp.isnan(ll_indiv))
    assert not jnp.any(jnp.isinf(ll_indiv))
    assert not jnp.any(jnp.isneginf(ll_indiv))


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
    set_vars = {}
    config = ConfigData(
        avail=np.array(df["avail"]),
        panels=np.array(df["person_id_contiguous"]),
        weights=np.array(df["weight"]),
        n_draws=3,
        set_vars=set_vars,
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
