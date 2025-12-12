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


def make_empty_data():
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


@pytest.fixture
def simple_data():
    return make_simple_data()


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
        optim_method="L-BFGS-B",
        init_coeff=None,
        skip_std_errs=True,
    )
    with pytest.raises(ValueError):
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
    for include_correlations in [True, False]:
        for rand_var_types in rand_var_type_combos:
            number_normal_and_lognormal = rand_var_types.count("n") + rand_var_types.count("ln")
            varnames = [f"x{i}" for i in range(X.shape[1])]

            model = MixedLogit()
            randvars = {varnames[i]: rand_var_types[i] for i in range(len(rand_var_types)) if rand_var_types[i]}
            set_vars = {}
            config = ConfigData(
                avail=avail,
                panels=panels,
                weights=weights,
                n_draws=3,
                set_vars=set_vars,
                optim_method="L-BFGS-B",
                init_coeff=None,
                include_correlations=include_correlations,
                skip_std_errs=True,
            )
            if include_correlations and number_normal_and_lognormal < 2:
                with pytest.raises(ValueError):
                    result = model.fit(X, y, varnames, alts, ids, randvars, config)
                continue

            result = model.fit(X, y, varnames, alts, ids, randvars, config)

            assert result is not None
            assert "fun" in result
            number_normal_and_lognormal = rand_var_types.count("n") + rand_var_types.count("ln")
            assert (
                len(result.x)
                == 2 * len(rand_var_types)
                - rand_var_types.count(None)
                + number_normal_and_lognormal * (number_normal_and_lognormal - 1) / 2 * include_correlations
            )

    # config = ConfigData(
    #     avail=avail,
    #     panels=panels,
    #     weights=weights,
    #     n_draws=3,
    #     set_vars=set_vars,
    #     optim_method="L-BFGS-B",
    #     init_coeff=None,
    #     skip_std_errs=True,
    #     include_correlations=True,
    # )
    # result = model.fit(X, y, varnames, alts, ids, randvars, config)

    # assert result is not None
    # assert "fun" in result

    # no_weights_or_panel_config = ConfigData(
    #     avail=avail, n_draws=3, set_vars=set_vars, optim_method="L-BFGS-B", init_coeff=None, skip_std_errs=False
    # )
    # result = model.fit(X, y, varnames, alts, ids, randvars, no_weights_or_panel_config)
    # assert result is not None
    # assert "fun" in result

    # randvars = {varnames[0]: "n"}
    # set_vars = {varnames[1]: 0.0}
    # config = ConfigData(
    #     avail=avail,
    #     panels=panels,
    #     weights=weights,
    #     n_draws=3,
    #     set_vars=set_vars,
    #     optim_method="L-BFGS-B",
    #     init_coeff=None,
    #     skip_std_errs=True,
    # )
    # result = model.fit(X, y, varnames, alts, ids, randvars, config)
    # assert result is not None
    # assert "fun" in result
    # assert len(result.x) == 4  # two from normal distribution, two from un-parameterised variables


def test_2mixed_logit_fit():
    X, y, ids, alts, avail, panels, weights = make_simple_data()
    include_correlations = False
    rand_var_types = ("n", None, None)
    number_normal_and_lognormal = rand_var_types.count("n") + rand_var_types.count("ln")
    if number_normal_and_lognormal < 2:
        include_correlations = False
    varnames = [f"x{i}" for i in range(X.shape[1])]

    model = MixedLogit()
    randvars = {varnames[i]: rand_var_types[i] for i in range(len(rand_var_types)) if rand_var_types[i]}
    set_vars = {}
    config = ConfigData(
        avail=avail,
        panels=panels,
        weights=weights,
        n_draws=3,
        set_vars=set_vars,
        optim_method="L-BFGS-B",
        init_coeff=None,
        include_correlations=include_correlations,
        skip_std_errs=True,
    )
    result = model.fit(X, y, varnames, alts, ids, randvars, config)

    assert result is not None
    assert "fun" in result
    number_normal_and_lognormal = rand_var_types.count("n") + rand_var_types.count("ln")
    assert (
        len(result.x)
        == 2 * len(rand_var_types)
        - rand_var_types.count(None)
        + number_normal_and_lognormal * (number_normal_and_lognormal - 1) / 2 * include_correlations
    )


@pytest.mark.skip(reason="different python versions and OSs give different results")
def test_mixed_logit_fit_against_previous_results(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data

    varnames = [f"x{i}" for i in range(X.shape[1])]

    model = MixedLogit()
    randvars = {varnames[0]: "n"}
    set_vars = {}

    config = ConfigData(
        skip_std_errs=True,
        init_coeff=None,
        optim_method="L-BFGS-B",
        set_vars=set_vars,
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


def save_simple_data_output():
    X, y, ids, alts, avail, panels, weights = make_simple_data()

    varnames = [f"x{i}" for i in range(X.shape[1])]

    model = MixedLogit()
    randvars = {varnames[0]: "n"}
    set_vars = {}

    config = ConfigData(
        skip_std_errs=True,
        init_coeff=None,
        optim_method="L-BFGS-B",
        set_vars=set_vars,
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
