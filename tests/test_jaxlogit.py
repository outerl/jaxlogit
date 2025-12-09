import pytest
import numpy as np

import jax
import jax.numpy as jnp
import json

from jaxlogit.mixed_logit import (
    MixedLogit,
    ConfigData,
    _transform_rand_betas,
    loglike_individual,
    neg_loglike,
    probability_individual,
)

SEED = 0


def make_simple_data():
    N = 6  # individuals
    J = 3  # alternatives
    K = 2  # variables
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
    randvars = {varnames[0]: "n"}
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

    with open("tests/simple_data_output.json", "r") as f:
        previous_model = json.load(f)

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
    (
        betas,
        Xdf,
        Xdr,
        panels,
        draws,
        weights,
        avail,
        mask,
        values_for_mask,
        mask_chol,
        values_for_chol_mask,
        rand_idx_norm,
        rand_idx_truncnorm,
        draws_idx_norm,
        draws_idx_truncnorm,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        coef_names,
        rand_idx_stddev,
        rand_idx_chol,
    ) = model.data_prep(df[varnames], df["choice"], varnames, df["alt"], df["custom_id"], randvars, config)

    ll_indiv = loglike_individual(
        betas,
        Xdf,
        Xdr,
        panels,
        draws,
        weights,
        avail,
        mask,
        values_for_mask,
        mask_chol,
        values_for_chol_mask,
        rand_idx_norm,
        rand_idx_truncnorm,
        draws_idx_norm,
        draws_idx_truncnorm,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        False,
        rand_idx_stddev,
        rand_idx_chol,
    )
    assert ll_indiv.shape[0] == num_panels
    assert not jnp.any(jnp.isnan(ll_indiv))

    nll = neg_loglike(
        betas,
        Xdf,
        Xdr,
        panels,
        draws,
        weights,
        avail,
        mask,
        values_for_mask,
        mask_chol,
        values_for_chol_mask,
        rand_idx_norm,
        rand_idx_truncnorm,
        draws_idx_norm,
        draws_idx_truncnorm,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        False,
        rand_idx_stddev,
        rand_idx_chol,
        0,
    )
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
    (
        betas,
        Xdf,
        Xdr,
        panels,
        draws,
        weights,
        avail,
        mask,
        values_for_mask,
        mask_chol,
        values_for_chol_mask,
        rand_idx_norm,
        rand_idx_truncnorm,
        draws_idx_norm,
        draws_idx_truncnorm,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        coef_names,
        rand_idx_stddev,
        rand_idx_chol,
    ) = model.data_prep(df[varnames], df["choice"], varnames, df["alt"], df["custom_id"], randvars, config)

    probs = probability_individual(
        betas,
        Xdf,
        Xdr,
        panels,
        draws,
        weights,
        avail,
        mask,
        values_for_mask,
        mask_chol,
        values_for_chol_mask,
        rand_idx_norm,
        rand_idx_truncnorm,
        draws_idx_norm,
        draws_idx_truncnorm,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        False,
        rand_idx_stddev,
        rand_idx_chol,
    )
    assert probs.shape[0] == Xdf.shape[0]
    assert not jnp.any(jnp.isnan(probs))


@pytest.mark.skip(reason="Not working properly")
def test_transform_rand_betas_shapes():
    Kr = 2
    N = 4
    R = 3
    betas = jnp.arange(Kr + Kr + Kr * (Kr + 1) // 2, dtype=float)
    draws = jnp.ones((N, Kr, R))
    rand_idx = jnp.arange(Kr)
    sd_start_idx = Kr
    sd_slice_size = Kr
    chol_start_idx = sd_start_idx + sd_slice_size
    chol_slice_size = (sd_slice_size * (sd_slice_size + 1)) // 2 - sd_slice_size
    idx_ln_dist = jnp.array([], dtype=int)
    out = _transform_rand_betas(
        betas, draws, rand_idx, sd_start_idx, sd_slice_size, chol_start_idx, chol_slice_size, idx_ln_dist, False
    )
    assert out.shape == (N, Kr, R)
    out_corr = _transform_rand_betas(
        betas, draws, rand_idx, sd_start_idx, sd_slice_size, chol_start_idx, chol_slice_size, idx_ln_dist, True
    )
    assert out_corr.shape == (N, Kr, R)


@pytest.mark.skip(reason="Not working properly")
def test_transform_rand_betas_jit():
    Kr = 2
    N = 3
    R = 2
    betas = jnp.arange(Kr + Kr + Kr * (Kr + 1) // 2, dtype=float)
    draws = jnp.ones((N, Kr, R))
    rand_idx = jnp.arange(Kr)
    sd_start_idx = Kr
    sd_slice_size = Kr
    chol_start_idx = sd_start_idx + sd_slice_size
    chol_slice_size = (sd_slice_size * (sd_slice_size + 1)) // 2 - sd_slice_size
    idx_ln_dist = jnp.array([], dtype=int)
    fn = jax.jit(
        _transform_rand_betas,
        static_argnames=[
            "sd_start_index",
            "sd_slice_size",
            "chol_start_idx",
            "chol_slice_size",
            "include_correlations",
        ],
    )
    out = fn(betas, draws, rand_idx, sd_start_idx, sd_slice_size, chol_start_idx, chol_slice_size, idx_ln_dist, True)
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

    with open("tests/simple_data_output.json", "w") as f:
        json.dump(model, f)


def main():
    save_simple_data_output()


if __name__ == "__main__":
    main()
