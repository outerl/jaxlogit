import pytest
import numpy as np
import jax
import jax.numpy as jnp

from jaxlogit.mixed_logit import (
    MixedLogit,
    _transform_rand_betas,
    loglike_individual,
    neg_loglike,
    probability_individual,
)


@pytest.fixture
def simple_data():
    N = 6  # individuals
    J = 3  # alternatives
    K = 2  # variables
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


def test_mixed_logit_fit(simple_data):
    X, y, ids, alts, avail, panels, weights = simple_data
    varnames = [f"x{i}" for i in range(X.shape[1])]

    model = MixedLogit()
    randvars = {varnames[0]: "n"}
    fixedvars = {}
    result = model.fit(
        X=X,
        y=y,
        varnames=varnames,
        ids=ids,
        alts=alts,
        avail=avail,
        panels=panels,
        weights=weights,
        n_draws=3,
        randvars=randvars,
        fixedvars=fixedvars,
        optim_method="L-BFGS-B",
        init_coeff=None,
        skip_std_errs=True,
    )
    assert result is not None
    assert "fun" in result


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
    (
        betas,
        Xdf,
        Xdr,
        panels_,
        draws,
        weights_,
        avail_,
        scale_d,
        mask,
        values_for_mask,
        rvidx,
        rand_idx,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        coef_names,
    ) = model.data_prep_for_fit(
        X=df[varnames],
        y=df["choice"],
        varnames=varnames,
        ids=df["custom_id"],
        alts=df["alt"],
        avail=df["avail"],
        panels=df["person_id_contiguous"],
        weights=df["weight"],
        n_draws=3,
        randvars=randvars,
        fixedvars=fixedvars,
        init_coeff=None,
        include_correlations=False,
    )

    ll_indiv = loglike_individual(
        betas,
        Xdf,
        Xdr,
        panels_,
        draws,
        weights_,
        avail_,
        scale_d,
        mask,
        values_for_mask,
        rvidx,
        rand_idx,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        False,
    )
    assert ll_indiv.shape[0] == num_panels
    assert not jnp.any(jnp.isnan(ll_indiv))

    nll = neg_loglike(
        betas,
        Xdf,
        Xdr,
        panels_,
        draws,
        weights_,
        avail_,
        scale_d,
        mask,
        values_for_mask,
        rvidx,
        rand_idx,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        False,
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
    (
        betas,
        Xdf,
        Xdr,
        panels_,
        draws,
        weights_,
        avail_,
        scale_d,
        mask,
        values_for_mask,
        rvidx,
        rand_idx,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        coef_names,
    ) = model.data_prep_for_fit(
        X=df[varnames],
        y=df["choice"],
        varnames=varnames,
        ids=df["custom_id"],
        alts=df["alt"],
        avail=df["avail"],
        panels=df["person_id_contiguous"],
        weights=df["weight"],
        n_draws=3,
        randvars=randvars,
        fixedvars=fixedvars,
        init_coeff=None,
        include_correlations=False,
    )

    probs = probability_individual(
        betas,
        Xdf,
        Xdr,
        panels_,
        draws,
        weights_,
        avail_,
        scale_d,
        mask,
        values_for_mask,
        rvidx,
        rand_idx,
        fixed_idx,
        num_panels,
        idx_ln_dist,
        False,
    )
    assert probs.shape[0] == Xdf.shape[0]
    assert not jnp.any(jnp.isnan(probs))


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
