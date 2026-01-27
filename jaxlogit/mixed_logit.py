import logging
import jax
import jax.numpy as jnp
import numpy as np

from jaxlogit._choice_model import ChoiceModel, diff_nonchosen_chosen
from jaxlogit._variables import ParametersSetup
from jaxlogit._optimize import _minimize, hessian
from jaxlogit.draws import truncnorm_ppf, generate_draws
from jaxlogit.utils import get_panel_aware_batch_indices
from jaxlogit._config_data import ConfigData

_logger = logging.getLogger(__name__)

"""
Notations
---------
    N : Number of choice situations
    P : Number of observations per panel
    J : Number of alternatives
    K : Number of variables (Kf: fixed, Kr: random)
"""


class MixedLogit(ChoiceModel):
    """Class for estimation of Mixed Logit Models."""

    def __init__(self):
        super(MixedLogit, self).__init__()
        self._rvidx = None  # Index of random variables (True when random var)
        self._rvdist = None  # List of mixing distributions of rand vars

    def _setup_input_data(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        randvars,
        config: ConfigData,
        predict_mode=False,
    ):
        # TODO: replace numpy random structure with jax
        if config.random_state is not None:
            np.random.seed(config.random_state)

        self._check_long_format_consistency(ids, alts)
        y = self._format_choice_var(y, alts) if not predict_mode else None
        X, Xnames = self._setup_design_matrix(X)
        self._model_specific_validations(randvars, Xnames)

        self._setup_randvars_info(randvars, Xnames)

        N, J, K = X.shape[0], X.shape[1], X.shape[2]

        # TODO TN: normal and lognormals only
        num_normal_based_params = len(jnp.where(self._rvidx_normal_bases)[0])
        num_truncnorm_based_params = len(jnp.where(self._rvidx_truncnorm_based)[0])
        num_random_params = len(randvars)
        assert num_normal_based_params + num_truncnorm_based_params == num_random_params, (
            "Distributions other than normal, log-normal, truncated normal not implemented yet"
        )
        # lower triangular matrix elements of correlations for random variables, minus the diagonal
        num_cholesky_params = (
            0
            if not config.include_correlations
            else num_normal_based_params * (num_normal_based_params + 1) // 2 - num_normal_based_params
        )

        if config.panels is not None:
            if config.panels.shape == (N,):
                panels = config.panels.shape
            else:
                # Convert panel ids to indexes
                panels = config.panels.reshape(N, J)[:, 0]
                panels_idx = np.empty(N)
                for i, u in enumerate(np.unique(panels)):
                    panels_idx[np.where(panels == u)] = i
                config.panels = panels_idx.astype(int)

        # Reshape arrays in the format required for the rest of the estimation
        X = X.reshape(N, J, K)
        y = y.reshape(N, J, 1) if not predict_mode else None

        if config.avail is not None:
            config.avail = config.avail.reshape(N, J)

        if config.weights is not None and not(config.setup_completed):
            config.weights = config.weights.reshape(N, J)[:, 0]
            if config.panels is not None:
                panel_change_idx = np.concatenate(([0], np.where(config.panels[:-1] != config.panels[1:])[0] + 1))
                config.weights = config.weights[panel_change_idx]

        # initial values for coefficients. One for each provided variable, plus a std dev for each random variable,
        # plus correlation coefficients for random variables if requested.
        num_coeffs = K + num_random_params + num_cholesky_params
        if config.init_coeff is None:
            betas = np.repeat(0.1, num_coeffs)
        else:
            betas = config.init_coeff
            if len(config.init_coeff) != num_coeffs:
                raise ValueError(f"The length of init_coeff must be {num_coeffs}, but got {len(config.init_coeff)}.")

        # Add std dev and correlation coefficients to the coefficient names
        coef_names = np.append(Xnames, np.char.add("sd.", Xnames[self._rvidx]))
        # cholesky params only for normal/lognormal if include_correlations
        if config.include_correlations:
            corr_names = [
                f"chol.{i}.{j}"
                for idx_j, j in enumerate(Xnames[self._rvidx_normal_bases])
                for i in Xnames[self._rvidx_normal_bases][:idx_j]
            ]
            coef_names = np.append(coef_names, corr_names)

        assert len(coef_names) == num_coeffs, (
            f"Wrong number of coefficients set up, this is a data prep bug. Expected {num_coeffs}, got {len(coef_names)}. {coef_names}."
        )
        _logger.debug(f"Set up {num_coeffs} initial coefficients for the model: {dict(zip(coef_names, betas))}")

        return (
            jnp.array(betas),
            jnp.array(X),
            None if predict_mode else jnp.array(y),
            jnp.array(config.panels) if config.panels is not None else None,
            jnp.array(config.weights) if config.weights is not None else None,
            jnp.array(config.avail) if config.avail is not None else None,
            Xnames,
            coef_names,
        )

    def setup_draws_from_config(self, N: int, config: ConfigData):
        """Returns the draws.

        Formats the draws according to the panels.

        Args:
            N: number of observations. Size of X, the data.
            config: The data config for the fit/predict
        """

        # Generate draws
        n_samples = N if config.panels is None else np.max(config.panels) + 1
        _logger.debug(f"Generating {config.n_draws} number of draws for each observation and random variable")

        draws = generate_draws(n_samples, config.n_draws, self._rvdist, config.halton, halton_opts=config.halton_opts)
        if draws.size == 0:
            return draws

        if config.panels is not None:
            draws = draws[config.panels]  # (N,num_random_params,n_draws)
        draws = jnp.array(draws)

        _logger.debug(f"Draw generation done, shape of draws: {draws.shape}, number of draws: {config.n_draws}")

        return draws

    def data_prep(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        randvars,
        config: ConfigData,
        predict_mode=False,
    ):
        # Handle array-like inputs by converting everything to numpy arrays
        (
            X,
            y,
            varnames,
            alts,
            ids,
            config.weights,
            config.panels,
            config.avail,
        ) = self._as_array(
            X,
            y,
            varnames,
            alts,
            ids,
            config.weights,
            config.panels,
            config.avail,
        )

        self._validate_inputs(X, y, alts, varnames, config.weights, predict_mode=predict_mode, setup_completed=config.setup_completed)

        self._pre_fit(alts, varnames, config.maxiter)

        (
            betas,
            X,
            y,
            panels,
            weights,
            avail,
            Xnames,
            coef_names,
        ) = self._setup_input_data(X, y, varnames, alts, ids, randvars, config, predict_mode=predict_mode)
        config.setup_completed = True

        parameter_info = ParametersSetup(
            self._rvdist,
            self._rvidx,
            self._rvidx_normal_bases,
            self._rvidx_truncnorm_based,
            coef_names,
            betas,
            config,
        )

        draws = self.setup_draws_from_config(X.shape[0], config)

        # panels are 0-based and contiguous by construction, so we can use the maximum value to determine the number
        # of panels. We provide this number explicitly to the log-likelihood function for jit compilation of
        # segment_sum (product of probabilities over panels)
        num_panels = 0 if panels is None else int(jnp.max(panels)) + 1

        if not predict_mode:
            # This here is estimation specific - we compute the difference between the chosen and non-chosen
            # alternatives because we only need the probability of the chosen alternative in the log-likelihood
            Xd, avail = diff_nonchosen_chosen(X, y, avail)  # Setup Xd as Xij - Xi*
        else:
            Xd = X

        # split data for non-random and random parameters to speed up calculations
        rvidx = jnp.array(self._rvidx, dtype=bool)
        # rand_idx = jnp.where(rvidx)[0]
        Xdf = Xd[:, :, ~rvidx]  # Data for fixed (non-random) parameters
        Xdr = Xd[:, :, rvidx]  # Data for random parameters

        return (betas, Xdf, Xdr, panels, weights, avail, num_panels, coef_names, draws, parameter_info)

    def fit(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        randvars,  # TODO: check if this works for zero randvars
        config: ConfigData,
        # optim_method="trust-region",  # "trust-region", "L-BFGS-B", "BFGS"
        # force_positive_chol_diag=True,  # use softplus for the cholesky diagonal elements
        # hessian_by_row=True,  # calculate the hessian row by row in a for loop to save memory at the expense of runtime
        verbose=1,
    ):
        """Fit Mixed Logit models.

        Parameters
        ----------
        X : array-like, shape (n_samples*n_alts, n_variables)
            Input data for explanatory variables in long format.
        y : array-like, shape (n_samples*n_alts,)
            Chosen alternatives or one-hot encoded representation of the choices.
        varnames : list-like, shape (n_variables,)
            Names of explanatory variables that must match the number and order of
            columns in ``X``.
        alts : array-like, shape (n_samples*n_alts,)
            Alternative values in long format.
        ids : array-like, shape (n_samples*n_alts,)
            Identifiers for the samples in long format.
        randvars : dict
            Names (keys) and mixing distributions (values) of variables that have
            random parameters as coefficients. Possible mixing distributions are:

            - ``'n'``: normal
            - ``'ln'``: lognormal
            - ``'t'``: triangular
            - ``'tn'``: truncated normal

        verbose : int, default=1
            Verbosity of messages to show during estimation.

            - 0: No messages
            - 1: Some messages
            - 2: All messages

        Returns
        -------
        result
            The estimated model parameters result.
        """

        (betas, Xdf, Xdr, panels, weights, avail, num_panels, coef_names, draws, parameter_info) = self.data_prep(
            X,
            y,
            varnames,
            alts,
            ids,
            randvars,
            config,
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

        if parameter_info.idx_ln_dist.shape[0] > 0:
            _logger.info(
                f"Lognormal distributions found for {parameter_info.idx_ln_dist.shape[0]} random variables, applying transformation."
            )

        if panels is not None:
            _logger.info(f"Data contains {num_panels} panels.")

        _logger.debug(f"Shape of Xdf: {Xdf.shape}, shape of Xdr: {Xdr.shape}")

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
        if optim_res is None:
            _logger.error("Optimization failed, returning None.")
            return None

        _logger.info(
            f"Optimization finished, success = {optim_res['success']}, final loglike = {-optim_res['fun']:.2f}"
            + f", final gradient max = {optim_res['jac'].max():.2e}, norm = {jnp.linalg.norm(optim_res['jac']):.2e}."
        )

        if config.skip_std_errs:
            _logger.info("Skipping H_inv and grad_n calculation due to skip_std_errs=True")
        else:
            _logger.info("Calculating gradient of individual log-likelihood contributions")
            grad = jax.jacfwd(loglike_individual)
            optim_res["grad_n"] = grad(jnp.array(optim_res["x"]), *fargs[:-1])

            _logger.info(
                f"Calculating Hessian, by row={config.hessian_by_row}, finite diff={config.finite_diff_hessian}"
            )

            H = hessian(
                neg_loglike,
                jnp.array(optim_res["x"]),
                config.hessian_by_row,
                config.finite_diff_hessian,
                fargs,
            )

            _logger.info("Inverting Hessian")
            # remove masked parameters to make it invertible
            if parameter_info.mask is not None:
                mask_for_hessian = jnp.array([x for x in range(0, H.shape[0]) if x not in parameter_info.mask])
                h_free = H[jnp.ix_(mask_for_hessian, mask_for_hessian)]
                h_inv_nonfixed = jax.lax.stop_gradient(jnp.linalg.inv(h_free))
                h_inv = jnp.zeros_like(H)
                h_inv = h_inv.at[jnp.ix_(mask_for_hessian, mask_for_hessian)].set(h_inv_nonfixed)
            else:
                h_inv = jax.lax.stop_gradient(jnp.linalg.inv(H))

            optim_res["hess_inv"] = h_inv
            # TODO: Do we want to use Hinv = jnp.linalg.pinv(np.dot(optim_res["grad_n"].T, optim_res["grad_n"])) as fallback?

        self._post_fit(optim_res, coef_names, Xdf.shape[0], parameter_info.mask, config.set_vars, config.skip_std_errs)
        return optim_res

    def _setup_randvars_info(self, randvars, Xnames):
        """Set up information about random variables and their mixing distributions.
        _rvidx: boolean array indicating which variables are random
        _rvdist: list of mixing distributions for each random variable
        """
        self.randvars = randvars
        self._rvidx, self._rvdist = [], []
        self._rvidx_normal_bases = []
        self._rvidx_truncnorm_based = []
        for var in Xnames:
            if var in self.randvars.keys():
                self._rvidx.append(True)
                self._rvdist.append(self.randvars[var])
                if self.randvars[var] in ["n", "ln"]:
                    self._rvidx_normal_bases.append(True)
                    self._rvidx_truncnorm_based.append(False)
                elif self.randvars[var] == "n_trunc":
                    self._rvidx_normal_bases.append(False)
                    self._rvidx_truncnorm_based.append(True)
                else:
                    raise ValueError("Only normal, log-normal, truncated normal distributions are implemented for now")
            else:
                self._rvidx.append(False)
                self._rvidx_normal_bases.append(False)
                self._rvidx_truncnorm_based.append(False)

        self._rvidx = np.array(self._rvidx)
        self._rvidx_normal_bases = np.array(self._rvidx_normal_bases)
        self._rvidx_truncnorm_based = np.array(self._rvidx_truncnorm_based)

    def _model_specific_validations(self, randvars, Xnames):
        """Conduct validations specific for mixed logit models."""
        if randvars is None:
            raise ValueError("The 'randvars' parameter is required for Mixed Logit estimation")
        if not set(randvars.keys()).issubset(Xnames):
            raise ValueError("Some variable names in 'randvars' were not found in the list of variable names")
        if not set(randvars.values()).issubset(["n", "ln", "t", "tn", "n_trunc", "u"]):
            raise ValueError("Wrong mixing distribution in 'randvars'. Accepted distrubtions are n, ln, t, u, tn")

    def summary(self):
        """Show estimation results in console."""
        super(MixedLogit, self).summary()

    def predict(self, X, varnames, alts, ids, randvars, config: ConfigData):
        assert config.init_coeff is not None
        (betas, Xdf, Xdr, panels, weights, avail, num_panels, coef_names, draws, parameter_info) = self.data_prep(
            X,
            None,
            varnames,
            alts,
            ids,
            randvars,
            config,
            predict_mode=True,
        )

        fargs = (Xdf, Xdr, panels, weights, avail, num_panels, config.force_positive_chol_diag, draws, parameter_info)

        probs = probability_individual(betas, *fargs)
        # uq_alts, idx = np.unique(alts, return_index=True)
        # uq_alts = uq_alts[np.argsort(idx)]
        # return pd.DataFrame(probs, columns=uq_alts)
        return probs


def _apply_distribution(betas_random, idx_ln_dist):
    """Apply the mixing distribution to the random betas."""

    if jax.config.jax_enable_x64:
        UTIL_MAX = 700
    else:
        UTIL_MAX = 87

    for i in idx_ln_dist:
        betas_random = betas_random.at[:, i, :].set(jnp.exp(betas_random[:, i, :].clip(-UTIL_MAX, UTIL_MAX)))

    return betas_random


def _transform_rand_betas(betas, force_positive_chol_diag, draws, parameter_info: ParametersSetup):
    """Compute the products between the betas and the random coefficients.

    This method also applies the associated mixing distributions
    """

    diag_vals = betas[
        parameter_info.rand_idx_stddev
    ]  # jax.lax.dynamic_slice(betas, (sd_start_index,), (sd_slice_size,))
    if force_positive_chol_diag:
        diag_vals = jax.nn.softplus(diag_vals)
        if parameter_info.mask_chol is not None:
            # Apply mask to the diagonal values of the Cholesky matrix again.
            # Could work around this by setting asserted params to softplus-1(x) but we also want to ensure
            # 0 values are propagated correctly for, e.g., ECs with less than full rank cov matrix.
            diag_vals = diag_vals.at[parameter_info.mask_chol].set(parameter_info.values_for_chol_mask)

    ### Normal/lognormal part
    br_mean = betas[parameter_info.rand_idx_norm]
    br_std_dev = diag_vals[parameter_info.draws_idx_norm]
    if parameter_info.rand_idx_chol is not None:
        # chol_start_idx = sd_start_index + sd_slice_size
        # chol_slice_size = (sd_slice_size * (sd_slice_size + 1)) // 2 - sd_slice_size
        sd_slice_size = len(br_mean)

        # Build lower-triangular Cholesky matrix
        tril_rows, tril_cols = jnp.tril_indices(sd_slice_size)
        L = jnp.zeros((sd_slice_size, sd_slice_size), dtype=betas.dtype)
        diag_mask = tril_rows == tril_cols
        off_diag_mask = ~diag_mask
        off_diag_vals = betas[
            parameter_info.rand_idx_chol
        ]  # jax.lax.dynamic_slice(betas, (chol_start_idx,), (chol_slice_size,))

        tril_vals = jnp.where(diag_mask, br_std_dev[tril_rows], off_diag_vals[jnp.cumsum(off_diag_mask) - 1])
        L = L.at[tril_rows, tril_cols].set(tril_vals)

        N, _, R = draws.shape
        draws_flat = draws[:, parameter_info.draws_idx_norm, :].transpose(0, 2, 1).reshape(-1, sd_slice_size)
        correlated_flat = (L @ draws_flat.T).T
        cov = correlated_flat.reshape(N, R, sd_slice_size).transpose(0, 2, 1)
    else:
        cov = draws[:, parameter_info.draws_idx_norm, :] * br_std_dev[None, :, None]

    # betas random
    betas_random = jnp.empty_like(draws)  # num_obs, num_rand_vars, num_draws

    for i, idx_norm in enumerate(parameter_info.draws_idx_norm):
        betas_random = betas_random.at[:, idx_norm, :].set(br_mean[None, i, None] + cov[:, i, :])

    # apply lognormal part if there are any
    betas_random = _apply_distribution(betas_random, parameter_info.idx_ln_dist)

    ### Truncated normal part
    br_mean = betas[parameter_info.rand_idx_truncnorm]
    br_std_dev = diag_vals[parameter_info.draws_idx_truncnorm]
    for i, idx_truncnorm in enumerate(parameter_info.draws_idx_truncnorm):
        betas_random = betas_random.at[:, idx_truncnorm, :].set(
            truncnorm_ppf(draws[:, idx_truncnorm, :], br_mean[i], br_std_dev[i])
        )

    return betas_random


### TODO: re-write for JAX, make whole class derive from pytree, etc. Until then, this is a separate method.
def neg_loglike(
    betas,
    Xdf,
    Xdr,
    panels,
    weights,
    avail,
    num_panels,
    force_positive_chol_diag,
    draws,
    parameter_info: ParametersSetup,
    batch_size,
):
    loglik_individ = loglike_individual(
        betas, Xdf, Xdr, panels, weights, avail, num_panels, force_positive_chol_diag, draws, parameter_info
    )

    loglik = loglik_individ.sum()
    return -loglik


def neg_loglike_grad_batched(
    betas,
    Xdf,
    Xdr,
    panels,
    weights,
    avail,
    num_panels,
    force_positive_chol_diag,
    draws,
    parameter_info: ParametersSetup,
    batch_size,
):
    if panels is None:
        # Simple case: no panels, just batch observations
        n_obs = Xdf.shape[0]
        batch_indices = [
            (i, min(i + batch_size, n_obs), min(i + batch_size, n_obs) - i) for i in range(0, n_obs, batch_size)
        ]
    else:
        # Panel case: group panels into batches
        if not jnp.all(jnp.diff(panels) >= 0):
            raise ValueError(
                "Panel array must be sorted for batching, please re-format input data or run with batch_size=None."
            )  # note we made sure panel indexes are 0-based and contiguous during data prep, but not that they are sorted
        num_batches = int(np.ceil(len(panels) / batch_size))
        batch_indices = get_panel_aware_batch_indices(panels, num_batches)

    loglik = jnp.array(0.0)
    grad_loglik = jnp.zeros_like(betas)
    num_panels_counter = 0

    for start, end, num_panels_this_batch in batch_indices:
        # panels need to start at 0 and be contiguous for segment_sum to work correctly
        batch_panels = panels[start:end] - panels[start] if panels is not None else None
        loglik_individ, grad_loglike_individ = loglike_and_grad_individual(
            betas,
            Xdf[start:end, :, :],
            Xdr[start:end, :, :],
            batch_panels,
            weights[num_panels_counter : num_panels_counter + num_panels_this_batch] if weights is not None else None,
            avail[start:end] if avail is not None else None,
            num_panels_this_batch,
            force_positive_chol_diag,
            draws[start:end, :, :],
            parameter_info,
        )

        num_panels_counter += num_panels_this_batch
        loglik = loglik + loglik_individ
        grad_loglik = grad_loglik + grad_loglike_individ

        # logger.debug(
        #    f"Batch ({start},{end},{num_panels_this_batch}: {loglik_individ:.3f}, {grad_loglike_individ}"
        #    + f", acc: {loglik:.3f}, {grad_loglik}, {num_panels_counter})."
        # )

    return -loglik, -grad_loglik  # is this right 0 both -1?


def loglike_individual_sum(
    betas,
    Xdf,
    Xdr,
    panels,
    weights,
    avail,
    num_panels,
    force_positive_chol_diag,
    draws,
    parameter_info: ParametersSetup,
):
    ll = loglike_individual(
        betas, Xdf, Xdr, panels, weights, avail, num_panels, force_positive_chol_diag, draws, parameter_info
    )
    return ll.sum()


loglike_and_grad_individual = jax.jit(
    jax.value_and_grad(loglike_individual_sum, argnums=0),
    static_argnames=["num_panels", "force_positive_chol_diag", "parameter_info"],
)


def loglike_individual(
    betas,
    Xdf,
    Xdr,
    panels,
    weights,
    avail,
    num_panels,
    force_positive_chol_diag,
    draws,
    parameter_info: ParametersSetup,
):
    """Compute the log-likelihood.

    Fixed and random parameters are handled separately to speed up the estimation and the results are concatenated.
    """

    if jax.config.jax_enable_x64:
        UTIL_MAX = 700  # ONLY IF 64bit precision is used
        LOG_PROB_MIN = 1e-300
    else:
        UTIL_MAX = 87
        LOG_PROB_MIN = 1e-30

    # mask for asserted parameters.
    if parameter_info.mask is not None:
        betas = betas.at[parameter_info.mask].set(parameter_info.values_for_mask)

    # Utility for fixed parameters
    Bf = betas[parameter_info.non_random_idx]  # Fixed betas
    Vdf = jnp.einsum("njk,k -> nj", Xdf, Bf)  # (N, J-1)

    # Utility for random parameters
    if parameter_info.random_idx.size != 0:
        Br = _transform_rand_betas(
            betas, force_positive_chol_diag, draws, parameter_info
        )  # Br shape: (num_obs, num_rand_vars, num_draws)
        parameterised_random_variable_contribution = jnp.einsum("njk,nkr -> njr", Xdr, Br)
    else:
        parameterised_random_variable_contribution = jnp.zeros_like(Vdf[:, :, None])

    # Vdr shape: (N,J-1,R)
    Vd = Vdf[:, :, None] + parameterised_random_variable_contribution
    eVd = jnp.exp(jnp.clip(Vd, -UTIL_MAX, UTIL_MAX))
    eVd = eVd if avail is None else eVd * avail[:, :, None]
    proba_n = 1 / (1 + eVd.sum(axis=1))  # (N,R)

    if panels is not None:
        # # no grads for segment_prod for non-unique panels. need to use sum of logs and then exp as workaround
        # proba_ = jax.ops.segment_prod(proba_n, panels, num_segments=num_panels)
        proba_n = jnp.exp(
            jnp.clip(
                jax.ops.segment_sum(
                    jnp.log(jnp.clip(proba_n, LOG_PROB_MIN, jnp.inf)),
                    panels,
                    num_segments=num_panels,
                ),
                -UTIL_MAX,
                UTIL_MAX,
            )
        )
    num_draws = max(1, draws.shape[2])
    loglik = jnp.log(jnp.clip(proba_n.sum(axis=1) / num_draws, LOG_PROB_MIN, jnp.inf))

    # loglik = jnp.log(jnp.clip(proba_n.sum(axis=1) / draws.shape[2], LOG_PROB_MIN, jnp.inf))

    if weights is not None:
        loglik = loglik * weights

    return loglik


def probability_individual(
    betas,
    Xdf,
    Xdr,
    panels,
    weights,
    avail,
    num_panels,
    force_positive_chol_diag,
    draws,
    parameter_info: ParametersSetup,
):
    """Compute the probabilities of all alternatives."""

    if jax.config.jax_enable_x64:
        UTIL_MAX = 700  # ONLY IF 64bit precision is used
    else:
        UTIL_MAX = 87

    R = max(1, draws.shape[2])
    # R = draws.shape[2]

    # mask for asserted parameters.
    if parameter_info.mask is not None:
        betas = betas.at[parameter_info.mask].set(parameter_info.values_for_mask)

    # Utility for fixed parameters
    Bf = betas[parameter_info.non_random_idx]  # Fixed betas
    Vdf = jnp.einsum("njk,k -> nj", Xdf, Bf)  # (N, J)

    if parameter_info.random_idx.size:
        Br = _transform_rand_betas(
            betas, force_positive_chol_diag, draws, parameter_info
        )  # Br shape: (num_obs, num_rand_vars, num_draws)
        parameterised_random_variable_contribution = jnp.einsum("njk,nkr -> njr", Xdr, Br)
    else:
        parameterised_random_variable_contribution = jnp.zeros_like(Vdf[:, :, None])

    # Vdr shape: (N,J,R)
    Vd = Vdf[:, :, None] + parameterised_random_variable_contribution
    eVd = jnp.exp(jnp.clip(Vd, -UTIL_MAX, UTIL_MAX))
    eVd = eVd if avail is None else eVd * avail[:, :, None]

    proba_n = eVd / eVd.sum(axis=1)[:, None, :]  # (N,J,R)

    # TODO: check if this is still correct - might need to be over different dimension? - Let's leave this out for now
    # if panels is not None:
    #     # # no grads for segment_prod for non-unique panels. need to use sum of logs and then exp as workaround
    #     # proba_ = jax.ops.segment_prod(proba_n, panels, num_segments=num_panels)
    #     proba_n = jnp.exp(
    #         jnp.clip(
    #             jax.ops.segment_sum(
    #                 jnp.log(jnp.clip(proba_n, LOG_PROB_MIN, jnp.inf)),
    #                 panels,
    #                 num_segments=num_panels,
    #             ),
    #             -UTIL_MAX,
    #             UTIL_MAX,
    #         )
    #     )

    mean_proba_n = proba_n.sum(axis=2) / R

    return mean_proba_n
