import logging
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Dict, Union, Sequence
from pandas import Series

from ._choice_model import ChoiceModel, diff_nonchosen_chosen
from ._optimize import _minimize, gradient, hessian
from .draws import generate_draws, truncnorm_ppf
from .utils import get_panel_aware_batch_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

"""
Notations
---------
    N : Number of choice situations
    P : Number of observations per panel
    J : Number of alternatives
    K : Number of variables (Kf: fixed, Kr: random)
"""

ArrayLike = Union[np.ndarray, Series, Sequence[Any]]


@dataclass
class ConfigData:
    """Configurations for the fit and predict functions with default values.

    Member variables:
        weights: array-like, shape (n_samples,), default=None
            Sample weights in long format.

        avail: array-like, shape (n_samples*n_alts,), default=None
            Availability of alternatives for the choice situations. One when available or zero otherwise.

        panels: array-like, shape (n_samples*n_alts,), default=None
            Identifiers in long format to create panels in combination with ``ids``

        init_coeff: numpy array, shape (n_variables,), default=None
            Initial coefficients for estimation.

        maxiter: int, default=2000
            Maximum number of iterations

        random_state: int, default=None
            Random seed for numpy random generator

        n_draws: int, default=1000
            Number of random draws to approximate the mixing distributions of the random coefficients

        halton: bool, default=True
            Whether the estimation uses halton draws.

        halton_opts: dict, default=None
            Options for generation of halton draws. The dictionary accepts the following options (keys):

                shuffle: bool, default=False
                    Whether the Halton draws should be shuffled

                drop: int, default=100
                    Number of initial Halton draws to discard to minimize correlations between Halton sequences

                primes: list
                    List of primes to be used as base for generation of Halton sequences.

        tol_opts: dict, default=None
            Options for tolerance of optimization routine. The dictionary accepts the following options (keys):

                ftol: float, default=1e-10
                    Tolerance for objective function (log-likelihood)

                gtol: float, default=1e-5
                    Tolerance for gradient function.

        num_hess: bool, default=False
            Whether numerical hessian should be used for estimation of standard errors

        fixedvars: dict, default=None
            Specified variable names (keys) of variables to be fixed to the given value (values)

        optim_method: str, default="trust-region" ##############################
            Optimization method to use for model estimation. It can be `trust-region`, `BFGS` or `L-BFGS-B`.

        skip_std_errs: bool, default=False
            Whether estimation of standard errors should be skipped

        include_correlations: bool, default=False
            Whether correlations between variables should be included as explanatory variables

        force_positive_chol_diag:bool, default=True


        hessian_by_row: bool, default=True
            whether to calculate the hessian row by row in a for loop to save
            memory at the expense of runtime

        finite_diff_hessian: bool, default=False
            Whether the hessian should be computed using finite difference.
            If true, this will stay within memory limits.

        batch_size: int, default=None
            Size of batches used to avoid GPU memory overflow.
    """

    weights: Optional[ArrayLike] = None
    avail: Optional[ArrayLike] = None
    panels: Optional[ArrayLike] = None
    init_coeff: Optional[ArrayLike] = None
    maxiter: int = 2000
    random_state: Optional[int] = None
    n_draws: int = 1000
    halton: bool = True
    halton_opts: Optional[Dict] = None
    tol_opts: Optional[Dict] = None
    num_hess: bool = False
    fixedvars: Any = None
    optim_method: str = "trust-region"
    skip_std_errs: bool = False
    include_correlations: bool = False
    force_positive_chol_diag: bool = True
    hessian_by_row: bool = True
    finite_diff_hessian: bool = False
    batch_size: Optional[int] = None


class MixedLogit(ChoiceModel):
    """Class for estimation of Mixed Logit Models."""

    def __init__(self):
        super(MixedLogit, self).__init__()
        self._rvidx = None  # Index of random variables (True when random var)
        self._rvdist = None  # List of mixing distributions of rand vars

    def set_data_variables(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        randvars,
        weights,
        avail,
        panels,
        init_coeff,
        maxiter,
        random_state,
        n_draws,
        halton,
        halton_opts,
        tol_opts,
        num_hess,
        fixedvars,
        optim_method,
        skip_std_errs,
        include_correlations,
        force_positive_chol_diag,
        hessian_by_row,
        finite_diff_hessian,
        batch_size,
    ):
        # Set class variables to enable simple pickling and running things post-estimation for analysis. This will be
        # replaced by proper database/dataseries structure in the future.
        self.X_raw = X
        self.y_raw = y
        self.varnames_raw = varnames
        self.alts_raw = (alts,)
        self.ids_raw = (ids,)
        self.randvars_raw = (randvars,)
        self.weights_raw = (weights,)
        self.avail_raw = (avail,)
        self.panels_raw = (panels,)
        self.init_coeff_raw = (init_coeff,)
        self.maxiter_raw = (maxiter,)
        self.random_state_raw = (random_state,)
        self.n_draws_raw = (n_draws,)
        self.halton_raw = (halton,)
        self.halton_opts_raw = (halton_opts,)
        self.tol_opts_raw = (tol_opts,)
        self.num_hess_raw = (num_hess,)
        self.fixedvars_raw = (fixedvars,)
        self.optim_method_raw = (optim_method,)
        self.skip_std_errs_raw = (skip_std_errs,)
        self.include_correlations_raw = (include_correlations,)
        self.force_positive_chol_diag_raw = (force_positive_chol_diag,)
        self.hessian_by_row_raw = (hessian_by_row,)
        self.finite_diff_hessian_raw = (finite_diff_hessian,)
        self.batch_size_raw = (batch_size,)

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

        # Generate draws
        n_samples = N if config.panels is None else np.max(config.panels) + 1
        logger.debug(f"Generating {config.n_draws} number of draws for each observation and random variable")
        draws = generate_draws(n_samples, config.n_draws, self._rvdist, config.halton, halton_opts=config.halton_opts)
        if config.panels is not None:
            draws = draws[config.panels]  # (N,num_random_params,n_draws)
        draws = jnp.array(draws)
        logger.debug(f"Draw generation done, shape of draws: {draws.shape}, number of draws: {config.n_draws}")

        if config.weights is not None:  # Reshape weights to match input data
            print("here", config.weights)
            # weights = weights.reshape(N, J)[:, 0]
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
        logger.debug(f"Set up {num_coeffs} initial coefficients for the model: {dict(zip(coef_names, betas))}")

        return (
            jnp.array(betas),
            jnp.array(X),
            None if predict_mode else jnp.array(y),
            jnp.array(config.panels) if config.panels is not None else None,
            draws,
            jnp.array(config.weights) if config.weights is not None else None,
            jnp.array(config.avail) if config.avail is not None else None,
            Xnames,
            coef_names,
        )

    def set_variable_indices(self, include_correlations):
        """Find and save indexes of types of random variables."""
        ### WIP
        # want idx_norml, idx_trunc for mean into betas.
        # rvidx = jnp.array(self._rvidx, dtype=bool)
        rand_idx_norm = jnp.where(self._rvidx_normal_bases)[0]
        rand_idx_truncnorm = jnp.where(self._rvidx_truncnorm_based)[0]

        # #std dev is different: in order
        sd_start_idx = len(self._rvidx)  # start of std devs
        sd_slice_size = len(jnp.where(self._rvidx)[0])  # num all std devs
        # TODO TN: separate rand_idx_stddev for n_trunc and n/ln
        rand_idx_stddev = jnp.arange(sd_start_idx, sd_start_idx + sd_slice_size, dtype=jnp.int32)
        # rand_idx_stddev = jnp.argwhere

        # only for n/ln, not n_trunc
        chol_start_idx = sd_start_idx + sd_slice_size  # start: after all std devs
        sd_chol_slice_size = len(jnp.where(self._rvidx_normal_bases)[0])  # number of elements based on n/ln dists
        chol_slice_size = (sd_chol_slice_size * (sd_chol_slice_size + 1)) // 2 - sd_chol_slice_size
        rand_idx_chol = (
            None
            if not include_correlations
            else jnp.arange(chol_start_idx, chol_start_idx + chol_slice_size, dtype=jnp.int32)
        )

        draws_idx_norm = jnp.array([k for k, dist in enumerate(self._rvdist) if dist in ["n", "ln"]], dtype=jnp.int32)
        draws_idx_truncnorm = jnp.array(
            [k for k, dist in enumerate(self._rvdist) if dist == "n_trunc"], dtype=jnp.int32
        )

        return (
            rand_idx_norm,
            rand_idx_truncnorm,
            rand_idx_stddev,
            rand_idx_chol,
            draws_idx_norm,
            draws_idx_truncnorm,
            sd_start_idx,
            sd_slice_size,
        )

    def set_fixed_variable_indicies(
        self, mask_chol, values_for_chol_mask, fixedvars, coef_names, sd_start_idx, sd_slice_size, betas
    ):
        mask = np.zeros(len(fixedvars), dtype=np.int32)
        values_for_mask = np.zeros(len(fixedvars), dtype=np.int32)
        for i, (k, v) in enumerate(fixedvars.items()):
            idx = np.where(coef_names == k)[0]
            if len(idx) == 0:
                raise ValueError(f"Variable {k} not found in the model.")
            if len(idx) > 1:
                raise ValueError(f"Variable {k} found more than once, this should never happen.")
            idx = idx[0]
            mask[i] = idx
            assert v is not None
            betas = betas.at[idx].set(v)
            values_for_mask[i] = v

            if (idx >= sd_start_idx) & (idx < sd_start_idx + sd_slice_size):
                mask_chol.append(idx - sd_start_idx)
                values_for_chol_mask.append(v)

        mask = jnp.array(mask)
        values_for_mask = jnp.array(values_for_mask)
        mask_chol = jnp.array(mask_chol, dtype=jnp.int32)
        values_for_chol_mask = jnp.array(values_for_chol_mask)
        return mask, values_for_mask, mask_chol, values_for_chol_mask

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

        self._validate_inputs(X, y, alts, varnames, ids, config.weights, predict_mode=predict_mode)

        self._pre_fit(alts, varnames, config.maxiter)

        (
            betas,
            X,
            y,
            panels,
            draws,
            weights,
            avail,
            Xnames,
            coef_names,
        ) = self._setup_input_data(
            X,
            y,
            varnames,
            alts,
            ids,
            randvars,
            config,
        )

        (
            rand_idx_norm,
            rand_idx_truncnorm,
            rand_idx_stddev,
            rand_idx_chol,
            draws_idx_norm,
            draws_idx_truncnorm,
            sd_start_idx,
            sd_slice_size,
        ) = self.set_variable_indices(config.include_correlations)

        # Set up index into _rvdist for lognormal distributions. This is used to apply the lognormal transformation
        # to the random betas
        idx_ln_dist = jnp.array([i for i, x in enumerate(self._rvdist) if x == "ln"], dtype=jnp.int32)

        # Mask fixed coefficients and set up array with values for the loglikelihood function
        mask = None
        values_for_mask = None
        # separate mask for fixing values of cholesky coeffs after softplus transformation
        mask_chol = []
        values_for_chol_mask = []

        if config.fixedvars is not None:
            mask, values_for_mask, mask_chol, values_for_chol_mask = self.set_fixed_variable_indicies(
                mask_chol, values_for_chol_mask, config.fixedvars, coef_names, sd_start_idx, sd_slice_size, betas
            )

        if (config.fixedvars is None) or (len(mask_chol) == 0):
            mask_chol = None
            values_for_chol_mask = None

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

        # split data for fixed and random parameters to speed up calculations
        rvidx = jnp.array(self._rvidx, dtype=bool)
        # rand_idx = jnp.where(rvidx)[0]
        fixed_idx = jnp.where(~rvidx)[0]
        Xdf = Xd[:, :, ~rvidx]  # Data for fixed parameters
        Xdr = Xd[:, :, rvidx]  # Data for random parameters

        return (
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
            rand_idx_norm,  # index into betas for normal draws
            rand_idx_truncnorm,  # index into betas for truncated normal draws
            draws_idx_norm,  # index into draws for normal draws
            draws_idx_truncnorm,  # index into draws for truncated normal draws
            fixed_idx,
            num_panels,
            idx_ln_dist,
            coef_names,
            rand_idx_stddev,
            rand_idx_chol,
        )

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

        Args:
            X: array-like, shape (n_samples*n_alts, n_variables)
                Input data for explanatory variables in long format

            y: array-like, shape (n_samples*n_alts,)
                Chosen alternatives or one-hot encoded representation of the choices

            varnames: list-like, shape (n_variables,)
                Names of explanatory variables that must match the number and order of columns in ``X``

            alts: array-like, shape (n_samples*n_alts,)
                Alternative values in long format

            ids: array-like, shape (n_samples*n_alts,)
                Identifiers for the samples in long format.

            randvars: dict
                Names (keys) and mixing distributions (values) of variables that have random parameters as coefficients.
                Possible mixing distributions are: ``'n'``: normal, ``'ln'``: lognormal, ``'t'``: triangular,
                ``'tn'``: truncated normal

            verbose: int, default=1
                Verbosity of messages to show during estimation. 0: No messages, 1: Some messages, 2: All messages

        Returns:
            Return the estimated model parameters result
        """

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
        ) = self.data_prep(
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
            config.force_positive_chol_diag,
            rand_idx_stddev,
            rand_idx_chol,
            config.batch_size,
        )

        if idx_ln_dist.shape[0] > 0:
            logger.info(
                f"Lognormal distributions found for {idx_ln_dist.shape[0]} random variables, applying transformation."
            )

        if panels is not None:
            logger.info(f"Data contains {num_panels} panels.")

        logger.debug(f"Shape of Xdf: {Xdf.shape}, shape of Xdr: {Xdr.shape}")

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
                "disp": verbose > 1,
            },
            jit_loglik=config.batch_size is None,
        )
        if optim_res is None:
            logger.error("Optimization failed, returning None.")
            return None

        logger.info(
            f"Optimization finished, success = {optim_res['success']}, final loglike = {-optim_res['fun']:.2f}"
            + f", final gradient max = {optim_res['jac'].max():.2e}, norm = {jnp.linalg.norm(optim_res['jac']):.2e}."
        )

        if config.skip_std_errs:
            logger.info("Skipping H_inv and grad_n calculation due to skip_std_errs=True")
        else:
            logger.info("Calculating gradient of individual log-likelihood contributions")
            optim_res["grad_n"] = gradient(loglike_individual, jnp.array(optim_res["x"]), *fargs[:-1])

            try:
                logger.info(
                    f"Calculating Hessian, by row={config.hessian_by_row}, finite diff={config.finite_diff_hessian}"
                )
                H = hessian(
                    neg_loglike, jnp.array(optim_res["x"]), config.hessian_by_row, config.finite_diff_hessian, *fargs
                )

                logger.info("Inverting Hessian")
                # remove masked parameters to make it invertible
                if mask is not None:
                    mask_for_hessian = jnp.array([x for x in range(0, H.shape[0]) if x not in mask])
                    h_free = H[jnp.ix_(mask_for_hessian, mask_for_hessian)]
                    h_inv_nonfixed = jax.lax.stop_gradient(jnp.linalg.inv(h_free))
                    h_inv = jnp.zeros_like(H)
                    h_inv = h_inv.at[jnp.ix_(mask_for_hessian, mask_for_hessian)].set(h_inv_nonfixed)
                else:
                    h_inv = jax.lax.stop_gradient(jnp.linalg.inv(H))

                optim_res["hess_inv"] = h_inv
            # TODO: narrow down to actual error here
            # TODO: Do we want to use Hinv = jnp.linalg.pinv(np.dot(optim_res["grad_n"].T, optim_res["grad_n"])) as fallback?
            except Exception as e:
                logger.error(f"Numerical Hessian calculation failed with {e} - parameters might not be identified")
                optim_res["hess_inv"] = jnp.eye(len(optim_res["x"]))

        self._post_fit(optim_res, coef_names, Xdf.shape[0], mask, config.fixedvars, config.skip_std_errs)
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
        ) = self.data_prep(
            X,
            None,
            varnames,
            alts,
            ids,
            randvars,
            config,
            predict_mode=True,
        )

        fargs = (
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
            config.force_positive_chol_diag,  ###
            rand_idx_stddev,
            rand_idx_chol,
        )

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


def _transform_rand_betas(
    betas,
    draws,
    # rand_idx,  # position of mean variables in betas
    rand_idx_norm,  # position of mean of norm/lognorm variables in beta
    rand_idx_truncnorm,  # position of mean of truncated normal variables beta
    draws_idx_norm,  # position of normal random variables in draws and std devs
    draws_idx_truncnorm,  # position of truncated normal random variables in draws
    rand_idx_stddev,  # position of std dev variables in betas
    rand_idx_chol,  # position of cholesky variables in betas
    idx_ln_dist,
    force_positive_chol_diag,
    mask_chol,
    values_for_chol_mask,
):
    """Compute the products between the betas and the random coefficients.

    This method also applies the associated mixing distributions
    """

    diag_vals = betas[rand_idx_stddev]  # jax.lax.dynamic_slice(betas, (sd_start_index,), (sd_slice_size,))
    if force_positive_chol_diag:
        diag_vals = jax.nn.softplus(diag_vals)
        if mask_chol is not None:
            # Apply mask to the diagonal values of the Cholesky matrix again.
            # Could work around this by setting asserted params to softplus-1(x) but we also want to ensure
            # 0 values are propagated correctly for, e.g., ECs with less than full rank cov matrix.
            diag_vals = diag_vals.at[mask_chol].set(values_for_chol_mask)

    ### Normal/lognormal part
    br_mean = betas[rand_idx_norm]
    br_std_dev = diag_vals[draws_idx_norm]
    if rand_idx_chol is not None:
        # chol_start_idx = sd_start_index + sd_slice_size
        # chol_slice_size = (sd_slice_size * (sd_slice_size + 1)) // 2 - sd_slice_size
        sd_slice_size = len(br_mean)

        # Build lower-triangular Cholesky matrix
        tril_rows, tril_cols = jnp.tril_indices(sd_slice_size)
        L = jnp.zeros((sd_slice_size, sd_slice_size), dtype=betas.dtype)
        diag_mask = tril_rows == tril_cols
        off_diag_mask = ~diag_mask
        off_diag_vals = betas[rand_idx_chol]  # jax.lax.dynamic_slice(betas, (chol_start_idx,), (chol_slice_size,))

        tril_vals = jnp.where(diag_mask, br_std_dev[tril_rows], off_diag_vals[jnp.cumsum(off_diag_mask) - 1])
        L = L.at[tril_rows, tril_cols].set(tril_vals)

        N, _, R = draws.shape
        draws_flat = draws[:, draws_idx_norm, :].transpose(0, 2, 1).reshape(-1, sd_slice_size)
        correlated_flat = (L @ draws_flat.T).T
        cov = correlated_flat.reshape(N, R, sd_slice_size).transpose(0, 2, 1)
    else:
        cov = draws[:, draws_idx_norm, :] * br_std_dev[None, :, None]

    # betas random
    betas_random = jnp.empty_like(draws)  # num_obs, num_rand_vars, num_draws

    for i, idx_norm in enumerate(draws_idx_norm):
        betas_random = betas_random.at[:, idx_norm, :].set(br_mean[None, i, None] + cov[:, i, :])

    # apply lognormal part if there are any
    betas_random = _apply_distribution(betas_random, idx_ln_dist)

    ### Truncated normal part
    br_mean = betas[rand_idx_truncnorm]
    br_std_dev = diag_vals[draws_idx_truncnorm]
    for i, idx_truncnorm in enumerate(draws_idx_truncnorm):
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
    force_positive_chol_diag,
    rand_idx_stddev,
    rand_idx_chol,
    batch_size,
):
    loglik_individ = loglike_individual(
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
        force_positive_chol_diag,
        rand_idx_stddev,
        rand_idx_chol,
    )

    loglik = loglik_individ.sum()
    return -loglik


def neg_loglike_grad_batched(
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
    force_positive_chol_diag,
    rand_idx_stddev,
    rand_idx_chol,
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
            draws[start:end, :, :],
            weights[num_panels_counter : num_panels_counter + num_panels_this_batch] if weights is not None else None,
            avail[start:end] if avail is not None else None,
            mask,
            values_for_mask,
            mask_chol,
            values_for_chol_mask,
            rand_idx_norm,
            rand_idx_truncnorm,
            draws_idx_norm,
            draws_idx_truncnorm,
            fixed_idx,
            num_panels_this_batch,
            idx_ln_dist,
            force_positive_chol_diag,
            rand_idx_stddev,
            rand_idx_chol,
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
    force_positive_chol_diag,
    rand_idx_stddev,
    rand_idx_chol,
):
    ll = loglike_individual(
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
        force_positive_chol_diag,
        rand_idx_stddev,
        rand_idx_chol,
    )
    return ll.sum()


loglike_and_grad_individual = jax.jit(
    jax.value_and_grad(loglike_individual_sum, argnums=0),
    static_argnames=["num_panels", "force_positive_chol_diag"],
)


def loglike_individual(
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
    force_positive_chol_diag,
    rand_idx_stddev,
    rand_idx_chol,
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
    if mask is not None:
        betas = betas.at[mask].set(values_for_mask)

    # Utility for fixed parameters
    Bf = betas[fixed_idx]  # Fixed betas
    Vdf = jnp.einsum("njk,k -> nj", Xdf, Bf)  # (N, J-1)

    # Utility for random parameters
    Br = _transform_rand_betas(
        betas,
        draws,
        rand_idx_norm,
        rand_idx_truncnorm,
        draws_idx_norm,
        draws_idx_truncnorm,
        rand_idx_stddev,
        rand_idx_chol,
        idx_ln_dist,
        force_positive_chol_diag,
        mask_chol,
        values_for_chol_mask,
    )  # Br shape: (num_obs, num_rand_vars, num_draws)

    # Vdr shape: (N,J-1,R)
    Vd = Vdf[:, :, None] + jnp.einsum("njk,nkr -> njr", Xdr, Br)
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

    loglik = jnp.log(jnp.clip(proba_n.sum(axis=1) / draws.shape[2], LOG_PROB_MIN, jnp.inf))

    if weights is not None:
        loglik = loglik * weights

    return loglik


def probability_individual(
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
    force_positive_chol_diag,
    rand_idx_stddev,
    rand_idx_chol,
):
    """Compute the probabilities of all alternatives."""

    if jax.config.jax_enable_x64:
        UTIL_MAX = 700  # ONLY IF 64bit precision is used
    else:
        UTIL_MAX = 87

    R = draws.shape[2]

    # mask for asserted parameters.
    if mask is not None:
        betas = betas.at[mask].set(values_for_mask)

    # Utility for fixed parameters
    Bf = betas[fixed_idx]  # Fixed betas
    Vdf = jnp.einsum("njk,k -> nj", Xdf, Bf)  # (N, J)

    Br = _transform_rand_betas(
        betas,
        draws,
        rand_idx_norm,
        rand_idx_truncnorm,
        draws_idx_norm,
        draws_idx_truncnorm,
        rand_idx_stddev,
        rand_idx_chol,
        idx_ln_dist,
        force_positive_chol_diag,
        mask_chol,
        values_for_chol_mask,
    )

    # Vdr shape: (N,J,R)
    Vd = Vdf[:, :, None] + jnp.einsum("njk,nkr -> njr", Xdr, Br)
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
