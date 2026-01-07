from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data

from jaxlogit.mixed_logit import MixedLogit
from jaxlogit._config_data import ConfigData

import numpy as np


class MixedLogitEstimator(BaseEstimator):
    def __init__(
        self,
        weights=None,
        avail=None,
        panels=None,
        init_coeff=None,
        maxiter=2000,
        random_state=None,
        n_draws=1000,
        halton=True,
        halton_opts=None,
        tol_opts=None,
        num_hess=False,
        set_vars=None,
        optim_method="L-BFGS-B",
        skip_std_errs=False,
        include_correlations=False,
        force_positive_chol_diag=True,
        hessian_by_row=True,
        finite_diff_hessian=False,
        batch_size=None,
        verbose=1,
    ):
        # Store all parameters as attributes
        self.weights = weights
        self.avail = avail
        self.panels = panels
        self.init_coeff = init_coeff
        self.maxiter = maxiter
        self.random_state = random_state
        self.n_draws = n_draws
        self.halton = halton
        self.halton_opts = halton_opts
        self.tol_opts = tol_opts
        self.num_hess = num_hess
        self.set_vars = set_vars
        self.optim_method = optim_method
        self.skip_std_errs = skip_std_errs
        self.include_correlations = include_correlations
        self.force_positive_chol_diag = force_positive_chol_diag
        self.hessian_by_row = hessian_by_row
        self.finite_diff_hessian = finite_diff_hessian
        self.batch_size = batch_size
        self.verbose = verbose

    def _get_config(self):
        """Create ConfigData instance from estimator attributes."""
        return ConfigData(
            weights=self.weights,
            avail=self.avail,
            panels=self.panels,
            init_coeff=self.init_coeff,
            maxiter=self.maxiter,
            random_state=self.random_state,
            n_draws=self.n_draws,
            halton=self.halton,
            halton_opts=self.halton_opts,
            tol_opts=self.tol_opts,
            num_hess=self.num_hess,
            set_vars=self.set_vars,
            optim_method=self.optim_method,
            skip_std_errs=self.skip_std_errs,
            include_correlations=self.include_correlations,
            force_positive_chol_diag=self.force_positive_chol_diag,
            hessian_by_row=self.hessian_by_row,
            finite_diff_hessian=self.finite_diff_hessian,
            batch_size=self.batch_size,
        )

    def fit(self, X, y, varnames=None, alts=None, ids=None, randvars=None):
        """Fit Mixed Logit model.

        Parameters
        ----------
        X : array-like of shape (n_samples*n_alts, n_features)
            Input data for explanatory variables in long format.
        y : array-like of shape (n_samples*n_alts,)
            Chosen alternatives or one-hot encoded representation of the choices.
        varnames : list-like of shape (n_features,), required
            Names of explanatory variables that must match the number and order of
            columns in ``X``.
        alts : array-like of shape (n_samples*n_alts,), required
            Alternative values in long format.
        ids : array-like of shape (n_samples*n_alts,), required
            Identifiers for the samples in long format.
        randvars : dict, required
            Names (keys) and mixing distributions (values) of variables that have
            random parameters as coefficients. Possible mixing distributions are:
            - ``'n'``: normal
            - ``'ln'``: lognormal
            - ``'t'``: triangular
            - ``'tn'``: truncated normal

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        Additional configuration options such as `weights`, `avail`, `panels`,
        `maxiter`, `n_draws`, and optimization settings are specified in `__init__`.
        """
        # Validate required parameters
        if varnames is None:
            raise ValueError(
                "varnames is required. Provide a list of variable names that match the number and order of columns in X."
            )
        if alts is None:
            raise ValueError("alts is required. Provide alternative identifiers in long format.")
        if ids is None:
            raise ValueError("ids is required. Provide sample identifiers in long format.")
        if randvars is None:
            raise ValueError(
                "randvars is required. Provide a dict with variable names as keys "
                "and mixing distributions ('n', 'ln', 't', 'tn') as values."
            )

        # Validate data format
        X, y = validate_data(self, X=X, y=y, accept_sparse=False, dtype="numeric")

        # Create config from estimator attributes
        config = self._get_config()

        # Initialize and fit the underlying MixedLogit model
        self.model_ = MixedLogit()
        self.model_.fit(X, y, varnames, alts, ids, randvars, config, self.verbose)

        # Store sklearn-required attributes
        self.classes_ = np.unique(y)  # Required by ClassifierMixin

        # Expose fitted model attributes at wrapper level
        self.convergence = self.model_.convergence
        self.coeff_ = self.model_.coeff_
        self.covariance = self.model_.covariance
        self.stderr = self.model_.stderr
        self.zvalues = self.model_.zvalues
        self.pvalues = self.model_.pvalues
        self.loglikelihood = self.model_.loglikelihood
        self.estimation_message = self.model_.estimation_message
        self.coeff_names = self.model_.coeff_names
        self.total_iter = self.model_.total_iter
        self.estim_time_sec = self.model_.estim_time_sec
        self.sample_size = self.model_.sample_size
        self.aic = self.model_.aic
        self.bic = self.model_.bic
        self.total_fun_eval = self.model_.total_fun_eval
        self.mask = self.model_.mask
        self.fixedvars = self.model_.fixedvars

        # Conditionally expose gradient and hessian if available
        if hasattr(self.model_, "grad_n"):
            self.grad_n = self.model_.grad_n
        if hasattr(self.model_, "hess_inv"):
            self.hess_inv = self.model_.hess_inv

        return self


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(MixedLogitEstimator())
