from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


from jaxlogit.mixed_logit import MixedLogit
from jaxlogit._config_data import ConfigData
from jaxlogit.utils import wide_to_long


import numpy as np


class MixedLogitEstimator(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        alternatives=(),
        varnames=(),
        randvars=(),
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
        """Initialises a jaxlogit estimator with configurations for the fit and predict functions.

        Parameters
        ----------
        alternatives : list-like
            Names of valid alternatives that were chosen.
        varnames : list-like of shape (n_features,), required
            Names of explanatory variables that must match the number and order of
            columns in ``X``.
        randvars : dict, required
            Names (keys) and mixing distributions (values) of variables that have
            random parameters as coefficients. Possible mixing distributions are:
            - ``'n'``: normal
            - ``'ln'``: lognormal
            - ``'t'``: triangular
            - ``'tn'``: truncated normal
        weights : array-like, shape (n_samples,), optional
            Sample weights in long format.
        avail : array-like, shape (n_samples*n_alts,), optional
            Availability of alternatives for the choice situations. One when
            available or zero otherwise.
        panels : array-like, shape (n_samples*n_alts,), optional
            Identifiers in long format to create panels in combination with ``ids``.
        init_coeff : numpy.ndarray, shape (n_variables,), optional
            Initial coefficients for estimation.
        maxiter : int, default=2000
            Maximum number of iterations.
        random_state : int, optional
            Random seed for numpy random generator.
        n_draws : int, default=1000
            Number of random draws to approximate the mixing distributions of the
            random coefficients.
        halton : bool, default=True
            Whether the estimation uses halton draws.
        halton_opts : dict, optional
            Options for generation of halton draws. The dictionary accepts the
            following options (keys):

            - shuffle : bool, default=False
                Whether the Halton draws should be shuffled.
            - drop : int, default=100
                Number of initial Halton draws to discard to minimize correlations
                between Halton sequences.
            - primes : list
                List of primes to be used as base for generation of Halton sequences.
        tol_opts : dict, optional
            Options for tolerance of optimization routine. The dictionary accepts
            the following options (keys):

            - ftol : float, default=1e-10
                Tolerance for objective function (log-likelihood).
            - gtol : float, default=1e-5
                Tolerance for gradient function.
        num_hess : bool, default=False
            Whether numerical hessian should be used for estimation of standard errors.
        set_vars : dict, optional
            Specified variable names (keys) of variables to be set to the given
            value (values).
        optim_method : {'trust-region', 'BFGS', 'L-BFGS-B'}, default='L-BFGS-B'
            Optimization method to use for model estimation.
        skip_std_errs : bool, default=False
            Whether estimation of standard errors should be skipped.
        include_correlations : bool, default=False
            Whether correlations between variables should be included as explanatory
            variables.
        force_positive_chol_diag : bool, default=True
            Whether to force positive diagonal elements in Cholesky decomposition.
        hessian_by_row : bool, default=True
            Whether to calculate the hessian row by row in a for loop to save
            memory at the expense of runtime.
        finite_diff_hessian : bool, default=False
            Whether the hessian should be computed using finite difference. If True,
            this will stay within memory limits.
        batch_size : int, optional
            Size of batches used to avoid GPU memory overflow.

        """
        # Store all parameters as attributes
        self.alternatives = alternatives
        self.varnames = varnames
        self.randvars = randvars
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

    def predict(self, X, y=None):
        if self.coeff_ is None:
            raise NotFittedError

        randvars = dict(self.randvars)

        # Input validation
        # X = validate_data(self, X, reset=False)

        # create config from estimator attributes
        config = self._get_config()
        config.init_coeff = self.coeff_

        modified_X = X.assign(CHOICE=np.tile("None", len(X)))
        modified_X.insert(loc=0, column="custom_id", value=np.arange(len(X)))

        df = wide_to_long(
            modified_X,
            id_col="custom_id",  # internal names can be hard coded
            alt_name="alt",
            sep="_",
            alt_list=self.classes_,
            empty_val=0,
            varying=self.varnames,
            alt_is_prefix=True,
        )

        # initialize and fit the underlying MixedLogit model
        self.model_ = MixedLogit()
        mean_probabilities = self.model_.predict(
            df[self.varnames], self.varnames, self.alts, self.ids, randvars, config
        )

        predicted_alternatives_indicies = np.argmax(mean_probabilities, axis=1)

        predicted_alternatives = np.array([self.classes_[index] for index in predicted_alternatives_indicies])

        return predicted_alternatives

    def fit(self, X, y):
        """Fit Mixed Logit model.

        Parameters
        ----------
        X : array-like of shape (n_samples*n_alts, n_features)
            Input data for explanatory variables in long format with alternative and ids in line.
        y : array-like of shape (n_samples*n_alts,)
            Chosen alternatives or one-hot encoded representation of the choices.
        # alts : array-like of shape (n_samples*n_alts,), required
        #     Alternative values in long format.
        # ids : array-like of shape (n_samples*n_alts,), required
        #     Identifiers for the samples in long format.


        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        Additional configuration options such as `weights`, `avail`, `panels`,
        `maxiter`, `n_draws`, and optimization settings are specified in `__init__`.
        """
        if self.alternatives is None or self.alternatives == ():
            self.classes_ = np.unique(y)
        else:
            self.classes_ = self.alternatives

        randvars = dict(self.randvars)

        # Validate required parameters

        # validate data format
        # X, y = validate_data(self, X=X, y=y, accept_sparse=False, dtype="numeric")

        # create config from estimator attributes
        config = self._get_config()

        # send to long format
        modified_X = X.assign(CHOICE=y)
        modified_X.insert(loc=0, column="custom_id", value=np.arange(len(X)))

        df = wide_to_long(
            modified_X,
            id_col="custom_id",  # internal names can be hard coded
            alt_name="alt",
            sep="_",
            alt_list=self.classes_,
            empty_val=0,
            varying=self.varnames,
            alt_is_prefix=True,
        )

        self.alts = df["alt"]
        self.ids = df["custom_id"]

        # initialize and fit the underlying MixedLogit model
        self.model_ = MixedLogit()
        self.model_.fit(
            df[self.varnames], df["CHOICE"], self.varnames, self.alts, self.ids, randvars, config, self.verbose
        )

        # expose fitted model attributes at wrapper level
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

        # expose gradient and hessian if available
        if hasattr(self.model_, "grad_n"):
            self.grad_n = self.model_.grad_n
        if hasattr(self.model_, "hess_inv"):
            self.hess_inv = self.model_.hess_inv

        return self
