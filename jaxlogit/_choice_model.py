import logging

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import t
from time import time
from abc import ABC

logger = logging.getLogger(__name__)


class ChoiceModel(ABC):  # noqa: B024
    """Base class for estimation of discrete choice models."""

    def __init__(self):
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.total_fun_eval = 0

    def _reset_attributes(self):
        self.coeff_names = None
        self.coeff_ = None
        self.stderr = None
        self.zvalues = None
        self.pvalues = None
        self.loglikelihood = None
        self.total_fun_eval = 0

    def _as_array(
        self,
        X,
        y,
        varnames,
        alts,
        ids,
        weights,
        panels,
        avail,
    ):
        """Returns given parameters as np arrays if they exist"""
        toTransform = (
            X,
            y,
            varnames,
            alts,
            ids,
            weights,
            panels,
            avail,
        )
        transformed = []
        for array in toTransform:
            transformed.append(np.asarray(array) if array is not None else None)
        return tuple(transformed)

    def _pre_fit(self, alts, varnames, maxiter):
        self._reset_attributes()
        self._fit_start_time = time()
        self._varnames = list(varnames)  # Easier to handle with lists
        self.alternatives = np.sort(np.unique(alts))
        self.maxiter = maxiter

    def _post_fit(
        self,
        optim_res,
        coeff_names,
        sample_size,
        mask=None,
        fixedvars=None,
        skip_std_errors=False,
    ):
        logger.info("Post fit processing")
        self.convergence = optim_res["success"]
        self.coeff_ = optim_res["x"]

        if skip_std_errors:
            self.covariance = jnp.eye(len(optim_res["x"]), len(optim_res["x"]))
        else:
            self.grad_n = optim_res["grad_n"]
            self.hess_inv = optim_res["hess_inv"]
            self.covariance = jax.lax.stop_gradient(self._robust_covariance(optim_res["hess_inv"], optim_res["grad_n"]))
            if mask is not None:
                self.covariance = self.covariance.at[mask, mask].set(0)
        self.stderr = jnp.sqrt(jnp.diag(self.covariance))
        # masked values lead to zero division warning - ignore
        with np.errstate(divide="ignore"):
            self.zvalues = self.coeff_ / self.stderr
        self.pvalues = 2 * t.cdf(-np.abs(self.zvalues), df=sample_size)
        self.loglikelihood = -optim_res["fun"]
        self.estimation_message = optim_res["message"]
        self.coeff_names = coeff_names
        self.total_iter = optim_res["nit"]
        self.estim_time_sec = time() - self._fit_start_time
        self.sample_size = sample_size
        corr_ = 0 if fixedvars is None else len(fixedvars)
        self.aic = 2.0 * (len(self.coeff_) - corr_ - self.loglikelihood)
        self.bic = np.log(sample_size) * (len(self.coeff_) - corr_) - 2.0 * self.loglikelihood
        self.total_fun_eval = optim_res["nfev"]
        self.mask = mask
        self.fixedvars = fixedvars

        if not self.convergence:
            logger.warning("**** The optimization did not converge after {} iterations. ****".format(self.total_iter))
            logger.info("Message: " + optim_res["message"])

    def _robust_covariance(self, hess_inv, grad_n):
        """Estimates the robust covariance matrix.

        This follows the methodology lined out in p.486-488 in the Stata 16 reference manual.
        Benchmarked against Stata 17.
        """
        n = grad_n.shape[0]
        grad_n_sub = grad_n - (jnp.sum(grad_n, axis=0) / n)  # subtract out mean gradient value
        inner = jnp.transpose(grad_n_sub) @ grad_n_sub
        correction = (n) / (n - 1)
        covariance = correction * (hess_inv @ inner @ hess_inv)
        return covariance

    def _setup_design_matrix(self, X):
        """Sets up and reshapes input data."""
        J = len(self.alternatives)
        N = int(len(X) / J)
        varnames = self._varnames.copy()

        # TODO: are the following two lines still necessary?
        aspos = [varnames.index(i) for i in varnames]  # Position of AS vars
        X = X[:, aspos]

        X = X.reshape(N, J, -1)

        return X, np.array(varnames)

    def _check_long_format_consistency(self, ids, alts):
        """Ensure that data in long format is consistent.

        It raises an error if the array of alternative indexes is incomplete
        """
        if ids is None or alts is None:
            raise ValueError("no inputs can be None")
        uq_alts, idx = np.unique(alts, return_index=True)
        uq_alts = uq_alts[np.argsort(idx)]
        expected_alts = np.tile(uq_alts, int(len(ids) / len(uq_alts)))
        if not np.array_equal(alts, expected_alts):
            raise ValueError(f"inconsistent alts values in long format, expected {expected_alts}, got {uq_alts}")
        _, obs_by_id = np.unique(ids, return_counts=True)
        if not np.all(obs_by_id % len(uq_alts) == 0):  # Multiple of J
            raise ValueError("inconsistent alts and ids values in long format")

    def _format_choice_var(self, y, alts):
        """Format choice (y) variable as one-hot encoded."""
        uq_alts = np.unique(alts)
        J, N = len(uq_alts), len(y) // len(uq_alts)
        # When already one-hot encoded the sum by row is one
        if isinstance(y[0], (np.number, np.bool_)) and np.array_equal(y.reshape(N, J).sum(axis=1), np.ones(N)):
            return y
        else:
            y1h = (y == alts).astype(int)  # Apply one hot encoding
            if np.array_equal(y1h.reshape(N, J).sum(axis=1), np.ones(N)):
                return y1h
            else:
                raise ValueError("inconsistent 'y' values. Make sure the data has one choice per sample")

    def _validate_inputs(self, X, y, alts, varnames, ids, weights, predict_mode=False):
        """Validate potential mistakes in the input data."""
        if varnames is None:
            raise ValueError("The parameter varnames is required")
        if alts is None:
            raise ValueError("The parameter alternatives is required")
        if X.ndim != 2:
            raise ValueError("X must be an array of two dimensions in long format")
        if not predict_mode and y.ndim != 1:
            raise ValueError("y must be an array of one dimension in long format")
        if len(varnames) != X.shape[1]:
            raise ValueError("The length of varnames must match the number of columns in X")

    def summary(self):
        """Show estimation results in console."""
        if self.coeff_ is None:
            logger.info("The current model has not been estimated.")
            return
        if self.convergence:
            logger.info("Optimization terminated successfully.")
        else:
            logger.warning("Convergence not reached. The estimates may not be reliable.")

        print("    Message: {}".format(self.estimation_message))
        print("    Iterations: {}".format(self.total_iter))
        print("    Function evaluations: {}".format(self.total_fun_eval))
        print("Estimation time= {:.1f} seconds".format(self.estim_time_sec))
        print("-" * 75)
        print("{:19} {:>13} {:>13} {:>13} {:>13}".format("Coefficient", "Estimate", "Std.Err.", "z-val", "P>|z|"))
        print("-" * 75)
        fmt = "{:19} {:13.7f} {:13.7f} {:13.7f} {:13.3g} {:3}"
        for i in range(len(self.coeff_)):
            signif = ""
            if self.pvalues[i] < 0.001:
                signif = "***"
            elif self.pvalues[i] < 0.01:
                signif = "**"
            elif self.pvalues[i] < 0.05:
                signif = "*"
            elif self.pvalues[i] < 0.1:
                signif = "."
            print(
                fmt.format(
                    self.coeff_names[i][:19],
                    self.coeff_[i],
                    self.stderr[i],
                    self.zvalues[i],
                    self.pvalues[i],
                    signif,
                )
            )
        print("-" * 75)
        print("Significance:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print("")
        print("Log-Likelihood= {:.3f}".format(self.loglikelihood))
        print("AIC= {:.3f}".format(self.aic))
        print("BIC= {:.3f}".format(self.bic))


def diff_nonchosen_chosen(X, y, avail):
    # Setup Xd as Xij - Xi* (difference between non-chosen and chosen alternatives)
    N, J, K = X.shape  # number of choice situations, alternatives, and variables
    X, y = (
        X.reshape(N * J, K),
        y.astype(bool).reshape(
            N * J,
        ),
    )
    Xd = X[~y, :].reshape(N, -1, K) - X[y, :].reshape(N, -1, K)
    avail = avail.reshape(N * J)[~y].reshape(N, J - 1) if avail is not None else None
    return Xd, avail
