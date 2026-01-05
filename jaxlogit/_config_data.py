from dataclasses import dataclass
import numpy as np
from typing import Any, Union, Sequence
from pandas import Series

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

        set_vars: dict, default=None
            Specified variable names (keys) of variables to be set to the given value (values)

        optim_method: str, default="trust-region"
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

    weights: ArrayLike | None = None
    avail: ArrayLike | None = None
    panels: ArrayLike | None = None
    init_coeff: ArrayLike | None = None
    maxiter: int = 2000
    random_state: int | None = None
    n_draws: int = 1000
    halton: bool = True
    halton_opts: dict | None = None
    tol_opts: dict | None = None
    num_hess: bool = False
    set_vars: dict[str, float] | None = None
    optim_method: str = "trust-region"
    skip_std_errs: bool = False
    include_correlations: bool = False
    force_positive_chol_diag: bool = True
    hessian_by_row: bool = True
    finite_diff_hessian: bool = False
    batch_size: int | None = None
