# -*- coding: utf-8 -*-
import numpy as np
import pytest
from pytest import approx

from jaxlogit.mixed_logit import (
    MixedLogit
)

X = np.array([[2, 1], [1, 3], [3, 1], [2, 4], [2, 1], [2, 4]])
y = np.array([0, 1, 0, 1, 0, 1])
ids = np.array([1, 1, 2, 2, 3, 3])
alts = np.array([1, 2, 1, 2, 1, 2])
varnames = ["a", "b"]
N, J, K = 3, 2, 2


def test__setup_design_matrix():
    """
    Ensures that jaxlogit shapes data properly
    Inspired heavily by tests in xlogit
    """
    model = MixedLogit()
    model._pre_fit(alts, varnames, 0)
    X_, Xnames_ = model._setup_design_matrix(X)
    assert X_.shape == (3, 2, 2)
    assert list(Xnames_) == ["a", "b"]


def test__validate_inputs():
    """
    Covers potential mistakes in parameters of the fit method that xlogit
    should be able to identify

    Adapted from xlogit tests
    """
    model = MixedLogit()
    validate = model._validate_inputs
    with pytest.raises(ValueError):  # match between columns in X and varnames
        validate(X, y, alts, varnames=["a"], ids=ids, weights=None)

    with pytest.raises(ValueError):  # alts can't be None
        validate(X, y, None, varnames=["a"], ids=ids, weights=None)

    with pytest.raises(ValueError):  # varnames can't be None
        validate(X, y, alts, None, ids=ids, weights=None)

    with pytest.raises(ValueError):  # X dimensions
        validate(np.array([]), y, alts, varnames=None, ids=ids, weights=None)

    with pytest.raises(ValueError):  # y dimensions
        validate(X, np.array([]), alts, varnames=None, ids=ids, weights=None)

def test__format_choice_var():
    """
    Ensures that the variable y is properly formatted as needed by internal
    procedures regardless of the input data type.
    Adapted from xlogit tests
    """
    model = MixedLogit()
    expected = np.array([1, 0, 0, 1, 1, 0])

    y1 = np.array([1, 1, 2, 2, 1, 1])
    assert np.array_equal(model._format_choice_var(y1, alts), expected)

    y2 = np.array(['a', 'a', 'b', 'b', 'a', 'a'])
    alts2 = np.array(['a', 'b', 'a', 'b', 'a', 'b',])
    assert np.array_equal(model._format_choice_var(y2, alts2), expected)

def test__robust_covariance():
    """
    Ensures that the robust covariance is estimated properly.
    Output is tested against results calculated in spreadsheet software.

    Adapted from xlogit tests
    """
    hess_inv = np.array([[1, .5], [.5, 4]])
    grad_n = np.array([[0, 0], [.05, .05], [-0.05, -0.05]])

    robust_cov = np.array([[0.016875, 0.050625], [0.050625, 0.151875]])

    model = MixedLogit()

    test_robust_cov = model._robust_covariance(hess_inv, grad_n)

    sum_sq_diff = np.sum(np.power(robust_cov-test_robust_cov,2))

    assert sum_sq_diff == approx(0)
