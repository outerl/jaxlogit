from jaxlogit._choice_model import ChoiceModel, diff_nonchosen_chosen
import numpy as np
import pandas as pd
import pytest
from pytest import approx
from jaxlogit.mixed_logit import MixedLogit
from time import time

X = np.array([[2, 1], [1, 3], [3, 1], [2, 4], [2, 1], [2, 4]])
y = np.array([0, 1, 0, 1, 0, 1])
ids = np.array([1, 1, 2, 2, 3, 3])
alts = np.array([1, 2, 1, 2, 1, 2])
varnames = ["a", "b"]
N, J, K = 3, 2, 2


@pytest.fixture
def setup():
    choiceModel = ChoiceModel()
    choiceModel.alternatives = alts
    choiceModel._varnames = varnames
    return choiceModel


def test__reset_attributes(setup):
    choiceModel = setup
    choiceModel.coeff_names = varnames
    choiceModel.coeff_ = {"key": 1}
    choiceModel.stderr = 10
    choiceModel.zvalues = 0.1
    choiceModel.pvalues = 0.2
    choiceModel.loglikelihood = 0.5
    choiceModel.total_fun_eval = 5
    choiceModel._reset_attributes()
    assert choiceModel.coeff_names is None
    assert choiceModel.coeff_ is None
    assert choiceModel.stderr is None
    assert choiceModel.zvalues is None
    assert choiceModel.pvalues is None
    assert choiceModel.loglikelihood is None
    assert 0 == choiceModel.total_fun_eval


def test__pre_fit(setup):
    choiceModel = setup
    choiceModel.coeff_names = ["a"]
    choiceModel.coeff_ = {"key": 1}
    choiceModel.stderr = 10
    choiceModel.zvalues = 0.1
    choiceModel.pvalues = 0.2
    choiceModel.loglikelihood = 0.5
    choiceModel.total_fun_eval = 5

    choiceModel._pre_fit(alts, varnames, 100)
    assert choiceModel._fit_start_time == pytest.approx(time(), abs=1)

    assert choiceModel.coeff_names is None
    assert choiceModel.coeff_ is None
    assert choiceModel.stderr is None
    assert choiceModel.zvalues is None
    assert choiceModel.pvalues is None
    assert choiceModel.loglikelihood is None
    assert 0 == choiceModel.total_fun_eval
    assert np.array_equal(choiceModel.alternatives, np.sort(np.unique(alts)))
    assert choiceModel.maxiter == 100


def test__setup_design_matrix_smoke_test(setup):
    choiceModel = setup
    choiceModel.alternatives = np.sort(np.unique(alts))
    obtained = choiceModel._setup_design_matrix(X)
    assert obtained[0].shape == (3, 2, 2)
    assert varnames == list(obtained[1])


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


def test__check_long_format_consistency(setup):
    choiceModel = setup
    with pytest.raises(ValueError):
        choiceModel._check_long_format_consistency(None, alts)
    with pytest.raises(ValueError):
        choiceModel._check_long_format_consistency(ids, None)
    with pytest.raises(ValueError):
        choiceModel._check_long_format_consistency(np.unique(ids), np.unique(alts))
    with pytest.raises(ValueError):
        choiceModel._check_long_format_consistency(np.unique(ids), alts)
    with pytest.raises(ValueError):
        choiceModel._check_long_format_consistency([1, 2, 3, 4, 5], [1, 2, 3])
    choiceModel._check_long_format_consistency(ids, alts)


def test__format_choice_var_y(setup):
    choiceModel = setup

    y = np.asarray(pd.Series([1, 0, 0, 1]))
    assert np.array_equal(y, choiceModel._format_choice_var(y, alts))

    y = np.array([1, 0, 0, 1])
    assert np.array_equal(y, choiceModel._format_choice_var(y, alts))

    y = np.array([1, 0, 1, 0])
    assert np.array_equal(y, choiceModel._format_choice_var(y, alts))

    with pytest.raises(ValueError):
        y = np.array([1, 0, 1, 0, 0])
        assert np.array_equal(y, choiceModel._format_choice_var(y, alts))


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

    y2 = np.array(["a", "a", "b", "b", "a", "a"])
    alts2 = np.array(
        [
            "a",
            "b",
            "a",
            "b",
            "a",
            "b",
        ]
    )
    assert np.array_equal(model._format_choice_var(y2, alts2), expected)


def test__robust_covariance():
    """
    Ensures that the robust covariance is estimated properly.
    Output is tested against results calculated in spreadsheet software.

    Adapted from xlogit tests
    """
    hess_inv = np.array([[1, 0.5], [0.5, 4]])
    grad_n = np.array([[0, 0], [0.05, 0.05], [-0.05, -0.05]])

    robust_cov = np.array([[0.016875, 0.050625], [0.050625, 0.151875]])

    model = MixedLogit()

    test_robust_cov = model._robust_covariance(hess_inv, grad_n)

    sum_sq_diff = np.sum(np.power(robust_cov - test_robust_cov, 2))

    assert sum_sq_diff == approx(0)


def test_diff_nonchosen_chosen(setup):
    X_ = np.array([np.array([[2, 1], [1, 3], [3, 1]]), np.array([[2, 4], [2, 1], [2, 4]])])
    y = np.array([0, 0, 1, 0, 0, 1])
    Xd, avail = diff_nonchosen_chosen(X_, y, None)
    expected = np.array([[[-1, 0], [-2, 2]], [[0, 0], [0, -3]]])
    assert np.array_equal(expected, Xd)
