from jaxlogit._choice_model import ChoiceModel, diff_nonchosen_chosen
import numpy as np
import pandas as pd
import pytest
from pytest import approx
from jaxlogit.mixed_logit import MixedLogit
from time import time
from jax.scipy.optimize import OptimizeResults

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
    # Values are random and made up purely for testing purposes
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


ys = [np.asarray(pd.Series([1, 0, 0, 1])), np.array([1, 0, 0, 1]), np.array([1, 0, 1, 0])]


@pytest.mark.parametrize("y", ys)
def test__format_choice_var_y(setup, y):
    choiceModel = setup
    assert np.array_equal(y, choiceModel._format_choice_var(y, alts))


def test__format_choice_var_y_bad_input(setup):
    choiceModel = setup
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
    with pytest.raises(
        ValueError, match=r"The length of varnames must match the number of columns in X"
    ):  # match between columns in X and varnames
        validate(X, y, alts, varnames=["a"], weights=None)

    with pytest.raises(ValueError, match=r"The parameter alternatives is required"):
        validate(X, y, None, varnames=["a"], weights=None)

    with pytest.raises(ValueError, match=r"The parameter varnames is required"):
        validate(X, y, alts, None, weights=None)

    with pytest.raises(ValueError, match=r"X must be an array of two dimensions in long format"):  # X dimensions
        validate(np.array([]), y, alts, varnames=["a", "b"], weights=None)

    with pytest.raises(ValueError, match=r"y must be an array of one dimension in long format"):  # y dimensions
        validate(X, np.array(0), alts, varnames=["a", "b"], weights=None, predict_mode=False)

    with pytest.raises(ValueError, match=r"The length of weights must be divisble by the first two dimensions of X"):
        validate(X, np.array([1, 2]), alts, varnames=["a", "b"], weights=np.ones(7))

    validate(X, y, alts, varnames, np.ones(6))


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
    Xd, _ = diff_nonchosen_chosen(X_, y, None)
    expected = np.array([[[-1, 0], [-2, 2]], [[0, 0], [0, -3]]])
    assert np.array_equal(expected, Xd)


@pytest.fixture
def fit_setup(setup):
    choiceModel = setup
    choiceModel._fit_start_time = time() - 5
    optim_res = OptimizeResults(
        np.array([10, 11, 12]), True, -1, 5.12, None, np.array([[1, 0.5], [0.5, 4]]), 100, 9, None
    )
    coeff_names = ["a", "b", "c"]
    sample_size = 2
    fixedvars = {"a": 1.0}
    return choiceModel, optim_res, coeff_names, sample_size, fixedvars


def test_post_fit_basic(fit_setup):
    choiceModel, optim_res, coeff_names, sample_size, fixedvars = fit_setup
    choiceModel._post_fit(optim_res, coeff_names, sample_size, None, fixedvars, True)

    assert np.array_equal(coeff_names, choiceModel.coeff_names)
    # assert optim_res.message == choiceModel.estimation_message
    assert choiceModel.total_iter == optim_res.nit
    assert 5 == pytest.approx(choiceModel.estim_time_sec, abs=2)
    assert sample_size == choiceModel.sample_size
    assert choiceModel.total_fun_eval == optim_res.nfev
    assert choiceModel.loglikelihood == -5.12
    assert choiceModel.aic == -2 * choiceModel.loglikelihood + 2 * 2
    assert choiceModel.bic == pytest.approx(-2 * choiceModel.loglikelihood + 2 * np.log(sample_size))
    assert choiceModel.mask is None


def test_post_fit_skip_stderr(fit_setup):
    choiceModel, optim_res, coeff_names, sample_size, fixedvars = fit_setup
    choiceModel._post_fit(optim_res, coeff_names, sample_size, None, fixedvars, True)

    assert np.array_equal(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), choiceModel.covariance)
    assert np.array_equal(np.array([1, 1, 1]), choiceModel.stderr)
    assert np.array_equal(np.array([10, 11, 12]), choiceModel.zvalues)


def test_post_fit_stderr(fit_setup):
    choiceModel, optim_res, coeff_names, sample_size, fixedvars = fit_setup
    optim_res = OptimizeResults(np.array([10, 11]), True, -1, 5.12, None, np.array([[1, 0.5], [0.5, 4]]), 100, 9, None)
    grad_n = np.array([[0, 0], [0.05, 0.05], [-0.05, -0.05]])
    coeff_names = ["a", "b"]
    # reuse values from test_robust_covariance
    choiceModel._post_fit(
        optim_res, coeff_names, sample_size, 2, fixedvars, False, hess_inv=optim_res.hess_inv, grad_n=grad_n
    )

    assert np.array_equal(choiceModel.hess_inv, optim_res.hess_inv)
    expected = np.array([[0.016875, 0.050625], [0.050625, 0.15187502]])
    for i in range(len(expected)):
        for j in range(len(expected[i])):
            assert expected[i][j] == pytest.approx(choiceModel.covariance[i][j])

    # this functionality is further tested in the system tests
    expected = np.array([0.12990381, 0.38971147])
    for i in range(len(expected)):
        assert expected[i] == pytest.approx(choiceModel.stderr[i])
    expected = np.array([76.980034, 28.226011])
    for i in range(len(expected)):
        assert expected[i] == pytest.approx(choiceModel.zvalues[i])
    expected = np.array([0.000168707, 0.00125281])
    for i in range(len(expected)):
        assert expected[i] == pytest.approx(choiceModel.pvalues[i], rel=1e-3)
