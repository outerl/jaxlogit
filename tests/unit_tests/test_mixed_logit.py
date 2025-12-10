import numpy as np
import pytest
from jaxlogit.mixed_logit import MixedLogit, ConfigData

# Setup data used for tests
X = np.array([[2, 1], [1, 3], [3, 1], [2, 4], [2, 1], [2, 4]])
y = np.array([0, 1, 0, 1, 0, 1])
ids = np.array([1, 1, 2, 2, 3, 3])
alts = np.array([1, 2, 1, 2, 1, 2])
panels = np.array([1, 1, 1, 1, 2, 2])
varnames = ["a", "b"]
randvars = {"a": "n", "b": "n"}
N, J, K, R = 3, 2, 2, 5

MIN_COMP_ZERO = 1e-300
MAX_COMP_EXP = 700


# def test_predict():
#     """
#     Ensures that returned choice probabilities are consistent.
#     Taken from xlogit tests
#     Probabilities incompatible. Leaving in case it helps future tests
#     """
#     # There is no need to initialize a random seed as the halton draws produce
#     # reproducible results
#     betas = np.array([.1, .1, .1, .1])
#     X_ = X.reshape(N, J, K)

#     model = MixedLogit()
#     model._rvidx,  model._rvdist = np.array([True, True]), np.array(['n', 'n'])
#     model.alternatives =  np.array([1, 2])
#     model.coeff_ = betas
#     model.randvars = randvars
#     model._isvars, model._asvars, model._varnames = [], varnames, varnames
#     model.coeff_names = np.array(["a", "b", "sd.a", "sd.b"])

#     #model.fit(X, y, varnames, alts, ids, randvars, verbose=0, halton=True)
#     prob = model.predict(X, varnames, alts, ids,
#                                           randvars, None)

#     # Compute choice probabilities by hand
#     draws = generate_halton_draws(N, R, K)  # (N,Kr,R)
#     Br = betas[None, [0, 1], None] + draws*betas[None, [2, 3], None]
#     V = np.einsum('njk,nkr -> njr', X_, Br)
#     eV = np.exp(V)
#     e_proba = eV/np.sum(eV, axis=1, keepdims=True)
#     expec_proba = e_proba.mean(axis=-1)
#     expec_ypred = model.alternatives[np.argmax(expec_proba, axis=1)]
#     alt_list, counts = np.unique(expec_ypred, return_counts=True)
#     expec_freq = dict(zip(list(alt_list), list(np.round(counts/np.sum(counts), 3))))

#     # assert np.array_equal(expec_ypred, y_pred)
#     assert expec_freq == prob


def test_validate_inputs():
    """
    Covers potential mistakes in parameters of the fit method that jaxlogit
    should be able to identify
    Taken from xlogit tests
    """
    model = MixedLogit()
    with pytest.raises(ValueError):  # wrong distribution
        config = ConfigData(
            halton=True,
            n_draws=10,
            maxiter=0,
        )
        model.fit(X, y, varnames=varnames, alts=alts, ids=ids, verbose=0, randvars={"a": "fake"}, config=config)

    with pytest.raises(ValueError):  # wrong var name
        config = ConfigData(
            halton=True,
            maxiter=0,
            n_draws=10,
        )

        model.fit(X, y, varnames=varnames, alts=alts, ids=ids, verbose=0, randvars={"fake": "n"}, config=config)
