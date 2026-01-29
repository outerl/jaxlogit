import numpy as np
import pytest
from jaxlogit.mixed_logit import MixedLogit
from jaxlogit._config_data import ConfigData


# Setup data used for tests
X = np.array([[2, 1], [1, 3], [3, 1], [2, 4], [2, 1], [2, 4]])
y = np.array([0, 1, 0, 1, 0, 1])
ids = np.array([1, 1, 2, 2, 3, 3])
alts = np.array([1, 2, 1, 2, 1, 2])
panels = np.array([1, 1, 1, 1, 2, 2])
varnames = ["a", "b"]
randvars = {"a": "n", "b": "n"}
N, J, K, R = 3, 2, 2, 5

def test_nesting_accepted():
    model = MixedLogit()
    with pytest.raises(
        ValueError, match=r'Variable "c" not in varnames'
    ):
        model._make_nests({'a': ['b', 'c']}, ['a', 'b'])
    with pytest.raises(
        ValueError, match=r'Variable "c" appears in two nests'
    ):
        model._make_nests({'a': ['b', 'c'], 'c': ['c']}, ['a', 'b', 'c'])
    model._make_nests({'a': ['b', 'c'], 'd': 'd'}, ['a', 'b', 'c', 'd']) # Test values without list around them
    assert {'a': ['b', 'c'], 'd': ['d'], 'a': ['a']} == model._make_nests({'a': ['b', 'c']}, ['a', 'b', 'c', 'd']) # Test variables not initially in nests get their own