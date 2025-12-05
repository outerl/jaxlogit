import pandas as pd
import numpy as np
from jaxlogit.utils import wide_to_long, lrtest, get_panel_aware_batch_indices
from jaxlogit.mixed_logit import MixedLogit
import jax.numpy as jnp
import pytest


dfw = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "time_car": [1, 1, 1, 1, 1],
        "time_bus": [2, 2, 2, 2, 2],
        "cost_bus": [3, 3, 3, 3, 3],
        "income": [9, 8, 7, 6, 5],
        "age": [0.6, 0.5, 0.4, 0.3, 0.2],
        "y": ["bus", "bus", "bus", "car", "car"],
    }
)


def test_wide_to_long():
    """
    Ensures a pandas dataframe is properly converted from wide to long format
    Adapted from xlogit tests
    """
    expec = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "alt": ["car", "bus", "car", "bus", "car", "bus", "car", "bus", "car", "bus"],
            "time": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "cost": [0, 3, 0, 3, 0, 3, 0, 3, 0, 3],
            "income": [9, 9, 8, 8, 7, 7, 6, 6, 5, 5],
            "age": [0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2],
            "y": ["bus", "bus", "bus", "bus", "bus", "bus", "car", "car", "car", "car"],
        }
    )
    dfl = wide_to_long(dfw, id_col="id", alt_list=["car", "bus"], alt_name="alt", varying=["time", "cost"], empty_val=0)
    assert dfl.equals(expec)


def test_wide_to_long_validation():
    with pytest.raises(ValueError):
        wide_to_long(None, None, None, None)
        wide_to_long(dfw, None, None, None)
        wide_to_long(None, "id", None, None)
        wide_to_long(None, None, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], None)
        wide_to_long(None, None, None, "alt")


def test_lrtest():
    """
    Ensures a correct result of the lrtest. The comparison values were
    obtained from comparison with lrtest in R's lmtest package
    Adapted from xlogit tests
    """
    general = MixedLogit()
    general.loglikelihood = 1312
    restricted = MixedLogit()
    restricted.loglikelihood = -1305
    general.loglikelihood = -1312
    general.coeff_ = np.zeros(4)
    restricted.coeff_ = np.zeros(2)

    obtained = lrtest(general, restricted)
    expected = {"pval": 0.0009118819655545164, "chisq": 14, "degfree": 2}
    for key in obtained:
        assert obtained[key] == expected[key]


def test_lrtest_validation():
    general = MixedLogit()
    general.loglikelihood = 1312
    restricted = MixedLogit()
    restricted.loglikelihood = -1305
    general.loglikelihood = -1312
    general.coeff_ = np.zeros(2)
    restricted.coeff_ = np.zeros(4)

    with pytest.raises(ValueError):
        lrtest(general, restricted)


@pytest.fixture
def panelsAndBatches():
    panels = jnp.arange(1, 6)
    batch_size = 5
    num_batches = int(np.ceil(len(panels) / batch_size))
    return (panels, num_batches)


def test_get_panel_aware_batch_indices_validate(panelsAndBatches):
    (panels, num_batches) = panelsAndBatches
    with pytest.raises(ValueError):
        get_panel_aware_batch_indices(panels, 0)
        get_panel_aware_batch_indices(panels, -1)
        get_panel_aware_batch_indices(None, num_batches)


def test_get_panel_aware_batch_indices(panelsAndBatches):
    (panels, num_batches) = panelsAndBatches

    # Edge cases:
    expected = [
        (np.zeros(1, dtype=np.int32), np.array(2, dtype=np.int32), 2),
        (np.array(2, dtype=np.int32), np.array(4, dtype=np.int32), 2),
    ]
    assert expected == get_panel_aware_batch_indices(jnp.arange(1, 5), 2)

    expected = [(np.zeros(1, dtype=np.int32), np.array(5, dtype=np.int32), 5)]
    assert expected == get_panel_aware_batch_indices(panels, num_batches)
