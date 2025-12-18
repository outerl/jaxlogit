import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import chi2


def wide_to_long(dataframe, id_col, alt_list, alt_name, varying=None, sep="_", alt_is_prefix=False, empty_val=np.nan, panels=False):
    """Reshapes pandas DataFrame from wide to long format.

    Parameters
    ----------
    dataframe : pandas DataFrame
        The wide-format DataFrame.

    id_col : str
        Column that uniquely identifies each sample.

    alt_list : list-like
        List of choice alternatives.

    alt_name : str
        Name of the alternatives column in returned dataset.

    varying : list-like
        List of column names that vary across alternatives.

    sep : str, default='_'
        Separator of column names that vary across alternatives.

    avail: array-like, shape (n_samples,), default=None
        Availability of alternatives for the choice situations. One when
        available or zero otherwise.

    alt_is_prefix : bool
        True if alternative is prefix of the variable name or False if it is
        suffix.

    empty_val : int, float or str, default=np.nan
        Value to fill when alternative not available for a certain variable.


    Returns
    -------
    DataFrame in long format.
    """
    varying = varying if varying is not None else []

    # Validations
    if dataframe is None or id_col is None or alt_list is None or alt_name is None:
        raise ValueError("Dataframe, id_col, alt_list, and alt_name cannot be None")
    if any(col in varying for col in dataframe.columns):
        raise ValueError("varying can't be identical to a column name")
    if alt_name in dataframe.columns:
        raise ValueError(f"alt_name {alt_name} can't be identical to a column name")

    # Initialize new dataframe with id and alt columns
    newcols = {id_col: np.repeat(dataframe[id_col].values, len(alt_list)), alt_name: np.tile(alt_list, len(dataframe))}
    conc_cols = []

    # Reshape columns that vary across alternatives
    patt = "{alt}{sep}{col}" if alt_is_prefix else "{col}{sep}{alt}"
    count_match_patt = 0
    for col in varying:
        series = []
        for alt in alt_list:
            c = patt.format(alt=alt, sep=sep, col=col)
            conc_cols.append(c)
            if c in dataframe.columns:
                series.append(dataframe[c].values)
                count_match_patt += 1
            else:
                series.append(np.repeat(empty_val, len(dataframe)))
        newcols[col] = np.stack(series, axis=1).ravel()
    if count_match_patt == 0 and len(varying) > 0:
        raise ValueError(f"no column matches the pattern {patt}")

    # Reshape columns that do NOT vary across alternatives
    non_varying = [c for c in dataframe.columns if c not in conc_cols + [id_col]]
    for col in non_varying:
        newcols[col] = np.repeat(dataframe[col].values, len(alt_list))

    df = pd.DataFrame(newcols)
    if panels:
        df.loc[df["choice"] != df["alt"], "choice"] = 0
        df.loc[df["choice"] == df["alt"], "choice"] = 1
        assert (df["choice"].sum()) == (df.shape[0] / len(alt_list))

    return df


def lrtest(general_model, restricted_model):
    """Conducts likelihood-ratio test.

    Parameters
    ----------
    general_model : jaxlogit Model
        Fitted model that contains all parameters (unrestricted)

    restricted_model : jaxlogit Model
        Fitted model with less parameters than ``general_model``.

    Returns
    -------
    lrtest_result : dict
        p-value result, chisq statistic, and degrees of freedom used in test
    """
    if len(general_model.coeff_) <= len(restricted_model.coeff_):
        raise ValueError("The general_model is expected to have less estimatesthan the restricted_model")
    genLL, resLL = general_model.loglikelihood, restricted_model.loglikelihood
    degfreedom = len(general_model.coeff_) - len(restricted_model.coeff_)
    stat = 2 * (resLL - genLL)
    return {"pval": chi2.sf(stat, df=degfreedom), "chisq": stat, "degfree": degfreedom}


def get_panel_aware_batch_indices(panel_ids: jnp.ndarray, n_batches: int) -> list[tuple[int, int, int]]:
    """
    Calculates batch indices ensuring that panels are not split across batches.

    Args:
        panel_ids: A 1D numpy array of panel IDs, which must be sorted.
        n_batches: The desired number of batches.

    Requires:
        jnp.all(jnp.diff(panel_ids) >= 0)

    Returns:
        A list of (start, end, num_panels_in_batch) tuples for slicing the data.
    """
    if n_batches <= 0:
        raise ValueError("Number of batches must be positive.")
    if panel_ids is None:
        raise ValueError("panel_ids cannot be None")

    n_obs = len(panel_ids)
    if n_obs == 0:
        return []

    # Find the indices where a new panel begins
    panel_change_points = jnp.where(jnp.diff(panel_ids) != 0)[0] + 1
    panel_start_indices = jnp.concatenate((jnp.array([0]), panel_change_points))

    num_panels = len(panel_start_indices)
    if n_batches >= num_panels:
        # If more batches than panels, each panel is a batch
        panel_end_indices = jnp.concatenate((panel_start_indices[1:], [n_obs]))
        # Each batch has 1 panel
        num_panels_in_batch = jnp.ones(num_panels, dtype=int)
        return list(zip(panel_start_indices, panel_end_indices, num_panels_in_batch))

    # Determine how many panels should go into each batch
    panels_per_batch = jnp.ceil(num_panels / n_batches).astype(int)

    batch_indices = []
    for i in range(0, num_panels, panels_per_batch):
        start_obs_idx = panel_start_indices[i]

        # Determine the end index for the current batch
        end_panel_idx = min(i + panels_per_batch, num_panels)
        if end_panel_idx == num_panels:
            end_obs_idx = jnp.array(n_obs)
        else:
            end_obs_idx = panel_start_indices[end_panel_idx]

        num_panels_in_batch = end_panel_idx - i
        batch_indices.append((start_obs_idx, end_obs_idx, int(num_panels_in_batch)))

    return batch_indices
