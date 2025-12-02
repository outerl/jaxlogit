from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class Parameters:
    """Parameters of the distributions for coefficients of explanatory variables.

    #TODO: see whether any functions can be abstracted here
    #TODO: docstring description of each
    #TODO: add class logic
    """

    Xdf: jnp.ndarray
    Xdr: jnp.ndarray
    draws: jnp.ndarray
    mask: jnp.ndarray
    values_for_mask: jnp.ndarray
    mask_chol: jnp.ndarray
    values_for_chol_mask: jnp.ndarray
    rand_idx_norm: jnp.ndarray
    rand_idx_truncnorm: jnp.ndarray
    draws_idx_norm: jnp.ndarray
    draws_idx_truncnorm: jnp.ndarray
    fixed_idx: jnp.ndarray
    idx_ln_dist: jnp.ndarray
    rand_idx_stddev: jnp.ndarray
    rand_idx_chol: jnp.ndarray


@dataclass
class InventoryItem:
    """Class for keeping track of an item in inventory."""

    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
