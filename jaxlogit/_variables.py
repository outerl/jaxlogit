from dataclasses import dataclass
import jax.numpy as jnp
from typing import Optional


@dataclass
class ParametersSetup:
    """EverythingParameters of the distributions for coefficients of explanatory variables.

    #TODO: see whether any functions can be abstracted here
    #TODO: docstring description of each
    #TODO: add class logic
    """

    draws: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None
    values_for_mask: Optional[jnp.ndarray] = None
    mask_chol: Optional[jnp.ndarray] = None
    values_for_chol_mask: Optional[jnp.ndarray] = None
    rand_idx_norm: Optional[jnp.ndarray] = None  # index into betas for normal draws
    rand_idx_truncnorm: Optional[jnp.ndarray] = None  # index into betas for truncated normal draws
    draws_idx_norm: Optional[jnp.ndarray] = None  # index into draws for normal draws
    draws_idx_truncnorm: Optional[jnp.ndarray] = None  # index into draws for truncated normal draws
    fixed_idx: Optional[jnp.ndarray] = None
    idx_ln_dist: Optional[jnp.ndarray] = None
    rand_idx_stddev: Optional[jnp.ndarray] = None
    rand_idx_chol: Optional[jnp.ndarray] = None

    def get_batched_version(self, start: int, end: int):
        return ParametersSetup(
            self.draws[start:end, :, :],
        )


# @dataclass
# class InventoryItem:
#     """Class for keeping track of an item in inventory."""

#     name: str
#     unit_price: float
#     quantity_on_hand: int = 0

#     def total_cost(self) -> float:
#         return self.unit_price * self.quantity_on_hand
