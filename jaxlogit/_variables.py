from dataclasses import dataclass
import jax.numpy as jnp


@dataclass
class ParametersSetup:
    """EverythingParameters of the distributions for coefficients of explanatory variables.

    #TODO: see whether any functions can be abstracted here
    #TODO: docstring description of each
    #TODO: add class logic
    """

    draws: jnp.ndarray | None = None
    mask: jnp.ndarray | None = None
    values_for_mask: jnp.ndarray | None = None
    mask_chol: jnp.ndarray | None = None
    values_for_chol_mask: jnp.ndarray | None = None
    rand_idx_norm: jnp.ndarray | None = None  # index into betas for normal draws
    rand_idx_truncnorm: jnp.ndarray | None = None  # index into betas for truncated normal draws
    draws_idx_norm: jnp.ndarray | None = None  # index into draws for normal draws
    draws_idx_truncnorm: jnp.ndarray | None = None  # index into draws for truncated normal draws
    fixed_idx: jnp.ndarray | None = None
    idx_ln_dist: jnp.ndarray | None = None
    rand_idx_stddev: jnp.ndarray | None = None
    rand_idx_chol: jnp.ndarray | None = None

    def get_batched_version(self, start: int, end: int):
        return ParametersSetup(
            self.draws[start:end, :, :],
            self.mask,
            self.values_for_mask,
            self.mask_chol,
            self.values_for_chol_mask,
            self.rand_idx_norm,
            self.rand_idx_truncnorm,
            self.draws_idx_norm,
            self.draws_idx_truncnorm,
            self.fixed_idx,
            self.idx_ln_dist,
            self.rand_idx_stddev,
            self.rand_idx_chol,
        )

    def freeze(self):
        """Convert this mutable setup into an immutable, hashable version."""
        return FrozenParametersSetup(
            draws=self.draws,
            mask=self.mask,
            values_for_mask=self.values_for_mask,
            mask_chol=self.mask_chol,
            values_for_chol_mask=self.values_for_chol_mask,
            rand_idx_norm=self.rand_idx_norm,
            rand_idx_truncnorm=self.rand_idx_truncnorm,
            draws_idx_norm=self.draws_idx_norm,
            draws_idx_truncnorm=self.draws_idx_truncnorm,
            fixed_idx=self.fixed_idx,
            idx_ln_dist=self.idx_ln_dist,
            rand_idx_stddev=self.rand_idx_stddev,
            rand_idx_chol=self.rand_idx_chol,
        )


@dataclass(frozen=True)
class FrozenParametersSetup:
    draws: jnp.ndarray | None
    mask: jnp.ndarray | None
    values_for_mask: jnp.ndarray | None
    mask_chol: jnp.ndarray | None
    values_for_chol_mask: jnp.ndarray | None
    rand_idx_norm: jnp.ndarray | None
    rand_idx_truncnorm: jnp.ndarray | None
    draws_idx_norm: jnp.ndarray | None
    draws_idx_truncnorm: jnp.ndarray | None
    fixed_idx: jnp.ndarray | None
    idx_ln_dist: jnp.ndarray | None
    rand_idx_stddev: jnp.ndarray | None
    rand_idx_chol: jnp.ndarray | None

    def get_batched_version(self, start: int, end: int):
        """Frozen version of batched slicing. Returns another FrozenParametersSetup."""
        return FrozenParametersSetup(
            draws=self.draws[start:end, :, :] if self.draws is not None else None,
            mask=self.mask,
            values_for_mask=self.values_for_mask,
            mask_chol=self.mask_chol,
            values_for_chol_mask=self.values_for_chol_mask,
            rand_idx_norm=self.rand_idx_norm,
            rand_idx_truncnorm=self.rand_idx_truncnorm,
            draws_idx_norm=self.draws_idx_norm,
            draws_idx_truncnorm=self.draws_idx_truncnorm,
            fixed_idx=self.fixed_idx,
            idx_ln_dist=self.idx_ln_dist,
            rand_idx_stddev=self.rand_idx_stddev,
            rand_idx_chol=self.rand_idx_chol,
        )

    def __hash__(self):
        # Use id() of arrays to make object hashable for JAX static args
        return hash(
            (
                id(self.draws),
                id(self.mask),
                id(self.values_for_mask),
                id(self.mask_chol),
                id(self.values_for_chol_mask),
                id(self.rand_idx_norm),
                id(self.rand_idx_truncnorm),
                id(self.draws_idx_norm),
                id(self.draws_idx_truncnorm),
                id(self.fixed_idx),
                id(self.idx_ln_dist),
                id(self.rand_idx_stddev),
                id(self.rand_idx_chol),
            )
        )
