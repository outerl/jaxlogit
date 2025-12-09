import logging
import jax.numpy as jnp
from ._config_data import ConfigData
from .draws import generate_draws
import numpy as np


logger = logging.getLogger(__name__)


class ParametersSetup:
    """EverythingParameters of the distributions for coefficients of explanatory variables.

    #TODO: see whether any functions can be abstracted here
    #TODO: docstring description of each
    #TODO: add class logic
    """

    def __init__(
        self, rvdist, rvidx, rvidx_normal_bases, rvidx_truncnorm_based, coef_names, betas, config: ConfigData
    ) -> None:
        self._frozen = False

        sd_start_idx, sd_slice_size = self.setup_betas_indicies(
            rvidx_normal_bases, rvidx_truncnorm_based, rvidx, rvdist, config
        )

        self.setup_fixed_variable_masks(config.fixedvars, coef_names, sd_start_idx, sd_slice_size, betas)

        rvidx = jnp.array(rvidx, dtype=bool)
        self.fixed_idx = jnp.where(~rvidx)[0]

        self._hash = None
        self._frozen = True

    def setup_betas_indicies(self, rvidx_normal_bases, rvidx_truncnorm_based, rvidx, rvdist, config: ConfigData):
        self.rand_idx_norm = jnp.where(rvidx_normal_bases)[0]
        self.rand_idx_truncnorm = jnp.where(rvidx_truncnorm_based)[0]

        # #std dev is different: in order
        sd_start_idx = len(rvidx)  # start of std devs
        sd_slice_size = len(jnp.where(rvidx)[0])  # num all std devs
        # TODO TN: separate rand_idx_stddev for n_trunc and n/ln
        self.rand_idx_stddev = jnp.arange(sd_start_idx, sd_start_idx + sd_slice_size, dtype=jnp.int32)

        chol_start_idx = sd_start_idx + sd_slice_size  # start: after all std devs
        sd_chol_slice_size = len(jnp.where(rvidx_normal_bases)[0])  # number of elements based on n/ln dists
        chol_slice_size = (sd_chol_slice_size * (sd_chol_slice_size + 1)) // 2 - sd_chol_slice_size
        self.rand_idx_chol = (
            None
            if not config.include_correlations
            else jnp.arange(chol_start_idx, chol_start_idx + chol_slice_size, dtype=jnp.int32)
        )

        self.draws_idx_norm = jnp.array([k for k, dist in enumerate(rvdist) if dist in ["n", "ln"]], dtype=jnp.int32)
        self.draws_idx_truncnorm = jnp.array([k for k, dist in enumerate(rvdist) if dist == "n_trunc"], dtype=jnp.int32)

        # Set up index into _rvdist for lognormal distributions. This is used to apply the lognormal transformation
        # to the random betas
        self.idx_ln_dist = jnp.array([i for i, x in enumerate(rvdist) if x == "ln"], dtype=jnp.int32)

        return sd_start_idx, sd_slice_size

    def setup_draws_from_config(self, N: int, rvdist, config: ConfigData):
        """Returns the draws.

        Formats the draws according to the panels.

        Args:
            N: number of observations. Size of X, the data.
            config: The data config for the fit/predict
        """

        # Generate draws
        n_samples = N if config.panels is None else np.max(config.panels) + 1
        logger.debug(f"Generating {config.n_draws} number of draws for each observation and random variable")

        draws = generate_draws(n_samples, config.n_draws, rvdist, config.halton, halton_opts=config.halton_opts)
        if config.panels is not None:
            draws = draws[config.panels]  # (N,num_random_params,n_draws)
        draws = jnp.array(draws)

        logger.debug(f"Draw generation done, shape of draws: {draws.shape}, number of draws: {config.n_draws}")

        return draws

    def setup_fixed_variable_masks(self, fixedvars, coef_names, sd_start_idx, sd_slice_size, betas):
        if fixedvars == None:
            self.mask = None
            self.values_for_mask = None
            self.mask_chol = None
            self.values_for_chol_mask = None
            return

        self.setup_fixed_variable_indicies(fixedvars, coef_names, sd_start_idx, sd_slice_size, betas)

    def setup_fixed_variable_indicies(self, fixedvars, coef_names, sd_start_idx, sd_slice_size, betas):
        mask_chol = []
        values_for_chol_mask = []
        mask = np.zeros(len(fixedvars), dtype=np.int32)
        values_for_mask = np.zeros(len(fixedvars), dtype=np.int32)
        for i, (k, v) in enumerate(fixedvars.items()):
            idx = np.where(coef_names == k)[0]
            if len(idx) == 0:
                raise ValueError(f"Variable {k} not found in the model.")
            if len(idx) > 1:
                raise ValueError(f"Variable {k} found more than once, this should never happen.")
            idx = idx[0]
            mask[i] = idx
            assert v is not None
            # TODO: move setting betas out of the __init__ function
            betas = betas.at[idx].set(v)
            values_for_mask[i] = v

            if (idx >= sd_start_idx) & (idx < sd_start_idx + sd_slice_size):
                mask_chol.append(idx - sd_start_idx)
                values_for_chol_mask.append(v)

        if len(mask_chol) == 0:
            self.mask_chol = None
            self.values_for_chol_mask = None
        else:
            self.mask_chol = jnp.array(mask_chol, dtype=jnp.int32)
            self.values_for_chol_mask = jnp.array(values_for_chol_mask)

        self.mask = jnp.array(mask)
        self.values_for_mask = jnp.array(values_for_mask)

    def __setattr__(self, attr, value):
        if attr != "_hash" and getattr(self, "_frozen", None):
            raise AttributeError("Trying to set attribute on a frozen instance")
        return super().__setattr__(attr, value)

    def __hash__(self):
        if self._hash is None:
            hash_values = []

            for attr_name in sorted(vars(self)):
                if attr_name == "_hash":
                    continue
                attr_value = getattr(self, attr_name)

                if attr_value is None:
                    hash_values.append(hash(None))
                elif hasattr(attr_value, "__array__"):
                    # For integer JAX arrays, tobytes() is fine
                    hash_values.append(hash(attr_value.tobytes()))
                else:
                    hash_values.append(hash(attr_value))

            self._hash = hash(tuple(hash_values))
        return self._hash

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ParametersSetup):
            return False
        return all(
            getattr(self, attr_name) == getattr(value, attr_name) for attr_name in vars(self) if attr_name != "_hash"
        )[0]
