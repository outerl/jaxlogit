# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: venv (3.10.12)
#     language: python
#     name: python3
# ---

# %% [markdown] id="TJHlxbR5kEe-"
# # Example of running jaxlogit with batched draws
#
# jaxlogit's default way of processing random draws for simulation is to generate them once at the beginning and then run calculate the loglikelihood at each step of the optimization routine with these draws. The size of the corresponding array is (number_of_observations x number_of_random_variables x  number_of_draws) which can get very large. In case this is too large for local memory, jaxlogit can dynamcially generate draws on each iteration. The advantage of this is that calculations can now be batched, i.e., processed on smaller subsets and then added up. This reduces memory load that the cost of runtime. Note that jax still calculates gradients so this method also has memory limits.
# %% colab={"base_uri": "https://localhost:8080/"} id="NQbZt7CVh8f_" outputId="b823e80f-fd47-4dd1-8656-3fd0d6a1e26a"
import pandas as pd
import numpy as np
import jax

from jaxlogit.mixed_logit import MixedLogit, ConfigData
# %%
#  64bit precision
jax.config.update("jax_enable_x64", True)
# %% [markdown]
# ## Electricity Dataset batching example
#
# From xlogit's examples. Since this example shows how batching reduces memory load, to speed up test times we skip the calculation of std errors and **reduce the maximum interations to 10**.
# %%
df = pd.read_csv("https://raw.github.com/arteagac/xlogit/master/examples/data/electricity_long.csv")
# %%
n_obs = df['chid'].unique().shape[0]
n_vars = 6
n_draws = 5000
maxiter = 10

size_in_ram = (n_obs * n_vars * n_draws * 8) / (1024 ** 3)  # in GB

print(
    f"Data has {n_obs} observations, we use {n_vars} random variables in the model. We work in 64 bit precision, so each element is 8 bytes."
    + f" For {n_draws} draws, the array of draws is about {size_in_ram:.2f} GB."
)

varnames = ['pf', 'cl', 'loc', 'wk', 'tod', 'seas']
# %% [markdown]
# ## Four batches
# First we try four batches
# %%
n_batches = 4
batch_size = np.ceil(n_obs/n_batches)
print(f"For {n_batches} batches and {n_obs} obervations, batch size is {batch_size}")

model = MixedLogit()

config = ConfigData(
    panels=df['id'],
    n_draws=n_draws,
    skip_std_errs=True,  # skip standard errors to speed up the example
    batch_size=batch_size,
    optim_method="L-BFGS-B",  # "L-BFGS-B", "BFGS"lver
    maxiter=maxiter,
)

res = model.fit(
    X=df[varnames],
    y=df['choice'],
    varnames=varnames,
    ids=df['chid'],
    alts=df['alt'],
    randvars={'pf': 'n', 'cl': 'n', 'loc': 'n', 'wk': 'n', 'tod': 'n', 'seas': 'n'},
    config=config
)
display(model.summary())

# %% [markdown]
# ## No batches
# %%
model = MixedLogit()

config = ConfigData(
    panels=df['id'],
    n_draws=n_draws,
    skip_std_errs=True,  # skip standard errors to speed up the example
    batch_size=None,
    optim_method="L-BFGS-B",
    maxiter=maxiter,
)

res = model.fit(
    X=df[varnames],
    y=df['choice'],
    varnames=varnames,
    ids=df['chid'],
    alts=df['alt'],
    randvars={'pf': 'n', 'cl': 'n', 'loc': 'n', 'wk': 'n', 'tod': 'n', 'seas': 'n'},
    config=config
)
display(model.summary())