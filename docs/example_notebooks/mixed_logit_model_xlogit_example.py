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

# # Mixed Logit [markdown]
# Based on the xlogit example [Mixed Logit](https://xlogit.readthedocs.io/en/latest/notebooks/mixed_logit_model.html).
# %%
import pandas as pd
import numpy as np
import jax

from jaxlogit.mixed_logit import MixedLogit, ConfigData

# %%
# 64bit precision
jax.config.update("jax_enable_x64", True)
# %% [markdown]
# ## Electricity Dataset
# %% [markdown]
# The electricity dataset contains 4,308 choices among four electricity suppliers based on the attributes of the offered plans, which include prices(pf), contract lengths(cl), time of day rates (tod), seasonal rates(seas), as well as attributes of the suppliers, which include whether the supplier is local (loc) and well-known (wk). The data was collected through a survey where 12 different choice situations were presented to each participant. The multiple responses per participants were organized into panels. Given that some participants answered less than 12 of the choice situations, some panels are unbalanced, which `jaxlogit` is able to handle. [Revelt and Train (1999)](https://escholarship.org/content/qt1900p96t/qt1900p96t.pdf) provide a detailed description of this dataset. 
# %% [markdown]
# ### Read data
# %% [markdown]
# The dataset is already in long format so no reshaping is necessary, it can be used directly in jaxlogit.
# %%
df = pd.read_csv("https://raw.githubusercontent.com/outerl/jaxlogit/refs/heads/main/examples/electricity_long.csv")
df
# %% [markdown]
# ### Fit the model
# %% [markdown]
# Note that the parameter `panels` was included in the `fit` function in order to take into account panel structure of this dataset during estimation.
# %%
varnames = ['pf', 'cl', 'loc', 'wk', 'tod', 'seas']
model = MixedLogit()

config = ConfigData(
    n_draws=600,
    panels=df['id'],
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
model.summary()
# %%
# Note the sd. variables in jaxlogit are softplus transformed by default such that they are always positive. To compare these to xlogits results at https://github.com/arteagac/xlogit/blob/master/examples/mixed_logit_model.ipynb
# use jax.nn.softplus(params) for non-asserted sd. params. Or run w/o softplus:
model = MixedLogit()

config = ConfigData(
    force_positive_chol_diag=False,
    panels=df['id'],
    n_draws=600,
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
model.summary()
# %% [markdown]
# ## Fishing Dataset
# %% [markdown]
# This example illustrates the estimation of a Mixed Logit model for choices of 1,182 individuals for sport fishing modes using `jaxlogit`. The goal is to analyse the market shares of four alternatives (i.e., beach, pier, boat, and charter) based on their cost and fish catch rate. [Cameron (2005)](http://cameron.econ.ucdavis.edu/mmabook/mma.html) provides additional details about this dataset. The following code illustrates how to use `jaxlogit` to estimate the model parameters.
# %% [markdown]
# ### Read data
# %% [markdown]
# The data to be analyzed can be imported to Python using any preferred method.
# %%
import pandas as pd
df = pd.read_csv("https://raw.github.com/arteagac/xlogit/master/examples/data/fishing_long.csv")
df
# %% [markdown]
# ### Fit model
# %% [markdown]
# Once the data is in the `Python` environment, `jaxlogit` can be used to fit the model, as shown below. The `MultinomialLogit` class is imported from `jaxlogit`, and its constructor is used to initialise a new model. The `fit` method estimates the model using the input data and estimation criteria provided as arguments to the method's call. The arguments of the `fit` methods are described in [jaxlogit's documentation](https://outerl.github.io/jaxlogit/api.html).
#
# %%
varnames = ['price', 'catch']
model = MixedLogit()

config = ConfigData(
    n_draws=2000,  # Note using 1000 draws here leads to sd.catch going to zero, need more draws to find minimum at positive stddev
)

model.fit(
    X=df[varnames],
    y=df['choice'],
    varnames=varnames,
    alts=df['alt'],
    ids=df['id'],
    randvars={'price': 'n', 'catch': 'n'},
    config=config
)
model.summary()
# %%
# sd. vals agree with xlogit results except for sign of sd.catch, which is due to xlogit not restricting the sd devs to positive parameters and the log-likelihood being symmetric wrt to sign of normal std dev for non-correlated parameters. 
jax.nn.softplus(model.coeff_[len(model._rvidx):])
# %% [markdown]
# ## Car Dataset
# %% [markdown]
# The fourth example uses a stated preference panel dataset for choice of car. Three alternatives are considered, with up to 6 choice situations per individual. This again is an unbalanced panel with responses of some individuals less than 6 situations. The dataset contains 8 explanatry variables: price, operating cost, range, and binary indicators to indicate whether the car is electric, hybrid, and if performance is high or medium respectively. This dataset was taken from Kenneth Train's MATLAB codes for estimation of Mixed Logit models as shown in this link: https://eml.berkeley.edu/Software/abstracts/train1006mxlmsl.html
# %% [markdown]
# ### Read data
# %%
import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.github.com/arteagac/xlogit/master/examples/data/car100_long.csv")
# %% [markdown]
# Since price and operating cost need to be estimated with negative coefficients, we reverse the variable signs in the dataframe. 

# %%
df['price'] = -df['price']/10000
df['opcost'] = -df['opcost']
df
# %% [markdown]
# ### Fit the model
# %%
varnames = ['hiperf', 'medhiperf', 'price', 'opcost', 'range', 'ev', 'hybrid'] 
model = MixedLogit()

config = ConfigData(
    n_draws = 1000,
    panels=df['person_id'],
)

model.fit(
    X=df[varnames],
    y=df['choice'],
    varnames=varnames,
    alts=df['alt'],
    ids=df['choice_id'],
    randvars = {'price': 'ln', 'opcost': 'n',  'range': 'ln', 'ev':'n', 'hybrid': 'n'}, 
    config=config
)
model.summary()
# %% [markdown]
# ### Softplus the standard deviations to make them positive.
# %%
jax.nn.softplus(model.coeff_[len(model._rvidx):])
# %% [markdown]
# ## References
# %% [markdown]
# - Bierlaire, M. (2018). PandasBiogeme: a short introduction. EPFL (Transport and Mobility Laboratory, ENAC).
#
# - Brathwaite, T., & Walker, J. L. (2018). Asymmetric, closed-form, finite-parameter models of multinomial choice. Journal of Choice Modelling, 29, 78–112. 
#
# - Cameron, A. C., & Trivedi, P. K. (2005). Microeconometrics: methods and applications. Cambridge university press.
#
# - Croissant, Y. (2020). Estimation of Random Utility Models in R: The mlogit Package. Journal of Statistical Software, 95(1), 1-41.
#
# - Revelt, D., & Train, K. (1999). Customer-specific taste parameters and mixed logit. University of California, Berkeley.
#
#
