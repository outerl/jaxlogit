# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown] 
# # Mixed Logit with correlations
# Using swissmetro data, comparing results to biogeme (Bierlaire, M. (2018). PandasBiogeme: a short introduction. EPFL (Transport and Mobility Laboratory, ENAC))
# %%
import pandas as pd
import numpy as np
import jax

from jaxlogit.mixed_logit import MixedLogit, ConfigData

# 64bit precision
jax.config.update("jax_enable_x64", True)
# %% [markdown] 
# ## Swissmetro Dataset
# The swissmetro dataset contains stated-preferences for three alternative transportation modes that include car, train and a newly introduced mode: the swissmetro. This dataset is commonly used for estimation examples with the `Biogeme` and `PyLogit` packages. The dataset is available at http://transp-or.epfl.ch/data/swissmetro.dat and [Bierlaire et. al., (2001)](https://transp-or.epfl.ch/documents/proceedings/BierAxhaAbay01.pdf) provides a detailed discussion of the data as wells as its context and collection process. The explanatory variables in this example include the travel time (`TT`) and cost `CO` for each of the three alternative modes.

# This example also adds alternative-specific constraints to represent unobserved factors, and shows how known parameter values can be set. It also shows the functionality of considering the correlation between normally distributed variables.  
# 
# ### Read data
# The dataset is imported to the Python environment using `pandas`. Then, two types of samples, ones with a trip purpose different to commute or business and ones with an unknown choice, are filtered out. The original dataset contains 10,729 records, but after filtering, 6,768 records remain for following analysis. Finally, a new column that uniquely identifies each sample is added to the dataframe and the `CHOICE` column, which originally contains a numerical coding of the choices, is mapped to a description that is consistent with the alternatives in the column names.
# %%
df_wide = pd.read_table("http://transp-or.epfl.ch/data/swissmetro.dat", sep="\t")

# Keep only observations for commute and business purposes that contain known choices
df_wide = df_wide[(df_wide["PURPOSE"].isin([1, 3]) & (df_wide["CHOICE"] != 0))]

df_wide["custom_id"] = np.arange(len(df_wide))  # Add unique identifier
df_wide["CHOICE"] = df_wide["CHOICE"].map({1: "TRAIN", 2: "SM", 3: "CAR"})
df_wide
# %% [markdown] 
# ### Reshape data
# The imported dataframe is in wide format, and it needs to be reshaped to long format for processing by `jaxlogit`, which offers the `wide_to_long` utility for this reshaping process from `xlogit`. The user specifies the column that uniquely identifies each sample, the names of the alternatives, the columns that vary across alternatives, and whether the alternative names are a prefix or suffix of the column names. Additionally, the user can specify a value (`empty_val`) to be used by default when an alternative is not available for a certain variable. Additional usage examples for the `wide_to_long` function are available in xlogit's documentation at https://xlogit.readthedocs.io/en/latest/notebooks/convert_data_wide_to_long.html. Also, details about the function parameters are available at the [API reference ](https://xlogit.readthedocs.io/en/latest/api/utils.html#xlogit.utils.wide_to_long).
# %%
from jaxlogit.utils import wide_to_long

df = wide_to_long(
    df_wide,
    id_col="custom_id",
    alt_name="alt",
    sep="_",
    alt_list=["TRAIN", "SM", "CAR"],
    empty_val=0,
    varying=["TT", "CO", "HE", "AV", "SEATS"],
    alt_is_prefix=True,
)
df
# %% [markdown] 
# ### Create model specification
# Following the reshaping, users can create or update the dataset's columns in order to accommodate their model specification needs, if necessary. The code below shows how the columns `ASC_TRAIN` and `ASC_CAR` were created to incorporate alternative-specific constants in the model. In addition, the example illustrates an effective way of establishing variable interactions (e.g., trip costs for commuters with an annual pass) by updating existing columns conditional on values of other columns. Column operations provide users with an intuitive and highly-flexible mechanism to incorporate model specification aspects, such as variable transformations, interactions, and alternative specific coefficients and constants. By operating the dataframe columns, any utility specification can be accommodated in `jaxlogit`. 
# %%
df["ASC_TRAIN"] = np.ones(len(df)) * (df["alt"] == "TRAIN")
df["ASC_CAR"] = np.ones(len(df)) * (df["alt"] == "CAR")
df["TT"], df["CO"] = df["TT"] / 100, df["CO"] / 100  # Scale variables
annual_pass = (df["GA"] == 1) & (df["alt"].isin(["TRAIN", "SM"]))
df.loc[annual_pass, "CO"] = 0  # Cost zero for pass holders
# %% [markdown] 
# ### Estimate model parameters
# The `fit` method estimates the model by taking as input the data from the previous step along with additional specification criteria, such as the distribution of the random parameters (`randvars`), the number of random draws (`n_draws`), and the availability of alternatives for the choice situations (`avail`). We set the optimization method as `L-BFGS-B` as this is a robust routine that usually helps solve convergence issues.  Once the estimation routine is completed, the `summary` method can be used to display the estimation results.

# The ConfigData class is used to store optional arguments to the `fit` method.
# %%
varnames = ["ASC_CAR", "ASC_TRAIN", "CO", "TT"]
model = MixedLogit()

config = ConfigData(
    n_draws=1500,
    avail=(df["AV"]),
    panels=(df["ID"]),
)

res = model.fit(df[varnames], df["CHOICE"], varnames, df["alt"], df["custom_id"], {"TT": "n"}, config)
model.summary()
# %% [markdown] 
# ## Example of fixing parameters
# Here we add the alternative specific constraint for the swissmetro and set it to 0.
# %%
# we left this one out before, let's add it and assert parameters to 0
df["ASC_SM"] = np.ones(len(df)) * (df["alt"] == "SM")
# %%
varnames = ["ASC_CAR", "ASC_TRAIN", "ASC_SM", "CO", "TT"]
set_vars = {"ASC_SM": 0.0}  # Fixing parameters
model = MixedLogit()

config = ConfigData(
    avail=df["AV"],
    panels=df["ID"],
    set_vars=set_vars,
    n_draws=1500,
)

res = model.fit(
    X=df[varnames],
    y=df["CHOICE"],
    varnames=varnames,
    alts=df["alt"],
    ids=df["custom_id"],
    randvars={"TT": "n"},
    config=config,
)
model.summary()
#  %% [markdown] 
# ## Error components with correlations
# By default, allowing correlation adds variables for the correlation of all normally and log normally distrubuted variables. For variable x and y, it adds a new variable called chol.x.y. Correlation variables representing the correlation between variables that we do not want to be correlated can be set to 0. Here some variables are excluded or set to known values according to research done in [J. Walker's PhD thesis (MIT 2001)](https://transp-or.epfl.ch/courses/dca2012/WalkerPhD.pdf).
# %%
varnames = ["ASC_CAR", "ASC_TRAIN", "ASC_SM", "CO", "TT"]

randvars = {"ASC_CAR": "n", "ASC_TRAIN": "n", "ASC_SM": "n"}
set_vars = {
    "ASC_SM": 0.0,
    "sd.ASC_TRAIN": 1.0,
    "sd.ASC_CAR": 0.0,
    "chol.ASC_CAR.ASC_TRAIN": 0.0,
    "chol.ASC_CAR.ASC_SM": 0.0,
}  # Identification of error components, see J. Walker's PhD thesis (MIT 2001)

config = ConfigData(
    avail=df["AV"],
    panels=df["ID"],
    set_vars=set_vars,
    include_correlations=True,  # Enable correlation between random parameters
)


model = MixedLogit()
res = model.fit(
    X=df[varnames],
    y=df["CHOICE"],
    varnames=varnames,
    alts=df["alt"],
    ids=df["custom_id"],
    randvars=randvars,
    config=config,
)
model.summary()
