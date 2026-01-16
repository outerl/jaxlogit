# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown] 
# # Scikit learn interface and Cross Validation
# Uses the swissmetro data. Based on previous example for this dataset, which is based on the xlogit example [Mixed Logit](https://xlogit.readthedocs.io/en/latest/notebooks/mixed_logit_model.html).
# %%
import pandas as pd
import numpy as np
import jax

#  64bit precision
jax.config.update("jax_enable_x64", True)

# %% [markdown] 
# ## Import Swissmetro Dataset
# The alternatives are car, train or SM (the Swissmetro). The explanatory variables are cost, travel time and alternative specific constants for the train and car options. See the previous example for the Swissmetro Dataset for a detailed explaination [here](https://outerl.github.io/jaxlogit/example_notebooks/mixed_logit_correlated_example.html#Swissmetro-Dataset) 
#
# 
# ### Read data
# The dataset is imported and filtered.
# %%
df_wide = pd.read_table("http://transp-or.epfl.ch/data/swissmetro.dat", sep="\t")

# Keep only observations for commute and business purposes that contain known choices
df_wide = df_wide[(df_wide["PURPOSE"].isin([1, 3]) & (df_wide["CHOICE"] != 0))]
df_wide["CHOICE"] = df_wide["CHOICE"].map({1: "TRAIN", 2: "SM", 3: "CAR"})

df_wide["custom_id"] = np.arange(len(df_wide))  # Add unique identifier
df_wide
# %% [markdown] 
# ### Reshape data
# This scikit learn interface uses the data in wide format. Here are data transformations and adding alternative specific constraints using pandas dataframes. Data headings for each alternative and variable pair is in the form `alternative_variable`, so for the cost of the train option, it would be `TRAIN_CO`.
# %%
varnames = ["CO", "TT"]
alternatives = ["TRAIN", "CAR", "SM"]
seperator = "_"
alt_is_prefix = True

for alternative in alternatives:
    # alternative specific constants for train and car
    for alternative_constant in ["TRAIN", "CAR"]:
        if alternative_constant == alternative:
            df_wide[alternative + seperator + 'ASC' + seperator + alternative_constant] = np.ones(len(df_wide))
        else:
            df_wide[alternative + seperator + 'ASC' + seperator + alternative_constant] = np.zeros(len(df_wide))
    
    # scale time and cost
    for var in varnames:
        df_wide[alternative + seperator + var] = df_wide[alternative + seperator + var]/100


varnames = ["CO", "TT", "ASC_TRAIN", "ASC_CAR"]
all_varnames = [alternative + seperator + varname for alternative in alternatives for varname in varnames]
all_varnames
# %% [markdown] 
# ### Creating and fitting a model
# Options for the model are given in the creation of the esimtator. Note that variable names must be included here. Panel data is currently not supported.
#
# Then the model can be fit when given the data.
# %%
from jaxlogit.scikit_wrapper import MixedLogitEstimator

mixed_logit_estimator = MixedLogitEstimator(
    varnames=varnames, 
    randvars = {'TT': 'n'},
    n_draws=1500
)
X=df_wide[all_varnames]
y=df_wide["CHOICE"]

mixed_logit_estimator.fit(X, y)
# %% [markdown] 
# ### Scikit learn utilities
# From this interface utilties for splitting up data in to training and testing data and cross validation can be used. 
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
mixed_logit_estimator.fit(X_train, y_train)

mixed_logit_estimator.predict(X_test)
# %%
mixed_logit_estimator.score(X_test, y_test)
# %%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(mixed_logit_estimator, X, y, cv=5)
scores