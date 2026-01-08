#!/usr/bin/env python
# coding: utf-8

# # Summary of time taken and brier scores for jaxlogit, xlogit, and biogeme
# Where the estimation is using draws = 600 (suboptimal but highest without running out of memory in biogeme), and training and test data is separated.
# 
# | | jaxlogit | xlogit | biogeme |
# |---|---|---|---|
# |Making Model | 37.7s | 16.9s | 4:15 |
# |Estimating | 1.6s | 0.0s | 15.4s |
# |Brier Score | 0.6345 | 0.6345 | 0.6345 |

# # Setup

# In[ ]:


import pandas as pd
import numpy as np
import jax
import pathlib
import xlogit
import sklearn

from jaxlogit.mixed_logit import MixedLogit, ConfigData
from jaxlogit.utils import wide_to_long

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Draws, log, MonteCarlo, PanelLikelihoodTrajectory
import biogeme.database as db
from biogeme.expressions import Variable

logger = blog.get_screen_logger()
logger.setLevel(blog.INFO)

#  64bit precision
jax.config.update("jax_enable_x64", True)


# # Get the full electricity dataset
# 
# Use for jaxlogit and xlogit. Adjustusting n_draws can improve accuracy, but Biogeme cannot handle 600 or more draws with this data set.

# In[ ]:


varnames = ['pf', 'cl', 'loc', 'wk', 'tod', 'seas']
n_draws = 600


# Reshape the data so it can be passed to test_train_split in a wide format. Additionally, xlogit and jaxlogit require long format while biogeme requires a wide format.

# In[ ]:


df_long = pd.read_csv(pathlib.Path.cwd().parent.parent / "examples" / "electricity_long.csv")
choice_df = df_long.loc[df_long['choice'] == 1, ['id', 'chid', 'alt']]
choice_df = choice_df.rename(columns={'alt': 'choice'})
df_wide = df_long.pivot(index=['id', 'chid'], columns='alt', values=varnames)
df_wide.columns = [f'{var}_{alt}' for var, alt in df_wide.columns]
df_wide = df_wide.reset_index()
df = df_wide.merge(
    choice_df,
    on=['id', 'chid'],
    how='inner',
    validate='one_to_one'
)

df_wide_train, df_wide_test = sklearn.model_selection.train_test_split(df, train_size=0.8)
df_train = wide_to_long(df_wide_train, "chid", [1,2,3,4], "alt", varying=varnames, panels=True)
df_train = df_train.sort_values(['chid', 'alt'])
df_test = wide_to_long(df_wide_test, "chid", [1,2,3,4], "alt", varying=varnames, panels=True)
df_test = df_test.sort_values(['chid', 'alt'])

df_small_train, _ = sklearn.model_selection.train_test_split(df, train_size=0.1)
database_train = db.Database('electricity', df_small_train)
database_train.panel('id')
database_test = db.Database('electricity', df_wide_test)


# jaxlogit and xlogit setup:

# In[ ]:


X_train = df_train[varnames]
y_train = df_train['choice']

ids_train = df_train['chid']
alts_train = df_train['alt']
panels_train = df_train['id']

X_test = df_test[varnames]
y_test = df_test['choice']

ids_test = df_test['chid']
alts_test = df_test['alt']
panels_test = df_test['id']


# In[ ]:


randvars = {'pf': 'n', 'cl': 'n', 'loc': 'n', 'wk': 'n', 'tod': 'n', 'seas': 'n'}

model_jax = MixedLogit()
model_x = xlogit.MixedLogit()

config = ConfigData(
    panels=panels_train,
    n_draws=n_draws,
    skip_std_errs=True,  # skip standard errors to speed up the example
    batch_size=None,
    optim_method="L-BFGS-B",
)
init_coeff = None


# Biogeme setup:

# In[ ]:


X = {
    name: {
        j: Variable(f"{name}_{j}")
        for j in [1,2,3,4]
    }
    for name in varnames
}

alt_1 = Beta('alt_1', 0, None, None, 0)
alt_2 = Beta('alt_2', 0, None, None, 0)
alt_3 = Beta('alt_3', 0, None, None, 0)
alt_4 = Beta('alt_4', 0, None, None, 1)

pf_mean = Beta('pf_mean', 0, None, None, 0)
pf_sd = Beta('pf_sd', 1, None, None, 0)
cl_mean = Beta('cl_mean', 0, None, None, 0)
cl_sd = Beta('cl_sd', 1, None, None, 0)
loc_mean = Beta('loc_mean', 0, None, None, 0)
loc_sd = Beta('loc_sd', 1, None, None, 0)
wk_mean = Beta('wk_mean', 0, None, None, 0)
wk_sd = Beta('wk_sd', 1, None, None, 0)
tod_mean = Beta('tod_mean', 0, None, None, 0)
tod_sd = Beta('tod_sd', 1, None, None, 0)
seas_mean = Beta('seas_mean', 0, None, None, 0)
seas_sd = Beta('seas_sd', 1, None, None, 0)

pf_rnd = pf_mean + pf_sd * Draws('pf_rnd', 'NORMAL')
cl_rnd = cl_mean + cl_sd * Draws('cl_rnd', 'NORMAL')
loc_rnd = loc_mean + loc_sd * Draws('loc_rnd', 'NORMAL')
wk_rnd = wk_mean + wk_sd * Draws('wk_rnd', 'NORMAL')
tod_rnd = tod_mean + tod_sd * Draws('tod_rnd', 'NORMAL')
seas_rnd = seas_mean + seas_sd * Draws('seas_rnd', 'NORMAL')

choice = Variable('choice')

V = {
    j: pf_rnd * X['pf'][j] + cl_rnd * X['cl'][j] + loc_rnd * X['loc'][j] + wk_rnd * X['wk'][j] + tod_rnd * X['tod'][j] + seas_rnd * X['seas'][j]
    for j in [1,2,3,4]
}


# # Make the models
# Jaxlogit:

# In[ ]:


model_jax.fit(
    X=X_train,
    y=y_train,
    varnames=varnames,
    ids=ids_train,
    alts=alts_train,
    randvars=randvars,
    config=config
)
display(model_jax.summary())
init_coeff_j = model_jax.coeff_


# xlogit:

# In[ ]:


model_x.fit(
    X=X_train,
    y=y_train,
    varnames=varnames,
    ids=ids_train,
    alts=alts_train,
    randvars=randvars,
    panels=panels_train,
    n_draws=n_draws,
    skip_std_errs=True,  # skip standard errors to speed up the example
    batch_size=None,
    optim_method="L-BFGS-B",
)
display(model_x.summary())
init_coeff_x = model_x.coeff_


# Biogeme:

# In[ ]:


prob = models.logit(V, None, choice)
logprob = log(MonteCarlo(PanelLikelihoodTrajectory(prob)))

the_biogeme = bio.BIOGEME(
    database_train, logprob, number_of_draws=n_draws, seed=999, generate_yaml=False, generate_html=False
)
the_biogeme.model_name = 'model_b'
results = the_biogeme.estimate()
print(results)


# # Compare parameters:

# In[ ]:


print("{:>13} {:>13} {:>13} {:>13}".format("Estimate", "Jaxlogit", "xlogit", "Biogeme"))
print("-" * 58)
fmt = "{:13} {:13.7f} {:13.7f} {:13.7f}"
coeff_names = {'pf': 'pf_mean', 'sd.pf': 'pf_sd', 'cl': 'cl_mean', 'sd.cl': 'cl_sd', 'loc': 'loc_mean', 'sd.loc': 'loc_sd', 'wk': 'wk_mean', 'sd.wk': 'wk_sd', 'tod': 'tod_mean', 'sd.tod': 'tod_sd', 'seas': 'seas_mean', 'sd.seas': 'seas_sd'}
for i in range(len(model_jax.coeff_)):
    name = model_jax.coeff_names[i]
    print(fmt.format(name[:13], 
                     model_jax.coeff_[i], 
                     model_x.coeff_[i], 
                     results.get_beta_values()[coeff_names[name]]))
print("-" * 58)


# # Predict
# jaxlogit:

# In[ ]:


model = model_jax 
config = ConfigData(
    panels=panels_test,
    n_draws=n_draws,
    skip_std_errs=True,  # skip standard errors to speed up the example
    batch_size=None,
    optim_method="L-BFGS-B",
)
config.init_coeff = init_coeff_j


# In[ ]:


prob_jj = model.predict(X_test, varnames, alts_test, ids_test, randvars, config)


# xlogit:

# In[ ]:


_, prob_xx = model_x.predict(X_test, varnames, alts_test, ids_test, isvars=None, panels=panels_test, n_draws=n_draws, return_proba=True)


# Biogeme:


# In[ ]:


P = {
    j: MonteCarlo(models.logit(V, None, j))
    for j in [1, 2, 3, 4]
}

simulate = {
    f'Prob_alt{j}': P[j]
    for j in [1, 2, 3, 4]
}

biogeme_sim = bio.BIOGEME(database_test, simulate)
biogeme_sim.model_name = 'per_choice_probs'

probs = biogeme_sim.simulate(results.get_beta_values())


# Compute the brier score:

# In[ ]:


print("{:>9} {:>9} {:>9}".format("Jaxlogit", "xlogit", "Biogeme"))
print("-" * 31)
fmt = "{:9f} {:9f} {:9f}"
print(fmt.format(sklearn.metrics.brier_score_loss(np.reshape(y_test, (prob_jj.shape[0], -1)), prob_jj),
                 sklearn.metrics.brier_score_loss(np.reshape(y_test, (prob_xx.shape[0], -1)), prob_xx),
                 sklearn.metrics.brier_score_loss(df_wide_test['choice'], probs)))
print("-" * 31)

