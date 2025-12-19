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

# %% [markdown]
# # Comparison of biogeme and jaxlogit on swissmetro data
#
# Comparing panel estimation among other things. Biogeme code mostly from examples at https://biogeme.epfl.ch/sphinx/auto_examples/swissmetro/index.html

# %%
import pandas as pd

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, bioDraws, log, MonteCarlo, PanelLikelihoodTrajectory
import biogeme.database as db
from biogeme.expressions import Variable

# %%
logger = blog.get_screen_logger(level=blog.INFO)

# %%
df = pd.read_table("http://transp-or.epfl.ch/data/swissmetro.dat", sep='\t')

# %%
(((df.PURPOSE != 1) * (df.PURPOSE != 3) + (df.CHOICE == 0)) > 0).value_counts()

# %%
database = db.Database('swissmetro', df)

GROUP = Variable('GROUP')
SURVEY = Variable('SURVEY')
SP = Variable('SP')
ID = Variable('ID')
PURPOSE = Variable('PURPOSE')
FIRST = Variable('FIRST')
TICKET = Variable('TICKET')
WHO = Variable('WHO')
LUGGAGE = Variable('LUGGAGE')
AGE = Variable('AGE')
MALE = Variable('MALE')
INCOME = Variable('INCOME')
GA = Variable('GA')
ORIGIN = Variable('ORIGIN')
DEST = Variable('DEST')
TRAIN_AV = Variable('TRAIN_AV')
CAR_AV = Variable('CAR_AV')
SM_AV = Variable('SM_AV')
TRAIN_TT = Variable('TRAIN_TT')
TRAIN_CO = Variable('TRAIN_CO')
TRAIN_HE = Variable('TRAIN_HE')
SM_TT = Variable('SM_TT')
SM_CO = Variable('SM_CO')
SM_HE = Variable('SM_HE')
SM_SEATS = Variable('SM_SEATS')
CAR_TT = Variable('CAR_TT')
CAR_CO = Variable('CAR_CO')
CHOICE = Variable('CHOICE')

exclude = ((PURPOSE != 1) * (PURPOSE != 3) + (CHOICE == 0)) > 0
print(f"Removing {(((df.PURPOSE != 1) * (df.PURPOSE != 3) + (df.CHOICE == 0)) > 0).sum()} rows from the database based on the exclusion criteria.")
database.remove(exclude)

SM_COST = database.define_variable('SM_COST', SM_CO * (GA == 0))
TRAIN_COST = database.define_variable('TRAIN_COST', TRAIN_CO * (GA == 0))
CAR_AV_SP = database.define_variable('CAR_AV_SP', CAR_AV * (SP != 0))
TRAIN_AV_SP = database.define_variable('TRAIN_AV_SP', TRAIN_AV * (SP != 0))
TRAIN_TT_SCALED = database.define_variable('TRAIN_TT_SCALED', TRAIN_TT / 100)
TRAIN_COST_SCALED = database.define_variable('TRAIN_COST_SCALED', TRAIN_COST / 100)
SM_TT_SCALED = database.define_variable('SM_TT_SCALED', SM_TT / 100)
SM_COST_SCALED = database.define_variable('SM_COST_SCALED', SM_COST / 100)
CAR_TT_SCALED = database.define_variable('CAR_TT_SCALED', CAR_TT / 100)
CAR_CO_SCALED = database.define_variable('CAR_CO_SCALED', CAR_CO / 100)

# %%
# panel data
database.panel('ID')

# %%
B_COST = Beta('B_COST', 0.1, None, None, 0)
B_COST_S = Beta('B_COST_S', 0.75, None, None, 0)
B_COST_RND = B_COST + B_COST_S * bioDraws('b_cost_rnd', 'NORMAL_MLHS_ANTI')

B_TIME = Beta('B_TIME', 0.1, None, None, 0)
B_TIME_S = Beta('B_TIME_S', 0.75, None, None, 0)
B_TIME_RND = B_TIME + B_TIME_S * bioDraws('b_time_rnd', 'NORMAL_MLHS_ANTI')

ASC_CAR = Beta('ASC_CAR', 0.1, None, None, 0)
ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 1) 
ASC_SM = Beta('ASC_SM', 0.1, None, None, 0)

V1 = ASC_TRAIN + B_TIME_RND * TRAIN_TT_SCALED + B_COST_RND * TRAIN_COST_SCALED
V2 = ASC_SM + B_TIME_RND * SM_TT_SCALED + B_COST_RND * SM_COST_SCALED
V3 = ASC_CAR + B_TIME_RND * CAR_TT_SCALED + B_COST_RND * CAR_CO_SCALED

V = {1: V1, 2: V2, 3: V3}
av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

prob = models.logit(V, av, CHOICE)
logprob = log(MonteCarlo(PanelLikelihoodTrajectory(prob)))

the_biogeme = bio.BIOGEME(
    database, logprob, number_of_draws=1000, seed=999,generate_html=False
)
the_biogeme.modelName = 'test'
the_biogeme.generate_pickle = False

# %%
the_biogeme.calculate_init_likelihood()

# %%
results = the_biogeme.estimate()
pandas_results = results.get_estimated_parameters()

print(results.short_summary())

# %%
pandas_results

# %% [markdown]
# ## jaxlogit

# %%
import numpy as np

import jax

from jaxlogit.mixed_logit import MixedLogit, ConfigData
from jaxlogit.utils import wide_to_long

#  64bit precision
jax.config.update("jax_enable_x64", True)

# %%
df_wide = database.data.copy()

df_wide['custom_id'] = np.arange(len(df_wide))  # Add unique identifier
df_wide['CHOICE'] = df_wide['CHOICE'].map({1: 'TRAIN', 2:'SM', 3: 'CAR'})
df_wide['TRAIN_AV'] = df_wide['TRAIN_AV'] * (df_wide['SP'] != 0)
df_wide['CAR_AV'] = df_wide['CAR_AV'] * (df_wide['SP'] != 0)

df_jxl = wide_to_long(
    df_wide, id_col='custom_id', alt_name='alt', sep='_',
    alt_list=['TRAIN', 'SM', 'CAR'], empty_val=0,
    varying=['TT', 'CO', 'HE', 'AV', 'SEATS'], alt_is_prefix=True
)

df_jxl['ASC_TRAIN'] = np.where(df_jxl['alt'] == 'TRAIN', 1, 0)
df_jxl['ASC_CAR'] = np.where(df_jxl['alt'] == 'CAR', 1, 0)
df_jxl['ASC_SM'] = np.where(df_jxl['alt'] == 'SM', 1, 0)

df_jxl['TT'] = df_jxl['TT'] / 100.0
df_jxl['CO'] = df_jxl['CO'] / 100.0

df_jxl.loc[(df_jxl['GA'] == 1) & (df_jxl['alt'].isin(['TRAIN', 'SM'])), 'CO'] = 0  # Cost zero for pass holders

# %%
varnames = ['ASC_SM', 'ASC_CAR', 'ASC_TRAIN', 'TT', 'CO']

randvars = {'CO': 'n', 'TT': 'n'}  

fixedvars = {'ASC_TRAIN': 0.0}

do_panel = True

model = MixedLogit()

config = ConfigData(
    avail=df_jxl['AV'],
    panels=None if do_panel is False else df_jxl["ID"],
    n_draws=1000,
    fixedvars=fixedvars,
    init_coeff=None,
    include_correlations=False,
    optim_method='trust-region',
    skip_std_errs=False,
    force_positive_chol_diag=False,  # not using softplus for std devs here for comparability with biogeme
)

res = model.fit(
    X=df_jxl[varnames],
    y=df_jxl['CHOICE'],
    varnames=varnames,
    alts=df_jxl['alt'],
    ids=df_jxl['custom_id'],
    randvars=randvars,
    config=config
)
model.summary()

# %%
