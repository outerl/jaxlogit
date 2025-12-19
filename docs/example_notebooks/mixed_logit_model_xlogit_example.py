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
# # Mixed Logit

# %% colab={"base_uri": "https://localhost:8080/"} id="NQbZt7CVh8f_" outputId="b823e80f-fd47-4dd1-8656-3fd0d6a1e26a"
import pandas as pd
import numpy as np
import jax

from jaxlogit.mixed_logit import MixedLogit, ConfigData

# %%
#  64bit precision
jax.config.update("jax_enable_x64", True)

# %% [markdown] id="MP77ezqVfvRI"
# ## Swissmetro Dataset

# %% [markdown] id="BOWB3Lffg5Qc"
#
# The swissmetro dataset contains stated-preferences for three alternative transportation modes that include car, train and a newly introduced mode: the swissmetro. This dataset is commonly used for estimation examples with the `Biogeme` and `PyLogit` packages. The dataset is available at http://transp-or.epfl.ch/data/swissmetro.dat and [Bierlaire et. al., (2001)](https://transp-or.epfl.ch/documents/proceedings/BierAxhaAbay01.pdf) provides a detailed discussion of the data as wells as its context and collection process. The explanatory variables in this example include the travel time (`TT`) and cost `CO` for each of the three alternative modes.

# %% [markdown] id="n4No84MAeFOM"
# ### Read data

# %% [markdown] id="TEzmVzYDdLS8"
# The dataset is imported to the Python environment using `pandas`. Then, two types of samples, ones with a trip purpose different to commute or business and ones with an unknown choice, are filtered out. The original dataset contains 10,729 records, but after filtering, 6,768 records remain for following analysis. Finally, a new column that uniquely identifies each sample is added to the dataframe and the `CHOICE` column, which originally contains a numerical coding of the choices, is mapped to a description that is consistent with the alternatives in the column names. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="4jqERhnWhGCc" outputId="6bbdca2a-1670-4836-c0d5-d16915ee9597"
df_wide = pd.read_table("http://transp-or.epfl.ch/data/swissmetro.dat", sep='\t')

# Keep only observations for commute and business purposes that contain known choices
df_wide = df_wide[(df_wide['PURPOSE'].isin([1, 3]) & (df_wide['CHOICE'] != 0))]

df_wide['custom_id'] = np.arange(len(df_wide))  # Add unique identifier
df_wide['CHOICE'] = df_wide['CHOICE'].map({1: 'TRAIN', 2:'SM', 3: 'CAR'})
df_wide

# %% [markdown] id="-GRMhgM2eIPz"
# ### Reshape data

# %% [markdown] id="r9OxW-yNhcal"
# The imported dataframe is in wide format, and it needs to be reshaped to long format for processing by `xlogit`, which offers the convenient `wide_to_long` utility for this reshaping process. The user needs to specify the column that uniquely identifies each sample, the names of the alternatives, the columns that vary across alternatives, and whether the alternative names are a prefix or suffix of the column names. Additionally, the user can specify a value (`empty_val`) to be used by default when an alternative is not available for a certain variable. Additional usage examples for the `wide_to_long` function are available in xlogit's documentation at https://xlogit.readthedocs.io/en/latest/notebooks/convert_data_wide_to_long.html. Also, details about the function parameters are available at the [API reference ](https://xlogit.readthedocs.io/en/latest/api/utils.html#xlogit.utils.wide_to_long).

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="1KM-BvFvhWed" outputId="33a6bacf-9674-4fec-eeca-90ab763a3308"
from jaxlogit.utils import wide_to_long

df = wide_to_long(df_wide, id_col='custom_id', alt_name='alt', sep='_',
                  alt_list=['TRAIN', 'SM', 'CAR'], empty_val=0,
                  varying=['TT', 'CO', 'HE', 'AV', 'SEATS'], alt_is_prefix=True)
df

# %% [markdown] id="PhLjuzaSeVjE"
# ### Create model specification

# %% [markdown] id="dgJP2WdQeXiY"
# Following the reshaping, users can create or update the dataset's columns in order to accommodate their model specification needs, if necessary. The code below shows how the columns `ASC_TRAIN` and `ASC_CAR` were created to incorporate alternative-specific constants in the model. In addition, the example illustrates an effective way of establishing variable interactions (e.g., trip costs for commuters with an annual pass) by updating existing columns conditional on values of other columns. Although apparently simple, column operations provide users with an intuitive and highly-flexible mechanism to incorporate model specification aspects, such as variable transformations, interactions, and alternative specific coefficients and constants. By operating the dataframe columns, any utility specification can be accommodated in `xlogit`. As shown in [this specification example](https://xlogit.readthedocs.io/en/latest/notebooks/multinomial_model.html#Create-model-specification), highly-flexible utility specifications can be modeled in `xlogit` by operating the dataframe columns.

# %% id="MsSu2jqKeoz-"
df['ASC_TRAIN'] = np.ones(len(df))*(df['alt'] == 'TRAIN')
df['ASC_CAR'] = np.ones(len(df))*(df['alt'] == 'CAR')
df['TT'], df['CO'] = df['TT']/100, df['CO']/100  # Scale variables
annual_pass = (df['GA'] == 1) & (df['alt'].isin(['TRAIN', 'SM']))
df.loc[annual_pass, 'CO'] = 0  # Cost zero for pass holders

# %% [markdown] id="JgjEi8QLexj6"
# ### Estimate model parameters

# %% [markdown] id="gzuSO2UBe99t"
# The `fit` method estimates the model by taking as input the data from the previous step along with additional specification criteria, such as the distribution of the random parameters (`randvars`), the number of random draws (`n_draws`), and the availability of alternatives for the choice situations (`avail`). We set the optimization method as `L-BFGS-B` as this is a robust routine that usually helps solve convergence issues.  Once the estimation routine is completed, the `summary` method can be used to display the estimation results.

# %% colab={"base_uri": "https://localhost:8080/"} id="dctkrAPBez4T" outputId="eaacd08d-fec7-4b8d-b0dd-757e4f3eac15"
varnames=['ASC_CAR', 'ASC_TRAIN', 'CO', 'TT']
model = MixedLogit()

config = ConfigData(
    avail=df['AV'],
    panels=df["ID"],
    n_draws=1500,
)

res = model.fit(
    X=df[varnames],
    y=df['CHOICE'],
    varnames=varnames,
    alts=df['alt'],
    ids=df['custom_id'],
    randvars={'TT': 'n'},
    config=config
)
model.summary()

# %% [markdown] id="iwsZACbNgJwd"
# The negative signs for the cost and time coefficients suggest that decision makers experience a general disutility with alternatives that have higher waiting times and costs, which conforms to the underlying decision making theory. Note that these estimates are highly consistent with those returned by Biogeme (https://biogeme.epfl.ch/examples/swissmetro/05normalMixtureIntegral.html)

# %% [markdown] id="dt6rAYtH3Djj"
# ## Electricity Dataset

# %% [markdown] id="_k8_iTHPn7l9"
# The electricity dataset contains 4,308 choices among four electricity suppliers based on the attributes of the offered plans, which include prices(pf), contract lengths(cl), time of day rates (tod), seasonal rates(seas), as well as attributes of the suppliers, which include whether the supplier is local (loc) and well-known (wk). The data was collected through a survey where 12 different choice situations were presented to each participant. The multiple responses per participants were organized into panels. Given that some participants answered less than 12 of the choice situations, some panels are unbalanced, which `xlogit` is able to handle. [Revelt and Train (1999)](https://escholarship.org/content/qt1900p96t/qt1900p96t.pdf) provide a detailed description of this dataset. 

# %% [markdown] id="wOjSrftv3Gtm"
# ### Read data

# %% [markdown] id="ymoa7_h4oZo_"
# The dataset is already in long format so no reshaping is necessary, it can be used directly in xlogit.

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="fLgHickp3IJw" outputId="d5873903-e4b9-4278-cc97-a7f96c016b2a"
df = pd.read_csv("https://raw.github.com/arteagac/xlogit/master/examples/data/electricity_long.csv")
df

# %% [markdown] id="gFUpTIpU3-Oi"
# ### Fit the model

# %% [markdown] id="0p7isMbqoIYz"
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

# %% [markdown] id="r9gxNL0XePRc"
# ## Fishing Dataset

# %% [markdown] id="Innu2ypbmKui"
# This example illustrates the estimation of a Mixed Logit model for choices of 1,182 individuals for sport fishing modes using `xlogit`. The goal is to analyze the market shares of four alternatives (i.e., beach, pier, boat, and charter) based on their cost and fish catch rate. [Cameron (2005)](http://cameron.econ.ucdavis.edu/mmabook/mma.html) provides additional details about this dataset. The following code illustrates how to use `xlogit` to estimate the model parameters.

# %% [markdown] id="cqBJWh8eOQDp"
# ### Read data

# %% [markdown] id="GqCeQywgozbK"
# The data to be analyzed can be imported to Python using any preferred method. In this example, the data in CSV format was imported using the popular `pandas` Python package. However, it is worth highlighting that `xlogit` does not depend on the `pandas` package, as `xlogit` can take any array-like structure as input. This represents an additional advantage because `xlogit` can be used with any preferred dataframe library, and not only with `pandas`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="9jDr3PIveaG8" outputId="490ab844-e202-4ba3-cdcf-cf64c54fc698"
import pandas as pd
df = pd.read_csv("https://raw.github.com/arteagac/xlogit/master/examples/data/fishing_long.csv")
df

# %% [markdown] id="rffV7cx8ORpP"
# ### Fit model

# %% [markdown] id="UFcmUz8Xo6UT"
# Once the data is in the `Python` environment, `xlogit` can be used to fit the model, as shown below. The `MultinomialLogit` class is imported from `xlogit`, and its constructor is used to initialize a new model. The `fit` method estimates the model using the input data and estimation criteria provided as arguments to the method's call. The arguments of the `fit` methods are described in [`xlogit`'s documentation](https://https://xlogit.readthedocs.io/en/latest/api/).
#

# %% colab={"base_uri": "https://localhost:8080/"} id="FIZwBe0zedfh" outputId="46561fa7-3339-4fb1-9c93-3c2a4a646c96"
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

# %% [markdown] id="mWU80LmcODPY"
# ## Car Dataset

# %% [markdown] id="q1zgiBGKouPr"
# The fourth example uses a stated preference panel dataset for choice of car. Three alternatives are considered, with upto 6 choice situations per individual. This again is an unbalanced panel with responses of some individuals less than 6 situations. The dataset contains 8 explanaotry variables: price, operating cost, range, and binary indicators to indicate whether the car is electric, hybrid, and if performance is high or medium respectively. This dataset was taken from Kenneth Train's MATLAB codes for estimation of Mixed Logit models as shown in this link: https://eml.berkeley.edu/Software/abstracts/train1006mxlmsl.html

# %% [markdown] id="SoSyQfjqkNU3"
# ### Read data

# %% id="v8AAMruCj8tt"
import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.github.com/arteagac/xlogit/master/examples/data/car100_long.csv")

# %% [markdown] id="0HY7mT__Lj5b"
# Since price and operating cost need to be estimated with negative coefficients, we reverse the variable signs in the dataframe. 

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="TQ33gsZZLkP5" outputId="7acb35c7-38b4-4017-b0ac-ae1dd2ef8b59"
df['price'] = -df['price']/10000
df['opcost'] = -df['opcost']
df

# %% [markdown] id="_ZQf9DFKFE5j"
# ### Fit the model

# %% colab={"base_uri": "https://localhost:8080/"} id="_MhfvmWgFCX6" outputId="28de1f2f-2e7c-46be-aa60-105df6b669bc"
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

# %%
jax.nn.softplus(model.coeff_[len(model._rvidx):])

# %% [markdown] id="pEiWWuCciEJ6"
# ## References

# %% [markdown] id="UkfFmr7fiFxc"
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
