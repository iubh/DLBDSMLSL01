# IU - International University of Applied Science
# Machine Learning - Supervised Learning
# Course Code: DLBDSMLSL01

# Generalized Linear Model (GLM)

# %% load packages
import statsmodels.api as sm
import pandas as pd

# %% load data
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.drop(columns=['Salary'])
y = dataset.iloc[:,1].values

# %% specify exogeneous and endogeneous variables
exog, endog = sm.add_constant(x), y

# %% specify the model
mod = sm.GLM(endog, exog,   
        family=sm.families.Poisson(
             link=sm.families.links.log))

# %% fit the model
res = mod.fit()

# %% print model summary
print(res.summary())
# %%
