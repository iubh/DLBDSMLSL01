# IU - International University of Applied Science
# Machine Learning - Supervised Learning
# Course Code: DLBDSMLSL01

# Lasso and Ridge Regression

# %% load packages
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet  

# %% load data
dataset = pd.read_csv('Salary_Data.csv')
X_train = dataset.drop(columns=['Salary'])
y_train = dataset.iloc[:,1].values

# %% build and train a ridge model
ridge_regressor = Ridge(alpha=1)
ridge_regressor.fit(X_train,y_train)

# %% print the regression coefficients
print(ridge_regressor.coef_)

# %% build and train a lasso model
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train,y_train)

# %% print the regression coefficients
print(lasso_reg.coef_)

# %% build and train an elastic net model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5) 
elastic_net.fit(X_train,y_train)

# %% print the regression coefficients
print(elastic_net.coef_)

# %% add a redundant variable
dataset['YearsExp/100'] = dataset['YearsExperience']/100
X_train = dataset.drop(columns=['Salary'])
y_train = dataset.iloc[:,1].values

# %% build and train an elastic net model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5) 
elastic_net.fit(X_train,y_train)

# %% print the regression coefficients
print(elastic_net.coef_)

# %% build and train an elastic net model
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.99) 
elastic_net.fit(X_train,y_train)

# %% print the regression coefficients
print(elastic_net.coef_)


# %%
