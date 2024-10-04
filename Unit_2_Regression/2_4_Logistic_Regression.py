# IU - International University of Applied Science
# Machine Learning - Supervised Learning
# Course Code: DLBDSMLSL01

# Logistic Regression

# %% load packages
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

# %% load data
iris = datasets.load_iris()

# %% prepare the data
X = iris["data"][:, 3:] # petal width
y = (iris["target"] == 2).astype(int) # 1 if Iris virginica, else 0

# %% define and fit the model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# %% generate new data to base predictions on
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)

# %% predict probabilities for the new data
y_proba = log_reg.predict_proba(X_new)

# %% plot the probabilities for each class
plt.plot(X_new, y_proba[:, 1], "g-", 
    label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", 
         label="Not Iris virginica")

