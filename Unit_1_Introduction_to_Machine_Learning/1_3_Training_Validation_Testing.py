# IU - International University of Applied Science
# Machine Learning - Supervised Learning
# Course Code: DLBDSMLSL01

# Training, Validation, Testing

# %% import libraries
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, cross_val_score

# %% load data
iris = datasets.load_iris(as_frame=True)

# %% prepare the data
X = iris.data
y = iris.target

# %% use the whole labeled data set for training (not a good idea)
lm = linear_model.LinearRegression()
mod_1 = lm.fit(X, y)

print('R² = ', mod_1.score(X, y))
# console output: R² = 0.9303939218549564 # WRONGWRONGWRONG!

# %% split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# %% use only the training data set for model training
mod_2 = lm.fit(X_train, y_train)

# %% report the coefficient of determination based on the testing data set
print('R² = ', mod_2.score(X_test, y_test))
# console output: R² =  0.8911391254405155

# %% cross validation
cv_scores = cross_val_score(mod_2, X_train, y_train, cv=5)

# %% aggregate CV results
print('Training R² = ', cv_scores.mean())
# console output: Training R² =  0.9286611936278207

# %%
