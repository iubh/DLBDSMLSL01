# IU - International University of Applied Science
# Machine Learning - Supervised Learning
# Course Code: DLBDSMLSL01

# k-Nearest Neighbor

# %% load packages
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# %% load the data
dataset = load_breast_cancer()

# %% prepare the data
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
X = X[['mean smoothness', 'mean concavity', 'radius error']]
y = pd.Categorical.from_codes(dataset.target, dataset.target_names)
y = pd.get_dummies(y, drop_first=True)

# %% split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# %% specify and fit the kNN model
knn = KNeighborsClassifier(n_neighbors=4, metric='manhattan')
knn.fit(X_train, y_train.values.ravel())

# %% use the model to predict values
y_pred = knn.predict(X_test)

# %% print the confusion matrix
print(confusion_matrix(y_test, y_pred))

# %% print accuracy
accuracy_score(y_test, y_pred)
# console output: 0.8671328671328671