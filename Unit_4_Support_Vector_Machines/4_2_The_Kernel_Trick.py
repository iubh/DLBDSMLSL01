# IU - International University of Applied Science
# Machine Learning - Supervised Learning
# Course Code: DLBDSMLSL01

# The Kernel Trick

# %% load packages
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# %% load and prepare the data
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Categorical.from_codes(dataset.target, dataset.target_names)
y = pd.get_dummies(y, drop_first=True)

# %% split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.3, random_state=42)

# %% specify and train the model
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train.values.ravel())

# %% use the model to predict values
y_pred = clf.predict(X_test)

# %% print the confusion matrix
print(confusion_matrix(y_test, y_pred))

# %% print accuracy
print(accuracy_score(y_test, y_pred))
# console output: 0.935672514619883