# IU - International University of Applied Science
# Machine Learning - Supervised Learning
# Course Code: DLBDSMLSL01

# Random Forest

# %% load libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# %% load dataset and print 1st five rows
dataset = pd.read_csv('takingawalk_dataset.csv', sep=';')
print(dataset.head())

# %% prepare the data
X = dataset.drop(columns=['Label', 'Week'])
y = dataset['Label']
X = pd.get_dummies(X)
print(X.columns)

# %% split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size=0.3, shuffle=True, random_state=42)

# %% specify and train the model
clf = RandomForestClassifier(n_estimators=100, max_depth=3,
                             random_state=42)
clf.fit(X_train, y_train)

# %% use the model to predict values
y_pred = clf.predict(X_test)

# %% print the confusion matrix
print(confusion_matrix(y_test, y_pred))

# %% print accuracy
print(accuracy_score(y_test, y_pred))
# console output: 0.9375

# %% extract feature importances
feature_scores = pd.Series(clf.feature_importances_,
    index=X_train.columns).sort_values(ascending=False)
print(feature_scores)

# %%
