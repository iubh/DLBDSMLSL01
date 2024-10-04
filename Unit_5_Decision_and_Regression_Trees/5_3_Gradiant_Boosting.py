# IU - International University of Applied Science
# Machine Learning - Supervised Learning
# Course Code: DLBDSMLSL01

# Gradient Boosting

# %% load packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
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

# %% specify the classifier to be trained
clf = GradientBoostingClassifier(n_estimators=100,
    max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# %% use the model to predict values
y_pred = clf.predict(X_test)

# %% print the confusion matrix
print(confusion_matrix(y_test, y_pred))
# console output:
# [[10  1]
#  [ 0  5]]

# %% print accuracy
print(accuracy_score(y_test, y_pred))
# console output: 0.9375

# %% extract feature importances
feature_scores = pd.Series(clf.feature_importances_,
    index=X_train.columns).sort_values(ascending=False)
print(feature_scores)

# console output:
# Wind_No            0.196809
# Outlook_Rainy      0.187876
# Outlook_Sunny      0.187626
# Wind_Yes           0.185413
# Humidity_Normal    0.129333
# Humidity_High      0.112943
# dtype: float64