# IU - International University of Applied Science
# Machine Learning - Supervised Learning
# Course Code: DLBDSMLSL01

# Naive Bayes

# %% load packages
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# %% load the data
exam_data = pd.read_csv('bayes_data.csv', sep=';')
print(exam_data)

# %% prepare the data
X = exam_data.drop(columns=['Passed'])
y = exam_data['Passed']
X = pd.get_dummies(X)

# %% specify and fit the model
model = GaussianNB()
model.fit(X, y)

# %% load and prepare the test data
test_data = pd.read_csv('bayes_test_data.csv', sep=';')
X_test = test_data.drop(columns=['Passed'])
y_test = test_data['Passed']
X_test = pd.get_dummies(X_test)

# %% use the model to predict values
y_pred = model.predict(X_test)
print('Prediction results:')
print(y_pred)
