# IU - International University of Applied Science
# Machine Learning - Supervised Learning
# Course Code: DLBDSMLSL01

# Linear Regression

# %% load packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %% load data
dataset = pd.read_csv('Salary_Data.csv')
X_train = dataset.drop(columns=['Salary'])
y_train = dataset.iloc[:,1].values

# %% build and train the model
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# %% plot the data and the model
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('Salary vs Experience(Train set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary')

# %% print the intercept and coefficients
print(regressor.intercept_, regressor.coef_)

# %% use the model to predict values
# y_pred = regressor.predict(X_newdata)