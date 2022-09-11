#Data Preprocessing
#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("50_Startups.csv")

#Independent Variable Matrix/ Vector
X = dataset.iloc[:,:-1].values

#Making Dependent Variable Matrix/ Vector
y= dataset.iloc[:, 4].values

#Encoding/Labeling Categorical Data
#Encoding/Labeling Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)

#Avoiding the Dummy Variable Trap
X = X[:,1:]

#Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

#1st Iteration
X_opt = X[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

#2nd Iteration
X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

#3rd Iteration
X_opt = X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

#4th Iteration
X_opt = X[:,[0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

#Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train_be, X_test_be, y_train_be, y_test_be = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)


#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor_be = LinearRegression()
regressor_be.fit(X_train_be, y_train_be)

#Predicting the Test set results
y_pred_be = regressor_be.predict(X_test_be)
