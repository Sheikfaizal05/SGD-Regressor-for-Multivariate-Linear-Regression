# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets.
2.Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance.
3.Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data.
4.Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error.
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:SHEIK FAIZAL S
RegisterNumber:24900982
*/
```
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
x = data.data[:, :3]
y = np.column_stack((data.target, data.data[:, 6]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train, y_train)

y_pred = multi_output_sgd.predict(x_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test, y_pred)

print("Predictions:\n", np.array2string(y_pred, separator=', '))
print("Squared Error:", mse)
```

## Output:
 [[ 1.10294192, 35.91524355],
 [ 1.51829697, 35.80348534],
 [ 2.2781599 , 35.71878728],
 ...,
 [ 4.31784449, 35.02074527],
 [ 1.70545882, 35.76972047],
 [ 1.81947425, 35.71433187]]
Squared Error: 2.6030095425883086

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
