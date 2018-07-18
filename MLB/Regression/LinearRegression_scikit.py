# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:59:29 2017

@author: BEAST
Scikit-Learn의 LinearRegression을 통한 계산

"""

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def predict(x):
    return w0 + w1*x

X_train =np.array([ [10], [20],[30], [50]])
y_train =np.array([ [25], [45],[65], [105] ])


model = LinearRegression(fit_intercept=True) # 상수항이 있으면
model.fit(X_train, y_train)

X_test = 40
y_predict = model.predict(X_test)


y_pred    = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
print(mse)




print("가중치: ", model.coef_)
print("상수 : ", model.intercept_)
print("예상 값 :", " x 값 :", X_test, " y_predict :", y_predict)

w1 = model.coef_
w0 = model.intercept_

x_new = np.arange(0, 51)
y_new1 = predict(x_new)
y_new = y_new1.reshape(-1,1)




plt.scatter(X_train, y_train, label="data")
plt.plot(x_new, y_new, 'r-', label="regression")
plt.scatter(X_test, y_predict, label="predict")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression with sckit_learn")
plt.legend()
plt.show()