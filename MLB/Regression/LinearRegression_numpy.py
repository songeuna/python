# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:59:29 2017

@author: BEAST

LinearRegression_numpy.py
션형 회귀 분석
  numpy 패키지를 사용한 벡터의 내적을 통한 계산

"""


import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def predict(x):
    return w0 + w1*x


X1 =np.array([ [10], [20],[30], [50]])
y_label  =np.array([ [25], [45],[65],  [105] ])

X_train = sm.add_constant(X1) # 오그멘테이션
print()

print("X1 : ", X_train, 'shape', X1.shape)


w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), y_label)
# 2 * 4 행렬  .  4*1  행렬 --> \ : 2 * 1 행렬

temp = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T)

print(temp.shape)

print(w.shape)


w0 = w[0]
w1 = w[1]


X_test = 40
y_predict = predict(X_test)


print("가중치: ", w1)
print("상수 : ", w0)
print("예상 값 :", " x 값 :", X_test, " y_predict :", y_predict)



x_new = np.arange(0,51)
y_new = predict(x_new)

print(x_new)
print(y_new)


plt.scatter(X1, y_label, label="data")


plt.scatter(X_test, y_predict, label="predict")
plt.plot(x_new, y_new, 'r-', label="regression")
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("Linear Regression _ with numpy")
plt.legend()
plt.show()
