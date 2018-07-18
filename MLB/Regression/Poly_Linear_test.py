# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:42:10 2017

@author: BEAST
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:35:29 2017

@author: BEAST
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.metrics import r2_score

def make_nl_sample():
    np.random.seed(0)
    samples_number = 50
    X = np.sort(np.random.rand(samples_number))
    y = np.sin(2 * np.pi * X) + np.random.randn(samples_number) * 0.2
    X = X[:, np.newaxis]
    return (X, y)

X_train, y_train = make_nl_sample()

model = LinearRegression().fit(X_train, y_train)        #훈련값과 목적값으로 model을 훈련시킴
predict = model.predict(X_train)        #예측값을 만듬

n_degree = 3                        #3차항으로 다항 회기?를 만듬

# Polynomial regression
poly_linear_model = LinearRegression()      #3차만드는 함수와 linear함수는 같음

# 차수에 맞는 형태의 데이터 형태 변환       #가지고 있는 데이터를 차수형태로 변형해야 그래프를 그릴수있음
polynomial = PolynomialFeatures(n_degree)                       #데이터 변환 함수
X_train_transformed = polynomial.fit_transform(X_train)         #데이터 변환 함수

#print("X_train_transformed.shape :", X_train_transformed.shape)

#linear 데이터를 주면 linear하게, 다항 변환형태를 주면 다항모델로..

poly_linear_model.fit(X_train_transformed, y_train)         #다항 회기분석 모델을 만들어서 형태에 맞게 변환하고 fit\
#y_train값은 하나, 따라서 X_train에 대한 값을 매핑만하면 되기때문에 다항형태로 변환하지 않아도 됌.
pre2 = poly_linear_model.predict(X_train_transformed)       #훈련한값을 넣었음, 훈련한값을 넣으면 얼마나 똑같이 리턴하는지 알아보기위해서

'''
print(y_train)
print(predict)
print(pre2)
'''

'''
print(X_train.shape)
print(y_train.shape)
'''

linear_r2_score = r2_score(y_train, predict)
poly_r2_score = r2_score(y_train, pre2)

plt.scatter(X_train, y_train, label='Training Data')
plt.plot(X_train, predict, label="Linear Regression", color = 'r')
plt.plot(X_train, pre2, label='Poly Regression', color='b')
plt.legend()        #옆에 네모박스에 설명을 주겠따
plt.title("Degree : {}\n linear_r2_score : {:.2e}\n poly_r2_score : {:.2e} ".format(n_degree, linear_r2_score, poly_r2_score))
plt.show()



