# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:56:10 2017

@author: BEAST
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

boston = load_boston()
# boston 객체 리턴
print(boston.DESCR)
print(type(boston.DESCR))
print("boston.data.shape :", boston.data.shape, " type :", type(boston.data))
print("boston.target.shape :", boston.target.shape)


# Training Data/Test Data 나누기
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)
# 80프로 정도만 학습시킴
print(X_train.shape)
print(X_test.shape)

#print(X_train[:, 12])

model_boston = LinearRegression(fit_intercept=True).fit(X_train, y_train)
# 선형 회귀 분석 수해하기
#model = LinearRegression().fit(X_train, y_train)

print('model_boston.coef_', model_boston.coef_)
print('model_boston.intercept_', model_boston.intercept_)


#print(boston.keys())
#print(boston.DESCR)

#print("boston.data")
#print(boston.data)

#print("boston.target")

#print(boston.target)

#print("boston.feature_names", boston.feature_names)


# 테스트 데이터에 대한 예측 수행하기
predictions = model_boston.predict(X_test)


# 평가....
print("훈련 세트 점수 : ", model_boston.score(X_train, y_train))
# 테스트 데이터와 테스트 목표 값을 넣어주면, 예측 값을 만들어 비교 해 준다. --> R2 스코어
print("테스트 세트 점수 : ", model_boston.score(X_test, y_test))
print("R2_score : ", r2_score(y_test, predictions)) #r2_score가 1이면 두 데이터가 똑같다.

print("mse :",  mean_squared_error(y_test, predictions))

get_error(y_test, predictions)
plt.scatter(y_test, predictions)
plt.xlabel(" Real Boston House Price")
plt.ylabel(" Predict Boston House Price")
plt.show()