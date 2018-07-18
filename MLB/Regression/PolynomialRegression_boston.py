
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:56:10 2017
다항 회귀 분석

보스톤 LSTAT와 MEDV의 관계 분석 ....


@author: BEAST
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


# 보스톤 데이터 set 로드 (scikit-learn)
boston = load_boston()

#머신러닝 훈련을 위한 데이터 생성
bospd= pd.DataFrame(boston.data)
bospd.columns = boston.feature_names

#print(boston.DESCR)

# 훈련 데이터 설정 (인구가 낮은 정도)
X_train = bospd[['LSTAT']].values
y_train = boston.target
#print(X_train.shape)

# 최소값에서 최대값까지 1씩 증가 데이터 생성
X_test  = np.arange(X_train.min(), X_train.max(), 1)[:, np.newaxis]
#X_test  = np.arange(X_train.min(), X_train.max(), 1).reshape(-1,1)
#print(X_test)

# Linear regression
model_boston = LinearRegression()
model_boston.fit(X_train, y_train)
linear_pred = model_boston.predict(X_test)


# Polynomial regression Degress 2
poly_linear_model2 = LinearRegression()
polynomial2 = PolynomialFeatures(degree=2)
#다차에 맞게 데이터 변형
X_train_transformed2 = polynomial2.fit_transform(X_train)
#print("X_train_transformed.shape :", X_train_transformed.shape)
poly_linear_model2.fit(X_train_transformed2, y_train)
# 훈련 데이터 적용
X_test_transformed2 = polynomial2.fit_transform(X_test)
pre2 = poly_linear_model2.predict(X_test_transformed2)


# Polynomial regression Degress 5
poly_linear_model5 = LinearRegression()
polynomial5 = PolynomialFeatures(degree=10)
X_train_transformed5 = polynomial5.fit_transform(X_train)
#print("X_train_transformed.shape :", X_train_transformed.shape)
poly_linear_model5.fit(X_train_transformed5, y_train)


X_test_transformed5 = polynomial5.fit_transform(X_test)
pre5 = poly_linear_model5.predict(X_test_transformed5)


# 평가.... R2
print(r2_score(y_train, model_boston.predict(X_train)))
print(r2_score(y_train, poly_linear_model2.predict(X_train_transformed2)))
print(r2_score(y_train, poly_linear_model5.predict(X_train_transformed5)))



#get_error(y_test, predictions)
plt.scatter(X_train, y_train, label='Training Data', c='grey')
plt.plot(X_test, linear_pred, linestyle='-', label='Linear Regression', c='green')
plt.plot(X_test, pre2, linestyle='--', label='2 Degress Poly Regression', c='red')
plt.plot(X_test, pre5, linestyle=':', label='5 Degress Regression',c='blue')
plt.xlabel(" LSTAT")
plt.ylabel(" HOUSE Price")
plt.legend()
plt.show()