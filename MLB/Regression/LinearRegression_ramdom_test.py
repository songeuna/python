# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:42:00 2017

@author: BEAST

Scikit-Learn 패키지를 이용한 선형 회귀 분석 모델 작성.

"""

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# scikit_learn에서 제공하는 데이터 제공 함수
X_train, y_train, coef = make_regression(n_samples=50, n_features=1, bias=50, noise=20, coef=True, random_state=1)              #학습
                            #샘플데이터를 만듬
# 상수항이 있으면
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# 선형회귀 직선을 작성하기 위해 데이터 생성
# X_train 데이터의 최대, 최소 값 사이를 100의 데이터로 구분한다.
x_new = np.linspace(np.min(X_train), np.max(X_train), 100)              #matplot에서 그림을 그려야하기 때문

#print(x_new)
X_new = x_new.reshape(-1, 1)
#print(X_new)
#print('X_new :', X_new)

# 그래프를 그리기 위한 y 예측 값 --> 직선을 그리기 위한 x 값과 그에 따른 y 값 정의
y_predict = model.predict(X_new)

# 예측 값
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)       # 예측값과 학습값을 비교
print('MSE :' ,mse)

plt.plot(X_new, y_predict, 'g-', label="regression")
plt.scatter(X_train, y_train, c='r', label="data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()
plt.show()
