# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:59:29 2017

@author: BEAST

선형 회귀 계산 공식을 이용한 회귀 분석

"""
import matplotlib.pyplot as plt
import numpy as np

def predict(x):
    return w0 + w1 * x

sample_data = [[10, 25], [20, 45], [30, 65], [50, 105]]

#print(type(sample_data))

X_train = []
y_train = []

X_train_a = []
y_train_a = []

total_size = 0
sum_xy = 0
sum_x = 0
sum_y = 0
sum_x_square = 0

for row in sample_data:
    X_train = row[0]
    y_train = row[1]
    X_train_a.append(row[0])
    y_train_a.append(row[1])

    print(X_train)
    sum_xy += X_train * y_train
    sum_x += X_train
    sum_y += y_train
    sum_x_square += X_train * X_train
    total_size += 1

print("sum_x :", sum_x)
print("sum_y :", sum_y)

print("sum_xy :", sum_xy)
print("sum_x* sum_y :", sum_x * sum_y)
print("sum_x_square :", sum_x_square)
print("sum_x*sum_x :", sum_x * sum_x)
print("sum_x_square*sum_y :", sum_x_square * sum_y)
print("sum_x_square*sum_y :", sum_x_square * sum_y)

w1 = (total_size * sum_xy - sum_x * sum_y) / (total_size * sum_x_square - sum_x * sum_x)
w0 = (sum_x_square * sum_y - sum_xy * sum_x) / (total_size * sum_x_square - sum_x * sum_x)

X_test = 40
y_predict = predict(X_test)

print("가중치: ", w1)
print("상수 : ", w0)
print("예상 값 :", " x 값 :", X_test, " y_predict :", y_predict)

x_new = np.arange(0, 51)
y_new = predict(x_new)

print(x_new.shape)
print(y_new.shape)

# plt.scatter(sample_data[:3,0],sample_data[-1,:3], label="data")

print(sample_data[0:3])

print(X_train)
# list 객체를 인자로 받음.
plt.scatter(X_train_a, y_train_a, label="data")
plt.scatter(X_test, y_predict, label="predict")
plt.plot(x_new, y_new, 'r-', label="regression")
plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("Linear Regression")
plt.legend()
plt.show()

'''
model = LinearRegression(fit_intercept=True) # 상수항이 있으면
model.fit(x_train, y_trina)

x_test = 40
y_predict = model.predict(x_test)



print("가중치: ", model.coef_)
print("상수 : ", model.intercept_)
print("예상 값 :", " x 값 :", x_test, " y_predict :", y_predict)
'''