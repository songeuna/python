# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:29:03 2017


숫자인식 예제
PythonProgramming.net 참조
@author: BEAST
"""



import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import numpy as np

digits = datasets.load_digits()
#X_train,y_train = digits.data[:-10], digits.target[:-10]
X_train,y_train = digits.data, digits.target
print('digits.images :', digits.images)

print('digits.data : ' , digits.data)
print('digits.data : ' , digits.data.shape)

print('digits.target :', digits.target)
print('digits.target :', digits.target.shape)


digits_index = 3

print(digits.data[digits_index].reshape(1,-1))

svm_model = svm.SVC(gamma=0.00001, C=100)
svm_model.fit(X_train,y_train)

#x_test = digits.data[digits_index].reshape(1,-1)
x_test = np.array([[ 0.,  0.,  0., 4. ,15., 12.,  0.,  0. , 0.,  0. , 3. ,16., 15., 14.,  0.,  0. , 0. , 0.,
   8., 13. , 8. ,16.,  0.,  0. , 0.,  0.,  1.,  6., 15. ,11. , 0. , 0.,  0.,  1. , 8., 13.,
  15.,  1. , 0. , 0. , 0.,  9., 16., 16.,  5.,  0.,  0.,  0. , 0.,  3., 13., 16., 16., 11.,
   5.,  0.,  0.,  0.,  0.,  3., 11., 16.,  9.,  0.]])


print('x_test : ' , x_test)

y_test= svm_model.predict(x_test)
print("result " , y_test)

print("Predict Label :",svm_model.predict(digits.data[digits_index].reshape(1,-1))[0])
print("Target Label :", digits.target[digits_index])
plt.imshow(digits.images[digits_index], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()