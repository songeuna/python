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

digits = datasets.load_digits()

X_train, y_train = digits.data, digits.target

digits_index = 9

svm_model = svm.SVC(gamma=0.0001,C=100)
svm_model.fit(X_train, y_train)


plt.imshow(digits.images[digits_index], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()