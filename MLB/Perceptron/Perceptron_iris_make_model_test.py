# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:18:08 2017

Perceptron 클래스를 직접 만들어 Iris 데이터 예측에 적용한다.
Perceptron 클래스는 벡터의 내적으로 작성한다.

@author: BEAST
"""

# perceptron.py
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt




# Perceptron 모델을 직접 만들어 사용해 본다.
class Perceptron(object):



    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        """

        return self

    # predict 함수 사용 시 내적 계산을 수행 함.
    def net_input(self, X):
        """Calculate net input"""
        return self

    # net_input 함수를 통해 내적된 결과를 1과 -1로 표현한다
    def predict(self, X):
        """Return class label after unit step"""
        return self


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


# 0과 2 인덱스 값 사용  (100 X 2 )
X_train = []
# 목표 값 설정 (100 X 1)
y_train = []
y_train = np.where(y_train == 'Iris-setosa', -1, 1)


plt.figure()
plt.scatter(X_train[:50, 0], X_train[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X_train[50:100, 0], X_train[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
#plt.show()

plt.figure()
pn = Perceptron(0.1, 10)
pn.fit(X_train, y_train)
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
#plt.show()


plt.figure()
plot_decision_regions(X_train, y_train, classifier=pn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
