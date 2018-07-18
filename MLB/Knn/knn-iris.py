# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:26:32 2017

@author: BEAST
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:22:23 2017
K-nn 알고리즘에서 k의 갯수를 변경할 경우 결과 확인하기.

@author: BEAST
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset  분꽃받침 너비,
X_train = iris.data[:, :2]
y_train = iris.target

print(y_train)

weights = 'distance'
h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# we create an instance of Neighbours Classifier and fit the data.
k_list = [1, 10, 15, 100]
for k in k_list:
    # print(weights)

    # we create an instance of Neighbours Classifier and fit the data.
    knn_model = neighbors.KNeighborsClassifier(k, weights=weights)
    knn_model.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    print(len(xx), len(yy))
    print(np.c_[xx.ravel(), yy.ravel()].shape)

    Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()

    #plt.subplot(2, 1, 1)


    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (k, weights))

plt.show()
