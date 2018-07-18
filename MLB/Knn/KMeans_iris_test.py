# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:44:21 2017

@author: BEAST
"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score

#np.random.seed(5)
iris = datasets.load_iris()
X_train = iris.data
y_train = iris.target

print('y_train : ', y_train)

estimators = [('K_means 8', KMeans(n_clusters=8)),
              ('K_Means 3', KMeans(n_clusters=3))]
              #('K_means 5', KMeans(n_clusters=5))]

fignum = 1
titles = ['8 clusters', '3 clusters']
#estimators = np.ndarray([])
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X_train)
    labels = est.labels_

    print(name, ':', est.labels_)
    ax.scatter(X_train[:,3], X_train[:,0], X_train[:,2],                # x좌표, y좌표, z좌표
                c=labels.astype(np.float), edgecolor='k')


    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

    print('Sihouette_score : ', silhouette_score(X_train, est.labels_, metric='euclidean',
                                                  sample_size=len(X_train)))
# Plot the ground truth

fig = plt.figure(fignum, figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X_train[y_train == label, 3].mean(),
              X_train[y_train == label, 0].mean(),
              X_train[y_train == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results
y = np.choose(y_train, [1, 2, 0]).astype(np.float)
ax.scatter(X_train[:, 3], X_train[:, 0], X_train[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')  # 3
ax.set_ylabel('Sepal length') # 0
ax.set_zlabel('Petal length') # 2
ax.set_title('Ground Truth')
ax.dist = 12

plt.show()





