# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:45:46 2017

@author: BEAST
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Input data
X_train = np.array([[3.2, 3.1], [4.2, 4.2], [1.9, 6.5], [4.1, 5.0], [5.1, 6.9],
                    [2.3, 5.3], [3.2, 5.5], [3.5, 3.7], [4.5, 4.1], [3.4, 5.9],
                    [4.1, 3.5], [4.1, 5.7], [3.1, 4.2], [5.2, 4.2], [4.7, 6.5]])


# 이웃의 갯수 선정
k = 5

# New_input_data
new_input_data = [4.3, 4.7]

# Plot input data
#plt.figure()
plt.subplot(2, 1,  1)
plt.scatter(X_train[:,0], X_train[:,1], marker='o', s=100, color='k', label='Input Data')
plt.scatter(new_input_data[0], new_input_data[1], marker='o', s=100, color='r', label='New Input Data')
plt.legend()

# Build K Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
knn_model.fit(X_train)


# 최 근접 이웃  인덱
distance, indices = knn_model.kneighbors([new_input_data])

print("distance : ", distance)
print('indices : ', indices)

cnt = 0
# 값 확
for index in indices[0]:
    print("좌표값 : {0}, 거리 : [1]".format(X_train[index], distance[0][cnt]))
    cnt += 1



# Visualize the nearest neighbors along with the test datapoint
#plt.figure()
plt.subplot(2, 1,  2)
plt.title( '{0} Nearest neighbors'.format(k))


plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=75, color='k', label='Input Data')
#plt.scatter(X_train[indices][0][:][:, 0], X_train[indices][0][:][:, 1], marker='o', s=250, color='k', facecolors='none')
plt.scatter(new_input_data[0], new_input_data[1],
        marker='o', s=200, color='r', label='New Input Data')
# Print the 'k' nearest neighbors
print("\nK Nearest Neighbors:")

#화살표 그리는 patplot
for rank, index in enumerate(indices[0][:k], start=1):
    print(str(rank) + " ==>", X_train[index])
    plt.arrow(new_input_data[0], new_input_data[1], X_train[index, 0] - new_input_data[0], X_train[index, 1] - new_input_data[1], head_width=0, fc='k', ec='k')


plt.legend()
plt.show()


