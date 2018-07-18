# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:18:43 2017

@author: BEAST
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


# Needed to show the plots inline
# %matplotlib inline
# Data

def make_cal(X_train, y_train, lb):
    # Create the model
    net = Perceptron(max_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
    net.fit(X_train, y_train)
    # Print the results
    print("Prediction :", str(net.predict(X_train)))
    print("Actual     " + str(y_train))
    print("Accuracy :", str(net.score(X_train, y_train)))

    plt.figure()
    # Plot the original data
    plt.scatter(X_train[:,0], X_train[:,1], c=colormap[y_train], s=40)
    #plt.scatter(X_train[:,0], X_train[:,1], c=colormap[y_train], s=40)
    plt.title("Perceptron [{0}] calculation  : ".format(lb))
    print("Coefficient :", net.coef_)

    # Output the values
    print("Codfficient 0:", net.coef_[0,0])
    print("Codfficient 1:", net.coef_[0,1])
    print("Bias :", net.intercept_)

    # Calc the hyperplane (decision boundary)
    ymin, ymax = plt.ylim()

    w = net.coef_[0]
    a = -w[1]/w[0]
    xx = np.linspace(ymin, ymax)
    #yy = a * xx -(net.intercept_[0])/w[0]

    # Plot the line
#    plt.plot(yy, xx, 'k-')
    #plt.show()


colormap = np.array(['r', 'k'])

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# Labels
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])
y_nand = np.array([1, 1, 1, 0])
y_xor = np.array([0, 1, 1, 0])

make_cal(X, y_and, 'and')

make_cal(X, y_or, 'or')

make_cal(X, y_nand, 'nand')

#make_cal(X, y_xor, 'xor')
#plt.show()

