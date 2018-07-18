import statsmodels.api as sm
import numpy as np
X1 = np.array([[10], [20], [30], [50]])

X_train = sm.add_constant(X1)  #오그멘테이션
print(X_train)
print(X_train.shape)

# X1.shape : (4:1)
print(np.ones((X1.shape[0],1)))
X_train1 = np.hstack([np.ones((X1.shape[0], 1)), X1])
print(X_train1)