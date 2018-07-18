'''
x1 = list(range(10))
x2 = x1[0:3]

print('x2 is : ', x2)

x2[1] = 100
print('x2 is : ', x2)

print('x1 is : ', x1)
'''
import numpy as np

y1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print('y1 is : ', y1)

y2 = y1[0:3]
print('y2 is : ', y2)

y2[1] = 100
print('y2 is : ', y2)

print('y1 is : ', y1)