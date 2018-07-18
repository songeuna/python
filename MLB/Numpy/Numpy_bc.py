import numpy as np

'''
x = np.array([[2, 3], [4, 5]])
y = 5

x1 = x + y

print('x type is ', type(x))
print('y type is ', type(y))
print(x1)                           # y값을 [[5,5],[5,5]] 배열로 브로드캐스팅해서 계산
'''

x2 = np.array([[2, 3], [4, 5]])
y2 = np.array([5, 10])

x3 = x2 + y2
print('x2 shape is : ', x2.shape)
print('y2 shape is : ', y2.shape)
print('x3 shape is : ', x3.shape)
print('x3 is : \n', x3)