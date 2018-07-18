import numpy as np

'''
[1 2 3]
[4 5 6]
'''

'''

#행렬 내적
a = np.array([[1,2,3],[4,5,6]])

print(a.shape)
print(a)

b = np.array([[7,8],[9,10],[11,12]])
c = np.dot(a,b)                             #numpy 안에있는 행렬 내적 함수

print("////////////\n" , a)
print("////////////\n" , b)
print("////////////\n", c.shape)
print("////////////\n", c)
'''

'''
#전치행렬
a = np.array([[1,2,3],[4,5,6]])
at = a.T                                    #numpy 안에있는 전치행렬 함수

print(a)
print(at)
'''

'''
#역행렬
import numpy.linalg as lin

a = np.array([[1,2],[3,4]])
print(a,"\n****************")

c = lin.inv(a)                              #inv = 역행렬 구하는 함수
print(c.shape,"\n****************")
print(c,"\n****************")

e1 = np.dot(a,c)
print(e1)
'''

