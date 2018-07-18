'''
#기울기
def getDifferential(f,x):                   #미분
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def equation1(x):
    return 10*x**3 + 5*x**2 + 4*x

def equation2(x):
    return 0.01*x**2 + 0.1*x

d= getDifferential(equation1, 5)
print(d)

d = getDifferential(equation2, 10)
print(d)
'''

#편미분
def getDifferential(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)

#식 f = x**2 + y**2
#점 x=3, y=4일때 편미분

def function_1(x):
    return x*x +  4.0 ** 2.0                    # x^2 + 4^2 // (3,4)일때의 미분, y = 4 인걸 수치적으로 넣었음 // 값엔 변화 X
print(getDifferential(function_1, 3.0))

def function_2(y):
    return 3.0 ** 2.0 + y*y
print(getDifferential(function_2, 4.0))