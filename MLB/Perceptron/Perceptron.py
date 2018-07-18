import numpy as np

def and_cal(x):
    w1, w2, theta = 0.5, 0.5, 0.7

    for x1 in x:
        tmp = x1[0] * w1 + x1[1] * w2

        if tmp <= theta:
            print(x1[0], ' AND ', x1[1], ' = ', 0)
        elif tmp > theta:
            print(x1[0], ' AND ', x1[1], ' = ', 1)

def or_cal(x):
    w1, w2, theta = 0.5, 0.5, 0.2

    for x1 in x:
        tmp = x1[0] * w1 + x1[1] * w2

        if tmp <= theta:
            print(x1[0], ' OR ', x1[1], ' = ', 0)
        elif tmp > theta:
            print(x1[0], ' OR ', x1[1], ' = ', 1)

def nand_cal(x):
    w1, w2, theta = 0.5, 0.5, 0.5

    for x1 in x:
        tmp = x1[0] * w1 + x1[1] * w2

        if tmp > theta:
            print(x1[0], ' NAND ', x1[1], ' = ', 0)
        elif tmp <= theta:
            print(x1[0], ' NAND ', x1[1], ' = ', 1)

X_train = np.array([[0,0], [1,0], [0,1], [1,1]])

and_cal(X_train)
or_cal(X_train)
nand_cal(X_train)


