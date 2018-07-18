# 2진으로 표현하겠다

from sklearn.preprocessing import Binarizer
import numpy as np
binarizer = Binarizer()                         # 임계값을 주지 않음(default 값 : 0)

X = np.array([[1, -1], [-1, 0], [0, -2], [0, 2]])
'''
print("X : \n", X)
print("변환 값 : \n",binarizer.transform(X))     # 숫자를 기준으로 나눔 / 0이상은 1, 0이하는 0으로 반환
'''

binarizer1 = Binarizer(threshold=1.5)           # 임계값을 줌

print("임계값을 주지 않은 변환 : \n", binarizer.transform(X))
print("임계값을 준 변환 : \n", binarizer1.transform(X))