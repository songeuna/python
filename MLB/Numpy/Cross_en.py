#타겟(목표) / 7번 숫자를 의미하는 목표 데이터
t= [0,0,0,0,0,0,0,1,0,0]

#예측 y -> 모델에서 출력된 데이터
y1 = [0.1, 0, 0, 0.1, 0.05, 0.05, 0, 0.7, 0, 0]         #7일 확률이 70%
y2 = [0, 0.1, 0.7, 0.05, 0.05, 0.1, 0, 0, 0, 0]         #2일 확률이 70%

import numpy as np
def cross_entropy(t,y):
    tmp = 1e-7
    return -np.sum(t * np.log(y+tmp))

print(cross_entropy(np.array(t),np.array(y1)))
print(cross_entropy(np.array(t),np.array(y2)))
