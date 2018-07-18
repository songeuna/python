from sklearn.preprocessing import OneHotEncoder
import numpy as np

ohe = OneHotEncoder()

X = np.array([[2], [1], [3], [10]])
print("X : ", X)

print("X' onehot result : ", ohe.fit_transform(X).toarray())

print("입력 값의 구분 갯수 : ", ohe.n_values_)
print("원소들이 어떻게 나누어졌나 : ", ohe.feature_indices_)    #입력이 벡터인 경우 각 원소를 나타내는 슬라이싱(slice) 정보
                                                                #나머지 값의 표시는 안되있지만, 내부적으로는 자리를 비워둠. 따라서 마지막 값의 갯수까지 공간
print("색인값 : ", ohe.active_features_)