from sklearn.preprocessing import scale, minmax_scale, StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

stdscale = StandardScaler()
minmax_scaler = MinMaxScaler()
input_data = (np.arange(5, dtype=np.float)-2).reshape(-1,1)
print(input_data)

minmax_scaler_data = minmax_scaler.fit_transform(input_data)
print('평균 : ', minmax_scaler_data.mean(axis=0))
print('표준편차 : ', minmax_scaler_data.std(axis=0))

df1 = pd.DataFrame(np.hstack([input_data, minmax_scale(input_data)]),       # minmax_scale() : 0~1값에서 표준화
                             columns=["input data", "minmax_scale"])

df2 = pd.DataFrame(np.hstack([input_data, scale(input_data)]),   # scale() : 정규분포로 표준화하겠다
                             columns=["input data", "minmax_scale"])

print(np.arange(4, dtype=np.float)-2)
#print(input_data)

print(df1)
print(df2)