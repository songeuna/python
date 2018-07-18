from pandas import DataFrame, Series
'''
s = Series([1, 2, 3, None, 5])                  # 1차원

# 데이터의 갯수 카운팅
print('Series Count :', s.count())

data = {'연도' : [2010, 2011, 2012, 2011, 2012],
        '가격' : [10, 15, 20, None, 5]}

s1 = DataFrame(data)
print('\ns :\n', s)

print('\ns1 : \n', s1)
print(s1.count())
'''
data1 = {'연도' : [2010, 2011, 2012, 2011, 2012],
        '가격' : [10, 15, 20, 30, 5]}

s2 = DataFrame(data1)

s3 = s2.T
print('\ns2 :\n', s2)
print('\ns3 :\n', s3)
print('\nsum :\n', s2.sum())
print('\nsum1 :\n', s3.sum())
print('\nsum2 :\n', s3.sum(axis=1))