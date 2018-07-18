from pandas import DataFrame, Series

data = {'지역' : ['서울', '서울', '서울', '인천', '인천'],
        '연도' : [2010,   2011,   2012,   2011,   2012],
        '가격' : [10,     15,     20,     3,      5]}

df = DataFrame(data)
df1 = DataFrame(data, columns = ['지역', '연도', '가격'],
                index = ['첫째', '둘째', '셋째', '넷째', '다섯째'])

print('df is : \n', df)
print('\ndf1 is : \n', df1)
print('\ndf1 columns is : \n', df1.columns)
print('\ndf1 values is : \n', df1.values)
print('\ndf1 index is : \n', df1.index)
print('\nindex 연도 is : \n', df1.연도)
print('\nindex 연도 is : \n', df1['연도'])

print('\n리스트 인덱싱 is : \n', df1[["연도", "지역"]])
print('\n리스트 인덱싱 type is : \n', type(df1[["연도", "지역"]]))

df2 = DataFrame(data, columns=['지역', '연도', '가격', '인구'],
                index=['첫째', '둘째', '셋째', '넷째', '다섯째'])

print('\ndf2 is : \n', df2)
df2.인구 = 100

val = Series([500, 500], index=['첫째', '넷째'])

df2.인구 = val
print('\ndf2 is : \n', df2)

# columns 삭제
del df2['인구']
print('\ndel 인구 : \n', df2)

# columns과 index 변경
df2 = df2.T
print('\ndf2 :\n', df2)
print('\ndf2 :\n', df2.T)