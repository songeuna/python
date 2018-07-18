import pandas as pd

iris = pd.read_csv('c:/python/iris.csv',
                   names=['sl', 'sw', 'pl', 'pw', 'regression'])

print(iris)
print(type(iris))

print('head(5) : \n', iris.head())      # 기본적으로 5개로 지정
print('head(25) : \n', iris.head(25))
print('tail(5) : \n', iris.tail())
print('tail(25) : \n', iris.head(25))