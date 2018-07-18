import pandas as pd

iris2 = pd.read_table('c:/python/iris2.txt', sep = '\s+',
                      names=['s1', 'sw', 'pl', 'pw', 'regression'],
                      skiprows=[0,1])

print(iris2)

iris2.to_csv('c:/python/iris2.csv')