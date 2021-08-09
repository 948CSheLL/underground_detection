import pandas as pd

data1 = {"a":[1.,3.,5.,2.],
         "b":[4.,8.,3.,7.],
         "c":[5.,45.,67.,34]}


df1 = pd.DataFrame(data1)
print(df1)
'''
     a    b     c
0  1.0  4.0   5.0
1  3.0  8.0  45.0
2  5.0  3.0  67.0
3  2.0  7.0  34.0

'''

df2=df1.sum(axis=1)# 按行相加
print(df2)
print(type(df2))
'''
0    10.0
1    56.0
2    75.0
3    43.0
dtype: float64

'''

print(df1.div(df2, axis='rows'))#每一行除以对应的向量
print(type(df1.div(df2, axis='rows')))#每一行除以对应的向量
'''
          a         b         c
0  0.100000  0.400000  0.500000
1  0.053571  0.142857  0.803571
2  0.066667  0.040000  0.893333
3  0.046512  0.162791  0.790698
'''