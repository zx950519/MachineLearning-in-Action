# Pandas常用函数 

# 参考源
    https://morvanzhou.github.io/tutorials/data-manipulation/np-pd/3-1-pd-intro/
<br>

## 生成数据集
    //Series基于List
    s = pd.Series([1,3,6,np.nan,44,1])
    //DataFrame基于Dict
    dates = pd.DataFramedate_range('20160101',periods=6)
    df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
    //特殊方式
    df = pd.DataFrame({"A":1., "B": pd.Timestamp("20180417"), "C":pd.Series(1, index=list(range(4)), dtype='float32'),
                        "D":np.array([3] * 4, dtype="int32"), "E": pd.Categorical(["test", "train", "test", "train"]),
                        "F":"foo"})
                        
## 查看数据集属性
    print(df.dtypes)
    
## 查看列的序号
    print(df.index)
    
## 查看数据的名称
    print(df.columns)
    
## 查看所有值
    print(df.values)
    
## 查看数据的总结
    print(df.describe())
## 转置
    df.T
## 根据数据的index排序
    df.sort_index(axis=1, ascending=False))
## 根据数据的值排序
    df.sort_index(by="B")
## 数据筛选
    数据集df为:
                 A   B   C   D
    2016-01-31   0   1   2   3
    2016-02-01   4   5   6   7
    2016-02-02   8   9  10  11
    2016-02-03  12  13  14  15
    2016-02-04  16  17  18  19
    
    选取B列：df["B"]或df.B
    选取第n行:df[n-1:n]
    选取多列：df[0:3]或df["2016-02-01":"2016-02-04"]
    选取多行：df[0:n]
    根据标签：df.loc["2016-02-01"]或df2.loc[:, ["A", "B"]]或df.loc['2016-02-01',['A','B']]
    根据序列：df.iloc[3:5, 1:3]或df.ix[0:3, ["A","C"]]
    通过判断筛选：df[df.A>8]
## 数据设置
    数据集df为:
                 A   B   C   D
    2016-01-31   0   1   2   3
    2016-02-01   4   5   6   7
    2016-02-02   8   9  10  11
    2016-02-03  12  13  14  15
    2016-02-04  16  17  18  19
    利用索引或标签：df.iloc[2,2] = 1111或df.loc['20130101','B'] = 2222
    根据条件设置：df.B[df.A>4] = 0
    按行或按列设置df["F"] = np.nan
    添加数据：df['E'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130101',periods=6)) 
## 处理丢失数据
    丢弃nan所在的行列:df.dropna(axis=0/1, how="any"/"all")
        axis:0处理行,1处理列
        how:any只要存在就丢弃,all必须全部是nan才会丢弃
    将nan替换为其他值,例如0:df.fillna(0)
    判断是否有缺失数据：df.isnull()
    检查数据中是否存在nan：np.any(df.isnull()) == True  
## 数据合并
    concat合并：
                df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
                df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
                df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
                res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
                结果为:
                    #     a    b    c    d
                    # 0  0.0  0.0  0.0  0.0
                    # 1  0.0  0.0  0.0  0.0
                    # 2  0.0  0.0  0.0  0.0
                    # 0  1.0  1.0  1.0  1.0
                    # 1  1.0  1.0  1.0  1.0
                    # 2  1.0  1.0  1.0  1.0
                    # 0  2.0  2.0  2.0  2.0
                    # 1  2.0  2.0  2.0  2.0
                    # 2  2.0  2.0  2.0  2.0
    join合并：  
                df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
                df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])
                res = pd.concat([df1, df2], axis=0, join='outer')
                结果为:
                    #     a    b    c    d    e
                    # 1  0.0  0.0  0.0  0.0  NaN
                    # 2  0.0  0.0  0.0  0.0  NaN
                    # 3  0.0  0.0  0.0  0.0  NaN
                    # 2  NaN  1.0  1.0  1.0  1.0
                    # 3  NaN  1.0  1.0  1.0  1.0
                    # 4  NaN  1.0  1.0  1.0  1.0
    concat+inner:
                res = pd.concat([df1, df2], axis=0, join='inner'，ignore_index=True))
                结果为：
                    #     b    c    d
                    # 0  0.0  0.0  0.0
                    # 1  0.0  0.0  0.0
                    # 2  0.0  0.0  0.0
                    # 3  1.0  1.0  1.0
                    # 4  1.0  1.0  1.0
                    # 5  1.0  1.0  1.0
    concat+axes:
                df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
                df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])
                res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])
                结果为:
                    #     a    b    c    d    b    c    d    e
                    # 1  0.0  0.0  0.0  0.0  NaN  NaN  NaN  NaN
                    # 2  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
                    # 3  0.0  0.0  0.0  0.0  1.0  1.0  1.0  1.0
    append:
                df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
                df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
                df3 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
                s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
                
                #将df2合并到df1的下面，以及重置index，并打印出结果
                res = df1.append(df2, ignore_index=True)
                结果为:
                    #     a    b    c    d
                    # 0  0.0  0.0  0.0  0.0
                    # 1  0.0  0.0  0.0  0.0
                    # 2  0.0  0.0  0.0  0.0
                    # 3  1.0  1.0  1.0  1.0
                    # 4  1.0  1.0  1.0  1.0
                    # 5  1.0  1.0  1.0  1.0
                #合并多个df，将df2与df3合并至df1的下面，以及重置index，并打印出结果
                res = df1.append([df2, df3], ignore_index=True)
                结果为:
                    #     a    b    c    d
                    # 0  0.0  0.0  0.0  0.0
                    # 1  0.0  0.0  0.0  0.0
                    # 2  0.0  0.0  0.0  0.0
                    # 3  1.0  1.0  1.0  1.0
                    # 4  1.0  1.0  1.0  1.0
                    # 5  1.0  1.0  1.0  1.0
                    # 6  1.0  1.0  1.0  1.0
                    # 7  1.0  1.0  1.0  1.0
                # 8  1.0  1.0  1.0  1.0
                #合并series，将s1合并至df1，以及重置index，并打印出结果
                res = df1.append(s1, ignore_index=True)
                结果为：
                    #     a    b    c    d
                    # 0  0.0  0.0  0.0  0.0
                    # 1  0.0  0.0  0.0  0.0
                    # 2  0.0  0.0  0.0  0.0
                    # 3  1.0  2.0  3.0  4.0
        merge:
                #依据一组key合并
                left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                             'A': ['A0', 'A1', 'A2', 'A3'],
                             'B': ['B0', 'B1', 'B2', 'B3']})
                right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                              'C': ['C0', 'C1', 'C2', 'C3'],
                              'D': ['D0', 'D1', 'D2', 'D3']})
                res = pd.merge(left, right, on='key')
                结果为：
                    #    A   B   key C   D
                    # 0  A0  B0  K0  C0  D0
                    # 1  A1  B1  K1  C1  D1
                    # 2  A2  B2  K2  C2  D2
                    # 3  A3  B3  K3  C3  D3
                #依据两组key合并
                left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                      'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
                right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                       'key2': ['K0', 'K0', 'K0', 'K0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})
                res = pd.merge(left, right, on=['key1', 'key2'], how='inner')
                结果为:
                    #    A   B key1 key2   C   D
                    # 0  A0  B0   K0   K0  C0  D0
                    # 1  A2  B2   K1   K0  C1  D1
                    # 2  A2  B2   K1   K0  C2  D2
                ps:how的值共有四种--left/right/outer/inner
        indicator:
                df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
                df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
                res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
                结果为:
                    #   col1 col_left  col_right indicator_column
                    # 0   0.0        a        NaN        left_only
                    # 1   1.0        b        2.0             both
                    # 2   2.0      NaN        2.0       right_only
                    # 3   2.0      NaN        2.0       right_only
        依据index合并:
                left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])
                right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                                      'D': ['D0', 'D2', 'D3']},
                                     index=['K0', 'K2', 'K3'])
                #依据左右资料集的index进行合并，how='outer',并打印出
                res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
                结果为:
                    #      A    B    C    D
                    # K0   A0   B0   C0   D0
                    # K1   A1   B1  NaN  NaN
                    # K2   A2   B2   C2   D2
                    # K3  NaN  NaN   C3   D3
                ps:how的取值可以为--outer/inner
