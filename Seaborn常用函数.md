# Seaborn常用函数 

# 参考源
    http://seaborn.pydata.org/tutorial.html
    https://zhuanlan.zhihu.com/p/27683042
    https://segmentfault.com/a/1190000005092460
    https://blog.csdn.net/Leo00000001/article/details/70226600
<br>

## 头文件引入
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from pandas import Series, DataFrame
    from pandas.tools.plotting import andrews_curves
    from pandas.tools.plotting import parallel_coordinates
    from pandas.tools.plotting import radviz

# 数据读入
    //数据的格式为DataFrame
    iris = pd.read_csv("./iris.csv")
    //用head函数查看数据结构
    print(iris.head())
    //调用counts()函数查看花的种类
    print(iris.Species.unique())
    //调用counts()函数查看花的种类
    print(iris["Species"].value_counts())
    //按照种类分组，Species共三种类型：Iris-setosa、Iris-versicolor、Iris-virginica
    iris_groupby = iris.groupby(iris["Species"])
    print(iris_groupby.get_group("Iris-setosa"))
    print(iris_groupby.get_group("Iris-versicolor"))
    print(iris_groupby.get_group("Iris-virginica"))
    
# 绘图

## 直方图
    sns.jointplot(x="横坐标", y="纵坐标", data=数据源, size=图像大小)
    例如:# sns.jointplot(x="SepalLength", y="SepalWidth", data=iris, size=5 )
    plt.show()
