import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import *

from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import load_boston
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn import metrics
from sklearn.decomposition import PCA

# Title-机器学习及实践(代码实现)
# Creator-ZhouXiang
# Environment-py3.x

# 乳腺
def test_01():
    # 创建特征列表
    column_names = ["SC", "CT", "UCS", "UOCS", "MA", "SECS", "BN", "BC", "NN", "M", "C"]
    # 获取数据
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                       names=column_names)
    # 将？替换为标准缺失值
    data = data.replace(to_replace="?", value=np.nan)
    # 丢弃带有缺失值的数据
    data = data.dropna(how="any")
    # 输出数据及维度
    # print(data.shape)
    # print(data)
    # 随机采样 25%用于测试 75%用于训练
    X_train, X_test, Y_train, Y_test = train_test_split(data[column_names[1:10]], data[column_names[10]],
                                                        test_size=0.25, random_state=33)
    # # 查看训练样本数量及其分布
    # print(Y_train.value_counts())
    # # 查看测试样本数量及其分布
    # print(Y_test.value_counts())

    # 标准化数据 保证每个维度的特征方差为1 均值为0 使得预测结果不会被某些维度过大的特征值主导
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    # 初始化逻辑回归器&随机梯度下降器
    lr = LogisticRegression()
    sgdc = SGDClassifier(max_iter=1000)
    # 调用逻辑回归中的fit模块训练模型参数
    lr.fit(X_train, Y_train)
    # 使用训练好的模型lr对X_test进行预测
    lr_y_predict = lr.predict(X_test)
    # 调用随机梯度下降器中的fit模块训练模型参数
    sgdc.fit(X_train, Y_train)
    # 使用训练好的模型sgdc对X_test进行预测
    sgdc_y_predict = sgdc.predict(X_test)

    # 使用逻辑回归模型自带的评分函数score获得模型在测试集上的准确性结果
    print("LR分类器的准确率为:", lr.score(X_test, Y_test))
    print(classification_report(Y_test, lr_y_predict, target_names=["Benign", "Malignant"]))
    # 使用随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果
    print("SGD分类器的准确率为:", sgdc.score(X_test, Y_test))
    print(classification_report(Y_test, sgdc_y_predict, target_names=["Benign", "Malignant"]))

    # 说明
    # 线性分类器可以说是最为基本和常用的机器学习模型
    # 逻辑回归模型对参数采用精确解析的方式 计算时间长但模型性能高
    # 随机梯度下降模型估计模型参数耗时小 但模型性能略低
    # 当训练规模大于10万时可考虑使用随机梯度下降

# 手写体
def test_02():
    # 加载数据
    digits = load_digits()
    # 查看维度
    # print(digits.data.shape)
    # 划分数据
    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
    # 查看数据
    # print(y_train.shape, y_test.shape)

    # 标准化数据 保证每个维度的特征方差为1 均值为0 使得预测结果不会被某些维度过大的特征值主导
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    # 初始化线性假设的支持向量机分类器LinearSVC
    lsvc = LinearSVC()
    # 训练模型
    lsvc.fit(x_train, y_train)
    # 使用训练好的模型进行预测
    y_predict = lsvc.predict(x_test)

    # 使用模型自带的评估函数进行准确性评测
    print("SVC的准确率为:", lsvc.score(x_test, y_test))
    print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))

# 新闻文本分类
def test_03():
    # 获取数据
    news = fetch_20newsgroups(subset="all")
    # 查看数据细节
    print(len(news.data))

    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
    vec = CountVectorizer()
    x_train = vec.fit_transform(x_train)
    x_test = vec.transform(x_test)

    # 初始化贝叶斯模型
    mnb = MultinomialNB()
    # 利用训练数据对模型参数进行估计
    mnb.fit(x_train, y_train)
    # 预测类别
    y_predict = mnb.predict(x_test)

    # 性能评估
    print("朴素贝叶斯的准确率:", mnb.score(x_test, y_test))
    print(classification_report(y_test, y_predict, target_names=news.target_names))

    # 说明
    # 朴素贝叶斯被广泛应用于海量互联网文本分类任务
    # 大大减少内存消耗以及计算时间
    # 但是模型训练时无法将各个特征之间的联系考虑在内
    # 在特征关联性较强的分类任务上性能不佳

# Iris花
def test_04():
    # 加载数据
    iris = load_iris()
    # 查看数据维度
    # print(iris.data.shape)
    # 查看数据说明
    # print(iris.DESCR)

    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
    # 初始化K近邻分类器
    knc = KNeighborsClassifier()
    knc.fit(x_train, y_train)
    # 预测类别
    y_predict = knc.predict(x_test)

    # 性能评估
    print("K近邻的准确率为:", knc.score(x_test, y_test))
    print(classification_report(y_test, y_predict, target_names=iris.target_names))

    # 说明
    # K近邻并无参数训练过程 属于无参数模型的一种
    # 该算法消耗内存大

# 泰坦尼克
def test_05():
    # 加载数据
    titanic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    # 观察数据
    # print(titanic.head())
    # 查看数据统计特性
    # print(titanic.info())

    # 数据补完
    x = titanic[["pclass", "age", "sex"]]
    y = titanic["survived"]
    x["age"].fillna(x["age"].mean(), inplace=True)
    # x.info()

    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
    # 特征抽取
    vec = DictVectorizer(sparse=True)
    # 凡是object类型的特征都单独剥离出来 单独形成一列特征 数值型保持不变
    x_train = vec.fit_transform(x_train.to_dict(orient="record"))
    x_test = vec.fit_transform(x_test.to_dict(orient="record"))
    print(vec.feature_names_)

    # 初始化决策树分类器
    dtc = DecisionTreeClassifier()
    # 模型学习
    dtc.fit(x_train, y_train)
    # 利用模型来预测
    y_predict = dtc.predict(x_test)

    # 输出预测准确性
    print("决策树的准确性为:", dtc.score(x_test, y_test))
    print(classification_report(y_predict, y_test, target_names=["died", "survived"]))

# 泰坦尼克-集成模型
def test_06():
    # 加载数据
    titantic = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    # 人工选取特征
    x = titantic[["pclass", "age", "sex"]]
    y = titantic["survived"]
    # 补全缺失信息
    x["age"].fillna(x["age"].mean(), inplace=True)

    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

    # 转化类别 生成特征向量
    vec = DictVectorizer(sparse=True)
    x_train = vec.fit_transform(x_train.to_dict(orient="record"))
    x_test = vec.transform(x_test.to_dict(orient="record"))

    # 使用单一决策树训练&分析
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    dtc_y_pred = dtc.predict(x_test)

    # 使用随机森林分类器训练&分析
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    rfc_y_pred = rfc.predict(x_test)

    # 使用随机梯度决策树训练&分析
    gbc = GradientBoostingClassifier()
    gbc.fit(x_train, y_train)
    gbc_y_pred = gbc.predict(x_test)

    # 输出结果
    print("单一决策树的性能:", dtc.score(x_test, y_test))
    print(classification_report(dtc_y_pred, y_test))

    print("随机森林的性能:", rfc.score(x_test, y_test))
    print(classification_report(rfc_y_pred, y_test))

    print("梯度提升决策树的性能:", gbc.score(x_test, y_test))
    print(classification_report(gbc_y_pred, y_test))

# 波士顿房价
def test_07():
    # 加载数据
    boston = load_boston()
    # 数据描述
    # print(boston.DESCR)

    x = boston.data
    y = boston.target

    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
    # 分析回归目标值差异
    # print("最大目标值:", np.max(boston.target))
    # print("最小目标值:", np.min(boston.target))
    # print("平均目标值:", np.mean(boston.target))

    # 特征&目标值标准化器的初始化
    ss_x = StandardScaler()
    ss_y = StandardScaler()

    # 标准化处理数据
    x_train = ss_x.fit_transform(x_train)
    x_test = ss_x.transform(x_test)
    y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
    y_test = ss_y.transform(y_test.reshape(-1, 1))

    # 初始化线性回归器
    lr = LinearRegression()
    # 参数估计
    lr.fit(x_train, y_train)
    # 回归预测
    lr_y_pred = lr.predict(x_test)

    # 初始化SGD回归器
    sgdr = SGDRegressor(max_iter=1000)
    # 参数估计
    sgdr.fit(x_train, y_train.ravel())
    # 回归预测
    sgdr_y_pred = sgdr.predict(x_test)

    # 输出线性分类器的结果
    print("线性回归性能:", lr.score(x_test, y_test))
    print("r2_score性能:", r2_score(y_test, lr_y_pred))
    print("均方性能:", mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_pred)))
    print("绝对均值性能:", mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_pred)))
    print()

    print("SGD回归性能:", sgdr.score(x_test, y_test))
    print("r2_score性能:", r2_score(y_test, sgdr_y_pred))
    print("均方性能:", mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_pred)))
    print("绝对均值性能:", mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_pred)))
    print()

    # 使用线性核函数训练&预测
    linear_svr = SVR(kernel="linear")
    linear_svr.fit(x_train, y_train.ravel())
    linear_svr_y_pred = linear_svr.predict(x_test)
    # 使用多项式核函数训练&预测
    poly_svr = SVR(kernel="poly")
    poly_svr.fit(x_train, y_train.ravel())
    poly_svr_y_pred = poly_svr.predict(x_test)
    # 使用线性核函数训练&预测
    rbf_svr = SVR(kernel="rbf")
    rbf_svr.fit(x_train, y_train.ravel())
    rbf_svr_y_pred = rbf_svr.predict(x_test)

    # 输出支持向量机回归的结果
    print("线性核函数性能:", linear_svr.score(x_test, y_test))
    print("r2_score性能:", r2_score(y_test, linear_svr_y_pred))
    print("均方性能:", mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_pred)))
    print("绝对均值性能:", mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_pred)))
    print()

    print("多项式核函数性能:", poly_svr.score(x_test, y_test))
    print("r2_score性能:", r2_score(y_test, poly_svr_y_pred))
    print("均方性能:", mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_pred)))
    print("绝对均值性能:", mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_pred)))
    print()

    print("径向核函数性能:", rbf_svr.score(x_test, y_test))
    print("r2_score性能:", r2_score(y_test, rbf_svr_y_pred))
    print("均方性能:", mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_pred)))
    print("绝对均值性能:", mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_pred)))
    print()

    # 初始化K近邻 根据距离平均回归 训练&预测
    uni_knr = KNeighborsRegressor(weights="uniform")
    uni_knr.fit(x_train, y_train)
    uni_knr_y_pred = uni_knr.predict(x_test)
    # 初始化K近邻 根据距离加权回归 训练&预测
    dis_knr = KNeighborsRegressor(weights="distance")
    dis_knr.fit(x_train, y_train)
    dis_knr_y_pred = dis_knr.predict(x_test)

    # 输出K近邻回归的结果
    print("平均K近邻性能:", uni_knr.score(x_test, y_test))
    print("r2_score性能:", r2_score(y_test, uni_knr_y_pred))
    print("均方性能:", mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_pred)))
    print("绝对均值性能:", mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_pred)))
    print()

    # 使用默认配置初始化决策树回归器
    dtr = DecisionTreeRegressor()
    dtr.fit(x_train, y_train)
    dtr_y_pred = dtr.predict(x_test)

    # 输出单一决策树回归的结果
    print("单一决策树回归性能:", dtr.score(x_test, y_test))
    print("r2_score性能:", r2_score(y_test, dtr_y_pred))
    print("均方性能:", mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_pred)))
    print("绝对均值性能:", mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_pred)))
    print()

    # 使用随机森林回归训练模型&预测
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)
    rfr_y_pred = rfr.predict(x_test)
    # 使用极端随机森林回归训练模型&预测
    etr = RandomForestRegressor()
    etr.fit(x_train, y_train.ravel())
    etr_y_pred = etr.predict(x_test)
    # 使用GradientBoostingRegerssor()训练模型&预测
    gbr = RandomForestRegressor()
    gbr.fit(x_train, y_train.ravel())
    gbr_y_pred = gbr.predict(x_test)

    # 输出集成决策树回归的结果
    print("随机森林回归性能:", rfr.score(x_test, y_test))
    print("r2_score性能:", r2_score(y_test, rfr_y_pred))
    print("均方性能:", mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_pred)))
    print("绝对均值性能:", mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_pred)))
    print()

    print("极端随机森林回归性能:", etr.score(x_test, y_test))
    print("r2_score性能:", r2_score(y_test, etr_y_pred))
    print("均方性能:", mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_pred)))
    print("绝对均值性能:", mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_pred)))
    print()

    print("使用GradientBoostingRegerssor()回归性能:", gbr.score(x_test, y_test))
    print("r2_score性能:", r2_score(y_test, gbr_y_pred))
    print("均方性能:", mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_pred)))
    print("绝对均值性能:", mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_pred)))
    print()

# K均值手写体
def test_08():
    # 加载数据
    digit_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)
    digit_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes", header=None)
    # 分离出64维度的像素特征与1维度的数字目标
    x_train = digit_train[np.arange(64)]
    y_train = digit_train[64]

    x_test = digit_test[np.arange(64)]
    y_test = digit_test[64]

    # 初始化K均值模型
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(x_train)
    # 逐条判断每个测试图像所属的聚类中心
    y_pred = kmeans.predict(x_test)

    # 使用ARI进行K均值聚类
    print(metrics.adjusted_rand_score(y_test, y_pred))

def test_09():
    # 分割子图
    plt.subplot(3, 2, 1)
    # 初始化数据点
    x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
    x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
    # x = np.array(zip(x1, x2)).reshape(len(x1), 2)
    x = np.array([[1, 1],[2, 3],[3,2],[1,2],[5,8],[6,6],[5,7],[5,6],[6,7],[7,1],[8,2],[9,1],[7,1],[9,3]])

    print(np.array(zip(x1, x2)))

    # 1号子图做出原始数据点阵的分布
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.title("Instances")
    plt.scatter(x1, x2)

    colors = ["b", "g", "r", "c", "m", "y", "k", "b"]
    markers = ["o", "s", "D", "v", "^", "p", "*", "+"]

    clusters = [2, 3, 4, 5, 8]
    subplot_counter = 1
    sc_scores = []
    for t in clusters:
        subplot_counter += 1
        plt.subplot(3, 2, subplot_counter)
        kmeans_model = KMeans(n_clusters=t).fit(x)

        for i, l in enumerate(kmeans_model.labels_):
            plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls="None")
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        sc_score = silhouette_score(x, kmeans_model.labels_, metric="euclidean")
        sc_scores.append(sc_score)

        plt.title("k=%s, silhouette coefficient=%0.003f" % (t, sc_score))

    plt.figure(figsize=(8, 8), dpi=80)
    plt.plot(clusters, sc_scores, "*-")
    plt.xlabel("Number of Cluster")
    plt.ylabel("Silhouette Coefficient Score")
    plt.show()

# 肘部观察法
def test_10():
    # 使用均匀分布函数随机三个簇 每个簇周围10个数据样本
    cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
    cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
    cluster3 = np.random.uniform(3.0, 4.0, (2, 10))
    # 绘制30个数据样本的分布图像
    x = np.hstack((cluster1, cluster2, cluster3)).T
    plt.scatter(x[:, 0], x[:, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    # 测试9种不同聚类中心数量下的聚类质量
    K = range(1, 10)
    meandistortions = []

    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        meandistortions.append(sum(np.min(cdist(x, kmeans.cluster_centers_, "euclidean"), axis=1))/x.shape[0])
    plt.plot(K, meandistortions, "bx-")
    plt.xlabel("k")
    plt.ylabel("Average Dispersion")
    plt.title("Selecting k with the Elbow Method")
    plt.show()

# 手写体利用PCA压缩
def test_11():
    # 加载数据
    digit_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
                              header=None)
    digit_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",
                             header=None)
    # 分割训练数据的特征向量&标记
    x_digits = digit_train[np.arange(64)]
    y_digits = digit_train[64]

    # 初始化一个可以将高维度特征向量压缩至2维的PCA
    estimator = PCA(n_components=2)
    x_pac = estimator.fit_transform(x_digits)

    # 绘图
    colors = ["black", "blue", "purple", "yellow", "white", "red", "lime", "cyan", "orange", "gray"]
    for i in range(len(colors)):
        px = x_pac[:, 0][y_digits.as_matrix() == i]
        py = x_pac[:, 1][y_digits.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()

    # # 对比
    # x_train = digit_train[np.arange(64)]
    # y_train = digit_train[64]
    # x_test = digit_test[np.array(64)]
    # y_test = digit_test[64]
    #
    # # 初始化线性SVC 对原始的64维像素特征的训练数据进行建模
    # svc = LinearSVC()
    # svc.fit(x_train, y_train)
    # y_predict = svc.predict(x_test)
    #
    # # 利用PCA将数据压缩到20个正交的维度上
    # estimator = PCA(n_components=20)
    #
    # # 利用训练特征决定20个正交维度的方向 并转化原训练特征
    # pca_x_train = estimator.fit_transform(x_train)
    # pca_x_test = estimator.transform(x_test)
    #
    # # 使用默认配置初始化线性SVC 对压缩的20维特征的训练数据进行建模
    # pca_svc = LinearSVC()
    # pca_svc.fit(pca_x_train, y_train)
    # pca_y_predict = pca_svc.predict(pca_x_test)
    #
    # # 原始维度评估
    # print(svc.score(x_test, y_test))
    # print(classification_report(y_test, y_predict, target_names=np.arange(10).astype(str)))
    #
    # # 降维评估
    # print(pca_svc.score(pca_x_test, y_test))
    # print(classification_report(y_test, pca_y_predict, target_names=np.arange(10).astype(str)))

# 对字典存储的数据进行特征抽取与向量化
def test_12():
    # 构造数据
    measurements = [{"city": "Dubai", "temperature": 33}, {"city": "London", "temperature": 12},
                    {"city": "Beijing", "temperature": 18}]
    # 初始化特征抽取其
    vec = DictVectorizer()
    # 输出特征矩阵
    print(vec.fit_transform(measurements).toarray())
    # 输出各个维度的特征含义
    print(vec.get_feature_names())

# 特征抽取对比
def test_13():
    # 数据下载
    news = fetch_20newsgroups(subset="all")
    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

    # 初始化CountVectorizer
    count_vec = CountVectorizer()
    # 转化为特征向量
    x_count_train = count_vec.fit_transform(x_train)
    x_count_test = count_vec.transform(x_test)
    # 初始化朴素贝叶斯分类器
    mnb_count = MultinomialNB()
    # 不去掉停用词学习参数
    mnb_count.fit(x_count_train, y_train)
    # 输出结果
    print(mnb_count.score(x_count_test, y_test))
    y_count_predict = mnb_count.predict(x_count_test)
    print(classification_report(y_test, y_count_predict, target_names=news.target_names))

    # # 初始化TfidfVectorizer 出现了爆内存的情况 暂时没找到有用解
    # tfidf_vec = TfidfTransformer()
    # # 转化为特征向量
    # x_tfidf_train = tfidf_vec.fit_transform(x_train)
    # x_tfidf_test = tfidf_vec.transform(x_test)
    # # 初始化朴素贝叶斯分类器
    # mnb_tfidf = MultinomialNB()
    # # 不去掉停用词学习参数
    # mnb_tfidf.fit(x_tfidf_train, x_tfidf_test)
    # # 输出结果
    # print(mnb_tfidf.score(x_count_test, y_test))
    # y_tfidf_predict = mnb_count.predict(x_count_test)
    # print(classification_report(y_test, y_tfidf_predict, target_names=news.target_names))

    # 初始化CountVectorizer&TfidfVectorizer 并去掉停用词
    count_filter_vec, tfidf_filter_vec = CountVectorizer(analyzer="word", stop_words="english"), \
                                         TfidfVectorizer(analyzer="word", stop_words="english")

    x_count_filter_train = count_filter_vec.fit_transform(x_train)
    x_count_filter_test = count_filter_vec.transform(x_test)
    x_tfidf_filter_train = tfidf_filter_vec.fit_transform(x_train)
    x_tfidf_filter_test = tfidf_filter_vec.transform(x_test)

    # 初始化朴素贝叶斯分类器
    mnb_count_filter = MultinomialNB()
    mnb_count_filter.fit(x_count_filter_train, y_train)
    # 性能评估
    print(mnb_count_filter.score(x_count_filter_test, y_test))
    y_count_filter_predict = mnb_count_filter.predict(x_count_filter_test)

    # 初始化朴素贝叶斯分类器
    mnb_tfidf_filter = MultinomialNB()
    mnb_tfidf_filter.fit(x_tfidf_filter_train, y_train)
    # 性能评估
    print(mnb_tfidf_filter.score(x_count_filter_test, y_test))
    y_tfidf_filter_predict = mnb_tfidf_filter.predict(x_tfidf_filter_test)

    print(classification_report(y_test, y_count_filter_predict, target_names=news.target_names))
    print(classification_report(y_test, y_tfidf_filter_predict, target_names=news.target_names))

# 拟合-披萨
def test_14():
    # 构造数据
    x_train = [[6], [8], [10], [14], [18]]
    y_train = [[7], [9], [13], [17.5], [18]]
    # 初始化线性回归模型
    regressor = LinearRegression()
    # 直接用直径作为特征训练模型
    regressor.fit(x_train, y_train)
    # 预测
    xx = np.linspace(0, 26, 100)
    xx = xx.reshape(xx.shape[0], 1)
    yy = regressor.predict(xx)
    # 绘图
    plt.scatter(x_train, y_train)
    plt1, = plt.plot(xx, yy, label="Degree=1")
    plt.axis([0, 25, 0, 25])
    plt.xlabel("Diameter of pizza")
    plt.ylabel("Price of Pizza")
    plt.legend(handles=[plt1])
    plt.show()
    # 打印结果
    print(regressor.score(x_train, y_train))

    # 使用2次多项式回归模型----------------------

    # 初始化
    poly2 = PolynomialFeatures(degree=2)
    # 映射出2次多项式特征
    x_train_poly2 = poly2.fit_transform(x_train)
    # 初始化线性回归器
    regressor_poly2 = LinearRegression()
    # 对2次多项式回归模型进行训练
    regressor_poly2.fit(x_train_poly2, y_train)
    # 新映射绘图用X轴采样数据
    xx_poly2 = poly2.transform(xx)
    # 使用2次多项式进行回归预测
    yy_poly2 = regressor_poly2.predict(xx_poly2)
    # 绘图
    plt.scatter(x_train, y_train)
    plt1, = plt.plot(xx, yy, label="Degree=1")
    plt2, = plt.plot(xx, yy_poly2, label="Degree=2")
    plt.axis([0, 25, 0, 25])
    plt.xlabel("Diameter of pizza")
    plt.ylabel("Price of Pizza")
    plt.legend(handles=[plt1, plt2])
    plt.show()
    # 打印结果
    print(regressor_poly2.score(x_train_poly2, y_train))

    # 使用4次多项式回归模型----------------------
    # 初始化4次多项式特征生成器
    poly4 = PolynomialFeatures(degree=4)
    x_train_poly4 = poly4.fit_transform(x_train)
    # 使用默认配置初始化4次多项式回归器
    regressor_poly4 = LinearRegression()
    # 对4次多项式回归模型进行训练
    regressor_poly4.fit(x_train_poly4, y_train)
    # 新映射绘图用X轴采样数据
    xx_poly4 = poly4.transform(xx)
    # 使用2次多项式进行回归预测
    yy_poly4 = regressor_poly4.predict(xx_poly4)
    # 绘图
    plt.scatter(x_train, y_train)
    plt3, = plt.plot(xx, yy, label="Degree=1")
    plt4, = plt.plot(xx, yy_poly4, label="Degree=2")
    plt.axis([0, 25, 0, 25])
    plt.xlabel("Diameter of pizza")
    plt.ylabel("Price of Pizza")
    plt.legend(handles=[plt1, plt2, plt3, plt4])
    plt.show()
    # 打印结果
    print(regressor_poly4.score(x_train_poly4, y_train))

if __name__ == "__main__":
    test_14()