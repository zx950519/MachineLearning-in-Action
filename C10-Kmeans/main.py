from numpy import *
import time
import matplotlib.pyplot as plt

# calculate Euclidean distance
# 计算二维距离
def euclDistance(vector1, vector2):
    # vector1与vector2类型为：ndarray
    # 每一个代表了一个实体点的横纵坐标
    # 返回两个点的二维距离
    return sqrt(sum(power(vector2 - vector1, 2)))


# init centroids with random samples
# 效果同函数randCent，查找初始质心
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    # numSamples值为80，代表有80行数据
    numSamples = dataSet.shape[0]
    # 构造一个80行2列的矩阵，值全为0
    # 第一列记录簇索引值，第二列存储误差(指当前点到簇质心的距离)
    clusterAssment = mat(zeros((numSamples, 2)))
    # 设定循环变量
    clusterChanged = True

    # 获取初始数据的质心，共计4个
    centroids = initCentroids(dataSet, k)
    # 统计迭代次数
    itor  = 1
    # 开始迭代
    while clusterChanged:
        # print("第"+str(itor)+"次迭代")
        # 迭代次数自增
        itor = itor + 1
        # 修改循环条件
        clusterChanged = False
        # 对数据集中的每个点
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            # 找到该点距离哪个质心最近
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 判断是否需要更新簇信息
            # 判断第i行0列的值是否与最近簇的下标一致
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
                # print(clusterAssment)
        ## step 4: update centroids
        # 更新簇中心
        for j in range(k):
            # clusterAssment[:, 0].A将矩阵第1列转化为数组
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            # 显示簇划分
            # print("第"+str(j)+"簇划分为:")
            # print(nonzero(clusterAssment[:, 0].A == j)[0])
            # axis不设置值，对m * n个数求均值，返回一个实数
            # axis = 0：压缩行，对各列求均值，返回1 * n矩阵
            # axis = 1 ：压缩列，对各行求均值，返回m * 1矩阵
            centroids[j, :] = mean(pointsInCluster, axis=0)
            # print("更新后的簇质心为:")
            # print(centroids[j, :])

    print("Congratulations, cluster complete!")
    return centroids, clusterAssment

# 计算向量的欧式距离
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB, 2)))

# 二分法-kmeans聚类
def bikMeans(dataSet, k, disMeas = distEclud):
    # 获取行数
    m = shape(dataSet)[0]
    # 构造一个80行2列的矩阵，值全为0
    # 第一列记录簇索引值，第二列存储误差(指当前点到簇质心的距离)
    clusterAssment = mat(zeros((m, 2)))
    # 初始簇
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    # print(centList)
    # print(len(centList))
    for j in range(m):
        # 计算所有点的均值，axis=0表示沿矩阵的列方向进行均值计算
        clusterAssment[j, 1] = disMeas(mat(centroid0), dataSet[j, :]) ** 2
    # 开始迭代过程
    while (len(centList) < k):
        lowestSSE = inf
        # 对每个质心
        for i in range(len(centList)):
            # 尝试划分每一簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            print("簇"+str(i)+"：内点数：")
            print("\t"+str(len(ptsInCurrCluster)))
            # 对簇内的点采用2分聚类
            centroidMat, splitClustAss = kmeans(ptsInCurrCluster, 2)
            # 所有参与分配的点到质心距离求和
            sseSplit = sum(splitClustAss[:, 1])
            # 所有未参与分配的点到质心距离求和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit and notSplit:", sseSplit, sseNotSplit)
            # 如果分割后的总误差更优，则更新
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 更新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit

        print("the bestCentToSplit is :", bestCentToSplit)
        print("the len of bestClustAss is:", len(bestClustAss))
        # 更新质心列表
        centList[bestCentToSplit] = bestNewCents[0, :]
        # 添加第二个质心
        centList.append(bestNewCents[1, :])
        # 更新簇的质心以及sse值
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment


# show your cluster only available with 2-D data
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large! please contact Zouxy")
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()

# 加载数据
def loadDataSet(path):
    print("step 1: loading data...")
    dataSet = []
    fileIn = open(path)
    for line in fileIn.readlines():
        # 处理每行的数据
        lineArr = line.strip().split('\t')
        line_Add = lineArr[0].split("   ")
        # print(line_Add[0]+"\t"+line_Add[1])
        # print(lineArr[0]+"\t"+lineArr[1])
        # 添加数据
        dataSet.append([float(line_Add[0]), float(line_Add[1])])
    # 返回结果集
    return dataSet

def randCent(dataSet, k):
    # 在本例中n=2,代表有两列
    n = shape(dataSet)[1]
    # 构建一个k行n列的矩阵，值全为0
    centroids = mat(zeros((k, n)))
    # 对每列，即每个维度
    for j in range(n):
        # 获取每个维度的最小值
        minJ = min(dataSet[:, j])
        # 每个维度最值的区间长度
        rangeJ = float(max(dataSet[:, j]) - minJ)
        # 获取一个位于区间中间的数
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def test_by_basic():
    dataSet = mat(loadDataSet("./testSet.txt"))
    # [:, n]推定为第n列的所有数据
    # 分别打印第一维(x轴)、第二维(y轴)最值
    # print(min(dataSet[:, 0]))
    # print(max(dataSet[:, 0]))
    # print(min(dataSet[:, 1]))
    # print(max(dataSet[:, 1]))

    # 构建簇质心
    # print("簇质心为")
    # print(randCent(dataSet, 4))
    print("step 2: clustered...")
    k = 4
    centroids, clusterAssment = kmeans(dataSet, k)

    # step 3: show the result
    print("step 3: show the result...")
    showCluster(dataSet, k, centroids, clusterAssment)

def test_by_binary():
    dataSet = mat(loadDataSet("./testSet.txt"))
    # [:, n]推定为第n列的所有数据
    # 分别打印第一维(x轴)、第二维(y轴)最值
    # print(min(dataSet[:, 0]))
    # print(max(dataSet[:, 0]))
    # print(min(dataSet[:, 1]))
    # print(max(dataSet[:, 1]))

    # 构建簇质心
    # print("簇质心为")
    # print(randCent(dataSet, 4))

    print("step 2: clustered...")
    k = 4
    # centroids, clusterAssment = kmeans(dataSet, k)
    centroids, clusterAssment = bikMeans(dataSet, k)

    # step 3: show the result
    print("step 3: show the result...")
    showCluster(dataSet, k, centroids, clusterAssment)

if __name__ == '__main__':
    # 普通聚类
    test_by_basic()
    # 二分聚类
    test_by_binary()
