from numpy import *
import numpy as np
import math

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open("testSet.txt")


# 加载数据
def loadDataSet():
    # 数据列表
    dataMat = []
    # 标签列表
    labelMat = []
    # 读取文件
    fr = open("./testSet.txt")

    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat



def sigmoid(intX):
    return 1.0/(1 + math.exp(-intX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() *  error
    return weights

def plotBastFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)

# sigmoid函数
def sigmoid(intX):
    return 1.0/(1 + exp(-intX))

# 梯度上升法更新最优拟合参数
# 参数dataMatIn:二维Numpy数组，每列代表不同的特征，每行代表一个训练样本
def gradAscent(dataMatIn, classLabels):
    # 转化为矩阵
    dataMatrix = mat(dataMatIn)
    # print(dataMatrix)
    # 额外转置
    labelMat = mat(classLabels).transpose()
    # 行数 列数
    m, n = shape(dataMatrix)
    # 学习步长
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    # 初始化权值参数向量 均设置为1.0
    weights = ones((n, 1))
    print("初始权值", weights)
    # 开始迭代
    for k in range(maxCycles):
        # 当前预测概率
        h = sigmoid(dataMatrix * weights)
        # 计算真实类别和预测类别的差值
        error = (labelMat - h)
        # 更新权重参数
        # 梯度上升迭代公式：w = w + a * Dwf(x)
        # 梯度下降迭代公式：w = w - a * Dwf(x)
        weights = weights + alpha * dataMatrix.transpose() * error
        # print(k ,"-目前权值", weights)

    return weights
# 绘图函数
def plotBastFit(wei):
    import matplotlib.pyplot as plt
    tmp = mat(wei)
    weights = tmp.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c="red", marker="s")
    ax.scatter(xcord2, ycord2, s=30, c="green")
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
# 重构数组
def recreate(weights):
    # 重构数组 使其满足绘图函数的参数形式
    jk = np.array([[weights[0]], [weights[1]], [weights[2]]])
    return jk
# 梯度上升法
def stocGradAscent0(dataMatrix, classLabels):
    # 获取行数 列数
    m, n = shape(dataMatrix)
    # 学习步长
    alpha = 0.01
    # 权重全设为1
    weights = np.ones(n)
    for i in range(m):
        # 计算当前sigmoid函数值
        h = sigmoid(sum(dataMatrix[i] * weights))
        # 计算当前残差
        error = classLabels[i] - h
        # 更新权值参数
        weights = weights + alpha * error * dataMatrix[i]

    return weights
# 修改版上升梯度
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    # 数组转换
    dataMat = array(dataMatrix)
    # 获取行数 列数
    m, n = shape(dataMat)
    # 权重全设为1
    weights = ones(n)
    for j in range(numIter):
        # 获取数据集行下标列表
        dataIndex = list(range(m))
        for i in range(m):
            # 动态设置步长
            alpha = 4/(1.0 + j + i) + 0.01
            # 获取随机样本
            randIndex = int(random.uniform(0, len(dataIndex)))
            # 计算当前sigmoid函数
            h = sigmoid(dataMat[randIndex] * weights)
            error = classLabels[randIndex] - h
            # 更新权重
            weights = weights + alpha * error * dataMat[randIndex]
            # 删除当前样本 保证只用一次
            del(dataIndex[randIndex])
    return weights

def test():

    dataArr, labelMat = loadDataSet()

    # weights = gradAscent(dataArr, labelMat)
    # plotBastFit(weights.getA())

    # weights = stocGradAscent0(array(dataArr), labelMat)
    # weights = recreate(weights)
    # plotBastFit(weights)

    print(type(array(dataArr)))
    print(array(dataArr))
    weights = stocGradAscent1(array(dataArr), labelMat)
    weights = recreate(weights)
    plotBastFit(weights)

# 分类决策函数
def classifyVector(intX, weights):
    prob = sigmoid(intX*weights)
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    # 打开文件
    frTrain = open("C:/Users/Alitria/Downloads/MLiA_SourceCode/machinelearninginaction/Ch05/horseColicTraining.txt")
    frTest = open("C:/Users/Alitria/Downloads/MLiA_SourceCode/machinelearninginaction/Ch05/horseColicTest.txt")
    # 训练集 测试集
    trainingSet = []
    trainingLabels = []
    # 读取训练集的每一行
    for line in frTrain.readlines():
        # 对当前特征进行分割
        currLine = line.strip().split()
        lineArr = []
        # 遍历每个特征
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 将样本标签存入标签列表
        trainingLabels.append(currLine[21])
        # 将该样本的特征向量添加到数据集
        trainingSet.append(lineArr)
    print(type(array(trainingSet)))
    print(array(trainingSet))
    # 更新logistic回归的权值参数
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    # 统计测试数据集预测样本数量和样本总数
    errorCount = 0;numTestVec = 0.0
    # 遍历测试数据集的每个样本
    for line in frTest.readlines():
        # 样本总数+1
        numTestVec += 1.0
        # 分割出各个特征及样本标签
        currLine = line.strip().split()
        # 新建特征向量
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 预测样本并与标签样本进行比较
        if(classifyVector(lineArr, trainWeights)!=currLine[21]):
            errorCount += 1
    # 计算预测错误率
    errorRate = (float(errorCount)/numTestVec)
    print("错误率为：", errorCount)
    return errorRate
# 多次测试算法求误差预测平均值
def multTest():
    numTests = 10
    errorRateSum = 0.0
    for k in range(numTests):
        errorRateSum += colicTest()
    print("平均错误率为:",errorRateSum/float(numTests))

if __name__ == '__main__':
    # 基本
    test()
    # 马的实现暂时存在未解决的bug
    # TypeError: ufunc 'subtract' did not contain a loop with signature matching types dtype('<U32') dtype('<U32') dtype('<U32')
    # 如果有能解决的大手 请联系我:726710192@qq.com

