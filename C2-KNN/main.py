from numpy import *
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    drawPicture(group[:, 0], group[:, 1])
    return group, labels

def drawPicture(x, y):
    # plt.scatter(group[:, 0], group[:, 1])
    plt.scatter(x, y)
    plt.show()

def classify(x, dataSet, labels, k):

    # 样本数
    dataSetSize = dataSet.shape[0]
    # tile(A,B)函数，按照B的模式，重复A若干次
    diffMat = tile(x, (dataSetSize, 1)) - dataSet
    # diffMat保存测试点与其他所有点的x,y坐标距离,**2即平方化处理
    sqDiffMat = diffMat ** 2
    # 按行累加平方和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方得欧氏距离
    distances = sqDistances ** 0.5
    # 按照升序排列
    sortedDistIndicies = distances.argsort()
    classCount={}
    # 投票
    for i in range(k):
        # 获取投票率高的结果
        voteIlable = labels[sortedDistIndicies[i]]
        # 得票+1
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    # 统计上一轮得票
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    # 返回唯一结果
    return sortedClassCount[0][0]

def file_to_matrix(filename):
    # 读取文件
    fr = open(filename)
    arrayOLines = fr.readlines()
    # 统计行数
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # 去掉换行符
        line = line.strip()
        # 分割字符串
        listFromLine = line.split("\t")
        # 重构欲返回的数据集
        returnMat[index,:] = listFromLine[0:3]
        # 把样本标签放到返回的数据集中
        classLabelVector.append(int(listFromLine[-1]))
        # 序号自增
        index += 1
    return returnMat, classLabelVector
# 归一化操作
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # 定义返回的结果集
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 做差
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 相除
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, range, minVals
# 图片转数组
def img_to_vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

# 简单KNN实例
def test_01():
    group, labels = createDataSet()
    x = [0.1, 0.1]
    className = classify(x, group, labels, 3)
    print(className)
# 约会网站实例
def datingClassTest():
    # 10%数据用于训练
    hoRatio = 0.10
    # 加载数据
    datingDataMat, datingLabels = file_to_matrix("./datingTestSet.txt")
    print(datingDataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
               15.0*array(datingLabels), 15*array(datingLabels))
    plt.show()
    # 归一化
    normMat, range, minVals = autoNorm(datingDataMat)
    # 数据总量
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    # 错误率
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d, result is :%s" % (
        classifierResult, datingLabels[i], classifierResult == datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount, "%")
# K近邻识别手写数字
def handwritingClassTest():

    hwLabels = []
    # 获取目录
    trainingFileList = listdir("./digits/trainingDigits")
    # 目录下文件数
    m = len(trainingFileList)
    # 构建训练集
    trainingMat = zeros((m, 1024))

    for i in range(m):
        # 文件名
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        # 输入数据转化
        trainingMat[i,:] = img_to_vector("./digits/trainingDigits/%s" % fileNameStr)
    # 测试文件目录
    testFileList = listdir("./digits/testDigits")
    # 错误数量
    errorCount = 0.0
    # 测试集数量
    mTest = len(testFileList)

    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img_to_vector("./digits/testDigits/%s" % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d, The predict result is: %s" % (
        classifierResult, classNumStr, classifierResult == classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    # 统计结果
    print("\nthe total number of errors is: %d / %d" % (errorCount, mTest))
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

if __name__ == '__main__':
    # test_01()
    datingClassTest()
    # handwritingClassTest()
    print("Test")