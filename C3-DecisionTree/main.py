# from math import log
from numpy import *
import math
import matplotlib.pyplot as plt

# 创建数据集
def createDataSet():
    dataSet = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return dataSet, labels
# 计算给定数据集的香农熵
def calaShannonEnt(dataSet):
    # 计算数据集长度
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 获取标签
        currentLabel = featVec[-1]
        # 如果标签不在新定义的字典里，创建新的标签值
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 该标签下含有数据的个数
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 计算同类标签出现的概率
        prob = float(labelCounts[key])/numEntries
        # 以2为底求对数
        shannonEnt -= prob * math.log(prob, 2)
        # print(type(log(prob,2)))

    return shannonEnt
# 划分数据集
def spiltDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 将相同数据特征的抽取出来
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet
# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    # 计算数据集的香农值
    baseEntropy = calaShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 获取第i个特征值所有的可能值
        featList = [example[i] for example in dataSet]
        # 获取不重复的可能取值
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 以i为数据集特征 value为返回值 划分数据集
            subDataSet = spiltDataSet(dataSet, i, value)
            # 数据集特征为i所占的比例
            prob = len(subDataSet)/float(len(dataSet))
            # 计算每种数据集的信息熵
            newEntropy += prob*calaShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 计算最好的信息增益
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
# 递归创建决策树
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的
    return sortedClassCount[0][0]

# 创建树的函数
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果类别相同 则停止划分
    if classList.count(classList[0])==len(classList):
        return classList[0]
    # 遍历玩所有特征值 返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最好的数据集划分方式
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取对应的标签值
    bestFeatLabel = labels[bestFeat]
    print("目前最优标签为：", bestFeatLabel)
    myTree = {bestFeatLabel:{}}
    print("目前构造的树为：", myTree)
    # 清空
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    print("所选的标签可出现的值为：", uniqueVals)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(spiltDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotMidTest(cntrPt, parentPt,txtString):
    xMid = (parentPt[0] + cntrPt[0])/2.0
    yMid = (parentPt[1] + cntrPt[1])/2.0
    createPlot.ax1.text(xMid, yMid, txtString)

# 绘制自身
# 若当前子节点不是叶子节点，递归
# 若当子节点为叶子节点，绘制该节点
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    # depth = getTreeDepth(myTree)
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yoff)
    plotMidTest(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xoff = plotTree.xoff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidTest((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0/plotTree.totalD

# figure points
# 画结点的模板
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt,  # 注释的文字，（一个字符串）
                            xy=parentPt,  # 被注释的地方（一个坐标）
                            xycoords='axes fraction',  # xy所用的坐标系
                            xytext=centerPt,  # 插入文本的地方（一个坐标）
                            textcoords='axes fraction', # xytext所用的坐标系
                            va="center",
                            ha="center",
                            bbox=nodeType,  # 注释文字用的框的格式
                            arrowprops=arrow_args)  # 箭头属性

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0

    plotTree(inTree, (0.5, 1.0),'') #树的引用作为父节点，但不画出来，所以用''
    plt.show()

def getNumLeafs(myTree):
    numLeafs = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

# 子树中树高最大的那一颗的高度+1作为当前数的高度
def getTreeDepth(myTree):
    maxDepth = 0    #用来记录最高子树的高度+1
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if(thisDepth > maxDepth):
            maxDepth = thisDepth
    return maxDepth

# 方便测试用的人造测试树
def retrieveTree(i):
    listofTrees = [{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
                   {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
                   ]
    return listofTrees[i]

#使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    #将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, "w")
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def test_01():
    # 获取数据
    dataSet, labels = createDataSet()
    # 建树
    myTree = createTree(dataSet, labels)
    # print(calaShannonEnt(dataSet))
    print("最终构造出的树为：", myTree)
    myTree = retrieveTree(0)
    # 绘图
    createPlot(myTree)

def test_02():
    fr = open(r"./lenses.txt")
    lenses = [inst.strip().split("\t") for inst in fr.readlines()]
    lensesLables = ["age", "prescript", "astigmatic", "tearRate"]
    lensesTree = createTree(lenses, lensesLables)
    createPlot(lensesTree)

if __name__ == '__main__':
    # test_01()
    test_02()


