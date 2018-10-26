from numpy import *


# 生成测试数据
def loadDataSet():
# 词条切分后的文档集合，列表每一行代表一个文档
    postingList = [['my', 'dog', 'has', 'flea',
                'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him',
                'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute',
                'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['my', 'licks', 'ate', 'my', 'steak', 'how',
                'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 由人工标注的每篇文档的类标签
    # 0-正常 1-侮辱
    classVec=[0, 1, 0, 1, 0, 1]
    return postingList, classVec

# 统计词条出现 生成词条字典
def createVocabList(dataSet):
    vocabSet = set([])
    # 遍历所有文档
    for document in dataSet:
        # 求并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 将文档转化为词条向量
# 参数：字典,文档
def setOfWords2Vec(vocabSet, inputSet):
    # 新建一个与字典长度一致的列表
    returnVec = [0] * len(vocabSet)
    # 遍历文档中的每一个词条
    for word in inputSet:
        if word in vocabSet:
            returnVec[vocabSet.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary! '%'word")
    return returnVec

# 训练算法
# 参数: 由每篇文档的词条向量组成的文档矩阵 每篇文档的类标签组成的向量
def trainNB0(trainMatrix, trainCategory):
    # 获取文档总数
    numTrainDocs = len(trainMatrix)
    # 获取词条向量的长度
    numWords = len(trainMatrix[0])
    # 所有文档汇中属于类"1"的比例
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建一个长度为词条向量等长的列表
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    # 遍历每一篇文档
    for i in range(numTrainDocs):
        # 如果该文档属于类别1
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            # trainMatrix[i]代表文档中出现的词汇在字典Set中是否出现 1-出现 0-未出现
            # [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocalList, inputSet):
    # 词袋向量
    returnVec = [0] * len(vocalList)
    for word in inputSet:
        if word in vocalList:
            returnVec[vocalList.index(word)] += 1
    return returnVec

def testingNB():
    # 文档矩阵 类标签向量
    listOPosts, listClasses = loadDataSet()
    # 获取所有文档中词的字典
    myVocabList = createVocabList(listOPosts)
    print("初始数据集", listOPosts)
    print(type(listOPosts))
    print("字典为：", myVocabList)
    # 列表
    trainMat = []
    # 对每篇文档处理
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 已获取字典在文档中的映射矩阵
    # 调用训练函数 获取概率值
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    # 测试文档
    testEntry = ["love", "my", "dalmation"]
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # print(thisDoc)
    # print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry1 = ["stupid", "garbage"]
    thisDoc1 = array(setOfWords2Vec(myVocabList, testEntry1))
    # print(testEntry1, 'classified as:', classifyNB(thisDoc1, p0V, p1V, pAb))

def testParse(bigString):
    import re
    listOfTokens=re.split(r'\\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullTest = []
    # 读取邮件
    for i in range(1, 26):

        wordList = testParse(open("./email/spam/%d.txt" %i).read())
        # 将得到的字符串列表添加到docList
        docList.append(wordList)
        # 将得到的字符串列表添加到fullTest
        fullTest.extend(wordList)
        # 添加类标签
        classList.append(1)

        wordList = testParse(open("./email/ham/%d.txt" %i).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)

    print("初始数据集：", docList)
    # -----------------------------------------------
    # 修改
    newDocList = []
    # print(type(newDocList))
    sum = 0
    for dl in docList:
        sp = dl[0].split()
        # print(type(sp))
        newDocList.append(sp)
        sum += len(sp)
        # print("!@#", len(sp), sp)
    # print(newDocList)
    print(sum)
    print(newDocList[40])
    # -----------------------------------------------
    # 构建字典
    vocabList = createVocabList(newDocList)
    # print(type(docList))
    print("字典为：", len(vocabList), vocabList)
    # 构建列表
    trainingSet = list(range(50))
    testSet = []
    # 随机选10个数作为测试集的索引
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    # 遍历训练集
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, fullTest[docIndex]))
        trainClasses.append(classList[docIndex])
    # 计算贝叶斯函数需要的概率值并返回
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 统计错误率
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is:", float(errorCount) / len(testSet))

# 遍历词汇表中每个词并统计出现次数 返回出现次数最多的30个单词
def calMostFrep(vocabList, fullTest):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullTest.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    # 文档列表 分类列表 测试集
    docList = [];
    classList = [];
    fullTest = []
    # 获取条目较少的RSS源的条目数
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    # 遍历每一个条目
    for i in range(minLen):
        # 解析和处理获取的相应数据
        wordList = testParse(feed1['entries'][i]['summary'])
        # 添加词条列表到docList
        docList.append(wordList)
        # 添加词条元素到fullTest
        fullTest.extend(wordList)
        # 类标签列表添加类1
        classList.append(1)
        # 同上
        wordList = testParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullTest.extend(wordList)
        # 此时添加类标签0
        classList.append(0)
    # 构建出现的所有词条列表
    vocabList = createVocabList(docList)
    # 找到出现的单词中频率最高的30个单词
    top30Words = calMostFrep(vocabList, fullTest)
    # 遍历每一个高频词，并将其在词条列表中移除
    # 这里移除高频词后错误率下降，如果继续移除结构上的辅助词
    # 错误率很可能会继续下降
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    # 下面内容与函数spaTest完全相同
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

# 最具表征性的词汇显示函数
def getTopWords(ny, sf):
    import operator
    # 利用RSS源分类器获取所有出现的词条列表，以及每个分类中每个单词出现的概率
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY=[]
    topSF=[]
    # 遍历每个类中各个单词的概率值
    for i in range(len(p0V)):
        # 往相应元组列表中添加概率值大于阈值的单词及其概率值组成的二元列表
        if(p0V[i]>-6.0):topSF.append((vocabList[i],p0V[i]))
        if(p1V[i]>-6.0):topNY.append((vocabList[i],p1V[i]))
    # 对列表按照每个二元列表中的概率值项进行排序，排序规则由大到小
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse = True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    #遍历列表中的每一个二元条目列表
    for item in sortedSF:
        #打印每个二元列表中的单词字符串元素
        print(item[0])
    #解析同上
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse = True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedNY:
        print(item[0])

if __name__ == '__main__':
    print("朴素贝叶斯")
    # 简单实例
    testingNB()
    # 垃圾邮件
    spamTest()
    # 区域倾向
    # import feedparser
    # ny = feedparser.parse("http://newyork.craigslist.org/stp/index.rss")
    # sf = feedparser.parse("http://sfbay.craigslist.org/stp/index.rss")
    # vocabList, pSF, pNY = localWords(ny, sf)
    # getTopWords(ny, sf)
