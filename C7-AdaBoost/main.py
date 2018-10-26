import numpy as np
import matplotlib.pyplot as plt

#生成一些训练数据和标签：
def loadSimData():
    # 输入：无
    # 功能：提供一个两个特征的数据集
    # 输出：带有标签的数据集

    # np是numpy的代名词
    # 生成数据矩阵
    datMat = np.matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    print("初始数据集为:")
    print(datMat)
    #生成标签集-List类型
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    # print(len(classLabels))
    return datMat, classLabels

def DrawData(data, labels):
        x0 = []
        y0 = []
        x1 = []
        y1 = []
        for i in range(len(labels)):
            if (labels[i] == -1):
                x0.append(data[i, 0])  # 点的 x y坐标
                y0.append(data[i, 1])
            else:
                x1.append(data[i, 0])  # 点的 x y坐标
                y1.append(data[i, 1])
        fig = plt.figure()  # 创建画笔
        ax = fig.add_subplot(111)  # 创建画布
        ax.scatter(x0, y0, marker='s', s=90)  # 画散点图， 参数： x,y坐标，以及形状 颜色 大小等
        ax.scatter(x1, y1, marker='o', s=90, c='red')
        plt.title('Original Data')  # 标题
        plt.show()  # 显示

# 函数stumpClassify()是将输入数据（矩阵）分为两类，首先将返回数组全部设置为1
# 然后将满足不等式要求的元素设置为-1，dimen保证可以基于数据集中任一元素进行比较
# 最后一个元素threshIneq：将不等号在大于和小于之间切换
def stumpClassify(dataMatrix,dimen,thresholdValue,thresholdIneq):
    # 输入：数据矩阵   特征维数    某一特征的分类阈值   分类不等号
    # 功能：输出决策树标签
    # 输出：标签

    # 通过比较阈值对数据进行分类，以阈值为界分为｛+1 -1｝两类 相当于剪枝分类器。
    # 输入参数 dimen 是哪个特征， threshIneq：有两种模式，在大于和小于之间切换不等式

    # 构造一个数组 初始化全1 np.shape(dataMatrix)[0]获取的是dataMatrix的列数
    # 例如 np.shape(dataMatrix)[0]=4 将会生成如下：
    # [[1.]
    #  [1.]
    #  [1.]
    #  [1.]]
    # 即一个四行一列的矩阵
    returnArray = np.ones((np.shape(dataMatrix)[0], 1))
    # 所有在阈值一边的数据会分类到-1 另一边分到+1
    if thresholdIneq == 'lt':
        returnArray[dataMatrix[:, dimen] <= thresholdValue] = -1
    else:
        returnArray[dataMatrix[:, dimen] > thresholdValue] = -1

    return returnArray

# 伪代码
# 将最小错误率minerror设为inf
# 对数据中每个特征：
#     对每个步长：
#         对每个不等号（分类错误的样本）：
#             建立一个单层决策树并利用加权数据集对它测试
#             如果错误率低于minerror，则将当前单层决策树设为最佳单层决策树

# 返回最佳的单层决策树(建立单层决策树)
def buildStump(dataArray, classLabels, D):
    # 输入：数据矩阵   对应的真实类别标签   特征的权值分布(样本初始权重向量)
    # 功能：在数据集上，找到加权错误率(分类错误率)最小的单层决策树，显然，该指标函数与权重向量有密切关系
    # 输出：最佳树桩(特征，分类特征阈值，不等号方向)，最小加权错误率，该权值向量D下的分类标签估计值

    #数据矩阵
    dataMatrix = np.mat(dataArray)
    #将标签转化成矩阵后转置
    labelMat = np.mat(classLabels).T
    #m,n分别为矩阵的行列数 例如np.shape(dataMatrix)返回的是(4,6) 那么m,n分别是4,6
    m, n = np.shape(dataMatrix)
    #步数
    stepNum = 10.0
    # 用来存放最后的单层决策树的字典(Map Directory)
    bestStump = {}
    # 类别估计值（预测标签）构造一个矩阵 该矩阵有m行1列，每行的元素都是0
    bestClassEst = np.mat(np.zeros((m, 1)))
    # 最小错误率初始值设置为 无限大
    minError = np.inf
    # 对数据中的每一个特征 n是列数
    # 第一层循环：在数据集所有特征上进行遍历
    for i in range(n):
        print("第1层第%d次循环" % i)
        # 得到某一特征中最值 rangeMin与rangeMax分别对应第i列的最值
        if i == 0:
            print("对X轴进行处理")
        else:
            print("对Y轴进行处理")
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        #获取步长
        stepSize = (rangeMax - rangeMin)/stepNum

        #第二层循环：在某特征的所有值上进行遍历(不断改变阈值，阈值的设置尽量大于特征值的范围)
        for j in range(-1, int(stepNum)+1):
            # 在大于和小于之间切换不等式 threshold意为阈值、入口、门槛
            # 对每个不等号(分类错误的样本)
            # 第三层循环：在大于和小于等于之间切换不等式。
            print("第2层第%d次循环" % int(j+1))
            for thresholdIneq in ["lt", "gt"]:
                # 阈值不断变大
                print(thresholdIneq)
                thresholdValue = rangeMin + float(j) * stepSize
                # 预测的标签值 通过调用上个函数返回分类预测结果
                predictClass = stumpClassify(dataMatrix, i, thresholdValue, thresholdIneq)
                # 构造一个全是1的矩阵(列向量)
                errArray = np.mat(np.ones((m, 1)))
                # 预测正确的为0 预测错误的为1
                errArray[predictClass == labelMat] = 0
                # 得到加权分类误差
                # 将错误向量和权重向量D相应元素相乘并求和
                # 就得到了数值weightedError，这就是AdaBoost和分类器交互的地方
                # 这里我们是基于权重向量D而不是其他错误计算指标来评价分类器
                weightError = D.T * errArray
                # print(weightError)
                print("划分: 维度 %d, 阈值: %.2f, 符号:%s, 错误率 %.3F" % (i, thresholdValue, thresholdIneq, weightError))

                # 如果加权分类误差小于最小误差，更新最小误差
                if weightError < minError:
                    # 更新
                    print("更新最小误差")
                    minError = weightError
                    # 更新类别估计值（预测标签）矩阵
                    bestClassEst = predictClass.copy()
                    # print(bestClassEst)
                    # 单层决策树的字典各个属性赋值
                    bestStump['dimen'] = i
                    bestStump['thresholdValue'] = thresholdValue
                    bestStump['thresholdIneq'] = thresholdIneq
                    print("当前更新后的单层决策树： 维度 %d , 阈值 %.2f , 类型 %s" % (i, thresholdValue, thresholdIneq))

    return bestClassEst, minError, bestStump

# 伪代码
# 对每次迭代：
#     利用buildStump函数找到最佳的单层决策树
#     将最佳单层决策树加入到单层决策树组
#     计算alpha
#     计算新的权重向量D
#     更新累计估计值
#     如果错误率等于0.0，则退出循环
def adaBoostTrainDS(dataArray,classLabels,numIt = 40):

    # 输入：数据集，标签向量，最大迭代次数
    # 功能：创建adaboost加法模型
    # 输出：多个弱分类器的数组

    # 定义弱分类数组，保存每个基本分类器bestStump
    # 存储每次迭代产生的最佳单层决策树和alpha值
    weakClass = []
    #样本个数
    m, n = np.shape(dataArray)
    # 初始的样本权重设置为一样的（1/m）
    # m行1列的矩阵
    D = np.mat(np.ones((m, 1))/m)
    # 记录每个数据点的类别估计 累计值
    aggClassEst = np.mat(np.zeros((m, 1)))
    #开始迭代 40次
    for i in range(numIt):
        print("第%d次迭代:" % i)
        # 调用函数返回（1）最佳单层决策树，（2）分类器估计错误率和（3）预测标签值
        bestClassEst, minError, bestStump = buildStump(dataArray, classLabels,D)#step1:找到最佳的单层决策树
        #输出权重向量D
        print("权重向量为：")
        print("D.T:", D.T)
        # 分类器的权重  分母max保证如果没有错误时防止分母为0
        alpha = float(0.5 * np.log((1-minError)/max(minError, 1e-16)))#step2: 更新alpha
        print("alpha值为：")
        print("alpha:", alpha)
        # 将alpha 放入到最佳决策树的字典中
        bestStump['alpha'] = alpha
        weakClass.append(bestStump)# step3:将基本分类器添加到弱分类的数组中
        print("classEst:", bestClassEst.T)

        #开始更新权重向量
        #classLabels 和classEst 都是m*1的，multiply函数对应元素相乘返回还是1*1的
        #multiply是numpy的乘法运算 对应元素相乘
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, bestClassEst)
        # print(expon)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()# step4:更新权重，该式是让D服从概率分布
        # 相当于更新强分类器 aggClassEst
        aggClassEst += alpha * bestClassEst#steo5:更新累计类别估计值
        print("aggClassEst:", aggClassEst.T)
        print(np.sign(aggClassEst) != np.mat(classLabels))

        # 错误率的累加
        # 判段预测值与实际值是否一致，不一致的赋为1，预测一致为0　可得预测错误的个数
        aggError = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        print("aggError",aggError.T)
        # 求得误差率
        aggErrorRate = aggError.sum()/m
        print("total error:",aggErrorRate)
        if aggErrorRate == 0.0: break

    return weakClass

# 测试
def adaTestClassify(dataToClassify,weakClass):
    # print("开始测试")
    #数据矩阵
    dataMatrix = np.mat(dataToClassify)
    # m是dataMatrix列的数量
    m = np.shape(dataMatrix)[0]
    #累计类别估计值
    aggClassEst = np.mat(np.zeros((m,1)))
    # 循环次数取决于弱分类器的数量
    for i in range(len(weakClass)):
        classEst = stumpClassify(dataToClassify,
                                 weakClass[i]['dimen'],
                                 weakClass[i]['thresholdValue'],
                                 weakClass[i]['thresholdIneq'])
        aggClassEst += weakClass[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)

def loadDataSet_b(fileName):
    numFeat = len(open(fileName).readline().split("\t"))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat, labelMat

if __name__  ==  '__main__':

    # 初始权值矩阵
    D = np.mat(np.ones((5, 1))/5)
    # 加载数据
    dataMatrix, classLabels = loadSimData()
    # 找到初始单层决策树
    bestClassEst, minError, bestStump = buildStump(dataMatrix, classLabels, D)
    # 获得决策树组
    weakClass = adaBoostTrainDS(dataMatrix, classLabels, 9)
    # 测试
    testClass = adaTestClassify(np.mat([0, 0]), weakClass)
    # 绘图
    # DrawData(dataMatrix, classLabels)


