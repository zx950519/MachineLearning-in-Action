import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import math

# 加载数据
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# 在样本集中随机选择第二个不等于alpha_i的优化向量alpha_j
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

# 调整alpha的值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 绘图
def showDataSet(dataMat, labelMat):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()


# 简单SMO
# 参数：dataMatIn 数据列表
# 参数：classLabels 标签列表
# 参数：C 权衡因子(增加松弛因子在目标优化函数中引入了惩罚项)
# 参数：toler 容错率
# 参数：maxIter 最大迭代次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 转化为矩阵或向量
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    # 获取行数和列数
    m, n = shape(dataMatrix)
    # 新建一个m行1列的向量
    alphas = mat(zeros((m, 1)))
    # 迭代次数为0
    iter = 0
    # 开始迭代
    while (iter < maxIter):
        # 用于记录alpha是否已经进行优化
        alphaPairsChanged = 0
        for i in range(m):
            # Step-1
            # fXi是预测的类别
            fXi = float((multiply(alphas, labelMat).T) * (dataMatrix * dataMatrix[i,:].T)) + b
            # 计算预测和实际的差别
            Ei = fXi -float(labelMat[i])
            # 检查正间隔、负间隔、alpha(不能等于0或C)
            if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ( (labelMat[i] * Ei > toler) and (alphas[i] > 0) ):
                # 随机选择另一个变量
                j = selectJrand(i, m)
                # fXj是预测的类别
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                # Step-1
                # 计算预测和实际的差别
                Ej = fXj - float(labelMat[j])
                # alpha原始值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # Step-2
                # 计算L H的值 用于将alpha[j]调整到0-C之间
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果LH一致 结束此次内循环
                if L == H:print("L==H");continue
                # Step-3
                # eta是alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                # 若eta>=0 结束此次内循环
                if(eta >= 0): print("eta>0"); continue
                # 计算获得新的alpha[j]值
                # Step-4
                # 更新
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                # Step-5
                # 修剪
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 如果alpha[j]变化不大 直接结束此次内循环
                if(abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                # Step-6
                # 计算alpha[i]的值
                # 对i的修改量与j相同 但方向相反
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # Step-7
                # 再计算两个alpha对应的b值
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                # Step-8
                # 更新b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                # 运行至此 alpha值改变 修改状态
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" %(iter, i, alphaPairsChanged))
        # 判断是否有alpha对改变 没有则进行下一次迭代
        if(alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" %iter)
    return b, alphas

def showClassifer(dataMat, labelMat, w, b):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)

    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])

    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c="none", alpha=0.7, linewidths=1.5, edgecolors="red")

    plt.show()

def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()

# 保存值的数据结构
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 误差缓存
        # 第一列表名eCache是否有效 第二列给出实际的E值
        self.eCache = mat(zeros((self.m, 2)))

# 格式化计算误差的函数
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 修改选择第二个变量alpha_j的方法
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    # 获取缓存中Ei不为0的样本对应的alpha_j列表
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    # 在误差不为0的列表中找出使Ei-Ej绝对值最大的alpha_j
    if(len(validEcacheList)>0):
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei-Ek)
            if(deltaE > maxDeltaE):
                # 选择具有最大的步长k
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

# 更新误差矩阵
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=("lin", 0)):
    # 保存关键数据
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
    iter = 0
    enrireSet = True
    alphaPairsChanged = 0
    # 选取第一个变量的三种情况 从间隔边界上选取或整个数据集
    while(iter < maxIter) and ((alphaPairsChanged > 0) or (enrireSet)):
        alphaPairsChanged = 0
        # 没有alpha更新对
        if enrireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            print("fullSet, iter: %d i:%d,pairs changed %d" %(iter, i, alphaPairsChanged))
        else:
            # 统计alphas向量中满足0<alpha<C的alpha列表
            nonBoundIs = nonzero(((oS.alphas.A)>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound,iter:%d i:%d,pairs changer %d" %(iter, i, alphaPairsChanged))
            iter += 1
        if enrireSet:enrireSet = False
        elif (alphaPairsChanged == 0):enrireSet = True
        print("iteration number: %d" %iter)
    return oS.b,oS.alphas

# 内循环寻找alpha_j
def innerL(i, oS):
    # 计算误差
    Ei = calcEk(oS, i)
    # 违背KKT条件
    if(((oS.labelMat[i]*Ei < -oS.tol)and(oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol)and(oS.alphas[i]>0))):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 计算上下界
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.C + oS.alphas[j] + oS.alphas[i])
        if L==H:print("L==H");return 0
        eta = 2.0*oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0:print("eta>=0");return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if(abs(oS.alphas[j]-alphaJold) < 0.00001):
            print("j not moving enough");return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS, i)

        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i]-alphaIold)*oS.X[i, :]*oS.X[i, :].T-\
            oS.labelMat[j] * (oS.alphas[j]-alphaJold) * oS.X[i, :]*oS.X[j, :].T

        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T

        if(0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif(0 < oS.alphas[j] and (oS.C > oS.alphas[j])):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def predict(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels)
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    result = dataArr[0] * mat(w) + b
    return sign(result)

# 核转化函数 径向基核函数
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    # 判断核函数类型
    if kTup[0] == "lin": K = X * A.T
    # rbf--径向基核函数
    # 将每个样本转化为高维空间
    elif kTup[0] == "rbf":
        for j in range(m):
            # print("第 %d 次原始数组" % j, X)
            # print("第 %d 次的A" % j, A)
            deltaRow = X[j, :] - A
            # print("第 %d 次的deltaRow" % j, deltaRow)
            K[j] = deltaRow * deltaRow.T
            # print("第 %d 次的k[j]" % j, K[j])
        K = exp(K / (-1*kTup[1]**2))
        # print("第 %d 次最终得到的K" % j, K)
    else:
        raise NameError("Houston We have a problem -- That Kernel is not recognized")
    return K

# 测试径向基核函数
def testRbf(k1 = 1.3):
    dataArr, labelArr = loadDataSet("./testSetRBF.txt")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ("rbf", k1))
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" %shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ("rbf", k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("The training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadDataSet("./testSetRBF2.txt")
    errorCount = 0
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ("rbf", k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("The best error rate is: %f" %(float(errorCount)/m))

if __name__=="__main__":
    dataArr, labelArr = loadDataSet("./testSet.txt")
    showDataSet(dataArr, labelArr)

    # b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    # w = get_w(dataArr, labelArr, alphas)
    # showClassifer(dataArr, labelArr, w, b)

    # b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    # w = get_w(dataArr, labelArr, alphas)
    # showClassifer(dataArr, labelArr, w, b)

    testRbf()
