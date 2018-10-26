import seaborn as sns
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def utlization():
    # 数据源 数据格式：序号,利用率
    path = "./s2.txt"
    # 坐标轴名称
    x_axis_name = "iterator"
    y_axis_name = "utilization(%)"
    # 横轴范围
    x_value_min = 0
    x_value_max = 62
    x_value_gap = 5
    x_min = 0
    x_max = 60
    # 纵轴范围
    y_value_min = 0.0
    y_value_max = 1.0 + 0.1
    y_value_gap = 0.05
    y_min = 0.0
    y_max = 1.05
    # 数据读入
    x = []
    y = []
    index = 1
    with open(path, "r") as csvfile:
        plots = csv.reader(csvfile, delimiter=",")
        for row in plots:
            # x.append(int(row[0]))
            x.append(index)
            index = index + 1
            y.append(float(row[1])/100)

    # 获取子图的图像&坐标轴对象
    fig, ax = plt.subplots()
    # 设定坐标轴label
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    # 设定刻度以及坐标范围
    ax.set_xticks(np.arange(x_value_min, x_value_max, x_value_gap))
    ax.set_yticks(np.arange(y_value_min, y_value_max, y_value_gap))
    ax.set_ylim([y_min, y_max])
    ax.set_xlim(x_min, x_max)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 绘制曲线
    plt.plot(x, y, "-", color="deepskyblue")
    # 左中右：springgreen/deepskyblue/ orangered

    # 设置是否绘制网格
    plt.grid(False)
    # 设置图例属性值
    # plt.legend(bbox_to_anchor=(0.88, 1.0), loc="best", borderaxespad=0.5)
    # plt.title("Computation Node1")
    # 绘图
    # plt.savefig("./node2.png")
    plt.show()

if __name__ == "__main__":
    utlization()