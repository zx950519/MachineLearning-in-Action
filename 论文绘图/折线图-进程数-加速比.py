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

def time_accle():
    # 数据源 数据格式：横坐标,折线1坐标,折线2坐标,折线3坐标,折线4坐标
    path = "./time.txt"
    # 坐标轴名称
    x_axis_name = "process"
    y_axis_name = "acceleration ratio"
    # 横轴范围
    x_value_min = 0
    x_value_max = 33
    x_value_gap = 2
    x_min = 0
    x_max = 33
    # 纵轴范围
    y_value_min = 0
    y_value_max = 25
    y_value_gap = 2
    y_min = 0
    y_max = 25
    # 数据读入
    x = []
    ssp_1 = []
    ssp_2 = []
    ssp_3 = []
    ssp_4 = []
    bsp = []
    dsp = []
    lsp = []
    mw = []
    my = []
    with open(path, "r") as csvfile:
        plots = csv.reader(csvfile, delimiter=",")
        for row in plots:
            x.append(int(row[0]))
            ssp_1.append(12600.0/float(row[1]))
            ssp_2.append(12600.0/float(row[2]))
            ssp_3.append(12600.0/float(row[3]))
            my.append(12600.0/float(row[4]))
            # bsp.append(float(row[4]))
            # dsp.append(float(row[5]))
            # lsp.append(float(row[6]))
            # mw.append(float(row[7]))

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
    plt.plot(x, ssp_1, ".-", color="springgreen", label="ssp=1")
    plt.plot(x, ssp_2, ".-", color="dodgerblue", label="ssp=2")
    plt.plot(x, ssp_3, ".-", color="red", label="ssp=3")
    plt.plot(x, my, ".-", color="k", label="my")
    # plt.plot(x, dsp, "s-", color="indianred", label="dsp")
    # plt.plot(x, lsp, ".-", color="y", label="lsp")
    # plt.plot(x, mw, ".-", color="k", label="mw")

    # 设置是否绘制网格
    plt.grid(False)
    # 设置图例属性值
    plt.legend(bbox_to_anchor=(1.0, 1.0), loc="best", borderaxespad=0.5)
    # plt.title("加速比")
    # plt.xlabel("核数")
    # plt.ylabel("时间(s)")
    # 绘图
    # plt.savefig("./time&proc.png")
    plt.show()

if __name__ == "__main__":
   time_accle()