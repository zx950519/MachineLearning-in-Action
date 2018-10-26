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

def bar():

    a = [666, 687, 652, 671, 709, 693, 2000, 2000, 0, 0]  # 时间
    b = [0.9643, 0.9744, 0.9533, 0.9549, 0.9512, 0.966, 0.0, 0.0, 0.0, 0.0] # 准确率
    l = [i for i in range(len(a))]
    lx = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.spines['top'].set_visible(False)
    plt.bar(l, a, 0.50, alpha=0.9, color="#87CEFA", label="时间开销")
    ax1.set_ylim([500, 1000])  # 设置y轴取值范围
    ax1.set_yticks(np.arange(500, 1015, 25))
    ax1.set_ylabel("时间开销(S)")
    plt.legend(prop={'family': 'SimHei', 'size': 8}, loc="upper right", bbox_to_anchor=(1.0, 0.92))

    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.plot(l, b, "or-", label="准确率")
    # ax1.yaxis.set_major_formatter(yticks)
    for i, (_x, _y) in enumerate(zip(l, b)):
        plt.text(_x, _y, b[i], color="black", fontsize=10, )  # 将数值显示在图形上
    ax2.set_ylim([-0.05, 1.0])
    ax2.set_yticks(np.arange(-0.05, 1.0125, 0.05))
    ax2.set_ylabel("准确率(%)")
    ax2.set_xlabel("不同的环境")
    plt.legend(prop={'family': 'SimHei', 'size': 8}, loc="upper right", bbox_to_anchor=(1.0, 0.85))

    plt.xticks(l, lx)
    # plt.title("有干扰条件下准确率与时间开销的关系")
    plt.show()

if __name__ == "__main__":
   bar()