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

def bsa():

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    X1 = [3.65, 5.65, 7.65, 9.65, 11.65, 13.65, 15.65, 17.65, 19.65, 21.65, 23.65, 25.65, 27.65, 29.65, 31.65]  # X是1,2,3,4,5,6,7,8,柱的个数
    X2 = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
    X3 = [4.35, 6.35, 8.35, 10.35, 12.35, 14.35, 16.35, 18.35, 20.35, 22.35, 24.35, 26.35, 28.35, 30.35, 32.35]
    Y1 = [0.997, 0.9866, 0.9793, 0.9758, 0.9668, 0.9644, 0.9603, 0.9575, 0.9524, 0.9501, 0.9483, 0.9472, 0.9335, 0.9430, 0.9397]
    Y2 = [0.987, 0.9836, 0.9723, 0.9741, 0.9702, 0.960, 0.9573, 0.9522, 0.9503, 0.9492, 0.9415, 0.9463, 0.9321, 0.9354, 0.9291]
    Y3 = [0.972, 0.9766, 0.9612, 0.965, 0.9568, 0.9541, 0.9403, 0.9375, 0.9224, 0.9101, 0.9183, 0.9272, 0.9035, 0.9430, 0.9397]

    ax.set_xticks(np.arange(2, 34, 2))
    ax.set_yticks(np.arange(0.85, 1.05, 0.00625))
    ax.set_ylim([0.85, 1])
    ax.set_xlim(2, 34)

    ax.set_ylabel("Accurary(%)")
    ax.set_xlabel("Computing Process")

    plt.bar(X1, Y1, alpha=0.9, width=0.35, facecolor='dodgerblue', edgecolor='white', label='BSP', lw=1)
    plt.bar(X2, Y2, alpha=0.9, width=0.35, facecolor='gold', edgecolor='white', label='SSP', lw=1)
    plt.bar(X3, Y3, alpha=0.9, width=0.35, facecolor='red', edgecolor='white', label='ASP', lw=1)

    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    bsa()