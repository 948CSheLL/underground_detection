# coding: utf-8

import numpy as np
from functools import partial
import os
import pandas as pd
import matplotlib.pyplot as plt  #  绘制图像的库

def plot_data(M1_without_noise=None,M2_without_noise=None,
              t=None,dir_save=None,plot_name=None,label1=None,label2=None):
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-8,1e1)
    p1 = plt.plot(t, M1_without_noise, '--', color="r", label=label1)
    p2 = plt.plot(t, M2_without_noise, ':', color="b", label=label2)
    plt.legend(loc='lower left')
    plt.xlabel("t /s")
    plt.ylabel("M")
    plt.title(plot_name)
    plt.savefig(dir_save+plot_name)
    plt.show()
    plt.close()


def get_data_from(filename):
    data = pd.read_csv(filename)
    unwanted_columns = ["Unnamed: 0", ]
    data = data.drop(unwanted_columns, axis=1)
    # print(data)
    result = np.array(data)
    # print(result)
    # print(result.shape)
    return result


# k,αβγ取值范围分别为 (0, [60, 200, 7, 4000000])
def func(t, k, a, b, R):  # 返回球体的响应方程，用于拟合算法，需要此方程,a,b,R其实是alpha, beta, gamma
    e = np.e
    return k * pow((1 + pow(t / a, 1 / 2)), -b) * pow(e, -t / R)


if __name__ == "__main__":
    data_snr00 = get_data_from(r".\snr00withoutnoise.csv")
    parameters_snr00 = get_data_from(r".\snr00withoutnoise_inv_para.csv")

    # sample 代表第几个样本
    # sample = 116
    sample_lineno = 0
    t_split = 200
    t = np.array(10 ** (np.linspace(-8, 0, t_split)))

    k1_ellipsoid, a1, b1, R1 = parameters_snr00[sample_lineno, 0:4]
    print(list(map(partial(func, k=k1_ellipsoid, a=a1, b=b1, R=R1), t)))
    M1_without_noise = np.array(list(map(partial(func, k=k1_ellipsoid, a=a1, b=b1, R=R1), t)))
    M2_without_noise = data_snr00[sample_lineno, 0:200]

    plot_name = "sample " + str(sample_lineno)
    label1 = "α=" + str(a1) + " β=" + str(b1)
    os.makedirs("./sample selected", exist_ok=True)
    dir_selected = './sample selected/'
    plot_data(M1_without_noise=M1_without_noise, M2_without_noise=M2_without_noise,
              t=t, dir_save=dir_selected, plot_name=plot_name, label1=label1)
