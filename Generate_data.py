import numpy as np
import time
from functools import partial
import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt  #  绘制图像的库
from matplotlib.pyplot import plot,savefig
# 材 质：行顺序： 钢 镍 铝  列顺序：相对磁导率 绝对磁导率 电导率
# 相对磁导率 = 绝对磁导率/4pi*10^-7  范围[1,1000]
attribute = np.array([[696.3028547, 875 * 1e-6, 50000000], [99.47183638, 125 * 1e-6, 14619883.04],
                      [1.000022202, 1.256665 * 1e-6, 37667620.91]])


def Msphere(c, c0, d, r, t):  # c是相对磁化率 μr；d是电导率 σ；r是半径；t是时间区间
    pi = np.pi
    e1 = pi
    e2 = pi * 1.5  # 二分范围
    # c=1.00125;              #相对磁化率 μr
    # c0 = 1.2566370614 * 1e-6  # 绝对磁化率 μ0
    # d=10**7                 #电导率 σ
    a1 = 1.38  # a=1.38
    # r=0.02                  #半径
    e = 2.718281828459
    global step
    step = 0

    def f(x, y):
        return np.tan(x) - ((y - 1) * x / ((y - 1) + x ** 2))  # 超越方程

    def erfen(a, b, c):  # a、b为方程的根区间，c是相对磁化率
        global step
        fhalf = f((a + b) / 2, c)
        half = (a + b) / 2
        fa = f(a, c)
        fb = f(b, c)
        step = step + 1
        if (fa == 0):
            return a
        if (fb == 0):
            return a
        if (fhalf == 0):
            return fhalf
        if np.sqrt(abs(fa * fb)) < 1e-10:
            return a
        if fhalf * fb < 0:
            return erfen(half, b, c)
        else:
            return erfen(a, half, c)

    x = erfen(e1, e2, c)  # x即为超越方程的解  δ1
    # print(x)

    #################  t0 t1 ############
    t0 = (d * c * c0 * r ** 2) / (x ** 2)
    if c > 2:
        t1 = (d * c * c0 * r ** 2) / ((c + 2) * (c - 1))
    else:
        t1 = t0
    #################  K  #####################
    k = (6 * pi * r ** 3 * c) / (c + 2)
    #################  a  ###################
    a = a1 * t1
    #################  b  ######################
    b = 2 * (c + 2) * pow(a, 1 / 2) / (pow((pi * c * c0 * d), 1 / 2) * r)
    ##################  γ  ####################
    r1 = (1 + pow((a1 * t1 / 2 * t0), 1 / 2)) / (1 + pow((a1 * t1 / 2 * t0), 1 / 2) - b / 4)  # b
    R = r1 * t0
    #################### Mspher #####################
    # print("k", k, "a", a, "b", b, "R", R)
    y = k * pow((1 + pow(t / a, 1 / 2)), -b) * pow(e, -t / R)
    return y  # y=f(t)，球体响应可拆分为两个方向上的响应，作为求解()函数的输入
    #################### 两个方向上的极化率值 #######################


#############超越方程解###############
def parameter_sphere(c, c0, d, r):  # c是相对磁化率 μr；d是电导率 σ；r是半径；t是时间区间
    # pi = 3.14
    pi = np.pi
    e1 = pi
    e2 = pi * 1.5  # 二分范围
    # c=1.00125;              #相对磁化率 μr
    # c0 = 1.2566370614 * 1e-6  # 绝对磁化率 μ0
    # d=10**7                 #电导率 σ
    a1 = 1.38  # a=1.38
    # r=0.02                  #半径
    e = np.e
    global step
    step = 0

    def f(x, y):
        return np.tan(x) - ((y - 1) * x / ((y - 1) + x ** 2))  # 超越方程  先验方程

    def erfen(a, b, c):  # a、b为方程的根区间，c是相对磁化率
        global step
        fhalf = f((a + b) / 2, c)
        half = (a + b) / 2
        fa = f(a, c)
        fb = f(b, c)
        step = step + 1
        if (fa == 0):
            return a
        if (fb == 0):
            return a
        if (fhalf == 0):
            return fhalf
        if np.sqrt(abs(fa * fb)) < 1e-10:
            return a
        if fhalf * fb < 0:
            return erfen(half, b, c)
        else:
            return erfen(a, half, c)

    x = erfen(e1, e2, c)  # x即为超越方程的解  δ1
    # print(x)

    #################  t0 t1 ############
    t0 = (d * c * c0 * r ** 2) / (x ** 2)
    if c > 20:
        t1 = (d * c * c0 * r ** 2) / ((c + 2) * (c - 1))
    else:
        t1 = t0
    #################  K  #####################
    k = (6 * pi * r ** 3 * c) / (c + 2)
    #################  a  ###################
    a = a1 * t1
    #################  b  ######################
    b = 2 * (c + 2) * pow(a, 1 / 2) / (pow((pi * c * c0 * d), 1 / 2) * r)
    ##################  γ  ####################
    r1 = (1 + pow((a1 * t1 / 2 * t0), 1 / 2)) / (1 + pow((a1 * t1 / 2 * t0), 1 / 2) - b / 4)  # b
    R = r1 * t0

    # 椭球体改变了K
    return k, a, b, R


def ellipsoid_k_plus(ta, tb, c):
    shape = ta / tb
    if shape < 1:  # h1，h2为退磁因子
        h1 = (shape ** 2 / (1 - shape ** 2)) * (
                ((np.arctanh(pow(1 - shape ** 2, 1 / 2))) / pow(1 - shape ** 2, 1 / 2)) - 1)  # 轴向-对应b
        h2 = (1 / (2 * (1 - shape ** 2))) * (
                1 - (shape ** 2 * np.arctanh(pow(1 - shape ** 2, 1 / 2)) / pow(1 - shape ** 2, 1 / 2)))  # 横向-对应a
    else:
        h1 = (shape ** 2 / (shape ** 2 - 1)) * (
                1 - ((np.arctan(pow(shape ** 2 - 1, 1 / 2))) / pow(shape ** 2 - 1, 1 / 2)))
        h2 = (1 / (2 * (shape ** 2 - 1))) * (
                (shape ** 2 * np.arctan(pow(shape ** 2 - 1, 1 / 2)) / pow(shape ** 2 - 1, 1 / 2)) - 1)

    k1_plus = ((2 * ta ** 2 * tb * (c + 2)) / (9 * tb ** 3 * c)) * (
            (1 / (1 - h1)) + ((c - 1) / (1 + h1 * (c - 1))))
    k2_plus = ((2 * ta ** 2 * tb * (c + 2)) / (9 * ta ** 3 * c)) * (
            (1 / (1 - h2)) + ((c - 1) / (1 + h2 * (c - 1))))
    return k1_plus, k2_plus


def ellipsoid_parameter(c, c0, d, ta, tb):
    k1, a1, b1, R1 = parameter_sphere(c, c0, d, tb)
    k2, a2, b2, R2 = parameter_sphere(c, c0, d, ta)
    k1_plus, k2_plus = ellipsoid_k_plus(ta, tb, c)
    k1_ellipsoid = k1_plus * k1
    k2_ellipsoid = k2_plus * k2

    return k1_ellipsoid, a1, b1, R1, k2_ellipsoid, a2, b2, R2




def qiujie(c, c0, d, ta, tb, t):  # ta,tb是椭球两个方向上的半径长度，小于两米
    shape = ta / tb
    if shape < 1:
        h1 = (shape ** 2 / (1 - shape ** 2)) * (
                ((np.arctanh(pow(1 - shape ** 2, 1 / 2))) / pow(1 - shape ** 2, 1 / 2)) - 1)  # 轴向-对应b
        h2 = (1 / (2 * (1 - shape ** 2))) * (
                1 - (shape ** 2 * np.arctanh(pow(1 - shape ** 2, 1 / 2)) / pow(1 - shape ** 2, 1 / 2)))  # 横向-对应a
    else:
        h1 = (shape ** 2 / (shape ** 2 - 1)) * (
                1 - ((np.arctan(pow(shape ** 2 - 1, 1 / 2))) / pow(shape ** 2 - 1, 1 / 2)))
        h2 = (1 / (2 * (shape ** 2 - 1))) * (
                (shape ** 2 * np.arctan(pow(shape ** 2 - 1, 1 / 2)) / pow(shape ** 2 - 1, 1 / 2)) - 1)

    y1 = Msphere(c, c0, d, tb, t)
    y2 = Msphere(c, c0, d, ta, t)
    k1_plus = ((2 * ta ** 2 * tb * (c + 2)) / (9 * tb ** 3 * c)) * (
            (1 / (1 - h1)) + ((c - 1) / (1 + h1 * (c - 1))))
    k2_plus = ((2 * ta ** 2 * tb * (c + 2)) / (9 * ta ** 3 * c)) * (
            (1 / (1 - h2)) + ((c - 1) / (1 + h2 * (c - 1))))
    M1 = k1_plus * y1  # r1=b 半径是b  轴向响应
    M2 = k2_plus * y2  # r2=a 半径是a  径向响应

    return M1, M2  # 返回两个方向上的响应


########### 拟合 ########################


def wgn(x, snr):  # 加上高斯噪声  snr信噪比 x好像是不同时间点处的响应？
    snr1 = 10 ** (snr / 10.0)
    xpower = np.array([np.sum(x ** 2, axis=1) / x.shape[1]]).T

    npower = xpower / snr1

    return np.random.randn(x.shape[0],x.shape[1]) * np.sqrt(npower)
    # numpy.random.randn()产生服从正态分布的随机数，会出现负值。

def wgn_one(x,snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)

    npower = float(xpower / snr)

    return np.random.randn(len(x)) * np.sqrt(npower)

def wgn_one_npower(x,snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)

    npower = float(xpower / snr)

    return npower


# feature_lable_attitude=generate_data(theta=theta, gama=gama, c=c, d=d, ta=ta, tb=tb, t=t,
#                  snr=snr,material_cnt=material_cnt,R_step=R_step,T_step=T_step,A_step=A_step,
#                  sample_num=sample_num,t_num= t_num)


def func(t, k, a, b, R):  # 返回球体的响应方程，用于拟合算法，需要此方程,a,b,R其实是alpha, beta, gamma
    e = np.e
    return k * pow((1 + pow(t / a, 1 / 2)), -b) * pow(e, -t / R)



def add_attitude(theta, gama, c, d, ta, tb, t):
    fai = 0
    theta = theta * np.pi / 180
    gama = gama * np.pi / 180
    sf = np.sin(fai)
    cf = np.cos(fai)
    st = np.sin(theta)
    ct = np.cos(theta)
    sg = np.sin(gama)
    cg = np.cos(gama)

    z = np.zeros(3)

    z[0] = qiujie(c, d, ta, tb, t)[0]
    z[1] = qiujie(c, d, ta, tb, t)[0]
    z[2] = qiujie(c, d, ta, tb, t)[1]

    PT = np.diag(z)

    RT = np.zeros([3, 3])
    RT[0][0] = cg
    RT[0][1] = 0
    RT[0][2] = -sg
    RT[1][0] = st * sg
    RT[1][1] = ct
    RT[1][2] = st * cg
    RT[2][0] = ct * sg
    RT[2][1] = -st
    RT[2][2] = ct * cg

    A = np.dot(RT, PT)
    A = np.dot(A, RT.T)
    px = A[0][0] + A[0][1] + A[0][2]
    py = A[1][0] + A[1][1] + A[1][2]
    pz = A[2][0] + A[2][1] + A[2][2]
    # print('px',px,'py',py,'pz',pz)

    return px, py, pz   # x, y, z轴上三个方向上的响应





def generate_data_attitude(theta=None, gama=None, c=None, d=None, ta=None, tb=None, t=None,
                 snr=None,material_cnt=None,R_step=None,T_step=None,A_step=None,
                 sample_num=None,t_num= None):
    feature_lable = np.zeros(
        [int(3 * (360 / A_step + 1) ** 2 * (2 / R_step + 1) ** 2 * (2 / T_step + 1)), int(3 * (2 / T_step + 1)+9)])
    px=py=pz=0
    while material_cnt <= 2:
        c = attribute[material_cnt, 0]
        d = attribute[material_cnt, 2]
        while theta <= 180:
            while gama <=180:
                while ta <= 2+R_step:
                    while tb <= 2+R_step:
                        while t<= 0.2:
                            px,py,pz=add_attitude(theta, gama, c, d, ta, tb, t)
                            px=wgn(px,snr)
                            py=wgn(py,snr)
                            pz=wgn(pz,snr)
                            feature_lable[sample_num, 3 * (t_num - 1)]=px
                            feature_lable[sample_num, 3 * (t_num - 1) + 1]=py
                            feature_lable[sample_num, 3 * (t_num - 1) + 2]=pz
                            t_num+=1
                            t+=T_step
                        feature_lable[sample_num, 7] =material_cnt
                        if ta>=tb:
                            feature_lable[sample_num, 8] = 0
                        else:
                            feature_lable[sample_num, 8] = 1
                        feature_lable[sample_num, 0] = theta
                        feature_lable[sample_num, 1] = gama
                        feature_lable[sample_num, 2] = c
                        feature_lable[sample_num, 3] = d
                        feature_lable[sample_num, 4] = snr
                        feature_lable[sample_num, 5] = ta
                        feature_lable[sample_num, 6] = tb
                        sample_num+=1
                        tb+=R_step
                    ta+=R_step
                gama+=A_step
            theta+=A_step
        material_cnt+=1
    return feature_lable


def plot_data(M1=None,M2=None,M1_without_noise=None,M2_without_noise=None,t=None,
              SNR=None,material=None,ta=None,tb=None,dir_save=None):
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-8,1e1)
    plt.plot(t, M1_without_noise,'--', color="r", label="M1_noiseless")
    plt.plot(t, M2_without_noise, '--', color="b", label="M2_noiseless")
    plt.plot(t, M1, 'o', color="y", label="M1")
    plt.plot(t, M2, 'o', color="g", label="M2")
    plt.xlabel("t /s")
    plt.ylabel("M")
    plt.title(str(material)+" ta="+"%.2f"%ta+" tb="+"%.2f"%tb+" SNR="+str(SNR)+"dB")
    plt.savefig(dir_save+str(SNR)+'dB.png')
    plt.show()
    plt.close()


def generate_data(snr=None,R_step=None,t_split=None):
    lable = []
    feature = []
    material_cnt=0
    t = np.array(10 ** (np.linspace(-8, 0, t_split)))
    sample_num=0
    plot_flag=0
    material_list=['steel','Ni','Al']
    while material_cnt <= 2:
        c = attribute[material_cnt, 0]
        c0 = attribute[material_cnt, 1]
        d = attribute[material_cnt, 2]
        ta=0.01
        while ta <= 1.5:
            tb=0.01
            while tb <= 1.5:
                if ta!=tb:
                    if snr==None:
                        k1_ellipsoid, a1, b1, R1, k2_ellipsoid, a2, b2, R2 = ellipsoid_parameter(c, c0, d, ta, tb)
                        M1_without_noise = np.array(list(map(partial(func, k=k1_ellipsoid, a=a1, b=b1, R=R1), t)))
                        M2_without_noise = np.array(list(map(partial(func, k=k2_ellipsoid, a=a2, b=b2, R=R2), t)))
                        M=np.hstack((M1_without_noise,M2_without_noise))
                        # 保存响应曲线的数据
                        if material_cnt==0:
                            if (ta==0.04)&(tb==0.08):
                                pd.DataFrame([M1_without_noise,M2_without_noise,t],index=['M1','M2','t']).to_csv('./response_curve.csv')
                                pd.DataFrame([[k1_ellipsoid, a1, b1, R1, k2_ellipsoid, a2, b2, R2]],columns=['k1_ellipsoid', 'a1',
                                                                                                   'b1', 'R1', 'k2_ellipsoid', 'a2', 'b2', 'R2']).to_csv('./response_parameter.csv')
                    else:
                        k1_ellipsoid, a1, b1, R1, k2_ellipsoid, a2, b2, R2 = ellipsoid_parameter(c, c0, d, ta, tb)
                        M1_without_noise = np.array(list(map(partial(func, k=k1_ellipsoid, a=a1, b=b1, R=R1), t)))
                        M2_without_noise = np.array(list(map(partial(func, k=k2_ellipsoid, a=a2, b=b2, R=R2), t)))
                        M_noise = np.hstack((M1_without_noise, M2_without_noise))
                        noise_power = wgn_one_npower(x=M_noise, snr=snr)
                        M1 = M1_without_noise + np.random.randn(len(M1_without_noise)) * np.sqrt(noise_power)
                        M2 = M2_without_noise + np.random.randn(len(M2_without_noise)) * np.sqrt(noise_power)
                        M=np.hstack((M1,M2))
                        # 输出第400个样本的分布情况
                        plot_flag+=1
                        if plot_flag==400:
                            os.makedirs("./sample selected",exist_ok=True)
                            dir_selected='./sample selected/'
                            plot_data(M1=M1,M1_without_noise=M1_without_noise,M2=M2,M2_without_noise=M2_without_noise,
                                      t=t,SNR=snr,material=material_list[material_cnt],ta=ta,tb=tb,dir_save=dir_selected)
                            data_collect=np.vstack((M1,M2,M1_without_noise,M2_without_noise,t))
                            print(data_collect)
                            data_collect=pd.DataFrame(data_collect,index=['M1','M2','M1_WITHOUT','M2_WITHOUT','t'])
                            print(data_collect)
                            data_collect.to_csv(dir_selected+'SNR='+str(snr)+'dB.csv')

                    material_flag = material_cnt
                    if ta > tb:
                        shape_flag = 0 # 径向大于轴向，扁椭球体
                    else:
                        shape_flag = 1 # 径向小于轴向，长椭球体
                    if sample_num == 0:
                        feature= M
                        inv_parameter=[k1_ellipsoid, a1, b1, R1, k2_ellipsoid, a2, b2, R2]
                        lable= [[material_flag,shape_flag]]

                    else:
                        feature=np.vstack((feature,M))
                        inv_parameter=np.vstack((inv_parameter, [k1_ellipsoid, a1, b1, R1, k2_ellipsoid, a2, b2, R2]))
                        lable=np.vstack((lable,[[material_flag,shape_flag]]))
                    sample_num += 1
                    print("这是第%d个样本"%sample_num,material_flag,shape_flag,snr)
                tb += R_step
            ta += R_step
        material_cnt += 1

    feature_lable=np.hstack((lable, feature))
    return feature_lable, inv_parameter


os.makedirs('generate data', exist_ok=True)
R_step=0.1

# 时域中基于模型的方法还需要保留每个样本的k1,k2,alpha1,
# alpha2,beta1,beta2,gamma1,gamma2值
data_sets=[[None, 5, 10, 15, 20, 25, 30],
           ['snr00withoutnoise', 'snr05', 'snr10', 'snr15', 'snr20', 'snr25', 'snr30']]
for i in range(len(data_sets[1])):
    feature_lable_withoutnoise, inv_parameter=generate_data(snr=data_sets[0][i], R_step=R_step,t_split=200)
    feature_lable_withoutnoise=pd.DataFrame(feature_lable_withoutnoise)
    feature_lable_withoutnoise.to_csv('./generate data/'+data_sets[1][i]+'.csv')
    pd.DataFrame(inv_parameter).to_csv("./generate data/"+data_sets[1][i]+"_inv_para.csv")






