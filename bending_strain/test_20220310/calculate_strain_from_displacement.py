"""
平板结构单行测点的弯曲应变计算，采用Savitzky-Golay滤波算法或全局最小二乘法

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def read_data(data_path=None, sample_num=25, plate_length=96):
    """
    读取位移数据,统一位移和x坐标的单位为mm。

    return:
        x_coor: x方向坐标（mm）
        dis_data_mm: 面外位移z（mm）
    """
    x_coor = np.arange(sample_num).reshape(-1, 1) * (plate_length / (sample_num - 1))

    dis_data = np.loadtxt(data_path)
    print("dis_data: ", dis_data.shape)
    dis_data_mm = dis_data[1:]

    return x_coor, dis_data_mm


def func_fit(x, dis, M=4):
    """
    方法一：最小二乘法、多项式拟合单行位移数据，曲线过于平滑，会丢失局部特征，误差随阶次增加。

    param:
        x: x方向坐标（mm）
        dis: 面外位移z（mm）
        M: 多项式阶次
    return:
        co_w: 拟合后多项式系数
        func: 拟合后位移方程
        y_estimate_lstsq: 拟合后测点面外位移z
    """
    X = x
    for i in range(2, M+1):
        X = np.column_stack((X, pow(x, i)))

    X = np.insert(X, 0, [1], 1)

    co_w, resl, _, _ = np.linalg.lstsq(X, dis, rcond=None)  # resl为残差的平方和
    # print("co_w:", co_w, "\nresl", resl)
    # print(co_w[::-1])
    func = np.poly1d(co_w[::-1])    # 构建多项式

    y_estimate_lstsq = X.dot(co_w)

    return co_w, func, y_estimate_lstsq



def sg_filter(y_noise, win_size=None, poly_order=None, deriv=0, delta=1):
    """
    方法二：对位移数据进行滑动滤波处理，可保留局部特征。

    param:
        y_noise: 原始面外位移z（mm）
    return:
        yhat: 拟合后测点面外位移z（mm）
        yhat_2_deri: 拟合后测点二阶导数

    """
    yhat = savgol_filter(y_noise, win_size, poly_order)    # window size 11, polynomial order 3

    yhat_2_deri = savgol_filter(y_noise, win_size, poly_order, deriv, delta=delta)   # 计算位移对位置的二阶导数

    return yhat, yhat_2_deri


def strain_calc(x, func_dis, palte_thick):
    """
    根据整个数据最小二乘法拟合后的位移方程的2阶导数计算应变
    param:
        x: x轴坐标点（mm）
        func_dis: 拟合后的位移函数
        palte_thick: 平板厚度（mm）
    return:
        first_deri: 一阶导数
        second_deri: 二阶导数
        strain: 应变（ɛ）
    """
    func_1_deri = np.polyder(func_dis, 1)
    first_deri = []
    func_2_deri = np.polyder(func_dis, 2)
    second_deri = []
    for _, x_value in enumerate(x):
        first_value = func_1_deri(x_value)
        first_deri.append(first_value)

        second_value = func_2_deri(x_value)     # 二阶导数计算
        second_deri.append(second_value)

    strain = np.array(second_deri) * palte_thick

    return first_deri, second_deri, strain


if __name__ == "__main__":
    # 读取数据
    plate_length = 110      # 单行测点实际总长度（mm）
    plate_thickness = 0.5   # 板的中性面到表面的厚度（mm）
    sample_num = 31         # 单行测点数量
    x_coor, dis_data_mm = read_data('bending_strain/test_20220310/test_dis_data.txt', sample_num, plate_length)

    # 仅绘制第一行数据
    only_one = 1
    if only_one:
        # max_dis_index = np.unravel_index(dis_data_mm.argmax(), dis_data_mm.shape)   # 最大值索引
        # print("最大位移的位置索引及值：", max_dis_index, "\t", dis_data_mm[max_dis_index])
        # dis_data_one = dis_data_mm[max_dis_index[0]]
        dis_data_one = dis_data_mm[:31]
        # 绘制原始位移散点及拟合后的位移曲线
        plt.figure(1)
        plt.plot(x_coor, dis_data_one, 'bo', label="dis_noise")

        co_w, func, y_estimate_lstsq = func_fit(x_coor, dis_data_one)   # 单行全部位移数据最小二乘拟合

        plt.plot(x_coor, y_estimate_lstsq, 'r', lw=2.0, label="dis_lstsq")

        # 绘制sg滤波后的数据及2阶导数
        interval_delta = plate_length / (sample_num - 1)
        win_size = sample_num // 5      # 滑动窗口选取为测点数的1/5，且为奇数
        if win_size % 2 == 0:
            win_size += 1
        dis_sg, sid_sg_deri = sg_filter(dis_data_one, 5, 3, 2, interval_delta) # 
        plt.plot(x_coor, dis_sg, 'y', lw=2.0, label="dis_sg")
        # plt.plot(x_coor, sid_sg_deri, 'p', lw=2.0, label="sid_sg_deri")

        plt.legend()
        plt.xlabel("x_coordinate [mm]")
        plt.ylabel("y_displacement [mm]")

        # 绘制一/二阶导数曲线及应变曲线
        first_deri, second_deri, strain = strain_calc(x_coor, func, plate_thickness)
        strain_lstsq = strain * 1e6

        strain_sg = sid_sg_deri * plate_thickness * 1e6

        plt.figure(2)
        # plt.plot(x_coor, first_deri, 'b', lw=2.0, label="first_deri")
        # plt.plot(x_coor, np.array(second_deri) * 1e6, 'g', lw=2.0, label="second_deri")
        plt.plot(x_coor, strain_lstsq, 'r', lw=2.0, label="strain_lstsq")
        plt.plot(x_coor, strain_sg, 'y', lw=2.0, label="strain_sg")
        plt.plot(x_coor, np.zeros_like(x_coor), 'b', label="y = 0")

        plt.legend()
        plt.xlabel("x_coordinate [mm]")
        plt.ylabel("y_strain [uɛ]")
        

        # 绘制应变片数据
        strain_gage = np.loadtxt("bending_strain/20230315/strain_data_0315.txt")
        plt.figure(3)
        plt.plot(strain_gage[:, 0], strain_gage[:, 1], 'b', lw=2.0, label="strain_gage")
        plt.show()