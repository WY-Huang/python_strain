"""
平板一条直线上测点的弯曲应变计算，采用SG滤波算法

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.signal import savgol_filter


def read_data(data_path=None, sample_num=25, plate_length=96):
    """
    读取位移数据,统一位移和x坐标的单位为mm
    """
    x_coor = np.arange(0, sample_num).reshape(sample_num, 1) * (plate_length / (sample_num - 1))

    dis_data = np.loadtxt(data_path)
    dis_data_mm = dis_data[100, 1:] / 1000

    return x_coor, dis_data_mm


def func_fit(x, dis, M=4):
    """
    最小二乘法、多项式、拟合位移数据
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


def f_fit(x_coord):
    """
    拟合函数
    """
    f = co_w[4] * x_coord ** 4 + co_w[3] * x_coord ** 3 + co_w[2] * x_coord ** 2 + co_w[1] * x_coord + co_w[0]

    return f


def sg_filter(y_noise, win_size=None, poly_order=None, deriv=0, delta=1):
    """
    对位移数据进行滤波处理
    """
    yhat = savgol_filter(y_noise, win_size, poly_order)    # window size 11, polynomial order 3

    yhat_2_deri = savgol_filter(y_noise, win_size, poly_order, deriv, delta=delta)   # 方法3

    return yhat, yhat_2_deri


def strain_calc(x, func_dis):
    """
    根据整个数据最小二乘法拟合后的位移方程的2阶导数计算应变
    """
    func_1_deri = np.polyder(func_dis, 1)
    first_deri = []
    func_2_deri = np.polyder(func_dis, 2)
    second_deri = []
    for _, x_value in enumerate(x):
        first_value = func_1_deri(x_value)
        first_deri.append(first_value)

        # second_value = derivative(f_fit, x_value, dx=1e-6, n=2)   # 方法1
        second_value = func_2_deri(x_value)                         # 方法2
        second_deri.append(second_value)

    strain = np.array(second_deri) * 0.25

    return first_deri, second_deri, strain


def dynamic_visualization(x_point, data_all, figure_num=1, ylabel="displacement [mm]", line_num=1000):
    """
    绘制所有测点在同一时刻下的位移/应变曲线,动态显示
    """
    data_all = np.array(data_all)
    plt.ion()
    plt.figure()
    print("start plot：", ylabel)

    for row in range(line_num):
        y_dis = data_all[row]

        # plt.clf()
        plt.cla()
        plt.plot(x_point, y_dis)
        plt.plot(x_point, np.zeros_like(x_point))

        plt.xlabel("x_position")
        plt.ylabel(ylabel)
        # plt.draw()
        if row != line_num-1:
            plt.pause(0.001)  # 显示秒数
        else:
            plt.pause(0)

    plt.ioff()  # 关闭interactive mode
    # plt.show()
    plt.close("all")


if __name__ == "__main__":
    # 读取数据
    plate_length = 110
    sample_num = 55
    x_coor, dis_data_mm = read_data('test_2022/dis_data_all.txt', sample_num, plate_length)   # dis_data_20230223/dis_data_all.txt

    # 仅绘制一张图
    only_one = 1
    if only_one:
        # max_dis_index = np.unravel_index(dis_data_mm.argmax(), dis_data_mm.shape)   # 最大值索引
        # print("最大位移的位置索引及值：", max_dis_index, "\t", dis_data_mm[max_dis_index])
        # dis_data_one = dis_data_mm[max_dis_index[0]]
        dis_data_one = dis_data_mm[:55]
        # 绘制原始位移散点及拟合后的位移曲线
        plt.figure(1)
        plt.plot(x_coor, dis_data_one, 'bo', label="dis_noise")

        co_w, func, y_estimate_lstsq = func_fit(x_coor, dis_data_one)

        plt.plot(x_coor, y_estimate_lstsq, 'r', lw=2.0, label="lstsq")

        # 绘制sg滤波后的数据及2阶导数
        interval_delta = plate_length / (sample_num - 1)
        dis_sg, sid_sg_deri = sg_filter(dis_data_one, 21, 3, 2, interval_delta)
        plt.plot(x_coor, dis_sg, 'y', lw=2.0, label="dis_sg")
        # plt.plot(x_coor, sid_sg_deri, 'p', lw=2.0, label="sid_sg_deri")

        plt.legend()
        plt.xlabel("x_coordinate [mm]")
        plt.ylabel("y_displacement [mm]")

        # 绘制一/二阶导数曲线及应变曲线
        first_deri, second_deri, strain = strain_calc(x_coor, func)
        # print(strain * 1e6)

        plt.figure(2)
        # plt.plot(x_coor, first_deri, 'b', lw=2.0, label="first_deri")
        # plt.plot(x_coor, np.array(second_deri) * 1e6, 'g', lw=2.0, label="second_deri")
        plt.plot(x_coor, strain * 1e6, 'r', lw=2.0, label="strain_lstsq")
        plt.plot(x_coor, sid_sg_deri * 0.25 * 1e6, 'y', lw=2.0, label="strain_sg")
        plt.plot(x_coor, np.zeros_like(x_coor), 'b', label="y = 0")

        plt.legend()
        plt.xlabel("x_coordinate [mm]")
        plt.ylabel("y_strain [uɛ]")
        plt.show()

    # 动态显示数据
    else:
        # 按时间显示原始位移数据
        dynamic_visualization(x_coor, dis_data_mm)

        # 按时间显示应变数据
        dis_row, dis_col = dis_data_mm.shape
        dis_fit_lstsq = []
        dis_fit_sg = []
        second_deri_sg = []
        strain_lstsq = []

        for dis_i, dis_data in enumerate(dis_data_mm):
            # fit method1
            co_w, func, y_estimate_lstsq = func_fit(x_coor, dis_data)   # 最小二乘拟合一行数据（一条线上的所有测点）
            dis_fit_lstsq.append(y_estimate_lstsq)
            # fit method2
            dis_sg, dis_sg_deri = sg_filter(dis_data, 21, 3, 2)
            dis_fit_sg.append(dis_sg)
            second_deri_sg.append(dis_sg_deri)
            # strain calculate
            first_deri, second_deri, strain = strain_calc(x_coor, func)
            strain_lstsq.append(strain)

        dynamic_visualization(x_coor, dis_fit_lstsq)  # 绘制lstsq拟合后的位移数据

        dynamic_visualization(x_coor, strain_lstsq, ylabel="strain [uɛ]")  # 绘制lstsq拟合后的应变数据



