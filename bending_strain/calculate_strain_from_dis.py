import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.signal import savgol_filter


def read_data(data_path=None, sample_num=25, plate_length=96):
    """
    读取位移数据,统一位移和x坐标的单位为mm
    """
    x_coor = np.arange(0, sample_num).reshape(sample_num, 1) * (plate_length / sample_num)

    dis_data = np.loadtxt(data_path)
    dis_data_mm = dis_data[10, 3::2] / 1000

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


def sg_filter(y_noise, win_size=None, poly_order=None, deriv=0):
    """
    对位移数据进行滤波处理
    """
    yhat = savgol_filter(y_noise, win_size, poly_order)    # window size 11, polynomial order 3

    yhat_2_deri = savgol_filter(y_noise, win_size, poly_order, deriv)   # 方法3

    return yhat, yhat_2_deri


def strain_calc(x, func_dis):
    """
    根据拟合后的位移方程的2阶导数计算应变
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


if __name__ == "__main__":
    x_coor, dis_data_mm = read_data('bending_strain/dis_data_20230223/dis_data_all.txt', 55, 110)

    # 绘制原始位移散点及拟合后的位移曲线
    plt.figure(1)
    plt.plot(x_coor, dis_data_mm, 'bo', label="dis_noise")

    co_w, func_fit, y_estimate_lstsq = func_fit(x_coor, dis_data_mm)

    plt.plot(x_coor, y_estimate_lstsq, 'r', lw=2.0, label="lstsq")

    # 绘制sg滤波后的数据及2阶导数
    dis_sg, sid_sg_deri = sg_filter(dis_data_mm, 21, 3, 2)
    plt.plot(x_coor, dis_sg, 'y', lw=2.0, label="dis_sg")
    # plt.plot(x_coor, sid_sg_deri, 'p', lw=2.0, label="sid_sg_deri")

    plt.legend()
    plt.xlabel("x_coordinate [mm]")
    plt.ylabel("y_displacement [mm]")

    # 绘制一/二阶导数曲线及应变曲线
    first_deri, second_deri, strain = strain_calc(x_coor, func_fit)
    # print(strain * 1e6)

    plt.figure(2)
    # plt.plot(x_coor, first_deri, 'b', lw=2.0, label="first_deri")
    plt.plot(x_coor, np.array(second_deri) * 1e6, 'g', lw=2.0, label="second_deri")
    plt.plot(x_coor, strain * 1e6, 'y', lw=2.0, label="strain")

    plt.legend()
    plt.xlabel("x_coordinate [mm]")
    plt.ylabel("y_strain [uɛ]")
    plt.show()


