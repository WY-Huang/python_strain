"""
Julight一维/二维应变数据验证
"""


import numpy as np
import matplotlib.pylab as plt


def func_fit(x, dis, M=3):
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
        strain: 应变（uɛ）
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

    strain = np.array(second_deri) * palte_thick * 0.5 * (-1) * 1e6

    return first_deri, second_deri, strain


if __name__ == "__main__":
    raw_data = np.loadtxt("/home/wanyel/vs_code/python_strain/bending_strain/verifacation/plate_point_dis.txt")
    print(raw_data.shape)
    pixel_x = 500 / (1220 - 120)
    x_p = raw_data[:, 0]
    x = (x_p - 120) * pixel_x

    pixel_y = 600 / (394 - 44)
    y_p = raw_data[:, 1]
    y = (220 - y_p) * pixel_y / 1e3

    plt.figure("Displacement")
    plt.plot(x, y, 'y', lw=2.0, label="dis_raw")
    plt.xlabel("x_position [mm]")
    plt.ylabel("y_displacement [mm]")

    co_w, func, y_estimate_lstsq = func_fit(x, y, M=6)   # 单行全部位移数据最小二乘拟合
    first_deri, second_deri, strain_lstsq = strain_calc(x, func, 5)

    plt.figure("Strain")
    # plt.plot(x, y_estimate_lstsq, 'r', lw=2.0, label="dis_lstsq")
    # plt.plot(x, first_deri, 'k', lw=2.0, label="first_deri")
    plt.plot(x, strain_lstsq, 'b', lw=2.0, label="strain_lstsq")

    plt.xlabel("x_position [mm]")
    plt.ylabel("y_strain [uɛ]")

    plt.legend()
    plt.show()