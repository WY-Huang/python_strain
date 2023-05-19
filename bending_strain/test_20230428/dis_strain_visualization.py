"""
二维弯曲应变的3D可视化
"""

import os

from tqdm import tqdm
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def visualization_3d(xc, yc, zc, xraw=None, yraw=None, zraw=[], 
                    flag=None, times=[], z_label='Z Position [um]'):
    """
    位移数据随时间变化的三维可视化（时域ODS）
    """
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection='3d')

    # cax = ax_3d.inset_axes([1.03, 0, 0.1, 1], transform=ax_3d.transAxes) # Colorbar is held by `cax`.
    # plt.ion()

    xs, ys = xc, yc
    # for idx in range(len(times)):
    for idx in range(1):
        title = f"Num.{idx} Time:[{times[idx]:f} s]"
        # zs = zc[idx]
        # zraw_i = zraw[idx]
        zs = zc.reshape((1, -1))[0]

        ax_3d.cla()
        # Plot a trisurf
        if flag == "trisurf":
            fig_tri = ax_3d.plot_trisurf(xs, ys, zs, cmap='gist_rainbow_r')

        # Plot a scatter
        elif flag == "scatter":
            ax_3d.scatter(xs, ys, zs, marker="o", color="y")
            # if xraw != None:
                # ax_3d.scatter(xraw, yraw, zraw_i, marker="^", color="r")

        # Plot a basic wireframe.
        elif flag == "wireframe":
            # Z = zs.reshape(num, num)
            # Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
            X, Y = np.meshgrid(xs, ys)
            Z = zc

            ax_3d.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

        # Plot the surface.
        elif flag == "surface":
            X, Y = np.meshgrid(xs, ys)
            Z = zc
            ax_3d.plot_surface(X, Y, Z, cmap=cm.autumn, linewidth=0, antialiased=False)

        ax_3d.set_title(title)

        ax_3d.set_xlabel('X Position [mm]')
        ax_3d.set_ylabel('Y Position [mm]')
        ax_3d.set_zlabel(z_label, labelpad=10)
        ax_3d.set_box_aspect([fig_y * x_y_ratio, fig_y, fig_y])
        # ax_3d.set_zlim(-1, 2)
        plt.colorbar(fig_tri)
        # cax.clear()
        # plt.colorbar(fig_tri, cax=cax)

        # plt.show()
        plt.pause(0.5)


def get_x_y_real(x_c, y_c):
    """
    获取真实的xyz值
    """
    x_c_r = (x_c - x_c.min()) / (x_c.max() - x_c.min()) * 40
    y_c_r = (y_c - y_c.min()) / (y_c.max() - y_c.min()) * 70

    return x_c_r, y_c_r


def np_move_avg(data, win_size, mode="valid"):
    """
    滑动平均滤波算法
    """
    data_p = data.copy()
    crop_data = np.convolve(data_p, np.ones(win_size) / win_size, mode=mode)
    insert_length = win_size // 2
    data_length = len(data_p)
    data_p[insert_length:data_length-insert_length] = crop_data
    return data_p


def sg_filter(y_noise, win_size=11, poly_order=3, deriv=0, delta=1):
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


def func_fit(x, dis, M=3):
    """
    方法一：最小二乘法、多项式拟合单行位移数据，曲线过于平滑，会丢失局部特征，误差随阶次增加。
         多项式次数从3到M，返回拟合误差最小的拟合系数及方程。
    param:
        x: x方向坐标（mm）
        dis: 面外位移z（mm）
        M: 最大多项式阶次
    return:
        co_w: 拟合后多项式系数
        resl: 拟合残差
        func: 拟合后位移方程
        y_estimate_lstsq: 拟合后测点面外位移z
    """
    co_w_list = []
    resl_list = []
    y_estimate_list = []
    for M_i in range(3, M+1):

        X = x.copy()
        for i in range(2, M_i+1):
            X = np.column_stack((X, pow(x, i)))

        X = np.insert(X, 0, [1], 1)

        co_w, resl, _, _ = np.linalg.lstsq(X, dis, rcond=None)  # resl为残差的平方和
        y_estimate_lstsq = X.dot(co_w)
        # print("co_w:", co_w[::-1], "\nresl", resl, "\ny_estimate_lstsq", y_estimate_lstsq)

        co_w_list.append(co_w)
        resl_list.append(resl)
        y_estimate_list.append(y_estimate_lstsq)
        
    min_index = resl_list.index(min(resl_list))

    func = np.poly1d(co_w_list[min_index][::-1])                            # 构建多项式

    return co_w_list[min_index], func, y_estimate_list[min_index], resl_list


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

    # ================================================================================
    # （1）原始位移及坐标数据读取显示
    # ================================================================================
    # 初始数据
    plate_length = 10.0     # 单行测点实际总长度（mm），实际长度46.2mm
    plate_thickness = 3     # 板的中性面到表面的厚度（mm），实际厚度3.42mm
    sample_num = 11         # 单行测点数量
    column = 6              # 需拟合的列数
    point_all = sample_num * column

    # 读取原始坐标点(x, y)数据
    coor_data = np.loadtxt("/home/wanyel/vs_code/python_strain/bending_strain/data_test/20230425/time_filter/point.txt")  # bending_strain/
    x_coor, y_coor = coor_data[:, 1], coor_data[:, 2]
    print("x_coor size: ", x_coor.shape, "\ny_coor size: ", y_coor.shape)

    # 真实坐标xy
    x_real, y_real = get_x_y_real(x_coor, y_coor)
    x_coor_real = x_real.reshape((-1, 1))
    y_coor_real = y_real.reshape((-1, 1))
    xy_save = False
    if xy_save:
        x_coor_temp = np.zeros((column, sample_num))
        y_coor_temp = np.zeros((column, sample_num))
        for idx in range(column):
            x_coor_temp[idx] = x_real[idx:point_all:column]
            y_coor_temp[idx] = y_real[idx:point_all:column]
        xy_coor_real = np.hstack((x_coor_temp.reshape((-1, 1)), y_coor_temp.reshape((-1, 1))))
        np.savetxt("/home/wanyel/vs_code/python_strain/bending_strain/data_test/20230425/xy_coor_real.txt", xy_coor_real)
        print("xy_coor_deleted shape: ", xy_coor_real.shape)

    fig_y = 2
    x_y_ratio = (x_coor.max() - x_coor.min()) / (y_coor.max() - y_coor.min())
    figsize = (fig_y * x_y_ratio, fig_y)

    # 读取所有原始坐标点的随时间变化的位移数据
    dis_data = np.loadtxt("/home/wanyel/vs_code/python_strain/bending_strain/data_test/20230425/time_filter/dis_merge_20230425.txt")
    data_shape = dis_data.shape
    dis_data_pure = dis_data[:, 1:]
    print("Out of plate displacement data size: ", data_shape)

    # 绘制原始xy坐标位置图
    show_xy_position = 0
    if show_xy_position:

        plt.figure("x_y_coordinate", figsize=figsize)
        plt.plot(x_coor, y_coor, marker='.', linestyle='')
        plt.pause(2)

        # 查看坐标点绘制顺序
        show_xy_order = 1
        if show_xy_order:
            for coor_index, coor_xy in enumerate(coor_data):
                plt.plot(coor_xy[1], coor_xy[2], marker='o', linestyle='')
                plt.pause(0.01)

        plt.show()

    # 绘制原始位移散点图
    show_dynamic_dis = 0
    if show_dynamic_dis:
        dis_time = dis_data[:, 0]
        visualization_3d(x_coor, y_coor, dis_data_pure, flag="scatter", times=dis_time)
        plt.show()

    # ================================================================================
    # （2）所有时刻位移数据拟合及应变计算（逐行拟合位移数据、并求二阶导数计算应变）
    # ================================================================================

    local_load = 1
    if local_load:
        xy_coor_re_arr = np.loadtxt("/home/wanyel/vs_code/python_strain/bending_strain/data_test/20230425/xy_coor_real.txt")
        x_coor_d, y_coor_d = xy_coor_re_arr[:, 0], xy_coor_re_arr[:, 1]

        y_estimate_lstsq_all = np.loadtxt("/home/wanyel/vs_code/python_strain/bending_strain/data_test/20230425/y_estimate_lstsq_185.txt")
        # dis_fit_sg_all = np.loadtxt("bending_strain/test_2022/dis_fit_sg.txt")
        strain_lstsq_all = np.loadtxt("/home/wanyel/vs_code/python_strain/bending_strain/data_test/20230425/strain_lstsq_185_copy.txt")
        # second_deri_sg_all = np.loadtxt("bending_strain/test_2022/strain_lstsq.txt")
    else:

        y_estimate_lstsq_all = np.zeros((5000, point_all))
        strain_lstsq_all = np.zeros((5000, point_all))
        row_i = 0
        for dis_point in tqdm(dis_data_pure):
            # 位移数据拟合
            y_estimate_lstsq_one = np.zeros((column, sample_num))
            strain_lstsq_one = np.zeros((column, sample_num))
            for col_i in range(column):
                x_coor_col = x_real[col_i:point_all:column]
                y_coor_col = y_real[col_i:point_all:column]
                dis_data_one = dis_point[col_i:point_all:column]
                # print("dis_data_one: ", dis_data_one.shape)

                # 滤波处理
                # dis_data_filter = np_move_avg(dis_data_one, 11)     # 1）滑动平均

                interval_delta = plate_length / (sample_num - 1)
                win_size = sample_num // 5                          # 滑动窗口选取为测点数的1/5，且为奇数
                if win_size % 2 == 0:
                    win_size += 1
                if win_size <= 11:
                    win_size = 11

                dis_sg, sid_sg_deri = sg_filter(dis_data_one, win_size, 3, 2, interval_delta) # 2）绘制sg滤波后的数据及2阶导数

                dis_data_filter = np_move_avg(dis_sg, 11)     # 1）滑动平均

                co_w, func, y_estimate_lstsq, resl_lists = func_fit(y_coor_col, dis_data_filter, 5)   # 单行全部位移数据最小二乘拟合
                first_deri, second_deri, strain_lstsq = strain_calc(y_coor_col, func, plate_thickness)
                # strain_sg = sid_sg_deri * plate_thickness * 0.5 * (-1) * 1e6

                y_estimate_lstsq_one[col_i] = y_estimate_lstsq
                strain_lstsq_one[col_i] = strain_lstsq

            # np.savetxt("/home/wanyel/vs_code/python_strain/bending_strain/data_test/20230425/y_estimate_lstsq_185.txt", y_estimate_lstsq_one)
            # np.savetxt("/home/wanyel/vs_code/python_strain/bending_strain/data_test/20230425/strain_lstsq_185.txt", strain_lstsq_one)

            y_estimate_lstsq_all[row_i] = y_estimate_lstsq_one.reshape((1,-1))
            strain_lstsq_all[row_i] = strain_lstsq_one.reshape((1,-1))
            row_i += 1

        # 保存拟合后的位移数据及应变数据
        save_flag = 1
        if save_flag:

            np.savetxt("/home/wanyel/vs_code/python_strain/bending_strain/data_test/20230425/y_estimate_lstsq_all.txt", y_estimate_lstsq_all)
            np.savetxt("/home/wanyel/vs_code/python_strain/bending_strain/data_test/20230425/strain_lstsq_all.txt", strain_lstsq_all)

    # 绘制拟合后的位移的散点图、应变图（三维显示）
    show_dynamic_dis_fit = 1
    if show_dynamic_dis_fit:
        dis_time = dis_data[:, 0]

        # method 1
        # visualization_3d(x_coor_d, y_coor_d, dis_fit_lstsq_all, x_coor, y_coor, dis_data_pure, "scatter", dis_time)
        
        # visualization_3d(x_coor_d, y_coor_d, y_estimate_lstsq_all, flag="trisurf", times=dis_time)
        # visualization_3d(x_coor_d, y_coor_d, strain_lstsq_all, flag="trisurf", times=dis_time, z_label="Strain [uɛ]")

        stress_lstsq_all = strain_lstsq_all * 71 * 1e-3     # 单位Mpa
        visualization_3d(x_coor_d, y_coor_d, stress_lstsq_all, flag="trisurf", times=dis_time, z_label="Stress [Mpa]")
        plt.show()