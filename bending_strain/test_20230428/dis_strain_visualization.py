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
    for idx in range(len(times)):
        title = f"Num.{idx} Time:[{times[idx]:f} s]"
        zs = zc[idx]
        # zraw_i = zraw[idx]

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
        ax_3d.set_zlabel(z_label)
        ax_3d.set_box_aspect([fig_y * x_y_ratio, fig_y, fig_y])
        # ax_3d.set_zlim(-1, 2)
        # plt.colorbar(fig_tri)
        # cax.clear()
        # plt.colorbar(fig_tri, cax=cax)

        # plt.show()
        plt.pause(0.1)


def get_x_y_z_data(x_c, y_c, dis_point):
    """
    获取真实的xyz值
    """
    x_c_r = (x_c - x_c.min()) / (x_c.max() - x_c.min()) * 40
    y_c_r = (y_c - y_c.min()) / (y_c.max() - y_c.min()) * 10


if __name__ == "__main__":

    # ================================================================================
    # （1）原始位移及坐标数据读取显示
    # ================================================================================

    # 读取原始坐标点(x, y)数据
    coor_data = np.loadtxt("/home/wanyel/vs_code/python_strain/bending_strain/test_20230428/data/point.txt")  # bending_strain/
    x_coor, y_coor = coor_data[:, 1], coor_data[:, 2]
    print("x_coor size: ", x_coor.shape, "\ny_coor size: ", y_coor.shape)

    fig_y = 2
    x_y_ratio = (x_coor.max() - x_coor.min()) / (y_coor.max() - y_coor.min())
    figsize = (fig_y * x_y_ratio, fig_y)

    # 读取所有原始坐标点的随时间变化的位移数据
    dis_data = np.loadtxt("/home/wanyel/vs_code/python_strain/bending_strain/test_20230428/data/dis_merge_20230426.txt")
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

    local_load = 0
    if local_load:
        xy_coor_deleted = np.loadtxt("bending_strain/test_2022/xy_coor_deleted.txt")
        x_coor_d, y_coor_d = xy_coor_deleted[:, 0], xy_coor_deleted[:, 1]

        dis_fit_lstsq_all = np.loadtxt("bending_strain/test_2022/dis_fit_lstsq.txt")
        dis_fit_sg_all = np.loadtxt("bending_strain/test_2022/dis_fit_sg.txt")
        strain_lstsq_all = np.loadtxt("bending_strain/test_2022/second_deri_sg.txt")
        second_deri_sg_all = np.loadtxt("bending_strain/test_2022/strain_lstsq.txt")
    else:
        data_dict_all = {"dis_fit_lstsq": [], "dis_fit_sg": [],
                         "second_deri_sg": [], "strain_lstsq": []}
        xy_save = True
        for dis_point in tqdm(dis_data_pure):

            # 解析某一时刻的位移数据z及坐标xy
            x_list, y_list, z_list = get_x_y_z_data(x_coor, y_coor, dis_point)
            if xy_save:
                x_coor_deleted = np.hstack(x_list).reshape((-1, 1))
                y_coor_deleted = np.hstack(y_list).reshape((-1, 1))
                xy_coor_deleted = np.hstack((x_coor_deleted, y_coor_deleted))
                np.savetxt("bending_strain/test_2022/xy_coor_deleted.txt", xy_coor_deleted)
                print("xy_coor_deleted shape: ", xy_coor_deleted.shape)
                xy_save = False

            # 位移数据拟合
            data_dict = {"dis_fit_lstsq": [], "dis_fit_sg": [],
                         "second_deri_sg": [], "strain_lstsq": []}

            for dis_i, dis_data_i in enumerate(z_list):
                # fit method1
                co_w, func, y_estimate_lstsq = func_fit(x_list[dis_i], dis_data_i)  # 最小二乘拟合一行数据（一条线上的所有测点）
                data_dict["dis_fit_lstsq"].append(y_estimate_lstsq)
                # fit method2
                dis_sg, dis_sg_deri = sg_filter(dis_data_i, 5, 3, 2)
                data_dict["dis_fit_sg"].append(dis_sg)
                data_dict["second_deri_sg"].append(dis_sg_deri)
                # strain calculate
                first_deri, second_deri, strain = strain_calc(x_list[dis_i], func)
                data_dict["strain_lstsq"].append(strain)

            dynamic_show = 0
            if dynamic_show:
                dynamic_visualization(x_list, z_list)  # 按时间显示原始位移数据

                dynamic_visualization(x_list, data_dict["dis_fit_lstsq"], linestyle='solid')  # 绘制lstsq拟合后的位移数据

                dynamic_visualization(x_list, data_dict["strain_lstsq"],
                                      ylabel="strain [uɛ]", linestyle='solid')  # 绘制lstsq拟合后的应变数据

                # 绘制应变图
                plt.show()

            for key, value in data_dict.items():
                data_dict_all[key].append(np.hstack(value))  # 将value转为一行，并添加到字典中

        # 保存拟合后的位移数据及应变数据
        save_flag = 1
        if save_flag:
            for keys, values in data_dict_all.items():
                np_values = np.array(values)
                np.savetxt(f"bending_strain/test_2022/{keys}.txt", np_values)

                print(f"{keys} 数据大小为：", np_values.shape)