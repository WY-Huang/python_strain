"""
平板三维数据处理
"""

import os
from tqdm import tqdm
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

from calculate_strain_from_dis import func_fit, sg_filter, strain_calc


def demo_3d_visual(random_ge=True):
    """
    测试代码
    """
    # 随机生成数据
    if random_ge:
        np.random.seed(19960808)
        num = 20
        xs = np.linspace(0, 200, num)
        ys = np.linspace(0, 100, num)
        # np.random.shuffle(ys)
        zs = np.random.rand(num)

        X, Y = np.meshgrid(xs, ys)
        Z = np.linspace(0, num, num ** 2).reshape(num, num)


def visualization_3d(xc=None, yc=None, zc=None, flag=None, times=None, z_label='Z Position [um]'):
    """
    位移数据的三维可视化（是否就是时域ODS？）
    """
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection='3d')
    # plt.ion()

    xs, ys = xc, yc
    for idx in range(len(times)):
        title = f"Num.{idx} Time:[{times[idx]:f} s]"
        zs = zc[idx]

        ax_3d.cla()
        # Plot a trisurf
        if flag == "trisurf":
            ax_3d.plot_trisurf(xs, ys, zs, cmap='gist_rainbow_r')

        # Plot a scatter
        elif flag == "scatter":
            ax_3d.scatter(xs, ys, zs, marker="o")

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
        # ax.set_zlim(-1, 2)

        # plt.show()
        plt.pause(0.001)


def data_merge(path, save_flg=True):
    """
    将每个测点的位移数据整合到一个文件
    """
    # for file in os.listdir(path):
    #     file_name_list = file.split("_")
    #     if file_name_list[2] == "Vib.txt":
    for index in range(1, 369):
        dis = np.loadtxt(path + f"0_{index}_Vib.txt")

        if index == 1:
            data_all = dis
        else:
            data_all = np.column_stack((data_all, dis[:, 1]))

    # 第一列为时间，后续每一列为一个测点随时间变化的位移数据
    if save_flg:
        np.savetxt("test_2022/dis_data_all.txt", data_all)
    print("数据大小为：", data_all.shape)


def get_x_y_z_data(x_coor_all, y_coor_all, dis_z):
    """
    获取每行x和z的坐标
    """
    assert len(x_coor_all) == len(dis_z) == len(y_coor_all)

    x_max = x_coor_all.max()
    x_min = x_coor_all.min()
    coor_list_x = []
    coor_list_y = []
    coor_list_z = []

    x_max_index = np.where(x_coor_all == x_max)
    x_min_index = np.where(x_coor_all == x_min)
    assert len(x_max_index) == len(x_min_index)
    # print("x_max_index: ", x_max_index, "\n", "x_min_index: ", x_min_index)

    delete_outlier = 1
    if delete_outlier:
        for row in range(len(x_max_index[0])):
            start_i = x_min_index[0][row]
            end_i = x_max_index[0][row] + 1
            coor_list_x.append(x_coor_all[start_i:end_i])
            coor_list_z.append(dis_z[start_i:end_i])
            coor_list_y.append(y_coor_all[start_i:end_i])

    else:
        x_max_i_start = 0
        for x_max_i in np.nditer(x_max_index):
            coor_list_x.append(x_coor_all[x_max_i_start:(x_max_i + 1)])
            coor_list_z.append(dis_z[x_max_i_start:(x_max_i + 1)])
            coor_list_y.append(y_coor_all[x_max_i_start:(x_max_i + 1)])
            x_max_i_start = x_max_i + 1

    print_flag = 0
    if print_flag:
        list_len_x = []
        for x in iter(coor_list_x):
            list_len_x.append(len(x))
        print("list_len_x:", list_len_x)

        list_len_z = []
        for z in iter(coor_list_z):
            list_len_z.append(len(z))
        print("list_len_z:", list_len_z)

    return coor_list_x, coor_list_y, coor_list_z


def dynamic_visualization(x_point_all, data_all, figure_num=1, ylabel="displacement [mm]", line_num=1000,
                          marker='.', linestyle=''):
    """
    绘制所有测点在同一时刻下的位移/应变曲线,动态显示
    """
    # data_all = np.array(data_all)
    plt.ion()
    plt.figure()
    print("start plot：", ylabel)

    line_num = len(data_all)
    for row in range(line_num):
        x_point = x_point_all[row]
        y_dis = data_all[row]

        # plt.clf()
        plt.cla()
        plt.plot(x_point, y_dis, marker=marker, linestyle=linestyle)
        plt.plot(x_point, np.zeros_like(x_point))

        plt.xlabel("x_position")
        plt.ylabel(ylabel)
        # plt.draw()
        if row != line_num - 1:
            plt.pause(0.1)  # 显示秒数
        else:
            plt.show()

    plt.ioff()  # 关闭interactive mode
    # plt.show()
    # plt.close("all")


if __name__ == "__main__":
    # 整合数据并保存到test_2022/dis_data_all.txt
    # data_merge("E:/舜宇2022/ldv/数据/位移/")

    # ================================================================================
    # （1）原始位移及坐标数据读取显示
    # ================================================================================

    # 读取原始坐标点(x, y)数据
    coor_data = np.loadtxt("test_2022/point.txt")  # bending_strain/
    x_coor, y_coor = coor_data[:, 1], coor_data[:, 2]

    # 读取所有原始坐标点的随时间变化的位移数据
    dis_data = np.loadtxt("test_2022/dis_data_all.txt")
    data_shape = dis_data.shape
    print("Out of plate displacement data size: ", data_shape)

    # 绘制原始xy坐标位置图
    show_xy_position = 0
    if show_xy_position:
        plt.figure("x_y_coordinate")
        plt.plot(x_coor, y_coor, marker='.', linestyle='')

        # X, Y = np.meshgrid(x_coor, y_coor)
        # plt.plot(X, Y, color='red', marker='.', linestyle='')  # 线型为空，即点与点之间不用线连接

        # plt.show()
        plt.pause(0.1)

        # 查看坐标点绘制顺序
        for coor_index, coor_xy in enumerate(coor_data):
            plt.plot(coor_xy[1], coor_xy[2], marker='o', linestyle='')
            plt.pause(0.01)

        plt.show()

    # 绘制原始位移散点图
    show_dynamic_dis = 0
    if show_dynamic_dis:
        dis_time = dis_data[:, 0]
        visualization_3d(x_coor, y_coor, dis_data[:, 1:], "scatter", dis_time)
        plt.show()

    # ================================================================================
    # （2）所有时刻位移数据拟合及应变计算（逐行拟合位移数据、并求二阶导数计算应变）
    # ================================================================================

    local_load = 1
    if local_load:
        xy_coor_deleted = np.loadtxt("test_2022/xy_coor_deleted.txt")
        x_coor_d, y_coor_d = xy_coor_deleted[:, 0], xy_coor_deleted[:, 1]

        dis_fit_lstsq_all = np.loadtxt("test_2022/dis_fit_lstsq.txt")
        dis_fit_sg_all = np.loadtxt("test_2022/dis_fit_sg.txt")
        strain_lstsq_all = np.loadtxt("test_2022/second_deri_sg.txt")
        second_deri_sg_all = np.loadtxt("test_2022/strain_lstsq.txt")
    else:
        data_dict_all = {"dis_fit_lstsq": [], "dis_fit_sg": [],
                         "second_deri_sg": [], "strain_lstsq": []}
        xy_save = True
        for dis_point in tqdm(dis_data[:, 1:]):

            # 解析某一时刻的位移数据z及坐标xy
            x_list, y_list, z_list = get_x_y_z_data(x_coor, y_coor, dis_point)
            if xy_save:
                x_coor_deleted = np.hstack(x_list).reshape((-1, 1))
                y_coor_deleted = np.hstack(y_list).reshape((-1, 1))
                xy_coor_deleted = np.hstack((x_coor_deleted, y_coor_deleted))
                np.savetxt("test_2022/xy_coor_deleted.txt", xy_coor_deleted)
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
                np.savetxt(f"test_2022/{keys}.txt", np_values)

                print(f"{keys} 数据大小为：", np_values.shape)

    # 绘制新的xy坐标位置图
    show_xy_position = 1
    if show_xy_position:
        plt.figure("x_y_coordinate")
        plt.plot(x_coor_d, y_coor_d, marker='.', linestyle='')

        plt.pause(0.1)

        # 查看坐标点绘制顺序
        for coor_index, coor_xy in enumerate(xy_coor_deleted):
            plt.plot(coor_xy[0], coor_xy[1], marker='o', linestyle='')
            plt.pause(0.01)

        plt.show()

    # 绘制拟合后的位移的散点图、应变图
    show_dynamic_dis_fit = 1
    if show_dynamic_dis_fit:
        dis_time = dis_data[:, 0]
        # visualization_3d(x_coor_d, y_coor_d, dis_fit_lstsq_all, "scatter", dis_time)
        # visualization_3d(x_coor_d, y_coor_d, strain_lstsq_all, "trisurf", dis_time, z_label="Strain [uɛ]")
        #
        # visualization_3d(x_coor_d, y_coor_d, dis_fit_sg_all, "scatter", dis_time)
        visualization_3d(x_coor_d, y_coor_d, second_deri_sg_all, "trisurf", dis_time, z_label="Strain [uɛ]")

        plt.show()
