"""
平板三维数据处理
"""

import os
from tqdm import tqdm
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

from calculate_strain_from_dis import func_fit, sg_filter, strain_calc


def dis_visualization_3d(xs=None, ys=None, zs=None, flag=None, random_ge=True, title='', z_label='Z Position [um]'):
    """
    位移数据的三维可视化（是否就是时域ODS？）
    """
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax.cla()

    # 随机生成数据
    if random_ge:
        np.random.seed(19960808)
        num = 20
        xs = np.linspace(0, 200, num)
        ys = np.linspace(0, 100, num)
        # np.random.shuffle(ys)
        zs = np.random.rand(num)

        X, Y = np.meshgrid(xs, ys)
        Z = np.linspace(0, num, num**2).reshape(num, num)

    # Plot a trisurf
    if flag == "trisurf":
        ax.plot_trisurf(xs, ys, zs, cmap='hot_r')

    # Plot a scatter
    if flag == "scatter":
        ax.scatter(xs, ys, zs, marker="o")

    # Plot a basic wireframe.
    if flag == "wireframe":
        # Z = zs.reshape(num, num)
        # Z = np.sin(np.sqrt(X ** 2 + Y ** 2))

        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    # Plot the surface.
    if flag == "surface":
        ax.plot_surface(X, Y, Z, cmap=cm.autumn, linewidth=0, antialiased=False)


    ax.set_xlabel('X Position [mm]')
    ax.set_ylabel('Y Position [mm]')
    ax.set_zlabel(z_label)
    # ax.set_zlim(-1, 2)

    ax.set_title(title)

    # plt.show()
    plt.pause(0.0001)


def data_merge(path, save_flag=True):
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
    if save_flag:
        np.savetxt("test_2022/dis_data_all.txt", data_all)
    print("数据大小为：", data_all.shape)


def get_x_z_data(x_coor_all, dis_z):
    """
    获取每行x和z的坐标
    """
    x_max = x_coor_all.max()
    coor_list_x = []
    coor_list_z = []
    dis_z_i = dis_z[1:]

    x_max_index = np.where(x_coor_all==x_max)
    # print("x_max_index: ", x_max_index)
    x_max_i_start = 0
    for x_max_i in np.nditer(x_max_index):
        coor_list_x.append(x_coor_all[x_max_i_start:(x_max_i+1)])
        coor_list_z.append(dis_z_i[x_max_i_start:(x_max_i+1)])
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

    return coor_list_x, coor_list_z


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
        if row != line_num-1:
            plt.pause(0.1)  # 显示秒数
        else:
            plt.show()

    plt.ioff()  # 关闭interactive mode
    # plt.show()
    # plt.close("all")



if __name__ == "__main__":
    # 整合数据并保存到test_2022/dis_data_all.txt
    # data_merge("E:/舜宇2022/ldv/数据/位移/")

    # 读取坐标点(x, y)数据
    coor_data = np.loadtxt("bending_strain/test_2022/point.txt")
    x_coor, y_coor = coor_data[:, 1], coor_data[:, 2]

    # 读取所有坐标点的随时间变化的位移数据
    dis_data = np.loadtxt("bending_strain/test_2022/dis_data_all.txt")
    data_shape = dis_data.shape
    print("dis data size: ", data_shape)

    # 绘制xy坐标位置图
    show_xy_position = 0
    if show_xy_position:
        plt.plot(x_coor, y_coor, marker='.', linestyle='')

        # X, Y = np.meshgrid(x_coor, y_coor)
        # plt.plot(X, Y, color='red', marker='.', linestyle='')  # 线型为空，即点与点之间不用线连接

        # plt.show()
        plt.pause(0.1)

        for coor_index, coor_xy in enumerate(coor_data):
            plt.plot(coor_xy[1], coor_xy[2], marker='^', linestyle='')
            plt.pause(0.01)
        
        plt.pause(0)

    # 绘制原始位移散点
    show_dynamic_dis = 0
    if show_dynamic_dis:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        dis_time = dis_data[:, 0]
        for index in range(data_shape[0]):
            title_time = f"Num.{index} Time:[{dis_time[index]:f} s]"
            dis_visualization_3d(x_coor, y_coor, dis_data[index, 1:], "scatter", False, title_time)

    # 所有时刻位移数据拟合及应变计算
    dis_fit_lstsq_all = []
    strain_lstsq_all = []
    for dis_point in tqdm(dis_data):

    
        # 解析某一时刻的位移数据z及坐标xy
        x_list, z_list = get_x_z_data(x_coor, dis_point)

        # 位移数据拟合
        dis_fit_lstsq = []
        dis_fit_sg = []
        second_deri_sg = []
        strain_lstsq = []

        for dis_i, dis_data in enumerate(z_list):
            # fit method1
            co_w, func, y_estimate_lstsq = func_fit(x_list[dis_i], dis_data)   # 最小二乘拟合一行数据（一条线上的所有测点）
            dis_fit_lstsq.append(y_estimate_lstsq)
            # fit method2
            dis_sg, dis_sg_deri = sg_filter(dis_data, 5, 3, 2)
            dis_fit_sg.append(dis_sg)
            second_deri_sg.append(dis_sg_deri)
            # strain calculate
            first_deri, second_deri, strain = strain_calc(x_list[dis_i], func)
            strain_lstsq.append(strain)

        dynamic_show = 0
        if dynamic_show:
            dynamic_visualization(x_list, z_list)   # 按时间显示原始位移数据
            
            dynamic_visualization(x_list, dis_fit_lstsq, linestyle='solid')  # 绘制lstsq拟合后的位移数据

            dynamic_visualization(x_list, strain_lstsq, ylabel="strain [uɛ]", linestyle='solid')  # 绘制lstsq拟合后的应变数据

            # 绘制应变图
            plt.show()

        show_dynamic_strain = 0
        if show_dynamic_strain:
            # 绘制应变散点
            # strain_one = sum(strain_lstsq, [])    # 列表展开，如果strain_lstsq为列表
            strain_one = np.hstack(strain_lstsq)

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            dis_visualization_3d(x_coor, y_coor, strain_one, "trisurf", False, z_label="Strain [uɛ]")
            plt.show()

        dis_fit_lstsq_flat = np.hstack(dis_fit_lstsq)
        dis_fit_lstsq_all.append(dis_fit_lstsq_flat)

        strain_lstsq_flat = np.hstack(strain_lstsq)
        strain_lstsq_all.append(strain_lstsq_flat)

    # 绘制拟合位移散点
    show_dynamic_dis_fit = 1
    if show_dynamic_dis_fit:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        dis_time = dis_data[:, 0]
        for index in range(data_shape[0]):
            title_time = f"Num.{index} Time:[{dis_time[index]:f} s]"
            dis_visualization_3d(x_coor, y_coor, dis_fit_lstsq_all[index], "scatter", False, title_time)
