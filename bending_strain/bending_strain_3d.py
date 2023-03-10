"""
带孔平板位移数据处理及三维可视化分析
"""

import os

from tqdm import tqdm
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.optimize import curve_fit

from calculate_strain_from_dis import func_fit, sg_filter, strain_calc


class Dynamic_3d_visulization():
    """
    通过 axs._offsets3d 更新三维数据,实际效果并不好
    """
    def __init__(self, xc, yc, zc, xraw, yraw, zraw, ftimes):
        self.xs = xc
        self.ys = yc
        self.zs = zc

        self.xraw = xraw
        self.yraw = yraw
        self.zraw = zraw

        self.time = ftimes
        self.time_range = len(ftimes)

        self.fig_3d = plt.figure()
        self.ax_3d = self.fig_3d.add_subplot(projection='3d')

        self.axs = self.ax_3d.scatter(0, 0, 0, marker="o", color="y")
        self.axc = self.ax_3d.scatter(0, 0, 0, marker="^", color="r")

        # axs = self.ax_3d.scatter(self.xs, self.ys, zs, marker="o", color="y")
        # axc = self.ax_3d.scatter(self.xraw , self.yraw, zraw_i, marker="^", color="r")

    def update(self, idx):
        """
        
        """
        title = f"Num.{idx} Time:[{self.time[idx]:f} s]"
        zs = self.zs[idx]
        zraw_i = self.zraw[idx]

        self.axs._offsets3d = (self.xs, self.ys, zs)
        # self.axc._offsets3d = (self.xs, self.ys, zraw_i)

        self.ax_3d.set_title(title)

        self.ax_3d.set_xlabel('X Position [mm]')
        self.ax_3d.set_ylabel('Y Position [mm]')
        self.ax_3d.set_zlabel('Z Position [um]')

        # return self.axc


    def animation(self):
        """
        
        """
        
        anim = FuncAnimation(self.fig_3d, self.update, frames=self.time_range, interval=300)

        plt.show()



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


def visualization_3d(xc=None, yc=None, zc=None, xraw=None, yraw=None, zraw=None, 
                     flag=None, times=None, z_label='Z Position [um]'):
    """
    位移数据的三维可视化（是否就是时域ODS？）
    """
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection='3d')

    cax = ax_3d.inset_axes([1.03, 0, 0.1, 1], transform=ax_3d.transAxes) # Colorbar is held by `cax`.
    # plt.ion()

    xs, ys = xc, yc
    for idx in range(len(times)):
        title = f"Num.{idx} Time:[{times[idx]:f} s]"
        zs = zc[idx]
        zraw_i = zraw[idx]

        ax_3d.cla()
        # Plot a trisurf
        if flag == "trisurf":
            fig_tri = ax_3d.plot_trisurf(xs, ys, zs, cmap='gist_rainbow_r')

        # Plot a scatter
        elif flag == "scatter":
            ax_3d.scatter(xs, ys, zs, marker="o", color="y")
            ax_3d.scatter(xraw, yraw, zraw_i, marker="^", color="r")

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
        # plt.colorbar(fig_tri)
        # cax.clear()
        # plt.colorbar(fig_tri, cax=cax)

        # plt.show()
        plt.pause(0.1)


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
        
        # 删除不规则点，空白区域填0(测试后发现不合适)， 填邻近值
        for r in range(2, 6):
            if r == 2:
                coor_list_x[r] = np.delete(coor_list_x[r], 13)
                coor_list_y[r] = np.delete(coor_list_y[r], 13)
                coor_list_z[r] = np.delete(coor_list_z[r], 13)

                coor_list_x[r] = np.insert(coor_list_x[r], 13, coor_list_x[0][13:15])
                coor_list_y[r] = np.insert(coor_list_y[r], 13, coor_list_y[r][:2])
                coor_list_z[r] = np.insert(coor_list_z[r], 13, coor_list_z[r][12:14])

            elif r == 3 or r == 4:
                coor_list_x[r] = np.insert(coor_list_x[r], 12, coor_list_x[0][12:16])
                coor_list_y[r] = np.insert(coor_list_y[r], 12, coor_list_y[r][:4])
                coor_list_z[r] = np.insert(coor_list_z[r], 12, coor_list_z[r][11:15])

            if r == 5:
                coor_list_x[r] = np.delete(coor_list_x[r], 12)
                coor_list_y[r] = np.delete(coor_list_y[r], 12)
                coor_list_z[r] = np.delete(coor_list_z[r], 12)

                coor_list_x[r] = np.insert(coor_list_x[r], 12, coor_list_x[0][12:15])
                coor_list_y[r] = np.insert(coor_list_y[r], 12, coor_list_y[r][:3])
                coor_list_z[r] = np.insert(coor_list_z[r], 12, coor_list_z[r][11:14])

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


def strain_hotmap(x, y, strain, times):
    """
    绘制应变随时间变化的热力图
    """
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (12, 4))

    for idx in range(len(times)):
        title = f"Num.{idx} Time:[{times[idx]:f} s]"
        strain_data = strain[idx].reshape(11, 31)
        ax.cla()
        cbar_ax.cla()
        sns.heatmap(ax = ax, data = strain_data, xticklabels=x, yticklabels=y, cmap = "gist_rainbow_r", cbar_ax = cbar_ax)

        plt.pause(0.001)

    plt.show()


def func_surface(xy, a, b, c, d, e, f):
    """
    二次曲面函数
    """
    x, y = xy
    z = a * x**3 + b * y**3 + c * x * y + d * x + e * y + f

    return z.ravel()


def func_surface_fit(func_s, xy, z):
    """
    曲面拟合
    """
    popt, pcov = curve_fit(func_s, xy, z)

    return popt, pcov


if __name__ == "__main__":
    # 整合数据并保存到test_2022/dis_data_all.txt
    # data_merge("E:/舜宇2022/ldv/数据/位移/")

    # ================================================================================
    # （1）原始位移及坐标数据读取显示
    # ================================================================================

    # 读取原始坐标点(x, y)数据
    coor_data = np.loadtxt("bending_strain/test_2022/point.txt")  # bending_strain/
    x_coor, y_coor = coor_data[:, 1], coor_data[:, 2]
    print("x_coor size: ", x_coor.shape, "\n", "y_coor size: ", y_coor.shape)

    # 读取所有原始坐标点的随时间变化的位移数据
    dis_data = np.loadtxt("bending_strain/test_2022/dis_data_all.txt")
    data_shape = dis_data.shape
    dis_data_pure = dis_data[:, 1:]
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
        show_xy_order = 0
        if show_xy_order:
            for coor_index, coor_xy in enumerate(coor_data):
                plt.plot(coor_xy[1], coor_xy[2], marker='o', linestyle='')
                plt.pause(0.01)

        plt.show()

    # 绘制原始位移散点图
    show_dynamic_dis = 0
    if show_dynamic_dis:
        dis_time = dis_data[:, 0]
        visualization_3d(x_coor, y_coor, dis_data_pure, "scatter", dis_time)
        plt.show()

    # ================================================================================
    # （2）所有时刻位移数据拟合及应变计算（逐行拟合位移数据、并求二阶导数计算应变）
    # ================================================================================

    local_load = 1
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

    # 绘制新的xy坐标位置图
    show_xy_position = 0
    if show_xy_position:
        plt.figure("x_y_coordinate")
        plt.plot(x_coor_d, y_coor_d, marker='.', linestyle='')

        plt.pause(0.01)

        # 查看坐标点绘制顺序
        show_xy_order = 0
        if show_xy_order:
            for coor_index, coor_xy in enumerate(xy_coor_deleted):
                plt.plot(coor_xy[0], coor_xy[1], marker='o', linestyle='')
                plt.pause(0.1)

        plt.show()

    # 绘制拟合后的位移的散点图、应变图（三维显示）
    show_dynamic_dis_fit = 1
    if show_dynamic_dis_fit:
        dis_time = dis_data[:, 0]

        # method 1
        # visualization_3d(x_coor_d, y_coor_d, dis_fit_lstsq_all, x_coor, y_coor, dis_data_pure, "scatter", dis_time)
        # visualization_3d(x_coor_d, y_coor_d, strain_lstsq_all, "trisurf", dis_time, z_label="Strain [uɛ]")
        visualization_3d(x_coor_d, y_coor_d, dis_fit_sg_all, x_coor, y_coor, dis_data_pure, "scatter", dis_time)
        # visualization_3d(x_coor_d, y_coor_d, second_deri_sg_all, "trisurf", dis_time, z_label="Strain [uɛ]")

        # plt.show()

        # method 2
        # draw = Dynamic_3d_visulization(x_coor_d, y_coor_d, dis_fit_lstsq_all, x_coor, y_coor, dis_data_pure, dis_time)
        # draw.animation()

    # 绘制应变热力图
    show_strain_hotmap = 0
    if show_strain_hotmap:
        dis_time = dis_data[:, 0]

        coor_axis = 0
        if coor_axis:
            x_c = x_coor_d[:31]
            y_c = y_coor_d[::31]
        else:
            x_c = np.arange(1,32)
            y_c = np.arange(1,12)
        
        strain_hotmap(x_c, y_c, dis_fit_sg_all, dis_time)

    # 曲面拟合
    surface_fit = 0
    if surface_fit:
        X_mesh, Y_mesh = np.meshgrid(x_coor, y_coor)
        XY = x_coor, y_coor
        Z = dis_data[1, 1:]

        popt, pcov = func_surface_fit(func_surface, XY, Z)  # 二次多项式拟合效果不太好
        print("popt params: ", popt)

        z_fit = func_surface(XY, *popt)

        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(projection='3d')

        ax_3d.scatter(x_coor, y_coor, z_fit, color='y')
        ax_3d.scatter(x_coor, y_coor, Z, color='red')

        plt.show()