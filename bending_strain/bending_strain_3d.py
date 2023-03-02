"""
平板三维数据处理
"""


import os
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt


def dis_visualization_3d(xs=None, ys=None, zs=None, flag=None, random_ge=True):
    """
    位移数据的三维可视化（是否就是时域ODS？）
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

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
        ax.plot_trisurf(xs, ys, zs, cmap='viridis')

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
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


    ax.set_xlabel('X Position [mm]')
    ax.set_ylabel('Y Position [mm]')
    ax.set_zlabel('Z Position [mm]')
    # ax.set_zlim(-1, 2)

    plt.show()


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


if __name__ == "__main__":
    # 整合数据并保存到test_2022/dis_data_all.txt
    # data_merge("E:/舜宇2022/ldv/数据/位移/")

    # 读取坐标点(x, y)数据
    coor_data = np.loadtxt("test_2022/point.txt")
    x_coor, y_coor = coor_data[:, 1], coor_data[:, 2]
    plt.plot(x_coor, y_coor, marker='.', linestyle='')

    # X, Y = np.meshgrid(x_coor, y_coor)
    # plt.plot(X, Y, color='red', marker='.', linestyle='')  # 线型为空，即点与点之间不用线连接

    plt.show()

    # 读取所有坐标点的随时间变化的位移数据
    dis_data = np.loadtxt("test_2022/dis_data_all.txt")

    # 绘制原始位移散点及拟合后的位移曲线
    dis_visualization_3d(x_coor, y_coor, dis_data[1, 1:], "trisurf", False)


    # 绘制应变图


