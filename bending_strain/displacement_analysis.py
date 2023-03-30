"""
位移数据的可视化分析(2D可视化)
"""


import time
import numpy as np
import matplotlib.pyplot as plt


def dis_data_visual(data_path=None):
    """
    单个测点的速度及位移随时间变化的可视化
    """
    dis_data = np.loadtxt(data_path, dtype=str)
    dis_data = np.delete(dis_data, 0, 0).astype(np.float64)

    x_time = dis_data[:, 1]
    y_velocity = dis_data[:, 2]
    y_displacement = dis_data[:, 3] / 1000  # unit to mm

    plt.figure()
    plt.plot(x_time, y_velocity, 'b', label='y_velocity [mm/s]')
    plt.plot(x_time, y_displacement, 'r', label='y_displacement [mm]')

    # 辅助分析线
    plt.plot(x_time, np.zeros_like(x_time), 'y', label='y = 0 [mm]')  # 绘制y=0的直线
    # y_list = [] # 绘制y=0时，垂直于x轴的辅助线
    # for y_index, y_data in enumerate(y_displacement):
    #     if y_data == 0:
    #         y_list.append(y_index)
    # plt.vlines(y_list, ymin=-1, ymax=1, color= 'pink', label="vline_y=0")

    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("displacement [mm]")
    # plt.show()


def data_merge(data_num, save_flag=False):
    """
    整合所有测点数据到单个txt文件
    """
    for i in range(1, data_num+1):
        data_path = f"/home/wanyel/vs_code/python_strain/bending_strain/dis_data_20230223/新建试验#{i}.txt"
        dis_str = np.loadtxt(data_path, dtype=str)
        dis_float = np.delete(dis_str, 0, 0).astype(np.float64)     # 删除第一行数据

        if i == 1:
            data_all = dis_float
        else:
            data_all = np.column_stack((data_all, dis_float[:, 2:]))
    
    # 第一列为序号，第二列为记录时间（ms），其它奇数列为速度（mm/s），偶数列为位移（um）
    if save_flag:
        np.savetxt("bending_strain/dis_data_20230223/dis_data_all.txt", data_all)
    print("数据大小为：", data_all.shape)


def time_dis(line_num=1000):
    """
    绘制所有测点在同一时刻下的位移曲线
    """
    dis_all = np.loadtxt("bending_strain/dis_data_20230223/dis_data_all.txt")

    plt.ion()
    plt.figure()

    x_point = np.arange(1, dis_all.shape[1] / 2)
    for row in range(line_num):
        y_dis = dis_all[row, 3::2]

        # plt.clf()
        plt.cla()
        plt.plot(x_point, y_dis/1000)
        plt.plot(x_point, np.zeros_like(x_point))

        plt.xlabel("x_index")
        plt.ylabel("displacement [mm]")
        # plt.draw()
        if row != line_num-1:
            plt.pause(0.001)  #显示秒数
        else:
            plt.pause(0)
        # plt.close()


if __name__ == "__main__":

    # 单个测点数据的可视化
    if 1:
        for data_i in range(1, 10):
            dis_data_visual(f"/home/wanyel/vs_code/python_strain/bending_strain/dis_data_20230223/新建试验#{data_i}.txt")

        plt.show()

    # data_merge(55, save_flag=True)

    # 同一时刻下，位移数据可视化
    # time_dis()