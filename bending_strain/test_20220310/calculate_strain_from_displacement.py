"""
平板结构单行测点的弯曲应变计算，采用Savitzky-Golay滤波算法或全局最小二乘法

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def read_data(data_path=None, sample_num=25, plate_length=96):
    """
    读取位移数据,统一位移和x坐标的单位为mm。

    return:
        x_coor: x方向坐标（mm）
        dis_data_mm: 面外位移z（mm）
    """
    x_coor = np.arange(sample_num) * (plate_length / (sample_num - 1))  # .reshape(1, -1)

    dis_data = np.loadtxt(data_path)
    # print("dis_data: ", dis_data.shape)
    dis_data_mm = dis_data[:, 1:] / 50

    return x_coor, dis_data_mm


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



def sg_filter(y_noise, win_size=None, poly_order=None, deriv=0, delta=1):
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
        strain: 应变（ɛ）
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

    strain = np.array(second_deri) * palte_thick

    return first_deri, second_deri, strain


def data_merge(source_path, save_path, point_num, save_flg=True):
    """
    将每个测点的位移数据整合到一个文件
    """
    # for file in os.listdir(path):
    #     file_name_list = file.split("_")
    #     if file_name_list[2] == "Vib.txt":
    for index in range(1, point_num+1):
        dis = np.loadtxt(source_path + f"0_{index}_Ref.txt")

        if index == 1:
            data_all = dis
        else:
            data_all = np.column_stack((data_all, dis[:, 1]))

    # 第一列为时间，后续每一列为一个测点随时间变化的位移数据
    if save_flg:
        np.savetxt(save_path, data_all)
    print("数据大小为：", data_all.shape)


def np_move_avg(data, win_size, mode="valid"):
    """
    滑动平均算法
    """
    data_p = data.copy()
    crop_data = np.convolve(data_p, np.ones(win_size) / win_size, mode=mode)
    insert_length = win_size // 2
    data_length = len(data_p)
    data_p[insert_length:data_length-insert_length] = crop_data
    return data_p


if __name__ == "__main__":
    # # 整合数据并保存到test_2022/dis_data_all.txt
    # data_merge("/home/wanyel/vs_code/python_strain/bending_strain/export_0323/", 
    #            "/home/wanyel/vs_code/python_strain/bending_strain/export_0323/strain_merge_20230323.txt", 155)

    # 读取数据
    plate_length = 40.0      # 单行测点实际总长度（mm），实际长度46.2mm
    plate_thickness = 1.71   # 板的中性面到表面的厚度（mm），实际厚度3.42mm
    sample_num = 31         # 单行测点数量
    x_coor, dis_data_mm = read_data('/home/wanyel/vs_code/python_strain/bending_strain/export_0323/dis_merge_20230323.txt', sample_num, plate_length)

    # 去除积分导致的每个点的第一个为0的数据
    x_coor = x_coor[1:]

    print("x_coor:", x_coor.shape, "\ndis_data_mm:", dis_data_mm.shape)
    # 仅绘制第一行数据
    only_one = 1
    if only_one:

        # max_dis_index = np.unravel_index(dis_data_mm.argmax(), dis_data_mm.shape)   # 最大值索引
        # print("最大位移的位置索引及值：", max_dis_index, "\t", dis_data_mm[max_dis_index])
        # dis_data_one = dis_data_mm[max_dis_index[0], 1:31]
        dis_data_one = dis_data_mm[4919, 1:31] # 0-31-62-93-124-155
        dis_data_filter = np_move_avg(dis_data_one, 5)
        # 绘制原始位移散点及拟合后的位移曲线
        plt.figure(1)
        plt.plot(x_coor, dis_data_one, 'bo', label="dis_noise")
        plt.plot(x_coor, dis_data_filter, 'mo', label="dis_filter")

        co_w, func, y_estimate_lstsq = func_fit(x_coor, dis_data_filter)   # 单行全部位移数据最小二乘拟合

        # plt.plot(x_coor, y_estimate_lstsq, 'r', lw=2.0, label="dis_lstsq")

        # 绘制sg滤波后的数据及2阶导数
        interval_delta = plate_length / (sample_num - 1)
        win_size = sample_num // 5      # 滑动窗口选取为测点数的1/5，且为奇数
        if win_size % 2 == 0:
            win_size += 1
        if win_size <= 11:
            win_size = 11

        dis_sg, sid_sg_deri = sg_filter(dis_data_filter, win_size, 3, 2, interval_delta) # 
        plt.plot(x_coor, dis_sg, 'y', lw=2.0, label="dis_sg")
        # plt.plot(x_coor, sid_sg_deri, 'p', lw=2.0, label="sid_sg_deri")

        plt.legend()
        plt.xlabel("x_coordinate [mm]")
        plt.ylabel("y_displacement [mm]")

        # 绘制一/二阶导数曲线及应变曲线
        first_deri, second_deri, strain = strain_calc(x_coor, func, plate_thickness)
        strain_lstsq = strain * 1e6

        strain_sg = sid_sg_deri * plate_thickness * 1e6

        plt.figure(2)
        # plt.plot(x_coor, first_deri, 'b', lw=2.0, label="first_deri")
        # plt.plot(x_coor, np.array(second_deri) * 1e6, 'g', lw=2.0, label="second_deri")
        # plt.plot(x_coor, strain_lstsq, 'r', lw=2.0, label="strain_lstsq")
        plt.plot(x_coor, strain_sg, 'y', lw=2.0, label="strain_sg")
        plt.plot(x_coor, np.zeros_like(x_coor), 'b', label="y = 0")

        plt.legend()
        plt.xlabel("x_coordinate [mm]")
        plt.ylabel("y_strain [uɛ]")

        # 绘制应变片数据
        strain = 1
        if strain:
            strain_gage = np.loadtxt("/home/wanyel/vs_code/python_strain/bending_strain/export_0323/strain_merge_20230323.txt")

            strain_gage_value = strain_gage[:, 1:] / 500000 / 2.5 / 2.08 * 1000000
            LDV_mean_strain = np.sum(strain_lstsq[12:20]) / 8

            plt.figure(3)
            plt.plot(strain_gage[:, 0], strain_gage_value[:, 1], 'b', label="strain_gage")  # , marker='.'
            # plt.plot(1, LDV_mean_strain, 'r', label="LDV_mean_strain", marker='^')
            plt.legend()
            plt.xlabel("x_time [s]")
            plt.ylabel("y_strain [uɛ]")
        
        plt.show()