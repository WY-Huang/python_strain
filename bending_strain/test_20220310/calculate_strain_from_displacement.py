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
    dis_data_mm = dis_data[:, 1:]       # 删除第一列时间信息

    return x_coor, dis_data_mm


def func_fit(x, dis, M=3):
    """
    方法一：最小二乘法、多项式拟合单行位移数据，曲线过于平滑，会丢失局部特征，误差随阶次增加。

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


def data_merge(source_path, save_file_name, point_num, save_flg=True, merge_type="Vib"):
    """
    将每个测点的位移数据整合到一个文件
    """
    # for file in os.listdir(path):
    #     file_name_list = file.split("_")
    #     if file_name_list[2] == "Vib.txt":
    for index in range(1, point_num+1):
        dis = np.loadtxt(source_path + f"0_{index}_{merge_type}.txt")

        if index == 1:
            data_all = dis
        else:
            data_all = np.column_stack((data_all, dis[:, 1]))

    # 第一列为时间，后续每一列为一个测点随时间变化的位移数据
    if save_flg:
        np.savetxt(save_file_name, data_all)
    print("数据大小为：", data_all.shape)


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


if __name__ == "__main__":
    # 整合数据并保存到test_2022/dis_data_all.txt
    merge = 0
    if merge:
        data_merge("/home/wanyel/vs_code/python_strain/bending_strain/export_0326_1Vpp/", 
                "/home/wanyel/vs_code/python_strain/bending_strain/export_0326_1Vpp/strain_merge_20230326_1.txt", 
                303,
                merge_type="Ref")

    # 读取数据
    plate_length = 40.0      # 单行测点实际总长度（mm），实际长度46.2mm
    plate_thickness = 3.42   # 板的中性面到表面的厚度（mm），实际厚度3.42mm
    sample_num = 101         # 单行测点数量
    x_coor, dis_data_mm = read_data('/home/wanyel/vs_code/python_strain/bending_strain/export_0326_1Vpp/dis_merge_20230326_1.txt', sample_num, plate_length)

    # 去除积分导致的每个点的第一个为0的数据
    x_coor = x_coor[1:]
    print("x_coor:", x_coor.shape, "\ndis_data_mm:", dis_data_mm.shape)

    # 仅绘制第 only_one 行的数据，否则绘制随时间变化的全部数据
    only_one = 0
    if only_one:

        # max_dis_index = np.unravel_index(dis_data_mm.argmax(), dis_data_mm.shape)   # 最大值索引
        # print("最大位移的位置索引及值：", max_dis_index, "\t", dis_data_mm[max_dis_index])
        # dis_data_one = dis_data_mm[max_dis_index[0], 1:31]
        dis_data_one = dis_data_mm[only_one, 1:101] # 0-31-62-93-124-155

        # 滤波处理
        # dis_data_filter = np_move_avg(dis_data_one, 11)     # 1）滑动平均

        interval_delta = plate_length / (sample_num - 1)
        win_size = sample_num // 5                          # 滑动窗口选取为测点数的1/5，且为奇数
        if win_size % 2 == 0:
            win_size += 1
        if win_size <= 11:
            win_size = 11

        dis_sg, sid_sg_deri = sg_filter(dis_data_one, win_size, 2, 2, interval_delta) # 2）绘制sg滤波后的数据及2阶导数

        dis_data_filter = np_move_avg(dis_sg, 11)     # 1）滑动平均

        # ================================================================== #
        # 绘制原始位移散点及拟合后的位移曲线
        # ================================================================== #
        co_w, func, y_estimate_lstsq, resl_lists = func_fit(x_coor, dis_data_filter, 5)   # 单行全部位移数据最小二乘拟合
        # print("resl_lists: ", resl_lists)

        plt.figure("Displacement")
        plt.plot(x_coor, dis_data_one, 'bo', label="dis_raw")
        plt.plot(x_coor, dis_data_filter, 'mo', label="dis_moveA_filter")
        plt.plot(x_coor, y_estimate_lstsq, 'r', lw=2.0, label="dis_lstsq")
        plt.plot(x_coor, dis_sg, 'yo', lw=2.0, label="dis_sg")
        plt.legend()
        plt.xlabel("x_coordinate [mm]")
        plt.ylabel("y_displacement [mm]")

        # ================================================================== #
        # 绘制一/二阶导数曲线及应变曲线，应变片范围点号40-64
        # ================================================================== #
        first_deri, second_deri, strain_lstsq = strain_calc(x_coor, func, plate_thickness)
        strain_sg = sid_sg_deri * plate_thickness * 1e6

        plt.figure("Strain")
        # plt.plot(x_coor, first_deri, 'b', lw=2.0, label="first_deri")
        # plt.plot(x_coor, np.array(second_deri) * 1e6, 'g', lw=2.0, label="second_deri")
        plt.plot(x_coor, strain_lstsq, 'ro', lw=2.0, label="strain_lstsq")
        # plt.plot(x_coor, strain_sg, 'y', lw=2.0, label="strain_sg")
        plt.plot(x_coor, np.zeros_like(x_coor), 'b', label="y = 0")
        plt.legend()
        plt.xlabel("x_coordinate [mm]")
        plt.ylabel("y_strain [uɛ]")

        # ================================================================== #
        # 绘制应变片数据
        # ================================================================== #
        show_strain = 1
        if show_strain:
            strain_gage = np.loadtxt("/home/wanyel/vs_code/python_strain/bending_strain/export_0326_1Vpp/strain_merge_20230326_1.txt")

            strain_gage_value = strain_gage[:, 1:] * (-1)
            # LDV_mean_strain = np.sum(strain_sg[12:20]) / 8

            plt.figure("LDV vs Strain_Gage")
            plt.plot(strain_gage[only_one, 0], strain_gage_value[only_one, 50], 'bo', label="strain_gage")  # , marker='.'
            plt.plot(1, strain_lstsq[50], 'r', label="LDV_mean_strain", marker='^')
            plt.legend()
            plt.xlabel("x_time [s]")
            plt.ylabel("y_strain [uɛ]")
        
        plt.show()

    else:
    # ================================================================== #
    # 绘制随时间变化的 应变片和LDV的对比数据
    # ================================================================== #
        ldv_strain = []
        for data_t in range(dis_data_mm.shape[0]):
            dis_data_one = dis_data_mm[data_t, 1:101]

            # 滤波处理
            # dis_data_filter = np_move_avg(dis_data_one, 11)     # 1）滑动平均

            interval_delta = plate_length / (sample_num - 1)
            win_size = sample_num // 5                          # 滑动窗口选取为测点数的1/5，且为奇数
            if win_size % 2 == 0:
                win_size += 1
            if win_size <= 11:
                win_size = 11

            dis_sg, sid_sg_deri = sg_filter(dis_data_one, win_size, 2, 2, interval_delta) # 2）绘制sg滤波后的数据及2阶导数
            dis_data_filter = np_move_avg(dis_sg, 11)     # 1）滑动平均

            co_w, func, y_estimate_lstsq, resl_lists = func_fit(x_coor, dis_data_filter, 5)   # 单行全部位移数据最小二乘拟合
            first_deri, second_deri, strain_lstsq = strain_calc(x_coor, func, plate_thickness)

            ldv_strain.append(strain_lstsq[50])

        show_strain = 1
        if show_strain:
            strain_gage = np.loadtxt("/home/wanyel/vs_code/python_strain/bending_strain/export_0326_1Vpp/strain_merge_20230326_1.txt")

            strain_gage_value = strain_gage[:, 1:]
            strain_gage_value = strain_gage_value / 2e5 * 4 * 1e6 / 10 / 2.08
            # LDV_mean_strain = np.sum(strain_sg[12:20]) / 8

            plt.figure("LDV vs Strain_Gage")
            plt.plot(strain_gage[:, 0], strain_gage_value[:, 50], 'b', label="strain_gage", marker='.')  # , marker='.'
            plt.plot(strain_gage[:, 0], ldv_strain, 'r', label="LDV_mean_strain", marker='^')
            plt.legend()
            plt.xlabel("x_time [s]")
            plt.ylabel("y_strain [uɛ]")
            
            plt.show()