import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

def read_data(sample_num=25, data_path=None, plate_length=96):
    '''
    读取位移数据,统一位移和x坐标的单位为mm
    '''
    x_coor = np.arange(0, sample_num).reshape(sample_num, 1) * (plate_length / sample_num)

    dis_data = np.loadtxt('dis_data_25.txt')
    dis_data_mm = dis_data / 1000

    return x_coor, dis_data_mm

def func_fit(x, dis, M=4):
    '''
    最小二乘法拟合位移数据
    '''
    X = x
    for i in range(2, M+1):
        X = np.column_stack((X, pow(x, i)))

    X = np.insert(X, 0, [1], 1)

    co_w, resl, _, _ = np.linalg.lstsq(X, dis, rcond=None)  # resl为残差的平方和
    # print("co_w:", co_w, "\nresl", resl)
    y_estimate_lstsq = X.dot(co_w)

    return co_w, y_estimate_lstsq

def f_fit(x_coord):
    '''
    拟合函数
    '''
    f = co_w[4] * x_coord ** 4 + co_w[3] * x_coord ** 3 + co_w[2] * x_coord ** 2 + co_w[1] * x_coord + co_w[0]

    return f

def strain_calc(x):
    '''
    根据拟合后的位移方程的2阶导数计算应变
    '''
    second_deri = []
    for _, x_value in enumerate(x):
        
        second_value = derivative(f_fit, x_value, dx=1e-6, n = 2)

        second_deri.append(second_value)

    strain = np.array(second_deri) * 0.25

    return second_deri, strain


if __name__ == "__main__":
    x_coor, dis_data_mm = read_data()

    # 绘制原始位移散点及拟合后的位移曲线
    plt.figure(1)
    plt.plot(x_coor, dis_data_mm, 'bo', label="dis_noise")

    co_w, y_estimate_lstsq = func_fit(x_coor, dis_data_mm)

    plt.plot(x_coor, y_estimate_lstsq, 'r', lw=2.0, label="lstsq")

    second_deri, strain = strain_calc(x_coor)

    plt.legend()

    # 绘制二阶导数曲线及应变曲线
    plt.figure(2)
    plt.plot(x_coor, second_deri, 'g', lw=2.0, label="second_deri")
    plt.plot(x_coor, strain, 'y', lw=2.0, label="strain")

    plt.legend()
    plt.show()


