from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt


def read_data(data_path=None, sample_num=25, plate_length=96):
    """
    读取位移数据,统一位移和x坐标的单位为mm
    """
    x_coor = np.arange(0, sample_num).reshape(sample_num, 1) * (plate_length / sample_num)

    dis_data = np.loadtxt(data_path)
    dis_data_mm = dis_data[10, 3::2] / 1000

    return x_coor, dis_data_mm


def dis_visualization_3d(flag=None, random_ge=True):
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


if __name__ == "__main__":
    # 读取数据
    # x_coor, dis_data_mm = read_data('bending_strain/dis_data_20230223/dis_data_all.txt', 55, 110)

    # 绘制原始位移散点及拟合后的位移曲线
    dis_visualization_3d("wireframe")


    # 绘制应变图


