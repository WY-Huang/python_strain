import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d


def CalNormal3D(point_1, point_2, point_3):
    """
    计算三点组成新平面的法向量normal_z，x轴向量normal_x，y轴向量normal_y

    平面方程: na * (x – n1) + nb * (y – n2) + nc * (z – n3) = 0;
    double na = (v2.y - v1.y)*(v3.z - v1.z) - (v2.z - v1.z)*(v3.y - v1.y);
    double nb = (v2.z - v1.z)*(v3.x - v1.x) - (v2.x - v1.x)*(v3.z - v1.z);
    double nc = (v2.x - v1.x)*(v3.y - v1.y) - (v2.y - v1.y)*(v3.x - v1.x);
    """
    na = (point_2[1] - point_1[1]) * (point_3[2] - point_1[2]) - (point_2[2] - point_1[2]) * (point_3[1] - point_1[1])
    nb = (point_2[2] - point_1[2]) * (point_3[0] - point_1[0]) - (point_2[0] - point_1[0]) * (point_3[2] - point_1[2])
    nc = (point_2[0] - point_1[0]) * (point_3[1] - point_1[1]) - (point_2[1] - point_1[1]) * (point_3[0] - point_1[0])
    normal_z = np.array([na, nb, nc])

    normal_x = point_2 - point_1
    normal_y = np.cross(normal_z, normal_x)

    return normal_x, normal_y, normal_z


def calCosine(norm_x, norm_y, norm_z):
    """
    计算新坐标系与原坐标系的旋转矩阵
    """
    norm_arr = np.array([norm_x, norm_y, norm_z])
    rotate_arr = np.empty([3, 3])
    for i in range(len(norm_arr)):
        for n in range(3):
            rotate_arr[i][n] =  norm_arr[i][n] / (np.linalg.norm(norm_arr[i]))
    
    print("rotate_arr:\n", rotate_arr)

    return rotate_arr


def strain_compute(node_coor, node_displace):
    """
    根据节点坐标和位移计算应变
    """

    # 三个节点坐标
    x1 = node_coor[0][0]
    y1 = node_coor[0][1]
    x2 = node_coor[1][0]
    y2 = node_coor[1][1]
    x3 = node_coor[2][0]
    y3 = node_coor[2][1]

    # 形函数系数（a, b, c）
    a1 = x2 * y3 - x3 * y2
    b1 = y2 - y3
    c1 = x3 - x2

    a2 = x3 * y1 - x1 * y3
    b2 = y3 - y1
    c2 = x1 - x3

    a3 = x1 * y2 - x2 * y1
    b3 = y1 - y2
    c3 = x2 - x1

    # 三角形单元面积A
    A = (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2) / 2

    # 应变矩阵B
    B_mat = np.matrix([[b1, 0, b2, 0, b3, 0],
                       [0, c1, 0, c2, 0, c3],
                       [c1, b1, c2, b2, c3, b3]])

    B = (1 / (2 * A)) * B_mat

    # 位移矩阵U
    U_array = node_displace.ravel()
    U = np.asmatrix(U_array)

    # 应变分量（Ex, Ey, Rxy）
    strain_conp = np.dot(B, U.T)

    # print(f"Ex = {strain_conp[0]}\nEy = {strain_conp[1]}\nRxy = {strain_conp[2]}")
    return strain_conp


def test_CalNormal3D():
    """
    测试CalNormal3D的返回向量
    """
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 1, 0])
    p3 = np.array([0, 0, 1])

    nx, ny, nz = CalNormal3D(p1, p2, p3)
    print("nx, ny, nz:\n", nx, ny, nz)

    rotate_arr = calCosine(nx, ny, nz)
    p1New = np.matmul(rotate_arr, p1)
    p2New = np.matmul(rotate_arr, p2)
    p3New = np.matmul(rotate_arr, p3)
    print("p1New, p2New, p3New:\n", p1New, p2New, p3New)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # 绘制平面
    ax.plot([1,0],[0,1],[0,0],color="y")
    ax.plot([0,0],[1,0],[0,1],color="y")
    ax.plot([0,1],[0,0],[1,0],color="y")
    # ax.quiver(1,0,0,0,1,0,arrow_length_ratio=0.1, color="y")
    # ax.quiver(0,1,0,0,0,1,arrow_length_ratio=0.1, color="y")
    # ax.quiver(0,0,1,1,0,0,arrow_length_ratio=0.1, color="y")

    # 绘制轴向量,前三个值为起点，后三个值为相对起点的位移量
    ax.quiver(0,0,0,nx[0],nx[1],nx[2],length=0.8,arrow_length_ratio=0.1, color="r")
    ax.quiver(0,0,0,ny[0],ny[1],ny[2],length=0.8,arrow_length_ratio=0.1, color="g")
    ax.quiver(0,0,0,nz[0],nz[1],nz[2],length=0.8,arrow_length_ratio=0.1, color="b")
    # 绘制文本
    ax.text(nx[0], nx[1], nx[2], "X", ha='center', va='center', color="r")
    ax.text(ny[0], ny[1], ny[2], "Y", ha='center', va='center', color="g")
    ax.text(nz[0], nz[1], nz[2], "Z", ha='center', va='center', color="b")

    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    # ax.set_zlim(0, 2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def draw_tri_grid():
    """
    绘制三角网格
    """
    np.random.seed(20230608)

    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(projection="3d")

    xdata = np.linspace(-2.5, 2.5, 10)
    ydata = np.linspace(-5, 5, 10)
    # z_data = (1/(1+np.exp(-xdata-ydata)))
    X, Y = np.meshgrid(xdata, ydata)
    Z = 1/(1+np.exp(-X-Y))
    gaussian = np.exp(-((pow(X, 2) + pow(Y, 2)) / pow(5, 2)))

    random_array = np.random.randn(10, 10) / 20
    x_d = X + random_array
    y_d = Y + random_array
    z_d = Z + random_array

    # ax3d = plt.axes(projection='3d')
    # ax3d.plot_surface(X, Y, Z)  # , cmap='plasma'
    # 初始数据
    ax3d.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(), edgecolor="r", alpha=0.2)  # , cmap='gist_rainbow_r'
    ax3d.scatter3D(X.reshape(-1), Y.reshape(-1), Z.reshape(-1), marker='o', s=30, color='b')

    # 位移后数据
    # ax3d.plot_trisurf(x_d.ravel(), y_d.ravel(), z_d.ravel(), facecolor="yellow", edgecolor="green", alpha=0.2)
    # ax3d.scatter3D(x_d.reshape(-1), y_d.reshape(-1), z_d.reshape(-1), marker='o', s=30, color='b')

    # 应变数据
    # ax3d.plot_trisurf(X.ravel(), Y.ravel(), gaussian.ravel(), cmap='jet')  # , cmap='gist_rainbow_r'
    # ax3d.scatter3D(X.reshape(-1), Y.reshape(-1), Z.reshape(-1), marker='o', s=30, color='b')

    # ax3d.set_title('Surface Plot in Matplotlib')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_box_aspect([5, 10, 2])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 绘制三角网格图
    # draw_tri_grid()

    test_CalNormal3D()