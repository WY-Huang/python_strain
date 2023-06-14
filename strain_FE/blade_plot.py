import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def creat_mesh_3D(x, y, nx, ny, e_type):
    '''
    生成节点坐标和单元索引
    x:x方向上的距离
    y:y方向上的距离
    nx:x方向上的element的数量
    ny:y方向上的element的数量
    e_type:SQ表示是矩形单元,TR表示三角单元

    '''
    q = np.array([[0., 0.], [x, 0.], [0, y], [x, y]])   # 矩形单元的四角坐标
    numN = (nx + 1) * (ny + 1)                          # 节点的数量
    numE = nx * ny                                      # 单元的数量
    NofE = 4                                            # 矩形element的角
    D = 3                                               # 三维坐标
    nodeCoor = np.zeros([numN, D])                      # nodes 坐标
    dx = q[1, 0] / nx                                   # dx,dy的计算
    dy = q[2, 1] / ny

    n = 0                                           # nodes 坐标计算
    for i in range(1, ny + 2):
        for j in range(1, nx + 2):
            nodeCoor[n, 0] = q[0, 0] + (j - 1) * dx
            nodeCoor[n, 1] = q[0, 1] + (i - 1) * dy
            nodeCoor[n, 2] = 1 / (1 + np.exp(-nodeCoor[n, 0] - nodeCoor[n, 1]))
            n += 1
    
    elementIndex = np.zeros([numE, NofE])                     # element 索引，一个element由四个角的节点进行索引

    for i in range(1, ny + 1):                      # 矩形单元的划分工作
        for j in range(1, nx + 1):
            # 从底层开始类推
            if j == 1:
                elementIndex[(i - 1) * nx + j - 1, 0] = (i - 1) * (nx + 1) + 1
                elementIndex[(i - 1) * nx + j - 1, 1] = elementIndex[(i - 1) * nx + j - 1, 0] + 1
                elementIndex[(i - 1) * nx + j - 1, 3] = elementIndex[(i - 1) * nx + j - 1, 0] + (nx + 1)
                elementIndex[(i - 1) * nx + j - 1, 2] = elementIndex[(i - 1) * nx + j - 1, 3] + 1
            else:
                elementIndex[(i - 1) * nx + j - 1, 0] = elementIndex[(i - 1) * nx + j - 2, 1]
                elementIndex[(i - 1) * nx + j - 1, 3] = elementIndex[(i - 1) * nx + j - 2, 2]
                elementIndex[(i - 1) * nx + j - 1, 1] = elementIndex[(i - 1) * nx + j - 1, 0] + 1
                elementIndex[(i - 1) * nx + j - 1, 2] = elementIndex[(i - 1) * nx + j - 1, 3] + 1

    # 三角形单元需要将每一个矩形单元进行拆分，即一分二成两个三角形
    if e_type == 'TR':
        NofE_new = 3            # 三角形的三个角
        numE_new = numE * 2     # 单元数量
        
        EI_new = np.zeros([numE_new, NofE_new])     # 新的三角单元索引

        # 对矩形单元进行逐个剖分
        for i in range(1, numE + 1):
            EI_new[2 * (i - 1), 0] = elementIndex[i - 1, 0]
            EI_new[2 * (i - 1), 1] = elementIndex[i - 1, 1]
            EI_new[2 * (i - 1), 2] = elementIndex[i - 1, 2]

            EI_new[2 * (i - 1) + 1, 0] = elementIndex[i - 1, 0]
            EI_new[2 * (i - 1) + 1, 1] = elementIndex[i - 1, 2]
            EI_new[2 * (i - 1) + 1, 2] = elementIndex[i - 1, 3]

        elementIndex = EI_new

    elementIndex = elementIndex.astype(int)
    
    return nodeCoor, elementIndex


def draw_mesh_3D(numNode, numElement, nodesCoor, elementsIndex, element_type, flag, title, color_value_x=None):
    """
    绘制网格线、单元编号、节点编号
    """
    if flag == "init_mesh":
        fig = plt.figure(figsize=(10, 8))
        ax3d = fig.add_subplot(projection="3d")

        count = 1
        # plot nodes num
        for i in range(numNode):
            ax3d.text(nodesCoor[i, 0], nodesCoor[i, 1], nodesCoor[i, 2], count, ha='center', va='center', color="g")
            count += 1

        if element_type == 'SQ':
            count2 = 1
            for i in range(numElement):
                # 计算中点位置，绘制单元编号
                plt.annotate(count2, xy=((nodesCoor[elementsIndex[i, 0] - 1, 0] + nodesCoor[elementsIndex[i, 1] - 1, 0]) / 2,
                                        (nodesCoor[elementsIndex[i, 0] - 1, 1] + nodesCoor[elementsIndex[i, 3] - 1, 1]) / 2),
                            c='blue')
                count2 += 1
                # 绘制节点连线
                x0, y0 = nodesCoor[elementsIndex[i, 0] - 1, 0], nodesCoor[elementsIndex[i, 0] - 1, 1]
                x1, y1 = nodesCoor[elementsIndex[i, 1] - 1, 0], nodesCoor[elementsIndex[i, 1] - 1, 1]
                x2, y2 = nodesCoor[elementsIndex[i, 2] - 1, 0], nodesCoor[elementsIndex[i, 2] - 1, 1]
                x3, y3 = nodesCoor[elementsIndex[i, 3] - 1, 0], nodesCoor[elementsIndex[i, 3] - 1, 1]
                plt.plot([x0, x1], [y0, y1], c='red', linewidth=2)
                plt.plot([x0, x3], [y0, y3], c='red', linewidth=2)
                plt.plot([x1, x2], [y1, y2], c='red', linewidth=2)
                plt.plot([x2, x3], [y2, y3], c='red', linewidth=2)

        if element_type == 'TR':
            count2 = 1
            for i in range(numElement):
                # 计算中点位置，绘制单元编号
                x_coor, y_coor, z = ((nodesCoor[elementsIndex[i, 0] - 1, 0] + nodesCoor[elementsIndex[i, 1] - 1, 0] + nodesCoor[elementsIndex[i, 2] - 1, 0]) / 3,
                          (nodesCoor[elementsIndex[i, 0] - 1, 1] + nodesCoor[elementsIndex[i, 1] - 1, 1] + nodesCoor[elementsIndex[i, 2] - 1, 1]) / 3,
                          (nodesCoor[elementsIndex[i, 0] - 1, 2] + nodesCoor[elementsIndex[i, 1] - 1, 2] + nodesCoor[elementsIndex[i, 2] - 1, 2]) / 3)
                ax3d.text(x_coor, y_coor, z, count2, ha='center', va='center', color="b")
                count2 += 1

                # 绘制节点连线
                x0, y0, z0 = nodesCoor[elementsIndex[i, 0] - 1, 0], nodesCoor[elementsIndex[i, 0] - 1, 1], nodesCoor[elementsIndex[i, 0] - 1, 2]
                x1, y1, z1 = nodesCoor[elementsIndex[i, 1] - 1, 0], nodesCoor[elementsIndex[i, 1] - 1, 1], nodesCoor[elementsIndex[i, 1] - 1, 2]
                x2, y2, z2 = nodesCoor[elementsIndex[i, 2] - 1, 0], nodesCoor[elementsIndex[i, 2] - 1, 1], nodesCoor[elementsIndex[i, 2] - 1, 2]
                plt.plot([x0, x1], [y0, y1], [z0, z1], c='red', linewidth=2)      
                plt.plot([x1, x2], [y1, y2], [z1, z2], c='red', linewidth=2)
                plt.plot([x0, x2], [y0, y2], [z0, z2], c='red', linewidth=2)

    elif flag == "strain_mesh":
        fig = plt.figure(title, figsize=(10, 8))
        ax3d = fig.add_subplot(projection="3d")

        tri = elementsIndex - 1     # 从0开始的单元索引（200, 3）
        vertices = nodesCoor[tri]
        # print(vertices)
        # 给每一个三角形添加颜色
        poly3d = Poly3DCollection(vertices, facecolors=color_value_x)      # , edgecolors='black', plt.cm.autumn(color_value_x[i])
        ax3d.add_collection3d(poly3d)
        ax3d.set_xlim([0, 2])
        ax3d.set_ylim([0, 4])
        ax3d.set_zlim([0, 1])

    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_box_aspect([5, 10, 2])

    plt.tight_layout()
    # plt.show()


def normalization(color_value):
    '''
    将应变值归一化，用于绘制热力图
    '''
    color_rgba = np.zeros((len(color_value), 4))
    x_max = color_value.max()
    x_min = color_value.min()
    for i, value in enumerate(color_value):
        color_value[i] = value / (x_max - x_min)
        color_rgba[i] = plt.cm.autumn(color_value_x[i])

    return color_rgba


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
    normal_y = np.cross(normal_z, normal_x) # 叉乘ab，得到垂直于ab的c向量
    print("nx, ny, nz:\n", normal_x, normal_y, normal_z)

    return normal_x, normal_y, normal_z


def calCosine(norm_x, norm_y, norm_z):
    """
    通过方向余弦，计算新坐标系与原坐标系的旋转矩阵
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
    x1 = node_coor[0][0]            # 三个节点坐标
    y1 = node_coor[0][1]
    x2 = node_coor[1][0]
    y2 = node_coor[1][1]
    x3 = node_coor[2][0]
    y3 = node_coor[2][1]

    a1 = x2 * y3 - x3 * y2          # 形函数系数（a, b, c）
    b1 = y2 - y3
    c1 = x3 - x2

    a2 = x3 * y1 - x1 * y3
    b2 = y3 - y1
    c2 = x1 - x3

    a3 = x1 * y2 - x2 * y1
    b3 = y1 - y2
    c3 = x2 - x1

    A = (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2) / 2     # 三角形单元面积A

    B_mat = np.matrix([[b1, 0, b2, 0, b3, 0],                               # 应变矩阵B
                       [0, c1, 0, c2, 0, c3],
                       [c1, b1, c2, b2, c3, b3]])

    B = (1 / (2 * A)) * B_mat

    U_array = node_displace.ravel()     # 位移矩阵U
    U = np.asmatrix(U_array)

    strain_conp = np.dot(B, U.T)        # 应变分量（Ex, Ey, Rxy）

    # print(f"Ex = {strain_conp[0]}\nEy = {strain_conp[1]}\nRxy = {strain_conp[2]}")
    return strain_conp


def calGlobalElementStrain(node1, node2, node3, dis1, dis2, dis3):
    """
    根据旋转矩阵（3*3）计算局部坐标系下的节点坐标<x,y,z>与位移<向量>（3*1）->（3*6）
    """
    # nodeDisArr = np.array([])
    nx, ny, nz = CalNormal3D(node1, node2, node3)   # 计算局部坐标系三个轴向量
    rotate_arr = calCosine(nx, ny, nz)              # 计算旋转矩阵

    p1New = np.matmul(rotate_arr, node1)            # 局部坐标系下新坐标
    p2New = np.matmul(rotate_arr, node2)
    p3New = np.matmul(rotate_arr, node3)
    print("p1New, p2New, p3New:\n", p1New, p2New, p3New)

    d1New = np.matmul(rotate_arr, dis1)             # 局部坐标系下新位移
    d2New = np.matmul(rotate_arr, dis2)
    d3New = np.matmul(rotate_arr, dis3)
    print("d1New, d2New, d3New:\n", d1New, d2New, d3New)

    u1x = np.dot(d1New, [1, 0, 0])                  # 新位移在新坐标系xy平面上的投影
    v1y = np.dot(d1New, [0, 1, 0])
    u2x = np.dot(d2New, [1, 0, 0])
    v2y = np.dot(d2New, [0, 1, 0])
    u3x = np.dot(d3New, [1, 0, 0])
    v3y = np.dot(d3New, [0, 1, 0])

    nodeCoorNew = np.array([p1New, p2New, p3New])   # 局部坐标系下新坐标及位移
    disNew = np.array([u1x, v1y, u2x, v2y, u3x, v3y])

    strainCon = strain_compute(nodeCoorNew, disNew) # 局部应变计算
    print("strainCon:\n", strainCon)

    strainLocal = strainCon
    strainLocal[2] = 0
    R_inv = np.linalg.inv(rotate_arr)               # 局部到全局的变换矩阵R_inv
    strainConGlobal = np.matmul(R_inv, strainLocal)
    strainConGlobal[2] = strainConGlobal[0] + strainConGlobal[1]
    print("strainConGlobal:\n", strainConGlobal)

    return strainConGlobal


def test_CalNormal3D(p1, p2, p3):
    """
    测试CalNormal3D的返回向量
    """
    nx, ny, nz = CalNormal3D(p1, p2, p3)    # 计算局部坐标系三个轴向量

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
    ax.text(nx[0], nx[1], nx[2], "X\'", ha='center', va='center', color="r")
    ax.text(ny[0], ny[1], ny[2], "Y\'", ha='center', va='center', color="g")
    ax.text(nz[0], nz[1], nz[2], "Z\'", ha='center', va='center', color="b")

    # ax.set_xlim(0, 2)
    # ax.set_ylim(0, 2)
    # ax.set_zlim(0, 2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == "__main__":
    """
    p1 = np.array([1, 0, 0])        # 节点坐标
    p2 = np.array([0, 1, 0])
    p3 = np.array([0, 0, 1])

    d1 = np.random.randn(3) / 20    # 节点位移
    d2 = np.random.randn(3) / 20
    d3 = np.random.randn(3) / 20
    
    draw_tri_grid()               # 绘制三角网格图

    # test_CalNormal3D(p1, p2, p3)            # 测试局部坐标系变换

    StrainGlobal = calGlobalElementStrain(p1, p2, p3, d1, d2, d3)      # 计算单个单元的全局坐标系下的应变分量
    """

    # 3D网格参数设置
    x, y = 2, 4
    nx = 10
    ny = 10
    element_type = 'TR'
    nodesCoor, elementsIndex = creat_mesh_3D(x, y, nx, ny, element_type)
    numN = np.size(nodesCoor, 0)
    numE = np.size(elementsIndex, 0)

    # draw_mesh_3D(numN, numE, nodesCoor, elementsIndex, element_type, "strain_mesh", "3D mesh generate")

    node_coor_tri = np.zeros([3, 3])
    node_disp_tri = np.zeros([3, 3])
    strain_conp_all = np.zeros([numE, 3, 1])
    for e in range(numE):
        node_1_index = elementsIndex[e][0]-1
        node_2_index = elementsIndex[e][1]-1
        node_3_index = elementsIndex[e][2]-1
        for k in range(3):
            node_coor_tri[k] = nodesCoor[elementsIndex[e][k]-1]
            # node_disp_tri[k] = node_disp_all[elementsIndex[e][k]-1]
        
        d1 = np.random.randn(3) / 20    # 节点位移
        d2 = np.random.randn(3) / 20
        d3 = np.random.randn(3) / 20
        strain_conp_all[e] = calGlobalElementStrain(node_coor_tri[0], node_coor_tri[1], node_coor_tri[2], d1, d2, d3)

    
    # 可视化应变分量
    color_value_x = strain_conp_all[:, 0, 0]
    color_x = normalization(color_value_x)

    color_value_y = strain_conp_all[:, 1, 0]
    color_y = normalization(color_value_y)

    color_value_xy = strain_conp_all[:, 2, 0]
    color_xy = normalization(color_value_xy)

    draw_mesh_3D(numN, numE, nodesCoor, elementsIndex, element_type, "strain_mesh", "color_value_x", color_x)
    draw_mesh_3D(numN, numE, nodesCoor, elementsIndex, element_type, "strain_mesh", "color_value_y", color_y)
    draw_mesh_3D(numN, numE, nodesCoor, elementsIndex, element_type, "strain_mesh", "color_value_xy", color_xy)

    plt.show()