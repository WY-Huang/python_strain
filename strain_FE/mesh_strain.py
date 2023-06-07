import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Polygon

def node_displacement(node_num):
    """
    生成节点位移x（0-10um）, y(0-20um)
    """
    node_disp = np.zeros([node_num, 2])
    # x_coor = np.linspace(0., 0.01, node_num)
    # y_coor = np.linspace(0., 0.02, node_num)

    x_coor = np.random.random(node_num) / 100
    y_coor = np.random.random(node_num) / 50

    for i in range(node_num):
        node_disp[i][0] = x_coor[i]
        node_disp[i][1] = y_coor[i]
    
    return np.sort(node_disp, axis=0)

def creat_mesh(x, y, nx, ny, e_type):
    '''
    生成节点坐标和单元索引
    x:x方向上的距离
    y:y方向上的距离
    nx:x方向上的element的数量
    ny:y方向上的element的数量
    e_type:SQ表示是矩形单元,TR表示三角单元

    '''
    # 矩形单元的四角坐标
    q = np.array([[0., 0.], [x, 0.], [0, y], [x, y]])
    # node的数量
    numN = (nx + 1) * (ny + 1)
    # element的数量
    numE = nx * ny
    # 矩形element的角
    NofE = 4
    # 二维坐标
    D = 2
    # nodes 坐标
    NC = np.zeros([numN, D])
    # dx,dy的计算
    dx = q[1, 0] / nx
    dy = q[2, 1] / ny
    # nodes 坐标计算
    n = 0
    for i in range(1, ny + 2):
        for j in range(1, nx + 2):
            NC[n, 0] = q[0, 0] + (j - 1) * dx
            NC[n, 1] = q[0, 1] + (i - 1) * dy

            n += 1
    # element 索引，一个element由四个角的节点进行索引
    EI = np.zeros([numE, NofE])

    for i in range(1, ny + 1):
        for j in range(1, nx + 1):
            # 从底层开始类推
            if j == 1:
                EI[(i - 1) * nx + j - 1, 0] = (i - 1) * (nx + 1) + 1
                EI[(i - 1) * nx + j - 1, 1] = EI[(i - 1) * nx + j - 1, 0] + 1
                EI[(i - 1) * nx + j - 1, 3] = EI[(i - 1) * nx + j - 1, 0] + (nx + 1)
                EI[(i - 1) * nx + j - 1, 2] = EI[(i - 1) * nx + j - 1, 3] + 1
            else:
                EI[(i - 1) * nx + j - 1, 0] = EI[(i - 1) * nx + j - 2, 1]
                EI[(i - 1) * nx + j - 1, 3] = EI[(i - 1) * nx + j - 2, 2]
                EI[(i - 1) * nx + j - 1, 1] = EI[(i - 1) * nx + j - 1, 0] + 1
                EI[(i - 1) * nx + j - 1, 2] = EI[(i - 1) * nx + j - 1, 3] + 1
    # 至此完成了矩形单元的划分工作

    # 三角形单元需要将每一个矩形单元进行拆分，即一分二成两个三角形
    if e_type == 'TR':
        # 三角形的三个角
        NofE_new = 3
        # 单元数量
        numE_new = numE * 2
        # 新的三角单元索引
        EI_new = np.zeros([numE_new, NofE_new])

        # 对矩形单元进行逐个剖分
        for i in range(1, numE + 1):
            EI_new[2 * (i - 1), 0] = EI[i - 1, 0]
            EI_new[2 * (i - 1), 1] = EI[i - 1, 1]
            EI_new[2 * (i - 1), 2] = EI[i - 1, 2]

            EI_new[2 * (i - 1) + 1, 0] = EI[i - 1, 0]
            EI_new[2 * (i - 1) + 1, 1] = EI[i - 1, 2]
            EI_new[2 * (i - 1) + 1, 2] = EI[i - 1, 3]

        EI = EI_new
    EI = EI.astype(int)
    return NC, EI

def strain_compute(node_coor, node_displace):
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

def draw_mesh(flag, title, color_value_x=None):
    if flag == "init_mesh":
        plt.figure(title)
        count = 1
        # plot nodes num
        for i in range(numN):
            plt.annotate(count, xy=(NC[i, 0], NC[i, 1]))
            count += 1

        if element_type == 'SQ':
            count2 = 1
            for i in range(numE):
                # 计算中点位置
                plt.annotate(count2, xy=((NC[EI[i, 0] - 1, 0] + NC[EI[i, 1] - 1, 0]) / 2,
                                        (NC[EI[i, 0] - 1, 1] + NC[EI[i, 3] - 1, 1]) / 2),
                            c='blue')
                count2 += 1
                # plot lines
                x0, y0 = NC[EI[i, 0] - 1, 0], NC[EI[i, 0] - 1, 1]
                x1, y1 = NC[EI[i, 1] - 1, 0], NC[EI[i, 1] - 1, 1]
                x2, y2 = NC[EI[i, 2] - 1, 0], NC[EI[i, 2] - 1, 1]
                x3, y3 = NC[EI[i, 3] - 1, 0], NC[EI[i, 3] - 1, 1]
                plt.plot([x0, x1], [y0, y1], c='red', linewidth=2)
                plt.plot([x0, x3], [y0, y3], c='red', linewidth=2)
                plt.plot([x1, x2], [y1, y2], c='red', linewidth=2)
                plt.plot([x2, x3], [y2, y3], c='red', linewidth=2)

        if element_type == 'TR':
            count2 = 1
            for i in range(numE):
                # 计算中点位置
                plt.annotate(count2, xy=((NC[EI[i, 0] - 1, 0] + NC[EI[i, 1] - 1, 0] + NC[EI[i, 2] - 1, 0]) / 3,
                                        (NC[EI[i, 0] - 1, 1] + NC[EI[i, 1] - 1, 1] + NC[EI[i, 2] - 1, 1]) / 3),
                            c='blue')
                count2 += 1
                x0, y0 = NC[EI[i, 0] - 1, 0], NC[EI[i, 0] - 1, 1]
                x1, y1 = NC[EI[i, 1] - 1, 0], NC[EI[i, 1] - 1, 1]
                x2, y2 = NC[EI[i, 2] - 1, 0], NC[EI[i, 2] - 1, 1]
                plt.plot([x0, x1], [y0, y1], c='red', linewidth=2)
                plt.plot([x1, x2], [y1, y2], c='red', linewidth=2)
                plt.plot([x0, x2], [y0, y2], c='red', linewidth=2)
        # plt.xlim(0, x)
        # plt.ylim(0, y)
        plt.axis("tight")   # equal

    elif flag == "strain_mesh":
        fig = plt.figure(title)
        sub = fig.add_subplot(111)
        x = np.squeeze(NC[:, 0])
        y = np.squeeze(NC[:, 1])
        tri = EI - 1
        triang = mtri.Triangulation(x, y, tri)

        # 给每一个三角形添加颜色
        triangles = tri
        for i in range(numE):
            vertices = np.zeros([3,2])
            for j in range(3):
                vertices[j,0] = x[triangles[i,j]]
                vertices[j,1] = y[triangles[i,j]]
            # x_center = (x[triangles[i,0]]+x[triangles[i,1]]+x[triangles[i,2]])/3
            poly = Polygon(vertices, color=plt.cm.autumn(color_value_x[i]))
            sub.add_patch(poly)


        # plt.tricontourf(triang, np.zeros_like(x))
        plt.triplot(triang, 'go-')
        # cbar = fig.colorbar(sub)

    # plt.show()

'''
def calculate_strain(displacements, coordinates, elements):
    """
    计算平面应变
    :param displacements: 位移向量，形如 [u1, v1, u2, v2, ..., un, vn]
    :param coordinates: 节点坐标，形如 [[x1, y1], [x2, y2], ..., [xn, yn]]
    :param elements: 单元信息，形如 [[node1, node2, node3], ..., [nodei, nodej, nodek]]
    :return: 应变矩阵，形如 [[exx1, eyy1, exy1], [exx2, eyy2, exy2], ..., [exxm, eyym, exym]]
    """
    strains = []
    for element in elements:
        # 获取单元上的三个节点的编号
        node1, node2, node3 = element
        # 获取单元上的三个节点的位移
        u1, v1 = displacements[2 * node1:2 * node1 + 2]
        u2, v2 = displacements[2 * node2:2 * node2 + 2]
        u3, v3 = displacements[2 * node3:2 * node3 + 2]
        # 获取单元上的三个节点的坐标
        x1, y1 = coordinates[node1]
        x2, y2 = coordinates[node2]
        x3, y3 = coordinates[node3]
        # 计算单元的位移矩阵
        B = np.array([
            [y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
            [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
            [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]
        ]) / (2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
        # 计算单元的应变
        strain = np.dot(B, np.array([u1, v1, u2, v2, u3, v3]))
        strains.append(strain)
    return strains
'''

def normalization(color_value):
    '''
    将应变值归一化，用于绘制热力图
    '''
    x_max = color_value.max()
    x_min = color_value.min()
    for i, value in enumerate(color_value):
        color_value[i] = value / (x_max - x_min)

    return color_value


if __name__ == "__main__":
    # 网格参数
    x, y = 10, 5
    nx = 10
    ny = 5
    element_type = 'TR'
    NC, EI = creat_mesh(x, y, nx, ny, element_type)
    numN = np.size(NC, 0)
    numE = np.size(EI, 0)

    # 可视化网格
    draw_mesh("init_mesh", "mesh generate")

    node_disp_all = node_displacement(numN)

    print("节点坐标：\n", NC)
    print("单元索引：\n", EI)
    print("节点位移：\n", node_disp_all)

    node_coor_tri = np.zeros([3, 2])
    node_disp_tri = np.zeros([3, 2])
    strain_conp_all = np.zeros([numE, 3, 1])
    for e in range(numE):
        node_1_index = EI[e][0]-1
        node_2_index = EI[e][1]-1
        node_3_index = EI[e][2]-1
        for k in range(3):
            node_coor_tri[k] = NC[EI[e][k]-1]
            node_disp_tri[k] = node_disp_all[EI[e][k]-1]
        
        strain_conp_all[e] = strain_compute(node_coor_tri, node_disp_tri)
    
    print("单元应变：\n", strain_conp_all)
    
    # 可视化应变分量
    color_value_x = strain_conp_all[:, 0, 0]
    color_value_x = normalization(color_value_x)

    color_value_y = strain_conp_all[:, 1, 0]
    color_value_y = normalization(color_value_y)

    color_value_xy = strain_conp_all[:, 2, 0]
    color_value_xy = normalization(color_value_xy)

    draw_mesh("strain_mesh", "strain x", color_value_x)
    draw_mesh("strain_mesh", "strain y", color_value_y)
    draw_mesh("strain_mesh", "strain xy", color_value_xy)

    plt.show()
