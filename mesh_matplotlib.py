# -*- coding: utf-8 -*-
'''
该程序用于创建二维有限元标准三角网格程序，其中利用了四角网格的剖分
来建立三角网格，计算得到代表网格node坐标列表，列表的索引为node 的
索引，得到element节点索引列表。
@作者：fm
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def creat_mesh(x, y, nx, ny, e_type):
    '''
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


if __name__ == "__main__":
    x, y = 1, 1
    nx = 5
    ny = 4
    element_type = 'TR'
    NC, EI = creat_mesh(x, y, nx, ny, element_type)

    print("节点坐标：", NC)
    print("单元索引：", EI)

    numN = np.size(NC, 0)
    numE = np.size(EI, 0)
    plt.figure(1)
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
            plt.plot([x0, x1], [y0, y1], c='red', linewidth=3)
            plt.plot([x0, x3], [y0, y3], c='red', linewidth=3)
            plt.plot([x1, x2], [y1, y2], c='red', linewidth=3)
            plt.plot([x2, x3], [y2, y3], c='red', linewidth=3)

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
            plt.plot([x0, x1], [y0, y1], c='red', linewidth=3)
            plt.plot([x1, x2], [y1, y2], c='red', linewidth=3)
            plt.plot([x0, x2], [y0, y2], c='red', linewidth=3)
    # plt.xlim(0, x)
    # plt.ylim(0, y)
    plt.axis("equal")
    # plt.show()

    plt.figure(2)
    x = np.squeeze(NC[:, 0])
    y = np.squeeze(NC[:, 1])
    tri = EI - 1
    triang = mtri.Triangulation(x, y, tri)
    plt.tricontourf(triang, np.zeros_like(x))
    plt.triplot(triang, 'go-')
    plt.show()

