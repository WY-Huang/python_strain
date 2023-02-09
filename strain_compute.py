"""
单个三角形单元，根据节点位移及坐标计算应变
"""

import numpy as np

# 三角形单元坐标（x，y）
x1 = 10
y1 = 10
x2 = 11
y2 = 10.5
x3 = 10.5
y3 = 11

# 节点位移（u，v）
u1 = 0.2
v1 = 0.3
u2 = 0.3
v2 = 0.1
u3 = 0.5
v3 = 0.2

# 形函数系数（a, b, c）
a1 = x2 * y3 - x3 * y2
b1 = y2 - y3
c1 = x3 -x2

a2 = x3 * y1 - x1 * y3
b2 = y3 - y1
c2 = x1 -x3

a3 = x1 * y2 - x2 * y1
b3 = y1 - y2
c3 = x2 -x1

# 三角形单元面积A
A = (x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2) / 2

# 应变矩阵B
B_mat = np.matrix([[b1, 0, b2, 0, b3, 0],
                [0, c1, 0, c2, 0, c3],
                [c1, b1, c2, b2, c3, b3]])

B = (1 / (2 * A)) * B_mat

# 位移矩阵U
U = np.matrix([u1, v1, u2, v2, u3, v3])

if __name__ == "__main__":
    # 应变分量（Ex, Ey, Rxy）
    # strain_conp = np.empty((1, 3))
    strain_conp = np.dot(B, U.T)

    print(f"Ex = {strain_conp[0]}\nEy = {strain_conp[1]}\nRxy = {strain_conp[2]}")