import numpy as np
import matplotlib.pyplot as plt
import pyuff


uff_file = pyuff.UFF('/home/wanyel/vs_code/python_strain/bending_strain/test_uff/export_time_500mVpp.uff')
data_type = uff_file.get_set_types()    # 文件中存放的数据类型
print(data_type)
data_num = uff_file.get_n_sets()        # 有效数据集的数量

print_str = "3".split()

if '1' in print_str:
    data_1 = uff_file.read_sets(0)            # 第1个数据类型：151，文件头，包含uff文件的基本信息
    print(data_1)

if '2' in print_str:
    data_2 = uff_file.read_sets(1)            # 第2个数据类型：164，数据的单位信息
    print(data_2)

if '3' in print_str:
    data_3 = uff_file.read_sets(2)            # 第3个数据类型：2411，节点编号及对应的坐标x, y, z数据
    print(data_3)

if '4' in print_str:
    data_4 = uff_file.read_sets(3)            # 第4个数据类型：82，结构的几何连线信息
    print(data_4)

if '5' in print_str:
    data_5 = uff_file.read_sets(4)            # 第5个数据类型：2412，单元编号对应的各顶点的节点号
    print(data_5)

if len(print_str) == 0:
    data_all = uff_file.read_sets()           # 读取所有数据类型的数据
    print(data_all)