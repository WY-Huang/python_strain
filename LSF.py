import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

SAMPLE_NUM = 100  # 要生成的sample个数
M = 7   # 多项式阶数

# 产生带有高斯噪声的信号
mid, sigma = 0, 0.1 # 设置均值和方差
noise = np.random.normal(mid, sigma, SAMPLE_NUM).reshape(SAMPLE_NUM, 1) # 生成SAMPLE_NUM个数据

# 产生SAMPLE_NUM个序号(范围是0-4pi)
x = np.arange(0, SAMPLE_NUM).reshape(SAMPLE_NUM, 1) / (SAMPLE_NUM-1)*(4*math.pi)

# generate y and y_noise, and both y's and y_noise's shape is (SAMPLE_NUM*1)
y = np.sin(x)
y_noise = np.sin(x) + noise

# 绿色曲线显示x - y，散点显示x - y_noise
plt.title("")
plt.plot(x, y, 'g', lw=2.0, label="sin(x)")
plt.plot(x, y_noise, 'bo', label="noise")
 

# generate Matrix X which has M order (SAMPLE_NUM, M)
X = x   
for i in range(2, M+1):
         X = np.column_stack((X, pow(x, i)))

# add 1 on the first column of X, now X's shape is (SAMPLE_NUM*(M+1))
X = np.insert(X, 0, [1], 1)
# print(X)aQQ

# calculate W, W's shape is ((M+1)*1)
# W=np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y_noise) # have no regularization
W = np.linalg.inv((X.T.dot(X)) + np.exp(-8) * np.eye(M+1)).dot(X.T).dot(y_noise) # introduce regularization
print("W:", W)
y_estimate = X.dot(W)

co_w, resl, _, _ = np.linalg.lstsq(X, y_noise, rcond=None)  # resl为残差的平方和
print("co_w:", co_w, "\nresl", resl)
y_estimate_lstsq = X.dot(W)

yhat = savgol_filter(y_noise.T, 21, 3)    # window size 11, polynomial order 3

#红色曲线显示x - y_estimate
plt.plot(x, y_estimate, 'r', lw=2.0, label="np_inv")
plt.plot(x, y_estimate_lstsq, 'y', lw=2.0, label="np_lstq")
plt.plot(x, yhat.T, 'b', lw=2.0, label="scipy_sg")

plt.legend()
plt.show()  
