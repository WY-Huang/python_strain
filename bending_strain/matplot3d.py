
# import matplotlib
# print(matplotlib.__version__)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import timeit


def parametric_curve():
    """
    3d参数曲线的绘制
    """
    ax = plt.figure().add_subplot(projection='3d')

    # Prepare arrays x, y, z
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    ax.plot(x, y, z, label='parametric curve')
    ax.legend()

    plt.show()


def scatterplot():
    """
    3D scatterplot
    """
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
        xs = randrange(n, 23, 32)
        ys = randrange(n, 0, 100)
        zs = randrange(n, zlow, zhigh)
        ax.scatter(xs, ys, zs, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin


def wireframe():
    """
    3D wireframe plot
    """
    from mpl_toolkits.mplot3d import axes3d

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Grab some test data.
    X, Y, Z = axes3d.get_test_data(0.05)

    # Plot a basic wireframe.
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    plt.show()


def test_wireframe():
    """
    
    """
    x = np.arange(0, 5, 1)
    y = np.arange(0, 5, 1)
    X, Y = np.meshgrid(x, y)

    Z = np.add(np.power(X, 1), np.power(X, 1))

    print(X)
    print(Y)
    print(Z)

def surface_plot():
    """
    3d surface plot
    """
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


# 创建一个函数模型用来生成数据
def func1(x, a, b, c, d):
    r = a * np.exp(-((x[0] - b) ** 2 + (x[1] - d) ** 2) / (2 * c ** 2))
    return r.ravel()
 

if __name__ == "__main__":
    # 1> 3d参数曲线的绘制
    # parametric_curve()

    # 2> 3d散点图的绘制
    # scatterplot()

    # 3> 3d线框图绘制
    # wireframe()
    # test_wireframe()

    # 3> 3d曲面图绘制
    # surface_plot()

    # 数据拟合及显示
    # 生成原始数据
    x1 = np.linspace(0, 10, 10).reshape(1, -1)
    x2 = np.linspace(0, 10, 10).reshape(1, -1)
    x = np.append(x1, x2, axis=0)
    X, Y = np.meshgrid(x1, x2)
    XX = np.expand_dims(X, 0)
    YY = np.expand_dims(Y, 0)
    xx = np.append(XX, YY, axis=0)
    y = func1(xx, 10, 5, 2, 5)
    # 对原始数据增加噪声
    yn = y + 0.002 * np.random.normal(size=xx.shape[1] * xx.shape[2])
 
    # 使用curve_fit函数拟合噪声数据
    t0 = timeit.default_timer()
    popt, pcov = curve_fit(func1, xx, yn)
    elapsed = timeit.default_timer() - t0
    print('Time: {} s'.format(elapsed))
 
    # popt返回最拟合给定的函数模型func的参数值
    print(popt)
 
    fig = plt.figure('拟合图')
    ax = Axes3D(fig)
    X, Y = np.meshgrid(x1, x2)
    XX = np.expand_dims(X, 0)
    YY = np.expand_dims(Y, 0)
    xx = np.append(XX, YY, axis=0)
    R = func1(xx, *popt)
    # R, _ = np.meshgrid(R, x1)
    # y = func1(xx, 10, 5, 2, 5)
    # # 对原始数据增加噪声
    # yn = y + 0.002 * np.random.normal(size=xx.shape[1] * xx.shape[2])
    R = R.reshape(10, 10)
    yn = yn.reshape(10, 10)
    ax.plot_surface(X, Y, R, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_surface(X, Y, yn, rstride=1, cstride=1, color='red')
 
    plt.show()
    y_predict_1 = func1(x, *popt)
    indexes_1 = getIndexes(y_predict_1, yn)
    print(indexes_1)

