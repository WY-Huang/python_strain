import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

np.random.seed(20230608)

fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(projection="3d")

xdata = np.linspace(-2.5, 2.5, 10)
ydata = np.linspace(-5, 5, 10)
# z_data = (1/(1+np.exp(-xdata-ydata)))
X,Y = np.meshgrid(xdata,ydata)
Z = 1/(1+np.exp(-X-Y))

random_array = np.random.randn(10, 10) / 20
x_d = X + random_array
y_d = Y + random_array
z_d = Z + random_array

# ax3d = plt.axes(projection='3d')
# ax3d.plot_surface(X, Y, Z)  # , cmap='plasma'
ax3d.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(), edgecolor="r", alpha=0.2)  # , cmap='gist_rainbow_r'
ax3d.scatter3D(X.reshape(-1), Y.reshape(-1), Z.reshape(-1), marker='o', s=30, color='b')
ax3d.plot_trisurf(x_d.ravel(), y_d.ravel(), z_d.ravel(), facecolor="yellow", edgecolor="green", alpha=0.2)

# ax3d.set_title('Surface Plot in Matplotlib')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_box_aspect([5, 10, 2])

plt.tight_layout()
plt.show()