import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from framework import LinearLayer, UnitStepLayer, SquaredLoss, Sequential

l1 = LinearLayer(2,1,'random')
l2 = UnitStepLayer()
l3 = SquaredLoss()
n = Sequential([l1,l2,l3])

x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)

xs, ys = np.meshgrid(x, y)
# z = calculate_R(xs, ys)
zs = (xs*10 + ys*3) > 0

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='hot')
zvett = np.zeros([x.size,y.size])
for i,xi in enumerate(x):
    for j,yi in enumerate(y):
        zvett[i,j] = n.forward(np.array([xi,yi]))

ax.scatter(xs.reshape((1,xs.size)), ys.reshape((1,ys.size)), zvett.reshape((1,zs.size)), c='r')
plt.show()

#dataVett = [[-1, -1], [0,1],[0,0],[1,1],[1,0],[5,5],[-5,5]]

#plt.scatter([x[0] for x in dataVett],[x[1] for x in dataVett],c=[cVett(t) for t in outputs],s=[dVett(t) for t in targetVett])
