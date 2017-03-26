import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from classicnetwork import Kohonen
from printers import Printer2D
from sklearn import datasets

classes = ["o","v","x",".","s"]
colors = ['r', 'g', 'b', 'y', 'o']
num_classes = 4
n = 200
epochs = 30

start_learning_rate = 0.15
stop_learning_rate = 0.05
total_iterations = 5
start_radius = 3
stop_radius = 1


model = Kohonen(
    input_size = 2,
    output_size = 25,
    topology = (5,5,False),
    weights = 'random',
    learning_rate = start_learning_rate,
    radius = start_radius
)


train = [((x,y),1) for x in np.linspace(-5,5,5) for y in np.linspace(-5,5,5)]

def data_gen(t=0):
    for epoch in range(epochs):
        np.random.shuffle(train)
        for data in train:
            model.forward(data[0],True)
            yield (model.W,data[0])
        model.learning_rate = stop_learning_rate+(start_learning_rate-stop_learning_rate)*(epochs-epoch)/epochs
        print model.learning_rate
        model.radius = stop_radius+(start_radius-stop_radius)*(epochs-epoch)/epochs

fig, ax = plt.subplots()

ax.grid()
xdata, ydata = [x for (x,y) in model.W], [y for (x,y) in model.W]
points, = plt.plot(xdata, ydata, color='r', marker='s', linestyle='none')
value, = ax.plot([x for ((x,y),t) in train], [y for ((x,y),t) in train],  color='g', marker='o', linestyle='none')
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)

def run(data):
    # update the data
    # print data
    xdata = [x for (x,y) in data[0]]
    ydata = [y for (x,y) in data[0]]
    # xmin, xmax = ax.get_xlim()

    # if t >= xmax:
    #     ax.set_xlim(xmin, 2*xmax)
    #     ax.figure.canvas.draw()
    points.set_xdata(xdata)
    points.set_ydata(ydata)

    return points

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10, repeat=False)
plt.show()