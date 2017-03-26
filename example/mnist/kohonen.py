from mnist_load import load_mnist_dataset
import numpy as np
import matplotlib.pyplot as plt
from classicnetwork import Kohonen


def plotWeights(model,ind):
    fig = plt.figure(ind)
    for i,img in enumerate(model.W):
        ax = fig.add_subplot(10,10,i+1)
        img = ax.imshow(img.reshape(28,28), cmap=plt.get_cmap('Greys'))


train = load_mnist_dataset("training","mnist")

start_learning_rate = 0.15
total_iterations = 5
start_radius = 2.1

model = Kohonen(
    input_size = 784,
    output_size = 100,
    topology = (10,10,False),
    weights = 'norm_random',
    learning_rate = start_learning_rate,
    radius = start_radius
)

for i in range(total_iterations):
    np.random.shuffle(train)
    for data in train:
        model.forward(data[0],True)

    model.learning_rate -= 0.01
    model.radius -= 0.05
    plotWeights(model,i)

model.save('kohonen.net')

plt.show()