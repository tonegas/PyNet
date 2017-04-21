from mnist_load import load_mnist_dataset
import numpy as np
import matplotlib.pyplot as plt
from standart_network.kohonen import Kohonen


def plot_weights(model,ind):
    fig = plt.figure(ind)
    for i,img in enumerate(model.W):
        ax = fig.add_subplot(10,10,i+1)
        img = ax.imshow(img.reshape(28,28), cmap=plt.get_cmap('Greys'))


train = load_mnist_dataset("training","mnist")

epochs = 6
start_learning_rate = 0.15
stop_learning_rate = 0.05
start_radius = 5
stop_radius = 2

model = Kohonen(
    input_size = 784,
    output_size = 100,
    topology = (10,10,False),
    weights = 'norm_random',
    learning_rate = start_learning_rate,
    radius = start_radius
)

for epoch in range(epochs):
    print epoch
    np.random.shuffle(train)
    for data in train:
        model.forward(data[0],True)

        model.learning_rate = stop_learning_rate+(start_learning_rate-stop_learning_rate)*(epochs-epoch)/epochs
        model.radius = stop_radius+(start_radius-stop_radius)*(epochs-epoch)/epochs
    plotWeights(model,epoch)

model.save('kohonen.net')

plt.show()