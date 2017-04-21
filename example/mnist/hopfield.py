from mnist_load import load_mnist_dataset
import numpy as np
import matplotlib.pyplot as plt
from standart_network.hopfield import Hopfield
from layers import NormalizationLayer, SignLayer
from network import Sequential


def plot_compare(image1,image2):
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    img1 = ax.imshow(image1, cmap=plt.get_cmap('Greys'))
    ax1 = fig.add_subplot(1,2,2)
    img2 = ax1.imshow(image2, cmap=plt.get_cmap('Greys'))
    plt.show()


train = load_mnist_dataset("training","mnist")

mean_val = [np.zeros(784) for i in range(10)]
tot_val = np.zeros(10)
for x,t in train:
    mean_val[np.argmax(t)] += x
    tot_val[np.argmax(t)] += 1

normalization_net = Sequential(
    NormalizationLayer(0,255,-1,1),
    SignLayer,
)

for i in range(10):
    mean_val[i] = mean_val[i]/tot_val[i]
    num = mean_val[i].reshape(28,28)
    plt.imshow(normalization_net.forward(num),cmap=plt.get_cmap('Greys'))
    # plt.imshow(num))
    plt.show()

hop_net = Hopfield(784)


stored_numers = [0,1] #numbers stored in the network

for i in stored_numers:
    hop_net.store(normalization_net.forward(mean_val[i]))

model = Sequential(
    normalization_net,
    hop_net
)

for (x,t) in train:
    y = model.forward(x)
    plot_compare(x.reshape(28,28),y.reshape(28,28))