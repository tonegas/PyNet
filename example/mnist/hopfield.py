from mnist_load import load_mnist_dataset
import numpy as np
import matplotlib.pyplot as plt
from classicnetwork import Hopfield
from layers import NormalizationLayer, SignLayer
from network import Sequential


def img_compare(image1,image2):
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

for i in range(10):
    mean_val[i] = mean_val[i]/tot_val[i]
    # plt.imshow(mean_val[i].reshape(28,28),cmap=plt.get_cmap('Greys'))
    # plt.show()

hop_net = Hopfield(784)
normalization_net = Sequential(
    NormalizationLayer(0,255,-1,1),
    SignLayer,
)

stored_numers = [3,7] #numbers stored in the network

for i in stored_numers:
    hop_net.store(normalization_net.forward(mean_val[i]))

model = Sequential(
    normalization_net,
    hop_net
)

for (x,t) in train:
    y = model.forward(x)
    img_compare(x.reshape(28,28),y.reshape(28,28))