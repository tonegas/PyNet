from mnist_load import load_mnist_dataset
import numpy as np
import matplotlib.pyplot as plt
from layers import NormalizationLayer, LinearLayer, TanhLayer, ReluLayer, SigmoidLayer
from network import Sequential
from genericlayer import StoreNetwork
from trainer import Trainer
from losses import NegativeLogLikelihoodLoss, CrossEntropyLoss, SquaredLoss, to_one_hot_vect
from optimizers import GradientDescent, GradientDescentMomentum
from printers import ShowTraining


from standart_network.autoencoder import AutoEncoder


num_classes = 10
name_net = "mnist.net"
load_net = False
epochs = 200


train = load_mnist_dataset(dataset = "training", path = "./mnist")
test = load_mnist_dataset(dataset = "testing", path = "./mnist")


if load_net:
    print "Load Network"
    model = StoreNetwork.load(name_net)
else:
    print "New Network"
    #Two layer network
    ae = AutoEncoder(784, [
        {"size" : 32, "output_layer" :TanhLayer},
        {"size" : 784, "output_layer" :TanhLayer}
    ])
    model = Sequential([
        NormalizationLayer(0,255,-0.1,0.1),
        ae,
        NormalizationLayer(-1,1,0,255),
    ])

train = []
ax.imshow(img.reshape(28,28), cmap=plt.get_cmap('Greys'))

display = ShowTraining(epochs_num = epochs)

trainer = Trainer(show_training = False, show_function = display.show)

J_list, dJdy_list, J_test = trainer.learn(
    model = model,
    train = train,
    # loss = NegativeLogLikelihoodLoss(),
    loss = CrossEntropyLoss(),
    # loss = SquaredLoss(),
    # optimizer = GradientDescent(learning_rate=0.3),
    optimizer = GradientDescentMomentum(learning_rate=0.35/10, momentum=0.5),
    epochs = epochs,
    batch_size = 10
)



raw_input('Press ENTER to exit')

model.save('model.net')