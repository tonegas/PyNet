from mnist_load import load_mnist_dataset
import numpy as np
import matplotlib.pyplot as plt
from layers import NormalizationLayer, LinearLayer, TanhLayer, ReluLayer, SigmoidLayer
from network import Sequential
from genericlayer import StoreNetwork
from trainer import Trainer
from losses import NegativeLogLikelihoodLoss, CrossEntropyLoss, SquaredLoss, to_one_hot_vect
from optimizers import GradientDescent, GradientDescentMomentum, AdaGrad
from printers import ShowTraining

from standart_network.autoencoder import AutoEncoder

num_classes = 10
name_net = "mnist.net"
load_net = False
epochs = 10


train = load_mnist_dataset(dataset = "training", path = "./mnist")
test = load_mnist_dataset(dataset = "testing", path = "./mnist")

train = [(t,t) for (t,v) in train]
test = [(t,t) for (t,v) in test]


if load_net:
    print "Load Network"
    model = StoreNetwork.load(name_net)
else:
    print "New Network"
    ae = AutoEncoder(784,[
        {'size' : 32, 'output_layer' : ReluLayer},
        {'size' : 784, 'output_layer' : SigmoidLayer},
    ])
    ae.choose_network()
    model = Sequential(
        NormalizationLayer(0,255,-0.1,0.1),
        ae,
        NormalizationLayer(0,1,0,255),
    )

display = ShowTraining(epochs_num = epochs)

trainer = Trainer(show_training = True, show_function = display.show)

J_list, dJdy_list, J_test = trainer.learn(
    model = model,
    train = train,
    test = test,
    # loss = NegativeLogLikelihoodLoss(),
    # loss = CrossEntropyLoss(),
    loss = SquaredLoss(),
    # optimizer = GradientDescent(learning_rate=0.3),
    optimizer = AdaGrad(learning_rate=0.7),
    epochs = epochs,
    batch_size = 256
)


raw_input('Press ENTER to exit')

model.save('model.net')
