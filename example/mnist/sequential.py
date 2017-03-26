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


num_classes = 10
name_net = "mnist.net"
load_net = False
epochs = 10


train = load_mnist_dataset(dataset = "training", path = "./mnist")
test = load_mnist_dataset(dataset = "testing", path = "./mnist")


def test_results(model,train,test):
    err = 0
    for i,(img,target) in enumerate(train):
        if np.argmax(model.forward(img)) != np.argmax(target):
            #print str(err)+' '+str(np.argmax(model.forward(train_data[ind])))+' '+str(np.argmax(train_targets[ind]))
            err += 1
    print (1.0-err/float(len(train)))*100.0

    err = 0
    for (img,target) in test:
        #print str(np.argmax(model.forward(test_data[ind])))+' '+str(np.argmax(test_targets[ind]))
        if np.argmax(model.forward(img)) != np.argmax(target):
            err += 1
    print (1.0-err/float(len(test)))*100.0


if load_net:
    print "Load Network"
    model = StoreNetwork.load(name_net)
else:
    print "New Network"
    #Two layer network
    model = Sequential([
        NormalizationLayer(0,255,-0.1,0.1),
        LinearLayer(784, 50, weights='norm_random'),
        TanhLayer,
        LinearLayer(50, 10, weights='norm_random'),
        TanhLayer,
        # NormalizationLayer(0,10,0,1),
        # SigmoidLayer()
    ])

display = ShowTraining(epochs_num = epochs)

trainer = Trainer(show_training = True, show_function = display.show)

validation = train[0:1000]
train = train[1001:]

J_list, dJdy_list, J_validation = trainer.learn(
    model = model,
    train = train,
    validation = validation,
    # loss = NegativeLogLikelihoodLoss(),
    loss = CrossEntropyLoss(),
    # loss = SquaredLoss(),
    # optimizer = GradientDescent(learning_rate=0.3),
    optimizer = GradientDescentMomentum(learning_rate=0.35/1000, momentum=0.95),
    epochs = epochs,
    batch_size = 1000
)

test_results(model,train,test)

raw_input('Press ENTER to exit')

model.save('model.net')