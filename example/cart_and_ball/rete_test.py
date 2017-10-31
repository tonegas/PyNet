import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import utils
from cart_and_ball_dyn import Cart, Ball
from layers import GenericLayer
from utils import to_one_hot_vect

from trainer import Trainer
from losses import HuberLoss, SquaredLoss
from optimizers import GradientDescent, GradientDescentMomentum, AdaGrad
from network import Sequential
from layers import  TanhLayer, LinearLayer, ReluLayer, NormalizationLayer, ComputationalGraphLayer, SelectVariableLayer
from printers import ShowTraining

epochs = 10

norm = NormalizationLayer(
    np.array([0.0,0.0]),
    np.array([5.0,5.0]),
    np.array([0.0,0.0]),
    np.array([1.0,1.0])
)

W1 = utils.SharedWeights(np.array([[0,0,0.0],[0,0,0.0]]),2+1,2)
#W1 = utils.SharedWeights('gaussian',2+1,2)
Q = Sequential(
    norm,
    LinearLayer(2,2,weights=W1),
    # TanhLayer
)
W2 = utils.SharedWeights(np.array([[10.0,-10.0,0.0],[-10.0,10.0,0.0]]),2+1,2)
#W2 = utils.SharedWeights('gaussian',2+1,2)
Q_hat = Sequential(
    norm,
    LinearLayer(2,2,weights=W2),
    # TanhLayer
)

# from computationalgraph import Input, HotVect
# x = Input(['x','a'],'x')
# a = Input(['x','a'],'a')
# c=ComputationalGraphLayer(x*HotVect(a))
# c.forward([[10,10],1])

printer = ShowTraining(epochs_num = epochs, weights_list={'Q':W1})
trainer = Trainer(show_training=True,  show_function = printer.show)

data_train = np.random.rand(1000,2)*5
train = []
for x in data_train:
    out = Q_hat.forward(x)
    train.append(Q_hat.forward(x)*utils.to_one_hot_vect(np.argmax(out),out.size))

data_test = np.random.rand(1000,2)*5
test = []
for x in data_test:
    out = Q_hat.forward(x)
    test.append(Q_hat.forward(x)*utils.to_one_hot_vect(np.argmax(out),out.size))

J_list, dJdy_list, J_test = trainer.learn(
    model = Q,
    train = zip(data_train,train),
    test = zip(data_test,test),
    # loss = NegativeLogLikelihoodLoss(),
    # loss = CrossEntropyLoss(),
    loss = SquaredLoss(),
    # optimizer = GradientDescent(learning_rate=0.3),
    optimizer = GradientDescentMomentum(learning_rate=0.35, momentum=0.5),
    epochs = epochs,
    batch_size = 100
)

raw_input('Press ENTER to exit')

Q.save('model.net')


