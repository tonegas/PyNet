import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from layers import LinearLayer, SoftMaxLayer, SigmoidLayer, HeavisideLayer, ConstantLayer, MulLayer, SumLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from optimizers import GradientDescent, GradientDescentMomentum
from network import Sequential, Parallel, SumGroup, ParallelGroup, MulGroup, MapGroup
from genericlayer import GenericLayer
from trainer import Trainer

#y = a*x^2+b*x+c

n = Sequential(
        ParallelGroup(
            Sequential(
                ParallelGroup(
                    GenericLayer,
                    LinearLayer(1,1)
                ),MulLayer,
            ),
            LinearLayer(1,1)
        ),SumLayer
    )

n2 = LinearLayer(3,1)


# train = []
# for i,x in enumerate(np.linspace(-1,1,50)):
#     train.append((np.array([x]),np.array([3.2*x**2+5.2*x+7.1])))
# print (train[0][0],train[0][1])
# print n.forward(train[0][0])
# print n.numeric_gradient(train[0][0]),n.backward(np.array([1]))
# print n.backward(np.array([1]))
# print n.backward_and_update(np.array([1.0]),StocaticGradientDescent(learning_rate=1.0),100)
# t = Trainer(show_training=True, depth=5)
# J_list, dJdy_list = t.learn(
#     model = n,
#     train = train,
#     loss = SquaredLoss(),
#     optimizer = StocaticGradientDescent(learning_rate=0.003),
#     epochs = 50
# )
# test = []
# for i,x in enumerate(np.linspace(-10,10,50)):
#     test.append((np.array([x]),np.array([3.2*x**2+5.2*x+7.1])))
#
# plt.figure(3)
# plt.title('Errors History (J)')
# plt.plot(np.array([x for (x,t) in test]), np.array([t for (x,t) in test]), color='red')
# plt.plot(np.array([x for (x,t) in test]), np.array([n.forward(x) for (x,t) in test]), color='green')
# # plt.ylim([0, 2])
# plt.xlabel('x')
# plt.ylabel('y')
#
# print (train[0][1],n.forward(train[0][0]))
#
# plt.figure(1)
# plt.title('Points and Function')
# plt.plot(xrange(len(J_list)), J_list, color='red')
#
# plt.show()


train2 = []
for i,x in enumerate(np.linspace(-2,2,5)):
    train2.append((np.array([x**2,x,1]),np.array([3.2*x**2+5.2*x+7.1])))

t = Trainer(show_training=True)

J_list, dJdy_list = t.learn(
    model = n2,
    train = train2,
    loss = SquaredLoss(),
    optimizer = GradientDescent(learning_rate=0.01),
    batch_size = len(train2),
    epochs = 50
)
test = []
for x in np.linspace(-10,10,50):
    test.append((np.array([x**2,x,1]),np.array([3.2*x**2+5.2*x+7.1])))

plt.figure(3)
plt.title('Points and Function')
plt.plot(np.array([x[1] for (x,t) in test]), np.array([t for (x,t) in test]), color='red')
plt.plot(np.array([x[1] for (x,t) in test]), np.array([n2.forward(x) for (x,t) in test]), color='green')
# plt.ylim([0, 2])
plt.xlabel('x')
plt.ylabel('y')

plt.figure(1)
plt.title('Errors History (J)')
plt.plot(xrange(len(J_list)), J_list, color='red')

plt.show()
