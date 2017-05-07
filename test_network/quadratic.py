import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from layers import LinearLayer, MulLayer, SumLayer, VWeightLayer, ComputationalGraphLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from optimizers import GradientDescent, GradientDescentMomentum, AdaGrad
from network import Sequential, Parallel
from groupnetworks import ParallelGroup, SumGroup, MulGroup
from genericlayer import GenericLayer
from trainer import Trainer
from computationalgraph import VWeight, Input
from printers import ShowTraining

# y = a*x^2+b*x+c
# n = Sequential(
#         ParallelGroup(
#             Sequential(
#                 ParallelGroup(
#                     GenericLayer,
#                     GenericLayer,
#                     VWeightLayer(1)
#                 ),MulLayer
#             ),
#             Sequential(
#                 ParallelGroup(
#                     GenericLayer,
#                     VWeightLayer(1),
#                 ),MulLayer
#             ),
#             VWeightLayer(1)
#         ),SumLayer
#     )

#equal to
# n = Sequential(
#         ParallelGroup(
#             Sequential(
#                 ParallelGroup(
#                     GenericLayer,
#                     GenericLayer,
#                     VWeightLayer(1)
#                 ),MulLayer
#             ),
#             LinearLayer(1,1)
#         ),SumLayer
#     )

#equal to
# n = Sequential(
#         ParallelGroup(GenericLayer,GenericLayer,GenericLayer),
#         SumGroup(
#             Sequential(
#                 ParallelGroup(GenericLayer,GenericLayer,GenericLayer),
#                 MulGroup(
#                     GenericLayer,
#                     GenericLayer,
#                     VWeightLayer(1)
#                 )
#             ),
#             Sequential(
#                 ParallelGroup(GenericLayer,GenericLayer),
#                 MulGroup(
#                     GenericLayer,
#                     VWeightLayer(1)
#                 )
#             ),
#             VWeightLayer(1)
#         )
#     )

epochs = 100

#equal to
varx = Input('x','x')
a = VWeight(1,L2=0.001,weights=np.array([10.0]))
b = VWeight(1,L2=0.001,weights=np.array([10.0]))
c = VWeight(1,L2=0.001,weights=np.array([10.0]))
# a = VWeight(1,weights=np.array([10.0]))
# b = VWeight(1,weights=np.array([10.0]))
# c = VWeight(1,weights=np.array([10.0]))
# x = Input(lv,'x')
n = ComputationalGraphLayer(a*varx**2+b*varx+c)

#
train = []
for i,x in enumerate(np.linspace(-0.2,0.2,50)):
    train.append((np.array([x]),np.array([5.2*x+7.1])))

test = []
for i,x in enumerate(np.linspace(-10,10,50)):
    test.append((np.array([x]),np.array([5.2*x+7.1])))

# print (train[0][0],train[0][1])
# print n.forward(train[0][0])
# print n.numeric_gradient(train[0][0])
# print n.backward(np.array([1.0]))

printer = ShowTraining(epochs_num = epochs, weights_list={'a':a.net.W,'b':b.net.W,'c':c.net.W})
t = Trainer(show_training = True, show_function = printer.show)

#
J_list, dJdy_list, J_test_list = t.learn(
    model = n,
    train = train,
    test = test,
    loss = SquaredLoss(),
    optimizer = GradientDescentMomentum(learning_rate=0.9,momentum=0.5),
    # batch_size = len(train),
    # optimizer = AdaGrad(learning_rate=0.6),
    epochs = epochs
)

#
plt.figure(4)
plt.title('Errors History (J)')
plt.plot(np.array([x for (x,t) in test]), np.array([t for (x,t) in test]), color='red')
plt.plot(np.array([x for (x,t) in test]), np.array([n.forward(x) for (x,t) in test]), color='green')
# plt.ylim([0, 2])
plt.xlabel('x')
plt.ylabel('y')

plt.ion()
plt.show()

# n2 = LinearLayer(3,1)
#
# train2 = []
# for i,x in enumerate(np.linspace(-2,2,5)):
#     train2.append((np.array([x**2,x,1]),np.array([3.2*x**2+5.2*x+7.1])))
#
# t = Trainer(show_training=True)
#
# J_list, dJdy_list = t.learn(
#     model = n2,
#     train = train2,
#     loss = SquaredLoss(),
#     optimizer = GradientDescent(learning_rate=0.1),
#     # batch_size = len(train2),
#     epochs = 50
# )
# test = []
# for x in np.linspace(-10,10,50):
#     test.append((np.array([x**2,x,1]),np.array([3.2*x**2+5.2*x+7.1])))
#
# plt.figure(3)
# plt.title('Points and Function')
# plt.plot(np.array([x[1] for (x,t) in test]), np.array([t for (x,t) in test]), color='red')
# plt.plot(np.array([x[1] for (x,t) in test]), np.array([n2.forward(x) for (x,t) in test]), color='green')
# # plt.ylim([0, 2])
# plt.xlabel('x')
# plt.ylabel('y')
#
# plt.figure(1)
# plt.title('Errors History (J)')
# plt.plot(xrange(len(J_list)), J_list, color='red')
#
# plt.show()

raw_input('Press ENTER to exit')