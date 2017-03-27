import unittest
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from layers import LinearLayer, SoftMaxLayer, SigmoidLayer, HeavisideLayer, ConstantLayer, MulLayer, SumLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from optimizers import GradientDescent, GradientDescentMomentum
from network import Sequential, Parallel, SumGroup, ParallelGroup, MulGroup, MapGroup
from genericlayer import GenericLayer
from trainer import Trainer
from printers import Printer2D
from classicnetwork import Vanilla, LSTM

n1 = Sequential([
    LinearLayer(2, 5, weights='random'),
    SigmoidLayer()
])

n2 = Sequential([
    LinearLayer(5, 1, weights='random'),
    SigmoidLayer()
])

n3 = Sequential([
    LinearLayer(2, 2, weights='random'),
    n1,
    n2
])



n4 = Sequential(
    ParallelGroup(LinearLayer(10,3),LinearLayer(10,5)),
    SumGroup(LinearLayer(3,3),LinearLayer(5,3)),
    LinearLayer(3,10)
)


# print n4.forward(np.array([1,2,3,1,1,1,1,1,1,1]))


x = np.array([1,2,3])
y = np.array([5,2,-1])

print x
print y


n5 = MulGroup(
    SumGroup(GenericLayer,GenericLayer),
    SumGroup(MulGroup(GenericLayer,GenericLayer),GenericLayer)
)
# print n5.forward([[x,y],[[x,y],np.array([3,3,3])]])
# print n5.backward(np.array([1,1,1]))
n6 = ParallelGroup(GenericLayer,ParallelGroup(GenericLayer,ConstantLayer(np.array([3,3,3]))))
# print n6.forward([x,y])

n7 = Sequential(n6,n5)
print n7.forward([x,y])
print n7.backward(np.array([1,1,1]))


#(x+y)*(x*y+3)

i = Sequential(MapGroup(GenericLayer,GenericLayer),SumLayer)
# print i.forward([x,y])
ii = Sequential(
        ParallelGroup(
                Sequential(MapGroup(GenericLayer,GenericLayer),MulLayer),
                ConstantLayer(np.array([3,3,3]))
        ),SumLayer
    )
# print ii.forward([x,y])

o = Sequential(
        ParallelGroup(
            i,ii
        ),MulLayer
    )

print o.forward([x,y])
print o.backward(np.array([1,1,1]))


#ibrido
oo = MulGroup(i,ii)
print oo.forward([[x,y],[x,y]])

oo = MulGroup(i,ii)
print oo.forward([[x,y],[x,y]])



# print oo.backward(np.array([1,1,1]))

# n1 = Vanilla(2,2,2)
# print n1.forward(np.array([1,2]))

# n1 = LSTM(3,2)
# for i in range(100):
#     print n1.forward(np.array([-1,-1,-1]))


# print n1.backward(np.array([1,1,1,1,1]))
# print n1.backward(np.array([1,1,1,1,1]))

# # print n3.forward(np.array([2,3]))
# # print n3.backward(np.array([2]))
#
# o = StocaticGradientDescent(learning_rate=0.01)
#
# t = Trainer()
# n3.forward(np.array([2,3]))
# t.train(n3,np.array([3]),o)
# # print n1.forward(np.array([2,3]))
#
# t.train(n3,np.array([3]),o,2)
# # print n1.forward(np.array([2,3]))
#
# # print n3.forward(np.array([2,3]))
#
# n4 = Parallel([
#     LinearLayer(5, 2, weights='random'),
#     n2,
# ])
#
# print n4.forward(np.array([2,3,6,7,8]))
# print n4.backward([3,4,5])
#
# print n2.forward(np.array([2,3,6,7,8]))
# t.train(n4,np.array([3,4,5]),o)
# print n4.forward(np.array([2,3,6,7,8]))
# print n2.forward(np.array([2,3,6,7,8]))
#
# print n2.forward(np.array([2,3,6,7,8]))
# t.train(n4,np.array([3,4,5]),o,2)
# print n4.forward(np.array([2,3,6,7,8]))
# print n2.forward(np.array([2,3,6,7,8]))