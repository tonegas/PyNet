import unittest
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from layers import LinearLayer, SoftMaxLayer, SigmoidLayer, UnitStepLayer
from losses import SquaredLoss, NegativeLogLikelihoodLoss, CrossEntropyLoss
from optimizers import StocaticGradientDescent, SGDMomentum
from sequential import Sequential, Parallel
from trainer import Trainer
from printers import Printer2D

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

# print n3.forward(np.array([2,3]))
# print n3.backward(np.array([2]))

o = StocaticGradientDescent(learning_rate=0.01)

t = Trainer()
n3.forward(np.array([2,3]))
t.train(n3,np.array([3]),o)
# print n1.forward(np.array([2,3]))

t.train(n3,np.array([3]),o,2)
# print n1.forward(np.array([2,3]))

# print n3.forward(np.array([2,3]))

n4 = Parallel([
    LinearLayer(5, 2, weights='random'),
    n2,
])

print n4.forward(np.array([2,3,6,7,8]))
print n4.backward([3,4,5])

print n2.forward(np.array([2,3,6,7,8]))
t.train(n4,np.array([3,4,5]),o)
print n4.forward(np.array([2,3,6,7,8]))
print n2.forward(np.array([2,3,6,7,8]))

print n2.forward(np.array([2,3,6,7,8]))
t.train(n4,np.array([3,4,5]),o,2)
print n4.forward(np.array([2,3,6,7,8]))
print n2.forward(np.array([2,3,6,7,8]))