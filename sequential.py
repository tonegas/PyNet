import numpy as np
from itertools import izip
from genericlayer import GenericLayer

class Sequential(GenericLayer):
    def __init__(self, layers = None):
        self.layers = [] if layers is None else layers

    def add(self,layer):
        self.layers.append(layer)

    def forward(self,x):
        aux_x = x
        for layer in self.layers:
            aux_x = layer.forward(aux_x)

        return aux_x

    def backward(self, in_delta, update = False):
        aux_in_delta = in_delta
        for layer in reversed(self.layers):
            aux_out_delta = layer.backward(aux_in_delta)
            if update:
                layer.update(aux_in_delta)

            aux_in_delta = aux_out_delta

        return aux_in_delta

    def learn_one(self, x, t, loss, learning_rate):
        y = self.forward(x)
        J = loss.calc_loss(y,t)
        delta = loss.calc_delta(y,t)
        self.backward(learning_rate * delta, True)
        return J, delta

    def learn(self, x_list, t_list, loss, learning_rate, epochs):
        J_list = np.zeros(x_list.size*epochs)
        delta_list = []
        for epoch in xrange(epochs):
            for x,t in izip(x_list,t_list):
                J, delta = self.learn_one(self, x, t, loss, True)
                J_list.append(J)

        return J, delta
