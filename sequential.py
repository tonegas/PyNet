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

    def backward(self, dJdy, optimizer = False):
        aux_dJdy = dJdy
        for layer in reversed(self.layers):
            aux_dJdx = layer.backward(aux_dJdy)
            if hasattr(layer,'dJdW_gradient') and optimizer and hasattr(optimizer, 'update'):
                optimizer.update(layer, layer.dJdW_gradient(aux_dJdy))

            aux_dJdy = aux_dJdx

        return aux_dJdy

    def learn_one(self, input, target, loss, optimizer):
        y = self.forward(input)
        J = loss.loss(y,target)
        dJdy = loss.dJdy_gradient(y,target)
        self.backward(dJdy, optimizer)
        return J, dJdy

    def learn(self, input_data, target_data, loss, optimizer, epochs):
        J_list = np.zeros(epochs)
        dJdy_list = np.zeros(epochs)
        for epoch in range(epochs):
            for x,t in izip(input_data,target_data):
                J, dJdy = self.learn_one(x, t, loss, optimizer)
                J_list[epoch] = np.sum(np.abs(J))
                dJdy_list[epoch] = np.sum(dJdy)

        return J_list, dJdy_list
