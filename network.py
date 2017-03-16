import numpy as np
from itertools import izip
from genericlayer import GenericLayer

class Sequential(GenericLayer):
    def __init__(self, layers = None):
        self.group = True
        self.layers = [] if layers is None else layers

    def add(self,layer):
        self.layers.append(layer)

    def forward(self,x):
        aux_x = x
        for layer in self.layers:
            aux_x = layer.forward(aux_x)
        return aux_x

    def backward(self, dJdy):
        aux_dJdy = dJdy
        for layer in reversed(self.layers):
            aux_dJdy = layer.backward(aux_dJdy)

        return aux_dJdy

class Parallel(GenericLayer):
    def __init__(self, elements = None):
        self.vect_size = []
        self.group = True
        self.elements = [] if elements is None else elements

    def add(self,element):
        self.elements.append(element)

    def forward(self,x):
        aux_y = []
        for element in self.elements:
            y = element.forward(x)
            self.vect_size.append(y.size)
            aux_y.append(y)
        return np.concatenate(aux_y,axis=0)

    def backward(self, dJdy):
        aux_dJdy = []
        a = 0
        for ind,element in enumerate(self.elements):
            b = a+self.vect_size[ind]
            aux_dJdy.append(element.backward(dJdy[a:b]))
            a = b
        return np.sum(np.array(aux_dJdy),0)

