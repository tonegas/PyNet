import numpy as np
from itertools import izip
from genericlayer import GenericLayer, WithElements

class Lock(GenericLayer):
    def __init__(self, net):
        self.net = net

    def forward(self, x, update = False):
        return self.net.forward(x)

    def backward(self, dJdy, optimizer = None):
        return self.net.backward(dJdy)

class Sequential(GenericLayer, WithElements):
    def __init__(self, *args):
        WithElements.__init__(self, *args)

    def forward(self, x, update = False):
        aux_x = x
        for layer in self.elements:
            aux_x = layer.forward(aux_x, update)
        return aux_x

    def backward(self, dJdy, optimizer = None):
        aux_dJdx = dJdy
        for layer in reversed(self.elements):
            aux_dJdx = layer.backward(aux_dJdx, optimizer)
        return aux_dJdx

class Parallel(GenericLayer, WithElements):
    def __init__(self, *args):
        self.vect_size = []
        WithElements.__init__(self, *args)

    def forward(self, x, update = False):
        aux_y = []
        for element in self.elements:
            y = element.forward(x, update)
            self.vect_size.append(y.size)
            aux_y.append(y)
        return np.concatenate(aux_y,axis=0)

    def backward(self, dJdy, optimizer = None):
        aux_dJdx = []
        a = 0
        for ind,element in enumerate(self.elements):
            b = a + self.vect_size[ind]
            aux_dJdx.append(element.backward(dJdy[a:b], optimizer))
            a = b
        return np.sum(np.array(aux_dJdx),0)


class SequentialSum(GenericLayer, WithElements):
    def __init__(self, *args):
        self.vect_size = []
        WithElements.__init__(self, *args)

    def forward(self, x, update = False):
        self.x_group = []
        for element in self.elements:
            self.x_group.append(element.forward(x, update))
        self.x = np.array(self.x_group)
        return np.sum(self.x,0)

    def backward(self, dJdy, optimizer = None):
        aux_dJdx = []
        for (x, element) in izip(self.x_group, self.elements):
            aux_dJdx.append(element.backward(np.ones(x.size)*dJdy, optimizer))
        return np.sum(np.array(aux_dJdx),0)

class SequentialMul(GenericLayer, WithElements):
    def __init__(self, *args):
        self.vect_size = []
        WithElements.__init__(self, *args)

    def forward(self, x, update = False):
        x_group = []
        for element in self.elements:
            x_group.append(element.forward(x, update))
        self.x = np.array(x_group)
        return np.prod(self.x,0)

    def backward(self, dJdy, optimizer = None):
        dJdx = []
        for i in range(self.x.shape[0]):
            dJdx.append(np.prod(np.delete(self.x,i,0),0))
        dJdx_group = np.array([element*dJdy for element in dJdx])
        aux_dJdx = []
        for (dJdx, element) in izip(dJdx_group, self.elements):
            aux_dJdx.append(element.backward(dJdx, optimizer))
        return np.sum(np.array(aux_dJdx),0)

class SequentialNegative(GenericLayer):
    def __init__(self, net):
        self.net = net

    def forward(self, x, update = False):
        return -np.array(self.net.forward(x, update))

    def backward(self, dJdy, optimizer = None):
        return -np.array(self.net.backward(dJdy, optimizer))
