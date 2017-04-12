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
        # print x
        for layer in self.elements:
            # print layer
            aux_x = layer.forward(aux_x, update)
            # print aux_x
        # print aux_x
        return aux_x

    def backward(self, dJdy, optimizer = None):
        aux_dJdx = dJdy
        for layer in reversed(self.elements):
            # print layer
            # print aux_dJdx
            aux_dJdx = layer.backward(aux_dJdx, optimizer)

        # print 'END'
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


class SumGroup(GenericLayer, WithElements):
    def forward(self, x_group, update = False):
        y_group = []
        for (x, element) in izip(x_group, self.elements):
            # print element
            y_group.append(element.forward(x, update))
        # print y_group
        return np.sum(np.array(y_group),0)

    def backward(self, dJdy, optimizer = None):
        dJdx_group = []
        for element in self.elements:
            # print 'dJdy'+str(aux_dJdy)
            dJdx_group.append(element.backward(dJdy, optimizer))
        return dJdx_group

class MulGroup(GenericLayer, WithElements):
    def forward(self, x_group, update = False):
        self.y_group = []
        for (x, element) in zip(x_group, self.elements):
            self.y_group.append(element.forward(x, update))
        return np.prod(np.array(self.y_group),0)

    def backward(self, dJdy, optimizer = None):
        dJdx_group = []
        for i,element in enumerate(self.elements):
            aux_dJdy = np.prod(np.array(self.y_group[:i]+self.y_group[i+1:]),0)*dJdy
            dJdx_group.append(element.backward(aux_dJdy, optimizer))

        return dJdx_group

class ParallelGroup(GenericLayer, WithElements):
    def forward(self, x, update = False):
        # print 'parallel-go'
        y_group = []
        for element in self.elements:
            # print x
            # print element
            y_group.append(element.forward(x, update))
        return y_group

    def backward(self, dJdy_group, optimizer = None):
        # print 'parallel-back'
        aux_dJdx = []
        for (dJdy, element) in izip(dJdy_group, self.elements):
            # print aux_dJdx
            aux_dJdx.append(element.backward(dJdy, optimizer))
            # print aux_dJdx

        # print 'END-parallel'
        # print aux_dJdx
        return np.sum(np.array(aux_dJdx),0)

class MapGroup(GenericLayer, WithElements):
    def forward(self, x_group, update = False):
        y_group = []
        for (x, element) in zip(x_group, self.elements):
            y_group.append(element.forward(x, update))
        return y_group

    def backward(self, dJdy_group, optimizer = None):
        aux_dJdx = []
        for (dJdy, element) in izip(dJdy_group, self.elements):
            aux_dJdx.append(element.backward(dJdy, optimizer))

        return aux_dJdx
