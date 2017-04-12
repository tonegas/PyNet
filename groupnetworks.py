import numpy as np
from itertools import izip

from genericlayer import GenericLayer, WithElements

class SumGroup(GenericLayer, WithElements):
    def forward(self, x_group, update = False):
        y_group = []
        for (x, element) in izip(x_group, self.elements):
            y_group.append(element.forward(x, update))
        return np.sum(np.array(y_group),0)

    def backward(self, dJdy, optimizer = None):
        dJdx_group = []
        for element in self.elements:
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
        y_group = []
        for element in self.elements:
            y_group.append(element.forward(x, update))
        return y_group

    def backward(self, dJdy_group, optimizer = None):
        aux_dJdx = []
        for (dJdy, element) in izip(dJdy_group, self.elements):
            aux_dJdx.append(element.backward(dJdy, optimizer))
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
