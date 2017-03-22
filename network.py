import numpy as np
from itertools import izip
from genericlayer import GenericLayer, WithElements

class Sequential(GenericLayer, WithElements):
    def forward(self,x):
        aux_x = x
        for layer in self.elements:
            aux_x = layer.forward(aux_x)
        return aux_x

    def backward(self, dJdy):
        dJdx = dJdy
        for layer in reversed(self.elements):
            dJdx = layer.backward(dJdx)

        return dJdx

    def backward_and_update(self, dJdy, optimizer, depth):
        aux_dJdx = dJdy
        for layer in reversed(self.elements):
            # print 'dJdy'+str(aux_dJdy)
            if depth - 1 >= 0:
                dJdx = layer.backward_and_update(aux_dJdx, optimizer, depth-1)
            else:
                dJdx = layer.backward(aux_dJdx)
            #print aux_dJdy
            if hasattr(layer,'dJdW_gradient') and hasattr(optimizer, 'update'):
                # print 'grad'+str(np.max(layer.dJdW_gradient(aux_dJdy)))
                optimizer.update(layer, layer.dJdW_gradient(aux_dJdx))

            aux_dJdx = dJdx

        return aux_dJdx

class Parallel(GenericLayer, WithElements):
    def __init__(self, *args):
        self.vect_size = []
        WithElements.__init__(self, *args)

    def forward(self,x):
        aux_y = []
        for element in self.elements:
            y = element.forward(x)
            self.vect_size.append(y.size)
            aux_y.append(y)
        return np.concatenate(aux_y,axis=0)

    def backward(self, dJdy):
        aux_dJdx = []
        a = 0
        for ind,element in enumerate(self.elements):
            b = a+self.vect_size[ind]
            aux_dJdx.append(element.backward(dJdy[a:b]))
            a = b
        return np.sum(np.array(aux_dJdx),0)

    def backward_and_update(self, dJdy, optimizer, depth):
        aux_dJdx = []
        a = 0
        for ind,element in enumerate(self.elements):
            b = a + self.vect_size[ind]
            if depth - 1 >= 0:
                aux_dJdx.append(element.backward_and_update(dJdy[a:b], optimizer, depth-1))
            else:
                aux_dJdx.append(element.backward(dJdy[a:b]))

            if hasattr(element,'dJdW_gradient') and hasattr(optimizer, 'update'):
                optimizer.update(element, element.dJdW_gradient(dJdy[a:b]))

            a = b
        return np.sum(np.array(aux_dJdx),0)


class SumGroup(GenericLayer, WithElements):
    def forward(self, x_group):
        y_group = []
        for (x, element) in izip(x_group, self.elements):
            y_group.append(element.forward(x))
        return np.sum(np.array(y_group),0)

    def backward(self, dJdy):
        dJdx_group = []
        for element in self.elements:
            dJdx_group.append(element.backward(dJdy))
        return dJdx_group

    def backward_and_update(self, dJdy, optimizer, depth):
        dJdx_group = []
        for element in self.elements:
            # print 'dJdy'+str(aux_dJdy)
            if depth - 1 >= 0:
                dJdx_group.append(element.backward_and_update(dJdy, optimizer, depth-1))
            else:
                dJdx_group.append(element.backward(dJdy))

            if hasattr(element,'dJdW_gradient') and hasattr(optimizer, 'update'):
                # print 'grad'+str(np.max(layer.dJdW_gradient(aux_dJdy)))
                optimizer.update(element, element.dJdW_gradient(dJdy))

        return dJdx_group

class MulGroup(GenericLayer, WithElements):
    def forward(self, x_group):
        self.y_group = []
        for (x, element) in zip(x_group, self.elements):
            self.y_group.append(element.forward(x))
        return np.prod(np.array(self.y_group),0)

    def backward(self, dJdy):
        dJdx_group = []
        for i,element in enumerate(self.elements):
            dJdx_group.append(element.backward(np.prod(np.array(self.y_group[:i]+self.y_group[i+1:]),0)*dJdy))

        return dJdx_group

    def backward_and_update(self, dJdy, optimizer, depth):
        dJdx_group = []
        for i,element in enumerate(self.elements):
            # print 'dJdy'+str(aux_dJdy)
            aux_dJdy = np.prod(np.array(self.y_group[:i]+self.y_group[i+1:]),0)*dJdy
            if depth - 1 >= 0:
                dJdx_group.append(element.backward_and_update(aux_dJdy, optimizer, depth-1))
            else:
                dJdx_group.append(element.backward(aux_dJdy))

            if hasattr(element,'dJdW_gradient') and hasattr(optimizer, 'update'):
                # print 'grad'+str(np.max(layer.dJdW_gradient(aux_dJdy)))
                optimizer.update(element, element.dJdW_gradient(aux_dJdy))

        return dJdx_group

class ParallelGroup(GenericLayer, WithElements):
    def forward(self, x):
        y_group = []
        for element in self.elements:
            y_group.append(element.forward(x))
        return y_group

    def backward(self, dJdy_group):
        aux_dJdx = []
        for (dJdy, element) in izip(dJdy_group, self.elements):
            aux_dJdx.append(element.backward(dJdy))

        return np.sum(np.array(aux_dJdx),0)

    def backward_and_update(self, dJdy_group, optimizer, depth):
        aux_dJdx = []
        for (dJdy, element) in izip(dJdy_group, self.elements):
            if depth - 1 >= 0:
                aux_dJdx.append(element.backward_and_update(dJdy, optimizer, depth-1))
            else:
                aux_dJdx.append(element.backward(dJdy))

            if hasattr(element,'dJdW_gradient') and hasattr(optimizer, 'update'):
                optimizer.update(element, element.dJdW_gradient(dJdy))

        return np.sum(np.array(aux_dJdx),0)

class MapGroup(GenericLayer, WithElements):
    def forward(self, x_group):
        y_group = []
        for (x, element) in zip(x_group, self.elements):
            y_group.append(element.forward(x))
        return y_group

    def backward(self, dJdy_group):
        aux_dJdx = []
        for (dJdy, element) in izip(dJdy_group, self.elements):
            aux_dJdx.append(element.backward(dJdy))
        return aux_dJdx

    def backward_and_update(self, dJdy_group, optimizer, depth):
        aux_dJdx = []
        for (dJdy, element) in izip(dJdy_group, self.elements):
            if depth - 1 >= 0:
                aux_dJdx.append(element.backward_and_update(dJdy, optimizer, depth-1))
            else:
                aux_dJdx.append(element.backward(dJdy))

            if hasattr(element,'dJdW_gradient') and hasattr(optimizer, 'update'):
                optimizer.update(element, element.dJdW_gradient(dJdy))

        return aux_dJdx