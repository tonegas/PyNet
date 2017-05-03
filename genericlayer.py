import inspect, os
import dill as pickle
import numpy as np

class StoreNetwork:
    def save(self, file):
        f = open(file, "w")
        pickle.dump(self,f)

    @staticmethod
    def load(file):
        if os.path.isfile(file):
            f = open(file, "r")
            return pickle.load(f)
        else:
            raise Exception('File does not exist!')

    @staticmethod
    def load_or_create(file, net):
        if os.path.isfile(file):
            f = open(file, "r")
            return pickle.load(f)
        else:
            return net

class GenericLayer(StoreNetwork):
    def numeric_gradient(self,x):
        dx = 0.00000001
        fx = self.forward(x)
        dJdx = np.zeros([fx.size,x.size])
        for r in xrange(x.size):
            dxvett = np.zeros(x.size)
            dxvett[r] = dx
            fxdx = self.forward(x+dxvett)
            dJdx[:,r] = (fxdx-fx)/dx
        return dJdx

    def forward(self, x, update = False):
        return x

    def backward(self, dJdy, optimizer = None):
        return dJdy

class WithElements:
    def __init__(self, *args):
        self.elements = []
        if len(args) == 1 and type(args[0]) == list:
            args = args[0]

        for element in args:
            self.add(element)

    def add(self, element):
        if inspect.isclass(element):
            element = element()

        self.elements.append(element)
        return self