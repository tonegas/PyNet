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

    def on_message(self, message, *args, **kwargs):
        pass

    def forward(self, x, update = False):
        return x

    def backward(self, dJdy, optimizer = None):
        return dJdy

    def printlayer(self, level):
        strlab = self.__class__.__name__
        if hasattr(self,'printelements'):
            strlab += self.printelements(level)

        return strlab

    def __str__(self):
        return self.printlayer(0)

class WithNet(GenericLayer):
    def __init__(self, net):
        self.net = net

    def forward(self, x, update = False):
        return self.net.forward(x, update)

    def backward(self, dJdy, optimizer = None):
        return self.net.backward(dJdy, optimizer)

    def printelements(self,level):
        return self.net.printelements(level+1)

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

    def on_message(self, message, *args, **kwargs):
        for element in self.elements:
            op = getattr(element, "on_message", None)
            if callable(op):
                element.on_message(message,*args,**kwargs)

    def printelements(self,level):
        strlab = '(\n'
        for element in self.elements:
            for l in range(level):
                strlab += '\t'
            strlab += element.printlayer(level+1)+'\n'
        for l in range(level-1):
            strlab += '\t'
        strlab += ')'
        return strlab

    # def __str__(self):
    #     strlab = self.__class__.__name__+'(\n'
    #     for element in self.elements:
    #         strlab+='\t'+str(element)+'\n'
    #     strlab+=')'
    #     return strlab